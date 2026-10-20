"""HDF5 storage of a dataset split and a PyTorch ``Dataset`` to read it.

File layout (one file per split)::

    hologram        (N, H, W) float32   noisy, flat-field corrected (1 = free space)
    hologram_clean  (N, H, W) float32   noise-free (optional, ``storage.store_clean``)
    phase           (N, H, W) float32   ground-truth phase phi in rad (optional)
    labels/<key>    (N,) float64        dz01, z01, z01_guess, fresnel_number, ...
    meta/<key>      (N,)                phantom_type, phase_shift, photons, a0, ...
    attrs           config (YAML), seed rule, label convention, versions

HDF5 is the common container at synchrotron beamlines and is read lazily per sample, which
suits a PyTorch ``DataLoader`` with several workers (each worker opens its own handle).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from .. import __version__
from ..phantoms import PHANTOM_TYPES
from .config import DatasetConfig
from .generator import LABEL_KEYS, META_KEYS, HologramGenerator

FORMAT_VERSION = 1
LABEL_CONVENTION = (
    "dz01 = z01 - z01_guess (true minus guess, metres); "
    "[2] Eq. (9) writes z01_hat = z01 + dz01 with the opposite sign"
)
TARGETS = ("dz01", "dz01_mm", "z01", "fresnel_number", "fresnel_ratio")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_split(
    config: DatasetConfig,
    split: str,
    path: str | Path,
    n_samples: int | None = None,
    device: str | None = None,
    log: Callable[[str], None] | None = print,
) -> Path:
    """Generate ``n_samples`` (default: ``config.splits[split]``) samples into ``path``.

    The file is written to a temporary name and renamed at the end, so an interrupted run
    never leaves a truncated file behind.
    """
    path = Path(path)
    n = config.splits[split] if n_samples is None else int(n_samples)
    generator = HologramGenerator(config, device=device)
    h, w = generator.fov_shape
    storage = config.storage
    options = {}
    if storage.compression:
        options = {"compression": storage.compression, "shuffle": True}
        if storage.compression == "gzip":
            options["compression_opts"] = storage.compression_level
    tmp = path.with_suffix(path.suffix + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with h5py.File(tmp, "w") as fh:
        fh.attrs.update(
            {
                "format_version": FORMAT_VERSION,
                "holofocus_version": __version__,
                "git_commit": _git_commit(),
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "split": split,
                "n_samples": n,
                "seed": config.seed,
                "seed_rule": "numpy SeedSequence(seed, spawn_key=(zlib.crc32(split), index))",
                "label_convention": LABEL_CONVENTION,
                "phantom_types": json.dumps(PHANTOM_TYPES),
                "fov_shape": generator.fov_shape,
                "simulation_shape": generator.simulation_shape,
                "pixel_size": config.geometry.pixel_size,
                "device": str(generator.device),
                "config_yaml": config.to_yaml(),
            }
        )
        chunk = (1, h, w)
        holo = fh.create_dataset("hologram", (n, h, w), np.float32, chunks=chunk, **options)
        clean = (
            fh.create_dataset("hologram_clean", (n, h, w), np.float32, chunks=chunk, **options)
            if storage.store_clean
            else None
        )
        phase = (
            fh.create_dataset("phase", (n, h, w), np.float32, chunks=chunk, **options)
            if storage.store_phase
            else None
        )
        labels = {k: fh.create_dataset(f"labels/{k}", (n,), np.float64) for k in LABEL_KEYS}
        meta = {
            k: fh.create_dataset(f"meta/{k}", (n,), np.int8 if k == "phantom_type" else np.float64)
            for k in META_KEYS
        }
        for i in range(n):
            sample = generator.sample(split, i)
            holo[i] = sample.hologram
            if clean is not None:
                clean[i] = sample.clean
            if phase is not None:
                phase[i] = sample.phase
            for k in LABEL_KEYS:
                labels[k][i] = sample.labels[k]
            for k in META_KEYS:
                meta[k][i] = sample.meta[k]
            if log is not None and (i + 1 == n or (i + 1) % max(n // 10, 10) == 0):
                rate = (time.perf_counter() - start) / (i + 1)
                log(f"{split}: {i + 1}/{n} samples, {rate:.3f} s/sample")
        fh.attrs["seconds"] = time.perf_counter() - start
    os.replace(tmp, path)
    return path


def read_labels(path: str | Path) -> dict[str, np.ndarray]:
    """All labels and per-sample metadata of a split as numpy arrays."""
    with h5py.File(path, "r") as fh:
        out = {k: fh[f"labels/{k}"][()] for k in fh["labels"]}
        out.update({k: fh[f"meta/{k}"][()] for k in fh["meta"]})
    return out


def read_config(path: str | Path) -> DatasetConfig:
    """The :class:`DatasetConfig` a split was generated with."""
    with h5py.File(path, "r") as fh:
        text = fh.attrs["config_yaml"]
    return DatasetConfig.from_dict(yaml.safe_load(text))


def target_values(labels: dict[str, np.ndarray], target: str) -> np.ndarray:
    """Training target derived from the stored labels.

    ``dz01`` (m) and ``dz01_mm`` are the primary target; ``fresnel_number`` is the true
    ``Fr`` and ``fresnel_ratio = Fr / Fr_guess - 1`` the relative Fresnel-number error
    (about ``dz01 / z01`` for ``z01 << z12``).
    """
    if target == "dz01":
        return labels["dz01"]
    if target == "dz01_mm":
        return labels["dz01"] * 1e3
    if target == "z01":
        return labels["z01"]
    if target == "fresnel_number":
        return labels["fresnel_number"]
    if target == "fresnel_ratio":
        return labels["fresnel_number"] / labels["fresnel_number_guess"] - 1.0
    raise ValueError(f"unknown target {target!r}, choose from {TARGETS}")


class HologramDataset(torch.utils.data.Dataset):
    """Holograms of one split with a selectable target (see :func:`target_values`).

    Items are ``(hologram, target)`` with ``hologram`` of shape ``(1, H, W)`` and a scalar
    ``target`` (float32), or ``(hologram, target, phase)`` with ``with_phase=True``.
    """

    def __init__(self, path: str | Path, target: str = "dz01", with_phase: bool = False):
        self.path = Path(path)
        self.target = target
        self.with_phase = with_phase
        self.labels = read_labels(self.path)
        self.targets = torch.as_tensor(target_values(self.labels, target), dtype=torch.float32)
        self._file: h5py.File | None = None
        self._pid: int | None = None

    def __len__(self) -> int:
        return len(self.targets)

    def _handle(self) -> h5py.File:
        # HDF5 handles must not be shared with forked DataLoader workers.
        if self._file is None or self._pid != os.getpid():
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    def __getitem__(self, index: int):
        fh = self._handle()
        hologram = torch.from_numpy(fh["hologram"][index])[None]
        if not self.with_phase:
            return hologram, self.targets[index]
        return hologram, self.targets[index], torch.from_numpy(fh["phase"][index])[None]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state
