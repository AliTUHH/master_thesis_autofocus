"""Run the classical autofocus of [2] on the samples of a dataset split.

Outputs in the run directory:

* ``run.yaml``: method config, ASRM settings for the data, dataset identity (path, SHA-256,
  seed, commit of the generator), software versions and host;
* ``results.csv``: one row per sample (:mod:`holofocus.baseline.results`), with the phase
  error against the ground truth if the split stores it (``storage.store_phase``);
* ``traces.jsonl``: every probing point ``(z01, MFE, seconds)`` per sample;
* ``summary.json``: aggregate accuracy and cost, rewritten after every sample.

Runs are resumable: samples already in ``results.csv`` are skipped, and a run directory
created with another config, dataset or warm start is refused.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from .. import __version__
from ..asrm import ASRMResult, asrm_reconstruct
from ..autofocus.search import find_focus
from ..dataset.storage import _git_commit, read_config, read_labels
from ..device import get_device, real_dtype
from ..yamlconfig import plain
from .config import BaselineConfig
from .results import append_row, append_trace, read_rows, summarize


def file_sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def dataset_identity(path: str | Path) -> dict:
    """Path, SHA-256, split, size, seed and generator commit of a split file."""
    with h5py.File(path, "r") as fh:
        attrs = fh.attrs
        identity = {
            "path": str(path),
            "sha256": file_sha256(path),
            "split": str(attrs["split"]),
            "n_samples": int(attrs["n_samples"]),
            "seed": int(attrs["seed"]),
            "generator_commit": str(attrs["git_commit"]),
            "created_utc": str(attrs["created_utc"]),
        }
    return identity


def _run_key(info: dict) -> tuple:
    """What must match to continue a run: method, settings, dataset content and warm start."""
    dataset = info.get("dataset") or {}
    return (
        info.get("config"),
        info.get("asrm_settings"),
        dataset.get("sha256"),
        info.get("warm_start"),
    )


@dataclass(frozen=True)
class WarmStart:
    """Initial simplexes from point estimates of ``z01``, e.g. from the learned model.

    With ``anchor="center"``, sample ``i`` starts from ``(c_i - w_i, c_i + w_i)`` with the
    estimate ``c_i`` and the half width ``w_i = max(half_width, sigmas * sigma_i)``; the pair
    is moved inside ``Omega_z`` if it sticks out. With ``anchor="vertex"`` it starts from
    ``(c_i, c_i + w_i)`` (``c_i - w_i`` at the upper end of ``Omega_z``), so the estimate
    itself is the first probe. ``source`` is a CSV with the columns ``index``, ``z01_est``
    and (for ``sigmas > 0``) ``sigma``, both in metres.
    """

    source: str
    sha256: str
    half_width: float
    sigmas: float
    centers: dict[int, float]
    sigma: dict[int, float]
    anchor: str = "center"

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        half_width: float = 5e-4,
        sigmas: float = 0.0,
        anchor: str = "center",
    ):
        if half_width <= 0 or sigmas < 0:
            raise ValueError("half_width must be positive and sigmas non-negative")
        if anchor not in ("center", "vertex"):
            raise ValueError(f"anchor must be 'center' or 'vertex', got {anchor!r}")
        centers, sigma = {}, {}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                index = int(row["index"])
                centers[index] = float(row["z01_est"])
                if sigmas > 0:
                    sigma[index] = float(row["sigma"])
        return cls(
            str(path),
            file_sha256(path),
            float(half_width),
            float(sigmas),
            centers,
            sigma,
            anchor,
        )

    def identity(self) -> dict:
        identity = {
            "source": self.source,
            "sha256": self.sha256,
            "half_width": self.half_width,
            "sigmas": self.sigmas,
        }
        # Runs made before the anchor existed keep their key.
        if self.anchor != "center":
            identity["anchor"] = self.anchor
        return identity

    def simplex(self, index: int, lower: float, upper: float) -> tuple[float, float]:
        width = self.half_width
        if self.sigmas > 0:
            width = max(width, self.sigmas * self.sigma[index])
        if self.anchor == "vertex":
            width = min(width, upper - lower)
            vertex = min(max(self.centers[index], lower), upper)
            return (vertex, vertex + width if vertex + width <= upper else vertex - width)
        width = min(width, 0.5 * (upper - lower))
        center = min(max(self.centers[index], lower + width), upper - width)
        return (center - width, center + width)


def phase_rmse(result: ASRMResult | None, phase: torch.Tensor | None) -> float | None:
    """RMS error of the reconstructed phase; ``None`` without truth or on another grid."""
    if result is None or phase is None or tuple(result.obj.shape) != tuple(phase.shape):
        return None
    return float((result.obj.real - phase.to(result.obj.real.dtype)).square().mean().sqrt())


def select_indices(n_samples: int, indices: Sequence[int] | None, limit: int | None) -> list[int]:
    """``indices`` (validated) or ``range(n_samples)``, cut to the first ``limit``."""
    chosen = list(range(n_samples)) if indices is None else [int(i) for i in indices]
    bad = [i for i in chosen if not 0 <= i < n_samples]
    if bad:
        raise ValueError(f"indices {bad} outside the split (0 .. {n_samples - 1})")
    return chosen if limit is None else chosen[:limit]


def run_baseline(
    config: BaselineConfig,
    data: str | Path,
    out: str | Path,
    indices: Sequence[int] | None = None,
    limit: int | None = None,
    device: str | None = None,
    warm_start: WarmStart | None = None,
    log: Callable[[str], None] | None = print,
) -> dict:
    """Autofocus every selected sample of the split file ``data``; returns the summary.

    Without ``warm_start`` the search starts from the ends of ``Omega_z`` as in [2].
    """
    data, out = Path(data), Path(out)
    dataset_config = read_config(data)
    labels = read_labels(data)
    n_split = len(labels["dz01"])
    chosen = select_indices(n_split, indices, limit)
    if warm_start is not None:
        missing = [i for i in chosen if i not in warm_start.centers]
        if missing:
            raise ValueError(f"warm start {warm_start.source} has no estimate for {missing}")
    binning = dataset_config.geometry.binning
    settings = config.reconstruction.settings(binning)
    torch_device = get_device(device or config.reconstruction.device)
    dtype = real_dtype(config.reconstruction.precision)

    identity = dataset_identity(data)
    run_info = {
        "config": config.to_dict(),
        "asrm_settings": plain(dataclasses.asdict(settings)),
        "dataset": identity,
        "dataset_name": dataset_config.name,
        "binning": binning,
    }
    if warm_start is not None:
        run_info["warm_start"] = warm_start.identity()
    out.mkdir(parents=True, exist_ok=True)
    run_file = out / "run.yaml"
    if run_file.exists():
        previous = yaml.safe_load(run_file.read_text())
        if _run_key(previous) != _run_key(run_info):
            raise ValueError(f"{out} holds a run with another config or dataset")
    else:
        meta = {
            **run_info,
            "holofocus_version": __version__,
            "git_commit": _git_commit(),
            "torch_version": str(torch.__version__),
            "device": str(torch_device),
            "threads": torch.get_num_threads(),
            "host": platform.node(),
            "cpu_count": os.cpu_count(),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        run_file.write_text(yaml.safe_dump(meta, sort_keys=False))

    results_file, traces_file = out / "results.csv", out / "traces.jsonl"
    done = {row["index"] for row in read_rows(results_file)}
    todo = [i for i in chosen if i not in done]
    if log is not None:
        log(
            f"{identity['split']} split of {dataset_config.name}: {len(chosen)} samples selected, "
            f"{len(chosen) - len(todo)} done, {len(todo)} to go ({torch_device}, "
            f"{torch.get_num_threads()} threads)"
        )
    search = config.search
    with h5py.File(data, "r") as fh:
        holograms = fh["hologram"]
        phases = fh["phase"] if "phase" in fh else None
        for count, index in enumerate(todo, start=1):
            hologram = torch.from_numpy(np.asarray(holograms[index])).to(torch_device, dtype)
            phase = None if phases is None else torch.from_numpy(np.asarray(phases[index]))
            z01_true = float(labels["z01"][index])
            z01_guess = float(labels["z01_guess"][index])
            geometry = dataset_config.geometry.at(z01_guess)
            simplex = None
            if warm_start is not None:
                bounds = (z01_guess - search.confidence, z01_guess + search.confidence)
                simplex = warm_start.simplex(index, *bounds)
            result = find_focus(
                hologram,
                geometry,
                settings,
                confidence=search.confidence,
                xatol=search.xatol,
                fatol=search.fatol,
                max_evaluations=search.max_evaluations,
                initial_simplex=simplex,
            )
            mfe_true, truth = None, None
            if config.evaluate_truth:
                truth = asrm_reconstruct(
                    hologram, dataset_config.geometry.at(z01_true).fresnel_number, settings
                )
                mfe_true = truth.final_loss
            phase = None if phase is None else phase.to(torch_device)
            error_um = (result.z01 - z01_true) * 1e6
            recon_seconds = [e.seconds for e in result.evaluations if e.reconstructed]
            row = {
                "index": index,
                "phantom_type": int(labels["phantom_type"][index]),
                "photons": float(labels["photons"][index]),
                "z01_true": z01_true,
                "z01_guess": z01_guess,
                "dz01_true": float(labels["dz01"][index]),
                "z01_est": result.z01,
                "error_um": error_um,
                "abs_error_um": abs(error_um),
                "n_reconstructions": result.n_reconstructions,
                "n_evaluations": result.n_evaluations,
                "nm_iterations": result.nm_iterations,
                "converged": result.converged,
                "seconds": result.seconds,
                "seconds_per_reconstruction": float(np.mean(recon_seconds)),
                "mfe_est": result.mfe,
                "mfe_true": mfe_true,
                "phase_rmse_est": phase_rmse(result.reconstruction, phase),
                "phase_rmse_true": phase_rmse(truth, phase),
            }
            append_row(results_file, row)
            append_trace(traces_file, {"index": index, **result.to_dict()})
            summary = summarize(read_rows(results_file))
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            if log is not None:
                log(
                    f"[{count}/{len(todo)}] sample {index}: dz01 {row['dz01_true'] * 1e3:+.3f} mm, "
                    f"error {error_um:+.1f} um, {result.n_reconstructions} reconstructions, "
                    f"{result.seconds:.0f} s"
                )
    summary = summarize(read_rows(results_file))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
