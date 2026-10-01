"""PyTorch dataset and data loaders for HoloForge HDF5 hologram files.

Each sample is ``(hologram, target)`` with ``hologram`` of shape ``[1, H, W]`` (float32) and a scalar
float32 ``target``. HoloForge stores the detector-plane *amplitude* ``|psi|`` (not the intensity) in
``images/hologram``; the normalisation modes below operate on that array as stored.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.representations import RepresentationTransform, build_representation
from src.utils.config import require_keys, resolve_path
from src.utils.physics import TARGET_MODES

__all__ = [
    "NORMALIZATIONS",
    "HologramHDF5Dataset",
    "dataset_kwargs_from_config",
    "make_dataloaders",
    "normalize_hologram",
]

NORMALIZATIONS: tuple[str, ...] = ("standardize", "divide_mean", "log", "none")
"""Per-image input normalisations: (x-mean)/std, x/mean (flat-field like), log(max(x, eps)), or raw."""

_LOG_EPS = 1e-6
_SETUP_GROUP = "metadata/setup"
_FLOAT32_SIGNIFICANT_DIGITS = 7


def _round_significant(values: np.ndarray, digits: int = _FLOAT32_SIGNIFICANT_DIGITS) -> np.ndarray:
    """Undo float32 storage noise of HoloForge labels (e.g. 238.4199981689453 -> 238.42) by rounding to
    ``digits`` significant digits; exact for distances drawn with 1 µm resolution below 1000 mm."""
    return np.array([float(f"{value:.{digits}g}") for value in values], dtype=np.float64)


def normalize_hologram(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply a per-image normalisation to a ``[1, H, W]`` (or ``[H, W]``) tensor."""
    if mode == "standardize":
        std = x.std()
        return (x - x.mean()) / (std if std > 0 else torch.ones_like(std))
    if mode == "divide_mean":
        mean = x.mean()
        return x / (mean if mean != 0 else torch.ones_like(mean))
    if mode == "log":
        return torch.log(torch.clamp(x, min=_LOG_EPS))
    if mode == "none":
        return x
    raise ValueError(f"unknown normalization {mode!r}; expected one of {NORMALIZATIONS}")


class HologramHDF5Dataset(Dataset):
    """Holograms and geometry labels from a HoloForge HDF5 file.

    The HDF5 handle is opened lazily per process (checked via PID), so the dataset can be used with
    ``DataLoader(num_workers > 0)`` under the fork start method.

    Args:
        path: HDF5 file written by HoloForge / :mod:`src.data.generate_data`.
        target_mode: ``"log_fr"`` (natural log of Fr, default), ``"fr"`` or ``"z01_mm"``.
        normalization: One of :data:`NORMALIZATIONS`.
        crop_size: Optional centre crop (pixels, applied before downsampling).
        downsample: Integer average-pooling factor (1 = none).
        hologram_key: Dataset inside the file to use as input (``images/hologram`` or ``images/gt_hologram``).
        representation: Network input derived from the preprocessed hologram, one of
            :data:`src.data.representations.REPRESENTATIONS` (``hologram`` = unchanged behaviour,
            ``log_power_spectrum``, ``hologram+spectrum`` (2 channels), ``radial_profile`` (1-D vector)).
        representation_kwargs: Options forwarded to :func:`src.data.representations.build_representation`
            (``window``, ``log_scale``, ``n_bins``, ``axis``).
    """

    def __init__(
        self,
        path: str | Path,
        target_mode: str = "log_fr",
        normalization: str = "standardize",
        crop_size: int | None = None,
        downsample: int = 1,
        hologram_key: str = "images/hologram",
        representation: str = "hologram",
        representation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.path}")
        if target_mode not in TARGET_MODES:
            raise ValueError(f"unknown target_mode {target_mode!r}; expected one of {TARGET_MODES}")
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"unknown normalization {normalization!r}; expected one of {NORMALIZATIONS}")
        if downsample < 1 or int(downsample) != downsample:
            raise ValueError(f"downsample must be a positive integer, got {downsample!r}")
        if crop_size is not None and crop_size < 1:
            raise ValueError(f"crop_size must be positive, got {crop_size!r}")

        self.target_mode = target_mode
        self.normalization = normalization
        self.crop_size = crop_size
        self.downsample = int(downsample)
        self.hologram_key = hologram_key
        self.transform: RepresentationTransform = build_representation(representation, **(representation_kwargs or {}))
        self._h5: h5py.File | None = None
        self._pid: int | None = None

        with h5py.File(self.path, "r") as handle:
            if hologram_key not in handle:
                raise KeyError(f"{hologram_key!r} not found in {self.path}; available: {list(handle.get('images', {}).keys())}")
            if _SETUP_GROUP not in handle:
                raise KeyError(f"{_SETUP_GROUP!r} missing in {self.path}; was the dataset generated with store_setup=True?")
            shape = handle[hologram_key].shape
            setup = handle[_SETUP_GROUP]
            self._setup_arrays = {key: setup[key][...].astype(np.float64) for key in setup.keys()}
        if len(shape) != 3:
            raise ValueError(f"expected (N, H, W) holograms, got shape {shape}")
        for key in ("z01", "z02", "Fr", "energy", "detector_px_size"):
            if key not in self._setup_arrays:
                raise KeyError(f"setup label {key!r} missing in {self.path}")
        self._num_samples, self._height, self._width = (int(s) for s in shape)
        if crop_size is not None and crop_size > min(self._height, self._width):
            raise ValueError(f"crop_size {crop_size} exceeds hologram size {(self._height, self._width)}")

        self.z01_mm: np.ndarray = _round_significant(self._setup_arrays["z01"])
        self.z02_mm: np.ndarray = _round_significant(self._setup_arrays["z02"])
        self.fr: np.ndarray = self._setup_arrays["Fr"]
        self.targets: np.ndarray = self._compute_targets().astype(np.float32)

    def _compute_targets(self) -> np.ndarray:
        if self.target_mode == "log_fr":
            return np.log(self.fr)
        if self.target_mode == "fr":
            return self.fr
        return self.z01_mm

    def _file(self) -> h5py.File:
        pid = os.getpid()
        if self._h5 is None or self._pid != pid:
            self._h5 = h5py.File(self.path, "r", swmr=False)
            self._pid = pid
        return self._h5

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self._num_samples
        if not 0 <= index < self._num_samples:
            raise IndexError(f"index {index} out of range for dataset of size {self._num_samples}")
        raw = self._file()[self.hologram_key][index]
        image = torch.from_numpy(np.asarray(raw, dtype=np.float32)).unsqueeze(0)
        image = self.preprocess(image)
        return image, torch.tensor(self.targets[index], dtype=torch.float32)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Centre-crop, average-pool, normalise and transform a ``[1, H, W]`` float tensor into the configured
        representation (also usable at inference)."""
        if self.crop_size is not None:
            _, height, width = image.shape
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2
            image = image[:, top : top + self.crop_size, left : left + self.crop_size]
        if self.downsample > 1:
            image = F.avg_pool2d(image.unsqueeze(0), kernel_size=self.downsample).squeeze(0)
        return self.transform(image, normalize_hologram(image, self.normalization))

    @property
    def representation(self) -> str:
        """Name of the input representation (``hologram`` by default)."""
        return self.transform.representation.value

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Shape ``(1, H, W)`` of the cropped/downsampled hologram before the representation transform."""
        height = (self.crop_size if self.crop_size is not None else self._height) // self.downsample
        width = (self.crop_size if self.crop_size is not None else self._width) // self.downsample
        return (1, height, width)

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of the tensors returned by ``__getitem__``: ``(C, H, W)`` for image representations
        (``C`` = 1 or 2), ``(n_bins,)`` for the radial profile."""
        return self.transform.output_shape(self.image_shape)

    @property
    def geometry(self) -> dict[str, np.ndarray]:
        """Per-sample arrays ``z01_mm``, ``z02_mm`` and ``fr`` (float64), aligned with the sample index."""
        return {"z01_mm": self.z01_mm, "z02_mm": self.z02_mm, "fr": self.fr}

    @property
    def setup_constants(self) -> dict[str, Any]:
        """Setup values that are constant across the file: ``energy_kev``, ``px_mm``, ``z02_mm`` (if fixed),
        ``detector_size``, ``padding_factor``, ``downsample_factor``. A varying z02 is reported as ``z02_mm_range``."""
        consts: dict[str, Any] = {
            "energy_kev": self._constant("energy"),
            "px_mm": self._constant("detector_px_size"),
            "detector_size": int(self._constant("detector_size")),
            "padding_factor": int(self._constant("padding_factor")),
            "downsample_factor": int(self._constant("downsample_factor")),
        }
        z02 = self.z02_mm
        if np.all(z02 == z02[0]):
            consts["z02_mm"] = float(z02[0])
        else:
            consts["z02_mm_range"] = [float(z02.min()), float(z02.max())]
        return consts

    def _constant(self, key: str) -> float:
        values = self._setup_arrays[key]
        if not np.all(values == values[0]):
            raise ValueError(f"setup value {key!r} varies across samples in {self.path}")
        return float(_round_significant(values[:1])[0])

    def summary(self) -> dict[str, Any]:
        """Human-readable description used in logs and ``meta``/``results`` files."""
        return {
            "path": str(self.path),
            "num_samples": len(self),
            "hologram_shape": (self._height, self._width),
            "output_shape": self.output_shape,
            "target_mode": self.target_mode,
            "normalization": self.normalization,
            "representation": self.transform.to_dict(),
            "z01_mm_range": [float(self.z01_mm.min()), float(self.z01_mm.max())],
            "fr_range": [float(self.fr.min()), float(self.fr.max())],
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_pid"] = None
        return state


def dataset_kwargs_from_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments of :class:`HologramHDF5Dataset` from the ``data`` section of an experiment config."""
    return {
        "target_mode": data_cfg.get("target_mode", "log_fr"),
        "normalization": data_cfg.get("normalization", "standardize"),
        "crop_size": data_cfg.get("crop_size"),
        "downsample": data_cfg.get("downsample", 1),
        "hologram_key": data_cfg.get("hologram_key", "images/hologram"),
        "representation": data_cfg.get("representation", "hologram"),
        "representation_kwargs": data_cfg.get("representation_kwargs") or {},
    }


def make_dataloaders(
    config: dict[str, Any],
    data_dir: str | Path | None = None,
    seed: int | None = None,
) -> tuple[dict[str, DataLoader], dict[str, HologramHDF5Dataset]]:
    """Build train/val/test loaders from the ``data`` section of an experiment config.

    Expects ``<data_dir>/{train,val,test}.hdf5``. Only the training loader shuffles (seeded).

    Args:
        config: Full experiment config with a ``data`` section (``dir``, ``batch_size``, ``num_workers``, ...).
        data_dir: Overrides ``config['data']['dir']``.
        seed: Seed for the shuffling generator of the training loader.

    Returns:
        ``(loaders, datasets)`` dictionaries keyed by split name.
    """
    require_keys(config, ("data",), "<root>")
    data_cfg = config["data"]
    directory = resolve_path(data_dir if data_dir is not None else data_cfg.get("dir", ""))
    if not directory.is_dir():
        raise FileNotFoundError(f"data directory not found: {directory}")
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    kwargs = dataset_kwargs_from_config(data_cfg)

    datasets: dict[str, HologramHDF5Dataset] = {}
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        path = directory / f"{split}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"missing split file {path}")
        datasets[split] = HologramHDF5Dataset(path, **kwargs)
        generator = None
        if split == "train":
            generator = torch.Generator()
            generator.manual_seed(seed if seed is not None else 0)
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            generator=generator,
            persistent_workers=num_workers > 0,
        )
    return loaders, datasets
