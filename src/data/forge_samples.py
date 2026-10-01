"""Plain reader for single HoloForge samples (hologram, optional phantom, geometry labels).

:class:`src.data.dataset.HologramHDF5Dataset` serves the ML pipeline (tensors, normalisation,
representations); the baselines and the downstream reconstruction test need the raw amplitude hologram
together with its ground-truth phantom and the per-sample geometry, which this module provides.

Conventions (see ``reports/baseline_model_based.md``):

* ``images/hologram`` holds the detector-plane **amplitude** ``|psi|`` (plus noise), *not* the intensity.
  HoloWizard's reconstruction/autofocus API expects intensities (it applies ``sqrt`` internally) ->
  :attr:`ForgeSample.hologram_intensity` squares the stored array.
* ``images/phantoms`` (if stored) is the complex object ``O = phi + i mu`` on the hologram grid: real part =
  phase shift in rad (``<= 0``), imaginary part = absorption (``>= 0``); the exit wave is ``exp(i O)``.
* ``metadata/setup/detector_px_size`` is the *effective* (binned) pixel size in mm and
  ``metadata/setup/detector_size`` the binned hologram size, so :func:`src.utils.physics.fresnel_number`
  applied to the stored values reproduces ``metadata/setup/Fr`` (float32 precision).
"""

from __future__ import annotations

import glob
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

__all__ = ["ForgeSample", "read_forge_samples", "iter_forge_samples", "expand_paths"]


@dataclass
class ForgeSample:
    """One hologram with its labels (units as in HoloWizard: mm, keV)."""

    hologram_amplitude: np.ndarray
    """Stored HoloForge hologram ``|psi_det|`` (+ noise), float32, shape ``(H, W)``."""
    phantom: np.ndarray | None
    """Complex object ``phi + i mu`` (complex64) or ``None`` if the file has no ``images/phantoms``."""
    fr: float
    """Stored pixel Fresnel number ``metadata/setup/Fr``."""
    z01_mm: float
    z02_mm: float
    energy_kev: float
    px_mm: float
    """Effective (binned) detector pixel size in mm."""
    padding_factor: float
    source: str = ""
    """File name of the HDF5 file."""
    index: int = -1

    @property
    def hologram_intensity(self) -> np.ndarray:
        """``|psi|^2`` as expected by ``holowizard.core`` (which takes the square root internally)."""
        return (self.hologram_amplitude.astype(np.float32)) ** 2

    @property
    def gt_phase(self) -> np.ndarray | None:
        """Ground-truth phase shift in rad (``<= 0``) or ``None`` without phantom."""
        return None if self.phantom is None else np.ascontiguousarray(self.phantom.real, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.hologram_amplitude.shape[0]), int(self.hologram_amplitude.shape[1])


def expand_paths(patterns: Iterable[str | Path]) -> list[Path]:
    """Expand glob patterns (sorted) and keep literal paths; raises if nothing matches."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if not matches and not Path(pattern).is_file():
            raise FileNotFoundError(f"no HDF5 file matches {pattern!r}")
        paths.extend(Path(m) for m in (matches or [str(pattern)]))
    return paths


def iter_forge_samples(
    paths: Iterable[str | Path],
    n_per_file: int | None = None,
    hologram_key: str = "images/hologram",
    indices: Iterable[int] | None = None,
) -> Iterator[ForgeSample]:
    """Yield the first ``n_per_file`` samples (or the given ``indices``) of every file."""
    wanted = None if indices is None else sorted(set(int(i) for i in indices))
    for path in expand_paths(paths):
        with h5py.File(path, "r") as handle:
            holograms = handle[hologram_key]
            phantoms = handle["images/phantoms"] if "images/phantoms" in handle else None
            setup = handle["metadata/setup"]
            total = holograms.shape[0]
            selection = [i for i in (wanted if wanted is not None else range(total)) if i < total]
            if n_per_file is not None:
                selection = selection[: int(n_per_file)]
            for i in selection:
                yield ForgeSample(
                    hologram_amplitude=holograms[i].astype(np.float32),
                    phantom=None if phantoms is None else phantoms[i].astype(np.complex64),
                    fr=float(setup["Fr"][i]),
                    z01_mm=float(setup["z01"][i]),
                    z02_mm=float(setup["z02"][i]),
                    energy_kev=float(setup["energy"][i]),
                    px_mm=float(setup["detector_px_size"][i]),
                    padding_factor=float(setup["padding_factor"][i]) if "padding_factor" in setup else 2.0,
                    source=path.name,
                    index=int(i),
                )


def read_forge_samples(
    paths: Iterable[str | Path],
    n_per_file: int | None = None,
    hologram_key: str = "images/hologram",
    indices: Iterable[int] | None = None,
) -> list[ForgeSample]:
    """List version of :func:`iter_forge_samples`."""
    return list(iter_forge_samples(paths, n_per_file=n_per_file, hologram_key=hologram_key, indices=indices))
