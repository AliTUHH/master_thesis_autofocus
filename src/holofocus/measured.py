"""Measured holograms of the Zenodo record of [1] (doi 10.5281/zenodo.8349365).

Flat-field corrected holograms of the P05 beamline (PETRA III), 2048^2 pixels of 6.5 um, in
``livereco_server/Examples/data/holograms/*.tiff``. Setups (energy, ``z02``, ``z01`` and the
flat-field offset ``A0``) follow the example scripts of the record
(``Examples/alg5_combined``) and [1] Tables 3-4; the search interval, the result of the MFE
autofocus and the visually determined focus are those of [2] Table 1. The record does not
contain the flow-cell measurement (v) of [2] Table 1.

Lengths in metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .geometry import P05_DETECTOR_SHAPE, P05_PIXEL_SIZE, ConeBeamGeometry


@dataclass(frozen=True)
class MeasuredSample:
    """One measured hologram with the reference values of [1] and [2].

    ``padding_factor`` is the array size of [2] Table 1 relative to the detector (type 1:
    14336^2 = 7 x 2048^2, type 2: 8192^2 = 4 x 2048^2). ``cutting_band`` is the number of
    detector pixels removed at every border before processing (``Padding.cutting_band`` of
    the example script, only used for the tooth).
    """

    name: str
    label: str
    energy_kev: float
    z02: float
    z01_example: float
    a0: float
    padding_factor: float
    simplex: tuple[float, float]
    z01_nm: float
    z01_visual: float
    nm_samples: int
    cutting_band: int = 0

    @property
    def z01_guess(self) -> float:
        """Centre of the initial simplex of [2] Table 1."""
        return 0.5 * (self.simplex[0] + self.simplex[1])

    @property
    def confidence(self) -> float:
        """Half width of ``Omega_z`` (5 mm for every sample of [2] Table 1)."""
        return 0.5 * (self.simplex[1] - self.simplex[0])

    def geometry(self, binning: int = 1, z01: float | None = None) -> ConeBeamGeometry:
        """Setup at ``z01`` (default: the guess) for the detector binned by ``binning``."""
        z01 = self.z01_guess if z01 is None else z01
        return ConeBeamGeometry(self.energy_kev, z01, self.z02, P05_PIXEL_SIZE * binning)

    def padding_for(self, padding_factor: float | None = None) -> float:
        """Padding factor for the cut hologram that keeps the array size of the full detector.

        The reference code removes the cutting band but sizes the arrays by the uncut
        detector; with the cut hologram as input the factor grows by ``n / (n - 2 band)``.
        """
        p = self.padding_factor if padding_factor is None else padding_factor
        n = P05_DETECTOR_SHAPE[0]
        return p * n / (n - 2 * self.cutting_band)


MEASURED_SAMPLES: dict[str, MeasuredSample] = {
    s.name: s
    for s in (
        MeasuredSample(
            "spider_hair", "spider hair", 11.0, 19.661, 79.95e-3, 0.98, 7.0,
            (74.8e-3, 84.8e-3), 79.4e-3, 79.4e-3, 13,
        ),
        MeasuredSample(
            "tooth", "tooth", 17.0, 19.652, 81.7085e-3, 0.96, 4.0,
            (77.9e-3, 87.9e-3), 80.9e-3, 81.0e-3, 11, cutting_band=80,
        ),
        MeasuredSample(
            "cactus_needle", "cactus needle", 17.0, 19.652, 285.415625e-3, 0.98, 7.0,
            (281.1e-3, 291.1e-3), 284.2e-3, 284.2e-3, 12,
        ),
        MeasuredSample(
            "magnesium_wire", "Mg wire", 11.0, 19.661, 470.515625e-3, 1.1, 4.0,
            (467.4e-3, 477.4e-3), 470.8e-3, 470.7e-3, 9,
        ),
    )
}  # fmt: skip
"""The four holograms of the record in the order of [2] Table 1 (i)-(iv)."""


def bin_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Mean over ``factor x factor`` blocks (detector binning of a flat-field corrected image)."""
    if factor == 1:
        return image
    h, w = image.shape
    if h % factor or w % factor:
        raise ValueError(f"image shape {image.shape} not divisible by {factor}")
    return image.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def load_measured(
    directory: str | Path, sample: MeasuredSample | str, binning: int = 4
) -> torch.Tensor:
    """Hologram ``<directory>/<name>.tiff`` with the cutting band removed, binned by the mean."""
    from PIL import Image

    if isinstance(sample, str):
        sample = MEASURED_SAMPLES[sample]
    image = np.asarray(Image.open(Path(directory) / f"{sample.name}.tiff"), dtype=np.float64)
    band = sample.cutting_band
    if band:
        image = image[band:-band, band:-band]
    return torch.from_numpy(bin_mean(image, binning).astype(np.float32))
