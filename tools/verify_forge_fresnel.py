"""Verify that HoloForge propagates holograms with exactly the Fresnel number it labels.

Background: on noise-free *weak* Mg phantoms (``configs/data_small_weak.yaml``) the CTF ring fit
(``src.baseline.ctf_ringfit``) measures ~0.4 % less Fr than the stored label, growing with Fr.  This
script separates the possible causes with controlled experiments:

A. label arithmetic (``calc_Fr`` vs. kernel pixel size, downsampling, hc constant, rounding, float32);
B. wave-field level: the HoloForge hologram of a controlled phantom vs. an independent float64 Fresnel
   propagator applied to the *identical* exit wave with the *label* Fr (rel. L2, max. amplitude difference,
   1-D minimisation of the mismatch over Fr -> Fr_sim / Fr_label);
C. every post-propagation stage (modulus, centre crop, intensity convention, noise, float32 storage) and
   the ring-fit Fr / label - 1 after each stage, for a linear-regime object and a nominal (Mg 2 um) object;
D. geometry dependence (binning 8 -> 256 px, no binning 512 px, binning 4, padding 4): ring-fit bias of the
   exact forward model vs. the linearised (first-order CTF) model of the same objects, and the linear
   scaling (1 + eps) the measured offset would correspond to, compared with 1/N-type candidates;
E. optionally a real HoloForge HDF5 dataset with stored phantoms (``--hdf5`` or ``--forge-dataset N``):
   stored gt_hologram vs. float64 reference of the stored phantom + label, ring fit on the stored hologram,
   on the exact reference, on the linearised model and on the object scaled by 0.5 / 2.

Usage::

    /tmp/holo311/bin/python tools/verify_forge_fresnel.py [--geometry all] [--n-objects 4]
        [--forge-dataset 100 | --hdf5 data/processed/small_weak_ph/test.hdf5] [--json reports/forge_label_verification.json]

Everything runs on the CPU with two threads; the default run takes a few minutes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py  # noqa: E402
import holowizard.forge.generators  # noqa: E402,F401  (import first: resolves HoloForge's circular imports)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from holowizard.forge.experiment.probe.probe import Probe  # noqa: E402
from holowizard.forge.experiment.simulation import NFHSimulation  # noqa: E402
from holowizard.forge.objects.phantom import Phantom  # noqa: E402
from holowizard.forge.utils import calc_Fr  # noqa: E402
from holowizard.forge.utils.utilities import crop_center  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from src.baseline.ctf_ringfit import RingFitConfig, ring_fit  # noqa: E402
from src.data.forge_setup import NFHRandomDistSetup  # noqa: E402
from src.utils.physics import HC_KEV_NM, fresnel_number  # noqa: E402

torch.set_num_threads(2)

HC_CODATA_KEV_NM = 1.23984198
MG_BETA_OVER_DELTA_11KEV = 0.00834  # xraylib, Mg at 11 keV; HoloForge uses A = 2*beta*k*t, i.e. A/|phi| = 2*beta/delta

__all__ = [
    "Geometry",
    "GEOMETRIES",
    "fresnel_propagate",
    "controlled_phantom",
    "forge_simulate",
    "ForgeSample",
    "linearised_hologram",
    "ringfit_offset_pct",
    "label_formula_checks",
    "propagation_check",
    "stage_check",
    "geometry_study",
    "dataset_check",
    "main",
]


# --------------------------------------------------------------------------------------------------
# geometry and physics helpers
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Geometry:
    """Detector geometry as given to HoloForge (physical detector, before binning)."""

    name: str
    detector_size: int
    detector_px_mm: float
    downsample: int
    padding: int
    energy_kev: float = 11.0
    z02_mm: float = 20000.0
    z01_list_mm: tuple[float, ...] = (60.0, 132.0, 200.0, 290.0)

    @property
    def n_pixels(self) -> int:
        return self.detector_size // self.downsample

    @property
    def grid(self) -> int:
        return self.n_pixels * self.padding

    @property
    def px_eff_mm(self) -> float:
        return self.detector_px_mm * self.downsample

    def fr(self, z01: float) -> float:
        return float(fresnel_number(z01, self.z02_mm, self.energy_kev, self.px_eff_mm))

    def describe(self) -> str:
        frs = [self.fr(z) for z in self.z01_list_mm]
        return (
            f"{self.name}: detector {self.detector_size} px / {self.downsample} -> N = {self.n_pixels} px, "
            f"px_eff = {self.px_eff_mm:.4f} mm, padding {self.padding} -> grid {self.grid}, "
            f"Fr in [{min(frs):.3e}, {max(frs):.3e}] (sampling limit 1/grid = {1 / self.grid:.3e})"
        )


GEOMETRIES: dict[str, Geometry] = {
    # (i) configs/data_small.yaml: P05 detector binned by 8
    "small": Geometry("small (2048/8 -> 256, pad 2)", 2048, 0.0065, 8, 2),
    # (ii) no binning, same effective pixel and Fr range but N = 512
    "nodown": Geometry("nodown (512/1 -> 512 @ 52 um, pad 2)", 512, 0.052, 1, 2),
    # (iii) binning 4: N = 512, px_eff 26 um -> Fr / 4 (z01 chosen above the sampling limit 1/1024)
    "ds4": Geometry("ds4 (2048/4 -> 512, pad 2)", 2048, 0.0065, 4, 2, z01_list_mm=(100.0, 200.0, 290.0)),
    # (iv) padding 4 on the small geometry: N = 256, grid 1024
    "pad4": Geometry("pad4 (2048/8 -> 256, pad 4)", 2048, 0.0065, 8, 4),
}


def fresnel_propagate(psi: np.ndarray, fr: float) -> np.ndarray:
    """Reference transfer-function propagator: ``ifft2(fft2(psi) * exp(-i*pi/Fr*(xi^2+eta^2)))`` in float64.

    Frequencies are ``fftfreq(N)`` in cycles/pixel (no ``d`` argument), i.e. exactly HoloForge's convention
    (``NFHSetup.create_kernel``), but evaluated in double precision with NumPy's FFT.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    n0, n1 = psi.shape
    f0 = np.fft.fftfreq(n0)
    f1 = np.fft.fftfreq(n1)
    u = f0[:, None] ** 2 + f1[None, :] ** 2
    return np.fft.ifft2(np.fft.fft2(psi) * np.exp(-1j * np.pi / fr * u))


def controlled_phantom(
    n_pixels: int, phi_max: float, seed: int, beta_over_delta: float = MG_BETA_OVER_DELTA_11KEV, n_shapes: int = 3
) -> np.ndarray:
    """Complex phantom ``phi + i*A`` (HoloForge convention: phase shift <= 0, absorption >= 0) on an N x N canvas.

    A few projected spheres / rectangles with Gaussian edge smoothing (sigma 1 px, like Forge's smoothing
    filter); ``phi_max`` is the peak phase shift in rad, the absorption follows Forge's ``A = 2*beta*k*t``.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n_pixels, :n_pixels]
    thickness = np.zeros((n_pixels, n_pixels))
    for _ in range(n_shapes):
        cx, cy = rng.uniform(n_pixels / 2 - n_pixels / 3, n_pixels / 2 + n_pixels / 3, 2)
        weight = rng.uniform(0.5, 1.0)
        if rng.random() < 0.5:
            radius = rng.uniform(n_pixels / 16, n_pixels / 6)
            chord = radius**2 - (xx - cx) ** 2 - (yy - cy) ** 2
            thickness += weight * np.sqrt(np.maximum(chord, 0.0)) / radius
        else:
            width, height = rng.uniform(n_pixels / 10, n_pixels / 4, 2)
            thickness += weight * ((np.abs(xx - cx) < width / 2) & (np.abs(yy - cy) < height / 2))
    thickness = gaussian_filter(thickness / max(thickness.max(), 1e-9), 1.0, truncate=1.0)
    phase = -phi_max * thickness
    absorption = 2.0 * beta_over_delta * phi_max * thickness
    return phase + 1j * absorption


@dataclass
class ForgeSample:
    """Everything HoloForge produces for one controlled phantom, plus the label it would store."""

    fr_label: float
    z01_mm: float
    z02_mm: float
    px_eff_mm: float
    psi_exit: np.ndarray
    """Exit wave on the padded simulation grid (complex128 copy of Forge's float32 tensor)."""
    psi_det: np.ndarray
    """Forge's propagated wave on the padded grid."""
    gt_hologram: np.ndarray
    """Forge's noise-free hologram: ``crop_center(|psi_det|)`` (what ``images/gt_hologram`` stores)."""
    kernel: np.ndarray
    phantom: np.ndarray


def forge_simulate(geometry: Geometry, z01_mm: float, phantom: np.ndarray) -> ForgeSample:
    """Run HoloForge's ``NFHSimulation`` (flat probe = 1, no flat-field) on ``phantom`` with the repo's setup class."""
    setup = NFHRandomDistSetup(
        detector_size=geometry.detector_size,
        detector_px_size=geometry.detector_px_mm,
        padding_factor=geometry.padding,
        downsample_factor=geometry.downsample,
        energy=geometry.energy_kev,
        z01_min=z01_mm,
        z01_max=z01_mm,
        z02_min=geometry.z02_mm,
        seed=0,
    )
    if setup.detector_size != phantom.shape[0]:
        raise ValueError(f"phantom must be {setup.detector_size} px for this geometry, got {phantom.shape}")
    grid = setup.probe_size
    forge_phantom = Phantom(torch.as_tensor(np.asarray(phantom, dtype=np.complex64)), setup.detector_size)
    probe = Probe(torch.ones((grid, grid), dtype=torch.float32), grid)
    simulation = NFHSimulation(setup)
    simulation.forward(forge_phantom, probe)  # draws (the fixed) z01, sets the label, propagates
    kernel = setup.create_kernel(setup.Fr).cpu().numpy()
    return ForgeSample(
        fr_label=float(setup.Fr),
        z01_mm=float(setup.z01),
        z02_mm=float(setup.z02),
        px_eff_mm=float(setup.detector_px_size),
        psi_exit=simulation.psi_exit.cpu().numpy().astype(np.complex128),
        psi_det=simulation.psi_det.cpu().numpy().astype(np.complex128),
        gt_hologram=simulation.gt_hologram.cpu().numpy().astype(np.float64),
        kernel=kernel,
        phantom=np.asarray(phantom, dtype=np.complex128),
    )


def _embed(phantom: np.ndarray, grid: int) -> np.ndarray:
    n = phantom.shape[0]
    out = np.zeros((grid, grid), dtype=np.complex128)
    lo = grid // 2 - n // 2
    out[lo : lo + n, lo : lo + n] = phantom
    return out


def _crop(image: np.ndarray, n: int) -> np.ndarray:
    grid = image.shape[0]
    lo = grid // 2 - n // 2
    return image[lo : lo + n, lo : lo + n]


def exact_hologram(phantom: np.ndarray, fr: float, grid: int, scale: float = 1.0) -> np.ndarray:
    """Float64 forward model ``|D_Fr(exp(i*scale*phi - scale*A))|`` cropped to the phantom size."""
    obj = _embed(phantom, grid) * scale
    psi = fresnel_propagate(np.exp(1j * obj.real - obj.imag), fr)
    return _crop(np.abs(psi), phantom.shape[0])


def linearised_hologram(phantom: np.ndarray, fr: float, grid: int) -> np.ndarray:
    """First-order (weak-object / CTF) model ``1 - Im D_Fr(phi) - Re D_Fr(A)`` of the same phantom."""
    obj = _embed(phantom, grid)
    amplitude = 1.0 - np.imag(fresnel_propagate(obj.real, fr)) - np.real(fresnel_propagate(obj.imag, fr))
    return _crop(amplitude, phantom.shape[0])


def ringfit_offset_pct(hologram: np.ndarray, fr_label: float, margin: float = 2.0, **options: Any) -> tuple[float, float]:
    """``(template, minima)`` ring-fit estimates relative to the label in percent (search range label / margin .. * margin)."""
    result = ring_fit(hologram, RingFitConfig(fr_min=fr_label / margin, fr_max=fr_label * margin, **options))
    return (result.fr / fr_label - 1.0) * 100.0, (result.fr_minima / fr_label - 1.0) * 100.0


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _fmt(value: float, digits: int = 3) -> str:
    return "nan" if not np.isfinite(value) else f"{value:+.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join([line, sep, *body])


# --------------------------------------------------------------------------------------------------
# A. label arithmetic
# --------------------------------------------------------------------------------------------------
def label_formula_checks(geometry: Geometry) -> dict[str, Any]:
    """Compare ``calc_Fr`` with the repo formula, the fine-grid Fr x s^2 identity, the hc constant and rounding."""
    z01, z02, energy = geometry.z01_list_mm[1], geometry.z02_mm, geometry.energy_kev
    px_fine, s = geometry.detector_px_mm, geometry.downsample
    setup = NFHRandomDistSetup(geometry.detector_size, px_fine, geometry.padding, s, energy, z01, z01, z02, seed=0)
    fr_setup = float(setup.Fr)
    fr_calc = float(calc_Fr(energy, z01, z02, setup.detector_px_size))
    fr_repo = float(fresnel_number(z01, z02, energy, geometry.px_eff_mm))
    fr_fine = float(calc_Fr(energy, z01, z02, px_fine))
    kernel = setup.create_kernel(fr_setup).cpu().numpy()
    f = np.fft.fftfreq(setup.probe_size)
    kernel_ref = np.exp(-1j * np.pi / fr_setup * (f[:, None] ** 2 + f[None, :] ** 2))
    lam_forge, lam_codata = HC_KEV_NM / energy, HC_CODATA_KEV_NM / energy
    # worst case of HoloWizard's 1 um rounding of z01 at the smallest z01 of the data_small range (all lengths in mm)
    z01_small = 50.0 + 0.0004
    fr_exact = geometry.px_eff_mm**2 * z01_small / ((lam_forge * 1e-6) * z02 * (z02 - z01_small))
    fr_rounded = float(fresnel_number(z01_small, z02, energy, geometry.px_eff_mm))
    checks = {
        "setup.Fr (label) == calc_Fr(E, z01, z02, px*s)": fr_setup / fr_calc - 1.0,
        "setup.Fr == src.utils.physics.fresnel_number": fr_setup / fr_repo - 1.0,
        "Fr(fine px) * s^2 == Fr(coarse px) (label of binned grid)": fr_fine * s**2 / fr_setup - 1.0,
        "kernel pixel size == label pixel size (setup.detector_px_size = px*s)": setup.detector_px_size / geometry.px_eff_mm - 1.0,
        "Forge kernel vs exp(-i pi/Fr (xi^2+eta^2)) max |diff| (float32 kernel)": float(np.abs(kernel - kernel_ref).max()),
        "hc = 1.2398 vs CODATA 1.23984198: rel. Fr change": lam_forge / lam_codata - 1.0,
        "1 um rounding of z01 = 50.0004 mm: rel. Fr change": fr_rounded / fr_exact - 1.0,
        "float32 storage of Fr: rel. change (sample)": float(np.float32(fr_setup)) / fr_setup - 1.0,
    }
    return {"z01_mm": z01, "z02_mm": z02, "energy_kev": energy, "fr": fr_setup, "checks": checks}


# --------------------------------------------------------------------------------------------------
# B. wave-field level comparison
# --------------------------------------------------------------------------------------------------
def propagation_check(sample: ForgeSample, scan: tuple[float, ...] = (-0.4, -0.1, 0.0, 0.1, 0.4)) -> dict[str, Any]:
    """Forge's ``psi_det`` vs. reference propagation of the identical exit wave with the label Fr."""
    psi_ref = fresnel_propagate(sample.psi_exit, sample.fr_label)
    n = sample.gt_hologram.shape[0]
    grid = sample.psi_det.shape[0]
    lo = grid // 2 - n // 2
    # Forge's gt_hologram must be the integer, centred crop [lo:lo+n] of |psi_det| (crop_center) up to float32 rounding
    crop_forge = crop_center(torch.abs(torch.as_tensor(sample.psi_det.astype(np.complex64))), (n, n)).numpy().astype(np.float64)
    crop_diff = float(np.abs(crop_forge - sample.gt_hologram).max())
    crop_diff_direct = float(np.abs(np.abs(sample.psi_det)[lo : lo + n, lo : lo + n] - sample.gt_hologram).max())
    result: dict[str, Any] = {
        "rel_l2_psi": _rel_l2(sample.psi_det, psi_ref),
        "max_abs_amplitude_diff": float(np.abs(np.abs(sample.psi_det) - np.abs(psi_ref)).max()),
        "rel_l2_hologram_contrast": float(np.linalg.norm(sample.gt_hologram - np.abs(psi_ref)[lo : lo + n, lo : lo + n]) / np.linalg.norm(sample.gt_hologram - 1.0)),
        "crop_is_centred_integer": crop_diff < 1e-6 and crop_diff_direct < 1e-6,
        "crop_max_abs_diff": max(crop_diff, crop_diff_direct),
        "crop_start": lo,
        "kernel_max_abs_diff": float(np.abs(sample.kernel - np.exp(-1j * np.pi / sample.fr_label * (np.fft.fftfreq(grid)[:, None] ** 2 + np.fft.fftfreq(grid)[None, :] ** 2))).max()),
    }
    # sensitivity: how far off would the reference have to be to explain a 0.1 / 0.4 % label error?
    spectrum = np.fft.fft2(sample.psi_exit)
    f = np.fft.fftfreq(grid)
    u = f[:, None] ** 2 + f[None, :] ** 2

    def mismatch(fr: float) -> float:
        psi = np.fft.ifft2(spectrum * np.exp(-1j * np.pi / fr * u))
        return _rel_l2(sample.psi_det, psi)

    result["scan"] = {f"{d:+.1f}%": mismatch(sample.fr_label * (1 + d / 100)) for d in scan}
    opt = minimize_scalar(mismatch, bounds=(sample.fr_label * 0.98, sample.fr_label * 1.02), method="bounded", options={"xatol": sample.fr_label * 1e-9})
    result["fr_fit_over_label_minus_1"] = float(opt.x / sample.fr_label - 1.0)
    result["min_rel_l2"] = float(opt.fun)
    return result


# --------------------------------------------------------------------------------------------------
# C. post-propagation stages
# --------------------------------------------------------------------------------------------------
def stage_check(sample: ForgeSample, noise_rel: float = 0.1, noise_realisations: int = 4, seed: int = 0) -> list[tuple[str, float, float]]:
    """Ring-fit offset (template, minima) in % after every stage between propagation and HDF5.

    HoloForge has no resampling stage at all: ``downsample_factor`` only coarsens the simulation grid
    (``Setup.__init__``: ``detector_size / s`` pixels of size ``px * s``), so the stages are the modulus,
    the centre crop, the float32 storage and the additive noise.  Noise is zero-mean and cannot move the
    CTF zeros; the last row lists mean and sd over ``noise_realisations`` of Gaussian noise with a standard
    deviation of ``noise_rel`` times the rms fringe contrast (scatter without bias).
    """
    rng = np.random.default_rng(seed)
    psi_ref = fresnel_propagate(sample.psi_exit, sample.fr_label)
    cropped = sample.gt_hologram
    noise_std = noise_rel * float(np.std(cropped - 1.0))
    stages = [
        ("reference float64 |psi| on the full padded grid", np.abs(psi_ref)),
        ("Forge |psi_det| on the full padded grid (float32 FFT)", np.abs(sample.psi_det)),
        ("crop_center -> images/gt_hologram (N px)", cropped),
        ("intensity convention |psi|^2 after crop (not used by Forge)", cropped**2),
        ("float32 HDF5 round trip of gt_hologram", cropped.astype(np.float32).astype(np.float64)),
    ]
    rows = []
    for name, image in stages:
        template, minima = ringfit_offset_pct(image, sample.fr_label)
        rows.append((name, template, minima))
    noisy_offsets = []
    for _ in range(noise_realisations):
        noisy = cropped + noise_std * rng.standard_normal(cropped.shape)
        noisy[noisy < 0] = 0  # HologramGenerator.create_hologram clips negatives
        noisy_offsets.append(ringfit_offset_pct(noisy, sample.fr_label)[0])
    rows.append((f"+ Gaussian noise std {noise_std:.1e} (= {noise_rel:.0%} of the fringe rms) on the amplitude, clip < 0: mean / sd of {noise_realisations}", float(np.mean(noisy_offsets)), float(np.std(noisy_offsets))))
    return rows


# --------------------------------------------------------------------------------------------------
# D. geometry study: exact vs. linearised forward model
# --------------------------------------------------------------------------------------------------
def geometry_study(geometry: Geometry, n_objects: int, phi_levels: tuple[float, ...] = (0.02, 0.33, 0.66)) -> dict[str, Any]:
    """Propagation agreement and ring-fit bias (exact vs. linear model) over Fr values and random objects."""
    n, grid = geometry.n_pixels, geometry.grid
    prop_l2, prop_fit = [], []
    bias: dict[str, dict[str, list[float]]] = {}
    paired: dict[str, dict[str, list[float]]] = {}
    linear_key = "linearised CTF model (first order)"
    for z01 in geometry.z01_list_mm:
        for k in range(n_objects):
            seed = 1000 * int(z01) + k
            phantom = controlled_phantom(n, phi_levels[1], seed)
            sample = forge_simulate(geometry, z01, phantom)
            fr = sample.fr_label
            key = f"{fr:.3e}"
            check = propagation_check(sample, scan=(0.0,))
            prop_l2.append(check["rel_l2_psi"])
            prop_fit.append(check["fr_fit_over_label_minus_1"])
            holograms = {f"forge gt_hologram (phi_max {phi_levels[1]:.2f})": sample.gt_hologram}
            for phi in phi_levels:
                holograms[f"exact model, phi_max {phi:.2f}"] = exact_hologram(phantom, fr, grid, scale=phi / phi_levels[1])
            holograms[linear_key] = linearised_hologram(phantom, fr, grid)
            offsets = {name: ringfit_offset_pct(image, fr)[0] for name, image in holograms.items()}
            for name, value in offsets.items():
                bias.setdefault(name, {}).setdefault(key, []).append(value)
                if name != linear_key:
                    # paired difference to the linear model of the *same* object: isolates the non-linear object term
                    paired.setdefault(name, {}).setdefault(key, []).append(value - offsets[linear_key])
    summary = {}
    for name, per_fr in bias.items():
        all_values = np.array([v for vals in per_fr.values() for v in vals])
        entry = {
            "mean_pct": float(all_values.mean()),
            "sd_pct": float(all_values.std()),
            "per_fr_mean_pct": {fr: float(np.mean(vals)) for fr, vals in per_fr.items()},
        }
        if name in paired:
            diffs = np.array([v for vals in paired[name].values() for v in vals])
            entry["minus_linear_mean_pct"] = float(diffs.mean())
            entry["minus_linear_sd_pct"] = float(diffs.std())
            entry["minus_linear_per_fr_pct"] = {fr: float(np.mean(vals)) for fr, vals in paired[name].items()}
        summary[name] = entry
    forge_mean = summary[f"forge gt_hologram (phi_max {phi_levels[1]:.2f})"]["mean_pct"]
    eps = forge_mean / 100.0 / 2.0  # a linear image scaling (1 + eps) changes Fr by ~2 eps
    candidates = {"1/N": 1 / n, "1/(N-1)": 1 / (n - 1), "1/grid": 1 / grid, "1/(grid-1)": 1 / (grid - 1), "1/(s*N)": 1 / (geometry.downsample * n)}
    return {
        "geometry": geometry.describe(),
        "n_pixels": n,
        "grid": grid,
        "num_samples": len(prop_l2),
        "propagation_rel_l2_max": float(max(prop_l2)),
        "propagation_fr_fit_minus_1_max_abs": float(max(abs(v) for v in prop_fit)),
        "ringfit": summary,
        "linear_scaling_eps_from_forge_offset": eps,
        "scaling_candidates": candidates,
    }


# --------------------------------------------------------------------------------------------------
# E. real HoloForge dataset with stored phantoms
# --------------------------------------------------------------------------------------------------
def dataset_check(h5_path: Path, num_samples: int | None = None, search: tuple[float, float] | None = None) -> dict[str, Any]:
    """Stored ``gt_hologram`` vs. float64 reference of the stored phantom + label; ring fit on model variants."""
    with h5py.File(h5_path, "r") as handle:
        if "images/phantoms" not in handle or "images/gt_hologram" not in handle:
            raise KeyError(f"{h5_path} needs images/phantoms and images/gt_hologram (store.phantom / store.gt_hologram)")
        fr = handle["metadata/setup/Fr"][...].astype(np.float64)
        probe_size = int(handle["metadata/setup/probe_size"][0])
        total = handle["images/gt_hologram"].shape[0]
        n = total if num_samples is None else min(int(num_samples), total)
        holograms = handle["images/gt_hologram"][:n].astype(np.float64)
        phantoms = handle["images/phantoms"][:n]
    fr = fr[:n]
    fr_min, fr_max = search if search else (fr.min() / 1.3, fr.max() * 1.3)
    variants = [
        "stored gt_hologram",
        "exact float64 reference (stored phantom, label Fr)",
        "linearised CTF model",
        "exact model, object x 0.5",
        "exact model, object x 2",
        "2nd-order transmission 1 + i phi - A - phi^2/2",
    ]
    offsets: dict[str, list[float]] = {name: [] for name in variants}
    minima: dict[str, list[float]] = {name: [] for name in variants}
    rel_l2 = []
    for i in range(n):
        phantom = phantoms[i].astype(np.complex128)
        obj = _embed(phantom, probe_size)
        second_order = _crop(np.abs(fresnel_propagate(1.0 + 1j * obj.real - obj.imag - 0.5 * obj.real**2, fr[i])), phantom.shape[0])
        images = {
            variants[0]: holograms[i],
            variants[1]: exact_hologram(phantom, fr[i], probe_size),
            variants[2]: linearised_hologram(phantom, fr[i], probe_size),
            variants[3]: exact_hologram(phantom, fr[i], probe_size, scale=0.5),
            variants[4]: exact_hologram(phantom, fr[i], probe_size, scale=2.0),
            variants[5]: second_order,
        }
        rel_l2.append(float(np.linalg.norm(images[variants[1]] - holograms[i]) / np.linalg.norm(holograms[i] - 1.0)))
        for name, image in images.items():
            result = ring_fit(image, RingFitConfig(fr_min=fr_min, fr_max=fr_max))
            offsets[name].append((result.fr / fr[i] - 1.0) * 100.0)
            minima[name].append((result.fr_minima / fr[i] - 1.0) * 100.0)
    terciles = np.percentile(fr, [100 / 3, 200 / 3])
    groups = [fr < terciles[0], (fr >= terciles[0]) & (fr < terciles[1]), fr >= terciles[1]]
    linear = np.array(offsets[variants[2]])
    summary = {}
    for name in variants:
        e = np.array(offsets[name])
        m = np.array(minima[name])
        d = e - linear
        summary[name] = {
            "template_bias_pct": float(e.mean()),
            "template_sd_pct": float(e.std()),
            "template_terciles_pct": [float(e[g].mean()) for g in groups],
            "minima_bias_pct": float(np.nanmean(m)),
            "slope_pct_per_fr": float(np.polyfit(fr, e, 1)[0]),
            "minus_linear_mean_pct": float(d.mean()),
            "minus_linear_sd_pct": float(d.std()),
        }
    phase = np.abs(phantoms.real).reshape(n, -1).max(axis=1)
    return {
        "file": str(h5_path),
        "num_samples": n,
        "fr_range": [float(fr.min()), float(fr.max())],
        "fr_terciles": [float(t) for t in terciles],
        "peak_phase_rad_mean": float(phase.mean()),
        "peak_phase_rad_max": float(phase.max()),
        "stored_vs_reference_rel_l2_contrast_max": float(max(rel_l2)),
        "stored_vs_reference_rel_l2_contrast_mean": float(np.mean(rel_l2)),
        "ringfit": summary,
    }


def generate_forge_dataset(num_samples: int, out_dir: Path) -> Path:
    """Generate ``num_samples`` holograms with ``configs/data_small_weak.yaml`` + stored phantoms (seed of the test split)."""
    from src.data.generate_data import generate_dataset
    from src.utils.config import load_yaml

    cfg = copy.deepcopy(load_yaml(PROJECT_ROOT / "configs" / "data_small_weak.yaml"))
    cfg["name"] = out_dir.name
    cfg["num_samples"] = {"train": 0, "val": 0, "test": int(num_samples)}
    cfg["store"]["phantom"] = True
    cfg["store"]["gt_hologram"] = True
    generate_dataset(cfg, out_dir, overwrite=True)
    import holowizard.forge.experiment as forge_experiment

    draws = getattr(forge_experiment.GLOBAL_EXP_SETUP, "num_draws", None)
    print(f"setup.kernel / get_distances() calls during generation: {draws} (expected {num_samples} + 1 initial draw)")
    return out_dir / "test.hdf5"


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--geometry", default="all", help="comma-separated subset of " + ",".join(GEOMETRIES) + " or 'all'")
    parser.add_argument("--n-objects", type=int, default=3, help="random objects per (geometry, z01)")
    parser.add_argument("--hdf5", default=None, help="HoloForge HDF5 with images/phantoms + images/gt_hologram")
    parser.add_argument("--forge-dataset", type=int, default=0, help="generate this many weak-object samples with HoloForge and check them")
    parser.add_argument("--num-samples", type=int, default=None, help="limit the dataset check to the first samples")
    parser.add_argument("--json", default=None, help="write all results to this JSON file")
    args = parser.parse_args(argv)

    names = list(GEOMETRIES) if args.geometry == "all" else [g.strip() for g in args.geometry.split(",")]
    results: dict[str, Any] = {"geometries": {}, "stages": {}, "label_formula": {}}
    t_start = time.perf_counter()

    print("=" * 100)
    print("A. label arithmetic (small geometry)")
    formula = label_formula_checks(GEOMETRIES["small"])
    results["label_formula"] = formula
    for name, value in formula["checks"].items():
        print(f"  {name:75s} {value:+.3e}")

    for name in names:
        geometry = GEOMETRIES[name]
        print("=" * 100)
        print(f"B/C. wave-field comparison and stages -- {geometry.describe()}")
        z01 = geometry.z01_list_mm[1]
        for phi_max, label in ((0.02, "linear-regime object, phi_max 0.02 rad"), (0.33, "nominal object, phi_max 0.33 rad (Mg 2 um)")):
            sample = forge_simulate(geometry, z01, controlled_phantom(geometry.n_pixels, phi_max, seed=7))
            check = propagation_check(sample)
            print(f"  [{label}] z01 = {z01} mm, Fr_label = {sample.fr_label:.6e}")
            print(f"    Forge psi_det vs reference(psi_exit, Fr_label): rel L2 = {check['rel_l2_psi']:.2e}, max |d amplitude| = {check['max_abs_amplitude_diff']:.2e}, "
                  f"kernel max |diff| = {check['kernel_max_abs_diff']:.1e}, crop centred/integer: {check['crop_is_centred_integer']} (start {check['crop_start']})")
            print("    rel L2 if the reference used Fr_label * (1+d): " + ", ".join(f"d={k}: {v:.2e}" for k, v in check["scan"].items()))
            print(f"    1-D fit of Fr minimising the mismatch: Fr_fit / Fr_label - 1 = {check['fr_fit_over_label_minus_1']:+.2e} (min rel L2 {check['min_rel_l2']:.2e})")
            rows = stage_check(sample)
            print("    " + _table(["stage", "ring fit Fr/label-1 [%] (template)", "(minima method; noise row: sd)"], [[r[0], _fmt(r[1]), _fmt(r[2])] for r in rows]).replace("\n", "\n    "))
            results["stages"][f"{name}/{label}"] = {"propagation": check, "stages": [{"stage": r[0], "template_pct": r[1], "minima_pct": r[2]} for r in rows]}

        print(f"D. geometry study -- {geometry.describe()}; {args.n_objects} objects x {len(geometry.z01_list_mm)} Fr values")
        study = geometry_study(geometry, args.n_objects)
        results["geometries"][name] = study
        print(f"  {study['num_samples']} samples; propagation: max rel L2 {study['propagation_rel_l2_max']:.2e}, max |Fr_fit/Fr_label - 1| {study['propagation_fr_fit_minus_1_max_abs']:.1e}")
        fr_keys = list(next(iter(study["ringfit"].values()))["per_fr_mean_pct"])
        headers = ["hologram model", "mean [%]", "sd [%]", *[f"Fr={fr}" for fr in fr_keys], "minus linear model: mean +- sd [%]"]
        rows = []
        for model, v in study["ringfit"].items():
            diff = f"{_fmt(v['minus_linear_mean_pct'])} +- {v['minus_linear_sd_pct']:.3f}" if "minus_linear_mean_pct" in v else "(reference)"
            rows.append([model, _fmt(v["mean_pct"]), f"{v['sd_pct']:.3f}", *[_fmt(x) for x in v["per_fr_mean_pct"].values()], diff])
        print("  " + _table(headers, rows).replace("\n", "\n  "))
        eps = study["linear_scaling_eps_from_forge_offset"]
        print(f"  linear scaling (1+eps) equivalent to the mean Forge offset: eps = {eps:+.2e}; candidates: " + ", ".join(f"{k} = {v:.2e}" for k, v in study["scaling_candidates"].items()))

    h5_path = Path(args.hdf5) if args.hdf5 else None
    if args.forge_dataset > 0:
        h5_path = generate_forge_dataset(args.forge_dataset, PROJECT_ROOT / "data" / "processed" / "verify_forge_weak")
    if h5_path is not None:
        print("=" * 100)
        print(f"E. HoloForge dataset {h5_path}")
        check = dataset_check(h5_path, args.num_samples)
        results["dataset"] = check
        print(f"  {check['num_samples']} samples, Fr in [{check['fr_range'][0]:.3e}, {check['fr_range'][1]:.3e}], peak phase shift mean {check['peak_phase_rad_mean']:.2f} rad (max {check['peak_phase_rad_max']:.2f})")
        print(f"  stored gt_hologram vs float64 reference of stored phantom + label: rel L2 of the contrast max {check['stored_vs_reference_rel_l2_contrast_max']:.1e}, mean {check['stored_vs_reference_rel_l2_contrast_mean']:.1e}")
        headers = ["hologram model", "bias [%]", "sd [%]", "Fr tercile 1", "tercile 2", "tercile 3", "slope [%/Fr]", "minima bias [%]", "minus linear model [%]"]
        rows = [
            [k, _fmt(v["template_bias_pct"]), f"{v['template_sd_pct']:.3f}", *[_fmt(t) for t in v["template_terciles_pct"]], f"{v['slope_pct_per_fr']:+.1f}", _fmt(v["minima_bias_pct"]), f"{_fmt(v['minus_linear_mean_pct'])} +- {v['minus_linear_sd_pct']:.3f}"]
            for k, v in check["ringfit"].items()
        ]
        print("  " + _table(headers, rows).replace("\n", "\n  "))

    results["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    print("=" * 100)
    print(f"done in {results['elapsed_s']} s")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
        print(f"results written to {out}")
    return results


if __name__ == "__main__":
    main()
