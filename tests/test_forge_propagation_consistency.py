"""HoloForge propagates with exactly the Fresnel number it labels; the ring fit is unbiased only for weak objects.

Background (see ``reports/forge_label_verification.md``): the CTF ring fit measured ~0.4 % less Fr than the
HoloForge label on noise-free Mg 1-2 um phantoms.  These tests pin down the verified facts:

* Forge's wave field equals an independent float64 Fresnel propagator applied to the identical exit wave
  with the *label* Fr (float32 precision), and the stored ``gt_hologram`` is the centred integer crop;
* for a real HoloForge dataset the stored hologram is reproduced from the stored phantom + label alone;
* the ring fit is unbiased (< 0.1 %) for objects in the linear (weak-object) regime;
* for the nominal Mg 1-2 um objects (0.17-0.66 rad) the exact forward model shifts the ring minima: the
  ring fit is biased by about -0.4 % (second-order object term, grows with phase shift and Fr).  This is a
  limitation of the CTF ring fit, not of the labels, and is recorded as a strict ``xfail``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import h5py
import holowizard.forge.experiment as forge_experiment
import numpy as np
import pytest

from src.data.generate_data import generate_dataset
from src.utils.config import PROJECT_ROOT, load_yaml
from src.utils.physics import fresnel_number
from tools.verify_forge_fresnel import (
    GEOMETRIES,
    controlled_phantom,
    exact_hologram,
    forge_simulate,
    fresnel_propagate,
    linearised_hologram,
    propagation_check,
    ringfit_offset_pct,
)

SMALL = GEOMETRIES["small"]  # configs/data_small.yaml geometry: 2048 px / 8 -> 256 px, 52 um, padding 2
NUM_WEAK_SAMPLES = 12


@pytest.fixture(scope="module")
def weak_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """First samples of ``configs/data_small_weak.yaml`` (same seed as the real test split) with stored phantoms."""
    cfg = copy.deepcopy(load_yaml(PROJECT_ROOT / "configs" / "data_small_weak.yaml"))
    cfg["name"] = "weak_verify"
    cfg["num_samples"] = {"train": 0, "val": 0, "test": NUM_WEAK_SAMPLES}
    cfg["store"]["phantom"] = True
    cfg["store"]["gt_hologram"] = True
    out_dir = tmp_path_factory.mktemp("weak_verify")
    generate_dataset(cfg, out_dir, overwrite=True)
    # the label of sample i must belong to the propagation of sample i: exactly one geometry draw per hologram
    assert forge_experiment.GLOBAL_EXP_SETUP.num_draws == NUM_WEAK_SAMPLES + 1
    return out_dir / "test.hdf5"


@pytest.mark.parametrize("z01_mm, phi_max", [(132.0, 0.33), (290.0, 0.02)])
def test_forge_wave_field_equals_reference_propagator_with_label_fr(z01_mm: float, phi_max: float) -> None:
    sample = forge_simulate(SMALL, z01_mm, controlled_phantom(SMALL.n_pixels, phi_max, seed=7))
    assert sample.fr_label == pytest.approx(fresnel_number(z01_mm, SMALL.z02_mm, SMALL.energy_kev, SMALL.px_eff_mm), rel=1e-12)
    check = propagation_check(sample, scan=(-0.4, 0.0))
    # identical exit wave, label Fr: agreement at float32 level (Forge uses complex64 FFTs)
    assert check["rel_l2_psi"] < 1e-5
    assert check["max_abs_amplitude_diff"] < 1e-4
    assert check["kernel_max_abs_diff"] < 1e-3
    # a 0.4 % Fr error would be two orders of magnitude more visible than the observed mismatch
    assert check["scan"]["-0.4%"] > 50 * check["rel_l2_psi"]
    assert abs(check["fr_fit_over_label_minus_1"]) < 1e-5
    # gt_hologram = centred integer crop of |psi_det| (crop_center), no resampling anywhere in Forge
    assert check["crop_is_centred_integer"]
    assert check["crop_start"] == SMALL.grid // 2 - SMALL.n_pixels // 2
    assert sample.gt_hologram.shape == (SMALL.n_pixels, SMALL.n_pixels)


def test_downsample_factor_only_coarsens_the_grid() -> None:
    """``downsample_factor`` s changes detector_size -> N/s and px -> s*px; the label uses the same px as the kernel."""
    fine = fresnel_number(132.0, SMALL.z02_mm, SMALL.energy_kev, SMALL.detector_px_mm)
    sample = forge_simulate(SMALL, 132.0, controlled_phantom(SMALL.n_pixels, 0.1, seed=1))
    assert sample.px_eff_mm == pytest.approx(SMALL.detector_px_mm * SMALL.downsample)
    assert sample.fr_label == pytest.approx(fine * SMALL.downsample**2, rel=1e-12)
    assert sample.psi_exit.shape == (SMALL.grid, SMALL.grid)


def test_stored_hologram_is_reproduced_from_stored_phantom_and_label(weak_dataset: Path) -> None:
    with h5py.File(weak_dataset, "r") as handle:
        fr = handle["metadata/setup/Fr"][...].astype(np.float64)
        probe_size = int(handle["metadata/setup/probe_size"][0])
        holograms = handle["images/gt_hologram"][...].astype(np.float64)
        phantoms = handle["images/phantoms"][...].astype(np.complex128)
    assert probe_size == 512 and holograms.shape == (NUM_WEAK_SAMPLES, 256, 256)
    for i in range(NUM_WEAK_SAMPLES):
        reference = exact_hologram(phantoms[i], fr[i], probe_size)
        rel_l2 = np.linalg.norm(reference - holograms[i]) / np.linalg.norm(holograms[i] - 1.0)
        assert rel_l2 < 1e-4, f"sample {i}: stored hologram deviates from the reference propagation ({rel_l2:.1e})"


def test_ring_fit_unbiased_for_linear_regime_objects_through_forge() -> None:
    """End-to-end through Forge with weak controlled objects (phi_max = 0.02 rad): |mean offset| < 0.1 %."""
    offsets = []
    for z01 in (90.0, 132.0, 220.0):
        for k in range(4):
            sample = forge_simulate(SMALL, z01, controlled_phantom(SMALL.n_pixels, 0.02, seed=100 * int(z01) + k))
            offsets.append(ringfit_offset_pct(sample.gt_hologram, sample.fr_label)[0])
    offsets = np.asarray(offsets)
    assert abs(offsets.mean()) < 0.1, offsets
    assert np.abs(offsets).max() < 0.3, offsets


def test_ring_fit_unbiased_for_linearised_model_of_forge_phantoms(weak_dataset: Path) -> None:
    """First-order CTF model of the real Mg 1-2 um phantoms with the stored label: no systematic offset."""
    with h5py.File(weak_dataset, "r") as handle:
        fr = handle["metadata/setup/Fr"][...].astype(np.float64)
        phantoms = handle["images/phantoms"][...].astype(np.complex128)
    offsets = np.asarray([ringfit_offset_pct(linearised_hologram(phantoms[i], fr[i], 512), fr[i])[0] for i in range(NUM_WEAK_SAMPLES)])
    assert abs(offsets.mean()) < 0.1, offsets


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CTF ring fit on the exact forward model of Mg 1-2 um phantoms (0.17-0.66 rad): the second-order "
        "object term -phi^2/2 acts as a frequency-dependent absorption and shifts the ring minima; measured "
        "offset about -0.4 % (-0.13 % -> -0.65 % over the Fr terciles of data_small_weak, growing with phase "
        "shift and Fr).  Labels are correct (see tests above); this documents the estimator's weak-object limit."
    ),
)
def test_ring_fit_matches_label_within_0p1_percent_for_nominal_mg_objects(weak_dataset: Path) -> None:
    with h5py.File(weak_dataset, "r") as handle:
        fr = handle["metadata/setup/Fr"][...].astype(np.float64)
        holograms = handle["images/gt_hologram"][...].astype(np.float64)
    offsets = np.asarray([ringfit_offset_pct(holograms[i], fr[i])[0] for i in range(NUM_WEAK_SAMPLES)])
    assert abs(offsets.mean()) < 0.1, f"mean ring-fit offset {offsets.mean():+.3f} % (per sample: {np.round(offsets, 3)})"


def test_nonlinear_object_term_explains_the_offset(weak_dataset: Path) -> None:
    """Paired per-object comparison: exact model minus linear model is clearly negative and grows with the object strength."""
    with h5py.File(weak_dataset, "r") as handle:
        fr = handle["metadata/setup/Fr"][...].astype(np.float64)
        phantoms = handle["images/phantoms"][...].astype(np.complex128)
    diffs: dict[float, list[float]] = {0.5: [], 1.0: [], 2.0: []}
    for i in range(NUM_WEAK_SAMPLES):
        linear = ringfit_offset_pct(linearised_hologram(phantoms[i], fr[i], 512), fr[i])[0]
        for scale, values in diffs.items():
            values.append(ringfit_offset_pct(exact_hologram(phantoms[i], fr[i], 512, scale=scale), fr[i])[0] - linear)
    means = {scale: float(np.mean(values)) for scale, values in diffs.items()}
    assert means[1.0] < -0.2, means
    assert means[2.0] < means[1.0] < means[0.5] < 0.0, means


def test_reference_propagator_is_unitary_and_matches_forge_convention() -> None:
    rng = np.random.default_rng(0)
    psi = np.exp(1j * 0.1 * rng.standard_normal((64, 64)))
    out = fresnel_propagate(psi, 0.01)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(psi), rel=1e-12)  # energy conservation
    back = fresnel_propagate(out, -0.01)
    np.testing.assert_allclose(back, psi, atol=1e-12)
