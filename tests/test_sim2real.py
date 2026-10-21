import math
from pathlib import Path

import numpy as np
import pytest
import torch

from holofocus.dataset import DatasetConfig, HologramGenerator
from holofocus.forward import object_from_phase, simulate_hologram
from holofocus.learned import LearnedAutofocus, LearnedConfig, build_model
from holofocus.learned.sim2real import (
    FEATURE_NAMES,
    MEASURED_MODELS,
    PERTURBATIONS,
    RadialPower,
    binning_phases,
    feature_distribution,
    measured_estimate,
    paired_shifts,
    percentile_rank,
    perturbation_changes,
    perturbed_config,
    spectral_features,
)
from holofocus.measured import MEASURED_SAMPLES, MeasuredSample, bin_mean

ROOT = Path(__file__).resolve().parents[1]
CI_DATASET = ROOT / "configs" / "dataset" / "ci.yaml"
CI_LEARNED = ROOT / "configs" / "learned" / "ci.yaml"


def test_measured_models_cover_the_measured_samples():
    assert set(MEASURED_MODELS) == set(MEASURED_SAMPLES)


@pytest.mark.parametrize("name", list(PERTURBATIONS))
def test_perturbations_change_only_their_keys(name):
    base = DatasetConfig.from_yaml(CI_DATASET)
    changes = perturbation_changes(name, "cactus_needle")
    config = perturbed_config(base, changes, f"ci-{name}")
    before, after = base.to_dict(), config.to_dict()
    for section, values in changes.items():
        assert set(values) <= set(before[section]), section
    assert after["name"] == f"ci-{name}" and after["seed"] == before["seed"]
    assert after["splits"] == before["splits"] and after["geometry"] == before["geometry"]
    for section in before:
        if section in ("name",):
            continue
        for key in before[section] if isinstance(before[section], dict) else ():
            expected = changes.get(section, {}).get(key, before[section][key])
            if isinstance(expected, tuple):
                expected = list(expected)
            assert after[section][key] == expected, (section, key)
    with pytest.raises(ValueError):
        perturbation_changes("gamma-rays", "tooth")
    with pytest.raises(ValueError):
        perturbed_config(base, {"optics": {"zoom": 2}}, "bad")


def test_combined_is_psf_noise_and_object():
    combined = perturbation_changes("combined", "spider_hair")
    assert combined["simulation"] == perturbation_changes("psf", "spider_hair")["simulation"]
    assert combined["noise"] == perturbation_changes("noise", "spider_hair")["noise"]
    assert combined["phantoms"] == perturbation_changes("object", "spider_hair")["phantoms"]


def test_perturbed_samples_pair_with_the_original():
    base = DatasetConfig.from_yaml(CI_DATASET)
    generators = {
        name: HologramGenerator(
            perturbed_config(base, perturbation_changes(name, "tooth"), name), device="cpu"
        )
        for name in PERTURBATIONS
    }
    reference = HologramGenerator(base, device="cpu")
    for index in (0, 3):
        ref = reference.sample("test", index)
        for name, generator in generators.items():
            sample = generator.sample("test", index)
            assert sample.labels["z01"] == ref.labels["z01"], name
            if name != "noise" and name != "combined":
                assert sample.meta["photons"] == ref.meta["photons"], name
            if name in ("psf", "flat-field", "noise"):
                np.testing.assert_array_equal(sample.phase, ref.phase)
        flat = generators["flat-field"].sample("test", index)
        residual = flat.hologram / ref.hologram - 1
        assert residual.std() == pytest.approx(MEASURED_MODELS["tooth"].flat_field, rel=1e-3)
        assert generators["noise"].sample("test", index).meta["photons"] == pytest.approx(
            MEASURED_MODELS["tooth"].photons
        )
        assert generators["psf"].sample("test", index).meta["psf_sigma"] == pytest.approx(1.6)


def _row(index, z01_est, z01_true=0.1):
    error = (z01_est - z01_true) * 1e6
    return {"index": index, "z01_est": z01_est, "z01_true": z01_true, "error_um": error}


def test_paired_shifts():
    reference = [_row(i, 0.1 + 1e-4 * (i - 2)) for i in range(5)]
    perturbed = [_row(i, r["z01_est"] + 50e-6) for i, r in enumerate(reference)][1:]
    out = paired_shifts(reference, perturbed)
    assert out["n"] == 4 and out["same_z01"]
    assert out["shift_um"]["mean"] == pytest.approx(50.0)
    assert out["shift_um"]["median"] == pytest.approx(50.0)
    assert out["shift_um"]["std"] == pytest.approx(0.0, abs=1e-6)
    assert out["reference_error_um"]["n"] == 4
    expected = np.mean([abs(r["error_um"]) for r in perturbed]) - np.mean(
        [abs(r["error_um"]) for r in reference[1:]]
    )
    assert out["mae_change_um"] == pytest.approx(expected)
    assert not paired_shifts(reference, [_row(0, 0.1, z01_true=0.2)])["same_z01"]
    with pytest.raises(ValueError):
        paired_shifts(reference, [_row(9, 0.1)])


def test_radial_power_of_white_noise_is_its_variance():
    rng = np.random.default_rng(0)
    radial = RadialPower((256, 256), bins=64, f2_max=0.25)
    variance = 4e-4
    profiles = [radial(1 + math.sqrt(variance) * rng.standard_normal((256, 256))) for _ in range(4)]
    level = np.exp(np.mean(profiles, axis=0))
    assert np.median(level) == pytest.approx(variance, rel=0.05)
    with pytest.raises(ValueError):
        radial(np.ones((128, 128)))
    with pytest.raises(ValueError):
        RadialPower((16, 16), bins=4096, f2_max=0.25)


def test_ctf_feature_finds_the_fresnel_number_and_blur_steepens_the_slope():
    n, fr = 256, 0.01
    rng = np.random.default_rng(1)
    phase = torch.from_numpy(1e-2 * rng.standard_normal((n, n)))
    sharp = simulate_hologram(object_from_phase(phase, None), fr).numpy()
    blurred = simulate_hologram(object_from_phase(phase, None), fr, psf_sigma=1.0).numpy()
    radial = RadialPower((n, n), bins=512, f2_max=0.25)
    right = spectral_features(radial, sharp, fr)
    wrong = spectral_features(radial, sharp, 1.37 * fr)
    assert right["ctf"] > 0.5 and right["ctf"] > wrong["ctf"] + 0.3
    assert len(right["ctf_bands"]) == 4
    assert spectral_features(radial, blurred, fr)["slope"] < right["slope"] - 5
    stats = feature_distribution([right, wrong])
    assert set(stats) == set(FEATURE_NAMES)
    assert stats["ctf"]["p05"] <= stats["ctf"]["median"] <= stats["ctf"]["p95"]


def test_percentile_rank():
    assert percentile_rank([1, 2, 3, 4], 2.5) == 50.0
    assert percentile_rank([1, 2, 3, 4], 2) == 37.5
    assert percentile_rank([1, 2, float("nan")], 5) == 100.0


def test_binning_phases():
    image = np.random.default_rng(2).random((16, 16))
    phases = binning_phases(image, 4)
    assert len(phases) == 16 and all(p.shape == (4, 4) for p in phases)
    np.testing.assert_array_equal(phases[0], bin_mean(image, 4))
    np.testing.assert_allclose([p.mean() for p in phases], image.mean(), rtol=1e-12)
    assert not np.array_equal(phases[1], phases[0])


def test_measured_estimate():
    config = LearnedConfig.from_yaml(CI_LEARNED)
    torch.manual_seed(0)
    autofocus = LearnedAutofocus(config, build_model(config), (128, 128), sigma_scale=2.0)
    sample = MeasuredSample(
        "test", "test", 11.0, 20.0, 0.1, 1.0, 4.0, (0.095, 0.105), 0.1002, 0.1001, 10
    )
    hologram = 1 + 0.01 * torch.rand((128, 128), generator=torch.Generator().manual_seed(3))
    phases = binning_phases(1 + 0.01 * np.random.default_rng(4).random((256, 256)), 2)
    out = measured_estimate(
        autofocus, hologram, sample, {"a": 0.1, "b": 0.2}, repeats=3, phases=phases
    )
    dz, sigma, sigma_raw = (float(v[0]) for v in autofocus.estimate(hologram))
    assert out["z01_est"] == pytest.approx(0.1 + dz) and out["sigma"] == pytest.approx(sigma)
    assert out["sigma"] == pytest.approx(2.0 * sigma_raw)
    width68 = out["intervals"]["68"]["width_um"]
    assert width68 == pytest.approx(2e6 * 0.9944578832 * sigma, rel=1e-6)
    assert out["intervals"]["95"]["width_um"] > width68
    a = out["references"]["a"]
    assert a["distance_um"] == pytest.approx((out["z01_est"] - 0.1) * 1e6)
    assert a["z_score"] == pytest.approx((out["z01_est"] - 0.1) / sigma)
    assert not out["references"]["b"]["inside"]["95"]
    assert out["binning_phases"]["n"] == 4 and out["seconds_per_estimate"]["repeats"] == 3
    with pytest.raises(ValueError):
        measured_estimate(autofocus, torch.ones((64, 64)), sample, {})
