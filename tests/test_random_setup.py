"""``NFHRandomDistSetup``: sampling bounds, reproducibility, Forge registration and per-sample labels."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import torch

import holowizard.forge.experiment.setup as forge_setup_module
from holowizard.forge.experiment.setup import NFHSetup
from holowizard.forge.utils import calc_Fr

from src.data.forge_setup import NFHRandomDistSetup, register_forge_setup
from src.data.generate_data import build_forge_config, generate_dataset, propagator_sampling_check, verify_split_labels
from src.utils.physics import fresnel_number

SETUP_KWARGS = dict(detector_size=2048, detector_px_size=0.0065, padding_factor=2, downsample_factor=32, energy=11.0)


def _draws(setup: NFHRandomDistSetup, n: int) -> np.ndarray:
    return np.array([setup.get_distances()[0] for _ in range(n)])


def test_uniform_draws_within_bounds_and_rounded() -> None:
    setup = NFHRandomDistSetup(**SETUP_KWARGS, z01_min=50.0, z01_max=300.0, z02_min=20000.0, seed=1)
    z01 = _draws(setup, 500)
    assert np.all(z01 >= 50.0) and np.all(z01 <= 300.0)
    assert np.allclose(z01, np.round(z01, 3))
    assert z01.std() > 20.0
    assert setup.z02 == 20000.0 and setup.z02_bounds == (20000.0, 20000.0)


def test_loguniform_draws_within_bounds_and_skewed() -> None:
    setup = NFHRandomDistSetup(**SETUP_KWARGS, z01_min=10.0, z01_max=1000.0, z02_min=20000.0, z01_distribution="loguniform", seed=2)
    z01 = _draws(setup, 2000)
    assert np.all(z01 >= 10.0) and np.all(z01 <= 1000.0)
    # log-uniform: the geometric mean is close to sqrt(10 * 1000) = 100
    assert np.exp(np.log(z01).mean()) == pytest.approx(100.0, rel=0.15)


def test_z02_range_is_sampled() -> None:
    setup = NFHRandomDistSetup(**SETUP_KWARGS, z01_min=50.0, z01_max=300.0, z02_min=19000.0, z02_max=21000.0, seed=3)
    z02 = np.array([setup.get_distances()[1] for _ in range(200)])
    assert np.all(z02 >= 19000.0) and np.all(z02 <= 21000.0)
    assert z02.std() > 100.0


def test_seed_reproducibility() -> None:
    a = NFHRandomDistSetup(**SETUP_KWARGS, z01_min=50.0, z01_max=300.0, z02_min=20000.0, seed=123)
    b = NFHRandomDistSetup(**SETUP_KWARGS, z01_min=50.0, z01_max=300.0, z02_min=20000.0, seed=123)
    c = NFHRandomDistSetup(**SETUP_KWARGS, z01_min=50.0, z01_max=300.0, z02_min=20000.0, seed=124)
    assert np.array_equal(_draws(a, 20), _draws(b, 20))
    assert not np.array_equal(_draws(a, 20), _draws(c, 20))


def test_state_fr_and_kernel_consistent() -> None:
    setup = NFHRandomDistSetup(**SETUP_KWARGS, z01_min=50.0, z01_max=300.0, z02_min=20000.0, seed=5)
    kernel = setup.kernel
    assert kernel.shape == (setup.probe_size, setup.probe_size) and kernel.dtype == torch.complex64
    expected_fr = calc_Fr(setup.energy, setup.z01, setup.z02, setup.detector_px_size)
    assert setup.Fr == pytest.approx(expected_fr, rel=1e-12)
    assert setup.detector_px_size == pytest.approx(0.0065 * 32)
    assert setup.as_dict()["Fr"] == setup.Fr and setup.as_dict()["z01"] == setup.z01
    # the kernel of the current draw equals the reference implementation of NFHSetup
    reference = NFHSetup.create_kernel(setup, setup.Fr)
    assert torch.allclose(kernel, reference)


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        NFHRandomDistSetup(**SETUP_KWARGS, z01_min=300.0, z01_max=50.0, z02_min=20000.0)
    with pytest.raises(ValueError):
        NFHRandomDistSetup(**SETUP_KWARGS, z01_min=50.0, z01_max=300.0, z02_min=20000.0, z01_distribution="normal")
    with pytest.raises(ValueError):
        NFHRandomDistSetup(**SETUP_KWARGS, z01_min=50.0, z01_max=30000.0, z02_min=20000.0)


def test_registration_makes_type_resolvable() -> None:
    register_forge_setup()
    assert getattr(forge_setup_module, "NFHRandomDistSetup") is NFHRandomDistSetup


def test_build_forge_config_structure(tiny_config: dict[str, Any]) -> None:
    forge_cfg = build_forge_config(tiny_config, "train", seed=7)
    assert forge_cfg["name"] == "train"
    assert forge_cfg["setup"]["type"] == "NFHRandomDistSetup"
    args = forge_cfg["setup"]["args"]
    assert args["seed"] == 7 and args["z01_min"] == 50.0 and args["z01_max"] == 300.0 and args["z02_max"] is None
    assert forge_cfg["labeller"]["args"]["store_setup"] is True
    assert forge_cfg["hologram_noise"] == {"type": "Gaussian", "args": {"mean": 0.0, "std": 1.0, "intensity": 0.05}}
    assert forge_cfg["probe_noise"] is None


def test_propagator_sampling_check(tmp_path: Path, tiny_config: dict[str, Any]) -> None:
    ok = propagator_sampling_check(tiny_config)
    assert ok["ok"] and ok["grid_size"] == 128 and ok["fr_limit"] == pytest.approx(1 / 128)
    aliased = copy.deepcopy(tiny_config)
    aliased["setup"]["downsample_factor"] = 1
    aliased["setup"]["detector_size"] = 64
    aliased["phantom"]["radius_range_px"] = [2, 16]
    aliased["phantom"]["size_range_px"] = [2, 32]
    assert not propagator_sampling_check(aliased)["ok"]
    aliased["num_samples"] = {"train": 1, "val": 0, "test": 0}
    with pytest.warns(UserWarning, match="sampling limit"):
        generate_dataset(aliased, tmp_path / "aliased", overwrite=True)


def test_build_forge_config_missing_key_raises(tiny_config: dict[str, Any]) -> None:
    broken = copy.deepcopy(tiny_config)
    del broken["setup"]["z02_mm"]
    with pytest.raises(KeyError, match="z02_mm"):
        build_forge_config(broken, "train", seed=1)


def test_labels_stored_per_sample(tiny_dataset_dir: Path, tiny_config: dict[str, Any]) -> None:
    """Each stored Fr must be the Fresnel number of the z01/z02 used for that very hologram."""
    with h5py.File(tiny_dataset_dir / "train.hdf5", "r") as handle:
        z01 = handle["metadata/setup/z01"][...].astype(np.float64)
        z02 = handle["metadata/setup/z02"][...].astype(np.float64)
        fr = handle["metadata/setup/Fr"][...].astype(np.float64)
        px = handle["metadata/setup/detector_px_size"][...].astype(np.float64)
        energy = handle["metadata/setup/energy"][...].astype(np.float64)
        holograms = handle["images/hologram"][...]
    n = tiny_config["num_samples"]["train"]
    assert holograms.shape == (n, 64, 64) and holograms.dtype == np.float32
    assert np.unique(z01).size == n, "z01 must vary per sample"
    assert np.all(z01 >= 50.0) and np.all(z01 <= 300.0)
    assert np.all(z02 == 20000.0)
    assert np.allclose(px, 0.0065 * 32)
    recomputed = fresnel_number(z01, z02, energy, px)
    np.testing.assert_allclose(fr, recomputed, rtol=1e-5)  # float32 storage limits agreement
    report = verify_split_labels(tiny_dataset_dir / "train.hdf5", (50.0, 300.0))
    assert report["z01_within_bounds"] and report["fr_rel_deviation_max"] < 1e-5


def test_hologram_encodes_fresnel_number(tiny_dataset_dir: Path) -> None:
    """Re-simulating a stored phantom-free hologram is impossible, but the propagation kernel used for a
    sample must differ between samples: holograms with different Fr may not be identical."""
    with h5py.File(tiny_dataset_dir / "train.hdf5", "r") as handle:
        holograms = handle["images/hologram"][...]
    assert not np.allclose(holograms[0], holograms[1])


def test_splits_are_disjoint_and_reproducible(tmp_path: Path, tiny_config: dict[str, Any]) -> None:
    cfg = copy.deepcopy(tiny_config)
    cfg["num_samples"] = {"train": 3, "val": 2, "test": 0}
    meta_a = generate_dataset(copy.deepcopy(cfg), tmp_path / "a", overwrite=True)
    meta_b = generate_dataset(copy.deepcopy(cfg), tmp_path / "b", overwrite=True)
    assert set(meta_a["splits"]) == {"train", "val"}
    with h5py.File(tmp_path / "a" / "train.hdf5", "r") as a, h5py.File(tmp_path / "b" / "train.hdf5", "r") as b:
        np.testing.assert_array_equal(a["images/hologram"][...], b["images/hologram"][...])
        np.testing.assert_array_equal(a["metadata/setup/z01"][...], b["metadata/setup/z01"][...])
    with h5py.File(tmp_path / "a" / "train.hdf5", "r") as train, h5py.File(tmp_path / "a" / "val.hdf5", "r") as val:
        assert not np.array_equal(train["metadata/setup/z01"][:2], val["metadata/setup/z01"][:2])
    assert (tmp_path / "a" / "meta.json").is_file()
    assert meta_b["versions"]["holowizard"] is not None


def test_generate_refuses_identical_seeds(tmp_path: Path, tiny_config: dict[str, Any]) -> None:
    cfg = copy.deepcopy(tiny_config)
    cfg["seeds"] = {"train": 1, "val": 1, "test": 2}
    with pytest.raises(ValueError, match="disjoint"):
        generate_dataset(cfg, tmp_path / "dup", overwrite=True)
