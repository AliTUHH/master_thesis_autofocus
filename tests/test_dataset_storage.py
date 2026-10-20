import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from holofocus.dataset import (
    LABEL_KEYS,
    TARGETS,
    DatasetConfig,
    HologramDataset,
    HologramGenerator,
    read_config,
    read_labels,
    target_values,
    write_split,
)

CI_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dataset" / "ci.yaml"
N = 6


@pytest.fixture(scope="module")
def config():
    return DatasetConfig.from_yaml(CI_CONFIG)


@pytest.fixture(scope="module")
def split_file(tmp_path_factory, config):
    path = tmp_path_factory.mktemp("data") / "val.h5"
    write_split(config, "val", path, n_samples=N, device="cpu", log=None)
    return path


def test_file_layout_and_attributes(split_file, config):
    assert not split_file.with_suffix(".h5.part").exists()
    with h5py.File(split_file, "r") as fh:
        h, w = config.geometry.fov_shape
        for name in ("hologram", "hologram_clean", "phase"):
            assert fh[name].shape == (N, h, w)
            assert fh[name].dtype == np.float32
        assert fh.attrs["n_samples"] == N
        assert fh.attrs["split"] == "val"
        assert fh.attrs["seed"] == config.seed
        assert tuple(fh.attrs["simulation_shape"]) == config.simulation_shape
    assert read_config(split_file) == config


def test_stored_samples_equal_generator_output(split_file, config):
    generator = HologramGenerator(config, device="cpu")
    labels = read_labels(split_file)
    with h5py.File(split_file, "r") as fh:
        for i in (0, N - 1):
            sample = generator.sample("val", i)
            np.testing.assert_array_equal(fh["hologram"][i], sample.hologram)
            np.testing.assert_array_equal(fh["hologram_clean"][i], sample.clean)
            np.testing.assert_array_equal(fh["phase"][i], sample.phase)
            for key in LABEL_KEYS:
                assert labels[key][i] == sample.labels[key]
            assert labels["phantom_type"][i] == sample.meta["phantom_type"]


def test_target_values(split_file):
    labels = read_labels(split_file)
    np.testing.assert_allclose(target_values(labels, "dz01_mm"), labels["dz01"] * 1e3)
    ratio = target_values(labels, "fresnel_ratio")
    # Delta Fr / Fr ~ dz01 / z01 for z01 << z02.
    np.testing.assert_allclose(ratio, labels["dz01"] / labels["z01_guess"], rtol=0.05)
    with pytest.raises(ValueError, match="unknown target"):
        target_values(labels, "focus")


@pytest.mark.parametrize("target", TARGETS)
def test_dataset_items(split_file, config, target):
    ds = HologramDataset(split_file, target=target)
    assert len(ds) == N
    hologram, value = ds[1]
    assert hologram.shape == (1, *config.geometry.fov_shape)
    assert hologram.dtype == torch.float32
    assert value.dtype == torch.float32 and value.ndim == 0
    expected = target_values(read_labels(split_file), target)[1]
    assert value.item() == pytest.approx(expected, rel=1e-6)


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_dataset_with_phase_pickles_and_loads_in_workers(split_file):
    ds = HologramDataset(split_file, with_phase=True)
    hologram, _, phase = ds[0]
    assert phase.shape == hologram.shape
    clone = pickle.loads(pickle.dumps(ds))
    torch.testing.assert_close(clone[0][0], hologram)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2)
    batches = list(loader)
    assert sum(b[0].shape[0] for b in batches) == N
    torch.testing.assert_close(batches[0][0][0], hologram)
