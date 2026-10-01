"""``HologramHDF5Dataset``: shapes, dtypes, target modes, normalisations, cropping and loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.data.dataset import NORMALIZATIONS, HologramHDF5Dataset, make_dataloaders, normalize_hologram
from src.utils.physics import fresnel_number


def test_sample_shape_and_dtype(tiny_dataset_dir: Path) -> None:
    ds = HologramHDF5Dataset(tiny_dataset_dir / "train.hdf5")
    image, target = ds[0]
    assert image.shape == (1, 64, 64) and image.dtype == torch.float32
    assert target.shape == () and target.dtype == torch.float32
    assert len(ds) == 6 and ds.output_shape == (1, 64, 64)
    assert ds[-1][1].item() == pytest.approx(ds.targets[-1])
    with pytest.raises(IndexError):
        ds[len(ds)]


@pytest.mark.parametrize("target_mode", ["log_fr", "fr", "z01_mm"])
def test_target_modes(tiny_dataset_dir: Path, target_mode: str) -> None:
    ds = HologramHDF5Dataset(tiny_dataset_dir / "train.hdf5", target_mode=target_mode)
    expected = {"log_fr": np.log(ds.fr), "fr": ds.fr, "z01_mm": ds.z01_mm}[target_mode]
    np.testing.assert_allclose(ds.targets, expected.astype(np.float32), rtol=1e-6)
    consts = ds.setup_constants
    np.testing.assert_allclose(fresnel_number(ds.z01_mm, ds.z02_mm, consts["energy_kev"], consts["px_mm"]), ds.fr, rtol=1e-5)


def test_setup_constants_and_geometry(tiny_dataset_dir: Path) -> None:
    ds = HologramHDF5Dataset(tiny_dataset_dir / "val.hdf5")
    consts = ds.setup_constants
    assert consts["energy_kev"] == 11.0
    assert consts["px_mm"] == pytest.approx(0.208)
    assert consts["z02_mm"] == 20000.0
    assert consts["detector_size"] == 64 and consts["padding_factor"] == 2 and consts["downsample_factor"] == 32
    geometry = ds.geometry
    assert set(geometry) == {"z01_mm", "z02_mm", "fr"} and geometry["z01_mm"].shape == (len(ds),)
    assert ds.summary()["num_samples"] == len(ds)


@pytest.mark.parametrize("mode", NORMALIZATIONS)
def test_normalizations(tiny_dataset_dir: Path, mode: str) -> None:
    ds = HologramHDF5Dataset(tiny_dataset_dir / "train.hdf5", normalization=mode)
    image, _ = ds[0]
    raw = HologramHDF5Dataset(tiny_dataset_dir / "train.hdf5", normalization="none")[0][0]
    assert torch.isfinite(image).all()
    if mode == "standardize":
        assert image.mean().item() == pytest.approx(0.0, abs=1e-5) and image.std().item() == pytest.approx(1.0, abs=1e-4)
    elif mode == "divide_mean":
        assert image.mean().item() == pytest.approx(1.0, abs=1e-5)
    elif mode == "log":
        torch.testing.assert_close(image, torch.log(torch.clamp(raw, min=1e-6)))
    else:
        torch.testing.assert_close(image, raw)


def test_normalize_constant_image_is_safe() -> None:
    constant = torch.full((1, 8, 8), 3.0)
    assert torch.isfinite(normalize_hologram(constant, "standardize")).all()
    assert torch.isfinite(normalize_hologram(torch.zeros(1, 8, 8), "divide_mean")).all()
    with pytest.raises(ValueError):
        normalize_hologram(constant, "bogus")


def test_crop_and_downsample(tiny_dataset_dir: Path) -> None:
    ds = HologramHDF5Dataset(tiny_dataset_dir / "train.hdf5", crop_size=32, downsample=2, normalization="none")
    image, _ = ds[0]
    assert image.shape == (1, 16, 16) and ds.output_shape == (1, 16, 16)
    full = HologramHDF5Dataset(tiny_dataset_dir / "train.hdf5", normalization="none")[0][0]
    pooled = torch.nn.functional.avg_pool2d(full[:, 16:48, 16:48].unsqueeze(0), 2).squeeze(0)
    torch.testing.assert_close(image, pooled)


def test_invalid_arguments(tiny_dataset_dir: Path, tmp_path: Path) -> None:
    path = tiny_dataset_dir / "train.hdf5"
    with pytest.raises(FileNotFoundError):
        HologramHDF5Dataset(tmp_path / "missing.hdf5")
    with pytest.raises(ValueError):
        HologramHDF5Dataset(path, target_mode="defocus_cm")
    with pytest.raises(ValueError):
        HologramHDF5Dataset(path, normalization="minmax")
    with pytest.raises(ValueError):
        HologramHDF5Dataset(path, crop_size=1000)
    with pytest.raises(KeyError):
        HologramHDF5Dataset(path, hologram_key="images/gt_hologram")


def test_make_dataloaders(tiny_dataset_dir: Path, tiny_config: dict[str, Any]) -> None:
    config = {"data": {"dir": str(tiny_dataset_dir), "batch_size": 4, "num_workers": 0, "target_mode": "log_fr"}}
    loaders, datasets = make_dataloaders(config, seed=0)
    assert set(loaders) == {"train", "val", "test"}
    images, targets = next(iter(loaders["train"]))
    assert images.shape == (4, 1, 64, 64) and targets.shape == (4,)
    assert len(datasets["test"]) == tiny_config["num_samples"]["test"]
    with pytest.raises(FileNotFoundError):
        make_dataloaders({"data": {"dir": str(tiny_dataset_dir / "nope")}})


def test_dataset_pickles_without_open_handle(tiny_dataset_dir: Path) -> None:
    import pickle

    ds = HologramHDF5Dataset(tiny_dataset_dir / "train.hdf5")
    _ = ds[0]
    clone = pickle.loads(pickle.dumps(ds))
    assert clone._h5 is None
    torch.testing.assert_close(clone[1][0], ds[1][0])
