"""``src.data.representations``: shapes, standardisation, radial profile on synthetic rings, dataset/model integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import HologramHDF5Dataset, dataset_kwargs_from_config
from src.data.representations import (
    REPRESENTATIONS,
    Representation,
    build_representation,
    frequency_grid,
    log_power_spectrum,
    power_spectrum,
    radial_bin_centres,
    radial_log_power_profile,
    radial_profile,
    relative_contrast,
)
from src.models.cnn import RadialProfileMLP, build_model, input_channels_from_config


def _hologram_batch(batch: int = 3, size: int = 64, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return 1.0 + 0.1 * torch.randn(batch, 1, size, size, generator=gen)


def test_power_spectrum_shapes_and_noise_level() -> None:
    x = _hologram_batch()
    spectrum = power_spectrum(x, window=False)
    assert spectrum.shape == x.shape and torch.isfinite(spectrum).all()
    # white noise of variance sigma^2 -> mean power sigma^2 (Parseval with the 1/(H*W) normalisation)
    sigma2 = relative_contrast(x).var(dim=(-2, -1), unbiased=False)
    torch.testing.assert_close(spectrum.mean(dim=(-2, -1)), sigma2, rtol=1e-4, atol=1e-6)
    # DC is removed by the relative contrast
    assert spectrum[..., 32, 32].abs().max() < 1e-6
    assert power_spectrum(x[0, 0]).shape == (64, 64)


def test_log_power_spectrum_is_standardised_and_handles_scales() -> None:
    x = _hologram_batch()
    for log_scale in ("auto", 1e3):
        out = log_power_spectrum(x, log_scale=log_scale)
        assert out.shape == x.shape
        torch.testing.assert_close(out.mean(dim=(-2, -1)), torch.zeros(3, 1), atol=1e-5, rtol=0)
        torch.testing.assert_close(out.std(dim=(-2, -1), unbiased=False), torch.ones(3, 1), atol=1e-4, rtol=0)
    pooled = log_power_spectrum(x, pool=2)
    assert pooled.shape == (3, 1, 32, 32)
    with pytest.raises(ValueError):
        log_power_spectrum(x, log_scale=-1.0)
    with pytest.raises(ValueError):
        log_power_spectrum(x, log_scale="bogus")


def test_spectrum_invariant_to_flips_and_rotations_but_not_to_zoom() -> None:
    x = _hologram_batch(1)[0]
    base = radial_log_power_profile(x, n_bins=64)
    for transformed in (torch.flip(x, dims=(-1,)), torch.flip(x, dims=(-2,)), torch.rot90(x, 1, dims=(-2, -1))):
        torch.testing.assert_close(radial_log_power_profile(transformed, n_bins=64), base, atol=1e-4, rtol=0)
    # the 2-D spectrum of a flipped image is the flipped spectrum
    spec = log_power_spectrum(x, window=False)
    flipped = log_power_spectrum(torch.flip(x, dims=(-1,)), window=False)
    torch.testing.assert_close(torch.roll(torch.flip(spec, dims=(-1,)), 1, dims=-1), flipped, atol=1e-4, rtol=0)


@pytest.mark.parametrize("fr", [0.01, 0.025])
def test_radial_profile_recovers_ring_period_in_freq_sq(fr: float) -> None:
    """A synthetic spectrum sin^2(pi u / Fr) has minima at u = n*Fr; uniform bins in u make them equidistant."""
    size, n_bins = 128, 256
    xi, eta = frequency_grid(size, size)
    u = xi**2 + eta**2
    spectrum = torch.sin(np.pi * u / fr) ** 2
    profile = radial_profile(spectrum, n_bins=n_bins, axis="freq_sq")
    centres = radial_bin_centres(n_bins)
    assert profile.shape == (n_bins,)
    expected_minima = [n * fr for n in range(1, int(0.25 / fr))]
    found = []
    for u_n in expected_minima:
        window = (centres > u_n - 0.3 * fr) & (centres < u_n + 0.3 * fr)
        u_min = float(centres[window][profile[window].argmin()])
        assert abs(u_min - u_n) < 1.5 * (0.25 / n_bins), f"minimum near {u_n} found at {u_min}"
        found.append(u_min)
    # the period in u (mean spacing of the minima) is the Fresnel number
    assert np.mean(np.diff(found)) == pytest.approx(fr, rel=0.02)


def test_radial_profile_batch_axes_and_empty_bins() -> None:
    spectrum = torch.rand(2, 1, 32, 32)
    prof = radial_profile(spectrum, n_bins=512)  # far more bins than radii on a 32 px grid -> empty bins get filled
    assert prof.shape == (2, 1, 512) and torch.isfinite(prof).all()
    prof_freq = radial_profile(spectrum, n_bins=16, axis="freq")
    assert prof_freq.shape == (2, 1, 16)
    # constant spectrum -> constant profile regardless of binning
    torch.testing.assert_close(radial_profile(torch.ones(8, 8), n_bins=10), torch.ones(10), atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        radial_profile(spectrum, n_bins=8, axis="polar")
    with pytest.raises(ValueError):
        radial_profile(torch.ones(8), n_bins=4)


def test_radial_log_power_profile_standardised() -> None:
    x = _hologram_batch(4)
    prof = radial_log_power_profile(x, n_bins=128)
    assert prof.shape == (4, 1, 128)
    torch.testing.assert_close(prof.mean(dim=-1), torch.zeros(4, 1), atol=1e-5, rtol=0)
    torch.testing.assert_close(prof.std(dim=-1, unbiased=False), torch.ones(4, 1), atol=1e-4, rtol=0)


def test_representation_factory_shapes() -> None:
    raw = _hologram_batch(1)[0]
    normalized = (raw - raw.mean()) / raw.std()
    expectations = {
        "hologram": (1, 64, 64),
        "log_power_spectrum": (1, 64, 64),
        "hologram+spectrum": (2, 64, 64),
        "radial_profile": (32,),
    }
    assert set(expectations) == set(REPRESENTATIONS)
    for name, shape in expectations.items():
        transform = build_representation(name, n_bins=32)
        out = transform(raw, normalized)
        assert tuple(out.shape) == shape == transform.output_shape((1, 64, 64))
        assert transform.channels == Representation(name).channels
        assert transform.to_dict()["representation"] == name
    assert torch.equal(build_representation("hologram")(raw, normalized), normalized)
    two = build_representation("hologram+spectrum")(raw, normalized)
    torch.testing.assert_close(two[0], normalized[0])
    torch.testing.assert_close(two[1], log_power_spectrum(raw)[0])
    assert build_representation("log_power_spectrum", pool=2).output_shape((1, 64, 64)) == (1, 32, 32)
    with pytest.raises(ValueError):
        build_representation("zoom")
    with pytest.raises(ValueError):
        build_representation("hologram", bogus=1)
    with pytest.raises(ValueError):
        build_representation("hologram+spectrum", pool=2)
    with pytest.raises(ValueError):
        build_representation("radial_profile", axis="polar")


def test_dataset_with_representations(tiny_dataset_dir: Path) -> None:
    path = tiny_dataset_dir / "train.hdf5"
    default = HologramHDF5Dataset(path)
    assert default.representation == "hologram" and default.output_shape == (1, 64, 64)
    spectrum = HologramHDF5Dataset(path, representation="log_power_spectrum")
    image, target = spectrum[0]
    assert image.shape == (1, 64, 64) and spectrum.output_shape == (1, 64, 64)
    assert image.mean().item() == pytest.approx(0.0, abs=1e-4) and image.std().item() == pytest.approx(1.0, abs=1e-3)
    assert target.item() == pytest.approx(default[0][1].item())
    both = HologramHDF5Dataset(path, representation="hologram+spectrum", crop_size=32)
    assert both[0][0].shape == (2, 32, 32) and both.output_shape == (2, 32, 32)
    torch.testing.assert_close(both[0][0][0], HologramHDF5Dataset(path, crop_size=32)[0][0][0])
    radial = HologramHDF5Dataset(path, representation="radial_profile", representation_kwargs={"n_bins": 48})
    assert radial[0][0].shape == (48,) and radial.output_shape == (48,)
    assert radial.summary()["representation"]["n_bins"] == 48
    kwargs = dataset_kwargs_from_config({"representation": "radial_profile", "representation_kwargs": {"n_bins": 16}})
    assert kwargs["representation"] == "radial_profile" and kwargs["representation_kwargs"] == {"n_bins": 16}
    with pytest.raises(ValueError):
        HologramHDF5Dataset(path, representation="wavelet")


def test_model_input_channels_follow_representation() -> None:
    base_model = {"arch": "cnn", "conv_channels": [4, 8], "fc_units": 8, "dropout_rate": 0.0, "pool_size": 2}
    assert input_channels_from_config({"data": {"representation": "hologram+spectrum"}, "model": base_model}) == 2
    assert input_channels_from_config({"data": {"representation": "log_power_spectrum"}, "model": base_model}) == 1
    assert input_channels_from_config({"data": {}, "model": base_model | {"in_channels": 3}}) == 3
    model = build_model({"data": {"representation": "hologram+spectrum"}, "model": base_model})
    with torch.no_grad():
        assert model(torch.randn(2, 2, 64, 64)).shape == (2,)
    mlp = build_model(
        {"data": {"representation": "radial_profile", "representation_kwargs": {"n_bins": 40}}, "model": {"arch": "mlp_radial", "hidden_units": [16]}}
    )
    assert isinstance(mlp, RadialProfileMLP)
    with torch.no_grad():
        assert mlp(torch.randn(5, 40)).shape == (5,)
        assert mlp(torch.randn(5, 1, 40)).shape == (5,)
    with pytest.raises(ValueError):
        mlp(torch.randn(5, 41))
    with pytest.raises(ValueError):
        RadialProfileMLP(n_bins=0)
    with pytest.raises(ValueError):
        build_model({"model": {"arch": "transformer"}})
