"""``src.baseline.ctf_ringfit``: Fresnel number of synthetic weak phase objects is recovered within 2 %."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from src.baseline.ctf_ringfit import (
    MIN_BINS_PER_PERIOD,
    RingFitConfig,
    default_n_bins,
    evaluate_file,
    fresnel_number_resolution_limit,
    main,
    max_resolvable_rings,
    radial_power_profile,
    resolvable_frequency_limit,
    ring_fit,
)
from src.utils.fresnel import propagate


def simulate_weak_phase_hologram(size: int, fr: float, phi_max: float = 0.2, seed: int = 0, pad: int = 2, noise: float = 0.0) -> np.ndarray:
    """Amplitude ``|D_Fr(exp(i phi))|`` of a few smooth shapes with the Fresnel kernel of the context formula
    ``exp(-i pi / Fr (xi^2 + eta^2))`` (frequencies in cycles/pixel, :mod:`src.utils.fresnel`), simulated on a
    padded grid and cropped."""
    rng = np.random.default_rng(seed)
    grid = size * pad
    yy, xx = np.mgrid[:grid, :grid]
    phase = np.zeros((grid, grid))
    for _ in range(3):
        cx, cy = rng.uniform(grid / 2 - size / 3, grid / 2 + size / 3, 2)
        if rng.random() < 0.5:
            phase += (np.hypot(xx - cx, yy - cy) < rng.uniform(size / 16, size / 6)) * rng.uniform(0.5, 1.0)
        else:
            w, h = rng.uniform(size / 10, size / 4, 2)
            phase += ((np.abs(xx - cx) < w / 2) & (np.abs(yy - cy) < h / 2)) * rng.uniform(0.5, 1.0)
    phase = gaussian_filter(phase * phi_max / max(phase.max(), 1e-9), 1.0)
    # the phase already lives on the padded grid, so propagate without further padding and crop the centre
    psi = propagate(np.exp(1j * phase), fr, pad_factor=1.0)
    lo, hi = grid // 2 - size // 2, grid // 2 + size // 2
    amplitude = np.abs(psi.numpy())[lo:hi, lo:hi].astype(np.float64)
    if noise > 0:
        amplitude = amplitude + noise * rng.standard_normal(amplitude.shape)
    return amplitude


@pytest.mark.parametrize("fr, seed", [(5.0e-3, 1), (8.0e-3, 2), (1.5e-2, 3)])
def test_weak_phase_object_fresnel_number_within_2_percent(fr: float, seed: int) -> None:
    hologram = simulate_weak_phase_hologram(256, fr, phi_max=0.2, seed=seed)
    result = ring_fit(hologram, RingFitConfig(fr_min=2e-3, fr_max=2.5e-2))
    assert result.fr == pytest.approx(fr, rel=0.02)
    # the minima regression agrees and finds many rings; the oscillation phase is that of a phase object (pi)
    assert result.fr_minima == pytest.approx(fr, rel=0.02)
    assert result.n_minima >= 5
    assert abs(abs(result.phase_rad) - np.pi) < 0.5
    assert result.score > 0.3 and result.elapsed_s < 5.0


def test_comb_template_and_noisy_input() -> None:
    fr = 1.0e-2
    hologram = simulate_weak_phase_hologram(256, fr, phi_max=0.3, seed=4, noise=0.002)
    for template in ("cos", "comb"):
        result = ring_fit(hologram, RingFitConfig(fr_min=2e-3, fr_max=2.5e-2, template=template))
        assert result.fr == pytest.approx(fr, rel=0.02), template


def test_downsampled_hologram_has_s_squared_fresnel_number() -> None:
    """Average pooling by s enlarges the pixel: rings move to u' = s^2 u, so the estimate must be divided by s^2."""
    fr = 4.0e-3
    hologram = simulate_weak_phase_hologram(256, fr, phi_max=0.2, seed=5)
    pooled = hologram.reshape(128, 2, 128, 2).mean(axis=(1, 3))
    result = ring_fit(pooled, RingFitConfig(fr_min=4e-3, fr_max=6e-2))
    assert result.fr / 4.0 == pytest.approx(fr, rel=0.03)


def test_radial_power_profile_properties() -> None:
    rng = np.random.default_rng(0)
    noise = 1.0 + 0.05 * rng.standard_normal((128, 128))
    u, profile, counts = radial_power_profile(noise, n_bins=256, window="none")
    assert u.shape == profile.shape == counts.shape == (256,)
    assert np.all(np.diff(u) > 0) and u[-1] < 0.25
    # white noise: flat profile at the noise variance (relative contrast ~ sigma^2)
    assert np.median(profile[u > 0.05]) == pytest.approx(0.05**2, rel=0.15)
    # equal-area bins in u: the mean pixel count per bin does not grow with the radius (lattice jitter aside)
    assert counts[20:80].mean() == pytest.approx(counts[180:240].mean(), rel=0.1)
    assert counts[20:].mean() == pytest.approx(np.pi * (0.25 / 256) * 128**2, rel=0.1)
    with pytest.raises(ValueError):
        radial_power_profile(np.ones((4, 4, 4)))


def test_resolution_criteria() -> None:
    # rings at |xi| are resolved while Fr > 2|xi|/N  <=>  |xi| < N Fr / 2
    assert resolvable_frequency_limit(3.0e-3, 256) == pytest.approx(0.384)
    assert resolvable_frequency_limit(1.0e-4, 2048) == pytest.approx(0.1024)
    assert resolvable_frequency_limit(1.0e-4, 2048, window="hann") == pytest.approx(0.0512)
    assert fresnel_number_resolution_limit(256) == pytest.approx(2 * 0.5 / 256)
    # P05 regime: Fr = 1e-4 on 2048 px -> rings only up to u = 0.1024^2 ~ 0.0105 -> ~104 rings
    assert max_resolvable_rings(1.0e-4, 2048) == 104
    assert max_resolvable_rings(5.0e-4, 2048) == 500  # limited by u_max = 0.25
    assert max_resolvable_rings(1.8e-2, 256) == 13


def test_default_binning_samples_smallest_period() -> None:
    assert default_n_bins(256) == 1024 and default_n_bins(32) == 256
    assert default_n_bins(256, fr_min=3.0e-3) == 1024  # 8 * 0.25 / 3e-3 = 667 < 4 N
    n_bins = default_n_bins(2048, fr_min=1.0e-4)  # P05 regime: 4 N = 8192 bins would give 3 bins per period
    assert n_bins == 20000 and 1.0e-4 / (0.25 / n_bins) >= MIN_BINS_PER_PERIOD
    assert default_n_bins(2048, fr_min=1.0e-6) == 65536  # upper clip


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        RingFitConfig(fr_min=1e-2, fr_max=1e-3)
    with pytest.raises(ValueError):
        RingFitConfig(fr_min=1e-3, fr_max=1e-2, window="kaiser")
    with pytest.raises(ValueError):
        RingFitConfig(fr_min=1e-3, fr_max=1e-2, template="sinc")
    cfg = RingFitConfig(fr_min=1e-3, fr_max=1e-2, grid_step=0.1)
    assert cfg.matched_detrend is True and len(cfg.fr_grid) >= 2
    assert RingFitConfig(fr_min=1e-3, fr_max=1e-2, template="comb").matched_detrend is False


def test_cli_on_tiny_dataset(tiny_dataset_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "ringfit"
    summary = main(["--data", str(tiny_dataset_dir / "test.hdf5"), "--out", str(out), "--n", "2", "--examples", "1", "--grid-step", "0.05"])
    assert (out / "summary.json").is_file() and (out / "samples.csv").is_file() and (out / "scatter.png").is_file()
    assert summary["num_samples"] == 2 and summary["hologram_shape"] == [64, 64]
    assert np.isfinite(summary["fr_rel_err_pct"]["median"]) and summary["ms_per_hologram"]["median"] > 0
    assert (out / summary["figures"]["example_000"]).is_file()  # figure names are stored relative to the output directory
    assert summary["data"].endswith("test.hdf5")
    direct = evaluate_file(tiny_dataset_dir / "test.hdf5", tmp_path / "ringfit2", num_samples=1, fr_range=(1e-2, 2e-1), n_examples=0, grid_step=0.05)
    assert direct["search_fr_range"] == [1e-2, 2e-1] and direct["num_samples"] == 1
