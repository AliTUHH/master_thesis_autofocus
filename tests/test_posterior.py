import math

import numpy as np
import pytest
from scipy import special, stats

from holofocus.learned.posterior import (
    NPE_PREDICTION_FIELDS,
    GaussianMixture,
    GridPosterior,
    calibration_summary,
    coverage_curve,
    gaussian_hpd_level,
    gaussian_pit,
    kolmogorov_sf,
    ks_uniform,
    log_density_um,
    mixture_summaries,
    pit_histogram,
    posterior_metrics,
    posterior_summaries,
    prediction_stats,
    summarize_npe_predictions,
    truth_statistics,
)

GRID = np.linspace(-1.0, 1.0, 2001)


def _gaussian(means, sigmas, grid=GRID):
    means, sigmas = np.atleast_1d(means)[:, None], np.atleast_1d(sigmas)[:, None]
    log_q = -0.5 * ((grid - means) / sigmas) ** 2 - np.log(sigmas * math.sqrt(2 * math.pi))
    return GridPosterior.from_log_density(grid, log_q)


def test_gaussian_moments_quantiles_and_intervals():
    post = _gaussian([0.1, -0.2], [0.05, 0.1])
    np.testing.assert_allclose(post.prior_mass, 1.0, atol=1e-6)
    np.testing.assert_allclose(post.median(), [0.1, -0.2], atol=1e-6)
    np.testing.assert_allclose(post.mean(), [0.1, -0.2], atol=1e-9)
    np.testing.assert_allclose(post.std(), [0.05, 0.1], rtol=1e-4)
    np.testing.assert_allclose(post.mode(), [0.1, -0.2], atol=1e-9)
    z68 = stats.norm.ppf(0.84)
    lower, upper = post.interval(0.68)
    np.testing.assert_allclose(lower, [0.1 - z68 * 0.05, -0.2 - z68 * 0.1], atol=2e-5)
    np.testing.assert_allclose(upper, [0.1 + z68 * 0.05, -0.2 + z68 * 0.1], atol=2e-5)
    np.testing.assert_allclose(post.hpd_width(0.68), 2 * z68 * np.array([0.05, 0.1]), atol=3e-3)
    np.testing.assert_allclose(
        post.quantile(0.975), [0.1 + 1.959964 * 0.05, -0.2 + 1.959964 * 0.1], atol=3e-5
    )
    with pytest.raises(ValueError):
        post.quantile(1.0)
    assert len(post) == 2 and len(post[1]) == 1
    np.testing.assert_allclose(post[1].median(), [-0.2], atol=1e-6)


def test_truncated_posterior_and_its_prior_mass():
    post = _gaussian([0.95], [0.1])
    assert post.prior_mass[0] == pytest.approx(stats.norm.cdf(0.5), abs=1e-4)
    assert post.median()[0] < 0.95 and post.interval(0.95)[1][0] <= 1.0
    assert post.cdf()[0, 0] == 0.0 and post.cdf()[0, -1] == pytest.approx(1.0)


def test_pit_and_hpd_level_match_the_gaussian_closed_forms():
    means, sigmas = np.array([0.0, 0.3, -0.1]), np.array([0.1, 0.05, 0.2])
    post = _gaussian(means, sigmas)
    truth = np.array([0.05, 0.2, -0.5])
    errors = means - truth
    np.testing.assert_allclose(post.pit(truth), stats.norm.cdf(truth, means, sigmas), atol=1e-4)
    np.testing.assert_allclose(gaussian_pit(errors, sigmas), stats.norm.cdf(-errors / sigmas))
    expected = 2 * stats.norm.cdf(np.abs(errors) / sigmas) - 1
    np.testing.assert_allclose(gaussian_hpd_level(errors, sigmas), expected, rtol=1e-12)
    boundary_cells = 2 * post.density_at(truth) * post.step
    assert np.all(np.abs(post.hpd_level(truth) - expected) <= boundary_cells)
    pdf = stats.norm.pdf(truth, means, sigmas)
    np.testing.assert_allclose(post.density_at(truth), pdf, rtol=1e-4)
    outside = np.array([-2.0, 2.0, 0.3])
    np.testing.assert_allclose(post.pit(outside)[:2], [0.0, 1.0])
    np.testing.assert_allclose(post.density_at(outside)[:2], [0.0, 0.0])
    assert post.hpd_level(np.array([0.0, 0.3, -0.1])).max() < 2e-3


def test_mixture_of_an_ensemble():
    left, right = _gaussian([-0.5], [0.05]), _gaussian([0.5], [0.05])
    mix = GridPosterior.mixture([left, right])
    np.testing.assert_allclose(mix.density, 0.5 * (left.density + right.density))
    assert mix.mean()[0] == pytest.approx(0.0, abs=1e-9)
    assert mix.hpd_width(0.68)[0] < 0.5 * (mix.interval(0.68)[1] - mix.interval(0.68)[0])[0]
    assert mix.hpd_level(np.array([0.0]))[0] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        GridPosterior.mixture([left, _gaussian([0.0, 0.1], [0.1, 0.1])])
    with pytest.raises(ValueError):
        GridPosterior.from_log_density(GRID, np.full((1, len(GRID)), -np.inf))
    with pytest.raises(ValueError):
        GridPosterior(GRID, np.ones((1, 5)))


def test_kolmogorov_distribution_and_ks_test():
    for lam in (0.3, 0.6, 1.0, 1.17, 1.19, 1.5, 2.5):
        assert kolmogorov_sf(lam) == pytest.approx(special.kolmogorov(lam), abs=1e-10)
    assert kolmogorov_sf(0.0) == 1.0
    rng = np.random.default_rng(0)
    for values in (rng.uniform(size=200), rng.beta(2, 2, size=200), rng.uniform(size=50)):
        ours, ref = ks_uniform(values), stats.kstest(values, "uniform")
        assert ours["statistic"] == pytest.approx(ref.statistic, rel=1e-12)
        assert ours["p_value"] == pytest.approx(ref.pvalue, abs=0.02)
    assert ks_uniform([])["n"] == 0


def test_calibration_of_calibrated_and_overconfident_posteriors():
    rng = np.random.default_rng(1)
    n = 4000
    errors = rng.normal(size=n)
    calibrated = calibration_summary(gaussian_pit(errors, 1.0), gaussian_hpd_level(errors, 1.0))
    assert calibrated["pit"]["ks"]["p_value"] > 0.01 and calibrated["hpd"]["ks"]["p_value"] > 0.01
    assert calibrated["hpd"]["coverage"]["68"] == pytest.approx(0.68, abs=0.025)
    assert calibrated["hpd"]["curve"]["max_abs_deviation"] < 0.03
    narrow = calibration_summary(gaussian_pit(errors, 0.6), gaussian_hpd_level(errors, 0.6))
    assert narrow["hpd"]["coverage"]["95"] < 0.85 and narrow["hpd"]["curve"]["mean_deviation"] < 0
    assert narrow["pit"]["below_0.025"] + narrow["pit"]["above_0.975"] > 0.1
    curve = coverage_curve([0.1, 0.5, 0.9], levels=[0.2, 0.6, 1.0])
    assert curve["coverage"] == pytest.approx([1 / 3, 2 / 3, 1.0])
    histogram = pit_histogram([0.01, 0.02, 0.99, 1.0], bins=4)
    assert histogram["counts"] == [2, 0, 0, 2] and histogram["expected"] == 1.0
    assert histogram["chi2"] == pytest.approx(4.0)


def _rows(post, truth, scale):
    stats_ = truth_statistics(post, truth, scale)
    rows = []
    for i in range(len(post)):
        z01_guess, dz01_true = 0.1, float(truth[i] * scale)
        est = {k: float(v[i]) for k, v in stats_.items()}
        row = dict.fromkeys(NPE_PREDICTION_FIELDS, 0.0)
        row.update(
            index=i,
            phantom_type=i % 4,
            photons=1e4,
            z01_true=z01_guess + dz01_true,
            z01_guess=z01_guess,
            dz01_true=dz01_true,
            dz01_est=est["dz01"],
            dz01_mean=est["dz01_mean"],
            dz01_mode=est["dz01_mode"],
            sigma=est["sigma"],
            z01_est=z01_guess + est["dz01"],
            error_um=(est["dz01"] - dz01_true) * 1e6,
            pit=est["pit"],
            hpd_level=est["hpd_level"],
            log_density_um=est["log_density_um"],
            prior_mass=est["prior_mass"],
            seconds=0.01,
        )
        for key in ("68", "95"):
            row[f"z01_lower_{key}"] = z01_guess + est[f"lower_{key}"]
            row[f"z01_upper_{key}"] = z01_guess + est[f"upper_{key}"]
            row[f"hpd_width_{key}_um"] = est[f"hpd_width_{key}"] * 1e6
        rows.append(row)
    return rows, stats_


def test_posterior_metrics_and_prediction_rows():
    scale = 5e-3
    rng = np.random.default_rng(2)
    means = rng.uniform(-0.5, 0.5, size=300)
    truth = means + 0.1 * rng.normal(size=300)
    post = _gaussian(means, np.full(300, 0.1))
    rows, stats_ = _rows(post, truth, scale)
    sigma_um = 0.1 * scale * 1e6
    nll = -np.mean(stats.norm.logpdf((truth - means) * scale * 1e6, scale=sigma_um))
    np.testing.assert_allclose(
        stats_["log_density_um"], log_density_um(post, truth, scale), rtol=1e-12
    )
    metrics = posterior_metrics(stats_, truth * scale)
    assert metrics["interval"]["nll"] == pytest.approx(nll, abs=1e-3)
    assert metrics["interval"]["sigma_median"] == pytest.approx(sigma_um, rel=1e-3)
    assert metrics["interval"]["width_um"]["68"]["median"] == pytest.approx(
        2 * stats.norm.ppf(0.84) * sigma_um, rel=1e-3
    )
    for key, level in (("68", 0.68), ("95", 0.95)):
        assert metrics["interval"]["coverage"][key] == pytest.approx(level, abs=0.06)
        assert (
            metrics["interval"]["hpd_coverage"][key]
            == metrics["calibration"]["hpd"]["coverage"][key]
        )
    assert metrics["prior_mass"]["min"] > 0.99
    errors = (stats_["dz01"] - truth * scale) * 1e6
    assert metrics["error_um"]["mae"] == pytest.approx(np.abs(errors).mean())
    again, dz01_true = prediction_stats(rows)
    np.testing.assert_allclose(dz01_true, truth * scale)
    for key in ("dz01", "lower_68", "upper_95", "hpd_width_68", "pit", "log_density_um"):
        np.testing.assert_allclose(again[key], stats_[key], atol=1e-15)
    summary = summarize_npe_predictions(rows)
    assert summary["n_samples"] == 300
    assert summary["interval"]["nll"] == pytest.approx(metrics["interval"]["nll"])
    assert summary["interval"]["coverage"] == metrics["interval"]["coverage"]
    assert set(summary["by_phantom_type"]) == {"triangle", "cell", "ball", "mixed"}
    assert summary["seconds_per_estimate"]["median"] == pytest.approx(0.01)
    assert set(posterior_summaries(post, scale)) == {
        "dz01",
        "dz01_mean",
        "dz01_mode",
        "sigma",
        "prior_mass",
        "lower_68",
        "upper_68",
        "hpd_width_68",
        "lower_95",
        "upper_95",
        "hpd_width_95",
    }


def test_gaussian_mixture_of_one_member_is_the_gaussian():
    means, sigmas = np.array([[0.3], [-1.2]]), np.array([[0.5], [2.0]])
    mix = GaussianMixture(means, sigmas)
    truth = np.array([0.8, 0.4])
    errors, s = means[:, 0] - truth, sigmas[:, 0]
    np.testing.assert_allclose(mix.median(), means[:, 0], atol=1e-12)
    np.testing.assert_allclose(mix.std(), s)
    for level in (0.68, 0.95):
        lower, upper = mix.interval(level)
        half = stats.norm.ppf(0.5 * (1 + level)) * s
        np.testing.assert_allclose(lower, means[:, 0] - half, atol=1e-10)
        np.testing.assert_allclose(upper, means[:, 0] + half, atol=1e-10)
        np.testing.assert_allclose(mix.hpd_width(level), 2 * half, rtol=2e-3)
    np.testing.assert_allclose(mix.pit(truth), gaussian_pit(errors, s), atol=1e-12)
    np.testing.assert_allclose(mix.hpd_level(truth), gaussian_hpd_level(errors, s), atol=2e-3)
    np.testing.assert_allclose(mix.log_density(truth), stats.norm.logpdf(truth, means[:, 0], s))


def test_gaussian_mixture_of_two_members():
    means, sigmas = np.array([[-1.0, 1.0], [0.0, 0.5]]), np.array([[0.3, 0.3], [1.0, 0.2]])
    mix = GaussianMixture(means, sigmas)
    np.testing.assert_allclose(mix.median()[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(mix.std()[0], math.sqrt(1.0 + 0.09))
    for q in (0.025, 0.16, 0.5, 0.84, 0.975):
        np.testing.assert_allclose(mix.cdf(mix.quantile(q)), q, atol=1e-12)
    x = np.array([0.0, 0.5])
    expected = 0.5 * (stats.norm.pdf(x[:, None], means, sigmas)).sum(axis=1)
    np.testing.assert_allclose(mix.density(x), expected)
    # between two separated peaks the density is low: nearly all mass lies above it
    assert mix.hpd_level(np.array([0.0, 0.5]))[0] > 0.99
    assert mix.hpd_width(0.95)[0] < mix.interval(0.95)[1][0] - mix.interval(0.95)[0][0]
    on_grid = mix.on_grid(np.linspace(-2.0, 2.0, 2001))
    np.testing.assert_allclose(
        on_grid.prior_mass, mix.cdf(np.full(2, 2.0)) - mix.cdf(np.full(2, -2.0))
    )
    np.testing.assert_allclose(on_grid.median(), mix.median(), atol=5e-3)
    summary = mixture_summaries(mix)
    assert set(summary) >= {"dz01", "dz01_mean", "sigma", "lower_68", "upper_95", "hpd_width_68"}
    same = GaussianMixture(np.array([[0.2, 0.2, 0.2]]), np.array([[0.4, 0.4, 0.4]]))
    np.testing.assert_allclose(same.interval(0.68)[1], 0.2 + stats.norm.ppf(0.84) * 0.4)
    for bad_means, bad_sigmas in (([[0.0, 1.0]], [[1.0]]), ([[0.0]], [[0.0]])):
        with pytest.raises(ValueError):
            GaussianMixture(np.array(bad_means), np.array(bad_sigmas))
    with pytest.raises(ValueError):
        mix.quantile(1.0)
