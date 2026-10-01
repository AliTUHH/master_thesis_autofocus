"""Tests for the simulation-based inference module (``src.sbi``): prior, embeddings, NPE smoke training,
posterior sampling, calibration diagnostics on synthetic Gaussian posteriors and the find_focus interface."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.sbi.calibration import (
    central_interval,
    coverage_curve,
    empirical_coverage,
    expected_coverage_error,
    rank_uniformity_test,
    risk_coverage_curve,
    sbc_ranks,
    uncertainty_error_relation,
    wilson_interval,
)
from src.sbi.npe import (
    HologramCNNEmbedding,
    PriorSpec,
    RadialProfileEmbedding,
    build_embedding_net,
    find_focus_bounds,
    load_posterior,
    log_prob_grid,
    member_log_prob,
    parameter_grid,
    point_estimates,
    posterior_members,
    posterior_to_find_focus_init,
    prior_from_config,
    prior_from_meta,
    read_meta,
    sample_posterior,
)
from src.utils.config import PROJECT_ROOT, load_yaml

torch.set_num_threads(2)


# --------------------------------------------------------------------------------------------------
# prior
# --------------------------------------------------------------------------------------------------
def test_prior_from_meta_json(tiny_dataset_dir: Path) -> None:
    meta = read_meta(tiny_dataset_dir)
    prior = prior_from_meta(meta)
    assert (prior.low, prior.high, prior.parameter, prior.source) == (50.0, 300.0, "z01_mm", "meta.json")
    box = prior.build()
    samples = box.sample((200,))
    assert samples.shape == (200, 1)
    assert float(samples.min()) >= 50.0 and float(samples.max()) <= 300.0
    assert prior.contains([50.0, 123.4, 300.0]).all() and not prior.contains(301.0)
    assert PriorSpec.from_dict(prior.to_dict()) == prior

    assert prior_from_config({"bounds": "meta"}, meta) == prior
    explicit = prior_from_config({"bounds": [100.0, 200.0]}, None)
    assert (explicit.low, explicit.high, explicit.source) == (100.0, 200.0, "config")


def test_prior_errors() -> None:
    with pytest.raises(ValueError):
        PriorSpec(low=300.0, high=50.0)
    with pytest.raises(ValueError, match="parameter"):
        PriorSpec(low=1.0, high=2.0, parameter="log_fr")
    with pytest.raises(NotImplementedError, match="log"):
        prior_from_meta({"config": {"setup": {"z01_mm": {"min": 50, "max": 300, "distribution": "loguniform"}}}})
    with pytest.raises(KeyError):
        prior_from_meta({"config": {}})
    with pytest.raises(ValueError):
        prior_from_config({"bounds": "meta"}, None)
    with pytest.raises(ValueError):
        prior_from_config({"bounds": [1.0]}, None)


# --------------------------------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------------------------------
def test_embedding_shapes() -> None:
    mlp = RadialProfileEmbedding(n_bins=256, hidden_units=(64, 32), output_dim=8)
    assert mlp(torch.randn(4, 256)).shape == (4, 8)
    assert mlp(torch.randn(4, 1, 256)).shape == (4, 8)
    with pytest.raises(ValueError):
        mlp(torch.randn(4, 128))

    cnn = HologramCNNEmbedding(in_channels=1, conv_channels=(4, 8), fc_units=16, pool_size=2, output_dim=6)
    assert cnn(torch.randn(2, 1, 64, 64)).shape == (2, 6)
    assert cnn(torch.randn(2, 1, 32, 48)).shape == (2, 6)  # resolution independent (adaptive pooling)
    with pytest.raises(ValueError):
        cnn(torch.randn(2, 64, 64))

    assert isinstance(build_embedding_net({"type": "mlp", "output_dim": 4}, (256,)), RadialProfileEmbedding)
    assert isinstance(build_embedding_net({"type": "cnn", "conv_channels": [4], "fc_units": 8}, (2, 64, 64)), HologramCNNEmbedding)
    assert isinstance(build_embedding_net({"type": "identity"}, (5,)), torch.nn.Identity)
    with pytest.raises(ValueError, match="1-D"):
        build_embedding_net({"type": "mlp"}, (1, 64, 64))
    with pytest.raises(ValueError, match="image"):
        build_embedding_net({"type": "cnn"}, (256,))
    with pytest.raises(ValueError, match="unknown"):
        build_embedding_net({"type": "transformer"}, (256,))


# --------------------------------------------------------------------------------------------------
# NPE smoke training on the tiny dataset
# --------------------------------------------------------------------------------------------------
def _smoke_config(runs_dir: Path, **overrides: dict) -> dict:
    config = copy.deepcopy(load_yaml(PROJECT_ROOT / "configs" / "sbi_radial.yaml"))
    config["data"].update({"batch_size": 3, "representation_kwargs": {"window": True, "log_scale": "auto", "n_bins": 64, "axis": "freq_sq"}})
    config["embedding"].update({"hidden_units": [32, 16], "output_dim": 4})
    config["density_estimator"].update({"hidden_features": 16, "num_transforms": 2, "num_bins": 4})
    config["train"].update(
        {"training_batch_size": 3, "stop_after_epochs": 2, "max_num_epochs": 3, "ensemble_size": 1, "seed": 1, "num_threads": 2}
    )
    config["eval"].update({"n_posterior_samples": 40, "n_examples": 2, "grid_points": 41, "sbi_diagnostics": False})
    config["paths"]["runs_dir"] = str(runs_dir)
    for section, values in overrides.items():
        config[section].update(values)
    return config


def test_npe_smoke_train_evaluate_and_reload(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    from src.sbi.train_npe import train

    config = _smoke_config(tmp_path / "runs")
    results = train(config, data_dir=tiny_dataset_dir, run_name="smoke_npe")
    run_dir = tmp_path / "runs" / "smoke_npe"
    for name in ("posterior.pt", "results.json", "history.json", "config.yaml", "loss_curves.png"):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "eval_test" / "eval_results.json").is_file()
    assert (run_dir / "eval_test" / "posterior_samples.npz").is_file()
    assert results["num_simulations"] == {"train": 6, "val": 3}  # train.hdf5 / val.hdf5 kept as given (files split)
    assert results["prior"]["low"] == 50.0 and results["prior"]["high"] == 300.0
    assert results["ensemble_size"] == 1 and len(results["epochs_trained"]) == 1 and results["epochs_trained"][0] <= 4
    test = results["test"]
    assert set(test["point_estimates"]) == {"mean", "median", "map"}
    assert test["point_estimates"]["mean"]["n"] == 3
    assert len(test["coverage"]) == 4 and all(0.0 <= record["coverage"] <= 1.0 for record in test["coverage"])
    assert 0.0 <= test["sbc"]["ks_pvalue"] <= 1.0
    assert test["inference_time"]["ms_sampling"] > 0
    for figure in test["figures"].values():
        assert (PROJECT_ROOT / figure).is_file() or Path(figure).is_file()

    saved = json.loads((run_dir / "eval_test" / "eval_results.json").read_text())
    assert saved["num_test_samples"] == 3 and saved["fraction_truth_inside_prior"] == 1.0

    bundle = load_posterior(run_dir / "posterior.pt")
    assert bundle["prior"]["low"] == 50.0 and bundle["versions"]["sbi"]
    assert bundle["dataset_kwargs"]["representation"] == "radial_profile"
    posterior = bundle["posterior"]
    assert len(posterior_members(posterior)) == 1
    x = torch.randn(2, 64)
    samples = sample_posterior(posterior, x, n_samples=30, seed=0)
    assert samples.shape == (2, 30)
    assert samples.min() >= 50.0 and samples.max() <= 300.0
    with pytest.raises(ValueError):
        sample_posterior(posterior, x, n_samples=1)


def test_npe_ensemble_smoke(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    from src.sbi.npe import fit_npe

    config = _smoke_config(tmp_path / "runs", train={"ensemble_size": 2})
    result = fit_npe(config, data_dir=tiny_dataset_dir)
    assert result.ensemble_size == 2 and len(result.members) == 2
    assert len(result.summary["epochs_trained"]) == 2 and len(result.summary["members"]) == 2
    assert result.num_parameters["total"] == 2 * result.num_parameters["per_member"]
    x = torch.randn(3, 64)
    samples = sample_posterior(result.posterior, x, n_samples=20, seed=0)
    assert samples.shape == (3, 20)
    assert samples.min() >= 50.0 and samples.max() <= 300.0

    # grid log-density (observation embedded once) must agree with sbi's log_prob (observation replicated)
    grid = parameter_grid(result.prior, 21)
    theta = torch.as_tensor(grid, dtype=torch.float32).reshape(-1, 1)
    member = result.members[0]
    fast = member_log_prob(member, theta, x[:1], normalize=False)
    reference = member.log_prob(theta, x=x[:1], norm_posterior=False)
    assert fast.shape == (21,) and torch.allclose(fast, reference, atol=1e-4)
    outside = member_log_prob(member, torch.tensor([[10.0], [400.0]]), x[:1])
    assert torch.isinf(outside).all() and (outside < 0).all()
    mixture = log_prob_grid(result.posterior, x[:1], grid, normalize=False)
    expected = torch.logsumexp(
        torch.stack([member_log_prob(m, theta, x[:1]) for m in result.members]) + torch.log(result.posterior.weights).reshape(-1, 1), dim=0
    )
    assert mixture.shape == (1, 21) and np.allclose(mixture[0], expected.numpy(), atol=1e-4)


def test_fit_npe_rejects_samples_outside_prior(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    from src.sbi.npe import fit_npe

    config = _smoke_config(tmp_path / "runs", prior={"bounds": [100.0, 150.0]})
    with pytest.raises(ValueError, match="outside the prior"):
        fit_npe(config, data_dir=tiny_dataset_dir)


# --------------------------------------------------------------------------------------------------
# calibration diagnostics on synthetic Gaussian posteriors
# --------------------------------------------------------------------------------------------------
def _gaussian_posteriors(n_obs: int, n_samples: int, scale: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Posterior centre = truth + N(0, 1) noise; samples ~ N(centre, scale^2).

    The posterior is calibrated for ``scale == 1`` (its spread equals the actual estimation noise),
    over-confident for ``scale < 1`` and under-confident for ``scale > 1``.
    """
    rng = np.random.default_rng(seed)
    truth = 150.0 + 40.0 * rng.normal(size=n_obs)
    centre = truth + rng.normal(size=n_obs)
    samples = centre[:, None] + scale * rng.normal(size=(n_obs, n_samples))
    return samples, truth


def test_coverage_matches_nominal_for_calibrated_posteriors() -> None:
    samples, truth = _gaussian_posteriors(3000, 400, scale=1.0)
    for record in empirical_coverage(samples, truth, (0.5, 0.68, 0.9, 0.95)):
        assert abs(record["coverage"] - record["level"]) < 0.03, record
        assert record["ci_low"] <= record["coverage"] <= record["ci_high"]
        assert record["n_covered"] == round(record["coverage"] * record["n"])
    curve = coverage_curve(samples, truth)
    assert expected_coverage_error(curve) < 0.02
    ranks = sbc_ranks(samples, truth)
    assert ranks.min() >= 0 and ranks.max() <= 400
    assert rank_uniformity_test(ranks, 400)["ks_pvalue"] > 0.01


def test_coverage_detects_over_and_under_confidence() -> None:
    over, truth = _gaussian_posteriors(2000, 300, scale=0.5)
    under, _ = _gaussian_posteriors(2000, 300, scale=2.0)
    cov_over = {r["level"]: r["coverage"] for r in empirical_coverage(over, truth, (0.68, 0.95))}
    cov_under = {r["level"]: r["coverage"] for r in empirical_coverage(under, truth, (0.68, 0.95))}
    assert cov_over[0.68] < 0.5 and cov_over[0.95] < 0.75
    assert cov_under[0.68] > 0.85 and cov_under[0.95] > 0.98
    assert rank_uniformity_test(sbc_ranks(over, truth), 300)["ks_pvalue"] < 1e-3
    assert rank_uniformity_test(sbc_ranks(over, truth), 300)["tail_fraction"] > 0.2  # U-shaped histogram


def test_central_interval_and_wilson() -> None:
    samples = np.tile(np.linspace(0.0, 1.0, 1001), (2, 1))
    lo, hi = central_interval(samples, 0.9)
    assert np.allclose(lo, 0.05, atol=1e-3) and np.allclose(hi, 0.95, atol=1e-3)
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high and abs(high - low - 0.196) < 0.01
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        central_interval(samples, 1.5)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_sbc_ranks_outside_support() -> None:
    samples = np.tile(np.linspace(50.0, 300.0, 100), (3, 1))
    ranks = sbc_ranks(samples, [10.0, 175.0, 400.0])
    assert ranks[0] == 0 and ranks[2] == 100 and 45 <= ranks[1] <= 55


def test_uncertainty_relation_and_risk_coverage() -> None:
    rng = np.random.default_rng(1)
    std = rng.uniform(1.0, 10.0, size=500)
    abs_err = np.abs(rng.normal(scale=std))  # errors scale with the uncertainty -> informative
    relation = uncertainty_error_relation(abs_err, std)
    assert relation["spearman_rho"] > 0.3 and relation["pearson_p"] < 1e-6
    curve = risk_coverage_curve(abs_err, std, (1.0, 0.8, 0.5))
    assert [r["keep_fraction"] for r in curve] == [1.0, 0.8, 0.5]
    assert curve[0]["n_kept"] == 500 and curve[1]["n_kept"] == 400
    assert curve[0]["mae"] > curve[1]["mae"] > curve[2]["mae"]
    flat = uncertainty_error_relation(abs_err, np.ones_like(std))
    assert np.isnan(flat["pearson_r"])
    with pytest.raises(ValueError):
        risk_coverage_curve(abs_err, std, (0.0,))


# --------------------------------------------------------------------------------------------------
# point estimates and the find_focus interface
# --------------------------------------------------------------------------------------------------
def test_point_estimates_with_grid_map() -> None:
    rng = np.random.default_rng(2)
    samples = 120.0 + 5.0 * rng.normal(size=(4, 2000))
    grid = np.linspace(50.0, 300.0, 1001)
    logp = -0.5 * ((grid[None, :] - np.array([[100.0], [150.0], [200.0], [250.3]])) / 5.0) ** 2
    est = point_estimates(samples, grid, logp)
    assert np.allclose(est["mean"], 120.0, atol=0.5) and np.allclose(est["median"], 120.0, atol=0.5)
    assert np.allclose(est["std"], 5.0, atol=0.4)
    assert np.allclose(est["map"], [100.0, 150.0, 200.0, 250.3], atol=0.05)  # sub-grid refinement (grid step 0.25)
    assert np.all(est["width_95"] > est["width_68"]) and np.allclose(est["width_95"], 2 * 1.96 * 5.0, rtol=0.1)
    with pytest.raises(ValueError):
        point_estimates(samples[0])


def test_posterior_to_find_focus_init() -> None:
    rng = np.random.default_rng(3)
    samples = 120.0 + 5.0 * rng.normal(size=20000)
    init = posterior_to_find_focus_init(samples)
    assert set(init) == {"z01", "z01_confidence"}
    assert abs(init["z01"] - 120.0) < 0.2
    assert abs(init["z01_confidence"] - 1.96 * 5.0) < 0.5
    lo, hi = find_focus_bounds(init)
    assert lo < 120.0 < hi and np.isclose(hi - lo, 2 * init["z01_confidence"])
    assert np.mean((samples >= lo) & (samples <= hi)) >= 0.95

    assert posterior_to_find_focus_init(samples, min_confidence_mm=50.0)["z01_confidence"] == 50.0
    assert abs(posterior_to_find_focus_init(samples, centre="mean")["z01"] - samples.mean()) < 1e-9
    skewed = np.concatenate([np.full(900, 100.0), np.linspace(100.0, 200.0, 100)])
    skew_init = posterior_to_find_focus_init(skewed, level=0.9)
    assert skew_init["z01"] == 100.0
    assert skew_init["z01_confidence"] >= np.quantile(skewed, 0.95) - 100.0  # covers the asymmetric upper tail
    for bad in ({"level": 1.0}, {"centre": "mode"}, {"min_confidence_mm": -1.0}):
        with pytest.raises(ValueError):
            posterior_to_find_focus_init(samples, **bad)
    with pytest.raises(ValueError):
        posterior_to_find_focus_init([1.0])


# --------------------------------------------------------------------------------------------------
# online simulator sketch
# --------------------------------------------------------------------------------------------------
def test_holoforge_simulator_uses_queued_z01(tiny_config: dict) -> None:
    from src.sbi.simulator import HoloForgeSimulator
    from src.utils.physics import fresnel_number

    simulator = HoloForgeSimulator(copy.deepcopy(tiny_config), preprocess=None, seed=5)
    theta = torch.tensor([[60.0], [250.5]])
    x = simulator(theta)
    assert x.shape == (2, 1, 64, 64) and x.dtype == torch.float32
    assert torch.isfinite(x).all() and float(x.mean()) > 0
    assert simulator.setup.z01 == 250.5 and simulator.setup.z02 == 20000.0
    expected_fr = fresnel_number(250.5, 20000.0, 11.0, 0.0065 * 32)
    assert np.isclose(simulator.setup.Fr, expected_fr, rtol=1e-6)
    with pytest.raises(ValueError, match="outside"):
        simulator.setup.queue_z01(400.0)

    standardised = HoloForgeSimulator(copy.deepcopy(tiny_config), preprocess=lambda h: (h - h.mean()) / h.std(), seed=5)
    y = standardised(torch.tensor([[120.0]]))
    assert y.shape == (1, 1, 64, 64) and abs(float(y.mean())) < 1e-4
