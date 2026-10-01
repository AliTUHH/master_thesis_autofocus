"""Evaluate a stored NPE posterior ``q(z01 | hologram)`` on an HDF5 hologram file.

Usage::

    python -m src.sbi.evaluate_npe --posterior runs/<name>/posterior.pt --data data/processed/small/test.hdf5
        [--n-posterior-samples 1000] [--out DIR] [--seed 0] [--threads 2] [--examples 5]

Per hologram the posterior is sampled (``n`` samples) and evaluated on a grid; reported are

* point estimators (posterior mean, median, MAP) with MAE/RMSE/p95/bias in mm and the relative Fr error
  (same metrics as :func:`src.utils.metrics.metrics_in_physical_units` for the regression models),
* posterior width (std, 68 %/95 % interval widths) and the HoloWizard start values
  (:func:`src.sbi.npe.posterior_to_find_focus_init`),
* **calibration**: empirical coverage of the 50/68/90/95 % central intervals with Wilson intervals, the
  coverage curve + expected coverage error, SBC ranks with a KS test (own implementation and, as a cross
  check, ``sbi.diagnostics.run_sbc``/``check_sbc``), TARP (``sbi.diagnostics.run_tarp``/``check_tarp``),
* **informativeness**: correlation of posterior std with |error|, risk-coverage curve (error after
  discarding the most uncertain holograms),
* **misspecification indicators**: leakage acceptance of the flow and posterior mass at the prior edges
  (useful for holograms whose z01 lies outside the training range),
* runtime per hologram (representation, embedding, sampling, MAP grid).

Everything is written to ``--out`` (default ``<posterior dir>/eval_<data stem>/``): ``eval_results.json``,
``posterior_samples.npz`` and the figures.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from src.data.dataset import HologramHDF5Dataset
from src.sbi.calibration import (
    DEFAULT_COVERAGE_LEVELS,
    coverage_curve,
    empirical_coverage,
    expected_coverage_error,
    rank_uniformity_test,
    risk_coverage_curve,
    sbc_ranks,
    uncertainty_error_relation,
)
from src.sbi.npe import (
    PriorSpec,
    find_focus_bounds,
    leakage_acceptance,
    load_posterior,
    load_simulations,
    log_prob_grid,
    parameter_grid,
    point_estimates,
    posterior_to_find_focus_init,
    sample_posterior,
)
from src.sbi.plots import (
    plot_coverage_curve,
    plot_interval_plot,
    plot_point_estimate_scatter,
    plot_posterior_examples,
    plot_sbc_histogram,
    plot_uncertainty_vs_error,
)
from src.utils.config import PROJECT_ROOT, resolve_path
from src.utils.metrics import metrics_in_physical_units

__all__ = ["evaluate_posterior", "measure_inference_time", "sbi_diagnostics", "main"]

_POINT_ESTIMATORS: tuple[str, ...] = ("mean", "median", "map")


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@torch.no_grad()
def measure_inference_time(
    posterior: Any,
    dataset: HologramHDF5Dataset,
    x: torch.Tensor,
    grid: np.ndarray,
    n_samples: int,
    num_holograms: int = 10,
    warmup: int = 2,
) -> dict[str, float]:
    """Median wall time per hologram in ms: representation (preprocessing of the raw hologram), embedding
    forward pass, drawing ``n_samples`` posterior samples (includes the embedding) and the MAP grid search."""
    n = min(num_holograms, len(dataset), x.shape[0])
    with h5py.File(dataset.path, "r") as handle:
        raws = [np.asarray(handle[dataset.hologram_key][i], dtype=np.float32) for i in range(n)]
    embedding = posterior.posterior_estimator.embedding_net
    theta_grid = torch.as_tensor(grid, dtype=torch.float32).reshape(-1, 1)
    timings: dict[str, list[float]] = {"representation": [], "embedding": [], "sampling": [], "map_grid": []}
    for i in range(n):
        raw = torch.from_numpy(raws[i]).unsqueeze(0)
        for _ in range(warmup if i == 0 else 0):
            dataset.preprocess(raw)
            embedding(x[i : i + 1])
            posterior.sample((n_samples,), x=x[i : i + 1], show_progress_bars=False)
            posterior.log_prob(theta_grid, x=x[i : i + 1], norm_posterior=False)
        start = time.perf_counter()
        dataset.preprocess(raw)
        timings["representation"].append(time.perf_counter() - start)
        start = time.perf_counter()
        embedding(x[i : i + 1])
        timings["embedding"].append(time.perf_counter() - start)
        start = time.perf_counter()
        posterior.sample((n_samples,), x=x[i : i + 1], show_progress_bars=False)
        timings["sampling"].append(time.perf_counter() - start)
        start = time.perf_counter()
        posterior.log_prob(theta_grid, x=x[i : i + 1], norm_posterior=False)
        timings["map_grid"].append(time.perf_counter() - start)
    result = {f"ms_{key}": float(np.median(values) * 1e3) for key, values in timings.items()}
    result["ms_total_representation_and_sampling"] = result["ms_representation"] + result["ms_sampling"]
    result["n_posterior_samples"] = int(n_samples)
    result["grid_points"] = int(len(grid))
    result["torch_num_threads"] = torch.get_num_threads()
    result["num_holograms_timed"] = int(n)
    return result


def sbi_diagnostics(posterior: Any, theta: torch.Tensor, x: torch.Tensor, n_samples: int) -> dict[str, Any]:
    """Cross-check with ``sbi``'s own SBC and TARP implementations (errors are reported, not raised)."""
    out: dict[str, Any] = {}
    try:
        from sbi.diagnostics import check_sbc, run_sbc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ranks, dap_samples = run_sbc(theta, x, posterior, num_posterior_samples=n_samples, show_progress_bar=False)
            checks = check_sbc(ranks, theta, dap_samples, num_posterior_samples=n_samples)
        out["sbc"] = {
            "ks_pvalue": float(checks["ks_pvals"].reshape(-1)[0]),
            "c2st_ranks": float(checks["c2st_ranks"].reshape(-1)[0]),
            "c2st_dap": float(checks["c2st_dap"].reshape(-1)[0]),
            "n_posterior_samples": int(n_samples),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics must never abort the evaluation
        out["sbc"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        from sbi.diagnostics import check_tarp, run_tarp

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ecp, alpha = run_tarp(theta, x, posterior, num_posterior_samples=n_samples, show_progress_bar=False)
            atc, ks_pvalue = check_tarp(ecp, alpha)
        out["tarp"] = {
            "atc": float(atc),
            "ks_pvalue": float(ks_pvalue),
            "ecp": ecp.reshape(-1).tolist(),
            "alpha": alpha.reshape(-1).tolist(),
            "max_abs_deviation": float((ecp.reshape(-1) - alpha.reshape(-1)).abs().max()),
        }
    except Exception as exc:  # noqa: BLE001
        out["tarp"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _edge_mass(samples: np.ndarray, prior: PriorSpec, margin: float = 0.05) -> dict[str, float]:
    """Fraction of posterior samples within ``margin`` of the prior width from the lower/upper bound."""
    band = margin * prior.width
    low = float(np.mean(samples <= prior.low + band))
    high = float(np.mean(samples >= prior.high - band))
    return {"margin_fraction_of_prior_width": margin, "lower_edge": low, "upper_edge": high, "total": low + high}


def _example_indices(truth: np.ndarray, n_examples: int) -> np.ndarray:
    """Indices spread over the range of true values (quantiles), so the examples show different z01."""
    n = min(n_examples, truth.size)
    if n <= 0:
        return np.empty(0, dtype=int)
    order = np.argsort(truth)
    positions = np.linspace(0, truth.size - 1, n).round().astype(int)
    return order[positions]


def evaluate_posterior(
    posterior_path: str | Path,
    data_path: str | Path,
    out_dir: str | Path | None = None,
    n_posterior_samples: int = 1000,
    n_examples: int = 5,
    coverage_levels: tuple[float, ...] = DEFAULT_COVERAGE_LEVELS,
    seed: int = 0,
    num_threads: int | None = None,
    grid_points: int = 1001,
    run_sbi_diagnostics: bool = True,
    num_timing_holograms: int = 10,
) -> dict[str, Any]:
    """Full evaluation of one posterior on one HDF5 file; writes ``eval_results.json``, samples and figures."""
    posterior_path = resolve_path(posterior_path)
    data_path = resolve_path(data_path)
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    bundle = load_posterior(posterior_path)
    posterior = bundle["posterior"]
    prior = PriorSpec.from_dict(bundle["prior"])
    dataset = HologramHDF5Dataset(data_path, **bundle["dataset_kwargs"])
    out = resolve_path(out_dir) if out_dir is not None else posterior_path.parent / f"eval_{data_path.stem}"
    out.mkdir(parents=True, exist_ok=True)

    theta, x = load_simulations(dataset)
    truth = theta.numpy().reshape(-1).astype(np.float64)
    inside_prior = prior.contains(truth)

    start = time.perf_counter()
    samples = sample_posterior(posterior, x, n_posterior_samples, seed=seed)
    sampling_wall = time.perf_counter() - start
    grid = parameter_grid(prior, grid_points)
    logp = log_prob_grid(posterior, x, grid)
    estimates = point_estimates(samples, grid, logp)

    setup = dataset.setup_constants | {"z02_mm": dataset.z02_mm}
    point_metrics = {name: metrics_in_physical_units(estimates[name], truth, "z01_mm", setup) for name in _POINT_ESTIMATORS}
    abs_err_mean = np.abs(estimates["mean"] - truth)
    abs_err_median = np.abs(estimates["median"] - truth)

    coverage = empirical_coverage(samples, truth, coverage_levels)
    curve = coverage_curve(samples, truth)
    ranks = sbc_ranks(samples, truth)
    ks = rank_uniformity_test(ranks, n_posterior_samples)
    relation = uncertainty_error_relation(abs_err_mean, estimates["std"])
    risk_mean = risk_coverage_curve(abs_err_mean, estimates["std"])
    risk_median = risk_coverage_curve(abs_err_median, estimates["std"])
    acceptance = leakage_acceptance(posterior, x)
    inits = [posterior_to_find_focus_init(row) for row in samples]
    init_z01 = np.array([init["z01"] for init in inits])
    init_conf = np.array([init["z01_confidence"] for init in inits])
    init_bounds = np.array([find_focus_bounds(init) for init in inits])
    init_covers = (truth >= init_bounds[:, 0]) & (truth <= init_bounds[:, 1])

    diagnostics = sbi_diagnostics(posterior, theta, x, n_posterior_samples) if run_sbi_diagnostics else {}
    timing = measure_inference_time(posterior, dataset, x, grid, n_posterior_samples, num_timing_holograms)
    timing["ms_sampling_full_pass_batched"] = float(sampling_wall * 1e3 / len(dataset))

    idx = _example_indices(truth, n_examples)
    with h5py.File(dataset.path, "r") as handle:
        holograms = np.stack([np.asarray(handle[dataset.hologram_key][int(i)], dtype=np.float32) for i in idx]) if idx.size else None
    labels = [f"#{int(i)}: z01={truth[i]:.1f} mm, Fr={dataset.fr[i]:.2e}\nmean {estimates['mean'][i]:.1f}, std {estimates['std'][i]:.1f} mm" for i in idx]
    figures = {
        "posterior_examples": plot_posterior_examples(
            grid, logp[idx], samples[idx], truth[idx], {k: estimates[k][idx] for k in _POINT_ESTIMATORS}, (prior.low, prior.high), out / "posterior_examples.png", holograms, labels
        ),
        "coverage_curve": plot_coverage_curve(curve, out / "coverage_curve.png", coverage),
        "sbc_histogram": plot_sbc_histogram(ranks, n_posterior_samples, out / "sbc_histogram.png", ks_pvalue=ks["ks_pvalue"]),
        "std_vs_abs_error": plot_uncertainty_vs_error(estimates["std"], abs_err_mean, out / "std_vs_abs_error.png", relation, risk_mean),
        "scatter_estimates": plot_point_estimate_scatter(truth, {k: estimates[k] for k in _POINT_ESTIMATORS}, out / "scatter_estimates_vs_true.png", estimates["std"], (prior.low, prior.high)),
        "interval_plot": plot_interval_plot(truth, estimates["median"], estimates["lo_95"], estimates["hi_95"], out / "interval_plot.png", 0.95, (prior.low, prior.high)),
    }

    results: dict[str, Any] = {
        "posterior": _portable(posterior_path),
        "data": dataset.summary() | {"path": _portable(data_path)},
        "prior": prior.to_dict(),
        "n_posterior_samples": int(n_posterior_samples),
        "grid_points": int(grid_points),
        "seed": int(seed),
        "num_test_samples": int(truth.size),
        "fraction_truth_inside_prior": float(inside_prior.mean()),
        "point_estimates": point_metrics,
        "posterior_width": {
            "std_mean_mm": float(estimates["std"].mean()),
            "std_median_mm": float(np.median(estimates["std"])),
            "std_min_mm": float(estimates["std"].min()),
            "std_max_mm": float(estimates["std"].max()),
            "width68_mean_mm": float(estimates["width_68"].mean()),
            "width95_mean_mm": float(estimates["width_95"].mean()),
            "prior_std_mm": float(prior.width / np.sqrt(12.0)),
        },
        "calibration": {
            "coverage": coverage,
            "coverage_curve": {key: value.tolist() for key, value in curve.items()},
            "expected_coverage_error": expected_coverage_error(curve),
            "sbc": ks | {"n_posterior_samples": int(n_posterior_samples), "ranks_mean_normalised": float(ranks.mean() / n_posterior_samples)},
            "sbi_diagnostics": diagnostics,
        },
        "informativeness": {
            "std_vs_abs_error_mean": relation,
            "risk_coverage_mean": risk_mean,
            "risk_coverage_median": risk_median,
        },
        "misspecification": {
            "leakage_acceptance_mean": float(acceptance.mean()),
            "leakage_acceptance_min": float(acceptance.min()),
            "edge_mass": _edge_mass(samples, prior),
        },
        "find_focus_init": {
            "level": 0.95,
            "z01_confidence_mean_mm": float(init_conf.mean()),
            "z01_confidence_median_mm": float(np.median(init_conf)),
            "fraction_truth_inside_bounds": float(init_covers.mean()),
            "example": {"z01": float(init_z01[0]), "z01_confidence": float(init_conf[0]), "true_z01": float(truth[0])},
        },
        "inference_time": timing,
        "figures": {key: _portable(Path(value)) for key, value in figures.items()},
        "versions": bundle.get("versions", {}),
    }
    np.savez(
        out / "posterior_samples.npz",
        samples=samples.astype(np.float32),
        z01_true_mm=truth,
        grid=grid,
        log_prob=logp.astype(np.float32),
        ranks=ranks,
        leakage_acceptance=acceptance,
        init_z01=init_z01,
        init_confidence=init_conf,
        **{f"est_{k}": v for k, v in estimates.items()},
    )
    (out / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate an NPE posterior q(z01 | hologram) on an HDF5 file.")
    parser.add_argument("--posterior", required=True, help="e.g. runs/<name>/posterior.pt")
    parser.add_argument("--data", required=True, help="e.g. data/processed/small/test.hdf5")
    parser.add_argument("--out", default=None, help="output directory (default: <posterior dir>/eval_<data stem>)")
    parser.add_argument("--n-posterior-samples", type=int, default=1000)
    parser.add_argument("--examples", type=int, default=5, help="number of posterior example panels")
    parser.add_argument("--seed", type=int, default=0, help="seed of the posterior sampling")
    parser.add_argument("--threads", type=int, default=None, help="PyTorch intra-op CPU threads")
    parser.add_argument("--grid-points", type=int, default=1001, help="grid resolution for MAP/density over the prior")
    parser.add_argument("--no-sbi-diagnostics", action="store_true", help="skip sbi's run_sbc/run_tarp cross-checks")
    args = parser.parse_args(argv)
    results = evaluate_posterior(
        args.posterior,
        args.data,
        args.out,
        n_posterior_samples=args.n_posterior_samples,
        n_examples=args.examples,
        seed=args.seed,
        num_threads=args.threads,
        grid_points=args.grid_points,
        run_sbi_diagnostics=not args.no_sbi_diagnostics,
    )
    pm = results["point_estimates"]
    cov = {record["level"]: record["coverage"] for record in results["calibration"]["coverage"]}
    print(
        f"n={results['num_test_samples']} | posterior mean MAE {pm['mean']['z01_mae_mm']:.2f} mm, median {pm['median']['z01_mae_mm']:.2f} mm, "
        f"MAP {pm['map']['z01_mae_mm']:.2f} mm | mean std {results['posterior_width']['std_mean_mm']:.2f} mm | "
        f"coverage 68 % {cov.get(0.68, float('nan')) * 100:.0f} %, 95 % {cov.get(0.95, float('nan')) * 100:.0f} % | "
        f"SBC KS p {results['calibration']['sbc']['ks_pvalue']:.3f} | "
        f"{results['inference_time']['ms_total_representation_and_sampling']:.1f} ms/hologram ({results['n_posterior_samples']} samples)"
    )
    print(f"results written to {Path(results['figures']['coverage_curve']).parent}")
    return results


if __name__ == "__main__":
    main()
