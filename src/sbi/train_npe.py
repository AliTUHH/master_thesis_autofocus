"""Train an amortised NPE posterior ``q(z01 | hologram)`` with ``sbi`` and evaluate it on the test split.

Usage::

    python -m src.sbi.train_npe --config configs/sbi_radial.yaml [--data-dir data/processed/small] [--run-name NAME]
        [--max-epochs N] [--no-eval]

Artifacts in ``<paths.runs_dir>/<run_name>/``: ``config.yaml`` (resolved), ``posterior.pt`` (pickled
``DirectPosterior`` + config + prior + geometry, see :func:`src.sbi.npe.save_posterior`), ``history.json``
(per-epoch losses), ``loss_curves.png``, ``results.json`` (training summary + test evaluation summary),
``tb/`` (TensorBoard log of ``sbi``) and ``eval_test/`` (full evaluation by :mod:`src.sbi.evaluate_npe`).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from src.sbi.evaluate_npe import evaluate_posterior
from src.sbi.npe import fit_npe, save_posterior
from src.sbi.plots import plot_npe_loss_curves
from src.utils.config import PROJECT_ROOT, load_yaml, require_keys, resolve_path, save_yaml

__all__ = ["train", "main"]


def _best_epoch(validation_loss: list[float]) -> int | None:
    if not validation_loss:
        return None
    return int(min(range(len(validation_loss)), key=lambda i: validation_loss[i]) + 1)


def train(config: dict[str, Any], data_dir: str | Path | None = None, run_name: str | None = None, evaluate: bool = True) -> dict[str, Any]:
    """Fit the posterior, store it and (optionally) run the full test evaluation; returns the results mapping."""
    require_keys(config, ("data", "prior", "embedding", "density_estimator", "train", "paths"), "<root>")
    if data_dir is not None:
        config["data"]["dir"] = str(data_dir)
    directory = resolve_path(config["data"]["dir"])
    run_name = run_name or f"{datetime.now():%Y%m%d_%H%M%S}_npe_{config['embedding'].get('type', 'mlp')}"
    run_dir = resolve_path(config["paths"].get("runs_dir", "runs")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config.yaml")
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    start = time.perf_counter()
    result = fit_npe(config, data_dir=directory, summary_writer=writer)
    writer.flush()
    writer.close()
    posterior_path = save_posterior(run_dir / "posterior.pt", result, config)

    summary = result.summary
    history = {
        "train_loss": summary.get("training_loss", []),
        "val_loss": summary.get("validation_loss", []),
        "epoch_time_s": summary.get("epoch_durations_sec", []),
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_epoch = _best_epoch(history["val_loss"])
    figures = {"loss_curves": str(plot_npe_loss_curves(history["train_loss"], history["val_loss"], run_dir / "loss_curves.png", best_epoch))}
    print(
        f"run {run_name}: prior z01 in [{result.prior.low}, {result.prior.high}] mm ({result.prior.source}), "
        f"input={result.input_shape}, embedding={config['embedding'].get('type', 'mlp')} ({result.num_parameters['embedding']:,} params), "
        f"estimator={config['density_estimator'].get('model', 'nsf')} ({result.num_parameters['total']:,} params total), "
        f"train/val={result.num_simulations['train']}/{result.num_simulations['val']} ({result.validation_split} split) | "
        f"{summary.get('epochs_trained')} epochs (best {best_epoch}), best val loss {summary.get('best_validation_loss'):.4f}, "
        f"{result.train_time_s:.1f} s",
        flush=True,
    )

    results: dict[str, Any] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "posterior": str(posterior_path),
        "prior": result.prior.to_dict(),
        "setup_constants": result.setup_constants,
        "input_shape": list(result.input_shape),
        "num_parameters": result.num_parameters,
        "num_simulations": result.num_simulations,
        "validation_split": result.validation_split,
        "epochs_trained": summary.get("epochs_trained"),
        "best_epoch": best_epoch,
        "best_validation_loss": summary.get("best_validation_loss"),
        "early_stopped": bool(summary.get("epochs_trained", 0) < int(config["train"].get("max_num_epochs", 500))),
        "train_time_s": result.train_time_s,
        "datasets": {split: ds.summary() for split, ds in result.datasets.items()},
        "figures": figures,
    }
    if evaluate:
        test_path = directory / "test.hdf5"
        if test_path.is_file():
            eval_cfg = config.get("eval", {}) or {}
            test_results = evaluate_posterior(
                posterior_path,
                test_path,
                out_dir=run_dir / "eval_test",
                n_posterior_samples=int(eval_cfg.get("n_posterior_samples", 1000)),
                n_examples=int(eval_cfg.get("n_examples", 5)),
                coverage_levels=tuple(eval_cfg.get("coverage_levels", (0.5, 0.68, 0.9, 0.95))),
                seed=int(eval_cfg.get("seed", 0)),
                grid_points=int(eval_cfg.get("grid_points", 1001)),
                run_sbi_diagnostics=bool(eval_cfg.get("sbi_diagnostics", True)),
            )
            results["test"] = {
                "eval_results": str(run_dir / "eval_test" / "eval_results.json"),
                "point_estimates": test_results["point_estimates"],
                "posterior_width": test_results["posterior_width"],
                "coverage": test_results["calibration"]["coverage"],
                "expected_coverage_error": test_results["calibration"]["expected_coverage_error"],
                "sbc": test_results["calibration"]["sbc"],
                "sbi_diagnostics": test_results["calibration"]["sbi_diagnostics"],
                "informativeness": test_results["informativeness"],
                "misspecification": test_results["misspecification"],
                "find_focus_init": test_results["find_focus_init"],
                "inference_time": test_results["inference_time"],
                "figures": test_results["figures"],
            }
            pm = test_results["point_estimates"]
            cov = {record["level"]: record["coverage"] for record in test_results["calibration"]["coverage"]}
            print(
                f"test: posterior mean MAE {pm['mean']['z01_mae_mm']:.2f} mm (RMSE {pm['mean']['z01_rmse_mm']:.2f}, p95 {pm['mean']['z01_p95_abs_err_mm']:.2f}), "
                f"median MAE {pm['median']['z01_mae_mm']:.2f} mm, MAP MAE {pm['map']['z01_mae_mm']:.2f} mm | "
                f"mean posterior std {test_results['posterior_width']['std_mean_mm']:.2f} mm | coverage 68 % {cov.get(0.68, float('nan')) * 100:.0f} %, "
                f"95 % {cov.get(0.95, float('nan')) * 100:.0f} % | SBC KS p {test_results['calibration']['sbc']['ks_pvalue']:.3f}",
                flush=True,
            )
        else:
            print(f"no test split at {test_path}; skipping evaluation", flush=True)
    results["total_time_s"] = time.perf_counter() - start
    results["torch_num_threads"] = torch.get_num_threads()
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"results: {run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir}/results.json", flush=True)
    return results


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train an NPE posterior q(z01 | hologram) with sbi.")
    parser.add_argument("--config", default="configs/sbi_radial.yaml", help="experiment YAML (sections data/prior/embedding/density_estimator/train/eval/paths)")
    parser.add_argument("--data-dir", default=None, help="directory with train/val/test.hdf5 + meta.json (overrides data.dir)")
    parser.add_argument("--run-name", default=None, help="name of the run folder below paths.runs_dir")
    parser.add_argument("--max-epochs", type=int, default=None, help="override train.max_num_epochs")
    parser.add_argument("--no-eval", action="store_true", help="skip the test-split evaluation")
    args = parser.parse_args(argv)
    config = load_yaml(resolve_path(args.config))
    if args.max_epochs is not None:
        config.setdefault("train", {})["max_num_epochs"] = args.max_epochs
    return train(config, data_dir=args.data_dir, run_name=args.run_name, evaluate=not args.no_eval)


if __name__ == "__main__":
    main()
