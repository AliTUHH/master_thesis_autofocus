"""Evaluate a trained checkpoint on a HoloForge HDF5 file.

Usage::

    python -m src.evaluate --checkpoint runs/<name>/best.pt --data data/processed/<name>/test.hdf5 [--out DIR]

Writes ``eval_results.json`` (metrics in target space and in physical units, inference timing) plus
scatter and error-over-z01 figures into ``--out`` (default: ``<checkpoint dir>/eval_<data stem>/``).
The functions here are also used by :mod:`src.train` for the final test evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import HologramHDF5Dataset, dataset_kwargs_from_config
from src.models.cnn import build_model
from src.utils.config import PROJECT_ROOT, resolve_path
from src.utils.metrics import metrics_in_physical_units, physical_errors, regression_metrics
from src.utils.plotting import plot_error_vs_z01, plot_example_holograms, plot_radial_profiles, plot_true_vs_pred
from src.utils.targets import TargetScaler
from src.utils.torch_utils import count_parameters, select_device

__all__ = [
    "load_checkpoint",
    "predict",
    "compute_metrics",
    "make_figures",
    "measure_inference_time",
    "evaluate_checkpoint",
    "main",
]

_TARGET_LABELS = {"log_fr": ("ln Fr", ""), "fr": ("Fr", ""), "z01_mm": ("z01", "mm")}


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model from the config stored in the checkpoint and load its weights (eval mode)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "config", "target_mode", "target_scaler"):
        if key not in checkpoint:
            raise KeyError(f"checkpoint {path} lacks {key!r}; was it written by src.train?")
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: TargetScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader; returns ``(pred_target, true_target)`` in the un-scaled target space."""
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for images, targets in loader:
        out = model(images.to(device, non_blocking=True))
        preds.append(scaler.inverse(out.detach().float().cpu()).numpy().reshape(-1))
        trues.append(targets.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(trues)


def compute_metrics(
    pred_target: np.ndarray,
    true_target: np.ndarray,
    target_mode: str,
    setup: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Target-space regression metrics and physical-unit metrics (z01 mm, relative Fr %)."""
    return {
        "target_space": regression_metrics(pred_target, true_target),
        "physical": metrics_in_physical_units(pred_target, true_target, target_mode, setup),
    }


def make_figures(
    pred_target: np.ndarray,
    true_target: np.ndarray,
    target_mode: str,
    setup: dict[str, Any],
    out_dir: Path,
    prefix: str = "",
) -> dict[str, str]:
    """Scatter plots (target space and z01 in mm) and the binned z01 error plot; returns the saved paths."""
    quantity, unit = _TARGET_LABELS[target_mode]
    values = physical_errors(pred_target, true_target, target_mode, setup)
    paths = {
        "scatter_target": plot_true_vs_pred(true_target, pred_target, out_dir / f"{prefix}scatter_target.png", quantity, unit),
        "scatter_z01": plot_true_vs_pred(
            values["z01_true_mm"], values["z01_pred_mm"], out_dir / f"{prefix}scatter_z01_mm.png", "z01", "mm"
        ),
        "error_vs_z01": plot_error_vs_z01(values["z01_true_mm"], values["z01_pred_mm"], out_dir / f"{prefix}error_vs_z01.png"),
    }
    return {key: str(path) for key, path in paths.items()}


def example_figure(dataset: HologramHDF5Dataset, out_dir: Path, num: int = 8, prefix: str = "") -> str:
    """Save a grid of the first ``num`` network inputs (first channel) labelled with their z01 and Fr;
    1-D radial profiles are drawn as curves."""
    num = min(num, len(dataset))
    inputs = torch.stack([dataset[i][0] for i in range(num)]).numpy()
    labels = [f"z01={dataset.z01_mm[i]:.1f} mm\nFr={dataset.fr[i]:.3e}" for i in range(num)]
    title = f"{dataset.representation} ({dataset.normalization}) input"
    if inputs.ndim == 2:
        path = plot_radial_profiles(inputs, labels, out_dir / f"{prefix}examples.png", title=title)
    else:
        path = plot_example_holograms(inputs, labels, out_dir / f"{prefix}examples.png", suptitle=title)
    return str(path)


@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    dataset: HologramHDF5Dataset,
    device: torch.device,
    batch_size: int,
    num_batches: int = 10,
    warmup: int = 2,
) -> dict[str, float]:
    """Forward-pass time per hologram in ms for batch size 1 and ``batch_size`` (excludes data loading)."""
    model.eval()
    sample = dataset[0][0]
    results: dict[str, float] = {}
    for label, bsz in (("batch_1", 1), (f"batch_{batch_size}", batch_size)):
        batch = sample.unsqueeze(0).expand(bsz, *sample.shape).contiguous().to(device)
        for _ in range(warmup):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(num_batches):
            start = time.perf_counter()
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1e3 / bsz)
        results[f"ms_per_hologram_{label}"] = float(np.median(timings))
    results["device"] = str(device)
    results["torch_num_threads"] = torch.get_num_threads()
    results["input_shape"] = list(sample.shape)
    return results


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_path: str | Path,
    out_dir: str | Path | None = None,
    batch_size: int | None = None,
    device_name: str | None = None,
    num_workers: int | None = None,
    num_threads: int | None = None,
) -> dict[str, Any]:
    """Full evaluation of one checkpoint on one HDF5 file; writes ``eval_results.json`` and figures.

    ``num_threads`` limits PyTorch's intra-op CPU threads; on shared or small CPUs the batch-1 latency
    can be an order of magnitude worse with the default thread count because of oversubscription.
    """
    checkpoint_path = resolve_path(checkpoint_path)
    data_path = resolve_path(data_path)
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    device = select_device(device_name or "auto")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    data_cfg = config.get("data", {})
    target_mode = checkpoint["target_mode"]
    scaler = TargetScaler.from_dict(checkpoint["target_scaler"])

    kwargs = dataset_kwargs_from_config(data_cfg) | {
        "target_mode": target_mode,
        "normalization": checkpoint.get("normalization", data_cfg.get("normalization", "standardize")),
    }
    dataset = HologramHDF5Dataset(data_path, **kwargs)
    bsz = int(batch_size or data_cfg.get("batch_size", 32))
    workers = int(num_workers if num_workers is not None else data_cfg.get("num_workers", 0))
    loader = DataLoader(dataset, batch_size=bsz, shuffle=False, num_workers=workers)

    out = resolve_path(out_dir) if out_dir is not None else checkpoint_path.parent / f"eval_{data_path.stem}"
    out.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    pred, true = predict(model, loader, device, scaler)
    wall = time.perf_counter() - start
    setup = dataset.setup_constants | {"z02_mm": dataset.z02_mm}
    metrics = compute_metrics(pred, true, target_mode, setup)
    figures = make_figures(pred, true, target_mode, setup, out)
    figures["examples"] = example_figure(dataset, out)
    timing = measure_inference_time(model, dataset, device, bsz)
    timing["ms_per_hologram_full_pass_incl_loading"] = float(wall * 1e3 / len(dataset))

    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "data": dataset.summary(),
        "target_mode": target_mode,
        "device": str(device),
        "num_parameters": count_parameters(model),
        "metrics": metrics,
        "inference_time": timing,
        "figures": figures,
    }
    np.savez(out / "predictions.npz", pred_target=pred, true_target=true, z01_true_mm=dataset.z01_mm)
    (out / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate an autofocus checkpoint on an HDF5 hologram file.")
    parser.add_argument("--checkpoint", required=True, help="e.g. runs/<name>/best.pt")
    parser.add_argument("--data", required=True, help="e.g. data/processed/<name>/test.hdf5")
    parser.add_argument("--out", default=None, help="output directory (default: <checkpoint dir>/eval_<data stem>)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | mps")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None, help="PyTorch intra-op CPU threads (latency on CPU depends on it)")
    args = parser.parse_args(argv)
    results = evaluate_checkpoint(
        args.checkpoint, args.data, args.out, args.batch_size, args.device, args.num_workers, args.threads
    )
    phys = results["metrics"]["physical"]
    tgt = results["metrics"]["target_space"]
    print(
        f"test n={tgt['n']}  target MAE={tgt['mae']:.4g}  RMSE={tgt['rmse']:.4g}  R2={tgt['r2']:.3f} | "
        f"z01 MAE={phys['z01_mae_mm']:.3f} mm  RMSE={phys['z01_rmse_mm']:.3f} mm  p95={phys['z01_p95_abs_err_mm']:.3f} mm  "
        f"bias={phys['z01_bias_mm']:+.3f} mm | Fr rel. err={phys['fr_rel_err_mean_pct']:.2f} % | "
        f"{results['inference_time']['ms_per_hologram_batch_1']:.2f} ms/hologram (batch 1, "
        f"{results['inference_time']['torch_num_threads']} threads)"
    )
    out_dir = Path(results["figures"]["scatter_target"]).parent
    print(f"results written to {out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir}")
    return results


if __name__ == "__main__":
    main()
