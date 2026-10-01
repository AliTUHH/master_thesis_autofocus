"""Train an autofocus regressor on HoloForge holograms.

Usage::

    python -m src.train --config configs/base.yaml [--data-dir data/processed/small] [--run-name NAME] [--epochs N] [--device cpu]

Artifacts are written to ``<paths.runs_dir>/<run_name>/``: ``best.pt`` (weights + config + target mode +
scaler + setup constants), ``config.yaml`` (resolved), ``history.json``, ``results.json`` (test metrics
of the best checkpoint), figures and a TensorBoard log directory ``tb/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import make_dataloaders
from src.evaluate import compute_metrics, example_figure, make_figures, measure_inference_time, predict
from src.models.cnn import build_model
from src.utils.config import PROJECT_ROOT, load_yaml, require_keys, resolve_path, save_yaml
from src.utils.metrics import metrics_in_physical_units, regression_metrics
from src.utils.plotting import plot_loss_curves
from src.utils.targets import TargetScaler
from src.utils.torch_utils import count_parameters, seed_everything, select_device

__all__ = ["train", "main", "LOSSES", "SCHEDULERS"]

LOSSES: tuple[str, ...] = ("mse", "huber")
SCHEDULERS: tuple[str, ...] = ("cosine", "plateau", "none")


def _build_loss(train_cfg: dict[str, Any]) -> nn.Module:
    name = str(train_cfg.get("loss", "mse"))
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss(delta=float(train_cfg.get("huber_delta", 1.0)))
    raise ValueError(f"unknown train.loss {name!r}; expected one of {LOSSES}")


def _build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict[str, Any], epochs: int):
    name = str(train_cfg.get("scheduler", "cosine"))
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=float(train_cfg.get("lr_min", 0.0)))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(train_cfg.get("plateau_factor", 0.5)), patience=int(train_cfg.get("plateau_patience", 5))
        )
    if name == "none":
        return None
    raise ValueError(f"unknown train.scheduler {name!r}; expected one of {SCHEDULERS}")


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    scaler: TargetScaler,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float | None,
    desc: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One pass over ``loader``; trains when ``optimizer`` is given. Returns mean loss and un-scaled predictions/targets."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.set_grad_enabled(training):
        for images, targets in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, file=sys.stdout):
            images = images.to(device, non_blocking=True)
            scaled = scaler.transform(targets.to(device, non_blocking=True))
            output = model(images)
            loss = criterion(output, scaled)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += loss.item() * images.shape[0]
            count += images.shape[0]
            preds.append(scaler.inverse(output.detach().float().cpu()).numpy().reshape(-1))
            trues.append(targets.numpy().reshape(-1))
    return total / max(count, 1), np.concatenate(preds), np.concatenate(trues)


def train(config: dict[str, Any], data_dir: str | Path | None = None, run_name: str | None = None) -> dict[str, Any]:
    """Train, select the best epoch on validation loss and evaluate that checkpoint on the test split."""
    require_keys(config, ("data", "model", "train", "paths"), "<root>")
    train_cfg = config["train"]
    data_cfg = config["data"]
    require_keys(train_cfg, ("epochs", "lr"), "train")
    epochs = int(train_cfg["epochs"])
    seed = int(train_cfg.get("seed", 0))
    seed_everything(seed, deterministic=bool(train_cfg.get("deterministic", True)))
    device = select_device(train_cfg.get("device", "auto"))

    if data_dir is not None:
        config["data"]["dir"] = str(data_dir)
    loaders, datasets = make_dataloaders(config, seed=seed)
    target_mode = datasets["train"].target_mode
    scaler = TargetScaler.from_values(datasets["train"].targets) if data_cfg.get("standardize_target", True) else TargetScaler.identity()
    setup_constants = datasets["train"].setup_constants

    run_name = run_name or f"{datetime.now():%Y%m%d_%H%M%S}_{config['model'].get('arch', 'cnn')}_{target_mode}"
    run_dir = resolve_path(config["paths"].get("runs_dir", "runs")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config.yaml")
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    model = build_model(config).to(device)
    criterion = _build_loss(train_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    scheduler = _build_scheduler(optimizer, train_cfg, epochs)
    patience = int(train_cfg.get("early_stopping_patience", 0))
    grad_clip = train_cfg.get("grad_clip")

    print(
        f"run {run_name}: device={device}, params={count_parameters(model):,}, target={target_mode}, "
        f"input={datasets['train'].output_shape}, train/val/test={len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}",
        flush=True,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time_s": [], "val_mae_target": [], "val_z01_mae_mm": []}
    best_val, best_epoch, epochs_without_improvement = float("inf"), 0, 0
    checkpoint_path = run_dir / "best.pt"
    val_setup = setup_constants | {"z02_mm": datasets["val"].z02_mm}
    start_all = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, _, _ = _run_epoch(model, loaders["train"], criterion, scaler, device, optimizer, grad_clip, f"epoch {epoch}/{epochs} train")
        val_loss, val_pred, val_true = _run_epoch(model, loaders["val"], criterion, scaler, device, None, None, f"epoch {epoch}/{epochs} val")
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_loss) if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()
        epoch_time = time.perf_counter() - epoch_start

        val_target = regression_metrics(val_pred, val_true)
        val_phys = metrics_in_physical_units(val_pred, val_true, target_mode, val_setup)
        for key, value in (
            ("train_loss", train_loss),
            ("val_loss", val_loss),
            ("lr", lr),
            ("epoch_time_s", epoch_time),
            ("val_mae_target", val_target["mae"]),
            ("val_z01_mae_mm", val_phys["z01_mae_mm"]),
        ):
            history[key].append(float(value))
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("val/mae_target", val_target["mae"], epoch)
        writer.add_scalar("val/z01_mae_mm", val_phys["z01_mae_mm"], epoch)
        writer.add_scalar("val/fr_rel_err_mean_pct", val_phys["fr_rel_err_mean_pct"], epoch)

        improved = val_loss < best_val
        if improved:
            best_val, best_epoch, epochs_without_improvement = val_loss, epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "target_mode": target_mode,
                    "normalization": datasets["train"].normalization,
                    "target_scaler": scaler.to_dict(),
                    "setup_constants": setup_constants,
                    "input_shape": list(datasets["train"].output_shape),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "versions": {"torch": torch.__version__, "numpy": np.__version__},
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        print(
            f"epoch {epoch:3d}/{epochs} | train {train_loss:.4e} | val {val_loss:.4e}{' *' if improved else '  '} | "
            f"val z01 MAE {val_phys['z01_mae_mm']:.2f} mm | lr {lr:.2e} | {epoch_time:.1f} s",
            flush=True,
        )
        if patience and epochs_without_improvement >= patience:
            print(f"early stopping after {epoch} epochs (no val improvement for {patience} epochs)", flush=True)
            break

    train_time = time.perf_counter() - start_all
    writer.flush()
    writer.close()
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    figures = {
        "loss_curves": str(plot_loss_curves(history["train_loss"], history["val_loss"], run_dir / "loss_curves.png", history["lr"], f"{train_cfg.get('loss', 'mse')} (scaled target)")),
    }

    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.to(device)
    test_pred, test_true = predict(model, loaders["test"], device, scaler)
    test_setup = setup_constants | {"z02_mm": datasets["test"].z02_mm}
    test_metrics = compute_metrics(test_pred, test_true, target_mode, test_setup)
    figures.update(make_figures(test_pred, test_true, target_mode, test_setup, run_dir, prefix="test_"))
    figures["examples"] = example_figure(datasets["test"], run_dir, prefix="test_")
    np.savez(run_dir / "test_predictions.npz", pred_target=test_pred, true_target=test_true, z01_true_mm=datasets["test"].z01_mm)

    results = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "device": str(device),
        "num_parameters": count_parameters(model),
        "target_mode": target_mode,
        "target_scaler": scaler.to_dict(),
        "setup_constants": setup_constants,
        "epochs_run": len(history["train_loss"]),
        "epochs_configured": epochs,
        "early_stopped": len(history["train_loss"]) < epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_time_s": train_time,
        "mean_epoch_time_s": float(np.mean(history["epoch_time_s"])),
        "test_metrics": test_metrics,
        "inference_time": measure_inference_time(model, datasets["test"], device, int(data_cfg.get("batch_size", 32))),
        "datasets": {split: ds.summary() for split, ds in datasets.items()},
        "figures": figures,
        "checkpoint": str(checkpoint_path),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    phys = test_metrics["physical"]
    print(
        f"best epoch {best_epoch} | test z01 MAE {phys['z01_mae_mm']:.3f} mm, RMSE {phys['z01_rmse_mm']:.3f} mm, "
        f"p95 {phys['z01_p95_abs_err_mm']:.3f} mm, bias {phys['z01_bias_mm']:+.3f} mm | Fr rel. err {phys['fr_rel_err_mean_pct']:.2f} % | "
        f"results: {run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir}/results.json",
        flush=True,
    )
    return results


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train a learning-based autofocus model.")
    parser.add_argument("--config", default="configs/base.yaml", help="experiment YAML (sections data/model/train/paths)")
    parser.add_argument("--data-dir", default=None, help="directory with train/val/test.hdf5 (overrides data.dir)")
    parser.add_argument("--run-name", default=None, help="name of the run folder below paths.runs_dir")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--device", default=None, help="override train.device (auto | cpu | cuda | mps)")
    args = parser.parse_args(argv)

    config = load_yaml(resolve_path(args.config))
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.device is not None:
        config.setdefault("train", {})["device"] = args.device
    return train(config, data_dir=args.data_dir, run_name=args.run_name)


if __name__ == "__main__":
    main()
