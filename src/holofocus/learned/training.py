"""Training of the learned autofocus with a validation split for model selection.

Outputs in the run directory:

* ``config.yaml``: the resolved :class:`~holofocus.learned.config.LearnedConfig`;
* ``history.csv``: per epoch the training loss and validation loss, MAE, RMSE, bias, NLL
  and 68 % coverage (errors in micrometres);
* ``model.pt``: weights of the epoch with the lowest validation MAE, the config, the field
  of view and the variance calibration ``sigma_scale``;
* ``run.yaml``: datasets (path, SHA-256, seed, generator commit), software, device, host,
  sizes and timing;
* ``summary.json``: validation metrics of the selected model with calibrated ``sigma``.

``sigma_scale = rms((mean - t) / sigma)`` on the validation split rescales the predicted
standard deviations so that their NLL is minimal there; the test split is never used.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

from .. import __version__
from ..baseline.runner import dataset_identity
from ..dataset.storage import _git_commit
from ..device import get_device
from .config import LearnedConfig
from .data import fov_shape, load_inputs, load_many, random_d4
from .metrics import calibration_scale, interval_metrics, point_metrics
from .models import build_model, count_parameters, gaussian_nll
from .representations import Representation

FORMAT_VERSION = 1
HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_mae_um",
    "val_rmse_um",
    "val_bias_um",
    "val_nll",
    "val_coverage_68",
    "learning_rate",
    "seconds",
)


def warmup_cosine(step: int, warmup: int, total: int, floor: float = 0.01) -> float:
    """Learning-rate factor: linear warm-up over ``warmup`` steps, cosine decay to ``floor``."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def predict_inputs(
    model: torch.nn.Module, inputs: torch.Tensor, device: torch.device, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance (target units) for every row of ``inputs``."""
    model.eval()
    means, variances = [], []
    for start in range(0, len(inputs), batch_size):
        mean, var = model(inputs[start : start + batch_size].to(device, torch.float32))
        means.append(mean.double().cpu())
        variances.append(var.double().cpu())
    return torch.cat(means).numpy(), torch.cat(variances).numpy()


def evaluation_metrics(
    mean: np.ndarray, var: np.ndarray, target: np.ndarray, target_scale: float, sigma_scale=1.0
) -> dict:
    """Point and interval metrics in micrometres of ``z01``."""
    errors_um = (mean - target) * target_scale * 1e6
    sigmas_um = sigma_scale * np.sqrt(var) * target_scale * 1e6
    return {
        "error_um": point_metrics(errors_um),
        "interval": interval_metrics(errors_um, sigmas_um),
    }


def _append_history(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row[k] for k in HISTORY_FIELDS})


def read_history(path: str | Path) -> list[dict]:
    with open(path, newline="") as fh:
        return [
            {k: int(v) if k == "epoch" else float(v) for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]


def train_model(
    config: LearnedConfig,
    train: str | Path | Sequence[str | Path],
    val: str | Path,
    out: str | Path,
    device: str | None = None,
    overwrite: bool = False,
    log: Callable[[str], None] | None = print,
) -> dict:
    """Train on the split file(s) ``train``, select and calibrate on ``val``; returns the
    summary. Several training files are concatenated in the given order."""
    trains = [Path(train)] if isinstance(train, (str, Path)) else [Path(p) for p in train]
    val, out = Path(val), Path(out)
    if (out / "model.pt").exists() and not overwrite:
        raise ValueError(f"{out} already holds a trained model (use overwrite)")
    shape = fov_shape(val)
    if any(fov_shape(path) != shape for path in trains):
        raise ValueError(f"{[str(p) for p in trains]} and {val} have different fields of view")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("history.csv", "summary.json"):
        (out / name).unlink(missing_ok=True)
    (out / "config.yaml").write_text(config.to_yaml())

    cfg = config.training
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    augment_rng = torch.Generator().manual_seed(cfg.seed)
    torch_device = get_device(device or config.device)
    representation = Representation(config.representation, shape, torch_device)
    start = time.perf_counter()
    train_data = load_many(trains, representation, config.target_scale, limit=cfg.limit)
    val_data = load_inputs(val, representation, config.target_scale)
    load_seconds = time.perf_counter() - start
    model = build_model(config).to(torch_device)
    augment = cfg.augment and config.representation.kind == "image"
    if log is not None:
        log(
            f"{config.name}: {len(train_data)} training / {len(val_data)} validation samples, "
            f"input {representation.input_shape}, {count_parameters(model)} parameters, "
            f"{torch_device}, {torch.get_num_threads()} threads (inputs in {load_seconds:.1f} s)"
        )

    batch = min(cfg.batch_size, len(train_data))
    steps_per_epoch = max(1, len(train_data) // batch)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine(step, cfg.warmup_epochs * steps_per_epoch, total_steps),
    )
    val_targets = val_data.targets.double().numpy()
    best = {"mae": math.inf, "epoch": 0, "state": None}
    train_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_data))
        losses = []
        for step in range(steps_per_epoch):
            pick = torch.from_numpy(order[step * batch : (step + 1) * batch])
            x = train_data.inputs[pick].to(torch_device, torch.float32)
            if augment:
                x = random_d4(x, augment_rng)
            mean, var = model(x)
            loss = gaussian_nll(mean, var, train_data.targets[pick].to(torch_device), cfg.beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        mean, var = predict_inputs(model, val_data.inputs, torch_device)
        target = torch.as_tensor(val_targets)
        val_loss = float(
            gaussian_nll(torch.as_tensor(mean), torch.as_tensor(var), target, cfg.beta)
        )
        metrics = evaluation_metrics(mean, var, val_targets, config.target_scale)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_mae_um": metrics["error_um"]["mae"],
            "val_rmse_um": metrics["error_um"]["rmse"],
            "val_bias_um": metrics["error_um"]["bias"],
            "val_nll": metrics["interval"]["nll"],
            "val_coverage_68": metrics["interval"]["coverage"]["68"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        _append_history(out / "history.csv", row)
        if row["val_mae_um"] < best["mae"]:
            best = {"mae": row["val_mae_um"], "epoch": epoch, "state": _copy_state(model)}
        if log is not None:
            log(
                f"epoch {epoch:3d}: train {row['train_loss']:+.4f}, val {val_loss:+.4f}, "
                f"MAE {row['val_mae_um']:.0f} um, RMSE {row['val_rmse_um']:.0f} um, "
                f"bias {row['val_bias_um']:+.0f} um, cov68 {row['val_coverage_68']:.2f} "
                f"({row['seconds']:.1f} s)"
            )
        if cfg.patience is not None and epoch - best["epoch"] >= cfg.patience:
            break
    train_seconds = time.perf_counter() - train_start

    model.load_state_dict(best["state"])
    mean, var = predict_inputs(model, val_data.inputs, torch_device)
    sigma_scale = calibration_scale(mean - val_targets, np.sqrt(var))
    summary = {
        "name": config.name,
        "best_epoch": best["epoch"],
        "epochs_run": epoch,
        "sigma_scale": sigma_scale,
        "validation": evaluation_metrics(mean, var, val_targets, config.target_scale, sigma_scale),
        "validation_uncalibrated": evaluation_metrics(mean, var, val_targets, config.target_scale),
        "train_seconds": train_seconds,
    }
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "config": config.to_dict(),
            "fov_shape": list(shape),
            "sigma_scale": sigma_scale,
            "best_epoch": best["epoch"],
            "state_dict": best["state"],
        },
        out / "model.pt",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    meta = {
        "config": config.to_dict(),
        "datasets": {
            "train": [dataset_identity(path) for path in trains],
            "val": dataset_identity(val),
        },
        "n_train": len(train_data),
        "n_val": len(val_data),
        "input_shape": list(representation.input_shape),
        "n_parameters": count_parameters(model),
        "best_epoch": best["epoch"],
        "epochs_run": epoch,
        "sigma_scale": sigma_scale,
        "load_seconds": load_seconds,
        "train_seconds": train_seconds,
        "holofocus_version": __version__,
        "git_commit": _git_commit(),
        "torch_version": str(torch.__version__),
        "device": str(torch_device),
        "threads": torch.get_num_threads(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "run.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    if log is not None:
        val_metrics = summary["validation"]
        log(
            f"best epoch {best['epoch']}: validation MAE {val_metrics['error_um']['mae']:.0f} um, "
            f"sigma scale {sigma_scale:.3f}, coverage 68/95 % "
            f"{val_metrics['interval']['coverage']['68']:.2f}/"
            f"{val_metrics['interval']['coverage']['95']:.2f} ({train_seconds:.0f} s)"
        )
    return summary


def _copy_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
