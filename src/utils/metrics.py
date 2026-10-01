"""Regression metrics in target space and in physical units (z01 in mm, relative Fr error in %)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.utils.physics import fresnel_number, target_to_fr, target_to_z01_mm

__all__ = ["regression_metrics", "metrics_in_physical_units", "physical_errors"]


def _as_1d(values: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def regression_metrics(pred: ArrayLike, true: ArrayLike) -> dict[str, float]:
    """MAE, RMSE, 95th percentile of |error|, bias (mean of pred-true), median |error|, R² and sample count."""
    pred_arr = _as_1d(pred, "pred")
    true_arr = _as_1d(true, "true")
    if pred_arr.shape != true_arr.shape:
        raise ValueError(f"shape mismatch: pred {pred_arr.shape} vs true {true_arr.shape}")
    err = pred_arr - true_arr
    abs_err = np.abs(err)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true_arr - true_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "p95_abs_err": float(np.percentile(abs_err, 95)),
        "bias": float(err.mean()),
        "median_abs_err": float(np.median(abs_err)),
        "r2": float(r2),
        "n": int(pred_arr.size),
    }


def physical_errors(
    pred_target: ArrayLike,
    true_target: ArrayLike,
    target_mode: str,
    setup: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Per-sample z01 (mm) and Fr values for predictions and ground truth, independent of the target mode.

    ``setup`` must provide ``z02_mm``, ``energy_kev`` and ``px_mm``; values may be scalars or per-sample
    arrays (NumPy broadcasting). Predicted z01 values outside ``(0, z02)`` cannot be converted to a
    Fresnel number and are clipped to ``[1e-3 mm, z02 - 1e-3 mm]`` for the Fr comparison only.
    """
    for key in ("z02_mm", "energy_kev", "px_mm"):
        if key not in setup:
            raise KeyError(f"setup is missing {key!r} (needed to convert {target_mode} to physical units)")
    pred = _as_1d(pred_target, "pred_target")
    true = _as_1d(true_target, "true_target")
    z02 = np.broadcast_to(np.asarray(setup["z02_mm"], dtype=np.float64), pred.shape)
    energy, px = setup["energy_kev"], setup["px_mm"]

    z01_pred = target_to_z01_mm(pred, target_mode, z02, energy, px)
    z01_true = target_to_z01_mm(true, target_mode, z02, energy, px)
    fr_true = target_to_fr(true, target_mode, z02, energy, px)
    if target_mode == "z01_mm":
        z01_clipped = np.clip(z01_pred, 1e-3, z02 - 1e-3)
        fr_pred = np.asarray(fresnel_number(z01_clipped, z02, energy, px), dtype=np.float64)
    else:
        fr_pred = target_to_fr(pred, target_mode, z02, energy, px)
    return {"z01_pred_mm": z01_pred, "z01_true_mm": z01_true, "fr_pred": fr_pred, "fr_true": fr_true}


def metrics_in_physical_units(
    pred_target: ArrayLike,
    true_target: ArrayLike,
    target_mode: str,
    setup: dict[str, Any],
) -> dict[str, float]:
    """Errors of z01 in mm and relative Fresnel-number errors in % regardless of the regression target.

    Returns ``z01_mae_mm``, ``z01_rmse_mm``, ``z01_p95_abs_err_mm``, ``z01_bias_mm``, ``z01_median_abs_err_mm``,
    ``fr_rel_err_mean_pct`` (mean |Fr_pred-Fr_true|/Fr_true), ``fr_rel_err_median_pct``, ``fr_rel_err_p95_pct``,
    ``fr_rel_bias_pct`` (signed mean) and ``n``.
    """
    values = physical_errors(pred_target, true_target, target_mode, setup)
    z01 = regression_metrics(values["z01_pred_mm"], values["z01_true_mm"])
    rel = (values["fr_pred"] - values["fr_true"]) / values["fr_true"] * 100.0
    abs_rel = np.abs(rel)
    return {
        "z01_mae_mm": z01["mae"],
        "z01_rmse_mm": z01["rmse"],
        "z01_p95_abs_err_mm": z01["p95_abs_err"],
        "z01_bias_mm": z01["bias"],
        "z01_median_abs_err_mm": z01["median_abs_err"],
        "fr_rel_err_mean_pct": float(abs_rel.mean()),
        "fr_rel_err_median_pct": float(np.median(abs_rel)),
        "fr_rel_err_p95_pct": float(np.percentile(abs_rel, 95)),
        "fr_rel_bias_pct": float(rel.mean()),
        "n": z01["n"],
    }
