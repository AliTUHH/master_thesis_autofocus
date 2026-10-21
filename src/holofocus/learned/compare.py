"""Comparison of the learned autofocus with the classical baseline on the same samples.

Methods, matched by sample ``index``:

* ``baseline_raw``: the estimates of [2] as they are (``holofocus-baseline`` results);
* ``baseline_corrected``: the same minus the mean error (offset) of the baseline on a
  validation subset. All corrected errors share the offset, so its standard error and
  bootstrap interval are reported with it;
* ``learned``: the network estimates (``holofocus-learn predict``);
* ``warm_start`` (optional): the baseline started from the learned estimates, and
  ``warm_start_corrected``, the same minus the offset (the warm start searches the same MFE
  minimum as the cold start).

``comparison.csv`` holds one row per sample (errors in micrometres), ``comparison.json`` the
metrics per method, the offset and paired differences of the absolute errors.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .metrics import bootstrap_ci, interval_metrics, point_metrics

COMPARISON_FIELDS = (
    "index",
    "phantom_type",
    "photons",
    "dz01_true_um",
    "learned_error_um",
    "learned_sigma_um",
    "learned_seconds",
    "raw_error_um",
    "corrected_error_um",
    "raw_reconstructions",
    "raw_seconds",
    "warm_error_um",
    "warm_corrected_error_um",
    "warm_reconstructions",
    "warm_seconds",
)
_INT_FIELDS = ("index", "phantom_type", "raw_reconstructions", "warm_reconstructions")


def offset_estimate(errors_um, level: float = 0.95, resamples: int = 10000, seed: int = 0) -> dict:
    """Mean error of the baseline with its standard error and bootstrap intervals."""
    err = np.asarray(errors_um, dtype=float)
    if err.size < 2:
        raise ValueError("the offset needs at least two validation samples")
    std = float(err.std(ddof=1))
    return {
        "n": int(err.size),
        "mean": float(err.mean()),
        "std": std,
        "se": std / math.sqrt(err.size),
        "ci": list(bootstrap_ci(err, np.mean, level, resamples, seed)),
        "median": float(np.median(err)),
        "median_ci": list(bootstrap_ci(err, np.median, level, resamples, seed)),
        "level": level,
    }


def _paired(abs_a: np.ndarray, abs_b: np.ndarray, seed: int = 0) -> dict:
    pairs = np.column_stack([abs_a, abs_b])
    return {
        "mae_difference_um": float(abs_a.mean() - abs_b.mean()),
        "mae_difference_ci": list(
            bootstrap_ci(pairs, lambda d: d[:, 0].mean() - d[:, 1].mean(), seed=seed)
        ),
        "fraction_first_better": float(np.mean(abs_a < abs_b)),
    }


def _cost(rows: list[dict], prefix: str) -> dict:
    seconds = np.array([r[f"{prefix}_seconds"] for r in rows], dtype=float)
    out = {"seconds_mean": float(seconds.mean()), "seconds_median": float(np.median(seconds))}
    if f"{prefix}_reconstructions" in rows[0]:
        recon = np.array([r[f"{prefix}_reconstructions"] for r in rows], dtype=float)
        out["reconstructions"] = {
            "mean": float(recon.mean()),
            "min": int(recon.min()),
            "max": int(recon.max()),
        }
    return out


def compare_methods(
    predictions: list[dict],
    baseline: list[dict],
    offset: dict,
    warm: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Per-sample rows and summary on the samples of ``baseline`` (results rows)."""
    learned = {r["index"]: r for r in predictions}
    warm_rows = None if warm is None else {r["index"]: r for r in warm}
    rows = []
    for b in sorted(baseline, key=lambda r: r["index"]):
        i = b["index"]
        if i not in learned or (warm_rows is not None and i not in warm_rows):
            raise ValueError(f"sample {i} of the baseline has no learned or warm-start estimate")
        p = learned[i]
        if abs(p["z01_true"] - b["z01_true"]) > 1e-12:
            raise ValueError(f"sample {i}: z01_true differs, the runs used different splits")
        row = {
            "index": i,
            "phantom_type": b["phantom_type"],
            "photons": b["photons"],
            "dz01_true_um": b["dz01_true"] * 1e6,
            "learned_error_um": p["error_um"],
            "learned_sigma_um": p["sigma"] * 1e6,
            "learned_seconds": p["seconds"],
            "raw_error_um": b["error_um"],
            "corrected_error_um": b["error_um"] - offset["mean"],
            "raw_reconstructions": b["n_reconstructions"],
            "raw_seconds": b["seconds"],
            "warm_error_um": None,
            "warm_corrected_error_um": None,
            "warm_reconstructions": None,
            "warm_seconds": None,
        }
        if warm_rows is not None:
            w = warm_rows[i]
            if abs(w["z01_true"] - b["z01_true"]) > 1e-12:
                raise ValueError(f"sample {i}: z01_true of the warm start differs")
            row.update(
                warm_error_um=w["error_um"],
                warm_corrected_error_um=w["error_um"] - offset["mean"],
                warm_reconstructions=w["n_reconstructions"],
                warm_seconds=w["seconds"],
            )
        rows.append(row)
    if not rows:
        raise ValueError("no baseline samples to compare")

    def errors(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=float)

    learned_err, raw_err, corr_err = (
        errors(k) for k in ("learned_error_um", "raw_error_um", "corrected_error_um")
    )
    methods = {
        "learned": {
            "error_um": point_metrics(learned_err),
            "interval": interval_metrics(learned_err, errors("learned_sigma_um")),
            **_cost(rows, "learned"),
        },
        "baseline_raw": {"error_um": point_metrics(raw_err), **_cost(rows, "raw")},
        "baseline_corrected": {"error_um": point_metrics(corr_err)},
    }
    paired = {
        "learned_vs_raw": _paired(np.abs(learned_err), np.abs(raw_err)),
        "learned_vs_corrected": _paired(np.abs(learned_err), np.abs(corr_err)),
    }
    if warm_rows is not None:
        warm_err = errors("warm_error_um")
        warm_corr = errors("warm_corrected_error_um")
        methods["warm_start"] = {"error_um": point_metrics(warm_err), **_cost(rows, "warm")}
        methods["warm_start_corrected"] = {"error_um": point_metrics(warm_corr)}
        paired["warm_vs_raw"] = _paired(np.abs(warm_err), np.abs(raw_err))
        paired["warm_vs_learned"] = _paired(np.abs(warm_err), np.abs(learned_err))
        paired["warm_corrected_vs_corrected"] = _paired(np.abs(warm_corr), np.abs(corr_err))
    summary = {
        "n_samples": len(rows),
        "indices": [r["index"] for r in rows],
        "offset_um": offset,
        "methods": methods,
        "paired": paired,
        "speedup_median": float(
            methods["baseline_raw"]["seconds_median"] / methods["learned"]["seconds_median"]
        ),
    }
    return rows, summary


def write_comparison(out: str | Path, rows: list[dict], summary: dict) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: "" if v is None else repr(float(v)) if isinstance(v, float) else v
                    for k, v in row.items()
                }
            )
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")


def read_comparison(path: str | Path) -> list[dict]:
    """Rows of ``comparison.csv`` (the file or its directory); empty cells become ``None``."""
    path = Path(path)
    if path.is_dir():
        path = path / "comparison.csv"
    with open(path, newline="") as fh:
        return [
            {
                k: None if v == "" else int(v) if k in _INT_FIELDS else float(v)
                for k, v in raw.items()
            }
            for raw in csv.DictReader(fh)
        ]
