"""Per-sample baseline results (CSV), probing traces (JSON lines) and the run summary.

``results.csv`` has one row per sample of the split, keyed by ``index``, so that Stage 4 can
evaluate a learned model on exactly the same samples. Distances are in metres, errors in
micrometres, times in seconds. ``error_um = (z01_est - z01_true) * 1e6``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

RESULT_FIELDS = (
    "index",
    "phantom_type",
    "photons",
    "z01_true",
    "z01_guess",
    "dz01_true",
    "z01_est",
    "error_um",
    "abs_error_um",
    "n_reconstructions",
    "n_evaluations",
    "nm_iterations",
    "converged",
    "seconds",
    "seconds_per_reconstruction",
    "mfe_est",
    "mfe_true",
)
INT_FIELDS = {"index", "phantom_type", "n_reconstructions", "n_evaluations", "nm_iterations"}
BOOL_FIELDS = {"converged"}


def append_row(path: str | Path, row: dict) -> None:
    """Append one result row, writing the header for a new file."""
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: _format(row.get(k)) for k in RESULT_FIELDS})


def _format(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def read_rows(path: str | Path) -> list[dict]:
    """Rows of ``results.csv`` with numbers parsed (empty cells become ``None``)."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for key, text in raw.items():
                if text == "" or text is None:
                    row[key] = None
                elif key in INT_FIELDS:
                    row[key] = int(text)
                elif key in BOOL_FIELDS:
                    row[key] = text == "true"
                else:
                    row[key] = float(text)
            rows.append(row)
    return rows


def append_trace(path: str | Path, record: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(record, allow_nan=True) + "\n")


def read_traces(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarize(rows: list[dict], thresholds_um: tuple[float, ...] = (50, 100, 250, 1000)) -> dict:
    """Accuracy, cost and the initial error of the guess over all rows."""
    if not rows:
        return {"n_samples": 0}
    err = np.array([r["error_um"] for r in rows], dtype=float)
    guess = np.array([r["dz01_true"] * 1e6 for r in rows], dtype=float)
    recon = np.array([r["n_reconstructions"] for r in rows], dtype=float)
    seconds = np.array([r["seconds"] for r in rows], dtype=float)
    per_recon = np.array([r["seconds_per_reconstruction"] for r in rows], dtype=float)
    abs_err = np.abs(err)
    out = {
        "n_samples": len(rows),
        "error_um": {
            "mean": float(err.mean()),
            "std": float(err.std(ddof=1)) if len(err) > 1 else 0.0,
            "mae": float(abs_err.mean()),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "median_abs": float(np.median(abs_err)),
            "p90_abs": float(np.percentile(abs_err, 90)),
            "max_abs": float(abs_err.max()),
        },
        "fraction_within_um": {str(int(t)): float(np.mean(abs_err <= t)) for t in thresholds_um},
        "guess_error_um": {
            "mae": float(np.abs(guess).mean()),
            "rmse": float(np.sqrt(np.mean(guess**2))),
        },
        "reconstructions": {
            "mean": float(recon.mean()),
            "min": int(recon.min()),
            "max": int(recon.max()),
        },
        "seconds_per_autofocus": {
            "mean": float(seconds.mean()),
            "median": float(np.median(seconds)),
            "total": float(seconds.sum()),
        },
        "seconds_per_reconstruction": float(np.mean(per_recon)),
        "converged": float(np.mean([bool(r["converged"]) for r in rows])),
    }
    truth = [r for r in rows if r.get("mfe_true") is not None]
    if truth:
        below = [r["mfe_est"] <= r["mfe_true"] for r in truth if not math.isnan(r["mfe_true"])]
        out["mfe_est_le_mfe_true"] = float(np.mean(below)) if below else math.nan
    return out
