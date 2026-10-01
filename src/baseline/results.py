"""Common per-sample result schema and summary statistics for all autofocus methods.

Every method (CTF ring fit, classical sharpness metric, HoloWizard model-based autofocus, learned
regressors) writes one row per hologram with at least :data:`RESULT_COLUMNS`, so that the methods can be
compared with the same code (:func:`summarize`, :func:`plot_scatter`, :func:`summary_markdown_table`) and
fed into the downstream reconstruction test (``python -m src.eval.downstream --candidates-csv ...``).

Columns (units):

``index``          sample index inside the HDF5 file
``source``         file name of the HDF5 file
``fr_true``        true pixel Fresnel number (label ``metadata/setup/Fr``)
``z01_true_mm``    true focus-object distance in mm
``fr_est``         estimated Fresnel number
``z01_est_mm``     estimated distance in mm (same z02/energy/pixel size as the label)
``rel_err_fr_pct`` signed relative error ``(fr_est / fr_true - 1) * 100`` in %
``dz01_mm``        signed distance error ``z01_est - z01_true`` in mm
``blur_px``        defocus blur ``sqrt(|e| / fr_true)`` in detector pixels (:func:`src.utils.fresnel.defocus_blur_px`)
``runtime_s``      wall-clock time of the estimate in seconds
``n_evals``        number of objective evaluations (reconstructions, metric evaluations, forward passes)
``method``         method identifier, e.g. ``holowizard_find_focus``, ``classical_tv``, ``ringfit_cos``, ``ml_cnn``

Additional, method-specific columns may follow the common ones.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.fresnel import blur_px_from_fresnel_numbers

__all__ = [
    "RESULT_COLUMNS",
    "make_result_row",
    "summarize",
    "summarize_by_method",
    "write_samples_csv",
    "read_samples_csv",
    "plot_scatter",
    "summary_markdown_table",
]

RESULT_COLUMNS: tuple[str, ...] = (
    "index",
    "source",
    "fr_true",
    "z01_true_mm",
    "fr_est",
    "z01_est_mm",
    "rel_err_fr_pct",
    "dz01_mm",
    "blur_px",
    "runtime_s",
    "n_evals",
    "method",
)

_WITHIN_PCT: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)


def make_result_row(
    index: int,
    source: str,
    fr_true: float,
    z01_true_mm: float,
    fr_est: float,
    z01_est_mm: float,
    runtime_s: float,
    n_evals: int,
    method: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build one result row; derived columns (relative error, dz01, blur) are computed here.

    ``extra`` key/value pairs are appended after the common columns (method-specific diagnostics).
    """
    fr_true = float(fr_true)
    fr_est = float(fr_est)
    row: dict[str, Any] = {
        "index": int(index),
        "source": str(source),
        "fr_true": fr_true,
        "z01_true_mm": float(z01_true_mm),
        "fr_est": fr_est,
        "z01_est_mm": float(z01_est_mm),
        "rel_err_fr_pct": (fr_est / fr_true - 1.0) * 100.0 if np.isfinite(fr_est) else float("nan"),
        "dz01_mm": float(z01_est_mm) - float(z01_true_mm),
        "blur_px": float(blur_px_from_fresnel_numbers(fr_est, fr_true)) if np.isfinite(fr_est) else float("nan"),
        "runtime_s": float(runtime_s),
        "n_evals": int(n_evals),
        "method": str(method),
    }
    for key, value in extra.items():
        if key in row:
            raise ValueError(f"extra column {key!r} clashes with a common column")
        row[key] = value
    return row


def _column(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=np.float64)


def summarize(rows: Sequence[dict[str, Any]], method: str | None = None) -> dict[str, Any]:
    """Aggregate statistics of the rows of one method (``method=None``: all rows).

    Relative Fresnel-number errors are heavy-tailed (gross failures of some estimators), therefore the
    median and the 95th percentile are reported next to the mean absolute error.  ``fraction_within_<p>pct``
    is the share of samples with ``|rel_err_fr_pct| < p``; the blur statistics use ``b = sqrt(|e|/Fr)``.
    """
    selected = [row for row in rows if method is None or row["method"] == method]
    if not selected:
        raise ValueError(f"no rows for method {method!r}")
    rel = _column(selected, "rel_err_fr_pct")
    finite = np.isfinite(rel)
    rel_ok = rel[finite]
    abs_rel = np.abs(rel_ok)
    dz = _column(selected, "dz01_mm")[finite]
    blur = _column(selected, "blur_px")[finite]
    runtime = _column(selected, "runtime_s")
    n_evals = _column(selected, "n_evals")
    per_eval = runtime[n_evals > 0] / n_evals[n_evals > 0]
    methods = sorted({row["method"] for row in selected})
    summary: dict[str, Any] = {
        "method": method if method is not None else (methods[0] if len(methods) == 1 else methods),
        "n": len(selected),
        "n_failed": int((~finite).sum()),
        "mae_rel_fr_pct": float(abs_rel.mean()) if abs_rel.size else float("nan"),
        "median_rel_fr_pct": float(np.median(abs_rel)) if abs_rel.size else float("nan"),
        "p95_rel_fr_pct": float(np.percentile(abs_rel, 95)) if abs_rel.size else float("nan"),
        "max_rel_fr_pct": float(abs_rel.max()) if abs_rel.size else float("nan"),
        "bias_rel_fr_pct": float(rel_ok.mean()) if rel_ok.size else float("nan"),
        "mae_z01_mm": float(np.abs(dz).mean()) if dz.size else float("nan"),
        "median_z01_mm": float(np.median(np.abs(dz))) if dz.size else float("nan"),
        "p95_z01_mm": float(np.percentile(np.abs(dz), 95)) if dz.size else float("nan"),
        "bias_z01_mm": float(dz.mean()) if dz.size else float("nan"),
        "blur_px_mean": float(blur.mean()) if blur.size else float("nan"),
        "blur_px_median": float(np.median(blur)) if blur.size else float("nan"),
        "blur_px_p95": float(np.percentile(blur, 95)) if blur.size else float("nan"),
        "mean_runtime_s": float(runtime.mean()),
        "median_runtime_s": float(np.median(runtime)),
        "mean_n_evals": float(n_evals.mean()),
        "mean_runtime_per_eval_s": float(per_eval.mean()) if per_eval.size else float("nan"),
    }
    for threshold in _WITHIN_PCT:
        summary[f"fraction_within_{threshold:g}pct"] = float((abs_rel < threshold).mean()) if abs_rel.size else float("nan")
    return summary


def summarize_by_method(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """:func:`summarize` for every method present in ``rows`` (sorted by method name)."""
    return [summarize(rows, method) for method in sorted({row["method"] for row in rows})]


def write_samples_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write rows to CSV (common columns first, extra columns in order of appearance)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    missing = [key for key in RESULT_COLUMNS if key not in rows[0]]
    if missing:
        raise ValueError(f"rows lack the common columns {missing}")
    fieldnames = list(RESULT_COLUMNS)
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return path


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _parse(value: str) -> Any:
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer() and "." not in value and "e" not in value.lower():
        return int(number)
    return number


def read_samples_csv(path: str | Path, methods: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Read a ``samples.csv`` written by any baseline/evaluation CLI (numbers are parsed, optional method filter).

    Files from :mod:`src.baseline.ctf_ringfit` versions without the common columns are upgraded on the fly
    (``fr_rel_err_pct`` -> ``rel_err_fr_pct``, ``z01_err_mm`` -> ``dz01_mm``, ``elapsed_ms`` -> ``runtime_s``).
    """
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [{key: _parse(value) for key, value in raw.items()} for raw in csv.DictReader(handle)]
    wanted = set(methods) if methods is not None else None
    upgraded: list[dict[str, Any]] = []
    for row in rows:
        row.setdefault("source", path.parent.name)
        row.setdefault("method", f"csv:{path.parent.name}")
        if "rel_err_fr_pct" not in row and "fr_rel_err_pct" in row:
            row["rel_err_fr_pct"] = row["fr_rel_err_pct"]
        if "dz01_mm" not in row and "z01_err_mm" in row:
            row["dz01_mm"] = row["z01_err_mm"]
        if "runtime_s" not in row:
            row["runtime_s"] = float(row["elapsed_ms"]) / 1e3 if row.get("elapsed_ms") is not None else float("nan")
        row.setdefault("n_evals", 0)
        if "blur_px" not in row and row.get("fr_est") is not None and row.get("fr_true"):
            row["blur_px"] = float(blur_px_from_fresnel_numbers(float(row["fr_est"]), float(row["fr_true"])))
        if wanted is None or row["method"] in wanted:
            upgraded.append(row)
    return upgraded


def plot_scatter(rows: Sequence[dict[str, Any]], path: str | Path, title: str | None = None) -> Path:
    """Three panels per method: estimated vs. true Fr (log-log), relative error over Fr, runtime over Fr."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({row["method"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fr_all = _column(rows, "fr_true")
    for method in methods:
        sel = [row for row in rows if row["method"] == method]
        fr_true = _column(sel, "fr_true")
        fr_est = _column(sel, "fr_est")
        axes[0].loglog(fr_true, fr_est, "o", alpha=0.7, ms=5, label=method)
        axes[1].semilogx(fr_true, _column(sel, "rel_err_fr_pct"), "o", alpha=0.7, ms=5, label=method)
        axes[2].loglog(fr_true, _column(sel, "runtime_s"), "o", alpha=0.7, ms=5, label=method)
    lim = [fr_all.min() * 0.5, fr_all.max() * 2.0]
    axes[0].plot(lim, lim, "k--", lw=0.8)
    axes[0].set_xlabel("true Fr")
    axes[0].set_ylabel("estimated Fr")
    axes[1].axhline(0.0, color="gray", lw=0.8)
    axes[1].set_xlabel("true Fr")
    axes[1].set_ylabel("relative Fr error [%]")
    axes[2].set_xlabel("true Fr")
    axes[2].set_ylabel("runtime per hologram [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def summary_markdown_table(summaries: Sequence[dict[str, Any]], labels: Sequence[str] | None = None) -> str:
    """Markdown table of :func:`summarize` dictionaries (one row per method/data set)."""
    header = (
        "| Method | n | MAE rel. Fr [%] | Median [%] | p95 [%] | Bias [%] | MAE z01 [mm] | Median z01 [mm] | "
        "blur b median / p95 [px] | within 2 % / 5 % | Runtime [s/hologram] | Evaluations |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for i, s in enumerate(summaries):
        label = labels[i] if labels is not None else str(s["method"])
        lines.append(
            f"| {label} | {s['n']} | {s['mae_rel_fr_pct']:.2f} | {s['median_rel_fr_pct']:.2f} | {s['p95_rel_fr_pct']:.2f} | "
            f"{s['bias_rel_fr_pct']:+.2f} | {s['mae_z01_mm']:.2f} | {s['median_z01_mm']:.2f} | "
            f"{s['blur_px_median']:.2f} / {s['blur_px_p95']:.2f} | "
            f"{100 * s['fraction_within_2pct']:.0f} % / {100 * s['fraction_within_5pct']:.0f} % | "
            f"{s['median_runtime_s']:.3g} | {s['mean_n_evals']:.1f} |"
        )
    return header + "\n".join(lines) + "\n"
