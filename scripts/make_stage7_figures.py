"""Stage 7 figures for the lab notebook (``docs/laborbuch/figures``).

* ``stage7-training.png``: validation curves per epoch on cpu-4x of the NPEs (main run, the four
  further seeds, mixture density) and the Gaussian CNN of Stage 4: MAE of the point estimate,
  NLL of the true ``dz01`` (nats, ``dz01`` in um) and coverage of the central 68 % interval
  (CNN with its raw sigma, before the scale fit); below, the validation NLL of the matched NPEs
  against the matched CNNs of Stage 5. Open circles: selected epochs.
* ``stage7-calibration.png``: on the calibration split, observed minus nominal coverage of the
  central intervals and of the HPD regions (expected coverage) over all levels, with the 95 %
  band of a calibrated method, and the PIT (SBC rank) histograms, for the main NPE also with
  sbi's ranks from posterior samples.
* ``stage7-comparison.png``: CDFs of the absolute error on the test split and on the 48 baseline
  samples of Stage 3 (raw and offset-corrected), 68 % widths of NPE and CNN per test sample, and
  the time per estimate with the parts of one NPE estimate.
* ``stage7-posteriors.png``: posterior densities of the NPEs and the CNN Gaussian for the example
  test samples (chosen before the results, plus the largest errors of the main NPE).
* ``stage7-robustness.png``: per measured sample and Stage 5 perturbation of its matched test
  split, the coverage, the widening of the 68 % intervals against the growth of the errors, and
  the MAE, matched NPE against matched CNN.
* ``stage7-measured.png``: the matched NPE posteriors of the four measured holograms with [2],
  our padding-4 baseline, the CNN of Stage 5 and the estimates of the 16 binning phases.
* ``stage7-metrics.json``: the numbers quoted in the lab notebook.

Usage::

    python scripts/make_stage7_figures.py                                   # results/
    python scripts/make_stage7_figures.py --quick --npe DIR --cnn DIR --learned DIR \\
        --measured DIR --out /tmp/figures                                   # smoke test (CI)

Figures whose inputs are missing are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from holofocus.learned import read_history, read_predictions  # noqa: E402
from holofocus.learned.metrics import interval_quantile  # noqa: E402
from holofocus.learned.posterior import (  # noqa: E402
    gaussian_hpd_level,
    gaussian_pit,
    ks_uniform,
)
from holofocus.measured import MEASURED_SAMPLES  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "laborbuch" / "figures"
RESULTS = REPO / "results"
MAIN = "npe-nsf"
ENSEMBLE = f"{MAIN}-ensemble"
SEEDS = tuple(f"{MAIN}-seed{k}" for k in range(1, 5))
VARIANTS = ("npe-mdn",)
SAMPLES = list(MEASURED_SAMPLES)
METHODS = {
    "cnn": ("Gaussian CNN (Stage 4)", "tab:blue", "-"),
    MAIN: ("NPE, spline flow", "tab:red", "-"),
    ENSEMBLE: ("NPE, 5 seeds", "tab:purple", "--"),
    "npe-mdn": ("NPE, mixture density", "tab:orange", ":"),
    "baseline_raw": ("NM baseline (Stage 3)", "0.55", "-."),
    "baseline_corrected": ("NM baseline, offset-corrected", "black", "-."),
}
MEASURED_REFERENCES = {
    "paper2_nm": ("[2] Nelder-Mead", "black", "-"),
    "paper2_visual": ("[2] visual", "black", ":"),
    "baseline_b1_p4": ("NM (ours), padding 4", "0.45", "-."),
}
LEVELS = np.linspace(0.01, 0.99, 99)
PIT_BINS = 20


def _save(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plain_log_y(ax) -> None:
    """Major ticks of a log y axis at 1, 2 and 5 (or more) times powers of ten inside its
    limits, as plain numbers; on axes spanning about a decade, matplotlib labels the minor
    ticks as 2×10^2 and the like."""
    lo, hi = ax.get_ylim()
    decades = range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
    for steps in ((1, 2, 5), (1, 2, 3, 5), (1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9)):
        ticks = [m * 10.0**d for d in decades for m in steps if lo <= m * 10.0**d <= hi]
        if len(ticks) >= 3:
            break
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        return [{k: _number(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def _number(text: str):
    try:
        return float(text)
    except ValueError:
        return text


def _get(record, *keys):
    for key in keys:
        if not isinstance(record, dict) or key not in record:
            return None
        record = record[key]
    return record


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def _legend(fig, axes, columns: int = 4) -> None:
    """One legend below the panels with the labels of all ``axes``."""
    handles = {}
    for a in np.atleast_1d(axes).flat:
        for h, label in zip(*a.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, h)
    if handles:
        fig.legend(
            handles.values(),
            handles.keys(),
            loc="outside lower center",
            ncols=max(1, min(columns, len(handles))),
            fontsize=8,
        )


# --- training ------------------------------------------------------------------------------


def _nll_um(run: Path, history: list[dict]) -> np.ndarray:
    """Validation NLL of ``dz01`` in um: the CNN logs it, the NPE logs ``-log q`` of
    ``t = dz01 / target_scale``."""
    if "val_nll" in history[0]:
        return np.array([h["val_nll"] for h in history])
    scale = yaml.safe_load((run / "config.yaml").read_text())["target_scale"]
    return np.array([h["val_loss"] for h in history]) + math.log(scale * 1e6)


def _curves(axes, run: Path, keys, color: str, style: str, label, lw=1.2, alpha=1.0):
    """Plot the validation curves ``keys`` of ``run`` on ``axes``; summary of the run."""
    if not (run / "history.csv").exists():
        return None
    history = read_history(run / "history.csv")
    epochs = np.array([h["epoch"] for h in history])
    values = {
        "mae": np.array([h["val_mae_um"] for h in history]),
        "nll": _nll_um(run, history),
        "coverage": np.array([h["val_coverage_68"] for h in history]),
    }
    summary = _json(run / "summary.json") or {}
    best = summary.get("best_epoch")
    for ax, key in zip(axes, keys, strict=True):
        ax.plot(epochs, values[key], style, color=color, lw=lw, alpha=alpha, label=label)
        if best in epochs:
            k = int(np.flatnonzero(epochs == best)[0])
            ax.plot(best, values[key][k], "o", color=color, mfc="white", ms=4, alpha=alpha)
    info = yaml.safe_load((run / "run.yaml").read_text()) if (run / "run.yaml").exists() else {}
    k = int(np.flatnonzero(epochs == best)[0]) if best in epochs else int(np.argmin(values["nll"]))
    return {
        "best_epoch": best,
        "epochs_run": summary.get("epochs_run"),
        "validation_mae_um": _get(summary, "validation", "error_um", "mae"),
        "validation_within_100": _get(summary, "validation", "error_um", "within", "100"),
        "validation_coverage": _get(summary, "validation", "interval", "coverage"),
        "validation_nll_um_at_best": float(values["nll"][k]),
        "validation_nll_um_last": float(values["nll"][-1]),
        "train_seconds": summary.get("train_seconds"),
        "load_seconds": info.get("load_seconds"),
        "threads": info.get("threads"),
        "n_train": info.get("n_train"),
        "n_parameters": info.get("n_parameters"),
    }


def training_figure(npe: Path, cnn: Path, learned: Path, out: Path, dpi: int) -> dict | None:
    if not (npe / MAIN / "history.csv").exists():
        return None
    matched = [s for s in SAMPLES if (npe / f"matched-{s}" / "history.csv").exists()]
    fig = plt.figure(figsize=(13.0, 7.6 if matched else 4.2), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 0.95]) if matched else (fig, None)
    axes = top.subplots(1, 3)
    keys = ("mae", "nll", "coverage")
    metrics = {}
    metrics["cnn"] = _curves(axes, cnn, keys, "tab:blue", "-", METHODS["cnn"][0])
    for k, name in enumerate(SEEDS):
        label = "NPE, seeds 1-4" if k == 0 else None
        metrics[name] = _curves(axes, npe / name, keys, "tab:red", "-", label, lw=0.7, alpha=0.35)
    for name in VARIANTS:
        _, color, style = METHODS[name]
        metrics[name] = _curves(axes, npe / name, keys, color, style, METHODS[name][0])
    metrics[MAIN] = _curves(axes, npe / MAIN, keys, "tab:red", "-", "NPE, spline flow (seed 0)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("validation MAE (um)")
    axes[1].set_ylabel("validation NLL (nats, dz01 in um)")
    axes[2].set_ylabel("validation coverage, 68 % interval")
    axes[2].axhline(0.68, color="0.4", lw=0.8, ls=":")
    nll = [m["validation_nll_um_at_best"] for m in metrics.values() if m]
    axes[1].set_ylim(min(nll) - 0.3, min(nll) + 2.5)
    _plain_log_y(axes[0])
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    info = yaml.safe_load((npe / MAIN / "run.yaml").read_text())
    top.suptitle(f"{info['n_train']} training, {info['n_val']} validation samples", fontsize=10)
    if matched:
        maxes = bottom.subplots(1, len(matched), sharey=False, squeeze=False)[0]
        for ax, sample in zip(maxes, matched, strict=True):
            metrics[f"matched-{sample}"] = _curves(
                [ax], npe / f"matched-{sample}", ("nll",), "tab:red", "-", "NPE"
            )
            metrics[f"cnn-matched-{sample}"] = _curves(
                [ax], learned / f"matched-{sample}", ("nll",), "tab:blue", "-", "CNN (Stage 5)"
            )
            best = [
                metrics[n]["validation_nll_um_at_best"]
                for n in (f"matched-{sample}", f"cnn-matched-{sample}")
                if metrics.get(n)
            ]
            ax.set_ylim(min(best) - 0.3, min(best) + 3.0)
            ax.set_title(MEASURED_SAMPLES[sample].label, fontsize=9)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        maxes[0].set_ylabel("validation NLL (nats, dz01 in um)")
        n_train = {m["n_train"] for k, m in metrics.items() if k.startswith("matched-") and m}
        sizes = ", ".join(str(n) for n in sorted(n_train, key=str))
        bottom.suptitle(f"matched datasets of Stage 5: {sizes} training samples", fontsize=10)
    _save(fig, out / "stage7-training.png", dpi)
    return {k: v for k, v in metrics.items() if v}


# --- calibration ---------------------------------------------------------------------------


def method_tables(npe: Path, cnn: Path, split: str) -> dict[str, dict[str, np.ndarray]]:
    """Per-sample errors, 68 % widths, PIT and HPD levels of the CNN and the NPEs on ``split``."""
    tables = {}
    path = cnn / split / "predictions.csv"
    if path.exists():
        rows = read_predictions(path)
        err = np.array([r["error_um"] for r in rows])
        sigma = np.array([r["sigma"] for r in rows]) * 1e6
        tables["cnn"] = {
            "error_um": err,
            "width_68_um": 2 * interval_quantile(0.68) * sigma,
            "pit": gaussian_pit(err, sigma),
            "hpd_level": gaussian_hpd_level(err, sigma),
        }
    for name in (MAIN, ENSEMBLE, *VARIANTS):
        path = npe / name / split / "predictions.csv"
        if path.exists():
            rows = read_predictions(path)
            tables[name] = {
                key: np.array([r[key] for r in rows])
                for key in ("error_um", "width_68_um", "pit", "hpd_level")
            }
    return tables


def _coverage_curves(pit: np.ndarray, hpd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage of the central intervals (from the PIT) and of the HPD regions per level."""
    central = np.array([np.mean(np.abs(pit - 0.5) <= a / 2) for a in LEVELS])
    regions = np.array([np.mean(hpd <= a) for a in LEVELS])
    return central, regions


def calibration_figure(npe: Path, cnn: Path, out: Path, dpi: int) -> dict | None:
    tables = method_tables(npe, cnn, "calibration")
    if MAIN not in tables or "cnn" not in tables:
        return None
    n = len(tables[MAIN]["pit"])
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), layout="constrained")
    band = 1.96 * np.sqrt(LEVELS * (1 - LEVELS) / n)
    for ax in axes[:2]:
        ax.fill_between(LEVELS, -band, band, color="0.85", label=f"95 % band, n = {n}")
        ax.axhline(0, color="0.3", lw=0.8)
    metrics = {}
    for name, t in tables.items():
        label, color, style = METHODS[name]
        central, regions = _coverage_curves(t["pit"], t["hpd_level"])
        axes[0].plot(LEVELS, central - LEVELS, style, color=color, label=label)
        axes[1].plot(LEVELS, regions - LEVELS, style, color=color, label=label)
        counts = np.histogram(np.clip(t["pit"], 0, 1), bins=PIT_BINS, range=(0, 1))[0]
        edges = np.linspace(0, 1, PIT_BINS + 1)
        axes[2].stairs(
            counts / (len(t["pit"]) / PIT_BINS),
            edges,
            baseline=None,
            color=color,
            ls=style,
            lw=1.2,
        )
        metrics[name] = {
            "n": len(t["pit"]),
            "central_coverage": {
                "68": float(np.mean(np.abs(t["pit"] - 0.5) <= 0.34)),
                "95": float(np.mean(np.abs(t["pit"] - 0.5) <= 0.475)),
            },
            "hpd_coverage": {
                "68": float(np.mean(t["hpd_level"] <= 0.68)),
                "95": float(np.mean(t["hpd_level"] <= 0.95)),
            },
            "central_max_abs_deviation": float(np.abs(central - LEVELS).max()),
            "hpd_max_abs_deviation": float(np.abs(regions - LEVELS).max()),
            "pit_ks": ks_uniform(t["pit"]),
            "pit_below_0.025": float(np.mean(t["pit"] < 0.025)),
            "pit_above_0.975": float(np.mean(t["pit"] > 0.975)),
            "width_68_median_um": float(np.median(t["width_68_um"])),
        }
    for name in (MAIN, *VARIANTS):
        checks = _json(npe / name / "calibration" / "sbi-checks.json")
        if checks is None:
            continue
        metrics.setdefault(name, {})["sbi"] = {
            "num_posterior_samples": checks["num_posterior_samples"],
            "sbc": checks["sbc"],
            "tarp": checks["tarp"],
            "coverage": checks["calibration"]["hpd"]["coverage"],
            "rank_ks_p_value": checks["calibration"]["pit"]["ks"]["p_value"],
            "seconds": checks["seconds"],
        }
        ranks = _csv(npe / name / "calibration" / "sbi-ranks.csv")
        if name == MAIN and ranks:
            r = np.array([row["rank"] for row in ranks])
            counts = np.histogram(r, bins=PIT_BINS, range=(0, 1))[0]
            axes[2].stairs(
                counts / (len(r) / PIT_BINS),
                np.linspace(0, 1, PIT_BINS + 1),
                baseline=None,
                color="tab:red",
                ls=(0, (1, 1)),
                lw=1.6,
                label=f"NPE, sbi ranks ({checks['num_posterior_samples']} samples)",
            )
    p = 1 / PIT_BINS
    half = 2.576 * math.sqrt(n * p * (1 - p)) / (n * p)
    axes[2].axhspan(max(0.0, 1 - half), 1 + half, color="0.85", zorder=0, label="99 % band per bin")
    axes[2].axhline(1, color="0.3", lw=0.8, zorder=0.5)
    axes[0].set_title("central intervals", fontsize=9)
    axes[1].set_title("HPD regions (expected coverage)", fontsize=9)
    axes[2].set_title("PIT / SBC rank histogram", fontsize=9)
    for ax in axes[:2]:
        ax.set_xlabel("nominal level")
        ax.set_ylabel("observed minus nominal coverage")
        ax.grid(alpha=0.3)
    axes[2].set_xlabel("PIT = posterior CDF at the true dz01")
    axes[2].set_ylabel("count / expected count")
    axes[2].grid(alpha=0.3)
    _legend(fig, axes, columns=4)
    fig.suptitle(f"calibration split of cpu-4x ({n} prior draws)", fontsize=10)
    _save(fig, out / "stage7-calibration.png", dpi)
    return metrics


# --- comparison ----------------------------------------------------------------------------


def _cdf(ax, values, **kwargs) -> None:
    v = np.sort(np.maximum(np.abs(values), 1.0))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", **kwargs)


def _error_panel(ax, rows: list[dict], title: str) -> dict:
    metrics = {}
    for name, (label, color, style) in METHODS.items():
        key = f"{name}_error_um"
        if key not in rows[0]:
            continue
        err = np.array([r[key] for r in rows])
        _cdf(ax, err, color=color, ls=style, lw=1.3, label=label)
        metrics[name] = {
            "mae_um": float(np.mean(np.abs(err))),
            "median_abs_um": float(np.median(np.abs(err))),
            "within_100": float(np.mean(np.abs(err) <= 100)),
        }
    ax.axvline(100, color="0.4", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("|error| (um)")
    ax.set_ylabel("fraction of samples")
    ax.set_title(f"{title} (n = {len(rows)})", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    return metrics


def comparison_figure(comparison: Path, timing: Path, out: Path, dpi: int) -> dict | None:
    test = _csv(comparison / "test.csv")
    if test is None:
        return None
    baseline = sorted(comparison.glob("baseline-*.csv"))
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.4), layout="constrained")
    metrics = {"test": _error_panel(axes[0], test, "test split")}
    if baseline:
        rows = _csv(baseline[0])
        metrics[baseline[0].stem] = _error_panel(axes[1], rows, "baseline samples of Stage 3")
    else:
        axes[1].set_axis_off()
    ax = axes[2]
    w_cnn = np.array([r["cnn_width_68_um"] for r in test])
    w_npe = np.array([r[f"{MAIN}_width_68_um"] for r in test])
    covered = np.array([r[f"{MAIN}_covered_68"] for r in test]) > 0
    ax.loglog(w_cnn[covered], w_npe[covered], ".", color="tab:red", ms=4, label="NPE covers")
    ax.loglog(w_cnn[~covered], w_npe[~covered], "x", color="tab:red", ms=4, label="NPE misses")
    lim = [min(w_cnn.min(), w_npe.min()) * 0.8, max(w_cnn.max(), w_npe.max()) * 1.25]
    ax.plot(lim, lim, color="0.3", lw=0.8)
    ratio = float(np.exp(np.median(np.log(w_npe / w_cnn))))
    ax.set_xlabel("68 % width of the CNN (um)")
    ax.set_ylabel("68 % width of the NPE (um)")
    ax.set_title(f"test split: median width ratio NPE / CNN {ratio:.2f}", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    metrics["width_68_ratio_median"] = ratio
    record = _json(timing)
    ax = axes[3]
    if record is None:
        ax.set_axis_off()
    else:
        names = [n for n in METHODS if n in record["methods"]]
        y = np.arange(len(names))[::-1] + 1
        for yy, name in zip(y, names, strict=True):
            m = record["methods"][name]
            ms = 1e3 * np.array([m["median"], m["p10"], m["p90"]])
            ax.barh(yy, ms[0], color=METHODS[name][1], alpha=0.8)
            ax.errorbar(ms[0], yy, xerr=[[ms[0] - ms[1]], [ms[2] - ms[0]]], color="0.2", capsize=3)
        left = 0.0
        parts = record["npe_parts_median"]
        for part, color in zip(parts, ("0.75", "tab:red", "salmon", "0.45"), strict=True):
            ax.barh(0, 1e3 * parts[part], left=left, color=color, label=part.replace("_", " "))
            left += 1e3 * parts[part]
        ax.set_yticks([*y, 0], [METHODS[n][0] for n in names] + ["NPE (seed 0), parts"])
        ax.set_xlabel(f"ms per estimate ({record['threads']} thread(s), {record['device']})")
        ax.set_title("time per estimate (median, 10-90 %)", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3, axis="x")
        metrics["timing_ms"] = {n: 1e3 * record["methods"][n]["median"] for n in names}
        metrics["timing_parts_ms"] = {k: 1e3 * v for k, v in parts.items()}
    _legend(fig, axes[:2], columns=6)
    _save(fig, out / "stage7-comparison.png", dpi)
    return metrics


# --- posterior examples --------------------------------------------------------------------


def posteriors_figure(path: Path, out: Path, dpi: int) -> dict | None:
    record = _json(path)
    if record is None:
        return None
    grid = np.array(record["dz01_mm"])
    samples = record["samples"]
    cols = min(4, len(samples))
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.4 * cols + 0.4, 2.7 * rows + 0.7),
        squeeze=False,
        layout="constrained",
    )
    metrics = {}
    for ax, (index, s) in zip(axes.flat, samples.items(), strict=False):
        lo, hi = [s["dz01_true_mm"]], [s["dz01_true_mm"]]
        for name in (MAIN, ENSEMBLE):
            if name not in s:
                continue
            label, color, style = METHODS[name]
            density = np.array(s[name]["density_per_mm"])
            ax.plot(grid, density, style, color=color, lw=1.2, label=label)
            mass = grid[density >= 1e-2 * density.max()]
            lo.append(mass.min())
            hi.append(mass.max())
        mu, sigma = s["cnn"]["dz01_mm"], s["cnn"]["sigma_mm"]
        lo.append(mu - 3 * sigma)
        hi.append(mu + 3 * sigma)
        pad = 0.1 * (max(hi) - min(lo)) + 0.02
        x = np.linspace(max(min(lo) - pad, grid[0]), min(max(hi) + pad, grid[-1]), 600)
        ax.plot(x, _gauss(x, mu, sigma), color="tab:blue", lw=1.2, label=METHODS["cnn"][0])
        ax.axvline(s["dz01_true_mm"], color="black", lw=1, label="true dz01")
        ax.set_xlim(x[0], x[-1])
        kind = "largest error" if int(index) in record["largest_error"] else "a priori"
        ax.set_title(
            f"test {index} ({kind}): type {s['phantom_type']}, {s['photons']:.0f} ph",
            fontsize=8,
        )
        ax.set_xlabel("dz01 (mm)")
        ax.grid(alpha=0.3)
        metrics[index] = {
            "true_mm": s["dz01_true_mm"],
            "npe_median_mm": _get(s, MAIN, "dz01_median_mm"),
            "cnn_mm": mu,
            "cnn_sigma_mm": sigma,
        }
    for ax in axes.flat[len(samples) :]:
        ax.set_axis_off()
    for ax in axes[:, 0]:
        ax.set_ylabel("density (1/mm)")
    _legend(fig, axes, columns=4)
    _save(fig, out / "stage7-posteriors.png", dpi)
    return metrics


# --- robustness ----------------------------------------------------------------------------


def _metric(entry, key: str, *keys):
    value = _get(entry, key, *keys) if entry is not None else None
    return np.nan if value is None else float(value)


def robustness_figure(sim2real: Path, out: Path, dpi: int) -> dict | None:
    records = {s: _json(sim2real / f"{s}.json") for s in SAMPLES}
    records = {s: r for s, r in records.items() if r is not None}
    if not records:
        return None
    fig, axes = plt.subplots(
        3,
        len(records),
        figsize=(3.4 * len(records) + 0.8, 8.6),
        sharex="col",
        squeeze=False,
        layout="constrained",
    )
    metrics = {}
    for col, (sample, rec) in enumerate(records.items()):
        perts = list(rec["perturbations"])
        names = ["none", *perts]
        x = np.arange(len(names))
        metrics[sample] = {}
        ratios = [1.0]
        for method, offset in ((MAIN, -0.1), ("cnn", 0.1)):
            color = METHODS[method][1]
            label = "matched NPE" if method == MAIN else "matched CNN (Stage 5)"
            entries = [rec["reference"][method]] + [rec["perturbations"][p][method] for p in perts]
            cov68 = [_metric(e, "coverage", "68") for e in entries]
            cov95 = [_metric(e, "coverage", "95") for e in entries]
            mae = np.array([_metric(e, "error_um", "mae") for e in entries])
            ratio = [1.0] + [
                _metric(e, "widening", "width_68_ratio", "geometric_mean") for e in entries[1:]
            ]
            ci = [[1.0, 1.0]] + [
                _get(e, "widening", "width_68_ratio", "geometric_mean_ci") or [np.nan, np.nan]
                for e in entries[1:]
            ]
            ci = np.array(ci, dtype=float)
            ratios += [*ci.ravel(), *(mae / mae[0])]
            axes[0, col].plot(x + offset, cov68, "o", color=color, label=f"{label}: coverage 68 %")
            axes[0, col].plot(
                x + offset, cov95, "o", color=color, mfc="white", label=f"{label}: coverage 95 %"
            )
            axes[1, col].errorbar(
                x + offset,
                ratio,
                yerr=[np.array(ratio) - ci[:, 0], ci[:, 1] - np.array(ratio)],
                fmt="o",
                color=color,
                capsize=2,
                label=f"{label}: 68 % width ratio (95 % CI)",
            )
            axes[1, col].plot(
                x + offset,
                mae / mae[0],
                "_",
                color=color,
                ms=12,
                mew=2,
                label=f"{label}: MAE ratio",
            )
            axes[2, col].plot(x + offset, mae, "s", color=color, label=f"{label}: MAE")
            metrics[sample][method] = {
                name: {
                    "coverage_68": cov68[k],
                    "coverage_95": cov95[k],
                    "mae_um": float(mae[k]),
                    "width_68_ratio": float(ratio[k]),
                    "mae_ratio": float(mae[k] / mae[0]),
                }
                for k, name in enumerate(names)
            }
        for level in (0.68, 0.95):
            axes[0, col].axhline(level, color="0.4", lw=0.8, ls=":")
        axes[1, col].axhline(1.0, color="0.4", lw=0.8, ls=":")
        axes[1, col].set_yscale("log")
        finite = np.array(ratios)[np.isfinite(ratios) & (np.array(ratios) > 0)]
        axes[1, col].set_ylim(min(0.5, finite.min() / 1.3), max(2.0, finite.max() * 1.3))
        axes[2, col].set_yscale("log")
        _plain_log_y(axes[1, col])
        _plain_log_y(axes[2, col])
        axes[0, col].set_ylim(0, 1.02)
        axes[0, col].set_title(MEASURED_SAMPLES[sample].label, fontsize=9)
        axes[2, col].set_xticks(x, names, rotation=45, ha="right", fontsize=8)
        for ax in axes[:, col]:
            ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel("coverage")
    axes[1, 0].set_ylabel("perturbed / matched test split")
    axes[2, 0].set_ylabel("MAE (um)")
    _legend(fig, axes, columns={1: 1, 2: 2, 3: 3}.get(len(records), 5))
    _save(fig, out / "stage7-robustness.png", dpi)
    return metrics


# --- measured ------------------------------------------------------------------------------


def measured_figure(npe_measured: Path, measured: Path, out: Path, dpi: int) -> dict | None:
    records = {s: _json(npe_measured / f"{s}.json") for s in SAMPLES}
    records = {s: r for s, r in records.items() if r is not None}
    if not records:
        return None
    fig, axes = plt.subplots(
        1,
        len(records),
        figsize=(3.6 * len(records) + 0.6, 3.9),
        squeeze=False,
        layout="constrained",
    )
    metrics = {}
    for ax, (name, rec) in zip(axes[0], records.items(), strict=True):
        z = np.array(rec["posterior"]["z01_mm"])
        density = np.array(rec["posterior"]["density_per_mm"])
        ax.plot(z, density, color="tab:red", lw=1.3, label="NPE (matched), posterior")
        for key, alpha in (("95", 0.12), ("68", 0.25)):
            i = rec["intervals"][key]
            ax.axvspan(
                i["lower"] * 1e3,
                i["upper"] * 1e3,
                color="tab:red",
                alpha=alpha,
                lw=0,
                label=f"NPE {key} % interval",
            )
        span = [rec["intervals"]["95"]["lower"] * 1e3, rec["intervals"]["95"]["upper"] * 1e3]
        for ref, (label, color, style) in MEASURED_REFERENCES.items():
            if ref in rec["references"]:
                value = rec["references"][ref]["z01"] * 1e3
                ax.axvline(value, color=color, ls=style, lw=1.1, label=label)
                span.append(value)
        cnn = _json(measured / "cnn" / f"{name}.json")
        if cnn is not None:
            mu, sigma = cnn["z01_est"] * 1e3, cnn["sigma"] * 1e3
            span += [mu - 2 * sigma, mu + 2 * sigma]
        pad = 0.1 * (max(span) - min(span)) + 0.05
        x = np.linspace(min(span) - pad, max(span) + pad, 600)
        if cnn is not None:
            ax.plot(x, _gauss(x, mu, sigma), color="tab:blue", lw=1.2, label="CNN (Stage 5)")
        phases = np.array(rec["binning_phases"]["z01_est"]) * 1e3
        ax.plot(
            phases, np.zeros_like(phases), "|", color="tab:red", ms=10, label="16 binning phases"
        )
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)
        ax.set_title(rec["label"], fontsize=9)
        ax.set_xlabel("z01 (mm)")
        ax.grid(alpha=0.3)
        ref = rec["references"].get("paper2_nm", {})
        metrics[name] = {
            "z01_est_mm": rec["z01_est"] * 1e3,
            "interval_68_mm": [
                rec["intervals"]["68"]["lower"] * 1e3,
                rec["intervals"]["68"]["upper"] * 1e3,
            ],
            "width_68_um": rec["intervals"]["68"]["width_um"],
            "minus_paper2_um": ref.get("distance_um"),
            "paper2_hpd_level": ref.get("hpd_level"),
            "paper2_inside": ref.get("inside"),
            "phases_std_um": rec["binning_phases"]["std_um"],
            "seconds_per_estimate": rec["seconds_per_estimate"]["median"],
        }
    axes[0, 0].set_ylabel("density (1/mm)")
    _legend(fig, axes, columns=4)
    _save(fig, out / "stage7-measured.png", dpi)
    return metrics


# --- metrics -------------------------------------------------------------------------------


def comparison_metrics(comparison: Path) -> dict | None:
    """The comparison rows and the paired differences of the main NPE against the CNN."""
    rows = _csv(comparison / "comparison.csv")
    if rows is None:
        return None
    out = {"rows": rows, "paired": {}}
    for path in sorted(comparison.glob("*.json")):
        record = json.loads(path.read_text())
        if "paired" in record:
            out["paired"][record["set"]] = record["paired"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="low resolution (CI)")
    parser.add_argument("--npe", type=Path, default=RESULTS / "npe", help="results of Stage 7")
    parser.add_argument(
        "--cnn", type=Path, default=RESULTS / "learned" / "spectrum-cnn-wide", help="Stage 4 CNN"
    )
    parser.add_argument("--learned", type=Path, default=RESULTS / "learned", help="matched CNNs")
    parser.add_argument("--measured", type=Path, default=RESULTS / "measured", help="Stage 5")
    args = parser.parse_args()
    dpi = 60 if args.quick else 120
    args.out.mkdir(parents=True, exist_ok=True)
    comparison = args.npe / "comparison"
    metrics: dict = {"quick": args.quick}
    steps = [
        ("training", lambda: training_figure(args.npe, args.cnn, args.learned, args.out, dpi)),
        ("calibration", lambda: calibration_figure(args.npe, args.cnn, args.out, dpi)),
        (
            "comparison",
            lambda: comparison_figure(comparison, args.npe / "timing.json", args.out, dpi),
        ),
        ("posteriors", lambda: posteriors_figure(comparison / "posteriors.json", args.out, dpi)),
        ("robustness", lambda: robustness_figure(args.npe / "sim2real", args.out, dpi)),
        (
            "measured",
            lambda: measured_figure(args.npe / "measured", args.measured, args.out, dpi),
        ),
        ("comparison_tables", lambda: comparison_metrics(comparison)),
    ]
    for name, fn in steps:
        start = time.perf_counter()
        value = fn()
        if value is None:
            print(f"{name}: inputs missing, skipped")
            continue
        metrics[name] = value
        print(f"{name}: {time.perf_counter() - start:.1f} s")
    with open(args.out / "stage7-metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
