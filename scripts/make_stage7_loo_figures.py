"""Figures of Stage 7 on the object model of Stage 6 (``docs/laborbuch/figures``).

* ``stage7-loo-training.png``: validation NLL per epoch of the NPEs on the leave-one-out sets
  (three seeds per sample) and of the leave-one-out CNNs of Stage 6 with the same seeds; open
  circles: selected epochs.
* ``stage7-loo-test.png``: on the leave-one-out test splits, per sample: MAE, coverage of the
  central 68 and 95 % intervals and median 68 % width of the NPE seeds, their ensemble, the CNN
  seeds and the guess (with the central intervals of the uniform prior).
* ``stage7-loo-calibration.png``: observed minus nominal coverage of the central intervals over
  all levels on the leave-one-out test splits, NPE seeds, ensemble and CNN (seed 0), with the
  95 % band of a calibrated method.
* ``stage7-loo-measured.png``: the posteriors of the NPE seeds and of their ensemble on the four
  measured holograms over the prior box, with [2] Table 1, our padding-4 search, the guess, the
  leave-one-out CNN of Stage 6 and the matched NPE of Stage 7 (simulator of Stage 5).
* ``stage7-loo-metrics.json``: the numbers quoted in the lab notebook.

Usage::

    python scripts/make_stage7_loo_figures.py                         # results/
    python scripts/make_stage7_loo_figures.py --quick --npe DIR --out /tmp/figures   # CI

Figures whose inputs are missing are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from holofocus.learned import read_predictions  # noqa: E402
from holofocus.learned.posterior import gaussian_pit, ks_uniform  # noqa: E402
from holofocus.measured import MEASURED_SAMPLES  # noqa: E402
from make_stage7_figures import (  # noqa: E402
    LEVELS,
    MEASURED_REFERENCES,
    _coverage_curves,
    _csv,
    _curves,
    _gauss,
    _json,
    _legend,
    _save,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "laborbuch" / "figures"
RESULTS = REPO / "results"
SAMPLES = list(MEASURED_SAMPLES)
SEEDS = (0, 1, 2)
NPE_COLOR, CNN_COLOR, ENSEMBLE_COLOR, GUESS_COLOR = "tab:red", "tab:blue", "tab:purple", "tab:green"


def run_name(sample: str, seed: int) -> str:
    return f"matched-{sample}" if seed == 0 else f"matched-{sample}-seed{seed}"


def cnn_run(cnn: Path, seeds: Path, sample: str, seed: int) -> Path:
    return cnn / f"loo-{sample}" if seed == 0 else seeds / f"seed-{seed}" / f"loo-{sample}"


def _seed_label(kind: str, seed: int) -> str:
    return f"{kind} (seed 0)" if seed == 0 else f"{kind}, seeds 1-2"


# --- training ------------------------------------------------------------------------------


def training_figure(npe: Path, cnn: Path, seeds: Path, out: Path, dpi: int) -> dict | None:
    samples = [s for s in SAMPLES if (npe / run_name(s, 0) / "history.csv").exists()]
    if not samples:
        return None
    fig, axes = plt.subplots(
        1,
        len(samples),
        figsize=(3.4 * len(samples) + 0.6, 3.6),
        squeeze=False,
        layout="constrained",
    )
    metrics = {}
    for ax, sample in zip(axes[0], samples, strict=True):
        best = []
        for kind, color, root in (("NPE", NPE_COLOR, None), ("CNN (Stage 6)", CNN_COLOR, cnn)):
            for seed in SEEDS:
                run = (
                    npe / run_name(sample, seed)
                    if root is None
                    else cnn_run(cnn, seeds, sample, seed)
                )
                lw, alpha = (1.3, 1.0) if seed == 0 else (0.8, 0.45)
                label = _seed_label(kind, seed) if seed in (0, 1) else None
                m = _curves([ax], run, ("nll",), color, "-", label, lw=lw, alpha=alpha)
                if m:
                    key = f"{'npe' if root is None else 'cnn'}-{sample}-seed{seed}"
                    metrics[key] = m
                    best.append(m["validation_nll_um_at_best"])
        ax.set_ylim(min(best) - 0.3, min(best) + 2.5)
        ax.set_title(MEASURED_SAMPLES[sample].label, fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("validation NLL (nats, dz01 in um)")
    n_train = {m["n_train"] for m in metrics.values() if m.get("n_train")}
    fig.suptitle(
        "leave-one-out sets of Stage 6: "
        + ", ".join(str(n) for n in sorted(n_train))
        + " training samples",
        fontsize=10,
    )
    _legend(fig, axes, columns=4)
    _save(fig, out / "stage7-loo-training.png", dpi)
    return metrics


# --- test splits ---------------------------------------------------------------------------


def _rows(comparison: Path) -> dict[str, dict[str, dict]]:
    """``comparison.csv`` as ``{sample: {method: row}}``."""
    rows = _csv(comparison / "comparison.csv") or []
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        out.setdefault(row["sample"], {})[row["method"]] = row
    return out


def test_figure(comparison: Path, out: Path, dpi: int) -> dict | None:
    table = _rows(comparison)
    samples = [s for s in SAMPLES if s in table]
    if not samples:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.0), layout="constrained")
    x = np.arange(len(samples))
    styles = {
        "guess": ("guess (prior intervals)", GUESS_COLOR, "x", -0.24),
        "cnn": ("CNN (Stage 6), seed 0", CNN_COLOR, "o", -0.12),
        "cnn-seed": ("CNN, seeds 1-2", CNN_COLOR, ".", -0.12),
        "npe": ("NPE, seed 0", NPE_COLOR, "o", 0.04),
        "npe-seed": ("NPE, seeds 1-2", NPE_COLOR, ".", 0.04),
        "ensemble": ("NPE, ensemble of 3", ENSEMBLE_COLOR, "D", 0.2),
    }
    keys = ("mae_um", "coverage_68", "coverage_95", "width_68_median_um")
    n = None
    for k, sample in enumerate(samples):
        for method, row in table[sample].items():
            style = "cnn-seed" if method.startswith("cnn-seed") else method
            style = "npe-seed" if method.startswith("npe-seed") else style
            label, color, marker, offset = styles[style]
            for ax, key in zip(axes, keys, strict=True):
                ax.plot(
                    k + offset,
                    row[key],
                    marker,
                    color=color,
                    ms=7 if marker != "." else 9,
                    mfc="white" if marker == "D" else color,
                    label=label,
                )
            n = int(row["n"])
    for ax, level in ((axes[1], 0.68), (axes[2], 0.95)):
        half = 1.96 * np.sqrt(level * (1 - level) / n)
        ax.axhspan(level - half, level + half, color="0.88", zorder=0, label=f"95 % band, n = {n}")
        ax.axhline(level, color="0.4", lw=0.8, ls=":")
    axes[0].set_ylabel("MAE (um)")
    axes[1].set_ylabel("coverage, central 68 % interval")
    axes[2].set_ylabel("coverage, central 95 % interval")
    axes[3].set_ylabel("median 68 % width (um)")
    axes[3].set_yscale("log")
    for ax in axes:
        ax.set_xticks(x, [MEASURED_SAMPLES[s].label for s in samples], fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"leave-one-out test splits of Stage 6 ({n} samples each)", fontsize=10)
    _legend(fig, axes, columns=7)
    _save(fig, out / "stage7-loo-test.png", dpi)
    return {s: table[s] for s in samples}


# --- calibration ---------------------------------------------------------------------------


def _tables(npe: Path, cnn: Path, sample: str) -> dict[str, dict[str, np.ndarray]]:
    tables = {}
    path = cnn / f"loo-{sample}" / "test" / "predictions.csv"
    if path.exists():
        rows = read_predictions(path)
        err = np.array([r["error_um"] for r in rows])
        sigma = np.array([r["sigma"] for r in rows]) * 1e6
        tables["cnn"] = {"pit": gaussian_pit(err, sigma), "hpd_level": None}
    names = {f"npe-seed{k}": npe / run_name(sample, k) for k in SEEDS}
    names["ensemble"] = npe / f"matched-{sample}-ensemble"
    for name, run in names.items():
        path = run / "test" / "predictions.csv"
        if path.exists():
            rows = read_predictions(path)
            tables[name] = {key: np.array([r[key] for r in rows]) for key in ("pit", "hpd_level")}
    return tables


def calibration_figure(npe: Path, cnn: Path, out: Path, dpi: int) -> dict | None:
    samples = [s for s in SAMPLES if (npe / run_name(s, 0) / "test" / "predictions.csv").exists()]
    if not samples:
        return None
    fig, axes = plt.subplots(
        1,
        len(samples),
        figsize=(3.5 * len(samples) + 0.6, 3.7),
        squeeze=False,
        layout="constrained",
    )
    metrics = {}
    for ax, sample in zip(axes[0], samples, strict=True):
        tables = _tables(npe, cnn, sample)
        n = len(tables["npe-seed0"]["pit"])
        band = 1.96 * np.sqrt(LEVELS * (1 - LEVELS) / n)
        ax.fill_between(LEVELS, -band, band, color="0.88", label=f"95 % band, n = {n}")
        ax.axhline(0, color="0.3", lw=0.8)
        metrics[sample] = {}
        for name, t in tables.items():
            pit = t["pit"]
            central = np.array([np.mean(np.abs(pit - 0.5) <= a / 2) for a in LEVELS])
            if name == "cnn":
                ax.plot(
                    LEVELS, central - LEVELS, color=CNN_COLOR, lw=1.3, label="CNN (Stage 6), seed 0"
                )
            elif name == "ensemble":
                ax.plot(
                    LEVELS,
                    central - LEVELS,
                    "--",
                    color=ENSEMBLE_COLOR,
                    lw=1.4,
                    label="NPE, ensemble of 3",
                )
            else:
                seed = int(name.removeprefix("npe-seed"))
                lw, alpha = (1.3, 1.0) if seed == 0 else (0.8, 0.5)
                label = _seed_label("NPE", seed) if seed in (0, 1) else None
                ax.plot(LEVELS, central - LEVELS, color=NPE_COLOR, lw=lw, alpha=alpha, label=label)
            entry = {
                "n": len(pit),
                "central_max_abs_deviation": float(np.abs(central - LEVELS).max()),
                "central_mean_deviation": float(np.mean(central - LEVELS)),
                "pit_mean": float(np.mean(pit)),
                "pit_ks": ks_uniform(pit),
                "pit_below_0.025": float(np.mean(pit < 0.025)),
                "pit_above_0.975": float(np.mean(pit > 0.975)),
            }
            if t["hpd_level"] is not None:
                _, regions = _coverage_curves(pit, t["hpd_level"])
                entry["hpd_max_abs_deviation"] = float(np.abs(regions - LEVELS).max())
            metrics[sample][name] = entry
        ax.set_title(MEASURED_SAMPLES[sample].label, fontsize=9)
        ax.set_xlabel("nominal level")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("observed minus nominal coverage")
    fig.suptitle("central intervals on the leave-one-out test splits", fontsize=10)
    _legend(fig, axes, columns=5)
    _save(fig, out / "stage7-loo-calibration.png", dpi)
    return metrics


# --- measured ------------------------------------------------------------------------------


def _density(ax, record: dict, color: str, style: str, lw: float, alpha: float, label) -> None:
    z = np.array(record["posterior"]["z01_mm"])
    ax.plot(
        z,
        np.array(record["posterior"]["density_per_mm"]),
        style,
        color=color,
        lw=lw,
        alpha=alpha,
        label=label,
    )


def measured_figure(npe: Path, npe_stage7: Path, out: Path, dpi: int) -> dict | None:
    records = {s: _json(npe / "measured" / f"{s}.json") for s in SAMPLES}
    records = {s: r for s, r in records.items() if r is not None}
    if not records:
        return None
    fig, axes = plt.subplots(
        1,
        len(records),
        figsize=(3.7 * len(records) + 0.6, 4.0),
        squeeze=False,
        layout="constrained",
    )
    metrics = {}
    for ax, (name, rec) in zip(axes[0], records.items(), strict=True):
        runs = rec["runs"]
        main = runs.get("ensemble") or runs["npe"]
        main_label = "NPE, ensemble of 3" if "ensemble" in runs else "NPE (seed 0)"
        for key, alpha in (("95", 0.10), ("68", 0.22)):
            i = main["intervals"][key]
            ax.axvspan(
                i["lower"] * 1e3,
                i["upper"] * 1e3,
                color=ENSEMBLE_COLOR,
                alpha=alpha,
                lw=0,
                label=f"{main_label}: {key} % interval",
            )
        for variant, r in runs.items():
            if variant == "ensemble":
                continue
            seed = 0 if variant == "npe" else int(variant.removeprefix("npe-seed"))
            lw, alpha = (1.2, 1.0) if seed == 0 else (0.8, 0.55)
            label = _seed_label("NPE", seed) if seed in (0, 1) else None
            _density(ax, r, NPE_COLOR, "-", lw, alpha, label)
        if "ensemble" in runs:
            _density(ax, runs["ensemble"], ENSEMBLE_COLOR, "--", 1.6, 1.0, main_label)
        stage7 = _json(npe_stage7 / f"{name}.json")
        if stage7 is not None:
            _density(
                ax, stage7, "0.5", ":", 1.2, 1.0, "matched NPE of Stage 7 (simulator of Stage 5)"
            )
        lo, hi = (v * 1e3 for v in rec["omega"])
        x = np.linspace(lo, hi, 800)
        cnn = rec.get("cnn_loo")
        if cnn is not None:
            ax.plot(
                x,
                _gauss(x, cnn["z01_est"] * 1e3, cnn["sigma"] * 1e3),
                color=CNN_COLOR,
                lw=1.2,
                label="CNN (Stage 6), seed 0",
            )
        for ref, (label, color, style) in MEASURED_REFERENCES.items():
            if ref in rec["references"]:
                ax.axvline(rec["references"][ref] * 1e3, color=color, ls=style, lw=1.1, label=label)
        ax.axvline(
            rec["z01_guess"] * 1e3, color=GUESS_COLOR, lw=1.1, label="guess (centre of Omega_z)"
        )
        phases = np.array(main["binning_phases"]["z01_est"]) * 1e3
        ax.plot(
            phases,
            np.zeros_like(phases),
            "|",
            color=ENSEMBLE_COLOR,
            ms=10,
            label="16 binning phases",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
        ax.set_title(rec["label"], fontsize=9)
        ax.set_xlabel("z01 (mm)")
        ax.grid(alpha=0.3)
        metrics[name] = {variant: _measured_metrics(r) for variant, r in runs.items()}
        metrics[name]["seeds"] = rec.get("seeds")
        if cnn is not None:
            metrics[name]["cnn_loo"] = {
                "z01_est_mm": cnn["z01_est"] * 1e3,
                "dz01_est_um": cnn["dz01_est"] * 1e6,
                "sigma_um": cnn["sigma"] * 1e6,
                "guess_inside": cnn["guess_inside"],
            }
    axes[0, 0].set_ylabel("density (1/mm)")
    _legend(fig, axes, columns=4)
    _save(fig, out / "stage7-loo-measured.png", dpi)
    return metrics


def _measured_metrics(r: dict) -> dict:
    refs = {
        key: {
            "distance_um": v["distance_um"],
            "inside": v["inside"],
            "hpd_level": v["hpd_level"],
        }
        for key, v in r["references"].items()
    }
    return {
        "z01_est_mm": r["z01_est"] * 1e3,
        "dz01_est_um": r["dz01_est"] * 1e6,
        "interval_68_mm": [
            r["intervals"]["68"]["lower"] * 1e3,
            r["intervals"]["68"]["upper"] * 1e3,
        ],
        "interval_95_mm": [
            r["intervals"]["95"]["lower"] * 1e3,
            r["intervals"]["95"]["upper"] * 1e3,
        ],
        "width_um": {k: i["width_um"] for k, i in r["intervals"].items()},
        "hpd_to_central": r["hpd_to_central"],
        "guess": {k: r["guess"][k] for k in ("pit", "hpd_level", "inside")},
        "references": refs,
        "modes_mm": [[m["z01"] * 1e3, m["relative_height"]] for m in r["modes"]],
        "edges": r["edges"],
        "prior_mass": r["prior_mass"],
        "phases_std_um": r["binning_phases"]["std_um"],
        "seconds_per_estimate": r["seconds_per_estimate"]["median"],
    }


# --- metrics -------------------------------------------------------------------------------


def comparison_metrics(comparison: Path) -> dict | None:
    """Paired differences and the spread over seeds of every sample."""
    out = {}
    for sample in SAMPLES:
        record = _json(comparison / f"{sample}.json")
        if record is not None:
            out[sample] = {
                "prior": record["prior"],
                "paired": record["paired"],
                "seeds": record["seeds"],
            }
    return out or None


def timing_metrics(path: Path) -> dict | None:
    record = _json(path)
    if record is None:
        return None
    return {
        "threads": record["threads"],
        "samples": {
            s: {
                "fov_shape": e["fov_shape"],
                "ms": {m: 1e3 * v["median"] for m, v in e["methods"].items()},
                "ratio_to_cnn": {m: v["ratio_to_cnn"] for m, v in e["methods"].items()},
                "npe_parts_ms": {k: 1e3 * v for k, v in e["npe_parts_median"].items()},
            }
            for s, e in record["samples"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="low resolution (CI)")
    parser.add_argument("--npe", type=Path, default=RESULTS / "npe-loo", help="NPE runs and study")
    parser.add_argument("--cnn", type=Path, default=RESULTS / "learned", help="<cnn>/loo-<sample>")
    parser.add_argument(
        "--cnn-seeds", type=Path, default=RESULTS / "objectmodel" / "seeds", help="CNN seeds 1-2"
    )
    parser.add_argument(
        "--npe-stage7", type=Path, default=RESULTS / "npe" / "measured", help="Stage 7 on DIR"
    )
    args = parser.parse_args()
    dpi = 60 if args.quick else 120
    args.out.mkdir(parents=True, exist_ok=True)
    comparison = args.npe / "comparison"
    metrics: dict = {"quick": args.quick}
    steps = [
        ("training", lambda: training_figure(args.npe, args.cnn, args.cnn_seeds, args.out, dpi)),
        ("test", lambda: test_figure(comparison, args.out, dpi)),
        ("calibration", lambda: calibration_figure(args.npe, args.cnn, args.out, dpi)),
        ("measured", lambda: measured_figure(args.npe, args.npe_stage7, args.out, dpi)),
        ("comparison", lambda: comparison_metrics(comparison)),
        ("timing", lambda: timing_metrics(args.npe / "timing.json")),
    ]
    for name, fn in steps:
        start = time.perf_counter()
        value = fn()
        if value is None:
            print(f"{name}: inputs missing, skipped")
            continue
        metrics[name] = value
        print(f"{name}: {time.perf_counter() - start:.1f} s")
    with open(args.out / "stage7-loo-metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
