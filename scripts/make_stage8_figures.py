"""Figures of Stage 8 (``docs/laborbuch/figures``).

* ``stage8-misspec.png``: the mean distance to the 10 nearest training samples of every test
  sample (histogram) and of the measured hologram (line) and its 16 binning phases (ticks),
  in the network input and in the embedding of each NPE seed, per probe.
* ``stage8-misspec-percentiles.png``: the percentiles of the measured holograms among the test
  split for every distance (dot) and the range over the binning phases (bar), with the
  positive control against the Stage 5 simulator if ``misspec`` ran with ``--control``.
* ``stage8-misspec-failures.png``: on the test split, the spread of the NPE seeds and the
  absolute error of the NPE ensemble against the rank of the embedding distance (seed 0), with
  the measured hologram (star; error against our padding-4 search).
* ``stage8-shrinkage.png``: shift of the NPE ensemble from the guess against the true offset on
  the test split, the 25 test samples nearest to the measured hologram in the embedding (seed 0)
  and the hologram (truth: padding 4).
* ``stage8-ensembles.png``: CNN ensemble against NPE ensemble with their seeds and the guess on
  the leave-one-out test splits and pooled: MAE, coverage, width, NLL.
* ``stage8-ensembles-measured.png``: the two ensembles on the measured holograms over the prior
  box, with [2], padding 4 and the guess.
* ``stage8-warmstart.png``: the padding-4 searches on the measured holograms, full, warm and in
  the 95 % intervals: the reconstructed points in order and their number.
* ``stage8-metrics.json``: the numbers quoted in the lab notebook.

Usage::

    python scripts/make_stage8_figures.py                         # results/
    python scripts/make_stage8_figures.py --quick --out /tmp/figures   # CI

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

from holofocus.measured import MEASURED_SAMPLES  # noqa: E402
from make_stage7_figures import MEASURED_REFERENCES, _csv, _json, _legend, _save  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "laborbuch" / "figures"
RESULTS = REPO / "results"
SAMPLES = list(MEASURED_SAMPLES)
NPE_COLOR, CNN_COLOR, ENSEMBLE_COLOR, GUESS_COLOR = "tab:red", "tab:blue", "tab:purple", "tab:green"
CNN_ENSEMBLE_COLOR = "navy"
CONTROL_COLOR = "tab:brown"
SPACES = {
    "input": "network input",
    "embedding": "embedding, NPE seed 0",
    "embedding-seed1": "embedding, NPE seed 1",
    "embedding-seed2": "embedding, NPE seed 2",
}
MAIN = "knn10"
SEARCHES = {
    "cold": ("full: guess +- 5 mm (Stage 3)", "black", "o"),
    "warm-stage5": ("warm: CNN of Stage 5", "0.6", "v"),
    "warm-cnn": ("warm: CNN ensemble", CNN_ENSEMBLE_COLOR, "s"),
    "warm-npe": ("warm: NPE ensemble", ENSEMBLE_COLOR, "D"),
    "box95-cnn": ("95 % interval, CNN ensemble", "tab:cyan", "^"),
    "box95-npe": ("95 % interval, NPE ensemble", "tab:pink", "P"),
}


def _misspec(directory: Path) -> dict[str, tuple[dict, list[dict]]]:
    out = {}
    for s in SAMPLES:
        record, rows = _json(directory / f"{s}.json"), _csv(directory / f"test-{s}.csv")
        if record is not None and rows:
            out[s] = (record, rows)
    return out


def _column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


def _ranks(values: np.ndarray) -> np.ndarray:
    """Percentile ranks among the values themselves (ties count half)."""
    v = np.asarray(values, dtype=float)
    below = (v[None, :] < v[:, None]).sum(axis=1)
    ties = (v[None, :] == v[:, None]).sum(axis=1)
    return 100.0 * (below + 0.5 * ties) / len(v)


# --- misspecification ----------------------------------------------------------------------


def misspec_figure(directory: Path, out: Path, dpi: int) -> dict | None:
    data = _misspec(directory)
    if not data:
        return None
    spaces = [s for s in SPACES if any(s in r["spaces"] for r, _ in data.values())]
    fig, axes = plt.subplots(
        len(spaces),
        len(data),
        figsize=(3.6 * len(data) + 0.6, 2.4 * len(spaces) + 0.8),
        squeeze=False,
        layout="constrained",
    )
    metrics = {}
    for col, (name, (record, rows)) in enumerate(data.items()):
        metrics[name] = {}
        for row, space in enumerate(spaces):
            ax = axes[row, col]
            entry = record["spaces"].get(space)
            key = f"{space}_{MAIN}"
            if entry is None or key not in rows[0]:
                ax.set_axis_off()
                continue
            test = _column(rows, key)
            m = entry["measures"][MAIN]
            ax.hist(test, bins=30, color="0.75", label="test split")
            title = f"{record['label']}, {SPACES[space]}"
            if "measured" in m:
                ax.axvline(m["measured"], color="black", lw=1.4, label="measured hologram")
                phases = np.array(m["phases"]["values"])
                ax.plot(phases, np.zeros_like(phases), "|", color="tab:orange", ms=12, mew=1.2)
                ax.plot([], [], "|", color="tab:orange", ms=10, label="16 binning phases")
                title += f"\n{m['percentile']:.1f} % (phases {m['phases']['percentile_min']:.0f}"
                title += f"-{m['phases']['percentile_max']:.0f} %)"
                metrics[name][space] = {
                    "percentile": m["percentile"],
                    "phases": [m["phases"]["percentile_min"], m["phases"]["percentile_max"]],
                }
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            if row == len(spaces) - 1:
                ax.set_xlabel("mean distance to the 10 nearest training samples", fontsize=8)
    _legend(fig, axes, columns=3)
    _save(fig, out / "stage8-misspec.png", dpi)
    return metrics


def _measure_label(space: str, measure: str, components: int) -> str:
    if measure.startswith("knn"):
        text = f"{measure.removeprefix('knn')} nearest"
    elif measure == "mahalanobis":
        text = f"Mahalanobis, {components} PC"
    else:
        text = f"residual outside {components} PC"
    return f"{'input' if space == 'input' else space.replace('embedding', 'emb.')}: {text}"


def _percentile_groups(record: dict) -> list[tuple[str, dict, bool]]:
    """``(space, entry, control)`` of a record: the leave-one-out spaces, then the positive
    control against the Stage 5 simulator."""
    groups = [(s, record["spaces"][s], False) for s in SPACES if s in record["spaces"]]
    control = (record.get("control") or {}).get("spaces", {})
    return groups + [(s, control[s], True) for s in ("input", "embedding") if s in control]


def percentiles_figure(directory: Path, out: Path, dpi: int) -> dict | None:
    data = _misspec(directory)
    data = {s: d for s, d in data.items() if d[0].get("measured")}
    if not data:
        return None
    rows = max(
        sum(len(e["measures"]) for _, e, _ in _percentile_groups(r)) for r, _ in data.values()
    )
    fig, axes = plt.subplots(
        1,
        len(data),
        figsize=(3.4 * len(data) + 2.0, 0.22 * rows + 1.8),
        squeeze=False,
        layout="constrained",
    )
    metrics, controls = {}, False
    for k, (ax, (name, (record, _))) in enumerate(zip(axes[0], data.items(), strict=True)):
        labels, positions, metrics[name] = [], [], {}
        y = 0.0
        for space, entry, control in _percentile_groups(record):
            controls |= control
            for measure, m in entry["measures"].items():
                if control:
                    color = CONTROL_COLOR
                else:
                    color = "black" if space == "input" else ENSEMBLE_COLOR
                lo, hi = m["phases"]["percentile_min"], m["phases"]["percentile_max"]
                ax.plot([lo, hi], [y, y], color=color, lw=3, alpha=0.35, solid_capstyle="butt")
                ax.plot(m["percentile"], y, "o", color=color, ms=5)
                label = _measure_label(space, measure, entry["components"])
                labels.append(f"Stage 5 {label}" if control else label)
                positions.append(y)
                metrics[name][f"{'control-' if control else ''}{space}/{measure}"] = m["percentile"]
                y -= 1
            y -= 0.6
        ax.set_yticks(positions, labels if k == 0 else [""] * len(labels), fontsize=7)
        ax.axvspan(95, 100, color="0.9", zorder=0)
        ax.set_xlim(0, 100)
        ax.set_xlabel("percentile among the test split (%)", fontsize=8)
        ax.set_title(record["label"], fontsize=9)
        ax.grid(alpha=0.3, axis="x")
    ax = axes[0, 0]
    ax.plot([], [], "o", color="black", label="measured hologram, network input")
    ax.plot([], [], "o", color=ENSEMBLE_COLOR, label="measured hologram, NPE embedding")
    if controls:
        ax.plot(
            [],
            [],
            "o",
            color=CONTROL_COLOR,
            label="against the Stage 5 simulator (input, embedding of its NPE)",
        )
    ax.plot([], [], color="0.5", lw=3, alpha=0.5, label="range over the 16 binning phases")
    _legend(fig, axes, columns=3)
    _save(fig, out / "stage8-misspec-percentiles.png", dpi)
    return metrics


def failures_figure(directory: Path, out: Path, dpi: int) -> dict | None:
    data = _misspec(directory)
    space = "embedding"
    data = {s: d for s, d in data.items() if f"{space}_{MAIN}" in d[1][0]}
    if not data:
        return None
    panels = [("npe_seed_std_um", "spread of the NPE seeds (um)", "seed_std_um")]
    panels.append(("npe_abs_error_um", "|error| of the NPE ensemble (um)", "abs_error_um"))
    panels = [p for p in panels if p[0] in next(iter(data.values()))[1][0]]
    fig, axes = plt.subplots(
        len(panels),
        len(data),
        figsize=(3.6 * len(data) + 0.6, 3.0 * len(panels) + 0.8),
        squeeze=False,
        layout="constrained",
    )
    metrics = {}
    for col, (name, (record, rows)) in enumerate(data.items()):
        distance = _column(rows, f"{space}_{MAIN}")
        rank = _ranks(distance)
        m = record["spaces"][space]["measures"][MAIN]
        measured = (record.get("measured_failures") or {}).get("npe", {})
        metrics[name] = {}
        for row, (key, label, measured_key) in enumerate(panels):
            ax = axes[row, col]
            value = _column(rows, key)
            ax.plot(rank, value, ".", color="0.55", ms=4, label="test samples")
            near, far = rank <= 25, rank >= 75
            for mask, text in ((near, "nearest"), (far, "farthest")):
                x = 12.5 if text == "nearest" else 87.5
                ax.plot(x, np.median(value[mask]), "_", color="black", ms=28, mew=2)
            ax.plot([], [], "_", color="black", ms=12, mew=2, label="median of a quarter")
            a = m["failures"][key]
            title = f"{record['label']}: Spearman {a['spearman']:+.2f} "
            title += f"[{a['ci'][0]:+.2f}, {a['ci'][1]:+.2f}]"
            if "percentile" in m:
                if measured_key == "abs_error_um" and "minus_p4_um" in measured:
                    y = abs(measured["minus_p4_um"])
                else:
                    y = measured.get(measured_key)
                if y is not None:
                    ax.plot(
                        m["percentile"],
                        y,
                        "*",
                        color="tab:orange",
                        ms=14,
                        mec="black",
                        label="measured hologram (error: against padding 4)",
                    )
            ax.set_yscale("log")
            ax.set_xlim(0, 100)
            ax.set_title(title, fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, which="both")
            if row == len(panels) - 1:
                ax.set_xlabel(
                    "rank of the embedding distance (seed 0) in the test split (%)", fontsize=8
                )
            metrics[name][key] = {
                "spearman": a["spearman"],
                "ci": a["ci"],
                "nearest_quarter_mean": a["nearest_quarter_mean"],
                "farthest_quarter_mean": a["farthest_quarter_mean"],
            }
    _legend(fig, axes, columns=3)
    _save(fig, out / "stage8-misspec-failures.png", dpi)
    return metrics


def shrinkage_figure(directory: Path, out: Path, dpi: int) -> dict | None:
    data = {s: d for s, d in _misspec(directory).items() if "npe_shift_um" in d[1][0]}
    if not data:
        return None
    fig, axes = plt.subplots(
        1, len(data), figsize=(3.6 * len(data) + 0.6, 4.0), squeeze=False, layout="constrained"
    )
    metrics = {}
    for ax, (name, (record, rows)) in zip(axes[0], data.items(), strict=True):
        offset, shift = _column(rows, "dz01_true_um") / 1e3, _column(rows, "npe_shift_um") / 1e3
        ax.plot(offset, shift, ".", color="0.6", ms=4, label="test samples")
        lim = 5.2
        ax.plot([-lim, lim], [-lim, lim], color="0.3", lw=0.8, ls=":", label="estimate = truth")
        ax.axhline(0, color=GUESS_COLOR, lw=0.9, label="estimate = guess")
        test = record["test"]
        title = f"{record['label']}: slope {test['npe_shrinkage']:.2f}"
        near = record.get("neighbours", {}).get("embedding")
        if near is not None:
            rows_near = [rows[i] for i in near["indices"]]
            ax.plot(
                _column(rows_near, "dz01_true_um") / 1e3,
                _column(rows_near, "npe_shift_um") / 1e3,
                "o",
                color=ENSEMBLE_COLOR,
                ms=4,
                mfc="none",
                label=f"{near['n']} nearest in the embedding (seed 0)",
            )
            title += f", nearest {near['npe_shrinkage']:.2f}"
        measured = (record.get("measured_failures") or {}).get("npe")
        if measured is not None and "p4_minus_guess_um" in measured:
            ax.plot(
                measured["p4_minus_guess_um"] / 1e3,
                measured["shift_um"] / 1e3,
                "*",
                color="tab:orange",
                ms=15,
                mec="black",
                label="measured hologram (truth: padding 4)",
            )
            title += f"\nhologram moved {measured['moved_fraction']:.2f} of the way"
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("truth minus guess (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        metrics[name] = {
            "test_shrinkage": test["npe_shrinkage"],
            "nearest_shrinkage": None if near is None else near["npe_shrinkage"],
            "moved_fraction": None if measured is None else measured.get("moved_fraction"),
        }
    axes[0, 0].set_ylabel("NPE ensemble minus guess (mm)", fontsize=8)
    _legend(fig, axes, columns=5)
    _save(fig, out / "stage8-shrinkage.png", dpi)
    return metrics


# --- ensembles -----------------------------------------------------------------------------


def ensembles_figure(directory: Path, out: Path, dpi: int) -> dict | None:
    rows = _csv(directory / "comparison.csv")
    if not rows:
        return None
    table: dict[str, dict[str, dict]] = {}
    for row in rows:
        table.setdefault(row["sample"], {})[row["method"]] = row
    samples = [s for s in (*SAMPLES, "pooled") if s in table]
    styles = {
        "guess": ("guess (prior intervals)", GUESS_COLOR, "x", -0.27),
        "cnn": ("CNN seeds", CNN_COLOR, ".", -0.15),
        "cnn-ensemble": ("CNN ensemble (mixture of 3)", CNN_ENSEMBLE_COLOR, "D", -0.15),
        "npe": ("NPE seeds", NPE_COLOR, ".", 0.1),
        "npe-ensemble": ("NPE ensemble (mixture of 3)", ENSEMBLE_COLOR, "D", 0.1),
    }
    keys = (
        ("mae_um", "MAE (um)"),
        ("coverage_68", "coverage, central 68 % interval"),
        ("coverage_95", "coverage, central 95 % interval"),
        ("width_68_median_um", "median 68 % width (um)"),
        ("nll", "NLL (per um)"),
    )
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.9), layout="constrained")
    n = None
    for k, sample in enumerate(samples):
        for method, row in table[sample].items():
            style = "cnn" if method.startswith("cnn-seed") else method
            style = "npe" if method.startswith("npe-seed") else style
            label, color, marker, offset = styles[style]
            for ax, (key, _) in zip(axes, keys, strict=True):
                if row.get(key, "") == "":
                    continue
                ax.plot(
                    k + offset,
                    row[key],
                    marker,
                    color=color,
                    ms=9 if marker == "." else 6,
                    mfc="white" if marker == "D" else color,
                    mew=1.5,
                    label=label,
                )
            if sample != "pooled":
                n = int(row["n"])
    for ax, level in ((axes[1], 0.68), (axes[2], 0.95)):
        half = 1.96 * np.sqrt(level * (1 - level) / n)
        ax.axhspan(level - half, level + half, color="0.88", zorder=0, label=f"95 % band, n = {n}")
        ax.axhline(level, color="0.4", lw=0.8, ls=":")
    axes[3].set_yscale("log")
    labels = [MEASURED_SAMPLES[s].label if s in MEASURED_SAMPLES else "pooled" for s in samples]
    for ax, (_, ylabel) in zip(axes, keys, strict=True):
        ax.set_xticks(np.arange(len(samples)), labels, fontsize=7, rotation=20)
        ax.axvline(len(samples) - 1.5, color="0.5", lw=0.6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"leave-one-out test splits ({n} samples each)", fontsize=10)
    _legend(fig, axes, columns=6)
    _save(fig, out / "stage8-ensembles.png", dpi)
    pooled = _json(directory / "pooled.json")
    metrics = {s: {m: {k: table[s][m].get(k) for k, _ in keys} for m in table[s]} for s in samples}
    if pooled is not None:
        metrics["pooled_paired"] = {
            pair: {
                key: d[key]
                for key in ("mae_um", "coverage_68", "coverage_95", "nll", "width_68_ratio_median")
                if key in d
            }
            for pair, d in pooled["paired"].items()
        }
    for s in SAMPLES:
        record = _json(directory / f"{s}.json")
        if record is not None:
            metrics[s]["paired"] = {
                pair: {
                    k: d[k]
                    for k in ("mae_um", "nll", "coverage_68", "width_68_ratio_median")
                    if k in d
                }
                for pair, d in record["paired"].items()
            }
    return metrics


def _density(ax, record: dict, color: str, style: str, label: str) -> None:
    z = np.array(record["posterior"]["z01_mm"])
    ax.plot(
        z, np.array(record["posterior"]["density_per_mm"]), style, color=color, lw=1.5, label=label
    )


def measured_figure(sources: Path, out: Path, dpi: int) -> dict | None:
    records = {
        s: (
            _json(sources / "cnn-loo-ensemble" / f"{s}.json"),
            _json(sources / "npe-loo-ensemble" / f"{s}.json"),
        )
        for s in SAMPLES
    }
    records = {s: r for s, r in records.items() if r[0] is not None and r[1] is not None}
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
    for ax, (name, (cnn, npe)) in zip(axes[0], records.items(), strict=True):
        for rec, color, style, label in (
            (cnn, CNN_ENSEMBLE_COLOR, "-", "CNN ensemble"),
            (npe, ENSEMBLE_COLOR, "--", "NPE ensemble"),
        ):
            i = rec["intervals"]["68"]
            ax.axvspan(
                i["lower"] * 1e3,
                i["upper"] * 1e3,
                color=color,
                alpha=0.12,
                lw=0,
                label=f"{label}: 68 % interval",
            )
            _density(ax, rec, color, style, label)
        refs = npe["references"]
        for ref, (label, color, style) in MEASURED_REFERENCES.items():
            if ref in refs:
                ax.axvline(refs[ref]["z01"] * 1e3, color=color, ls=style, lw=1.1, label=label)
        ax.axvline(
            npe["z01_guess"] * 1e3, color=GUESS_COLOR, lw=1.1, label="guess (centre of Omega_z)"
        )
        lo, hi = (v * 1e3 for v in npe["omega"])
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
        ax.set_title(npe["label"], fontsize=9)
        ax.set_xlabel("z01 (mm)")
        ax.grid(alpha=0.3)
        metrics[name] = {
            kind: {
                "z01_est_mm": rec["z01_est"] * 1e3,
                "minus_paper2_um": rec["references"]["paper2_nm"]["distance_um"],
                "minus_p4_um": rec["references"].get("baseline_b1_p4", {}).get("distance_um"),
                "width_68_um": rec["intervals"]["68"]["width_um"],
                "width_95_um": rec["intervals"]["95"]["width_um"],
                "seed_std_um": rec["seeds"]["std_um"],
            }
            for kind, rec in (("cnn-ensemble", cnn), ("npe-ensemble", npe))
        }
    axes[0, 0].set_ylabel("density (1/mm)")
    _legend(fig, axes, columns=4)
    _save(fig, out / "stage8-ensembles-measured.png", dpi)
    return metrics


# --- warm start ----------------------------------------------------------------------------


def warmstart_figure(measured: Path, out: Path, dpi: int) -> dict | None:
    summary = _json(measured / "warmstart.json")
    if summary is None:
        return None
    samples = [s for s in SAMPLES if s in summary["samples"]]
    fig, axes = plt.subplots(
        2,
        len(samples),
        figsize=(3.7 * len(samples) + 0.6, 7.0),
        squeeze=False,
        layout="constrained",
    )
    for col, name in enumerate(samples):
        entries = summary["samples"][name]
        ax = axes[0, col]
        cold = entries["cold"]["z01"] * 1e3
        for variant, e in entries.items():
            label, color, marker = SEARCHES[variant]
            record = json.loads((REPO / e["record"]).read_text())
            points = [v["z01"] * 1e3 for v in record["result"]["evaluations"] if v["reconstructed"]]
            ax.plot(
                np.arange(1, len(points) + 1),
                np.array(points) - cold,
                marker=marker,
                color=color,
                ms=4,
                lw=0.9,
                label=label,
            )
        ax.axhspan(
            -0.1, 0.1, color="0.88", zorder=0, label="+- 100 um (xatol) around the full search"
        )
        ax.set_title(MEASURED_SAMPLES[name].label, fontsize=9)
        ax.set_xlabel("reconstruction", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        variants = list(entries)
        x = np.arange(len(variants))
        counts = [entries[v]["n_reconstructions"] for v in variants]
        colors = [SEARCHES[v][1] for v in variants]
        ax.bar(x, counts, color=colors, alpha=0.8)
        for xi, v in zip(x, variants, strict=True):
            e = entries[v]
            text = "" if v == "cold" else f"{e['minus_cold_um']:+.0f}"
            ax.text(xi, counts[xi] + 0.2, text, ha="center", fontsize=6.5)
        ax.set_xticks(x, [v.replace("-", "\n", 1) for v in variants], fontsize=6.5)
        ax.set_ylim(0, max(counts) + 2.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, axis="y")
    axes[0, 0].set_ylabel("z01 minus the full search (mm)", fontsize=8)
    axes[1, 0].set_ylabel("reconstructions (label: z01 minus the full search, um)", fontsize=8)
    _legend(fig, axes[0], columns=4)
    _save(fig, out / "stage8-warmstart.png", dpi)
    return {
        "totals": summary["totals"],
        "samples": {
            s: {
                v: {
                    k: e.get(k)
                    for k in (
                        "z01",
                        "minus_cold_um",
                        "same_as_cold",
                        "mfe_relative_to_best",
                        "n_reconstructions",
                        "distinct_points",
                        "seconds",
                        "threads",
                        "seconds_per_reconstruction",
                        "start_minus_cold_um",
                        "bounds",
                    )
                }
                for v, e in summary["samples"][s].items()
            }
            for s in samples
        },
    }


def misspec_metrics(directory: Path) -> dict | None:
    data = _misspec(directory)
    if not data:
        return None
    out = {}
    for name, (record, _) in data.items():
        out[name] = {
            "spaces": {
                space: {
                    "rank": e["rank"],
                    "components": e["components"],
                    "components_for": e["components_for"],
                    "measures": {
                        m: {k: v[k] for k in ("measured", "percentile", "phases", "test") if k in v}
                        | {"spearman": {f: a["spearman"] for f, a in v["failures"].items()}}
                        for m, v in e["measures"].items()
                    },
                    "neighbourhood": e.get("neighbourhood"),
                }
                for space, e in record["spaces"].items()
            },
            "test": record["test"],
            "neighbours": {
                s: {k: v for k, v in n.items() if k != "indices"}
                for s, n in record.get("neighbours", {}).items()
            },
            "measured_failures": record.get("measured_failures"),
        }
        control = record.get("control")
        if control is not None:
            out[name]["control"] = {
                "model": control["model"],
                "spaces": {
                    space: {
                        "components": e["components"],
                        "measures": {
                            m: {k: v[k] for k in ("measured", "percentile", "phases", "test")}
                            | {"measured_to_test_max": v["measured"] / v["test"]["max"]}
                            for m, v in e["measures"].items()
                        },
                    }
                    for space, e in control["spaces"].items()
                },
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="low resolution (CI)")
    parser.add_argument("--misspec", type=Path, default=RESULTS / "misspecification")
    parser.add_argument("--ensembles", type=Path, default=RESULTS / "ensembles")
    parser.add_argument("--measured", type=Path, default=RESULTS / "measured")
    args = parser.parse_args()
    dpi = 60 if args.quick else 120
    args.out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"quick": args.quick}
    steps = [
        ("misspec", lambda: misspec_figure(args.misspec, args.out, dpi)),
        ("percentiles", lambda: percentiles_figure(args.misspec, args.out, dpi)),
        ("failures", lambda: failures_figure(args.misspec, args.out, dpi)),
        ("shrinkage", lambda: shrinkage_figure(args.misspec, args.out, dpi)),
        ("ensembles", lambda: ensembles_figure(args.ensembles, args.out, dpi)),
        ("ensembles_measured", lambda: measured_figure(args.measured, args.out, dpi)),
        ("warmstart", lambda: warmstart_figure(args.measured, args.out, dpi)),
        ("misspec_details", lambda: misspec_metrics(args.misspec)),
    ]
    for name, fn in steps:
        start = time.perf_counter()
        value = fn()
        if value is None:
            print(f"{name}: inputs missing, skipped")
            continue
        metrics[name] = value
        print(f"{name}: {time.perf_counter() - start:.1f} s")
    with open(args.out / "stage8-metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
