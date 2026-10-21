"""Stage 5 figures for the lab notebook (``docs/laborbuch/figures``).

* ``stage5-measured.png``: the estimates on the four measured holograms relative to the
  Nelder-Mead result of [2] (Table 1): the CNN with its 68 and 95 % intervals (the variant not
  selected open, the model of the realistic set green), our cold baselines at padding 4 and 2,
  the warm start from the CNN, the visual value of [2] and the distance of the example in [1],
  inside the box Omega_z around the guess; right, the time per autofocus.
* ``stage5-sim2real.png``: per sample the CDF of the CNN error on its matched simulated test
  split and on the test split with all measured properties (``combined``), against the
  disagreement of the CNN on the measured hologram with [2] and with our padding-4 baseline.
* ``stage5-perturbations.png``: per sample and simulator assumption the paired shift of the
  estimates (mean with 95 % CI, median) and the MAE, with the measured disagreement.
* ``stage5-realistic.png``: where a model of the realistic set exists (``perturb --out
  <sim2real>/realistic``), its errors against those of the matched model on the matched test
  split, with all measured properties and on its own test split, the measured disagreements of
  both, and the MAE of both under every perturbation.
* ``stage5-spectra.png``: the CNN input of each measured hologram against the 5-95 % band of
  its matched test split and the median with all measured properties; below, the percentile of
  the measured hologram in every set for the spectral features.
* ``stage5-noise.png``: noise power spectra of the background corners with the fits behind the
  PSF width (full resolution) and the flat-field residual (4x grid).
* ``stage5-holograms.png``: the measured holograms on the 4x grid next to a simulated test
  sample without and with all measured properties, on the colour scale of the measured one.
* ``stage5-training.png``: validation MAE per epoch of the matched models (and of the model of
  the realistic set, on its own validation split).
* ``stage5-metrics.json``: the numbers quoted in the lab notebook.

Usage::

    python scripts/make_stage5_figures.py --holograms DIR       # results/ and data/
    python scripts/make_stage5_figures.py --quick --learned /tmp/none --measured /tmp/none \\
        --sim2real /tmp/sim2real-ci --out /tmp/figures   # smoke test (CI)

Figures whose inputs are missing are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from holofocus.learned import read_history, read_predictions  # noqa: E402
from holofocus.learned.sim2real import (  # noqa: E402
    FEATURE_NAMES,
    PERTURBATIONS,
    blurred_white_spectrum,
    lorentzian_spectrum,
)
from holofocus.measured import MEASURED_SAMPLES, load_measured  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "laborbuch" / "figures"
RESULTS = REPO / "results"
SAMPLES = list(MEASURED_SAMPLES)
VARIANTS = ("", "-hf")
REALISTIC = "-realistic"
GUESS_MAE_UM = 2500.0
"""MAE of the guess itself (``dz01_est = 0``) for ``dz01`` uniform in +-5 mm."""
METHODS = {
    "cnn": ("CNN", "tab:blue", "o"),
    "realistic": ("CNN, realistic training set", "tab:green", "o"),
    "b1_p4": ("NM (ours), padding 4", "tab:red", "D"),
    "b1_p2": ("NM (ours), padding 2", "tab:orange", "s"),
    "warm": ("NM from the CNN, padding 4", "tab:purple", "^"),
    "paper2_visual": ("[2] visual", "black", "x"),
    "paper1_example": ("example of [1]", "0.45", "+"),
    "paper2_gpu": ("[2] on the GPU (estimate)", "black", "*"),
}
CORNER_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")


def _save(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _comparison_rows(measured: Path) -> dict[str, dict]:
    comparison = _json(measured / "comparison.json")
    if comparison is None:
        return {}
    return {name: d["row"] for name, d in comparison["samples"].items()}


def _short(name: str) -> str:
    return MEASURED_SAMPLES[name].label


def _percent(x: float) -> str:
    """A percentile as text, with one decimal where rounding would give 0 or 100."""
    return f"{x:.1f}" if 0 < x < 0.5 or 99.5 <= x < 100 else f"{x:.0f}"


def _point(ax, x, y, key: str, filled: bool = True, **kwargs) -> None:
    label, color, marker = METHODS[key]
    face = color if filled else "white"
    ax.plot(x, y, marker, color=color, mfc=face, ms=7, ls="none", label=label, **kwargs)


def _legend(fig, axes, columns: int) -> None:
    """One legend below the panels with the labels of all ``axes``."""
    handles = {}
    for a in np.atleast_1d(axes).flat:
        for h, label in zip(*a.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, h)
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="outside lower center",
        ncols=max(1, min(4, columns)),
        fontsize=8,
    )


def measured_figure(measured: Path, out: Path, dpi: int) -> dict | None:
    rows = _comparison_rows(measured)
    names = [n for n in SAMPLES if n in rows]
    if not names:
        return None
    fig, (ax, at) = plt.subplots(
        1,
        2,
        figsize=(12, 1.4 + 1.45 * len(names)),
        width_ratios=[2.2, 1.1],
        sharey=True,
        layout="constrained",
    )
    ticks = []
    for y, name in enumerate(names):
        row, sample = rows[name], MEASURED_SAMPLES[name]
        ref = sample.z01_nm

        def mm(z, ref=ref):
            return (z - ref) * 1e3

        box = (mm(sample.z01_guess - sample.confidence), mm(sample.z01_guess + sample.confidence))
        ax.fill_betweenx([y - 0.44, y + 0.44], *box, color="0.93", lw=0)
        cnn = _json(measured / "cnn" / f"{name}.json") or {"variants": {}, "selected": None}
        matched = [r for r in cnn["variants"] if not r.endswith(REALISTIC)]
        offsets = {cnn["selected"]: -0.34}
        for run in cnn["variants"]:
            offsets.setdefault(run, -0.1 if run.endswith(REALISTIC) else -0.22)
        for run, variant in cnn["variants"].items():
            dy = offsets[run]
            key = "realistic" if run.endswith(REALISTIC) else "cnn"
            color = METHODS[key][1]
            filled = run == cnn["selected"] or key == "realistic"
            for level, lw in (("95", 1.2), ("68", 3.5 if filled else 2.0)):
                i = variant["intervals"][level]
                ax.plot([mm(i["lower"]), mm(i["upper"])], [y + dy] * 2, color=color, lw=lw)
            _point(ax, mm(variant["z01_est"]), y + dy, key, filled=filled, zorder=3)
            if key == "cnn" and len(matched) > 1:
                ax.annotate(
                    "hf" if run.endswith("-hf") else "std",
                    (mm(variant["z01_est"]), y + dy),
                    xytext=(6, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )
        for key, dy in (("b1_p4", 0.13), ("b1_p2", 0.24), ("warm", 0.35)):
            if f"{key}_mm" in row:
                _point(ax, mm(row[f"{key}_mm"] * 1e-3), y + dy, key)
        _point(ax, mm(sample.z01_visual), y + 0.02, "paper2_visual")
        _point(ax, mm(sample.z01_example), y + 0.02, "paper1_example")
        fr = sample.geometry(4).fresnel_number
        ticks.append(f"{sample.label}\n{sample.energy_kev:.0f} keV, Fr {fr:.2e}")

        times = {"cnn": row["cnn_ms"] * 1e-3, "paper2_gpu": row["paper2_gpu_seconds_est"]}
        times.update(
            {k: row[f"{k}_seconds"] for k in ("b1_p4", "b1_p2", "warm") if f"{k}_seconds" in row}
        )
        for key, dy in (("cnn", -0.34), ("b1_p4", 0.13), ("b1_p2", 0.24), ("warm", 0.35)):
            if key in times:
                _point(at, times[key], y + dy, key)
        _point(at, times["paper2_gpu"], y + 0.02, "paper2_gpu")
    ax.axvline(0, color="black", lw=1, ls="--", label="[2] Nelder-Mead (Table 1)")
    ax.set_yticks(range(len(names)), ticks)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlabel("z01 - z01 of [2] (mm); grey: guess +- 5 mm")
    ax.grid(axis="x", alpha=0.3)
    at.set_xscale("log")
    threads = []
    groups = ((("cnn",), "CNN"), (("b1_p4", "warm"), "padding 4"), (("b1_p2",), "padding 2"))
    for keys, label in groups:
        counts = sorted({rows[n].get(f"{k}_threads") for n in names for k in keys} - {None})
        if counts:
            threads.append(f"{label}: {'/'.join(str(c) for c in counts)}")
    at.set_xlabel(
        "time per autofocus (s)" + (f"\nCPU threads, {', '.join(threads)}" if threads else ""),
        fontsize=9,
    )
    at.grid(axis="x", alpha=0.3, which="both")
    _legend(fig, [ax, at], 4)
    _save(fig, out / "stage5-measured.png", dpi)
    return {name: rows[name] for name in names}


def _cdf(ax, values, **kwargs) -> None:
    v = np.sort(np.asarray(values, dtype=float))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", **kwargs)


def sim2real_figure(learned: Path, measured: Path, sim2real: Path, out: Path, dpi: int):
    rows = _comparison_rows(measured)
    names = [n for n in SAMPLES if n in rows and (learned / rows[n]["cnn_run"]).exists()]
    if not names:
        return None
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.1 * len(names) + 0.8, 3.6), sharey=True, layout="constrained"
    )
    axes = np.atleast_1d(axes)
    metrics = {}
    for ax, name in zip(axes, names, strict=True):
        row = rows[name]
        test = [
            abs(r["error_um"])
            for r in read_predictions(learned / row["cnn_run"] / "test" / "predictions.csv")
        ]
        _cdf(ax, test, color="tab:blue", label="matched test split")
        d2 = abs(row["cnn_minus_paper2_um"])
        rank = row["disagreement_paper2_percentile"]
        title = (
            f"{_short(name)}\n|CNN - [2]| = {d2:.0f} um, larger than\n"
            f"{_percent(rank)} % of the matched test errors"
        )
        combined = sim2real / name / "combined" / "predictions.csv"
        entry = {"test_median_abs_um": float(np.median(test))}
        if combined.exists():
            errors = np.array([abs(r["error_um"]) for r in read_predictions(combined)])
            _cdf(ax, errors, color="tab:green", label="test split, all measured properties")
            entry["combined_median_abs_um"] = float(np.median(errors))
            pct = float(100 * np.mean(errors < d2))
            entry["measured_percentile_in_combined"] = pct
            title += f",\n{_percent(pct)} % with all measured properties"
        ax.axvline(d2, color="black", label="measured: |CNN - [2]|")
        if "cnn_minus_b1_p4_um" in row:
            ax.axvline(
                abs(row["cnn_minus_b1_p4_um"]),
                color="tab:red",
                ls="--",
                label="measured: |CNN - NM padding 4|",
            )
        ax.axvline(100, color="0.6", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(10, 2e4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("|error| (um)")
        ax.grid(alpha=0.3, which="both")
        metrics[name] = entry
    axes[0].set_ylabel("fraction of samples")
    _legend(fig, axes, 2 * len(names))
    _save(fig, out / "stage5-sim2real.png", dpi)
    return metrics


def perturbation_figure(sim2real: Path, measured: Path, out: Path, dpi: int) -> dict | None:
    records = {n: _json(sim2real / f"{n}.json") for n in SAMPLES}
    records = {n: r for n, r in records.items() if r is not None}
    if not records:
        return None
    rows = _comparison_rows(measured)
    names = list(records)
    fig, axes = plt.subplots(
        2,
        len(names),
        figsize=(3.2 * len(names) + 1.4, 6.0),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    perts = [p for p in PERTURBATIONS if any(p in r["perturbations"] for r in records.values())]
    metrics = {}
    for j, name in enumerate(names):
        rec = records[name]
        present = [p for p in perts if p in rec["perturbations"]]
        y = np.array([perts.index(p) for p in present])
        shift = [rec["perturbations"][p]["paired"]["shift_um"] for p in present]
        mean = np.array([s["mean"] for s in shift])
        ci = np.array([s["mean_ci"] for s in shift])
        ax = axes[0, j]
        ax.axvline(0, color="0.6", lw=0.8)
        ax.errorbar(
            mean,
            y,
            xerr=[mean - ci[:, 0], ci[:, 1] - mean],
            fmt="o",
            color="black",
            capsize=3,
            label="mean shift (95 % CI)",
        )
        ax.plot(
            [s["median"] for s in shift],
            y,
            "|",
            color="tab:green",
            ms=12,
            mew=2,
            label="median shift",
        )
        if name in rows:
            ax.axvline(
                rows[name]["cnn_minus_paper2_um"],
                color="tab:blue",
                ls="--",
                label="measured: CNN - [2]",
            )
            if "cnn_minus_b1_p4_um" in rows[name]:
                ax.axvline(
                    rows[name]["cnn_minus_b1_p4_um"],
                    color="tab:red",
                    ls=":",
                    label="measured: CNN - NM padding 4",
                )
        ax.set_title(f"{_short(name)} (n = {rec['reference']['n']})", fontsize=9)
        ax.set_xlabel("estimate shift (um)")
        ax.grid(axis="x", alpha=0.3)
        mae = [rec["perturbations"][p]["error_um"]["mae"] for p in present]
        bx = axes[1, j]
        bx.barh(y, mae, color="tab:green", alpha=0.6, label="MAE with the change")
        bx.axvline(
            rec["reference"]["error_um"]["mae"], color="tab:blue", label="MAE, matched test split"
        )
        bx.axvline(GUESS_MAE_UM, color="0.5", ls=":", label="MAE of the guess")
        bx.set_xlabel("MAE (um)")
        bx.grid(axis="x", alpha=0.3)
        metrics[name] = {
            "reference_mae_um": rec["reference"]["error_um"]["mae"],
            "n": rec["reference"]["n"],
            **{
                p: {
                    "shift_mean_um": rec["perturbations"][p]["paired"]["shift_um"]["mean"],
                    "shift_mean_ci_um": rec["perturbations"][p]["paired"]["shift_um"]["mean_ci"],
                    "shift_median_um": rec["perturbations"][p]["paired"]["shift_um"]["median"],
                    "mae_um": rec["perturbations"][p]["error_um"]["mae"],
                    "within_100": rec["perturbations"][p]["error_um"]["within"]["100"],
                    "coverage_68": rec["perturbations"][p]["interval"]["coverage"]["68"],
                }
                for p in present
            },
        }
    for row_axes in axes:
        row_axes[0].set_yticks(range(len(perts)), perts)
    axes[0, 0].set_ylim(len(perts) - 0.5, -0.5)
    _legend(fig, axes, 2 * len(names))
    _save(fig, out / "stage5-perturbations.png", dpi)
    return metrics


def _abs_errors(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.array([abs(r["error_um"]) for r in read_predictions(path)])


def realistic_figure(learned: Path, measured: Path, sim2real: Path, out: Path, dpi: int):
    """The model of the realistic set against the model of the matched set, per sample."""
    models = {}
    for name in SAMPLES:
        pair = (_json(sim2real / f"{name}.json"), _json(sim2real / "realistic" / f"{name}.json"))
        if None not in pair:
            models[name] = pair
    if not models:
        return None
    fig, axes = plt.subplots(
        2,
        len(models),
        figsize=(3.6 * len(models) + 1.4, 6.6),
        layout="constrained",
        squeeze=False,
    )
    rows = ["none", *PERTURBATIONS]
    metrics = {}
    for j, (name, records) in enumerate(models.items()):
        cnn = _json(measured / "cnn" / f"{name}.json") or {"variants": {}}
        ax, bx = axes[0, j], axes[1, j]
        entry = {}
        for k, (rec, base, color) in enumerate(
            zip(records, (sim2real, sim2real / "realistic"), ("tab:blue", "tab:green"), strict=True)
        ):
            run = Path(rec["model"]).name
            label = run.removeprefix(f"matched-{name}").removeprefix("-") or "std"
            reference = rec["reference"].get("path")
            curves = {
                "matched test split": (
                    REPO / reference if reference else learned / run / "test" / "predictions.csv",
                    "-",
                ),
                "all measured properties": (base / name / "combined" / "predictions.csv", "--"),
            }
            if run.endswith(REALISTIC):
                curves["own test split"] = (learned / run / "test" / "predictions.csv", ":")
            summary = {}
            for data, (path, style) in curves.items():
                errors = _abs_errors(path)
                if errors is None:
                    continue
                _cdf(ax, errors, color=color, ls=style, label=f"{label}: {data}")
                summary[data] = {
                    "median_abs_um": float(np.median(errors)),
                    "mae_um": float(errors.mean()),
                    "within_100": float(np.mean(errors <= 100)),
                }
            variant = cnn["variants"].get(run)
            if variant is not None:
                d2 = abs(variant["references"]["paper2_nm"]["distance_um"])
                ax.axvline(d2, color=color, lw=1.5, label=f"{label}: measured |CNN - [2]|")
                summary["measured_minus_paper2_um"] = variant["references"]["paper2_nm"][
                    "distance_um"
                ]
                combined = _abs_errors(base / name / "combined" / "predictions.csv")
                if combined is not None:
                    summary["measured_percentile_in_combined"] = float(100 * np.mean(combined < d2))
            mae = [rec["reference"]["error_um"]["mae"]] + [
                _get(rec, "perturbations", p, "error_um", "mae") or np.nan for p in PERTURBATIONS
            ]
            bx.barh(
                np.arange(len(rows)) + (k - 0.5) * 0.38,
                mae,
                height=0.36,
                color=color,
                alpha=0.7,
                label=f"{label}: MAE",
            )
            summary["mae_um"] = dict(zip(rows, (float(m) for m in mae), strict=True))
            entry[label] = {"run": run, **summary}
        ax.axvline(100, color="0.6", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(10, 2e4)
        ax.set_title(_short(name), fontsize=9)
        ax.set_xlabel("|error| (um)")
        ax.grid(alpha=0.3, which="both")
        bx.axvline(GUESS_MAE_UM, color="0.5", ls=":", label="MAE of the guess")
        bx.set_yticks(range(len(rows)), rows)
        bx.set_ylim(len(rows) - 0.5, -0.5)
        bx.set_xlabel("MAE on the perturbed test split (um)")
        bx.grid(axis="x", alpha=0.3)
        metrics[name] = entry
    axes[0, 0].set_ylabel("fraction of samples")
    _legend(fig, axes, 2 * len(models))
    _save(fig, out / "stage5-realistic.png", dpi)
    return metrics


def spectra_figure(sim2real: Path, out: Path, dpi: int) -> dict | None:
    records = {n: _json(sim2real / f"spectra-{n}.json") for n in SAMPLES}
    records = {n: r for n, r in records.items() if r is not None}
    if not records:
        return None
    names = list(records)
    fig, axes = plt.subplots(
        2,
        len(names),
        figsize=(3.4 * len(names) + 0.8, 6.6),
        height_ratios=[1.25, 1],
        layout="constrained",
        squeeze=False,
    )
    metrics = {}
    for j, name in enumerate(names):
        rec = records[name]
        measured = np.asarray(rec["measured_input"])
        f2 = (np.arange(len(measured)) + 0.5) * rec["input_f2_max"] / len(measured)
        ax = axes[0, j]
        matched = rec["sets"].get("matched")
        if matched is not None:
            q = matched["input"]
            ax.fill_between(
                f2,
                q["p05"],
                q["p95"],
                color="tab:blue",
                alpha=0.25,
                lw=0,
                label="matched test split, 5-95 %",
            )
            ax.plot(f2, q["p50"], color="tab:blue", lw=1, label="matched test split, median")
        if "combined" in rec["sets"]:
            ax.plot(
                f2,
                rec["sets"]["combined"]["input"]["p50"],
                color="tab:green",
                lw=1,
                label="all measured properties, median",
            )
        ax.plot(f2, measured, color="black", lw=1, label="measured")
        fr = rec["fresnel_number_paper2_nm"]
        if rec["input_f2_max"] / fr <= 30:
            for k in range(1, int(rec["input_f2_max"] / fr) + 1):
                ax.axvline(k * fr, color="0.8", lw=0.5, zorder=0)
        ax.set_title(f"{_short(name)}: CNN input", fontsize=9)
        ax.set_xlabel("f^2 (cycles^2 per binned pixel^2)")
        ax.grid(alpha=0.2)
        sets = list(rec["sets"])
        ranks = np.array(
            [[rec["sets"][s]["measured_percentile"][f] for f in FEATURE_NAMES] for s in sets]
        )
        bx = axes[1, j]
        bx.imshow(ranks, cmap="RdBu_r", vmin=0, vmax=100, aspect="auto")
        for (r, c), v in np.ndenumerate(ranks):
            bx.text(
                c,
                r,
                _percent(v),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(v - 50) > 35 else "black",
            )
        bx.set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=30, ha="right", fontsize=8)
        bx.set_yticks(range(len(sets)), sets, fontsize=8)
        bx.set_title("percentile of the measured hologram", fontsize=9)
        metrics[name] = {
            "measured": {f: rec["measured"][f] for f in FEATURE_NAMES},
            "measured_ctf_bands": rec["measured"]["ctf_bands"],
            "percentiles": {s: rec["sets"][s]["measured_percentile"] for s in sets},
            "ctf_bands_median": {s: rec["sets"][s]["ctf_bands_median"] for s in sets},
        }
    axes[0, 0].set_ylabel("standardized log power")
    _legend(fig, axes[0], 2 * len(names))
    _save(fig, out / "stage5-spectra.png", dpi)
    return metrics


def noise_figure(sim2real: Path, out: Path, dpi: int) -> dict | None:
    record = _json(sim2real / "noise.json")
    if record is None:
        return None
    names = [n for n in SAMPLES if n in record["samples"]]
    fig, axes = plt.subplots(
        2, len(names), figsize=(3.2 * len(names) + 0.6, 6.0), layout="constrained", squeeze=False
    )
    metrics = {}
    for j, name in enumerate(names):
        rec = record["samples"][name]
        for row, (key, model, title) in enumerate(
            (
                ("full_resolution", blurred_white_spectrum, "full resolution"),
                ("binned", lorentzian_spectrum, f"{record['binning']}x grid"),
            )
        ):
            ax = axes[row, j]
            for corner, color in zip(rec[key], CORNER_COLORS, strict=True):
                f = np.asarray(corner["f"])
                ax.plot(f, corner["nps"], ".", color=color, ms=4)
                fit = corner["fit"]
                label = (
                    f"sigma {fit['sigma']:.2f} px"
                    if key == "full_resolution"
                    else f"{100 * fit['rho']:.1f} %, f_c {fit['cutoff']:.2f}"
                )
                ax.plot(f, model(f, fit), color=color, lw=1, label=label)
            ax.set_yscale("log")
            ax.set_title(f"{_short(name)}, {title}", fontsize=9)
            ax.set_xlabel("f (cycles per pixel)")
            ax.legend(fontsize=7, title="corner fits", title_fontsize=7)
            ax.grid(alpha=0.3, which="both")
            if j == 0:
                ax.set_ylabel("NPS (relative to the corner mean)")
        metrics[name] = {
            "psf_sigma_median": rec["psf_sigma_median"],
            "flat_field_rho_median": rec["flat_field_rho_median"],
            "flat_field_cutoff_median": rec["flat_field_cutoff_median"],
            "mean_spectrum_fit": rec.get("mean_spectrum_fit"),
            "corner_means": rec["corner_means"],
        }
    _save(fig, out / "stage5-noise.png", dpi)
    return metrics


def holograms_figure(holograms: Path | None, data: Path, sets: Path, out: Path, dpi: int):
    if holograms is None:
        return None
    names = [n for n in SAMPLES if (holograms / f"{n}.tiff").exists()]
    if not names:
        return None
    columns = (
        "measured",
        "simulated (matched test, sample 0)",
        "same sample, all measured properties",
    )
    fig, axes = plt.subplots(
        len(names), 3, figsize=(9.6, 3.1 * len(names)), layout="constrained", squeeze=False
    )
    metrics = {}
    for i, name in enumerate(names):
        measured = load_measured(holograms, MEASURED_SAMPLES[name], 4).numpy()
        images = [measured]
        for path in (data / f"matched-{name}" / "test.h5", sets / name / "combined.h5"):
            if path.exists():
                with h5py.File(path, "r") as fh:
                    images.append(np.asarray(fh["hologram"][0]))
            else:
                images.append(None)
        lo, hi = np.percentile(measured, [1, 99])
        for j, image in enumerate(images):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if image is None:
                ax.set_axis_off()
                continue
            ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(
                f"{_short(name)}: {columns[j]}\nstd/mean {image.std() / image.mean():.3f}",
                fontsize=8,
            )
        metrics[name] = {
            "contrast": [None if im is None else float(im.std() / im.mean()) for im in images]
        }
    _save(fig, out / "stage5-holograms.png", dpi)
    return metrics


def training_figure(learned: Path, out: Path, dpi: int) -> dict | None:
    runs = {
        n: [
            learned / f"matched-{n}{s}"
            for s in (*VARIANTS, REALISTIC)
            if (learned / f"matched-{n}{s}" / "history.csv").exists()
        ]
        for n in SAMPLES
    }
    runs = {n: r for n, r in runs.items() if r}
    if not runs:
        return None
    fig, axes = plt.subplots(
        1, len(runs), figsize=(3.1 * len(runs) + 0.8, 3.4), sharey=True, layout="constrained"
    )
    axes = np.atleast_1d(axes)
    metrics = {}
    for ax, (name, paths) in zip(axes, runs.items(), strict=True):
        for path in paths:
            realistic = path.name.endswith(REALISTIC)
            style, color = (":", "tab:green") if realistic else ("-", "tab:blue")
            if path.name.endswith("-hf"):
                style = "--"
            history = read_history(path / "history.csv")
            epochs = [h["epoch"] for h in history]
            mae = [h["val_mae_um"] for h in history]
            summary = _json(path / "summary.json") or {}
            best = summary.get("best_epoch")
            label = path.name.removeprefix(f"matched-{name}").removeprefix("-") or "std"
            if realistic:
                label += " (own validation split)"
            ax.plot(epochs, mae, style, color=color, lw=1, label=label)
            if best is not None:
                ax.plot(best, mae[epochs.index(best)], "o", color=color, mfc="white")
            metrics[path.name] = {
                "best_epoch": best,
                "epochs_run": summary.get("epochs_run"),
                "validation_mae_um": _get(summary, "validation", "error_um", "mae"),
                "validation_within_100": _get(summary, "validation", "error_um", "within", "100"),
                "train_seconds": summary.get("train_seconds"),
            }
        ax.axhline(GUESS_MAE_UM, color="0.5", ls=":", label="MAE of the guess")
        ax.set_title(_short(name), fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("validation MAE (um)")
    _save(fig, out / "stage5-training.png", dpi)
    return metrics


def _get(record, *keys):
    for key in keys:
        if not isinstance(record, dict) or key not in record:
            return None
        record = record[key]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="low resolution (CI)")
    parser.add_argument("--holograms", type=Path, help="directory of the measured holograms")
    parser.add_argument("--learned", type=Path, default=RESULTS / "learned")
    parser.add_argument("--measured", type=Path, default=RESULTS / "measured")
    parser.add_argument("--sim2real", type=Path, default=RESULTS / "sim2real")
    parser.add_argument("--data", type=Path, default=REPO / "data")
    parser.add_argument("--sets", type=Path, default=REPO / "data" / "sim2real")
    args = parser.parse_args()
    dpi = 60 if args.quick else 120
    args.out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"quick": args.quick}
    steps = [
        ("measured", lambda: measured_figure(args.measured, args.out, dpi)),
        (
            "sim2real",
            lambda: sim2real_figure(args.learned, args.measured, args.sim2real, args.out, dpi),
        ),
        ("perturbations", lambda: perturbation_figure(args.sim2real, args.measured, args.out, dpi)),
        (
            "realistic",
            lambda: realistic_figure(args.learned, args.measured, args.sim2real, args.out, dpi),
        ),
        ("spectra", lambda: spectra_figure(args.sim2real, args.out, dpi)),
        ("noise", lambda: noise_figure(args.sim2real, args.out, dpi)),
        (
            "holograms",
            lambda: holograms_figure(args.holograms, args.data, args.sets, args.out, dpi),
        ),
        ("training", lambda: training_figure(args.learned, args.out, dpi)),
    ]
    for name, fn in steps:
        start = time.perf_counter()
        value = fn()
        if value is None:
            print(f"{name}: inputs missing, skipped")
            continue
        metrics[name] = value
        print(f"{name}: {time.perf_counter() - start:.1f} s")
    with open(args.out / "stage5-metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
