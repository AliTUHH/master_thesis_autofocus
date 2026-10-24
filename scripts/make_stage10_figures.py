"""Figures of Stage 10 (``docs/laborbuch/figures``).

* ``stage10-inputs.png``: the three inputs of one synthetic hologram (test sample 0 of
  cpu-4x): the hologram binned 2x of the image CNN, the polar and the radial log power
  spectrum.
* ``stage10-texture.png``: the radial log power of each measured hologram and the median of
  its fit set without texture, with the texture of Stage 6 and with that of Stage 10, and
  below their differences to the measured one, with the bands of the levels.
* ``stage10-spectra.png``: the 5-95 % band and the median of the radial log power of the
  leave-one-out test splits of Stages 6 and 10 with the measured hologram, and below the
  percentile of the measured hologram among each split, bin by bin.
* ``stage10-learned.png``: the share within 100 um, the MAE and the coverage of the 95 %
  interval of the four families (ensembles with 95 % CIs, seeds as points) on the synthetic
  samples and the leave-one-out test splits.
* ``stage10-measured.png``: the estimates and 95 % intervals of the ensembles and seeds on
  the measured holograms against [2] and Nelder-Mead cold.
* ``stage10-searches.png``: Brent in the 95 % intervals of the ensembles against Nelder-Mead
  and Brent cold: MAE and reconstructions on the synthetic samples, distance to [2] and
  reconstructions on the measured holograms.
* ``stage10-metrics.json``: the numbers quoted in the lab notebook.

Usage::

    python scripts/make_stage10_figures.py                              # results/
    python scripts/make_stage10_figures.py --quick --out /tmp/figures   # CI

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
import torch  # noqa: E402

from holofocus.dataset import DatasetConfig, HologramGenerator  # noqa: E402
from holofocus.learned.config import LearnedConfig  # noqa: E402
from holofocus.learned.representations import Representation  # noqa: E402
from holofocus.learned.sim2real import FEATURE_BANDS  # noqa: E402
from holofocus.measured import MEASURED_SAMPLES  # noqa: E402
from make_stage7_figures import _json, _legend, _save  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "laborbuch" / "figures"
STAGE10 = REPO / "results" / "stage10"
PROBES = tuple(MEASURED_SAMPLES)
FAMILIES = {
    "radial-cnn": ("CNN, radial spectrum", "tab:blue", "o"),
    "npe": ("NPE, radial spectrum", "tab:red", "D"),
    "polar-cnn": ("CNN, polar spectrum", "tab:green", "s"),
    "image-cnn": ("CNN, image", "tab:purple", "^"),
}
"""Label, colour and marker of every family of ``run_stage10_study.py``."""
TEXTURES = {
    "untextured": ("no texture", "0.55", ":"),
    "stage6": ("texture of Stage 6", "tab:orange", "--"),
    "stage10": ("texture of Stage 10 (with levels)", "tab:blue", "-"),
}
SPLITS = {"loo": ("Stage 6", "tab:orange"), "loo2": ("Stage 10", "tab:blue")}
LEVEL_BANDS = ("mid", "upper", "floor")
"""The bands of :data:`holofocus.learned.sim2real.FEATURE_BANDS` whose mean log power the fit
with levels compares."""
SETS = (
    "cpu-4x-test",
    "cpu-4x-baseline",
    "loo2-pooled",
    *(f"loo2-{name}" for name in PROBES),
)
SEARCHES = {
    "nm-cold": ("NM, guess ±5 mm", "black"),
    "brent-cold": ("Brent, guess ±5 mm", "0.5"),
    "brent-box95-cnn-stage9": ("Brent, 95 % CNN, Stage 9", "lightsteelblue"),
    "brent-box95-npe-stage9": ("Brent, 95 % NPE, Stage 9", "lightpink"),
    "brent-box95-cnn-stage8": ("Brent, 95 % CNN, Stage 8", "lightsteelblue"),
    "brent-box95-npe-stage8": ("Brent, 95 % NPE, Stage 8", "lightpink"),
    **{
        f"brent-box95-{f}": (f"Brent, 95 % {label}", color)
        for f, (label, color, _) in FAMILIES.items()
    },
}
"""Label and colour of every search of ``compare``: on the synthetic samples the references
of Stage 9 (the CNN of Stage 4, one seed; the five NPE seeds of Stage 7), on the measured
holograms the ensembles of Stage 8."""


def _set_label(name: str) -> str:
    if name.startswith("loo2-") and name[5:] in MEASURED_SAMPLES:
        return f"LOO {MEASURED_SAMPLES[name[5:]].label}"
    return {
        "cpu-4x-test": "cpu-4x test (250)",
        "cpu-4x-baseline": "cpu-4x, 48 of Stage 3",
        "loo2-pooled": "LOO, all four",
    }.get(name, name)


def _bands(ax, labels: bool) -> None:
    for name in LEVEL_BANDS:
        lo, hi = FEATURE_BANDS[name]
        ax.axvspan(lo, hi, color="0.85", alpha=0.5, lw=0)
        if labels:
            ax.text(
                (lo + hi) / 2,
                0.98,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6,
                color="0.35",
            )


# --- inputs --------------------------------------------------------------------------------


def inputs_figure(out: Path, dpi: int) -> dict | None:
    config = REPO / "configs" / "dataset" / "cpu-4x.yaml"
    if not config.exists():
        return None
    sample = HologramGenerator(DatasetConfig.from_yaml(config), device="cpu").sample("test", 0)
    hologram = torch.as_tensor(np.asarray(sample.hologram))
    inputs = {}
    for key, name in (
        ("image", "image-cnn"),
        ("polar", "polar-cnn"),
        ("radial", "spectrum-cnn-wide"),
    ):
        rep = LearnedConfig.from_yaml(REPO / "configs" / "learned" / f"{name}.yaml").representation
        inputs[key] = (rep, Representation(rep, tuple(hologram.shape))(hologram)[0].numpy())
    fig = plt.figure(figsize=(11, 4.3), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 2.3], height_ratios=[1, 1.2])
    ax = fig.add_subplot(grid[:, 0])
    rep, image = inputs["image"]
    ax.imshow(image[0], cmap="gray")
    ax.set_title(f"image CNN: binned {rep.binning}x, {image.shape[1]}x{image.shape[2]}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    rep, polar = inputs["polar"]
    ax = polar_ax = fig.add_subplot(grid[0, 1])
    step = 180 / rep.angles
    ax.imshow(
        polar[0],
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=(0, rep.f2_max, -step / 2, 180 - step / 2),
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(0, 180, 2 * step))
    ax.set_ylabel("direction (deg)", fontsize=8)
    ax.set_title(
        f"polar CNN: {rep.angles} sectors x {rep.bins} bins of f^2 (standardized log power)",
        fontsize=9,
    )
    ax.tick_params(labelsize=7, labelbottom=False)
    rep, radial = inputs["radial"]
    ax = fig.add_subplot(grid[1, 1], sharex=polar_ax)
    f2 = (np.arange(rep.bins) + 0.5) * rep.f2_max / rep.bins
    ax.plot(f2, radial[0], color="tab:blue", lw=0.6)
    ax.set_xlim(0, rep.f2_max)
    ax.set_xlabel("f^2 (cycles per binned pixel, squared)", fontsize=8)
    ax.set_ylabel("standardized log power", fontsize=8)
    ax.set_title(
        f"radial CNN and NPE: {rep.bins} bins of f^2, CTF minima every Fr = "
        f"{sample.labels['fresnel_number']:.5f}",
        fontsize=9,
    )
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    fig.suptitle(f"test sample 0 of cpu-4x, z01 = {sample.labels['z01'] * 1e3:.2f} mm", fontsize=10)
    _save(fig, out / "stage10-inputs.png", dpi)
    return {key: list(value.shape) for key, (_, value) in inputs.items()}


# --- texture and spectra -------------------------------------------------------------------


def texture_figure(stage10: Path, out: Path, dpi: int) -> dict | None:
    records = {n: _json(stage10 / f"texture-{n}.json") for n in PROBES}
    records = {n: r for n, r in records.items() if r is not None}
    if not records:
        return None
    fig, axes = plt.subplots(
        2, len(records), figsize=(3.6 * len(records), 6.2), sharex=True, layout="constrained"
    )
    axes = np.asarray(axes).reshape(2, len(records))
    metrics = {}
    for col, (name, r) in enumerate(records.items()):
        f2 = np.asarray(r["f2"])
        measured = np.asarray(r["measured_log_power"])
        top, bottom = axes[:, col]
        _bands(top, labels=True)
        _bands(bottom, labels=False)
        top.plot(f2, measured, color="black", lw=1.2, label="measured hologram")
        lines = []
        for key, (label, color, ls) in TEXTURES.items():
            s = r["settings"][key]
            power = np.asarray(s["log_power"])
            top.plot(f2, power, color=color, ls=ls, lw=1.0, label=label)
            bottom.plot(f2, power - measured, color=color, ls=ls, lw=1.0, label=label)
            lines.append(f"{label.split(' (')[0]}: {s['objective_levels']:.2f}")
        top.text(
            0.98,
            0.80,
            "objective with levels\n" + "\n".join(lines),
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.8, "lw": 0},
        )
        bottom.axhline(0, color="black", lw=0.8)
        top.set_title(MEASURED_SAMPLES[name].label, fontsize=9)
        bottom.set_xlabel("f^2 (cycles per binned pixel, squared)", fontsize=8)
        for ax in (top, bottom):
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        if col == 0:
            top.set_ylabel("radial log power (median of 16, nat)", fontsize=8)
            bottom.set_ylabel("simulated - measured (nat)", fontsize=8)
        metrics[name] = {
            "measured_excess_over_untextured": r["measured_excess_over_untextured"],
            **{
                key: {
                    k: r["settings"][key][k]
                    for k in (
                        "texture",
                        "objective",
                        "objective_levels",
                        "level_differences",
                        "power_to_measured",
                        "texture_power_to_measured_excess",
                    )
                }
                for key in TEXTURES
            },
        }
    _legend(fig, axes, columns=4)
    _save(fig, out / "stage10-texture.png", dpi)
    return metrics


def _inside_share(percentiles: np.ndarray, f2: np.ndarray, f2_max: float) -> float:
    """Share of the bins up to ``f2_max`` where the measured hologram lies between 5 and 95 %
    of a split."""
    keep = f2 <= f2_max
    return float(np.mean((percentiles[keep] >= 5) & (percentiles[keep] <= 95)))


def spectra_figure(stage10: Path, out: Path, dpi: int) -> dict | None:
    records = {n: _json(stage10 / f"spectra-{n}.json") for n in PROBES}
    records = {n: r for n, r in records.items() if r is not None and r["sets"]}
    if not records:
        return None
    fig, axes = plt.subplots(
        2,
        len(records),
        figsize=(3.6 * len(records), 6.2),
        sharex=True,
        layout="constrained",
        height_ratios=[1.4, 1],
    )
    axes = np.asarray(axes).reshape(2, len(records))
    metrics = {}
    for col, (name, r) in enumerate(records.items()):
        f2 = np.asarray(r["f2"])
        top, bottom = axes[:, col]
        _bands(top, labels=True)
        _bands(bottom, labels=False)
        metrics[name] = {}
        for key, (label, color) in SPLITS.items():
            s = r["sets"].get(key)
            if s is None:
                continue
            band = s["log_power"]
            top.fill_between(
                f2,
                band["p05"],
                band["p95"],
                color=color,
                alpha=0.25,
                lw=0,
                label=f"{label}: 5-95 %",
            )
            top.plot(f2, band["p50"], color=color, lw=0.8, ls="--", label=f"{label}: median")
            percentiles = np.asarray(s["measured_log_power_percentile"])
            bottom.plot(f2, percentiles, color=color, lw=0.9, label=f"{label}: measured among")
            metrics[name][key] = {
                "n": s["n"],
                "measured_percentile": s["measured_percentile"],
                "ctf_bands_percentile": s["ctf_bands_percentile"],
                "bins_inside_5_95_to_0.125": _inside_share(percentiles, f2, 0.125),
                "bins_inside_5_95": _inside_share(percentiles, f2, float(f2.max())),
            }
        top.plot(f2, r["measured_log_power"], color="black", lw=1.0, label="measured hologram")
        bottom.axhspan(5, 95, color="0.9", lw=0, zorder=0)
        bottom.set_ylim(-3, 103)
        top.set_title(MEASURED_SAMPLES[name].label, fontsize=9)
        bottom.set_xlabel("f^2 (cycles per binned pixel, squared)", fontsize=8)
        for ax in (top, bottom):
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        if col == 0:
            top.set_ylabel("radial log power (nat)", fontsize=8)
            bottom.set_ylabel("percentile of the measured (%)", fontsize=8)
    _legend(fig, axes, columns=4)
    _save(fig, out / "stage10-spectra.png", dpi)
    return metrics


# --- learned estimates ---------------------------------------------------------------------


def _metric(m: dict, key: str) -> tuple[float, float, float] | None:
    """Value and 95 % CI of a learned metric of ``compare``."""
    if key == "mae_um":
        lo, hi = m["mae_um_ci"]
        return m["mae_um"], lo, hi
    entry = m.get(key)
    if entry is None:
        return None
    lo, hi = entry["ci"]
    return entry["share"], lo, hi


def learned_figure(stage10: Path, out: Path, dpi: int) -> dict | None:
    record = _json(stage10 / "compare.json")
    if record is None:
        return None
    sets = [s for s in SETS if s in record["learned"]]
    if not sets:
        return None
    panels = (
        ("within_100um", "share within 100 um"),
        ("mae_um", "MAE (um)"),
        ("coverage_95", "coverage of the 95 % interval"),
    )
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 8.5), sharex=True, layout="constrained")
    width = 0.8 / len(FAMILIES)
    metrics = {}
    for ax, (key, ylabel) in zip(axes, panels, strict=True):
        for g, set_name in enumerate(sets):
            methods = record["learned"][set_name]["methods"]
            for k, (family, (label, color, marker)) in enumerate(FAMILIES.items()):
                x = g - 0.4 + width * (k + 0.5)
                ensemble = methods.get(f"{family}/ensemble")
                value = None if ensemble is None else _metric(ensemble, key)
                if value is not None:
                    v, lo, hi = value
                    ax.bar(x, v, width * 0.9, color=color, alpha=0.55, label=f"{label}: ensemble")
                    ax.errorbar(x, v, [[v - lo], [hi - v]], color="black", capsize=2, lw=0.8)
                seeds = [
                    _metric(m, key)
                    for method, m in methods.items()
                    if method.startswith(f"{family}/seed-")
                ]
                seeds = [s[0] for s in seeds if s is not None]
                if seeds:
                    ax.plot(
                        [x] * len(seeds),
                        seeds,
                        ls="none",
                        marker=marker,
                        ms=3.5,
                        mfc="white",
                        mec=color,
                        label=f"{label}: seeds",
                    )
        if key == "coverage_95":
            ax.axhline(0.95, color="black", lw=0.8, ls="--")
            ax.set_ylim(0.5, 1.02)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, axis="y")
    axes[-1].set_xticks(range(len(sets)), [_set_label(s) for s in sets], fontsize=7, rotation=15)
    for set_name in sets:
        methods = record["learned"][set_name]["methods"]
        metrics[set_name] = {
            method: {
                "n": m["n"],
                "within_100um": m["within_100um"]["share"],
                "mae_um": m["mae_um"],
                "coverage_95": None if "coverage_95" not in m else m["coverage_95"]["share"],
                "truth_outside_95": None
                if "coverage_95" not in m
                else m["coverage_95"]["n"] - m["coverage_95"]["k"],
            }
            for method, m in methods.items()
        }
    handles = {}
    for ax in axes:
        for h, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, h)
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncols=4, fontsize=7)
    _save(fig, out / "stage10-learned.png", dpi)
    return metrics


def _measured_methods(methods: dict) -> list[str]:
    """Methods of a measured hologram in the order of the figure: each family's ensemble, then
    its seeds, then the ensembles of Stage 8."""
    order = []
    for family in FAMILIES:
        order += [m for m in (f"{family}/ensemble",) if m in methods]
        order += sorted(m for m in methods if m.startswith(f"{family}/seed-"))
    return order + [m for m in methods if m.endswith("/stage8")]


def _measured_label(method: str) -> str:
    family, which = method.split("/")
    if which == "stage8":
        return f"{'CNN' if family == 'radial-cnn' else 'NPE'}, Stage 8 ensemble"
    return f"{FAMILIES[family][0]}, {'ensemble' if which == 'ensemble' else which}"


def measured_figure(stage10: Path, out: Path, dpi: int) -> dict | None:
    record = _json(stage10 / "compare.json")
    if record is None or not record["measured"]["samples"]:
        return None
    samples = record["measured"]["samples"]
    order = _measured_methods(next(iter(samples.values()))["methods"])
    fig, axes = plt.subplots(
        1,
        len(samples),
        figsize=(3.4 * len(samples) + 1.8, 0.28 * len(order) + 1.6),
        sharey=True,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    for ax, (name, s) in zip(axes, samples.items(), strict=True):
        paper2 = s["references"]["paper2_nm"]
        ax.axvspan(-0.1, 0.1, color="tab:green", alpha=0.12, lw=0, label="within 100 um of [2]")
        ax.axvline(0, color="black", lw=0.8, label="[2] Nelder-Mead")
        if s["references"].get("nm_cold") is not None:
            ax.axvline(
                (s["references"]["nm_cold"] - paper2) * 1e3,
                color="0.4",
                ls="-.",
                lw=0.8,
                label="NM, guess ±5 mm (b1-p4)",
            )
        for y, method in enumerate(order):
            e = s["methods"].get(method)
            if e is None:
                continue
            family, which = method.split("/")
            color = FAMILIES[family][1] if family in FAMILIES else "0.3"
            if which == "stage8":
                color = "0.3"
            ensemble = which in ("ensemble", "stage8")
            z = (e["z01_est"] - paper2) * 1e3
            ax.errorbar(
                z,
                y,
                xerr=[[z - (e["lower_95"] - paper2) * 1e3], [(e["upper_95"] - paper2) * 1e3 - z]],
                color=color,
                marker=FAMILIES.get(family, ("", "", "o"))[2] if which != "stage8" else "x",
                ms=5 if ensemble else 3.5,
                mfc=color if ensemble else "white",
                lw=1.6 if ensemble else 0.8,
                capsize=2,
            )
        ax.set_title(MEASURED_SAMPLES[name].label, fontsize=9)
        ax.set_xlabel("z01 - [2] (mm); bars: 95 % interval", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, axis="x")
    axes[0].set_yticks(range(len(order)), [_measured_label(m) for m in order], fontsize=7)
    axes[0].invert_yaxis()
    _legend(fig, axes, columns=3)
    _save(fig, out / "stage10-measured.png", dpi)
    return record["measured"]["summaries"]


# --- searches ------------------------------------------------------------------------------


def searches_figure(stage10: Path, out: Path, dpi: int) -> dict | None:
    record = _json(stage10 / "compare.json")
    if record is None:
        return None
    synthetic = record["searches"]["synthetic"]["summaries"]
    measured = record["searches"]["measured"]
    if not synthetic and not measured["samples"]:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    shown = [m for m in SEARCHES if m in synthetic]
    for ax, key, ylabel in (
        (axes[0, 0], "mae_um", "MAE (um, 95 % CI)"),
        (axes[0, 1], "reconstructions", "reconstructions per autofocus (mean, 95 % CI)"),
    ):
        for k, method in enumerate(shown):
            s = synthetic[method]
            label, color = SEARCHES[method]
            lo, hi = s[f"{key}_ci"]
            ax.bar(k, s[key], color=color, edgecolor="black", lw=0.5)
            ax.errorbar(k, s[key], [[s[key] - lo], [hi - s[key]]], color="black", capsize=3)
            if key == "reconstructions" and "truth_outside" in s:
                ax.text(
                    k,
                    hi,
                    f"{len(s['truth_outside'])}/{s['n_samples']}\noutside",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
        ax.set_xticks(
            range(len(shown)), [SEARCHES[m][0] for m in shown], fontsize=7, rotation=30, ha="right"
        )
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title("48 synthetic samples of Stage 3", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, axis="y")
    samples = measured["samples"]
    methods = [m for m in SEARCHES if any(m in s["searches"] for s in samples.values())]
    width = 0.8 / max(1, len(methods))
    for ax, key, ylabel in (
        (axes[1, 0], "minus_paper2_nm_um", "z01 found - [2] (um)"),
        (axes[1, 1], "n_reconstructions", "reconstructions"),
    ):
        for g, s in enumerate(samples.values()):
            for k, method in enumerate(methods):
                e = s["searches"].get(method)
                if e is None:
                    continue
                label, color = SEARCHES[method]
                ax.bar(
                    g - 0.4 + width * (k + 0.5),
                    e[key],
                    width * 0.9,
                    color=color,
                    edgecolor="black",
                    lw=0.5,
                    label=label,
                )
        if key == "minus_paper2_nm_um":
            ax.axhspan(-100, 100, color="tab:green", alpha=0.12, lw=0, zorder=0)
            ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(samples)), [MEASURED_SAMPLES[n].label for n in samples], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title("measured holograms, binning 1", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, axis="y")
    handles = dict(zip(*reversed(axes[1, 1].get_legend_handles_labels()), strict=True))
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncols=4, fontsize=7)
    _save(fig, out / "stage10-searches.png", dpi)
    return {
        "synthetic": synthetic,
        "measured": measured["totals"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="low resolution (CI)")
    parser.add_argument("--stage10", type=Path, default=STAGE10)
    args = parser.parse_args()
    dpi = 60 if args.quick else 120
    args.out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"quick": args.quick}
    steps = [
        ("inputs", lambda: inputs_figure(args.out, dpi)),
        ("texture", lambda: texture_figure(args.stage10, args.out, dpi)),
        ("spectra", lambda: spectra_figure(args.stage10, args.out, dpi)),
        ("learned", lambda: learned_figure(args.stage10, args.out, dpi)),
        ("measured", lambda: measured_figure(args.stage10, args.out, dpi)),
        ("searches", lambda: searches_figure(args.stage10, args.out, dpi)),
    ]
    for name, fn in steps:
        start = time.perf_counter()
        value = fn()
        if value is None:
            print(f"{name}: inputs missing, skipped")
            continue
        metrics[name] = value
        print(f"{name}: {time.perf_counter() - start:.1f} s")
    with open(args.out / "stage10-metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
