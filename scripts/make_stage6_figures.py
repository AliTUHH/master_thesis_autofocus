"""Stage 6 figures for the lab notebook (``docs/laborbuch/figures``).

* ``stage6-regions.png``: CTF correlation of each measured hologram at ``Fr`` of [2] over the
  band of the spectrum CNN and per band, for the whole hologram and for the object, its edge and
  the background around it (``scripts/run_stage6_study.py regions``), with the control at
  1.3 ``Fr``.
* ``stage6-fit.png``: the objective of the texture fit of each measured hologram over the RMS of
  the texture inside the object and of the film, at the best slope of each cell of the coarse
  grid (red: its best cell); the title gives the best texture after the finer grid.
* ``stage6-spectra.png``: per sample the log power minus its running mean over one CTF period,
  each curve scaled to the same RMS, for the measured hologram and a typical sample of the
  Stage 5 and of the leave-one-out test split; below, the CTF correlation per band of the
  measured hologram against the 5-95 % range of every simulated set.
* ``stage6-measured.png``: the estimates of the leave-one-out CNN and of the Stage 5 CNN on the
  four measured holograms relative to [2] (Table 1) with their 68 and 95 % intervals, the
  leave-one-out CNN trained with other seeds, our padding-4 search, the visual value of [2] and
  the guess; right, the test MAE of both CNNs on both test splits.
* ``stage6-sim2real.png``: per sample the CDF of the errors of both CNNs on both test splits,
  against the disagreement of the leave-one-out CNN on the measured hologram with [2] and with
  the padding-4 search, and the distance of the guess to [2].
* ``stage6-training.png``: validation MAE per epoch of the leave-one-out and the Stage 5 models.
* ``stage6-holograms.png``: the measured holograms on the 4x grid next to the typical samples of
  the Stage 5 and of the leave-one-out test split, on the colour scale of the measured one.
* ``stage6-metrics.json``: the numbers quoted in the lab notebook.

Usage::

    python scripts/make_stage6_figures.py --holograms DIR    # results/ and data/
    python scripts/make_stage6_figures.py --quick --out /tmp/figures   # smoke test (CI)

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
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from holofocus.learned import read_history, read_predictions  # noqa: E402
from holofocus.learned.objectmodel import TEXTURE_GRID  # noqa: E402
from holofocus.learned.sim2real import CTF_BANDS  # noqa: E402
from holofocus.measured import MEASURED_SAMPLES, load_measured  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "laborbuch" / "figures"
RESULTS = REPO / "results"
SAMPLES = list(MEASURED_SAMPLES)
GUESS_MAE_UM = 2500.0
"""MAE of the guess itself (``dz01_est = 0``) for ``dz01`` uniform in +-5 mm."""
BAND_TICKS = ("CNN band\n0.002-0.125", *(f"{lo:g}-{hi:g}" for lo, hi in CTF_BANDS))
REGIONS = {
    "full": ("whole hologram", "0.25"),
    "object": ("object", "tab:red"),
    "edge": ("edge of the object (+-16 px)", "tab:purple"),
    "background": ("background", "tab:blue"),
}
SETS = {
    "matched": ("Stage 5 test split (compact objects)", "tab:blue"),
    "untextured": ("own detector and fiber, no texture", "0.55"),
    "fit": ("own detector and fiber, fitted texture", "tab:orange"),
    "loo": ("leave-one-out test split", "tab:green"),
}
METHODS = {
    "loo": ("CNN, leave-one-out (Stage 6)", "tab:green", "o"),
    "stage5": ("CNN, Stage 5", "tab:blue", "o"),
    "b1_p4": ("NM (ours), padding 4", "tab:red", "D"),
    "paper2_visual": ("[2] visual", "black", "x"),
    "paper1_example": ("example of [1]", "0.45", "+"),
    "guess": ("guess", "0.45", "v"),
}
TESTS = {
    "loo_test": ("leave-one-out CNN, leave-one-out test", "tab:green", "-", True),
    "loo_on_matched_test": ("leave-one-out CNN, Stage 5 test", "tab:green", ":", False),
    "stage5_on_loo_test": ("Stage 5 CNN, leave-one-out test", "tab:blue", "--", False),
    "stage5_test": ("Stage 5 CNN, Stage 5 test", "tab:blue", "-", True),
}


def _save(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _short(name: str) -> str:
    return MEASURED_SAMPLES[name].label


def _percent(x: float) -> str:
    """A percentile as text, with one decimal where rounding would give 0 or 100."""
    text = f"{x:.0f}"
    return f"{x:.1f}" if (text == "0" and x > 0) or (text == "100" and x < 100) else text


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


def _comparison(objectmodel: Path) -> dict[str, dict]:
    """Per sample the row of ``comparison.json`` and the paths of its estimates and runs."""
    comparison = _json(objectmodel / "comparison.json")
    return {} if comparison is None else comparison["samples"]


def _repo_path(path: str) -> Path:
    return REPO / path


def _abs_errors(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.array([abs(r["error_um"]) for r in read_predictions(path)])


def _cdf(ax, values, **kwargs) -> None:
    v = np.sort(np.asarray(values, dtype=float))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", **kwargs)


def regions_figure(objectmodel: Path, out: Path, dpi: int) -> dict | None:
    record = _json(objectmodel / "regions.json")
    if record is None:
        return None
    samples = record["samples"]
    names = [n for n in SAMPLES if n in samples]
    main, other = (f"{f:g}" for f in record["fractions"])
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.3 * len(names) + 0.8, 3.8), sharey=True, layout="constrained"
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(BAND_TICKS))
    width = 0.8 / len(REGIONS)
    metrics = {}
    for ax, name in zip(axes, names, strict=True):
        entry = samples[name]
        values = {"full": entry["full"], **entry["regions"][main]}
        for k, (region, (label, color)) in enumerate(REGIONS.items()):
            v = values[region]
            dx = (k - (len(REGIONS) - 1) / 2) * width
            ax.bar(x + dx, [v["ctf"], *v["ctf_bands"]], width, color=color, label=label)
            if region == "full":
                ax.plot(
                    x + dx,
                    [v["control_ctf"], *v["control_ctf_bands"]],
                    "_",
                    color="black",
                    ms=9,
                    mew=2,
                    label=f"whole hologram at {record['control_factor']:g} Fr (control)",
                )
            else:
                o = entry["regions"][other][region]
                ax.plot(
                    x + dx,
                    [o["ctf"], *o["ctf_bands"]],
                    ".",
                    color="black",
                    ms=4,
                    label=f"region with the core threshold {other} instead of {main}",
                )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, BAND_TICKS, fontsize=6.5)
        fr = entry["fresnel_number_paper2_nm"]
        ax.set_title(f"{_short(name)} (Fr of [2] {fr:.3g})", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        metrics[name] = {
            region: {k: values[region][k] for k in ("area", "ctf", "ctf_bands")}
            for region in REGIONS
        }
        metrics[name]["control_ctf"] = entry["full"]["control_ctf"]
    axes[0].set_ylabel("CTF correlation")
    for ax in axes:
        ax.set_xlabel("f^2 band (cycles^2 per binned pixel^2)", fontsize=8)
    _legend(fig, axes, 3)
    _save(fig, out / "stage6-regions.png", dpi)
    return metrics


def fit_figure(objectmodel: Path, out: Path, dpi: int) -> dict | None:
    records = {n: _json(objectmodel / f"fit-{n}.json") for n in SAMPLES}
    records = {n: r for n, r in records.items() if r is not None and "best" in r}
    if not records:
        return None
    rms_grid, film_grid = TEXTURE_GRID["rms"], TEXTURE_GRID["background_rms"]
    grids, metrics = {}, {}
    for name, rec in records.items():
        grid = np.full((len(film_grid), len(rms_grid)), np.inf)
        for e in rec["settings"]:
            t = e["texture"]
            if e["stage"] != "coarse":
                continue
            i, j = film_grid.index(t["background_rms"]), rms_grid.index(t["rms"])
            grid[i, j] = min(grid[i, j], e["objective"])
        grids[name] = grid
        untextured = grid[0, 0]
        metrics[name] = {
            "best": rec["best"],
            "best_coarse": rec["best_coarse"],
            "untextured_objective": float(untextured),
            "settings": len(rec["settings"]),
            "n": rec["n"],
            "seconds": float(sum(e["seconds"] for e in rec["settings"])),
            "measured": rec["measured"],
            "best_median": next(
                e["median"] for e in rec["settings"] if e["texture"] == rec["best"]["texture"]
            ),
        }
    finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    norm = LogNorm(vmin=max(finite.min(), 0.1), vmax=min(finite.max(), 10.0))
    fig, axes = plt.subplots(
        1, len(records), figsize=(3.1 * len(records) + 1.0, 3.7), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for ax, (name, rec) in zip(axes, records.items(), strict=True):
        grid = grids[name]
        image = ax.imshow(grid, origin="lower", cmap="viridis_r", norm=norm)
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white")
        bc = rec["best_coarse"]
        corner = (rms_grid.index(bc["rms"]) - 0.5, film_grid.index(bc["background_rms"]) - 0.5)
        ax.add_patch(Rectangle(corner, 1, 1, fill=False, ec="red", lw=2))
        ax.set_xticks(range(len(rms_grid)), [f"{v:g}" for v in rms_grid], fontsize=7)
        ax.set_yticks(range(len(film_grid)), [f"{v:g}" for v in film_grid], fontsize=7)
        ax.set_xlabel("texture RMS inside the object (rad)", fontsize=8)
        t, objective = rec["best"]["texture"], rec["best"]["objective"]
        ax.set_title(
            f"{_short(name)}: best object {t['rms']:g}, film {t['background_rms']:g} rad,\n"
            f"slope {t['slope']:g}; objective {objective:.2f}, "
            f"without texture {metrics[name]['untextured_objective']:.2f}",
            fontsize=8,
        )
    axes[0].set_ylabel("film RMS over the field of view (rad)", fontsize=8)
    bar = fig.colorbar(image, ax=axes, shrink=0.8, label="objective (best slope per cell)")
    ticks = [t for t in (0.2, 0.5, 1, 2, 5, 10) if norm.vmin <= t <= norm.vmax]
    bar.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
    bar.minorticks_off()
    _save(fig, out / "stage6-fit.png", dpi)
    return metrics


def _detrended(log_power: np.ndarray, f2: np.ndarray, fresnel_number: float) -> np.ndarray:
    """Log power minus its running mean over one CTF period, as in the CTF correlation."""
    period = max(3, int(round(fresnel_number / (f2[1] - f2[0]))))
    trend = np.convolve(log_power, np.ones(period) / period, mode="same")
    out = log_power - trend
    out[: period // 2 + 1] = np.nan
    out[len(f2) - period // 2 - 1 :] = np.nan
    return out


def spectra_figure(objectmodel: Path, out: Path, dpi: int) -> dict | None:
    records = {n: _json(objectmodel / f"spectra-{n}.json") for n in SAMPLES}
    records = {n: r for n, r in records.items() if r is not None}
    if not records:
        return None
    fig, axes = plt.subplots(
        2,
        len(records),
        figsize=(3.5 * len(records) + 0.8, 6.8),
        height_ratios=[1.2, 1],
        layout="constrained",
        squeeze=False,
    )
    x = np.arange(len(BAND_TICKS))
    metrics = {}
    for j, (name, rec) in enumerate(records.items()):
        f2 = np.asarray(rec["f2"])
        fr = rec["fresnel_number_paper2_nm"]
        ax = axes[0, j]
        curves = [("measured", np.asarray(rec["measured_log_power"]), fr, "black")]
        for set_name in ("loo", "matched"):
            s = rec["sets"].get(set_name)
            if s is not None:
                example = s["example"]
                curves.append(
                    (
                        f"{SETS[set_name][0]}, typical sample",
                        np.asarray(example["log_power"]),
                        example["fresnel_number"],
                        SETS[set_name][1],
                    )
                )
        for k, (label, log_power, fresnel_number, color) in enumerate(curves):
            detrended = _detrended(log_power, f2, fresnel_number)
            # the CTF correlation does not depend on the scale, so every curve gets the same RMS
            detrended = 0.12 * detrended / np.nanstd(detrended)
            ax.plot(f2, detrended - k, color=color, lw=0.6, label=label)
        if f2[-1] / fr <= 40:
            for k in range(1, int(f2[-1] / fr) + 1):
                ax.axvline(k * fr, color="0.85", lw=0.5, zorder=0)
        ax.set_xlim(0.002, f2[-1])
        ax.set_ylim(-len(curves) + 0.4, 0.6)
        ax.set_title(f"{_short(name)}: log power minus running mean, same RMS", fontsize=9)
        ax.set_xlabel("f^2 (cycles^2 per binned pixel^2); grey: f^2 = k Fr", fontsize=8)
        ax.set_yticks([])
        bx = axes[1, j]
        sets = [s for s in SETS if s in rec["sets"]]
        for k, set_name in enumerate(sets):
            s, (label, color) = rec["sets"][set_name], SETS[set_name]
            dx = (k - (len(sets) - 1) / 2) * 0.17
            ctf = s["features"]["ctf"]
            lo = [ctf["p05"], *s["ctf_bands"]["p05"]]
            hi = [ctf["p95"], *s["ctf_bands"]["p95"]]
            median = [ctf["median"], *s["ctf_bands"]["median"]]
            bx.vlines(x + dx, lo, hi, color=color, lw=3, alpha=0.55, label=f"{label}, 5-95 %")
            bx.plot(x + dx, median, "o", color=color, ms=3.5)
        m = rec["measured"]
        bx.plot(x, [m["ctf"], *m["ctf_bands"]], "D", color="black", ms=5, label="measured")
        bx.axhline(0, color="black", lw=0.8)
        bx.set_xticks(x, BAND_TICKS, fontsize=6.5)
        bx.set_ylim(-0.6, 1.0)
        bx.grid(axis="y", alpha=0.3)
        bx.set_title("CTF correlation", fontsize=9)
        metrics[name] = {
            "measured": {"ctf": m["ctf"], "ctf_bands": m["ctf_bands"]},
            "sets": {
                s: {
                    "n": rec["sets"][s]["n"],
                    "ctf_median": rec["sets"][s]["features"]["ctf"]["median"],
                    "ctf_bands_median": rec["sets"][s]["ctf_bands"]["median"],
                    "ctf_percentile": rec["sets"][s]["measured_percentile"]["ctf"],
                    "ctf_bands_percentile": rec["sets"][s]["ctf_bands_percentile"],
                    "measured_percentile": rec["sets"][s]["measured_percentile"],
                }
                for s in sets
            },
        }
    axes[1, 0].set_ylabel("correlation with sin^2(pi f^2 / Fr)")
    _legend(fig, axes, 3)
    _save(fig, out / "stage6-spectra.png", dpi)
    return metrics


def measured_figure(objectmodel: Path, out: Path, dpi: int) -> dict | None:
    details = _comparison(objectmodel)
    rows = {name: d["row"] for name, d in details.items()}
    names = [n for n in SAMPLES if n in rows]
    if not names:
        return None
    fig, (ax, at) = plt.subplots(
        1,
        2,
        figsize=(12, 1.6 + 1.45 * len(names)),
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
        stage5 = _json(_repo_path(details[name]["stage5"]))
        estimates = {
            "loo": _json(_repo_path(details[name]["loo"])),
            "stage5": stage5["variants"][stage5["selected"]] if stage5 else None,
        }
        for (key, estimate), dy in zip(estimates.items(), (-0.3, -0.1), strict=True):
            if estimate is None:
                continue
            color = METHODS[key][1]
            for level, lw in (("95", 1.2), ("68", 3.5)):
                i = estimate["intervals"][level]
                ax.plot([mm(i["lower"]), mm(i["upper"])], [y + dy] * 2, color=color, lw=lw)
            _point(ax, mm(estimate["z01_est"]), y + dy, key, zorder=3)
        for entry in details[name].get("seeds", [])[1:]:
            ax.plot(
                mm(entry["z01_est_mm"] * 1e-3),
                y - 0.3,
                "|",
                color="darkgreen",
                ms=13,
                mew=1.5,
                zorder=4,
                label="CNN, leave-one-out, other training seeds",
            )
        _point(ax, mm(row["b1_p4_mm"] * 1e-3), y + 0.15, "b1_p4")
        _point(ax, mm(sample.z01_visual), y + 0.02, "paper2_visual")
        _point(ax, mm(sample.z01_example), y + 0.02, "paper1_example")
        _point(ax, mm(sample.z01_guess), y + 0.32, "guess", filled=False)
        ticks.append(f"{sample.label}\n{sample.energy_kev:.0f} keV")
        for k, (prefix, (label, color, _, filled)) in enumerate(TESTS.items()):
            if f"{prefix}_mae_um" in row:
                at.plot(
                    row[f"{prefix}_mae_um"],
                    y - 0.3 + 0.2 * k,
                    "s",
                    color=color,
                    mfc=color if filled else "white",
                    ms=7,
                    ls="none",
                    label=label,
                )
    ax.axvline(0, color="black", lw=1, ls="--", label="[2] Nelder-Mead (Table 1)")
    ax.set_yticks(range(len(names)), ticks)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlabel("z01 - z01 of [2] (mm); bars 68 and 95 %; grey: guess +- 5 mm")
    ax.grid(axis="x", alpha=0.3)
    at.axvline(GUESS_MAE_UM, color="0.5", ls=":", label="MAE of the guess")
    at.set_xscale("log")
    maes = [row[f"{p}_mae_um"] for row in rows.values() for p in TESTS if f"{p}_mae_um" in row]
    lo, hi = min([*maes, GUESS_MAE_UM]) / 1.3, max([*maes, GUESS_MAE_UM]) * 1.3
    at.set_xlim(lo, hi)
    mae_ticks = [t for t in (50, 100, 200, 500, 1000, 2000, 5000, 10000) if lo <= t <= hi]
    at.set_xticks(mae_ticks, [str(t) for t in mae_ticks])
    at.xaxis.set_minor_formatter(NullFormatter())
    at.set_xlabel("test MAE (um)")
    at.grid(axis="x", alpha=0.3, which="both")
    _legend(fig, [ax, at], 4)
    _save(fig, out / "stage6-measured.png", dpi)
    return {name: rows[name] for name in names}


def sim2real_figure(objectmodel: Path, out: Path, dpi: int) -> dict | None:
    details = _comparison(objectmodel)
    names = [n for n in SAMPLES if n in details]
    if not names:
        return None
    fig, axes = plt.subplots(
        1, len(names), figsize=(3.2 * len(names) + 0.8, 3.9), sharey=True, layout="constrained"
    )
    axes = np.atleast_1d(axes)
    metrics = {}
    for ax, name in zip(axes, names, strict=True):
        row = details[name]["row"]
        entry = {}
        for key, path in details[name]["tests"].items():
            errors = _abs_errors(_repo_path(path) / "predictions.csv")
            if errors is None:
                continue
            label, color, style, _ = TESTS[key]
            _cdf(ax, errors, color=color, ls=style, lw=1.2, label=label)
            entry[f"{key}_median_abs_um"] = float(np.median(errors))
        d2 = abs(row["loo_minus_paper2_um"])
        ax.axvline(d2, color="black", label="measured: |leave-one-out CNN - [2]|")
        title = f"{_short(name)}\n|CNN - [2]| = {d2:.0f} um"
        if "disagreement_paper2_percentile" in row:
            title += f", above {_percent(row['disagreement_paper2_percentile'])} %"
        if "loo_minus_b1_p4_um" in row:
            ax.axvline(
                abs(row["loo_minus_b1_p4_um"]),
                color="tab:red",
                ls="--",
                label="measured: |leave-one-out CNN - NM padding 4|",
            )
        if "guess_minus_paper2_um" in row:
            ax.axvline(
                abs(row["guess_minus_paper2_um"]),
                color="0.45",
                ls="-.",
                lw=1,
                label="measured: |guess - [2]|",
            )
        ax.axvline(100, color="0.6", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(10, 2e4)
        ax.set_title(title + "\nof its test errors", fontsize=9)
        ax.set_xlabel("|error| (um)")
        ax.grid(alpha=0.3, which="both")
        metrics[name] = entry
    axes[0].set_ylabel("fraction of samples")
    _legend(fig, axes, 3)
    _save(fig, out / "stage6-sim2real.png", dpi)
    return metrics


def training_figure(objectmodel: Path, out: Path, dpi: int) -> dict | None:
    details = _comparison(objectmodel)
    runs = {}
    for name in SAMPLES:
        if name not in details:
            continue
        paths = [_repo_path(details[name][k]) for k in ("loo_run", "stage5_run")]
        paths = [p for p in paths if (p / "history.csv").exists()]
        if paths:
            runs[name] = paths
    if not runs:
        return None
    fig, axes = plt.subplots(
        1, len(runs), figsize=(3.1 * len(runs) + 0.8, 3.4), sharey=True, layout="constrained"
    )
    axes = np.atleast_1d(axes)
    metrics = {}
    for ax, (name, paths) in zip(axes, runs.items(), strict=True):
        for path in paths:
            loo = path.name.startswith("loo-")
            color = "tab:green" if loo else "tab:blue"
            history = read_history(path / "history.csv")
            epochs = [h["epoch"] for h in history]
            mae = [h["val_mae_um"] for h in history]
            summary = _json(path / "summary.json") or {}
            best = summary.get("best_epoch")
            label = "leave-one-out" if loo else f"Stage 5 ({path.name})"
            ax.plot(epochs, mae, color=color, lw=1, label=label)
            if best is not None:
                ax.plot(best, mae[epochs.index(best)], "o", color=color, mfc="white")
            validation = summary.get("validation", {}).get("error_um", {})
            metrics[path.name] = {
                "best_epoch": best,
                "epochs_run": summary.get("epochs_run"),
                "validation_mae_um": validation.get("mae"),
                "train_seconds": summary.get("train_seconds"),
            }
        ax.axhline(GUESS_MAE_UM, color="0.5", ls=":", label="MAE of the guess")
        ax.set_title(f"{_short(name)} (own validation splits)", fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("validation MAE (um)")
    _save(fig, out / "stage6-training.png", dpi)
    return metrics


def holograms_figure(holograms: Path | None, objectmodel: Path, data: Path, out: Path, dpi: int):
    if holograms is None:
        return None
    names = [n for n in SAMPLES if (holograms / f"{n}.tiff").exists()]
    if not names:
        return None
    fig, axes = plt.subplots(
        len(names), 3, figsize=(9.6, 3.1 * len(names)), layout="constrained", squeeze=False
    )
    metrics = {}
    for i, name in enumerate(names):
        measured = load_measured(holograms, MEASURED_SAMPLES[name], 4).numpy()
        spectra = _json(objectmodel / f"spectra-{name}.json") or {"sets": {}}
        panels = [("measured", measured)]
        for set_name, stem in (("matched", "matched"), ("loo", "loo")):
            path = data / f"{stem}-{name}" / "test.h5"
            index = spectra["sets"].get(set_name, {}).get("example", {}).get("index", 0)
            image = None
            if path.exists():
                with h5py.File(path, "r") as fh:
                    image = np.asarray(fh["hologram"][index])
            panels.append((f"{SETS[set_name][0]}, sample {index}", image))
        lo, hi = np.percentile(measured, [1, 99])
        for j, (title, image) in enumerate(panels):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if image is None:
                ax.set_axis_off()
                continue
            ax.imshow(image, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(
                f"{_short(name)}: {title}\nstd/mean {image.std() / image.mean():.3f}", fontsize=8
            )
        metrics[name] = {
            "contrast": [None if im is None else float(im.std() / im.mean()) for _, im in panels]
        }
    _save(fig, out / "stage6-holograms.png", dpi)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="low resolution (CI)")
    parser.add_argument("--holograms", type=Path, help="directory of the measured holograms")
    parser.add_argument(
        "--objectmodel",
        type=Path,
        default=RESULTS / "objectmodel",
        help="results of run_stage6_study.py; comparison.json names the estimates and runs",
    )
    parser.add_argument("--data", type=Path, default=REPO / "data")
    args = parser.parse_args()
    dpi = 60 if args.quick else 120
    args.out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"quick": args.quick}
    steps = [
        ("regions", lambda: regions_figure(args.objectmodel, args.out, dpi)),
        ("fit", lambda: fit_figure(args.objectmodel, args.out, dpi)),
        ("spectra", lambda: spectra_figure(args.objectmodel, args.out, dpi)),
        ("measured", lambda: measured_figure(args.objectmodel, args.out, dpi)),
        ("sim2real", lambda: sim2real_figure(args.objectmodel, args.out, dpi)),
        ("training", lambda: training_figure(args.objectmodel, args.out, dpi)),
        (
            "holograms",
            lambda: holograms_figure(args.holograms, args.objectmodel, args.data, args.out, dpi),
        ),
    ]
    for name, fn in steps:
        start = time.perf_counter()
        value = fn()
        if value is None:
            print(f"{name}: inputs missing, skipped")
            continue
        metrics[name] = value
        print(f"{name}: {time.perf_counter() - start:.1f} s")
    with open(args.out / "stage6-metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
