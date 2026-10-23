"""Figures of Stage 9 (``docs/laborbuch/figures``).

* ``stage9-tooth-step.png``: the MFE of the tooth against ``z01`` with both numerics (searches
  and scans), the step enlarged, and below the RMS distance of the phase of every scan to that
  of the scan nearest to the minimum and, for the scans with one thread, the MFE next to the
  loss at the returned object.
* ``stage9-tooth-objects.png``: the phase of the tooth on both sides of the step at the same
  ``z01`` (one thread, and 3 threads with the library of before this stage), their difference
  and that of two neighbours on the same side, in the same colours.
* ``stage9-tooth-loss.png``: the relative difference of the loss histories of two scans against
  the iteration: the same ``z01`` with both numerics, the closest neighbours across the switch
  of the first stage and on the same branch.
* ``stage9-first-stage.png``: the loss after the first stage of the plan, at the lookahead and
  at the returned object, against ``z01``: the tooth around its switches with both numerics,
  where the padded arrays on both sides of a switch differ, and all four probes over
  ``Omega_z``.
* ``stage9-probes.png``: the MFE of spider hair, cactus needle and Mg wire against ``z01``
  with both numerics.
* ``stage9-numerics.png``: the MFE at exactly the same ``z01`` with 3 threads before this stage
  against one thread, on the measured holograms and on the synthetic samples.
* ``stage9-measured.png``: the searches on the measured holograms, Nelder-Mead and Brent, cold
  and in the 95 % intervals of the ensembles: the reconstructed points in order.
* ``stage9-reconstructions.png``: reconstructions per autofocus on the synthetic samples and
  what the 95 % intervals save with Nelder-Mead and with Brent.
* ``stage9-savings-width.png``: the saving per sample against the width of the interval.
* ``stage9-errors.png``: the absolute errors of all searches on the synthetic samples.
* ``stage9-metrics.json``: the numbers quoted in the lab notebook.

Usage::

    python scripts/make_stage9_figures.py                              # results/
    python scripts/make_stage9_figures.py --quick --out /tmp/figures   # CI

Figures whose inputs are missing are skipped; those of the objects need the npz of the scans.
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
from matplotlib.ticker import MaxNLocator  # noqa: E402

from holofocus.measured import MEASURED_SAMPLES  # noqa: E402
from make_stage7_figures import _csv, _json, _legend, _save  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "laborbuch" / "figures"
RESULTS = REPO / "results"
STAGE9 = RESULTS / "stage9"
BEFORE, AFTER = "threads-3-4349d6a", "threads-3"
ONE, THREE = "tab:blue", "tab:red"
NUMERICS = {1: ("one thread (this stage)", ONE), 3: ("3 threads, before this stage", THREE)}
METHODS = {
    "nm-cold": ("NM, guess ±5 mm", "black", "o", "-"),
    "nm-cold-repeat": ("NM, guess ±5 mm, again", "0.5", "o", ":"),
    "nm-box95-cnn": ("NM, 95 % CNN", "tab:blue", "s", "-"),
    "nm-box95-npe": ("NM, 95 % NPE", "tab:red", "D", "-"),
    "brent-cold": ("Brent, guess ±5 mm", "tab:green", "o", "--"),
    "brent-box95-cnn": ("Brent, 95 % CNN", "tab:cyan", "s", "--"),
    "brent-box95-npe": ("Brent, 95 % NPE", "tab:pink", "D", "--"),
}
"""Label, colour, marker and line of every search of ``compare``; on the synthetic samples
the CNN is the single network of Stage 4, on the measured holograms the ensemble of Stage 8."""


def _points(stage9: Path) -> list[dict]:
    rows = _csv(stage9 / "mfe-points.csv") or []
    for row in rows:
        row["numerics"] = int(row["numerics"])
    return rows


def _scans(stage9: Path, sample: str) -> list[dict]:
    """The scan points of ``sample`` with both numerics, with ``numerics`` set; those with
    3 threads and the library of this stage have the bits of one thread and count where no
    scan with one thread has their ``z01``."""
    out = []
    for directory, n in ((stage9, 1), (stage9 / AFTER, 1), (stage9 / BEFORE, 3)):
        record = _json(directory / f"scan-{sample}.json")
        if record is not None:
            seen = {p["z01"] for p in out if p["numerics"] == n}
            out += [
                {**p, "numerics": n, "every": record["loss_every"]}
                for p in record["points"]
                if p["z01"] not in seen
            ]
    return sorted(out, key=lambda p: (p["z01"], p["numerics"]))


def _phase(point: dict) -> np.ndarray | None:
    path = REPO / point["arrays"]
    return np.load(path)["phase"].astype(np.float64) if path.exists() else None


def _mfe_panel(ax, rows: list[dict], sample: str, lo: float, hi: float, log: bool) -> None:
    for n, (label, color) in NUMERICS.items():
        sel = [r for r in rows if r["sample"] == sample and r["numerics"] == n]
        searched = [r for r in sel if not r["source"].startswith("scan") and lo <= r["z01"] <= hi]
        scanned = [r for r in sel if r["source"].startswith("scan") and lo <= r["z01"] <= hi]
        ax.plot(
            [r["z01"] * 1e3 for r in searched],
            [r["ppm"] for r in searched],
            "o" if n == 1 else "x",
            ms=4,
            mfc="none",
            color=color,
            label=f"searches, {label}",
        )
        if scanned:
            scanned.sort(key=lambda r: r["z01"])
            ax.plot(
                [r["z01"] * 1e3 for r in scanned],
                [r["ppm"] for r in scanned],
                "-o" if n == 1 else "D",
                ms=3.5,
                lw=0.8,
                color=color,
                label=f"scan, {label}",
            )
    if log:
        ax.set_yscale("symlog", linthresh=10.0)
    ax.set_xlim(lo * 1e3, hi * 1e3)
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.set_xlabel("z01 (mm)", fontsize=8)
    ax.set_ylabel("MFE above the smallest (ppm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def tooth_step_figure(stage9: Path, out: Path, dpi: int) -> dict | None:
    rows = _points(stage9)
    scans = _scans(stage9, "tooth")
    if not rows or not scans:
        return None
    steps = _json(stage9 / "steps.json") or {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    _mfe_panel(axes[0, 0], rows, "tooth", 0.0804, 0.0811, log=True)
    axes[0, 0].set_title("tooth, 80.4-81.1 mm", fontsize=9)
    ones = [p for p in scans if p["numerics"] == 1]
    jumps = steps.get("probes", {}).get("tooth", {}).get("1", {}).get("jumps", [])
    lo, hi = (jumps[0]["z01"][0], jumps[0]["z01"][1]) if jumps else (0.08069, 0.08075)
    _mfe_panel(axes[0, 1], rows, "tooth", lo - 25e-6, hi + 25e-6, log=False)
    axes[0, 1].set_title("around the step", fontsize=9)
    for ax in axes[0]:
        for pair in steps.get("same_z01", []):
            if pair["sample"] == "tooth":
                z = pair["z01"] * 1e3
                ax.plot([z, z], [pair["ppm_one_thread"], pair["ppm_3_threads"]], "k:", lw=0.8)
    reference = min(ones, key=lambda p: p["mfe"])
    ref_phase = _phase(reference)
    metrics = {"reference_z01": reference["z01"]}
    ax = axes[1, 0]
    if ref_phase is not None:
        for n, (label, color) in NUMERICS.items():
            sel = [p for p in scans if p["numerics"] == n]
            dist = [
                (p["z01"] * 1e3, 1e3 * np.sqrt(np.mean((phase - ref_phase) ** 2)))
                for p in sel
                if (phase := _phase(p)) is not None
            ]
            if dist:
                ax.plot(
                    *zip(*dist, strict=True),
                    "-o" if n == 1 else "D",
                    ms=3.5,
                    lw=0.8,
                    color=color,
                    label=f"scan, {label}",
                )
                metrics[f"phase_rms_mrad_numerics_{n}"] = dist
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.set_xlabel("z01 (mm)", fontsize=8)
        ax.set_ylabel(f"RMS phase to {reference['z01'] * 1e3:.3f} mm (mrad)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.set_title("phase of the scans against the scan nearest the minimum", fontsize=9)
    else:
        ax.set_axis_off()
    smallest = _losses_panel(axes[1, 1], scans)
    metrics.update({f"scan_smallest_{key}_z01": z for key, z in smallest.items()})
    _legend(fig, axes[0], columns=2)
    _save(fig, out / "stage9-tooth-step.png", dpi)
    return metrics


def _losses_panel(ax, scans: list[dict]) -> dict:
    """The MFE, the smallest loss of the last stage and the loss at the returned object of the
    full scans with one thread, each above its smallest; returns where each is smallest."""
    sel = [p for p in scans if p["numerics"] == 1 and "object_loss" in p]
    if len(sel) < 2:
        ax.set_axis_off()
        return {}
    series = {
        "mfe": ("MFE (last iteration, at the lookahead)", "-o", [p["mfe"] for p in sel]),
        "last_stage": (
            "smallest loss of the last stage",
            "--^",
            [min(p["loss"][p["stage_ends"][-2] // p["every"] :]) for p in sel],
        ),
        "object_loss": ("at the returned object", ":s", [p["object_loss"] for p in sel]),
    }
    smallest = {}
    for key, (label, style, values) in series.items():
        ppm = [(v / min(values) - 1) * 1e6 for v in values]
        ax.plot([p["z01"] * 1e3 for p in sel], ppm, style, ms=3.5, lw=0.8, color=ONE, label=label)
        smallest[key] = sel[int(np.argmin(values))]["z01"]
    ax.set_yscale("symlog", linthresh=10.0)
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.set_xlabel("z01 (mm)", fontsize=8)
    ax.set_ylabel("loss above its smallest (ppm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("scans, one thread: the MFE and the loss at the object", fontsize=9)
    return smallest


def tooth_objects_figure(stage9: Path, out: Path, dpi: int) -> dict | None:
    scans = _scans(stage9, "tooth")
    threes = [p for p in scans if p["numerics"] == 3 and _phase(p) is not None]
    ones = {p["z01"]: p for p in scans if p["numerics"] == 1 and _phase(p) is not None}
    pairs = [(ones[p["z01"]], p) for p in threes if p["z01"] in ones]
    if not pairs:
        return None
    low, high = max(pairs, key=lambda pair: abs(pair[1]["mfe"] / pair[0]["mfe"] - 1))
    if high["mfe"] < low["mfe"]:
        low, high = high, low
    side = sorted(
        (p for p in ones.values() if p is not low and abs(p["mfe"] / low["mfe"] - 1) < 2e-4),
        key=lambda p: abs(p["z01"] - low["z01"]),
    )
    a, b = _phase(low), _phase(high)
    diff = b - a
    panels = [
        (a, f"{low['z01'] * 1e3:.5f} mm, {NUMERICS[low['numerics']][0]}", "gray", None),
        (b, f"{high['z01'] * 1e3:.5f} mm, {NUMERICS[high['numerics']][0]}", "gray", None),
    ]
    limit = float(np.percentile(np.abs(diff), 99.5))
    panels.append((diff, "difference (rad)", "RdBu_r", limit))
    metrics = {
        "low": {"z01": low["z01"], "numerics": low["numerics"], "mfe": low["mfe"]},
        "high": {"z01": high["z01"], "numerics": high["numerics"], "mfe": high["mfe"]},
        "mfe_relative_ppm": (high["mfe"] / low["mfe"] - 1) * 1e6,
        "phase_rms_difference_rad": float(np.sqrt(np.mean(diff**2))),
        "phase_max_abs_difference_rad": float(np.abs(diff).max()),
        "phase_std_rad": float(a.std()),
    }
    if side:
        near = side[0]
        same = _phase(near) - a
        panels.append(
            (same, f"{near['z01'] * 1e3:.4f} minus {low['z01'] * 1e3:.4f} mm", "RdBu_r", limit)
        )
        metrics["same_side"] = {
            "z01": near["z01"],
            "phase_rms_difference_rad": float(np.sqrt(np.mean(same**2))),
        }
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.4), layout="constrained")
    vmin, vmax = np.percentile(a, [0.5, 99.5])
    for ax, (image, title, cmap, lim) in zip(axes, panels, strict=True):
        if lim is None:
            shown = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            shown = ax.imshow(image, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=8)
        ax.set_axis_off()
        fig.colorbar(shown, ax=ax, shrink=0.7)
    _save(fig, out / "stage9-tooth-objects.png", dpi)
    return metrics


def tooth_loss_figure(stage9: Path, out: Path, dpi: int) -> dict | None:
    steps = _json(stage9 / "steps.json")
    scans = _scans(stage9, "tooth")
    if steps is None or not scans:
        return None
    by_key = {(p["z01"], p["numerics"]): p for p in scans}
    switch = steps.get("first_stage_switch", 0.01)
    chosen = []
    pairs = [p for p in steps["scans"] if p["sample"] == "tooth"]
    for p in pairs:
        if p["kind"] == "same z01":
            chosen.append((p, f"same z01 {p['z01'][0] * 1e3:.4f} mm: 3 threads before / one"))

    def switched(p):
        a, b = (by_key[z, n] for z, n in zip(p["z01"], p["numerics"], strict=True))
        return abs(b["stage_losses"][0] / a["stage_losses"][0] - 1.0) > switch

    def spacing_nm(p):
        return round(abs(p["z01"][1] - p["z01"][0]) * 1e9)

    neighbours = [p for p in pairs if p["kind"] == "neighbours" and p["numerics"] == [1, 1]]
    across = [p for p in neighbours if switched(p)]
    if across:
        nearest = min(across, key=spacing_nm)
        chosen.append(
            (
                nearest,
                f"across the switch: {nearest['z01'][1] * 1e3:.3f} / "
                f"{nearest['z01'][0] * 1e3:.3f} mm",
            )
        )
        centre = sum(nearest["z01"]) / 2
        same = [p for p in neighbours if not switched(p)]
        if same:
            near = min(same, key=lambda p: (spacing_nm(p), abs(sum(p["z01"]) / 2 - centre)))
            chosen.append(
                (
                    near,
                    f"same branch: {near['z01'][1] * 1e3:.3f} / {near['z01'][0] * 1e3:.3f} mm",
                )
            )
    if not chosen:
        return None
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    metrics = {}
    for p, label in chosen:
        a = by_key[p["z01"][0], p["numerics"][0]]
        b = by_key[p["z01"][1], p["numerics"][1]]
        rel = np.abs(np.asarray(b["loss"]) / np.asarray(a["loss"]) - 1.0)
        iterations = np.arange(1, len(rel) + 1) * a["every"]
        ax.semilogy(iterations, np.maximum(rel, 1e-12), lw=1, label=label)
        metrics[label] = p["loss"]
    for end in scans[0]["stage_ends"][:-1]:
        ax.axvline(end, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("iteration", fontsize=8)
    ax.set_ylabel("|loss b / loss a - 1|", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, out / "stage9-tooth-loss.png", dpi)
    return metrics


def _first_stage(stage9: Path, sample: str) -> tuple[list[dict], dict | None]:
    """The scan points of the first stage of ``sample`` with both numerics, and the record of
    the one-thread scan (for its grid)."""
    out, header = [], None
    for directory, n in ((stage9 / "first-stage", 1), (stage9 / "first-stage" / BEFORE, 3)):
        record = _json(directory / f"scan-{sample}.json")
        if record is not None:
            out += [{**p, "numerics": n} for p in record["points"]]
            header = header or record
    return sorted(out, key=lambda p: (p["z01"], p["numerics"])), header


def _first_stage_panel(ax, points: list[dict], lo: float, hi: float, smallest: dict) -> None:
    """Both losses after the first stage, each relative to its smallest with one thread."""
    for n, (label, color) in NUMERICS.items():
        sel = [p for p in points if p["numerics"] == n and lo <= p["z01"] <= hi]
        if not sel:
            continue
        z = [p["z01"] * 1e3 for p in sel]
        for key, style, what in (("mfe", "-o", "lookahead"), ("object_loss", ":s", "object")):
            ax.plot(
                z,
                [100 * (p[key] / smallest[key] - 1) for p in sel],
                style if n == 1 else ("D" if key == "mfe" else "x"),
                ms=2.5,
                lw=0.8,
                color=color,
                alpha=1.0 if key == "mfe" else 0.6,
                label=f"at the {what}, {label}",
            )
    ax.set_xlim(lo * 1e3, hi * 1e3)
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.set_xlabel("z01 (mm)", fontsize=8)
    ax.set_ylabel("loss after stage 1, above its smallest (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def first_stage_figure(stage9: Path, out: Path, dpi: int) -> dict | None:
    steps = _json(stage9 / "steps.json")
    points, _ = _first_stage(stage9, "tooth")
    if steps is None or not points:
        return None
    first = steps.get("first_stage", {})
    ones = [p for p in points if p["numerics"] == 1]
    smallest = {key: min(p[key] for p in ones) for key in ("mfe", "object_loss")}
    fig = plt.figure(figsize=(14, 7.5), layout="constrained")
    grid = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(grid[0, 0])
    _first_stage_panel(ax, points, 0.0805, 0.0809, smallest)
    ax.set_title("tooth, first stage of the plan", fontsize=9)
    ax = fig.add_subplot(grid[0, 1])
    _first_stage_panel(ax, points, 0.08066, 0.08073, smallest)
    ax.set_title("around the switches", fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    metrics = {
        "tooth_switches": {n: v["switches"] for n, v in first.get("tooth", {}).items()},
    }
    switches = first.get("tooth", {}).get("1", {}).get("switches", [])
    by_z = {p["z01"]: p for p in ones}
    pair = [by_z.get(z) for z in switches[0]["z01"]] if switches else []
    present = len(pair) == 2 and all(p is not None and (REPO / p["arrays"]).exists() for p in pair)
    for col, (name, unit, scale) in enumerate(
        (("phase", "mrad", 1e3), ("absorption", "1e-3", 1e3)), start=2
    ):
        ax = fig.add_subplot(grid[0, col])
        ax.set_axis_off()
        if not present:
            continue
        a, b = (np.load(REPO / p["arrays"])[name].astype(np.float64) for p in pair)
        diff = b - a
        limit = float(np.abs(diff).max())
        image = ax.imshow(diff * scale, cmap="RdBu_r", vmin=-limit * scale, vmax=limit * scale)
        fig.colorbar(image, ax=ax, shrink=0.8, label=f"difference of {name} ({unit})")
        rms = float(np.sqrt(np.mean(diff**2)))
        span = (
            f"range of the phase {a.max() - a.min():.1f} rad"
            if name == "phase"
            else f"largest {a.max():.3f} and {b.max():.3f}"
        )
        ax.set_title(
            f"{name} after stage 1, field of view:\n{pair[1]['z01'] * 1e3:.3f} minus "
            f"{pair[0]['z01'] * 1e3:.3f} mm (RMS {rms * scale:.1f} {unit}, {span})",
            fontsize=8,
        )
        metrics[f"tooth_first_switch_{name}"] = {
            "rms": rms,
            "max_abs": limit,
            "largest": [float(a.max()), float(b.max())],
            "range": float(a.max() - a.min()),
        }
    ax = fig.add_subplot(grid[1, :])
    for name, color in zip(
        MEASURED_SAMPLES, ("tab:green", "tab:purple", "tab:orange", "tab:brown"), strict=False
    ):
        sel, _ = _first_stage(stage9, name)
        sel = [p for p in sel if p["numerics"] == 1]
        if len(sel) < 10:
            continue
        guess = MEASURED_SAMPLES[name].z01_guess
        low = min(p["mfe"] for p in sel)
        ax.plot(
            [p["z01"] * 1e3 - guess * 1e3 for p in sel],
            [100 * (p["mfe"] / low - 1) for p in sel],
            "-o",
            ms=2,
            lw=0.7,
            color=color,
            label=MEASURED_SAMPLES[name].label,
        )
        metrics[f"omega_{name}"] = len(first.get(name, {}).get("1", {}).get("switches", []))
    ax.set_xlabel("z01 minus the guess (mm)", fontsize=8)
    ax.set_ylabel("loss after stage 1, above the smallest (%)", fontsize=8)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.set_title("first stage over all of Omega_z, one thread", fontsize=9)
    ax.legend(fontsize=7)
    fig.legend(handles, labels, loc="outside upper center", ncols=4, fontsize=7)
    _save(fig, out / "stage9-first-stage.png", dpi)
    return metrics


def probes_figure(stage9: Path, out: Path, dpi: int) -> dict | None:
    rows = _points(stage9)
    steps = _json(stage9 / "steps.json")
    if not rows or steps is None:
        return None
    names = [s for s in ("spider_hair", "cactus_needle", "magnesium_wire") if s in steps["probes"]]
    fig, axes = plt.subplots(3, len(names), figsize=(4 * len(names), 9.5), layout="constrained")
    axes = np.atleast_2d(axes).reshape(3, len(names))
    metrics = {}
    for col, name in enumerate(names):
        best = steps["probes"][name]["1"]["minimum"]["z01"]
        span = 1.2e-3 if name == "spider_hair" else 0.6e-3
        _mfe_panel(axes[0, col], rows, name, best - span, best + span, log=True)
        axes[0, col].set_title(MEASURED_SAMPLES[name].label, fontsize=9)
        near = 0.1e-3 if name == "spider_hair" else 0.25e-3
        _mfe_panel(axes[1, col], rows, name, best - near, best + near, log=False)
        axes[1, col].set_title("around the minimum", fontsize=8)
        metrics[name] = {
            n: {k: len(v[k]) for k in ("jumps", "reversals")}
            for n, v in steps["probes"][name].items()
        }
        smallest = _losses_panel(axes[2, col], _scans(stage9, name))
        metrics[name].update({f"scan_smallest_{key}_z01": z for key, z in smallest.items()})
    _legend(fig, axes[0], columns=2)
    _save(fig, out / "stage9-probes.png", dpi)
    return metrics


def numerics_figure(stage9: Path, out: Path, dpi: int) -> dict | None:
    steps = _json(stage9 / "steps.json")
    if steps is None:
        return None
    groups = {}
    for pair in steps["same_z01"]:
        groups.setdefault(MEASURED_SAMPLES[pair["sample"]].label, []).append(pair["relative_ppm"])
    repeat = steps.get("synthetic_repeat", [])
    if repeat:
        groups["synthetic, at the truth"] = [r["mfe_true_relative_ppm"] for r in repeat]
        groups["synthetic, common path"] = [x for r in repeat for x in r["common_relative_ppm"]]
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for k, values in enumerate(groups.values()):
        values = np.asarray(values)
        jitter = np.linspace(-0.15, 0.15, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(
            k + np.asarray(jitter),
            np.maximum(np.abs(values), 1e-3),
            c=np.where(values > 0, "tab:red", "tab:blue"),
            s=14,
        )
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)), list(groups), fontsize=7, rotation=15)
    ax.set_ylabel("|MFE with 3 threads before / one thread - 1| (ppm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("same z01, both numerics (red: 3 threads higher)", fontsize=9)
    _save(fig, out / "stage9-numerics.png", dpi)
    return {label: [float(v) for v in values] for label, values in groups.items()}


def measured_figure(stage9: Path, out: Path, dpi: int) -> dict | None:
    record = _json(stage9 / "compare.json")
    if record is None or not record["measured"]["samples"]:
        return None
    samples = record["measured"]["samples"]
    fig, axes = plt.subplots(
        2, len(samples), figsize=(3.6 * len(samples), 6.5), layout="constrained"
    )
    axes = np.atleast_2d(axes).reshape(2, len(samples))
    for col, (name, s) in enumerate(samples.items()):
        for row, family in enumerate(("nm", "brent")):
            ax = axes[row, col]
            for method, e in s["searches"].items():
                if not method.startswith(family) or method == "nm-cold-repeat":
                    continue
                label, color, marker, line = METHODS[method]
                z = np.asarray(e["evaluated"]) * 1e3
                ax.plot(
                    np.arange(1, len(z) + 1),
                    z,
                    marker=marker,
                    ls=line,
                    color=color,
                    ms=3.5,
                    lw=0.8,
                    label=label,
                )
            refs = s["paper2"]
            ax.axhline(refs["z01_nm"] * 1e3, color="black", lw=0.8, label="[2] Nelder-Mead")
            if refs.get("z01_visual") is not None:
                ax.axhline(
                    refs["z01_visual"] * 1e3, color="black", ls=":", lw=0.8, label="[2] visual"
                )
            cold = s["searches"]["nm-cold"]["z01"] * 1e3
            ax.axhline(cold, color="0.5", ls="-.", lw=0.8, label="NM, guess ±5 mm: result")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("evaluation", fontsize=8)
            ax.set_ylabel("z01 (mm)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(MEASURED_SAMPLES[name].label, fontsize=9)
    _legend(fig, axes, columns=4)
    _save(fig, out / "stage9-measured.png", dpi)
    return record["measured"]["totals"]


def reconstructions_figure(stage9: Path, out: Path, dpi: int) -> dict | None:
    record = _json(stage9 / "compare.json")
    if record is None or not record["synthetic"]["summaries"]:
        return None
    summaries = record["synthetic"]["summaries"]
    savings = record["synthetic"]["savings"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    ax = axes[0]
    shown = [m for m in METHODS if m in summaries and m != "nm-cold-repeat"]
    for k, m in enumerate(shown):
        s = summaries[m]
        label, color, _, _ = METHODS[m]
        lo, hi = s["reconstructions_ci"]
        ax.bar(k, s["reconstructions"], color=color, alpha=0.8)
        ax.errorbar(
            k,
            s["reconstructions"],
            [[s["reconstructions"] - lo], [hi - s["reconstructions"]]],
            color="black",
            capsize=3,
        )
    ax.set_xticks(
        range(len(shown)), [METHODS[m][0] for m in shown], fontsize=7, rotation=25, ha="right"
    )
    ax.set_ylabel("reconstructions per autofocus (mean, 95 % CI)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    for k, s in enumerate(savings.values()):
        for j, (family, color) in enumerate((("nm", "black"), ("brent", "tab:green"))):
            x = 3 * k + j
            value, (lo, hi) = (
                100 * s[family]["relative"],
                np.multiply(100, s[family]["relative_ci"]),
            )
            ax.bar(
                x, value, color=color, alpha=0.8, label="Nelder-Mead" if family == "nm" else "Brent"
            )
            ax.errorbar(x, value, [[value - lo], [hi - value]], color="black", capsize=3)
    ax.set_xticks(
        [3 * k + 0.5 for k in range(len(savings))],
        [f"95 % {key.upper()}" for key in savings],
        fontsize=8,
    )
    ax.set_ylabel("reconstructions saved against the cold search (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    ax.legend(unique.values(), unique.keys(), fontsize=7)
    _save(fig, out / "stage9-reconstructions.png", dpi)
    return {"summaries": summaries, "savings": savings}


def savings_width_figure(stage9: Path, out: Path, dpi: int) -> dict | None:
    record = _json(stage9 / "compare.json")
    rows = _csv(stage9 / "synthetic.csv")
    if record is None or not rows or not record["synthetic"]["savings"]:
        return None
    savings = record["synthetic"]["savings"]
    fig, axes = plt.subplots(1, len(savings), figsize=(5 * len(savings), 3.8), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, (key, s) in zip(axes, savings.items(), strict=True):
        for family, color, offset in (("nm", "black", -0.15), ("brent", "tab:green", 0.15)):
            cold, box = (f"{family}-cold", f"{family}-box95-{key}")
            pts = [
                (
                    r[f"{key}_width_um"] / 1e3,
                    r[f"{cold}_n_reconstructions"] - r[f"{box}_n_reconstructions"],
                )
                for r in rows
                if isinstance(r.get(f"{box}_n_reconstructions"), float)
                and isinstance(r.get(f"{cold}_n_reconstructions"), float)
            ]
            if pts:
                w, d = np.array(pts).T
                ax.scatter(
                    w,
                    d + offset,
                    s=10,
                    color=color,
                    alpha=0.6,
                    label="Nelder-Mead" if family == "nm" else "Brent",
                )
            centers = [np.mean(b["width_um"]) / 1e3 for b in s["by_width"]]
            ax.plot(centers, [b[family]["saved"] for b in s["by_width"]], "-", color=color, lw=1.5)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel(f"width of the 95 % interval, {key.upper()} (mm)", fontsize=8)
        ax.set_ylabel("reconstructions saved", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    _save(fig, out / "stage9-savings-width.png", dpi)
    return {key: s["by_width"] for key, s in savings.items()}


def errors_figure(stage9: Path, out: Path, dpi: int) -> dict | None:
    rows = _csv(stage9 / "synthetic.csv")
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    metrics = {}
    for m, (label, color, _, line) in METHODS.items():
        err = [abs(r[f"{m}_error_um"]) for r in rows if isinstance(r.get(f"{m}_error_um"), float)]
        if not err or m == "nm-cold-repeat":
            continue
        err = np.sort(err)
        ax.step(
            err,
            np.arange(1, len(err) + 1) / len(err),
            where="post",
            color=color,
            ls=line,
            label=f"{label} (MAE {err.mean():.0f} um)",
        )
        metrics[m] = {"n": len(err), "mae_um": float(err.mean())}
    ax.axvline(100, color="0.5", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("|z01 found - truth| (um)", fontsize=8)
    ax.set_ylabel("share of the samples", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, out / "stage9-errors.png", dpi)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="low resolution (CI)")
    parser.add_argument("--stage9", type=Path, default=STAGE9)
    args = parser.parse_args()
    dpi = 60 if args.quick else 120
    args.out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"quick": args.quick}
    steps = [
        ("tooth_step", lambda: tooth_step_figure(args.stage9, args.out, dpi)),
        ("tooth_objects", lambda: tooth_objects_figure(args.stage9, args.out, dpi)),
        ("tooth_loss", lambda: tooth_loss_figure(args.stage9, args.out, dpi)),
        ("first_stage", lambda: first_stage_figure(args.stage9, args.out, dpi)),
        ("probes", lambda: probes_figure(args.stage9, args.out, dpi)),
        ("numerics", lambda: numerics_figure(args.stage9, args.out, dpi)),
        ("measured", lambda: measured_figure(args.stage9, args.out, dpi)),
        ("reconstructions", lambda: reconstructions_figure(args.stage9, args.out, dpi)),
        ("savings_width", lambda: savings_width_figure(args.stage9, args.out, dpi)),
        ("errors", lambda: errors_figure(args.stage9, args.out, dpi)),
    ]
    for name, fn in steps:
        start = time.perf_counter()
        value = fn()
        if value is None:
            print(f"{name}: inputs missing, skipped")
            continue
        metrics[name] = value
        print(f"{name}: {time.perf_counter() - start:.1f} s")
    with open(args.out / "stage9-metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
