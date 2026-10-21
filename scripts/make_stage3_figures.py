"""Stage 3 figures for the lab notebook (``docs/laborbuch/figures``).

* ``stage3-simulated-curves.png``: [2] Fig. 7 for the first test sample of every phantom
  type: the true exit wave is de-focused, projected onto ``Omega_P`` and fed to the MFE (with
  the noise-free hologram) and to the six image criteria (on its phase). No reconstruction.
* ``stage3-reconstruction-curves.png``: [2] Fig. 8, one ASRM reconstruction per sampling
  point (``results/focus-curves``): synthetic samples centred on the true ``z01``, measured
  holograms on the visual focus of [2] Table 1.
* ``stage3-baseline.png``: accuracy and cost of the classical autofocus on the synthetic test
  split (``results/baseline/cpu-4x-test``).
* ``stage3-search.png``: Nelder-Mead probing sequences on the MFE.
* ``stage3-measured.png``: the autofocus on the measured holograms of [1] against [2] Table 1
  and the ``z01`` of the example scripts of [1] (``results/measured``); the phase panels need
  the ``*-phase.npy`` files of that run.
* ``stage3-bias.png``: fine MFE scans around the true ``z01`` for noisy, noise-free and
  grid-simulated holograms (``results/bias``).
* ``stage3-crosscheck.png``: ASRM port and ``find_focus`` against holowizard 3.0.6
  (``results/crosscheck``).
* ``stage3-metrics.json``: the numbers quoted in the lab notebook.

Usage::

    python scripts/make_stage3_figures.py --data data/cpu-4x/test.h5   # about 5 min
    python scripts/make_stage3_figures.py --quick --out /tmp/figures   # smoke test (CI)

Figures whose inputs are missing are skipped. Without ``--data`` the test phantoms are drawn
again with the generator of the dataset config (same seeds as the stored split).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from holofocus.asrm import stage_grid  # noqa: E402
from holofocus.autofocus import (  # noqa: E402
    FOCUS_EXTREMUM,
    IMAGE_CRITERIA,
    extremum_offset,
    normalize_curve,
    simulated_focus_curve,
    symmetric_offsets,
)
from holofocus.baseline.results import read_rows, read_traces, summarize  # noqa: E402
from holofocus.dataset import DatasetConfig, HologramGenerator, read_config, read_labels  # noqa: E402
from holofocus.forward import object_from_phase  # noqa: E402
from holofocus.measured import MEASURED_SAMPLES  # noqa: E402
from holofocus.phantoms import PHANTOM_TYPES  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "laborbuch" / "figures"
DEFAULT_RESULTS = REPO / "results"
CRITERIA = ("MFE", *IMAGE_CRITERIA)
TYPE_COLORS = {"cell": "black", "triangle": "tab:blue", "ball": "tab:red", "mixed": "tab:green"}
SAMPLE_COLORS = {
    "spider_hair": "black",
    "tooth": "tab:cyan",
    "cactus_needle": "tab:red",
    "magnesium_wire": "tab:orange",
}
NM_STOP_UM = 100.0  # simplex length at which Nelder-Mead stops ([2] Sec. 3.3)
PADDING = 2.0  # padding factor of configs/baseline/paper2-cpu.yaml


@dataclass
class Settings:
    config: str
    curve_points: int
    half_range: float
    max_index: int
    dpi: int


FULL = Settings("configs/dataset/cpu-4x.yaml", 201, 10e-3, 64, 120)
QUICK = Settings("configs/dataset/ci.yaml", 21, 10e-3, 64, 60)


def unimodality(offsets, values, kind: str) -> float:
    """Fraction of sampling steps that change the criterion in the right direction (1 = ideal).

    Steps towards ``dz01 = 0`` must decrease (``kind="min"``) or increase (``"max"``) the
    criterion and steps away from it the opposite; a monotonic or random curve gives about 0.5.
    """
    dz = np.asarray(offsets, dtype=float)
    diff = np.diff(np.asarray(values, dtype=float))
    away = np.where(dz[:-1] >= 0, 1.0, np.where(dz[1:] <= 0, -1.0, 0.0))
    expected = away if kind == "min" else -away
    valid = expected != 0
    return float(np.mean(np.sign(diff[valid]) == expected[valid]))


def curve_metrics(curve: dict) -> dict:
    dz = np.asarray(curve["dz01"], dtype=float)
    out = {}
    for name in CRITERIA:
        kind = FOCUS_EXTREMUM[name]
        out[name] = {
            "extremum": kind,
            "extremum_offset_mm": extremum_offset(dz, curve[name], kind) * 1e3,
            "unimodality": unimodality(dz, curve[name], kind),
            "argmin_mm": extremum_offset(dz, curve[name], "min") * 1e3,
            "argmax_mm": extremum_offset(dz, curve[name], "max") * 1e3,
            "unimodality_min": unimodality(dz, curve[name], "min"),
        }
    if "phase_rmse" in curve:
        rmse = np.asarray(curve["phase_rmse"], dtype=float)
        out["phase_rmse"] = {
            "min_offset_mm": float(dz[int(np.argmin(rmse))] * 1e3),
            "min": float(rmse.min()),
            "at_reference": float(rmse[int(np.argmin(np.abs(dz)))]),
        }
    return out


def criteria_figure(
    curves: dict, styles: dict, title: str, path: Path, dpi: int, focus_note: str = "at focus"
) -> None:
    """[2] Fig. 7-8 layout: MFE on top, the six image criteria below, all scaled to [0, 1].

    Panel titles name the extremum that marks the focus in [2] Fig. 7, followed by
    ``focus_note``.
    """
    with_rmse = any("phase_rmse" in c for c in curves.values())
    fig = plt.figure(figsize=(11.0, 9.4), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.15])
    top_axes = top.subplots(1, 2 if with_rmse else 1, squeeze=False)[0]
    ax = top_axes[0]
    for name, curve in curves.items():
        ax.plot(np.asarray(curve["dz01"]) * 1e3, normalize_curve(curve["MFE"]), **styles[name])
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.set(xlabel=r"$\Delta_{01}$ / mm", ylabel="A.U.", title="MFE")
    ax.legend(fontsize=8)
    if with_rmse:
        ax = top_axes[1]
        for name, curve in curves.items():
            if "phase_rmse" in curve:
                ax.plot(np.asarray(curve["dz01"]) * 1e3, curve["phase_rmse"], **styles[name])
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set(
            xlabel=r"$\Delta_{01}$ / mm",
            ylabel="RMS phase error / rad",
            title="reconstruction vs ground truth (synthetic)",
        )
    axes = bottom.subplots(2, 3, sharex=True, sharey=True)
    for ax, name in zip(axes.flat, IMAGE_CRITERIA, strict=True):
        for key, curve in curves.items():
            style = {k: v for k, v in styles[key].items() if k != "label"}
            ax.plot(np.asarray(curve["dz01"]) * 1e3, normalize_curve(curve[name]), **style)
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_title(f"{name} ({FOCUS_EXTREMUM[name]} {focus_note})" if name != "VAR" else name)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\Delta_{01}$ / mm")
    for ax in axes[:, 0]:
        ax.set_ylabel("A.U.")
    fig.suptitle(title)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def test_phantoms(settings: Settings, data: Path | None):
    """First test sample of every phantom type: ``config, {kind: record}``."""
    picks: dict[str, dict] = {}
    if data is not None:
        config = read_config(data)
        labels = read_labels(data)
        for index, code in enumerate(labels["phantom_type"].astype(int)):
            picks.setdefault(PHANTOM_TYPES[code], {"index": index})
        with h5py.File(data, "r") as fh:
            for record in picks.values():
                i = record["index"]
                record["phase"] = np.asarray(fh["phase"][i], dtype=np.float64)
                record["z01"] = float(labels["z01"][i])
                record["phase_shift"] = float(labels["phase_shift"][i])
        return config, picks
    config = DatasetConfig.from_yaml(REPO / settings.config)
    config = replace(config, simulation=replace(config.simulation, device="cpu"))
    generator = HologramGenerator(config, device="cpu")
    for index in range(settings.max_index):
        sample = generator.sample("test", index)
        kind = PHANTOM_TYPES[int(sample.meta["phantom_type"])]
        if kind not in picks:
            picks[kind] = {
                "index": index,
                "phase": sample.phase.astype(np.float64),
                "z01": float(sample.labels["z01"]),
                "phase_shift": float(sample.meta["phase_shift"]),
            }
        if len(picks) == len(PHANTOM_TYPES):
            break
    return config, picks


def simulated_curves_figure(settings: Settings, data: Path | None, out: Path) -> dict:
    config, picks = test_phantoms(settings, data)
    offsets = symmetric_offsets(settings.curve_points, settings.half_range)
    padded = stage_grid(config.geometry.fov_shape, 1, PADDING).shape
    curves, styles, metrics = {}, {}, {}
    for kind in PHANTOM_TYPES:
        if kind not in picks:
            continue
        record = picks[kind]
        obj = object_from_phase(torch.from_numpy(record["phase"]), config.phantoms.delta_beta)
        curve = simulated_focus_curve(obj, config.geometry.at(record["z01"]), offsets, padded)
        curves[kind] = curve
        styles[kind] = {
            "color": TYPE_COLORS[kind],
            "lw": 1.2,
            "label": f"{kind} (test {record['index']}, {record['phase_shift']:.2f} rad)",
        }
        metrics[kind] = {
            "index": record["index"],
            "z01": record["z01"],
            "phase_shift": record["phase_shift"],
            **curve_metrics(curve),
        }
    g = config.geometry
    criteria_figure(
        curves,
        styles,
        f"Focus curves from simulation ([2] Fig. 7): '{config.name}' test phantoms, noise-free\n"
        f"{g.fov_shape[0]}$^2$ px (binning {g.binning}x) padded to {padded[0]}$^2$, "
        f"{len(offsets)} points",
        out / "stage3-simulated-curves.png",
        settings.dpi,
    )
    return {"dataset": config.name, "points": len(offsets), "padded_shape": padded, **metrics}


def reconstruction_curves_figure(directory: Path, settings: Settings, out: Path) -> dict | None:
    files = sorted(directory.glob("synthetic-*.json")) + sorted(directory.glob("measured-*.json"))
    if not files:
        return None
    curves, styles, metrics = {}, {}, {}
    for path in files:
        record = json.loads(path.read_text())
        key = path.stem
        curve = {k: np.asarray(v) for k, v in record["curve"].items()}
        curves[key] = curve
        if record["kind"] == "synthetic":
            kind = PHANTOM_TYPES[record["phantom_type"]]
            label = f"{kind} (test {record['index']}, N = {record['photons']:.0f})"
            styles[key] = {"color": TYPE_COLORS[kind], "ls": "--", "lw": 1.1, "label": label}
        else:
            label = f"{record['name']} (measured, 4× grid, p = {record['padding_factor']:g})"
            styles[key] = {"color": SAMPLE_COLORS[record["sample"]], "lw": 1.3, "label": label}
        metrics[key] = {
            "name": record["name"],
            "reference": record["reference"],
            "z01_reference": record["z01_reference"],
            "points": len(curve["dz01"]),
            "seconds": record["seconds"],
            "threads": record["threads"],
            **curve_metrics(curve),
        }
    points = sorted({m["points"] for m in metrics.values()})
    criteria_figure(
        curves,
        styles,
        "Focus curves from ASRM reconstructions ([2] Fig. 8), one reconstruction per point "
        f"({'/'.join(map(str, points))} points)\nsynthetic: true $z_{{01}}$ at 0; measured: "
        "visual focus of [2] Table 1 at 0",
        out / "stage3-reconstruction-curves.png",
        settings.dpi,
        focus_note="in [2] Fig. 7",
    )
    return metrics


def _by_type(rows: list[dict], value) -> dict:
    out = {}
    for code, kind in enumerate(PHANTOM_TYPES):
        part = [value(r) for r in rows if r["phantom_type"] == code]
        if part:
            out[kind] = {
                "n": len(part),
                "mae": float(np.mean(np.abs(part))),
                "mean": float(np.mean(part)),
            }
    return out


def baseline_figure(run_dir: Path, settings: Settings, out: Path) -> dict | None:
    rows = read_rows(run_dir / "results.csv")
    if not rows:
        return None
    info = yaml.safe_load((run_dir / "run.yaml").read_text())
    summary = summarize(rows)
    err = np.array([r["error_um"] for r in rows])
    dz = np.array([r["dz01_true"] * 1e3 for r in rows])
    photons = np.array([r["photons"] for r in rows])
    recon = np.array([r["n_reconstructions"] for r in rows])
    seconds = np.array([r["seconds"] for r in rows])
    codes = np.array([r["phantom_type"] for r in rows])
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.4), layout="constrained")

    def scatter(ax, x, y, **kw):
        for code, kind in enumerate(PHANTOM_TYPES):
            sel = codes == code
            if sel.any():
                ax.plot(x[sel], y[sel], "o", ms=4, color=TYPE_COLORS[kind], label=kind, **kw)

    ax = axes[0, 0]
    scatter(ax, dz, err)
    ax.axhline(0, color="gray", lw=0.8)
    for s in (-1, 1):
        ax.axhline(s * NM_STOP_UM, color="gray", ls="--", lw=0.8)
    ax.set(
        xlabel=r"initial error $z_{01} - z_{01}^\mathrm{guess}$ / mm",
        ylabel=r"error $\hat z_{01} - z_{01}$ / um",
        title=f"bias {err.mean():+.0f} um, std {summary['error_um']['std']:.0f} um",
    )
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    scatter(ax, photons, np.abs(err))
    ax.axhline(NM_STOP_UM, color="gray", ls="--", lw=0.8, label="NM stop (100 um)")
    ax.set(
        xscale="log", yscale="log", xlabel="flat-field counts per pixel N", ylabel="|error| / um"
    )
    ax.set_title(
        f"MAE {summary['error_um']['mae']:.0f} um, RMSE {summary['error_um']['rmse']:.0f} um"
    )
    ax = axes[0, 2]
    edges = np.arange(np.floor(err.min() / 25) * 25, np.ceil(err.max() / 25) * 25 + 25, 25)
    ax.hist(err, bins=edges, color="C0")
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="error / um", ylabel="samples")
    within = summary["fraction_within_um"]
    ax.set_title(
        f"{100 * within['100']:.0f} % within 100 um, {100 * within['250']:.0f} % within 250 um"
    )
    ax = axes[1, 0]
    counts = np.bincount(recon)
    ks = np.nonzero(counts)[0]
    ax.bar(ks, counts[ks], color="C1")
    ax.set(xlabel="reconstructions per autofocus", ylabel="samples")
    ax2 = ax.twinx()
    ax2.plot(recon + np.random.default_rng(0).uniform(-0.15, 0.15, len(recon)), seconds, "k.", ms=3)
    ax2.set_ylabel("seconds per autofocus (CPU)")
    ax.set_title(
        f"{summary['reconstructions']['mean']:.1f} reconstructions, "
        f"{summary['seconds_per_autofocus']['median']:.0f} s per autofocus"
    )
    ax = axes[1, 1]
    truth = [r for r in rows if r["mfe_true"] is not None]
    rel = np.array([r["mfe_est"] / r["mfe_true"] - 1 for r in truth])
    scatter_rows = np.array([r["error_um"] for r in truth])
    sel_codes = np.array([r["phantom_type"] for r in truth])
    for code, kind in enumerate(PHANTOM_TYPES):
        sel = sel_codes == code
        if sel.any():
            ax.plot(scatter_rows[sel], 100 * rel[sel], "o", ms=4, color=TYPE_COLORS[kind])
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(
        xlabel="error / um",
        ylabel=r"MFE($\hat z_{01}$) / MFE($z_{01}$) - 1 / %",
        title=f"MFE at the estimate below the truth: {100 * np.mean(rel <= 0):.0f} %",
    )
    ax = axes[1, 2]
    pairs = [(r["phase_rmse_true"], r["phase_rmse_est"], r["phantom_type"]) for r in rows]
    pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
    if pairs:
        p_true, p_est, p_codes = (np.array(v) for v in zip(*pairs, strict=True))
        for code, kind in enumerate(PHANTOM_TYPES):
            sel = p_codes == code
            if sel.any():
                ax.plot(p_true[sel], p_est[sel], "o", ms=4, color=TYPE_COLORS[kind])
        lim = (0, 1.05 * max(p_true.max(), p_est.max()))
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set(xlim=lim, ylim=lim)
        ax.set_title(f"phase error: median ratio {np.median(p_est / p_true):.3f}")
    ax.set(xlabel=r"RMS phase error at $z_{01}$ / rad", ylabel=r"... at $\hat z_{01}$ / rad")
    asrm = info["asrm_settings"]
    stages = "/".join(f"{s['iterations']}@{s['down_sampling']}x" for s in asrm["stages"])
    fig.suptitle(
        f"Classical autofocus of [2] on the synthetic '{info['dataset_name']}' "
        f"{info['dataset']['split']} split: {len(rows)} samples, ASRM {stages}, padding "
        f"{asrm['padding_factor']:g}, NM on guess $\\pm$ 5 mm ({info['threads']} CPU threads)"
    )
    fig.savefig(out / "stage3-baseline.png", dpi=settings.dpi)
    plt.close(fig)
    terciles = np.percentile(photons, [100 / 3, 200 / 3])
    groups = np.digitize(photons, terciles)
    by_photons = {
        f"N {lo:.0f}-{hi:.0f}": {
            "n": int((groups == k).sum()),
            "mae": float(np.abs(err[groups == k]).mean()),
            "mean": float(err[groups == k].mean()),
        }
        for k, (lo, hi) in enumerate(
            zip([photons.min(), *terciles], [*terciles, photons.max()], strict=True)
        )
        if (groups == k).any()
    }
    slope, intercept = np.polyfit(dz, err, 1) if len(rows) > 2 else (np.nan, np.nan)
    return {
        "run": str(run_dir.resolve().relative_to(REPO))
        if run_dir.resolve().is_relative_to(REPO)
        else str(run_dir),
        "git_commit": info.get("git_commit"),
        "threads": info.get("threads"),
        "summary": summary,
        "by_type": _by_type(rows, lambda r: r["error_um"]),
        "by_photons": by_photons,
        "error_vs_dz01_fit": {"slope_um_per_mm": float(slope), "intercept_um": float(intercept)},
        "mfe_est_below_truth": float(np.mean(rel <= 0)) if len(rel) else None,
    }


def search_figure(run_dir: Path, settings: Settings, out: Path) -> dict | None:
    rows = {r["index"]: r for r in read_rows(run_dir / "results.csv")}
    traces = {t["index"]: t for t in read_traces(run_dir / "traces.jsonl")}
    if not rows or not traces:
        return None
    order = sorted(rows, key=lambda i: rows[i]["abs_error_um"])
    picks = {"smallest": order[0], "median": order[len(order) // 2], "largest": order[-1]}
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.4), layout="constrained")
    for ax, (what, index) in zip(axes, picks.items(), strict=False):
        row, trace = rows[index], traces[index]
        probes = [e for e in trace["evaluations"] if e["reconstructed"]]
        z = np.array([(e["z01"] - row["z01_true"]) * 1e3 for e in probes])
        rel = np.array([e["mfe"] for e in probes]) / row["mfe_est"] - 1
        ax.plot(z, rel, "o", color="C0", ms=5)
        for k, (x, y) in enumerate(zip(z, rel, strict=True), start=1):
            ax.annotate(str(k), (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)
        ax.axvline(-row["dz01_true"] * 1e3, color="gray", ls="--", lw=0.8, label="guess")
        if row["mfe_true"] is not None:
            truth = row["mfe_true"] / row["mfe_est"] - 1
            ax.plot(0.0, truth, "D", color="k", ms=6, label="MFE at the true z01")
        est = (row["z01_est"] - row["z01_true"]) * 1e3
        ax.plot(est, 0.0, "*", color="C3", ms=12, label="estimate")
        ax.set_yscale("symlog", linthresh=1e-5)
        kind = PHANTOM_TYPES[row["phantom_type"]]
        ax.set(
            xlabel=r"$\hat z_{01} - z_{01}$ / mm",
            ylabel=r"MFE / MFE($\hat z_{01}$) - 1",
            title=f"{what} error: test {index} ({kind}), {row['error_um']:+.0f} um",
        )
        ax.legend(fontsize=7)
    ax = axes[3]
    distances = []
    for trace in traces.values():
        probes = [e["z01"] for e in trace["evaluations"] if e["reconstructed"]]
        d = np.abs(np.array(probes) - trace["z01"]) * 1e6
        distances.append(d)
        ax.semilogy(np.arange(1, len(d) + 1), np.maximum(d, 1.0), color="gray", lw=0.6, alpha=0.6)
    longest = max(len(d) for d in distances)
    median = [np.median([d[k] for d in distances if len(d) > k]) for k in range(longest)]
    ax.semilogy(np.arange(1, longest + 1), np.maximum(median, 1.0), "k-", lw=2, label="median")
    ax.axhline(NM_STOP_UM, color="C3", ls="--", lw=0.8, label="100 um")
    ax.set(
        xlabel="reconstruction number",
        ylabel=r"|probe - final estimate| / um (floor 1 um)",
        title=f"all {len(traces)} searches",
    )
    ax.legend(fontsize=8)
    fig.suptitle("Nelder-Mead probing sequences on the MFE (synthetic test split)")
    fig.savefig(out / "stage3-search.png", dpi=settings.dpi)
    plt.close(fig)
    return {"examples": picks}


MEASURED_ORDER = ("p2", "b1-p2", "b1-p4", "paper-padding")


def measured_runs(directory: Path) -> dict[str, Path]:
    """Run directories below ``results/measured`` in the order of :data:`MEASURED_ORDER`."""
    if not directory.is_dir():
        return {}
    found = {d.name: d for d in directory.iterdir() if d.is_dir()}
    order = [n for n in MEASURED_ORDER if n in found] + sorted(set(found) - set(MEASURED_ORDER))
    return {n: found[n] for n in order}


def run_label(record: dict) -> str:
    data = "4×4 block mean" if record["binning"] > 1 else "ASRM down-sampling"
    return f"{data}, p = {record['padding_factor']:g}"


def measured_figure(directories: dict[str, Path], settings: Settings, out: Path) -> dict | None:
    records: dict[str, dict[str, dict]] = {}
    for label, directory in directories.items():
        for name in MEASURED_SAMPLES:
            path = directory / f"{name}.json"
            if path.exists():
                records.setdefault(name, {})[label] = json.loads(path.read_text())
    if not records:
        return None
    names = [n for n in MEASURED_SAMPLES if n in records]
    fig, axes = plt.subplots(
        2, len(names), figsize=(4.0 * len(names), 7.8), layout="constrained", squeeze=False
    )
    table = {}
    first = next(iter(directories))
    markers = dict(zip(directories, ("o", "s", "^", "D", "v", "P"), strict=False))
    for col, name in enumerate(names):
        sample = MEASURED_SAMPLES[name]
        ax = axes[0, col]
        phase_file = directories[first] / f"{name}-phase.npy"
        if phase_file.exists() and first in records[name]:
            phase = np.load(phase_file)
            lo, hi = np.percentile(phase, [0.5, 99.5])
            ax.imshow(phase, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(
                f"{sample.label}: phase at the estimate\n({run_label(records[name][first])})",
                fontsize=9,
            )
        else:
            ax.text(
                0.5, 0.5, "phase not available", ha="center", va="center", transform=ax.transAxes
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axes[1, col]
        table[name] = {
            "paper2": {
                "z01_nm_mm": sample.z01_nm * 1e3,
                "z01_visual_mm": sample.z01_visual * 1e3,
                "samples": sample.nm_samples,
            },
            "paper1_example_z01_mm": sample.z01_example * 1e3,
        }
        for label, record in records[name].items():
            result = record["result"]
            probes = [e for e in result["evaluations"] if e["reconstructed"]]
            z = np.array([e["z01"] * 1e3 for e in probes])
            mfe = np.array([e["mfe"] for e in probes])
            ax.plot(
                z,
                mfe / mfe.min(),
                markers[label],
                ms=5,
                mfc="none",
                label=f"{run_label(record)}: {len(probes)} rec.",
            )
            ax.plot(result["z01"] * 1e3, 1.0, markers[label], color="C3", ms=7)
            table[name][label] = {
                "binning": record["binning"],
                "z01_mm": result["z01"] * 1e3,
                "minus_paper2_nm_um": record["z01_minus_paper2_nm_um"],
                "minus_visual_um": record["z01_minus_visual_um"],
                "minus_paper1_example_um": (result["z01"] - sample.z01_example) * 1e6,
                "mfe_range_percent": 100.0 * (mfe.max() / mfe.min() - 1.0),
                "reconstructions": result["n_reconstructions"],
                "seconds": result["seconds"],
                "seconds_per_reconstruction": record["seconds_per_reconstruction"],
                "threads": record["threads"],
                "padding_factor": record["padding_factor"],
                "stage_arrays": record["stage_arrays"],
            }
        ax.axvline(
            sample.z01_nm * 1e3, color="k", lw=0.9, label=f"[2] NM {sample.z01_nm * 1e3:.1f} mm"
        )
        ax.axvline(
            sample.z01_visual * 1e3,
            color="gray",
            ls="--",
            lw=0.9,
            label=f"[2] visual {sample.z01_visual * 1e3:.1f} mm",
        )
        ax.axvline(
            sample.z01_example * 1e3,
            color="tab:green",
            ls=":",
            lw=1.2,
            label=f"[1] example {sample.z01_example * 1e3:.2f} mm",
        )
        ax.set(
            xlabel=r"$z_{01}$ / mm",
            ylabel="MFE / min MFE",
            title=f"{sample.label} ([2]: {sample.nm_samples} samples)",
        )
        ax.legend(fontsize=7)
    fig.suptitle(
        "Classical autofocus on the measured holograms of [1] "
        "(Zenodo 10.5281/zenodo.8349365)\non the 4× grid against [2] Table 1 "
        "(red: estimates; p: padding factor)"
    )
    fig.savefig(out / "stage3-measured.png", dpi=settings.dpi)
    plt.close(fig)
    return table


BIAS_STYLES = {
    "noisy": {"color": "tab:red", "marker": "o"},
    "clean": {"color": "tab:blue", "marker": "s"},
    "grid": {"color": "black", "marker": "^"},
    "grid-nofilter": {"color": "tab:green", "marker": "v", "ls": "--"},
}
BIAS_LABELS = {
    "noisy": "noisy (dataset)",
    "clean": "noise-free",
    "grid": "noise-free, reconstruction grid",
    "grid-nofilter": "noise-free, reconstruction grid,\nno object filters",
}


def bias_figure(directory: Path, run_dir: Path, settings: Settings, out: Path) -> dict | None:
    files = sorted(directory.glob("bias-*.json"))
    if not files:
        return None
    nm = {r["index"]: r for r in read_rows(run_dir / "results.csv")}
    fig, axes = plt.subplots(
        len(files), 2, figsize=(11.0, 4.2 * len(files)), layout="constrained", squeeze=False
    )
    metrics = {}
    for (ax_mfe, ax_phase), path in zip(axes, files, strict=True):
        rec = json.loads(path.read_text())
        kind = PHANTOM_TYPES[rec["phantom_type"]]
        entry = {"phantom_type": kind, "photons": rec["photons"], "minimum_um": rec["minimum_um"]}
        for variant, curve in rec["curves"].items():
            style = BIAS_STYLES.get(variant, {})
            dz_um = np.asarray(curve["dz01"]) * 1e6
            vertex = rec["minimum_um"][variant]["parabola"]
            ax_mfe.plot(
                dz_um,
                normalize_curve(curve["MFE"]),
                ms=4,
                lw=1.0,
                label=f"{BIAS_LABELS.get(variant, variant)}: {vertex:+.0f} µm",
                **style,
            )
            ax_mfe.axvline(vertex, color=style.get("color"), lw=0.8, ls=":")
            ax_phase.plot(dz_um, curve["phase_rmse"], ms=4, lw=1.0, **style)
            entry.setdefault("mfe_range", {})[variant] = [min(curve["MFE"]), max(curve["MFE"])]
        if rec["index"] in nm:
            error = nm[rec["index"]]["error_um"]
            entry["nelder_mead_error_um"] = error
            ax_mfe.axvline(
                error, color="tab:purple", lw=2.0, alpha=0.5, label="Nelder–Mead estimate"
            )
        for ax in (ax_mfe, ax_phase):
            ax.axvline(0.0, color="gray", lw=0.8)
        ax_mfe.set(
            xlabel=r"$\hat z_{01} - z_{01}$ / µm",
            ylabel="MFE, normalized",
            title=f"test {rec['index']} ({kind}, N = {rec['photons']:.0f}): MFE\n"
            "(dotted: parabola vertex)",
        )
        ax_phase.legend(*ax_mfe.get_legend_handles_labels(), fontsize=7, loc="upper left")
        ax_phase.set(
            xlabel=r"$\hat z_{01} - z_{01}$ / µm",
            ylabel="phase RMSE / rad",
            title="phase error of the reconstruction",
        )
        metrics[rec["index"]] = entry
    fig.suptitle("Location of the MFE minimum for different data models")
    fig.savefig(out / "stage3-bias.png", dpi=settings.dpi)
    plt.close(fig)
    return metrics


def crosscheck_figure(directory: Path, settings: Settings, out: Path) -> dict | None:
    reco_files = sorted(directory.glob("reconstruct_*.json"))
    focus_files = sorted(
        p for p in directory.glob("focus_*.json") if p.stem.split("_")[-1].isdigit()
    )
    if not reco_files and not focus_files:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), layout="constrained")
    metrics: dict = {}
    if reco_files:
        rec = json.loads(reco_files[0].read_text())
        hw = np.asarray(rec["loss"]["holowizard"])
        hf = np.asarray(rec["loss"]["holofocus"])
        it = np.arange(1, len(hw) + 1)
        ax = axes[0]
        ax.semilogy(
            it,
            hw,
            color="C0",
            lw=2.5,
            alpha=0.5,
            label=f"holowizard 3.0.6 ({rec['holowizard_seconds']:.0f} s)",
        )
        ax.semilogy(
            it, hf, color="k", lw=0.8, label=f"holofocus ({rec['holofocus_seconds']:.0f} s)"
        )
        for end in rec["stage_ends"][:-1]:
            ax.axvline(end, color="gray", ls=":", lw=0.8)
        ax.set(
            xlabel="iteration",
            ylabel="MFE (loss)",
            title=f"test {rec['index']}: ASRM [2] Table 3 schedule",
        )
        ax.legend(fontsize=8)
        ax = axes[1]
        ax.semilogy(it, np.abs(hf / hw - 1), color="C2", lw=0.8)
        ax.set(
            xlabel="iteration",
            ylabel="|loss_holofocus / loss_holowizard - 1|",
            title=f"phase max diff {rec['phase_max_abs_diff']:.1e} rad "
            f"(rms phase {rec['phase_rms']:.2f} rad)",
        )
        metrics["reconstruct"] = {k: v for k, v in rec.items() if k not in ("loss", "settings")}
    ax = axes[2]
    metrics["focus"] = {}
    for k, path in enumerate(focus_files):
        rec = json.loads(path.read_text())
        guess = rec["z01_guess_mm"]
        hw = np.asarray(rec["holowizard"]["points_mm"]) - guess
        hf = np.asarray(rec["holofocus"]["points_mm"]) - guess
        ax.plot(
            np.arange(1, len(hw) + 1),
            hw,
            "o",
            mfc="none",
            ms=8,
            color=f"C{k}",
            label=f"test {rec['index']}: holowizard",
        )
        ax.plot(
            np.arange(1, len(hf) + 1),
            hf,
            "x",
            ms=6,
            color=f"C{k}",
            label=f"test {rec['index']}: holofocus",
        )
        ax.axhline(rec["z01_true_mm"] - guess, color=f"C{k}", ls=":", lw=0.8)
        metrics["focus"][rec["index"]] = {
            "same_points": rec["same_points"],
            "estimate_diff_um": rec["estimate_diff_um"],
            "reconstructions": [
                rec["holowizard"]["reconstructions"],
                rec["holofocus"]["reconstructions"],
            ],
            "z01_mm": [rec["holowizard"]["z01_mm"], rec["holofocus"]["z01_mm"]],
            "z01_true_mm": rec["z01_true_mm"],
            "seconds": [rec["holowizard"]["seconds"], rec["holofocus"]["seconds"]],
        }
    ax.set(
        xlabel="reconstruction number",
        ylabel=r"probe $z_{01} - z_{01}^\mathrm{guess}$ / mm",
        title="find_focus probes, 10 % of the iterations\n(dotted: true $z_{01}$)",
    )
    ax.legend(fontsize=7, ncols=2)
    fig.suptitle("Cross-check of the ASRM port and the nested autofocus against holowizard 3.0.6")
    fig.savefig(out / "stage3-crosscheck.png", dpi=settings.dpi)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="16x binned config, 21 points")
    parser.add_argument("--data", type=Path, help="test split for the simulated curves (HDF5)")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--baseline", type=Path, help="baseline run directory")
    parser.add_argument("--threads", type=int, help="torch CPU threads")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    settings = QUICK if args.quick else FULL
    results = args.results
    baseline = args.baseline or results / "baseline" / "cpu-4x-test"
    measured = measured_runs(results / "measured")
    args.out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"settings": asdict(settings), "quick": args.quick}
    steps = [
        ("simulated_curves", lambda: simulated_curves_figure(settings, args.data, args.out)),
        (
            "reconstruction_curves",
            lambda: reconstruction_curves_figure(results / "focus-curves", settings, args.out),
        ),
        ("baseline", lambda: baseline_figure(baseline, settings, args.out)),
        ("search", lambda: search_figure(baseline, settings, args.out)),
        ("measured", lambda: measured_figure(measured, settings, args.out)),
        ("bias", lambda: bias_figure(results / "bias", baseline, settings, args.out)),
        ("crosscheck", lambda: crosscheck_figure(results / "crosscheck", settings, args.out)),
    ]
    for name, fn in steps:
        start = time.perf_counter()
        value = fn()
        if value is None:
            print(f"{name}: inputs missing, skipped")
            continue
        metrics[name] = value
        print(f"{name}: {time.perf_counter() - start:.1f} s")
    with open(args.out / "stage3-metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
