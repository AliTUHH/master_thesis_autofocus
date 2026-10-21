"""Stage 4 figures for the lab notebook (``docs/laborbuch/figures``).

* ``stage4-inputs.png``: what the networks see, for three validation samples: the binned
  hologram (image CNN) and the radial power spectrum on the ``f^2`` axis with the CTF minima
  ``f^2 = k Fr`` of the true Fresnel number (spectrum CNN).
* ``stage4-training.png``: validation MAE per epoch of the models trained on all samples
  (``--finals``) and of the study runs at 2000 and 10000 samples, learning curves per family
  (representation and channels) and the variants at 2000 samples of
  ``scripts/run_stage4_study.py`` (``results/learned/study``).
* ``stage4-test.png``: the main model on the full synthetic test split: estimate against
  truth with the predicted 1-sigma interval, error against ``dz01`` and the error histogram.
* ``stage4-calibration.png``: coverage of the central intervals against their nominal level
  (validation and test), error against predicted ``sigma`` and the ``z``-score histogram.
* ``stage4-dependence.png``: absolute error against the flat-field counts and MAE per phantom
  type, learned model and baseline.
* ``stage4-comparison.png``: the samples of the Stage 3 baseline run: errors of the raw and
  offset-corrected baseline, the learned model and the warm start, paired errors and the
  time per estimate.
* ``stage4-warmstart.png``: Nelder-Mead from the learned estimate against the cold start:
  errors, reconstructions per autofocus and probing sequences.
* ``stage4-metrics.json``: the numbers quoted in the lab notebook.

Usage::

    python scripts/make_stage4_figures.py                       # results/ and data/cpu-4x
    python scripts/make_stage4_figures.py --quick --main /tmp/learned-ci --finals /tmp/learned-ci \\
        --study /tmp/none --comparison /tmp/comparison --baseline /tmp/baseline \\
        --baseline-val /tmp/baseline --warm /tmp/warm --data /tmp/demo-dataset \\
        --out /tmp/figures   # smoke test (CI)

Figures whose inputs are missing are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from holofocus.baseline.results import read_rows, read_traces  # noqa: E402
from holofocus.dataset import read_config, read_labels  # noqa: E402
from holofocus.learned import (  # noqa: E402
    LearnedConfig,
    Representation,
    RepresentationConfig,
    read_history,
    read_predictions,
)
from holofocus.learned.compare import read_comparison  # noqa: E402
from holofocus.learned.metrics import (  # noqa: E402
    PHOTON_EDGES,
    interval_metrics,
    interval_quantile,
    photon_bin_labels,
    photon_bin_names,
    point_metrics,
)
from holofocus.phantoms import PHANTOM_TYPES  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "laborbuch" / "figures"
RESULTS = REPO / "results"
TYPE_COLORS = {"cell": "black", "triangle": "tab:blue", "ball": "tab:red", "mixed": "tab:green"}
METHOD_STYLE = {
    "baseline_val": ("baseline [2], validation", "0.55"),
    "raw": ("baseline [2]", "tab:red"),
    "corrected": ("baseline - offset", "tab:orange"),
    "learned": ("learned (CNN)", "tab:blue"),
    "warm": ("warm start", "tab:purple"),
    "warm_corrected": ("warm start - offset", "plum"),
}
SEED_RUNS = ("spectrum-n2000", "spectrum-seed1", "spectrum-seed2")


def _save(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _run_info(run: Path) -> dict:
    file = run / "run.yaml"
    return yaml.safe_load(file.read_text()) if file.exists() else {}


def _read_comparison(directory: Path) -> list[dict]:
    return read_comparison(directory) if (directory / "comparison.csv").exists() else []


def inputs_figure(data: Path, runs: list[Path], out: Path, dpi: int) -> dict | None:
    """Inputs of the first run of each representation in ``runs`` (defaults otherwise)."""
    path = data / "val.h5"
    if not path.exists():
        return None
    configs = {}
    for run in runs:
        if (run / "config.yaml").exists():
            representation = LearnedConfig.from_yaml(run / "config.yaml").representation
            configs.setdefault(representation.kind, representation)
    image_cfg = configs.get("image", RepresentationConfig(kind="image", binning=2))
    spectrum_cfg = configs.get("spectrum", RepresentationConfig(kind="spectrum"))
    labels = read_labels(path)
    shape = read_config(path).geometry.fov_shape
    order = np.argsort(labels["dz01"])
    picks = [int(order[min(4, len(order) - 1)]), int(order[len(order) // 2])]
    picks.append(int(order[max(len(order) - 5, 0)]))
    image = Representation(image_cfg, shape)
    spectrum = Representation(spectrum_cfg, shape)
    with h5py.File(path, "r") as fh:
        holograms = torch.from_numpy(np.stack([fh["hologram"][i] for i in picks]))
    images, spectra = image(holograms)[:, 0].numpy(), spectrum(holograms)[:, 0].numpy()
    width = spectrum_cfg.f2_max / spectrum_cfg.bins
    f2 = (np.arange(spectrum_cfg.bins) + 0.5) * width
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), layout="constrained")
    for col, i in enumerate(picks):
        fr = float(labels["fresnel_number"][i])
        crop = images[col]
        h, w = crop.shape
        sub = crop[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        ax = axes[0, col]
        ax.imshow(sub, cmap="gray", vmin=np.percentile(sub, 1), vmax=np.percentile(sub, 99))
        ax.set_title(
            f"val {i}: dz01 {labels['dz01'][i] * 1e3:+.2f} mm, "
            f"{PHANTOM_TYPES[int(labels['phantom_type'][i])]}, N {labels['photons'][i]:.0f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax = axes[1, col]
        ax.plot(f2, spectra[col], lw=0.6, color="tab:blue")
        for k in range(1, int(f2[-1] / fr) + 1):
            ax.axvline(k * fr, color="tab:red", lw=0.4, alpha=0.35)
        ax.set_xlim(0, min(f2[-1], 0.06))
        ax.set_xlabel("$f^2$ (cycles$^2$/px$^2$); red: $f^2 = k\\,Fr$")
        ax.set_title(f"spectrum input, Fr = {fr:.4e}", fontsize=9)
    axes[0, 0].set_ylabel(f"image input (centre half, binning {image_cfg.binning})")
    axes[1, 0].set_ylabel("standardized log power")
    _save(fig, out / "stage4-inputs.png", dpi)
    return {"val_indices": picks, "image_shape": list(image.input_shape)}


def _training_row(run: Path) -> dict:
    """Numbers of a training run and, if predicted, of its validation predictions."""
    summary = json.loads((run / "summary.json").read_text())
    history = read_history(run / "history.csv")
    info = _run_info(run)
    row = {
        "kind": info.get("config", {}).get("representation", {}).get("kind"),
        "n_train": info.get("n_train"),
        "n_parameters": info.get("n_parameters"),
        "best_epoch": summary["best_epoch"],
        "epochs": len(history),
        "val_error_um": summary["validation"]["error_um"],
        "val_interval": summary["validation"]["interval"],
        "sigma_scale": summary["sigma_scale"],
        "train_seconds": summary["train_seconds"],
        "seconds_per_epoch": float(np.mean([h["seconds"] for h in history])),
    }
    predictions = run / "val" / "summary.json"
    if predictions.exists():
        val = json.loads(predictions.read_text())
        row["val_fresnel_error_rel"] = val["fresnel_error_rel"]
        row["seconds_per_estimate"] = val["seconds_per_estimate"]["median"]
    return row


def _seed_average(runs: list[Path]) -> dict:
    """Validation metrics of the mean of several models (Gaussian mixture moments)."""
    tables = [
        {r["index"]: r for r in read_predictions(run / "val" / "predictions.csv")} for run in runs
    ]
    indices = sorted(tables[0])
    est = np.array([[t[i]["dz01_est"] for i in indices] for t in tables])
    var = np.array([[t[i]["sigma"] ** 2 for i in indices] for t in tables])
    truth = np.array([tables[0][i]["dz01_true"] for i in indices])
    err = (est.mean(axis=0) - truth) * 1e6
    sigma = np.sqrt(var.mean(axis=0) + est.var(axis=0)) * 1e6
    return {
        "runs": [run.name for run in runs],
        "error_um": point_metrics(err),
        "interval": interval_metrics(err, sigma),
    }


def _family(info: dict, any_seed: bool = False) -> str | None:
    """Learning-curve family of a run: representation and channels.

    Runs with another seed or ``beta`` than the configs (0 and 0.5) are variants without a
    family; ``any_seed`` admits other seeds.
    """
    config = info.get("config", {})
    training = config.get("training", {})
    if training.get("beta") != 0.5 or (training.get("seed") != 0 and not any_seed):
        return None
    channels = "/".join(str(c) for c in config.get("model", {}).get("channels", []))
    return f"{config.get('representation', {}).get('kind')} CNN, {channels}"


def training_figure(finals: list[Path], study: Path, out: Path, dpi: int) -> dict | None:
    runs = {run.name: run for run in finals if (run / "summary.json").exists()}
    for file in sorted(study.glob("*/summary.json")):
        runs.setdefault(file.parent.name, file.parent)
    if not runs:
        return None
    rows = {name: _training_row(run) for name, run in runs.items()}
    infos = {name: _run_info(run) for name, run in runs.items()}
    fig, axes = plt.subplots(
        1, 3, figsize=(17.5, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1.2, 1, 1.2]}
    )
    final_colors = ("tab:blue", "navy", "tab:cyan")
    curves = [(run.name, final_colors[i % 3]) for i, run in enumerate(finals) if run.name in runs]
    curves += [
        ("spectrum-n2000", "lightskyblue"),
        ("image-n10000", "tab:green"),
        ("image-n2000", "yellowgreen"),
    ]
    for name, color in curves:
        if name not in runs:
            continue
        history = read_history(runs[name] / "history.csv")
        label = f"{name} ({rows[name]['kind']}, n = {rows[name]['n_train']})"
        mae = [h["val_mae_um"] for h in history]
        axes[0].semilogy([h["epoch"] for h in history], mae, color=color, lw=1, label=label)
        best = rows[name]["best_epoch"]
        axes[0].plot(best, mae[best - 1], "o", color=color)
    axes[0].axhline(2500, color="0.6", ls=":", lw=1)
    axes[0].text(1, 2600, "guess dz01 = 0 (MAE 2.5 mm)", fontsize=8, color="0.4")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation MAE (um)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("validation MAE per epoch (dot: selected epoch)", fontsize=10)

    metrics: dict = {"runs": rows, "learning_curves": {}}
    families: dict[str, dict[int, tuple[float, str]]] = {}
    for name in runs:
        family = _family(infos[name])
        if family is not None and rows[name]["n_train"]:
            families.setdefault(family, {}).setdefault(
                rows[name]["n_train"], (rows[name]["val_error_um"]["mae"], name)
            )
    spectrum_colors = iter(("tab:blue", "navy", "tab:cyan", "tab:purple"))
    colors = {
        family: "tab:green" if family.startswith("image") else next(spectrum_colors, "0.4")
        for family in sorted(families, key=lambda f: (f.startswith("image"), f))
    }
    for family, color in colors.items():
        points = sorted(families[family].items())
        n = np.array([p[0] for p in points], dtype=float)
        mae = np.array([p[1][0] for p in points])
        entry: dict = {
            "points": [
                {"n_train": int(a), "val_mae_um": float(b), "run": run} for a, (b, run) in points
            ]
        }
        label = family
        if len(points) >= 3:
            entry["slope"] = float(np.polyfit(np.log(n), np.log(mae), 1)[0])
            label += f" (MAE ~ n^{entry['slope']:.2f})"
        metrics["learning_curves"][family] = entry
        axes[1].loglog(n, mae, "o-", color=color, label=label)
        for x, y in zip(n, mae, strict=True):
            axes[1].annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(5, 3))
    seeds = [
        name
        for name in runs
        if _family(infos[name]) is None and _family(infos[name], any_seed=True) in colors
    ]
    for i, name in enumerate(seeds):
        axes[1].plot(
            rows[name]["n_train"],
            rows[name]["val_error_um"]["mae"],
            "x",
            color=colors[_family(infos[name], any_seed=True)],
            label="other seeds" if i == 0 else None,
        )
    if families:
        axes[1].legend(fontsize=8)
    axes[1].set_title("learning curves (validation, seed 0)", fontsize=10)
    axes[1].set_xlabel("training samples")
    axes[1].set_ylabel("validation MAE (um)")

    variants = [n for n in rows if rows[n]["n_train"] == 2000]
    if variants:
        labels = list(variants)
        mae = [rows[n]["val_error_um"]["mae"] for n in variants]
        colors = ["tab:green" if rows[n]["kind"] == "image" else "tab:blue" for n in variants]
        notes = [rows[n].get("seconds_per_estimate") for n in variants]
        seeds = [runs[n] for n in SEED_RUNS if n in runs]
        if len(seeds) == len(SEED_RUNS) and all((r / "val").is_dir() for r in seeds):
            metrics["seed_average"] = _seed_average(seeds)
            labels.append("mean of 3 seeds")
            mae.append(metrics["seed_average"]["error_um"]["mae"])
            colors.append("lightsteelblue")
            notes.append(None)
        axes[2].bar(range(len(labels)), mae, color=colors)
        for i, ms in enumerate(notes):
            note = f"{mae[i]:.0f}" + ("" if ms is None else f"\n{1e3 * ms:.1f} ms")
            axes[2].text(i, mae[i] * 1.02, note, ha="center", va="bottom", fontsize=8)
        axes[2].set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
        axes[2].set_ylim(0, 1.25 * max(mae))
    axes[2].set_ylabel("validation MAE (um)")
    axes[2].set_title("variants at n = 2000 (MAE; time per estimate)", fontsize=10)
    _save(fig, out / "stage4-training.png", dpi)
    return metrics


def test_figure(predictions: Path, out: Path, dpi: int) -> dict | None:
    if not predictions.exists():
        return None
    rows = read_predictions(predictions)
    dz = np.array([r["dz01_true"] for r in rows]) * 1e3
    est = np.array([r["dz01_est"] for r in rows]) * 1e3
    sigma = np.array([r["sigma"] for r in rows]) * 1e3
    err = np.array([r["error_um"] for r in rows])
    colors = [TYPE_COLORS[PHANTOM_TYPES[r["phantom_type"]]] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), layout="constrained")
    axes[0].errorbar(dz, est, yerr=sigma, fmt="none", ecolor="0.7", lw=0.6)
    axes[0].scatter(dz, est, c=colors, s=8)
    axes[0].plot([-5, 5], [-5, 5], color="0.3", lw=0.8)
    axes[0].set_xlabel("true dz01 (mm)")
    axes[0].set_ylabel("estimated dz01 (mm), bars: 1 sigma")
    axes[1].scatter(dz, err, c=colors, s=8)
    axes[1].axhline(0, color="0.3", lw=0.8)
    axes[1].set_xlabel("true dz01 (mm)")
    axes[1].set_ylabel("error (um)")
    for name, color in TYPE_COLORS.items():
        axes[1].plot([], [], "o", color=color, ms=4, label=name)
    axes[1].legend(fontsize=8)
    m = point_metrics(err)
    lim = max(1.0, np.percentile(np.abs(err), 99))
    axes[2].hist(err, bins=np.linspace(-lim, lim, 41), color="tab:blue", alpha=0.7)
    axes[2].set_xlabel("error (um)")
    axes[2].set_title(
        f"n {m['n']}: bias {m['bias']:+.0f}, MAE {m['mae']:.0f}, RMSE {m['rmse']:.0f}, "
        f"median {m['median_abs']:.0f} um",
        fontsize=9,
    )
    fig.suptitle(f"learned estimate on the synthetic test split ({predictions.parents[1].name})")
    _save(fig, out / "stage4-test.png", dpi)
    return json.loads((predictions.parent / "summary.json").read_text())


def calibration_figure(model: Path, out: Path, dpi: int) -> dict | None:
    sets = {s: model / s / "predictions.csv" for s in ("val", "test")}
    sets = {s: p for s, p in sets.items() if p.exists()}
    if "test" not in sets:
        return None
    levels = np.linspace(0.05, 0.99, 48)
    quantiles = np.array([interval_quantile(p) for p in levels])
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), layout="constrained")
    metrics = {}
    for split, path in sets.items():
        rows = read_predictions(path)
        z = np.abs([r["z_score"] for r in rows])
        coverage = [float(np.mean(z <= q)) for q in quantiles]
        axes[0].plot(levels, coverage, label=f"{split} (calibrated)")
        metrics[split] = {"n": len(rows), "levels": levels.round(3).tolist(), "coverage": coverage}
        if split == "test":
            raw = np.abs([r["error_um"] * 1e-6 / r["sigma_raw"] for r in rows])
            axes[0].plot(levels, [np.mean(raw <= q) for q in quantiles], "--", label="test (raw)")
            err = np.abs([r["error_um"] for r in rows])
            sig = np.array([r["sigma"] * 1e6 for r in rows])
            axes[1].loglog(sig, np.maximum(err, 1.0), ".", ms=4)
            grid = np.array([sig.min(), sig.max()])
            axes[1].plot(grid, grid, color="0.3", lw=0.8, label="|e| = sigma")
            axes[1].plot(grid, 1.96 * grid, color="0.3", ls="--", lw=0.8, label="|e| = 1.96 sigma")
            axes[1].legend(fontsize=8)
            zs = np.array([r["z_score"] for r in rows])
            axes[2].hist(zs, bins=np.linspace(-4, 4, 33), density=True, alpha=0.7)
            x = np.linspace(-4, 4, 200)
            axes[2].plot(x, np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi), color="0.3")
            spearman = np.corrcoef(np.argsort(np.argsort(sig)), np.argsort(np.argsort(err)))[0, 1]
            metrics["test"]["spearman_sigma_abs_error"] = float(spearman)
    axes[0].plot([0, 1], [0, 1], color="0.3", lw=0.8)
    axes[0].set_xlabel("nominal level of the central interval")
    axes[0].set_ylabel("observed coverage")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("predicted sigma (um)")
    axes[1].set_ylabel("|error| (um)")
    axes[2].set_xlabel("z = error / sigma (test)")
    _save(fig, out / "stage4-calibration.png", dpi)
    return metrics


def dependence_figure(model: Path, comparison: Path, out: Path, dpi: int) -> dict | None:
    path = model / "test" / "predictions.csv"
    if not path.exists():
        return None
    rows = read_predictions(path)
    comp = _read_comparison(comparison)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), layout="constrained")
    photons = np.array([r["photons"] for r in rows])
    err = np.abs([r["error_um"] for r in rows])
    axes[0].loglog(photons, np.maximum(err, 1.0), ".", color="tab:blue", alpha=0.35, ms=4)
    metrics: dict = {"by_photons": {}, "by_phantom_type": {}}
    centers = np.sqrt(np.array(PHOTON_EDGES[:-1]) * np.array(PHOTON_EDGES[1:]))
    series = {"learned": (photons, err)}
    if comp:
        series["raw"] = (
            np.array([r["photons"] for r in comp]),
            np.abs([r["raw_error_um"] for r in comp]),
        )
        series["corrected"] = (series["raw"][0], np.abs([r["corrected_error_um"] for r in comp]))
        axes[0].loglog(
            series["raw"][0],
            np.maximum(series["raw"][1], 1.0),
            "s",
            ms=3,
            color=METHOD_STYLE["raw"][1],
            alpha=0.5,
        )
    for key, (n, e) in series.items():
        labels = np.array(photon_bin_labels(n))
        medians = [
            float(np.median(e[labels == b])) if np.any(labels == b) else np.nan
            for b in photon_bin_names()
        ]
        name, color = METHOD_STYLE[key]
        axes[0].plot(centers, medians, "o-", color=color, lw=2, label=f"{name}: median")
        metrics["by_photons"][key] = dict(zip(photon_bin_names(), medians, strict=True))
    axes[0].set_xlabel("flat-field counts per pixel")
    axes[0].set_ylabel("|error| (um)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"learned: {len(rows)} test samples; baseline: {len(comp)} samples")
    groups = {"learned (test)": (rows, "tab:blue")}
    if comp:
        indices = {r["index"] for r in comp}
        picked = [r for r in rows if r["index"] in indices]
        groups["learned (baseline samples)"] = (picked, "lightskyblue")
    bars = [
        (k, [r["error_um"] for r in v], [r["phantom_type"] for r in v], color)
        for k, (v, color) in groups.items()
    ]
    if comp:
        types = [r["phantom_type"] for r in comp]
        for key, column in (("raw", "raw_error_um"), ("corrected", "corrected_error_um")):
            name, color = METHOD_STYLE[key]
            bars.append((name, [r[column] for r in comp], types, color))
    width = 0.8 / len(bars)
    x = np.arange(len(PHANTOM_TYPES))
    for j, (label, errors, types, color) in enumerate(bars):
        errors, types = np.asarray(errors), np.asarray(types)
        mae = [
            float(np.mean(np.abs(errors[types == t]))) if np.any(types == t) else np.nan
            for t in range(len(PHANTOM_TYPES))
        ]
        offset = (j - (len(bars) - 1) / 2) * width
        axes[1].bar(x + offset, mae, width, label=label, color=color)
        metrics["by_phantom_type"][label] = dict(zip(PHANTOM_TYPES, mae, strict=True))
    axes[1].set_xticks(x, PHANTOM_TYPES)
    axes[1].set_ylabel("MAE (um)")
    axes[1].legend(fontsize=8)
    _save(fig, out / "stage4-dependence.png", dpi)
    return metrics


def comparison_figure(comparison: Path, baseline_val: Path, out: Path, dpi: int) -> dict | None:
    rows = _read_comparison(comparison)
    if not rows:
        return None
    summary = json.loads((comparison / "comparison.json").read_text())
    val_rows = read_rows(baseline_val / "results.csv")
    has_warm = rows[0]["warm_error_um"] is not None
    columns = {"baseline_val": [r["error_um"] for r in val_rows]}
    for key in ("raw", "corrected", "learned") + (("warm", "warm_corrected") if has_warm else ()):
        columns[key] = [r[f"{key}_error_um"] for r in rows]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 5.0),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.6, 1.0, 1.0]},
    )
    rng = np.random.default_rng(0)
    for j, (key, values) in enumerate(columns.items()):
        name, color = METHOD_STYLE[key]
        values = np.asarray(values, dtype=float)
        axes[0].boxplot([values], positions=[j], widths=0.5, showfliers=False)
        axes[0].scatter(j + rng.uniform(-0.15, 0.15, len(values)), values, s=10, color=color)
    offset = summary["offset_um"]
    axes[0].axhspan(offset["ci"][0], offset["ci"][1], color="tab:orange", alpha=0.15)
    axes[0].axhline(offset["mean"], color="tab:orange", lw=0.8, ls="--")
    axes[0].axhline(0, color="0.3", lw=0.8)
    axes[0].set_xticks(range(len(columns)), [METHOD_STYLE[k][0] for k in columns], fontsize=8)
    axes[0].set_ylabel("error (um)")
    axes[0].set_title(
        f"offset {offset['mean']:+.0f} +- {offset['se']:.0f} um (n = {offset['n']}, "
        f"95 % CI {offset['ci'][0]:+.0f} .. {offset['ci'][1]:+.0f})",
        fontsize=9,
    )
    learned = np.abs(columns["learned"])
    for key, marker in (("raw", "s"), ("corrected", "o")):
        name, color = METHOD_STYLE[key]
        axes[1].loglog(
            np.maximum(np.abs(columns[key]), 1.0),
            np.maximum(learned, 1.0),
            marker,
            ms=4,
            color=color,
            label=name,
        )
    lim = [1.0, 1.2 * max(learned.max(), np.abs(columns["raw"]).max(), 10.0)]
    axes[1].plot(lim, lim, color="0.3", lw=0.8)
    axes[1].set_xlabel("|error| of the baseline (um)")
    axes[1].set_ylabel("|error| of the learned model (um)")
    axes[1].legend(fontsize=8)
    methods = summary["methods"]
    names, seconds, colors = ["learned"], [methods["learned"]["seconds_median"]], ["tab:blue"]
    names.append("baseline [2]")
    seconds.append(methods["baseline_raw"]["seconds_median"])
    colors.append("tab:red")
    if has_warm:
        names.append("warm start")
        seconds.append(
            methods["warm_start"]["seconds_median"] + methods["learned"]["seconds_median"]
        )
        colors.append("tab:purple")
    axes[2].bar(names, seconds, color=colors)
    axes[2].set_yscale("log")
    for i, s in enumerate(seconds):
        axes[2].text(i, s * 1.3, f"{s:.3g} s", ha="center", fontsize=9)
    axes[2].set_ylabel("time per estimate (s, median)")
    _save(fig, out / "stage4-comparison.png", dpi)
    return summary


def warm_figure(comparison: Path, baseline: Path, warm: Path, out: Path, dpi: int) -> dict | None:
    rows = _read_comparison(comparison)
    if not rows or rows[0]["warm_error_um"] is None or not (warm / "traces.jsonl").exists():
        return None
    raw_traces = {t["index"]: t for t in read_traces(baseline / "traces.jsonl")}
    warm_traces = {t["index"]: t for t in read_traces(warm / "traces.jsonl")}
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), layout="constrained")
    raw = np.array([r["raw_error_um"] for r in rows])
    hot = np.array([r["warm_error_um"] for r in rows])
    axes[0].scatter(raw, hot, s=12, color="tab:purple")
    lim = [min(raw.min(), hot.min()) - 20, max(raw.max(), hot.max()) + 20]
    axes[0].plot(lim, lim, color="0.3", lw=0.8)
    axes[0].set_xlabel("error of the baseline (um)")
    axes[0].set_ylabel("error of the warm start (um)")
    n_raw = [r["raw_reconstructions"] for r in rows]
    n_warm = [r["warm_reconstructions"] for r in rows]
    edges = np.arange(min(n_raw + n_warm) - 0.5, max(n_raw + n_warm) + 1.5)
    axes[1].hist(
        [n_raw, n_warm],
        bins=edges,
        label=["baseline [2]", "warm start"],
        color=["tab:red", "tab:purple"],
    )
    axes[1].set_xlabel("reconstructions per autofocus")
    axes[1].legend(fontsize=8)
    for k, row in enumerate(rows[:3]):
        for traces, color in ((raw_traces, "tab:red"), (warm_traces, "tab:purple")):
            trace = traces.get(row["index"])
            if trace is None:
                continue
            z = [
                (e["z01"] - trace["z01_guess"]) * 1e3
                for e in trace["evaluations"]
                if e["reconstructed"]
            ]
            axes[2].plot(np.arange(1, len(z) + 1), z, "o-", ms=3, color=color, alpha=0.4 + 0.2 * k)
        axes[2].axhline(row["dz01_true_um"] * 1e-3, color="0.3", lw=0.6, ls=":")
    axes[2].set_xlabel("reconstruction")
    axes[2].set_ylabel("probed dz01 (mm); dotted: truth")
    axes[2].set_title("first three samples: baseline (red), warm start (purple)", fontsize=9)
    _save(fig, out / "stage4-warmstart.png", dpi)
    return {
        "reconstructions_raw": float(np.mean(n_raw)),
        "reconstructions_warm": float(np.mean(n_warm)),
        "warm": point_metrics(hot),
        "raw": point_metrics(raw),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="low resolution (CI)")
    parser.add_argument("--main", type=Path, default=RESULTS / "learned" / "spectrum-cnn")
    parser.add_argument(
        "--finals",
        type=Path,
        nargs="+",
        default=[RESULTS / "learned" / "spectrum-cnn", RESULTS / "learned" / "spectrum-cnn-wide"],
        help="models trained on all samples (training figure, with --main)",
    )
    parser.add_argument("--study", type=Path, default=RESULTS / "learned" / "study")
    parser.add_argument("--comparison", type=Path, default=RESULTS / "comparison" / "cpu-4x-test")
    parser.add_argument("--baseline", type=Path, default=RESULTS / "baseline" / "cpu-4x-test")
    parser.add_argument("--baseline-val", type=Path, default=RESULTS / "baseline" / "cpu-4x-val")
    parser.add_argument("--warm", type=Path, default=RESULTS / "baseline" / "cpu-4x-test-warm")
    parser.add_argument("--data", type=Path, default=REPO / "data" / "cpu-4x")
    args = parser.parse_args()
    dpi = 60 if args.quick else 120
    model = args.main
    finals = list(dict.fromkeys([model, *args.finals]))
    runs = [*finals, *sorted(p for p in args.study.glob("*") if p.is_dir())]
    args.out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"quick": args.quick, "main": str(model)}
    steps = [
        ("inputs", lambda: inputs_figure(args.data, runs, args.out, dpi)),
        ("training", lambda: training_figure(finals, args.study, args.out, dpi)),
        ("test", lambda: test_figure(model / "test" / "predictions.csv", args.out, dpi)),
        ("calibration", lambda: calibration_figure(model, args.out, dpi)),
        ("dependence", lambda: dependence_figure(model, args.comparison, args.out, dpi)),
        (
            "comparison",
            lambda: comparison_figure(args.comparison, args.baseline_val, args.out, dpi),
        ),
        (
            "warmstart",
            lambda: warm_figure(args.comparison, args.baseline, args.warm, args.out, dpi),
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
    with open(args.out / "stage4-metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
