"""Stage 5: the learned autofocus on the measured holograms and the sim-to-real gap (plan step 8).

* ``measured``: the spectrum CNNs of the matched datasets (``results/learned/matched-<sample>``
  and, if trained, the variant ``matched-<sample>-hf``) on the measured hologram, binned 4x as
  in Stage 3. The variant with the smaller validation MAE is the estimate of the sample; the
  record has the calibrated 68/95 % intervals, the distances to [2] Table 1, the example setup
  of [1] and the baselines of Stage 3, the spread over the 16 phases of the binning grid and
  the time per estimate (``results/measured/cnn/<sample>.json``). A model trained on the
  realistic set (``matched-<sample>-realistic``) is evaluated too but never selected: its
  validation split has other assumptions.
* ``perturb``: the test split of a matched dataset regenerated with one simulator assumption
  changed (:data:`holofocus.learned.sim2real.PERTURBATIONS`; ``data/sim2real/<sample>/<name>.h5``,
  not tracked) and predicted by the CNN (``results/sim2real/<sample>/<name>/``). The paired
  shifts and the errors against the unchanged test split go to
  ``results/sim2real/<sample>.{json,csv}``.
* ``spectra``: features of the radial power spectrum of the measured hologram against those of
  the matched test split and of the perturbed sets, and the CNN inputs (5/50/95 % per bin)
  (``results/sim2real/spectra-<sample>.json``).
* ``compare``: every estimate on the measured holograms in one table, with the error of the
  CNN on the matched test split (``results/measured/comparison.{json,csv}``).
* ``noise``: noise power spectra of the background corners of the measured holograms, at full
  resolution with the fit of a Gaussian-blurred plus a white part (the detector PSF) and on
  the 4x grid with the fit of the correlated residual of the simulator
  (``results/sim2real/noise.json``).

Usage::

    python scripts/run_stage5_study.py --threads 1 measured --holograms DIR
    python scripts/run_stage5_study.py --threads 3 perturb --samples cactus_needle tooth
    python scripts/run_stage5_study.py spectra --holograms DIR
    python scripts/run_stage5_study.py compare
    python scripts/run_stage5_study.py noise --holograms DIR
    # models trained elsewhere (docs/maxwell-gpu.md, 7.1), against the baselines of results/
    python scripts/run_stage5_study.py --learned L measured --holograms DIR --out M/cnn
    python scripts/run_stage5_study.py --learned L compare --cnn M/cnn --out M

``DIR`` is ``Examples/data/holograms`` of the Zenodo record 10.5281/zenodo.8349365.
``perturb`` reuses perturbed splits and predictions that still match the config and the model;
the other commands rewrite their outputs. The warm start (``run_stage3_experiments.py measured
--warm-start``) records the SHA-256 of the ``measured`` output it starts from.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from holofocus.baseline.runner import file_sha256
from holofocus.dataset import DatasetConfig, HologramGenerator, read_config, read_labels
from holofocus.dataset.storage import _git_commit, write_split
from holofocus.learned import (
    LearnedAutofocus,
    predict_split,
    read_predictions,
    summarize_predictions,
)
from holofocus.learned.sim2real import (
    CTF_BANDS,
    FEATURE_NAMES,
    MEASURED_MODELS,
    PERTURBATIONS,
    RadialPower,
    binning_phases,
    corner_spectra,
    feature_distribution,
    fit_blurred_white,
    fit_lorentzian,
    measured_estimate,
    paired_shifts,
    percentile_rank,
    perturbation_changes,
    perturbed_config,
    spectral_features,
)
from holofocus.measured import MEASURED_SAMPLES, load_measured

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
BINNING = 4
VARIANTS = ("", "-hf")
REALISTIC = "-realistic"
BASELINES = {"baseline_b1_p4": "b1-p4", "baseline_b1_p2": "b1-p2"}
WARM = "warm-b1-p4"
PROFILE_POINTS = 256


def environment() -> dict:
    return {
        "git_commit": _git_commit(),
        "torch_version": str(torch.__version__),
        "threads": torch.get_num_threads(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
    }


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()})


def runs_of(learned: Path, name: str, suffixes: tuple[str, ...] = VARIANTS) -> dict[str, Path]:
    """Trained variants of the matched model of ``name`` (run name -> directory)."""
    runs = {}
    for suffix in suffixes:
        run = learned / f"matched-{name}{suffix}"
        if (run / "model.pt").exists():
            runs[run.name] = run
    return runs


def validation_mae(run: Path) -> float:
    return float(json.loads((run / "summary.json").read_text())["validation"]["error_um"]["mae"])


def selected_run(learned: Path, name: str) -> Path:
    """The variant with the smallest validation MAE (simulated data only)."""
    runs = runs_of(learned, name)
    if not runs:
        raise FileNotFoundError(f"no trained model matched-{name}* in {learned}")
    return min(runs.values(), key=validation_mae)


def baseline_estimates(measured: Path, name: str) -> dict[str, float]:
    out = {}
    for key, run in BASELINES.items():
        path = measured / run / f"{name}.json"
        if path.exists():
            out[key] = float(json.loads(path.read_text())["result"]["z01"])
    return out


def cmd_measured(args) -> None:
    for name in args.samples:
        target = args.out / f"{name}.json"
        sample = MEASURED_SAMPLES[name]
        start = time.perf_counter()
        hologram = load_measured(args.holograms, sample, BINNING)
        load_seconds = time.perf_counter() - start
        native = load_measured(args.holograms, sample, 1).numpy()
        phases = binning_phases(native, BINNING)
        references = {
            "paper2_nm": sample.z01_nm,
            "paper2_visual": sample.z01_visual,
            "paper1_example": sample.z01_example,
            **baseline_estimates(args.measured, name),
        }
        selected = selected_run(args.learned, name)
        variants = {}
        for run_name, run in runs_of(args.learned, name, (*VARIANTS, REALISTIC)).items():
            autofocus = LearnedAutofocus.load(run, args.device)
            estimate = measured_estimate(
                autofocus, hologram, sample, references, args.repeats, phases
            )
            variants[run_name] = {
                "model": str(run.relative_to(REPO) if run.is_relative_to(REPO) else run),
                "model_sha256": file_sha256(run / "model.pt"),
                "config_name": autofocus.config.name,
                "representation": dataclasses.asdict(autofocus.config.representation),
                "validation_mae_um": validation_mae(run),
                **estimate,
            }
        best = variants[selected.name]
        record = {
            "sample": name,
            "label": sample.label,
            "binning": BINNING,
            "cutting_band": sample.cutting_band,
            "input_shape": list(hologram.shape),
            "geometry_guess": sample.geometry(BINNING).as_dict(),
            "paper2": {
                "simplex": list(sample.simplex),
                "z01_nm": sample.z01_nm,
                "z01_visual": sample.z01_visual,
                "samples": sample.nm_samples,
                "padding_factor": sample.padding_factor,
            },
            "selected": selected.name,
            "selection": "smallest validation MAE on the simulated validation split, among "
            f"the models of the matched dataset ({', '.join(runs_of(args.learned, name))})",
            "z01_est": best["z01_est"],
            "sigma": best["sigma"],
            "load_seconds": load_seconds,
            "variants": variants,
            **environment(),
        }
        write_json(target, record)
        for run_name, variant in variants.items():
            ref = variant["references"]
            print(
                f"{name}: {run_name}{' (selected)' if run_name == selected.name else ''} z01 = "
                f"{variant['z01_est'] * 1e3:.3f} mm, sigma {variant['sigma'] * 1e6:.0f} um; "
                f"minus [2] {ref['paper2_nm']['distance_um']:+.0f} um"
                + (
                    f", minus b1-p4 {ref['baseline_b1_p4']['distance_um']:+.0f} um"
                    if "baseline_b1_p4" in ref
                    else ""
                )
                + f"; {1e3 * variant['seconds_per_estimate']['median']:.1f} ms"
            )


def regeneration_check(path: Path, index: int = 0) -> dict:
    """Sample ``index`` of the split regenerated with this code against the stored one."""
    config = read_config(path)
    with h5py.File(path, "r") as fh:
        split = str(fh.attrs["split"])
        stored = np.asarray(fh["hologram"][index])
    sample = HologramGenerator(config, device="cpu").sample(split, index)
    return {
        "index": index,
        "hologram_max_difference": float(np.abs(sample.hologram - stored).max()),
    }


def perturbation_row(name: str, summary: dict, shifts: dict | None) -> dict:
    err, interval = summary["error_um"], summary["interval"]
    row = {
        "perturbation": name,
        "n": summary["n_samples"],
        "mae_um": err["mae"],
        "median_abs_um": err["median_abs"],
        "bias_um": err["bias"],
        "within_100": err["within"]["100"],
        "coverage_68": interval["coverage"]["68"],
        "coverage_95": interval["coverage"]["95"],
        "sigma_median_um": interval["sigma_median"],
    }
    s = (shifts or {}).get("shift_um", {})
    row.update(
        {
            "shift_mean_um": s.get("mean", 0.0),
            "shift_median_um": s.get("median", 0.0),
            "shift_std_um": s.get("std", 0.0),
            "shift_mean_abs_um": s.get("mean_abs", 0.0),
            "mae_change_um": (shifts or {}).get("mae_change_um", 0.0),
        }
    )
    return row


def set_is_current(path: Path, config: DatasetConfig, n: int) -> bool:
    """Whether the perturbed split ``path`` was made from ``config`` with at least ``n`` samples."""
    return (
        path.exists()
        and read_config(path).to_dict() == config.to_dict()
        and len(read_labels(path)["dz01"]) >= n
    )


def predictions_are_current(out: Path, model_sha256: str, data: Path) -> bool:
    """Whether ``out`` holds the predictions of the model ``model_sha256`` on ``data``."""
    info = out / "run.yaml"
    if not (out / "summary.json").exists() or not info.exists():
        return False
    run = yaml.safe_load(info.read_text())
    dataset = run.get("dataset") or {}
    return run.get("model_sha256") == model_sha256 and dataset.get("sha256") == file_sha256(data)


def cmd_perturb(args) -> None:
    if (args.data or args.model or args.reference) and len(args.samples) != 1:
        raise SystemExit("--data, --model and --reference need exactly one sample")
    for name in args.samples:
        data = args.data or REPO / "data" / f"matched-{name}" / "test.h5"
        model = args.model or selected_run(args.learned, name)
        model_sha256 = file_sha256(model / "model.pt")
        reference_path = args.reference or model / "test" / "predictions.csv"
        config = read_config(data)
        n = len(read_labels(data)["dz01"])
        n = n if args.limit is None else min(n, args.limit)
        autofocus = LearnedAutofocus.load(model, args.device)
        reference = [r for r in read_predictions(reference_path) if r["index"] < n]
        target = args.out / f"{name}.json"
        record = json.loads(target.read_text()) if target.exists() else {}
        if record.get("model_sha256") != model_sha256 or record.get("data") != str(data):
            record = {
                "sample": name,
                "data": str(data),
                "model": str(model),
                "model_sha256": model_sha256,
                "reference": str(reference_path),
                "measured_model": dataclasses.asdict(MEASURED_MODELS[name]),
                "regeneration": regeneration_check(data),
                "perturbations": {},
                **environment(),
            }
        record["measured_model"] = dataclasses.asdict(MEASURED_MODELS[name])
        reference_summary = summarize_predictions(reference)
        record["reference"] = {
            "path": str(reference_path),
            "n": len(reference),
            "error_um": reference_summary["error_um"],
            "interval": reference_summary["interval"],
        }
        for pert in list(record["perturbations"]):
            entry = record["perturbations"][pert]
            cfg = perturbed_config(config, entry["changes"], f"{config.name}-{pert}")
            if entry["changes"] != perturbation_changes(pert, name) or not (
                set_is_current(args.sets / name / f"{pert}.h5", cfg, 0)
                and predictions_are_current(
                    args.out / name / pert, model_sha256, args.sets / name / f"{pert}.h5"
                )
            ):
                del record["perturbations"][pert]
        for pert in args.perturbations:
            sets = args.sets / name / f"{pert}.h5"
            out = args.out / name / pert
            changes = perturbation_changes(pert, name)
            cfg = perturbed_config(config, changes, f"{config.name}-{pert}")
            if not set_is_current(sets, cfg, n):
                start = time.perf_counter()
                write_split(cfg, "test", sets, n_samples=n, device="cpu", log=None)
                print(f"{name} {pert}: {n} samples in {time.perf_counter() - start:.0f} s")
            if not predictions_are_current(out, model_sha256, sets):
                predict_split(autofocus, sets, out, model_path=model, log=None)
            summary = json.loads((out / "summary.json").read_text())
            shifts = paired_shifts(reference, read_predictions(out / "predictions.csv"))
            record["perturbations"][pert] = {
                "description": PERTURBATIONS[pert],
                "changes": changes,
                "paired": shifts,
                "error_um": summary["error_um"],
                "interval": summary["interval"],
            }
            print(
                f"{name} {pert}: MAE {summary['error_um']['mae']:.0f} um "
                f"(reference {shifts['reference_error_um']['mae']:.0f}), shift "
                f"{shifts['shift_um']['mean']:+.0f} um (median {shifts['shift_um']['median']:+.0f})"
            )
        done = record["perturbations"]
        record["perturbations"] = {p: done[p] for p in PERTURBATIONS if p in done}
        write_json(target, record)
        rows = [
            perturbation_row("none", {"n_samples": len(reference), **record["reference"]}, None)
        ]
        for pert, entry in record["perturbations"].items():
            rows.append(
                perturbation_row(
                    pert, {"n_samples": entry["paired"]["n"], **entry}, entry["paired"]
                )
            )
        write_csv(args.out / f"{name}.csv", rows)


def _profile_quantiles(inputs: np.ndarray) -> dict:
    """5/50/95 % of the CNN input per bin, averaged to :data:`PROFILE_POINTS` points."""
    k = max(1, inputs.shape[1] // PROFILE_POINTS)
    coarse = inputs[:, : k * (inputs.shape[1] // k)].reshape(len(inputs), -1, k).mean(axis=2)
    q = np.percentile(coarse, [5, 50, 95], axis=0)
    return {"p05": q[0].tolist(), "p50": q[1].tolist(), "p95": q[2].tolist()}


def set_features(path: Path, radial: RadialPower, autofocus, limit: int | None) -> tuple:
    labels = read_labels(path)
    n = len(labels["dz01"]) if limit is None else min(limit, len(labels["dz01"]))
    features, inputs = [], []
    with h5py.File(path, "r") as fh:
        for i in range(n):
            hologram = np.asarray(fh["hologram"][i])
            features.append(spectral_features(radial, hologram, float(labels["fresnel_number"][i])))
            rep = autofocus.representation(torch.from_numpy(hologram))
            inputs.append(rep[0, 0].cpu().numpy())
    return features, np.stack(inputs)


def cmd_spectra(args) -> None:
    for name in args.samples:
        sample = MEASURED_SAMPLES[name]
        run = selected_run(args.learned, name)
        autofocus = LearnedAutofocus.load(run, args.device)
        hologram = load_measured(args.holograms, sample, BINNING)
        radial = RadialPower(tuple(hologram.shape))
        fr_nm = sample.geometry(BINNING, sample.z01_nm).fresnel_number
        measured = spectral_features(radial, hologram.numpy(), fr_nm)
        measured_input = autofocus.representation(hologram)[0, 0].cpu().numpy()[None]
        sets = {"matched": args.data_root / f"matched-{name}" / "test.h5"}
        sets.update(
            {
                p: args.sets / name / f"{p}.h5"
                for p in PERTURBATIONS
                if (args.sets / name / f"{p}.h5").exists()
            }
        )
        record = {
            "sample": name,
            "model": str(run),
            "fresnel_number_paper2_nm": fr_nm,
            "f2_axis": {"bins": len(radial.f2), "f2_max": float(radial.f2[-1] + radial.f2[0])},
            "ctf_bands": [list(b) for b in CTF_BANDS],
            "measured": measured,
            "measured_input": _profile_quantiles(measured_input)["p50"],
            "input_f2_max": autofocus.config.representation.f2_max,
            "sets": {},
        }
        for set_name, path in sets.items():
            features, inputs = set_features(path, radial, autofocus, args.limit)
            record["sets"][set_name] = {
                "path": str(path),
                "n": len(features),
                "features": feature_distribution(features),
                "measured_percentile": {
                    f: percentile_rank([x[f] for x in features], measured[f]) for f in FEATURE_NAMES
                },
                "ctf_bands_median": [
                    float(np.nanmedian([x["ctf_bands"][k] for x in features]))
                    for k in range(len(CTF_BANDS))
                ],
                "input": _profile_quantiles(inputs),
            }
            ranks = record["sets"][set_name]["measured_percentile"]
            print(
                f"{name} {set_name}: " + ", ".join(f"{f} {ranks[f]:.0f} %" for f in FEATURE_NAMES)
            )
        write_json(args.out / f"spectra-{name}.json", record)


def _get(record: dict, *keys, default=None):
    for key in keys:
        if not isinstance(record, dict) or key not in record:
            return default
        record = record[key]
    return record


def cmd_compare(args) -> None:
    rows, details = [], {}
    cnn_dir = args.cnn or args.measured / "cnn"
    out = args.out or args.measured
    for name in args.samples:
        sample = MEASURED_SAMPLES[name]
        cnn_path = cnn_dir / f"{name}.json"
        if not cnn_path.exists():
            print(f"{name}: no CNN estimate, skipped")
            continue
        cnn = json.loads(cnn_path.read_text())
        best = cnn["variants"][cnn["selected"]]
        run = args.learned / cnn["selected"]
        test = json.loads((run / "test" / "summary.json").read_text())
        test_rows = read_predictions(run / "test" / "predictions.csv")
        abs_test = np.array([abs(r["error_um"]) for r in test_rows])
        cold = {}
        for key, sub in (("b1_p4", "b1-p4"), ("b1_p2", "b1-p2"), ("warm", WARM)):
            path = args.measured / sub / f"{name}.json"
            if path.exists():
                cold[key] = json.loads(path.read_text())
        # [2]: 6 s per reconstruction for type 2 (2048^2 arrays), 16 s for type 1 (3584^2).
        gpu_seconds = sample.nm_samples * (16.0 if sample.padding_factor > 4 else 6.0)
        row = {
            "sample": name,
            "z01_guess_mm": sample.z01_guess * 1e3,
            "paper2_nm_mm": sample.z01_nm * 1e3,
            "paper2_visual_mm": sample.z01_visual * 1e3,
            "paper1_example_mm": sample.z01_example * 1e3,
            "paper2_reconstructions": sample.nm_samples,
            "paper2_gpu_seconds_est": gpu_seconds,
            "cnn_run": cnn["selected"],
            "cnn_mm": cnn["z01_est"] * 1e3,
            "cnn_sigma_um": cnn["sigma"] * 1e6,
            "cnn_width68_um": best["intervals"]["68"]["width_um"],
            "cnn_width95_um": best["intervals"]["95"]["width_um"],
            "cnn_minus_paper2_um": best["references"]["paper2_nm"]["distance_um"],
            "cnn_z_paper2": best["references"]["paper2_nm"]["z_score"],
            "cnn_ms": 1e3 * best["seconds_per_estimate"]["median"],
            "cnn_phase_std_um": _get(best, "binning_phases", "std_um", default=float("nan")),
            "test_mae_um": test["error_um"]["mae"],
            "test_median_abs_um": test["error_um"]["median_abs"],
            "test_within_100": test["error_um"]["within"]["100"],
            "test_coverage_68": test["interval"]["coverage"]["68"],
            "test_coverage_95": test["interval"]["coverage"]["95"],
            "test_sigma_median_um": test["interval"]["sigma_median"],
            "disagreement_paper2_percentile": percentile_rank(
                abs_test, abs(best["references"]["paper2_nm"]["distance_um"])
            ),
        }
        for key in ("b1_p4", "b1_p2", "warm"):
            if key not in cold:
                continue
            result = cold[key]["result"]
            z = result["z01"]
            row[f"{key}_mm"] = z * 1e3
            row[f"{key}_minus_paper2_um"] = (z - sample.z01_nm) * 1e6
            row[f"{key}_reconstructions"] = result["n_reconstructions"]
            row[f"{key}_seconds"] = result["seconds"]
        if "b1_p4" in cold:
            z = cold["b1_p4"]["result"]["z01"]
            row["cnn_minus_b1_p4_um"] = (cnn["z01_est"] - z) * 1e6
            row["cnn_z_b1_p4"] = (cnn["z01_est"] - z) / cnn["sigma"]
            row["disagreement_b1_p4_percentile"] = percentile_rank(
                abs_test, abs(row["cnn_minus_b1_p4_um"])
            )
            if "warm" in cold:
                row["warm_minus_b1_p4_um"] = (cold["warm"]["result"]["z01"] - z) * 1e6
        realistic = cnn["variants"].get(f"matched-{name}{REALISTIC}")
        if realistic is not None:
            row.update(
                {
                    "realistic_mm": realistic["z01_est"] * 1e3,
                    "realistic_sigma_um": realistic["sigma"] * 1e6,
                    "realistic_width68_um": realistic["intervals"]["68"]["width_um"],
                    "realistic_minus_paper2_um": realistic["references"]["paper2_nm"][
                        "distance_um"
                    ],
                    "realistic_z_paper2": realistic["references"]["paper2_nm"]["z_score"],
                }
            )
            if "b1_p4" in cold:
                row["realistic_minus_b1_p4_um"] = (
                    realistic["z01_est"] - cold["b1_p4"]["result"]["z01"]
                ) * 1e6
            own = args.learned / f"matched-{name}{REALISTIC}"
            for key, split in (("test", "test"), ("matched_test", "matched-test")):
                path = own / split / "summary.json"
                if path.exists():
                    summary = json.loads(path.read_text())
                    row[f"realistic_{key}_mae_um"] = summary["error_um"]["mae"]
                    row[f"realistic_{key}_within_100"] = summary["error_um"]["within"]["100"]
        rows.append(row)
        details[name] = {"row": row, "cnn": str(cnn_path), "test_summary": str(run / "test")}
    if not rows:
        raise SystemExit("no sample with a CNN estimate")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    full = [{k: r.get(k, "") for k in fields} for r in rows]
    write_csv(out / "comparison.csv", full)
    write_json(out / "comparison.json", {"samples": details, **environment()})
    for r in rows:
        print(
            f"{r['sample']}: CNN {r['cnn_mm']:.3f} mm ({r['cnn_minus_paper2_um']:+.0f} um from [2]"
            + (
                f", {r['cnn_minus_b1_p4_um']:+.0f} um from b1-p4"
                if "cnn_minus_b1_p4_um" in r
                else ""
            )
            + f"), test MAE {r['test_mae_um']:.0f} um"
        )


CORNER = 256
"""Corner size in detector pixels (64 binned pixels)."""


def cmd_noise(args) -> None:
    record = {"corner_pixels": CORNER, "binning": BINNING, "samples": {}}
    for name in args.samples:
        sample = MEASURED_SAMPLES[name]
        native = load_measured(args.holograms, sample, 1).numpy()
        full = corner_spectra(native, CORNER, 1, bins=48, f_min=0.02)
        binned = corner_spectra(native, CORNER, BINNING, bins=16, f_min=0.03)
        for corner in full:
            corner["fit"] = fit_blurred_white(corner["f"], corner["nps"])
        for corner in binned:
            corner["fit"] = fit_lorentzian(corner["f"], corner["nps"])
        sigma = [c["fit"]["sigma"] for c in full]
        rho = [c["fit"]["rho"] for c in binned]
        cutoff = [c["fit"]["cutoff"] for c in binned]
        means = [c["mean"] for c in full]
        mean_full = fit_blurred_white(full[0]["f"], np.mean([c["nps"] for c in full], axis=0))
        mean_binned = fit_lorentzian(binned[0]["f"], np.mean([c["nps"] for c in binned], axis=0))
        record["samples"][name] = {
            "full_resolution": full,
            "binned": binned,
            "psf_sigma_median": float(np.median(sigma)),
            "flat_field_rho_median": float(np.median(rho)),
            "flat_field_cutoff_median": float(np.median(cutoff)),
            "mean_spectrum_fit": {"full_resolution": mean_full, "binned": mean_binned},
            "corner_means": means,
        }
        print(
            f"{name}: PSF sigma {', '.join(f'{s:.2f}' for s in sigma)} px; residual "
            f"{', '.join(f'{100 * r:.1f}' for r in rho)} % with cutoff "
            f"{', '.join(f'{c:.2f}' for c in cutoff)}; corner means "
            f"{', '.join(f'{m:.2f}' for m in means)}; mean spectrum: PSF sigma "
            f"{mean_full['sigma']:.2f} px, residual {100 * mean_binned['rho']:.1f} % with "
            f"cutoff {mean_binned['cutoff']:.2f}"
        )
    record.update(environment())
    write_json(args.out / "noise.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, help="torch CPU threads")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or auto")
    parser.add_argument("--learned", type=Path, default=RESULTS / "learned")
    parser.add_argument("--measured", type=Path, default=RESULTS / "measured")
    sub = parser.add_subparsers(dest="command", required=True)
    samples = {"nargs": "+", "choices": list(MEASURED_SAMPLES), "default": list(MEASURED_SAMPLES)}

    me = sub.add_parser("measured", help="the CNNs on the measured holograms")
    me.add_argument("--holograms", type=Path, required=True)
    me.add_argument("--samples", **samples)
    me.add_argument("--repeats", type=int, default=20, help="timed estimates per model")
    me.add_argument("--out", type=Path, default=RESULTS / "measured" / "cnn")
    me.set_defaults(func=cmd_measured)

    pe = sub.add_parser("perturb", help="test splits with one simulator assumption changed")
    pe.add_argument("--samples", **samples)
    pe.add_argument(
        "--perturbations", nargs="+", choices=list(PERTURBATIONS), default=list(PERTURBATIONS)
    )
    pe.add_argument("--limit", type=int, help="only the first N test samples")
    pe.add_argument("--data", type=Path, help="test split (default: data/matched-<sample>)")
    pe.add_argument("--model", type=Path, help="model run (default: the selected variant)")
    pe.add_argument("--reference", type=Path, help="predictions.csv of the unchanged split")
    pe.add_argument("--sets", type=Path, default=REPO / "data" / "sim2real")
    pe.add_argument("--out", type=Path, default=RESULTS / "sim2real")
    pe.set_defaults(func=cmd_perturb)

    sp = sub.add_parser("spectra", help="spectral features, measured against simulated")
    sp.add_argument("--holograms", type=Path, required=True)
    sp.add_argument("--samples", **samples)
    sp.add_argument("--limit", type=int, help="only the first N samples of every set")
    sp.add_argument("--data-root", type=Path, default=REPO / "data")
    sp.add_argument("--sets", type=Path, default=REPO / "data" / "sim2real")
    sp.add_argument("--out", type=Path, default=RESULTS / "sim2real")
    sp.set_defaults(func=cmd_spectra)

    co = sub.add_parser("compare", help="all estimates on the measured holograms")
    co.add_argument("--samples", **samples)
    co.add_argument("--cnn", type=Path, help="CNN estimates (default: <measured>/cnn)")
    co.add_argument("--out", type=Path, help="comparison.{csv,json} (default: <measured>)")
    co.set_defaults(func=cmd_compare)

    no = sub.add_parser("noise", help="noise spectra of the background corners")
    no.add_argument("--holograms", type=Path, required=True)
    no.add_argument("--samples", **samples)
    no.add_argument("--out", type=Path, default=RESULTS / "sim2real")
    no.set_defaults(func=cmd_noise)

    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
