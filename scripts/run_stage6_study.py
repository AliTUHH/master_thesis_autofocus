"""Stage 6: a textured object model for the measured holograms (Laborbuch 06).

* ``regions``: where the CTF modulation of each measured hologram comes from. CTF correlation
  at ``Fr`` of [2], and at 1.3 ``Fr`` as a control, of the whole hologram, of the object, of
  its edge and of the background around it; the regions come from the phase reconstruction of
  Stage 3 (:func:`holofocus.learned.objectmodel.region_weights`,
  ``results/objectmodel/regions.json``).
* ``fit``: the texture of each measured hologram. Test samples of the matched dataset with the
  detector and the bulk object of the measured hologram
  (:func:`holofocus.learned.objectmodel.fit_changes`) and every texture of the coarse grid,
  then of a finer grid around the best one; the texture whose median spectral features lie
  closest to those of the measured hologram (:func:`~holofocus.learned.objectmodel.fit_objective`)
  is the fit (``results/objectmodel/fit-<sample>.{json,csv}``).
* ``configs``: the leave-one-out dataset configs ``configs/dataset/loo-<sample>.yaml``
  (:func:`holofocus.learned.objectmodel.loo_changes`) and ``loo-<sample>-large.yaml`` with
  more training samples for GPU runs.
* ``spectra``: the spectral check without training. Percentiles of the features of each
  measured hologram in the Stage 5 test split, in the fit sets without and with the fitted
  texture and in the leave-one-out test split (``results/objectmodel/spectra-<sample>.json``).
* ``measured``: the leave-one-out CNNs ``results/learned/loo-<sample>`` on the measured
  holograms, against [2] and the padding-4 search (``results/measured/cnn-loo/<sample>.json``).
* ``compare``: leave-one-out CNN, Stage 5 CNN, [2] and padding 4 side by side, with the test
  errors of the runs that made the estimates (``model`` of each estimate record) on both test
  splits (``results/objectmodel/comparison.{csv,json}``).

Usage::

    python scripts/run_stage6_study.py regions --holograms DIR
    python scripts/run_stage6_study.py --threads 2 fit --holograms DIR --samples spider_hair
    python scripts/run_stage6_study.py configs
    python scripts/run_stage6_study.py spectra --holograms DIR
    python scripts/run_stage6_study.py --threads 1 measured --holograms DIR
    python scripts/run_stage6_study.py compare

``DIR`` is ``Examples/data/holograms`` of the Zenodo record 10.5281/zenodo.8349365.
``regions`` reads the phase files ``<sample>-phase.npy`` of the Stage 3 run
``results/measured/b1-p4`` (not tracked; ``scripts/run_stage3_experiments.py``). ``fit``
continues a record of the same sample, setup and number of samples and writes it after every
texture. Datasets, training and predictions between ``configs`` and ``measured`` are in
``docs/usage.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import textwrap
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from holofocus.baseline.runner import file_sha256
from holofocus.dataset import DatasetConfig, HologramGenerator, read_labels
from holofocus.learned import LearnedAutofocus, read_predictions
from holofocus.learned.objectmodel import (
    BINNING,
    DETECTOR_FITS,
    FIT_SCALES,
    LOO_FACTOR,
    REGION_FRACTIONS,
    TEXTURE_FITS,
    TEXTURE_FLOOR,
    Texture,
    fit_changes,
    fit_objective,
    loo_changes,
    loo_ranges,
    median_features,
    phase_per_pixel,
    refined_grid,
    region_ctf,
    region_weights,
    texture_grid,
)
from holofocus.learned.sim2real import (
    FEATURE_NAMES,
    MEASURED_MODELS,
    RadialPower,
    binning_phases,
    feature_distribution,
    measured_estimate,
    percentile_rank,
    perturbed_config,
    spectral_features,
)
from holofocus.measured import MEASURED_SAMPLES, load_measured
from run_stage5_study import (
    baseline_estimates,
    environment,
    repo_relative,
    validation_mae,
    write_csv,
    write_json,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
CONFIGS = REPO / "configs" / "dataset"
SPLIT = "test"
LOO_SEEDS = {name: 20261061 + k for k, name in enumerate(MEASURED_SAMPLES)}
LARGE_SEEDS = {name: 20261071 + k for k, name in enumerate(MEASURED_SAMPLES)}
LARGE_TRAIN = 50000


def matched_config(name: str) -> DatasetConfig:
    return DatasetConfig.from_yaml(CONFIGS / f"matched-{name}.yaml")


def loo_config(name: str, large: bool = False) -> DatasetConfig:
    """``configs/dataset/loo-<name>[-large].yaml``: the matched config of ``name`` with
    :func:`~holofocus.learned.objectmodel.loo_changes` and its own seed."""
    stem = f"loo-{name}" + ("-large" if large else "")
    data = perturbed_config(matched_config(name), loo_changes(name), stem).to_dict()
    data["seed"] = (LARGE_SEEDS if large else LOO_SEEDS)[name]
    if large:
        data["splits"] = {"train": LARGE_TRAIN}
    return DatasetConfig.from_dict(data)


def _g(value: float) -> str:
    return f"{value:.3g}".replace("e+0", "e")


def loo_header(name: str, large: bool) -> str:
    sample, r = MEASURED_SAMPLES[name], loo_ranges(name)
    others = ", ".join(MEASURED_SAMPLES[s].label for s in r.others)
    if large:
        text = (
            f"loo-{name}.yaml with {LARGE_TRAIN} training samples and another seed, for GPU runs "
            "(docs/maxwell-gpu.md)."
        )
        title = f"Large leave-one-out training data for the measured {sample.label} (Stage 6)."
    else:
        text = (
            f"matched-{name}.yaml changed by holofocus.learned.objectmodel.loo_changes: what "
            "the measured holograms show about detector and object comes from the other three "
            f"({others}) as ranges, widened by a factor of {LOO_FACTOR:g}; the setup and "
            f"delta/beta = {MEASURED_MODELS[name].delta_beta:g} (composition of the sample) stay. "
            f"Gaussian detector PSF of {_g(r.psf_sigma[0])}-{_g(r.psf_sigma[1])} unbinned "
            f"pixels, {_g(r.photons[0])}-{_g(r.photons[1])} photons per binned pixel "
            f"(log-uniform), flat-field residual of {_g(100 * r.flat_field[0])}-"
            f"{_g(100 * r.flat_field[1])} % RMS with the corner frequency "
            f"{_g(r.flat_field_cutoff)} per binned pixel; fibers of {_g(100 * r.fiber_diameter[0])}"
            f"-{_g(100 * r.fiber_diameter[1])} % of the field of view (log-uniform) with "
            f"{_g(r.phase_per_pixel[0])}-{_g(r.phase_per_pixel[1])} rad per binned pixel of "
            "projected thickness (log-uniform); a texture with the power-law slope "
            f"{_g(r.texture_slope[0])}-{_g(r.texture_slope[1])} inside the fiber and as a film "
            f"over the field of view, each {_g(r.texture_rms[0])}-{_g(r.texture_rms[1])} rad RMS "
            "(log-uniform). Own seed."
        )
        title = f"Leave-one-out training data for the measured {sample.label} (Stage 6)."
    lines = [title, ""] + textwrap.wrap(text, 94)
    return "".join(f"# {line}".rstrip() + "\n" for line in lines)


def own_values(name: str) -> dict:
    """What the measured hologram of ``name`` itself shows, in the keys of
    :class:`~holofocus.learned.objectmodel.LooRanges`; ``None`` where it shows nothing (the
    detector of the Mg wire, the slope of an untextured fit)."""
    model, detector, texture = MEASURED_MODELS[name], DETECTOR_FITS[name], TEXTURE_FITS[name]
    return {
        "psf_sigma": None if detector is None else detector.psf_sigma,
        "photons": model.photons,
        "flat_field": None if detector is None else model.flat_field,
        "flat_field_cutoff": None if detector is None else detector.flat_field_cutoff,
        "phase_per_pixel": phase_per_pixel(name),
        "fiber_diameter": list(model.fiber_diameter_range),
        "texture_rms": texture.rms,
        "background_texture_rms": texture.background_rms,
        "texture_slope": texture.slope if texture.amplitude > TEXTURE_FLOOR else None,
    }


def loo_coverage(name: str) -> dict:
    """The leave-one-out ranges of ``name`` and whether its own values lie inside them."""
    ranges = dataclasses.asdict(loo_ranges(name))
    own = own_values(name)
    inside = {}
    for key, value in own.items():
        texture = key in ("texture_rms", "background_texture_rms")
        bounds = ranges["texture_rms" if texture else key]
        if value is None or not isinstance(bounds, (list, tuple)):
            continue
        if texture and value <= TEXTURE_FLOOR:
            inside[key] = bounds[0] <= TEXTURE_FLOOR
            continue
        values = value if isinstance(value, list) else [value]
        inside[key] = all(bounds[0] <= v <= bounds[1] for v in values)
    return {"ranges": ranges, "own": own, "inside": inside}


def cmd_configs(args) -> None:
    coverage = {}
    for name in args.samples:
        for large in (False, True):
            config = loo_config(name, large)
            body = yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=None)
            path = args.out / f"{config.name}.yaml"
            path.write_text(loo_header(name, large) + body)
            print(f"{repo_relative(path)}: seed {config.seed}, splits {config.splits}")
        coverage[name] = loo_coverage(name)
        outside = [k for k, v in coverage[name]["inside"].items() if not v]
        print(f"{name}: own values outside the leave-one-out ranges: {outside or 'none'}")
    fits = {name: dataclasses.asdict(t) for name, t in TEXTURE_FITS.items()}
    write_json(
        args.results / "loo-ranges.json",
        {"texture_fits": fits, "samples": coverage, **environment()},
    )


def measured_features(holograms: Path, name: str) -> tuple[RadialPower, dict]:
    """Radial power of the measured hologram binned 4x and its features at ``Fr`` of [2]."""
    sample = MEASURED_SAMPLES[name]
    hologram = load_measured(holograms, sample, BINNING).numpy()
    radial = RadialPower(tuple(hologram.shape))
    fresnel_number = sample.geometry(BINNING, sample.z01_nm).fresnel_number
    return radial, spectral_features(radial, hologram, fresnel_number)


REGION_CONTROL = 1.3
"""``regions`` repeats every correlation at this multiple of ``Fr``, off the CTF of the
hologram."""


def _correlations(hologram: np.ndarray, weight: np.ndarray, fresnel_number: float) -> dict:
    ctf = region_ctf(hologram, weight, fresnel_number)
    control = region_ctf(hologram, weight, REGION_CONTROL * fresnel_number)
    return {
        "area": round(float(weight.mean()), 4),
        "ctf": round(ctf["ctf"], 4),
        "ctf_bands": [round(v, 4) for v in ctf["ctf_bands"]],
        "control_ctf": round(control["ctf"], 4),
        "control_ctf_bands": [round(v, 4) for v in control["ctf_bands"]],
    }


def sample_regions(name: str, holograms: Path, phases: Path) -> dict:
    """CTF correlations of the whole measured hologram and of its regions at both thresholds."""
    sample = MEASURED_SAMPLES[name]
    hologram = load_measured(holograms, sample, BINNING).numpy()
    phase_path = phases / f"{name}-phase.npy"
    phase = np.load(phase_path)
    if phase.shape != hologram.shape:
        raise SystemExit(f"{phase_path}: shape {phase.shape}, hologram {hologram.shape}")
    fresnel_number = sample.geometry(BINNING, sample.z01_nm).fresnel_number
    return {
        "fresnel_number_paper2_nm": fresnel_number,
        "phase_sha256": file_sha256(phase_path),
        "full": _correlations(hologram, np.ones_like(hologram), fresnel_number),
        "regions": {
            f"{fraction:g}": {
                region: _correlations(hologram, weight, fresnel_number)
                for region, weight in region_weights(phase, fraction).items()
            }
            for fraction in REGION_FRACTIONS
        },
    }


def cmd_regions(args) -> None:
    record = {
        "phases": repo_relative(args.phases),
        "fractions": list(REGION_FRACTIONS),
        "control_factor": REGION_CONTROL,
        "samples": {},
    }
    for name in args.samples:
        entry = sample_regions(name, args.holograms, args.phases)
        record["samples"][name] = entry
        rows = [("full", entry["full"])] + [
            (f"{region} ({fraction})", values)
            for fraction, regions in entry["regions"].items()
            for region, values in regions.items()
        ]
        for label, values in rows:
            print(
                f"{name} {label}: area {values['area']:.2f}, ctf {values['ctf']:+.2f} bands "
                f"{values['ctf_bands']}, control {values['control_ctf']:+.2f}"
            )
    write_json(args.out / "regions.json", {**record, **environment()})


def simulated_features(config: DatasetConfig, n: int, radial: RadialPower) -> list[dict]:
    """Spectral features of the first ``n`` test samples of ``config``."""
    generator = HologramGenerator(config, device="cpu")
    out = []
    for i in range(n):
        sample = generator.sample(SPLIT, i)
        features = spectral_features(radial, sample.hologram, sample.labels["fresnel_number"])
        out.append({**features, "phase_shift": sample.meta["phase_shift"]})
    return out


def _key(texture: Texture) -> tuple[float, float, float]:
    return (texture.rms, texture.background_rms, texture.slope)


def _fit_row(entry: dict) -> dict:
    m = entry["median"]
    return {
        **entry["texture"],
        "objective": entry["objective"],
        "ctf": m["ctf"],
        **{f"ctf_band{k + 1}": v for k, v in enumerate(m["ctf_bands"])},
        "upper_snr": m["upper_snr"],
        "spectral_slope": m["slope"],
        "floor": m["floor"],
        "low_excess": m["low_excess"],
        "stage": entry["stage"],
        "seconds": entry["seconds"],
    }


def fit_sample(name: str, holograms: Path, n: int, out: Path) -> None:
    target = out / f"fit-{name}.json"
    radial, measured = measured_features(holograms, name)
    base = matched_config(name)
    sample = MEASURED_SAMPLES[name]
    record = {
        "sample": name,
        "base_config": repo_relative(CONFIGS / f"matched-{name}.yaml"),
        "split": SPLIT,
        "n": n,
        "fresnel_number_paper2_nm": sample.geometry(BINNING, sample.z01_nm).fresnel_number,
        "measured": measured,
        "scales": FIT_SCALES,
        "settings": [],
    }
    if target.exists():
        old = json.loads(target.read_text())
        same = all(old.get(k) == record[k] for k in ("sample", "base_config", "split", "n"))
        if same and np.allclose(old["measured"]["ctf_bands"], measured["ctf_bands"]):
            record["settings"] = old["settings"]
    done = {_key(Texture(**e["texture"])) for e in record["settings"]}

    def run(texture: Texture, stage: str) -> None:
        if _key(texture) in done:
            return
        config = perturbed_config(base, fit_changes(name, texture), f"fit-{name}")
        start = time.perf_counter()
        features = simulated_features(config, n, radial)
        median = median_features(features)
        entry = {
            "texture": dataclasses.asdict(texture),
            "stage": stage,
            "median": median,
            "objective": fit_objective(median, measured),
            "phase_shift_median": float(np.median([f["phase_shift"] for f in features])),
            "seconds": time.perf_counter() - start,
        }
        if texture.amplitude == 0:
            entry["features"] = features
        record["settings"].append(entry)
        done.add(_key(texture))
        write_json(target, {**record, **environment()})
        print(
            f"{name} {stage} rms {texture.rms:<6g} film {texture.background_rms:<6g} slope "
            f"{texture.slope:<4g}: objective {entry['objective']:.2f}, ctf {median['ctf']:+.2f}"
            f" bands {[round(b, 2) for b in median['ctf_bands']]}, upper_snr "
            f"{median['upper_snr']:+.2f}, slope {median['slope']:+.1f} "
            f"({entry['seconds']:.0f} s)",
            flush=True,
        )

    for texture in texture_grid():
        run(texture, "coarse")
    coarse = [e for e in record["settings"] if e["stage"] == "coarse"]
    best_coarse = Texture(**min(coarse, key=lambda e: e["objective"])["texture"])
    for texture in refined_grid(best_coarse):
        run(texture, "refined")
    best = min(record["settings"], key=lambda e: e["objective"])
    config = perturbed_config(base, fit_changes(name, Texture(**best["texture"])), f"fit-{name}")
    best["features"] = simulated_features(config, n, radial)
    record["best"] = {"texture": best["texture"], "objective": best["objective"]}
    record["best_coarse"] = dataclasses.asdict(best_coarse)
    write_json(target, {**record, **environment()})
    write_csv(out / f"fit-{name}.csv", [_fit_row(e) for e in record["settings"]])
    m = measured
    print(
        f"{name}: best {best['texture']} (objective {best['objective']:.2f}); measured ctf "
        f"{m['ctf']:+.2f} bands {[round(b, 2) for b in m['ctf_bands']]}, upper_snr "
        f"{m['upper_snr']:+.2f}, slope {m['slope']:+.1f}"
    )


def cmd_fit(args) -> None:
    for name in args.samples:
        fit_sample(name, args.holograms, args.n, args.out)


PROFILE_F2_MAX = 0.125
"""The example spectra of ``spectra`` reach to the upper end of the band of the spectrum CNN."""


def distribution(features: list[dict], measured: dict) -> dict:
    """5/50/95 % of every feature over a set and the percentile of the measured hologram."""
    bands = np.array([f["ctf_bands"] for f in features], dtype=float)
    measured_bands = measured["ctf_bands"]
    return {
        "n": len(features),
        "features": feature_distribution(features),
        "measured_percentile": {
            k: percentile_rank([f[k] for f in features], measured[k]) for k in FEATURE_NAMES
        },
        "ctf_bands": {
            "median": np.nanmedian(bands, axis=0).tolist(),
            "p05": np.nanpercentile(bands, 5, axis=0).tolist(),
            "p95": np.nanpercentile(bands, 95, axis=0).tolist(),
        },
        "ctf_bands_percentile": [
            percentile_rank(bands[:, k], v) for k, v in enumerate(measured_bands)
        ],
    }


def _profile(radial: RadialPower, hologram: np.ndarray) -> list[float]:
    keep = radial.f2 < PROFILE_F2_MAX
    return [round(float(v), 4) for v in radial(hologram)[keep]]


def typical_index(features: list[dict]) -> int:
    """The sample whose CTF correlation lies closest to the median of its set."""
    ctf = np.array([f["ctf"] for f in features], dtype=float)
    return int(np.nanargmin(np.abs(ctf - np.nanmedian(ctf))))


def stored_features(path: Path, radial: RadialPower, limit: int | None) -> tuple[list, dict]:
    """Features of the holograms of a split file and the spectrum of its typical sample."""
    labels = read_labels(path)
    n = len(labels["dz01"]) if limit is None else min(limit, len(labels["dz01"]))
    features, profiles = [], []
    with h5py.File(path, "r") as fh:
        for i in range(n):
            hologram = np.asarray(fh["hologram"][i])
            fr = float(labels["fresnel_number"][i])
            features.append(spectral_features(radial, hologram, fr))
            profiles.append(_profile(radial, hologram))
    i = typical_index(features)
    example = {
        "index": i,
        "fresnel_number": float(labels["fresnel_number"][i]),
        "log_power": profiles[i],
    }
    return features, example


def generated_example(config: DatasetConfig, radial: RadialPower, index: int) -> dict:
    sample = HologramGenerator(config, device="cpu").sample(SPLIT, index)
    return {
        "index": index,
        "fresnel_number": sample.labels["fresnel_number"],
        "log_power": _profile(radial, sample.hologram),
    }


def cmd_spectra(args) -> None:
    for name in args.samples:
        sample = MEASURED_SAMPLES[name]
        hologram = load_measured(args.holograms, sample, BINNING).numpy()
        radial, measured = measured_features(args.holograms, name)
        fit = json.loads((args.out / f"fit-{name}.json").read_text())
        untextured = next(e for e in fit["settings"] if Texture(**e["texture"]).amplitude == 0)
        best = next(e for e in fit["settings"] if e["texture"] == fit["best"]["texture"])
        record = {
            "sample": name,
            "fresnel_number_paper2_nm": fit["fresnel_number_paper2_nm"],
            "f2": [float(v) for v in radial.f2[radial.f2 < PROFILE_F2_MAX]],
            "measured": measured,
            "measured_log_power": _profile(radial, hologram),
            "fit_texture": fit["best"]["texture"],
            "loo_ranges": dataclasses.asdict(loo_ranges(name)),
            "sets": {},
        }
        base = matched_config(name)
        sets = {
            "matched": ("Stage 5 test split: compact objects, no texture", None),
            "untextured": ("own detector and bulk object, no texture (fit)", untextured),
            "fit": ("own detector and bulk object, fitted texture (fit)", best),
            "loo": ("leave-one-out test split: ranges from the other three", None),
        }
        paths = {
            "matched": args.data_root / f"matched-{name}" / "test.h5",
            "loo": args.data_root / f"loo-{name}" / "test.h5",
        }
        for set_name, (description, entry) in sets.items():
            if entry is not None:
                features = entry["features"]
                texture = Texture(**entry["texture"])
                config = perturbed_config(base, fit_changes(name, texture), f"fit-{name}")
                example = generated_example(config, radial, typical_index(features))
                source = repo_relative(args.out / f"fit-{name}.json")
            elif paths[set_name].exists():
                features, example = stored_features(paths[set_name], radial, args.limit)
                source = repo_relative(paths[set_name])
            else:
                print(f"{name} {set_name}: {paths[set_name]} missing, skipped")
                continue
            record["sets"][set_name] = {
                "description": description,
                "source": source,
                **distribution(features, measured),
                "example": example,
            }
            ranks = record["sets"][set_name]["measured_percentile"]
            bands = record["sets"][set_name]["ctf_bands"]["median"]
            print(
                f"{name} {set_name}: "
                + ", ".join(f"{f} {ranks[f]:.0f} %" for f in FEATURE_NAMES)
                + f"; median bands {[round(b, 2) for b in bands]}"
            )
        print(f"{name} measured: bands {[round(b, 2) for b in measured['ctf_bands']]}")
        record.update(environment())
        write_json(args.out / f"spectra-{name}.json", record)


def cmd_measured(args) -> None:
    for name in args.samples:
        sample = MEASURED_SAMPLES[name]
        run = args.learned / f"loo-{name}"
        start = time.perf_counter()
        hologram = load_measured(args.holograms, sample, BINNING)
        load_seconds = time.perf_counter() - start
        phases = binning_phases(load_measured(args.holograms, sample, 1).numpy(), BINNING)
        references = {
            "paper2_nm": sample.z01_nm,
            "paper2_visual": sample.z01_visual,
            "paper1_example": sample.z01_example,
            **baseline_estimates(args.measured, name),
        }
        autofocus = LearnedAutofocus.load(run, args.device)
        estimate = measured_estimate(autofocus, hologram, sample, references, args.repeats, phases)
        record = {
            "sample": name,
            "label": sample.label,
            "binning": BINNING,
            "input_shape": list(hologram.shape),
            "model": repo_relative(run),
            "model_sha256": file_sha256(run / "model.pt"),
            "config_name": autofocus.config.name,
            "validation_mae_um": validation_mae(run),
            "load_seconds": load_seconds,
            **estimate,
            **environment(),
        }
        write_json(args.out / f"{name}.json", record)
        ref = estimate["references"]
        print(
            f"{name}: z01 = {estimate['z01_est'] * 1e3:.3f} mm, sigma "
            f"{estimate['sigma'] * 1e6:.0f} um; minus [2] {ref['paper2_nm']['distance_um']:+.0f} um"
            + (
                f", minus b1-p4 {ref['baseline_b1_p4']['distance_um']:+.0f} um"
                if "baseline_b1_p4" in ref
                else ""
            )
            + f"; {1e3 * estimate['seconds_per_estimate']['median']:.1f} ms"
        )


def _summary(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _estimate_columns(prefix: str, estimate: dict) -> dict:
    """Estimate, interval and distances to [2] and to the padding-4 search of one model."""
    row = {
        f"{prefix}_mm": estimate["z01_est"] * 1e3,
        f"{prefix}_sigma_um": estimate["sigma"] * 1e6,
        f"{prefix}_width68_um": estimate["intervals"]["68"]["width_um"],
    }
    for key, ref in (("paper2", "paper2_nm"), ("b1_p4", "baseline_b1_p4")):
        r = estimate["references"].get(ref)
        if r is None:
            continue
        row[f"{prefix}_minus_{key}_um"] = r["distance_um"]
        row[f"{prefix}_z_{key}"] = r["z_score"]
        row[f"{prefix}_{key}_in68"] = r["inside"]["68"]
        row[f"{prefix}_{key}_in95"] = r["inside"]["95"]
    return row


def _test_columns(prefix: str, summary: dict | None) -> dict:
    if summary is None:
        return {}
    err, interval = summary["error_um"], summary["interval"]
    return {
        f"{prefix}_mae_um": err["mae"],
        f"{prefix}_bias_um": err["bias"],
        f"{prefix}_within_100": err["within"]["100"],
        f"{prefix}_coverage_68": interval["coverage"]["68"],
        f"{prefix}_coverage_95": interval["coverage"]["95"],
        f"{prefix}_sigma_median_um": interval["sigma_median"],
    }


def cmd_compare(args) -> None:
    rows, details = [], {}
    for name in args.samples:
        sample = MEASURED_SAMPLES[name]
        loo_path = args.cnn / f"{name}.json"
        if not loo_path.exists():
            print(f"{name}: no estimate of the leave-one-out CNN, skipped")
            continue
        loo = json.loads(loo_path.read_text())
        stage5_path = args.measured / "cnn" / f"{name}.json"
        stage5 = json.loads(stage5_path.read_text())
        stage5_best = stage5["variants"][stage5["selected"]]
        b1_p4 = json.loads((args.measured / "b1-p4" / f"{name}.json").read_text())["result"]
        # the runs that made the estimates, so the test errors belong to the same models
        loo_run, stage5_run = REPO / loo["model"], REPO / stage5_best["model"]
        tests = {
            "loo_test": loo_run / "test",
            "loo_on_matched_test": loo_run / "matched-test",
            "stage5_on_loo_test": args.matched_model / name,
            "stage5_test": stage5_run / "test",
        }
        tests = {k: p for k, p in tests.items() if (p / "predictions.csv").exists()}
        test_rows = read_predictions(tests["loo_test"] / "predictions.csv")
        abs_test = np.array([abs(r["error_um"]) for r in test_rows])
        row = {
            "sample": name,
            "paper2_nm_mm": sample.z01_nm * 1e3,
            "b1_p4_mm": b1_p4["z01"] * 1e3,
            "b1_p4_minus_paper2_um": (b1_p4["z01"] - sample.z01_nm) * 1e6,
            "stage5_run": stage5["selected"],
            **_estimate_columns("stage5", stage5_best),
            **_estimate_columns("loo", loo),
            "loo_ms": 1e3 * loo["seconds_per_estimate"]["median"],
            "loo_threads": loo.get("threads"),
            "loo_phase_std_um": loo["binning_phases"]["std_um"],
        }
        for key, path in tests.items():
            row.update(_test_columns(key, _summary(path / "summary.json")))
        row["disagreement_paper2_percentile"] = percentile_rank(
            abs_test, abs(loo["references"]["paper2_nm"]["distance_um"])
        )
        if "baseline_b1_p4" in loo["references"]:
            row["disagreement_b1_p4_percentile"] = percentile_rank(
                abs_test, abs(loo["references"]["baseline_b1_p4"]["distance_um"])
            )
        rows.append(row)
        details[name] = {
            "row": row,
            "loo": repo_relative(loo_path),
            "stage5": repo_relative(stage5_path),
            "loo_run": repo_relative(loo_run),
            "stage5_run": repo_relative(stage5_run),
            "tests": {k: repo_relative(p) for k, p in tests.items()},
        }
        print(
            f"{name}: LOO CNN {row['loo_mm']:.3f} mm +- {row['loo_sigma_um']:.0f} um "
            f"({row['loo_minus_paper2_um']:+.0f} um from [2], in 68 %: {row['loo_paper2_in68']}; "
            f"{row.get('loo_minus_b1_p4_um', float('nan')):+.0f} um from b1-p4), Stage 5 "
            f"{row['stage5_minus_paper2_um']:+.0f} um from [2]; LOO test MAE "
            f"{row.get('loo_test_mae_um', float('nan')):.0f} um"
        )
    if not rows:
        raise SystemExit("no sample with an estimate of the leave-one-out CNN")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(args.out / "comparison.csv", [{k: r.get(k, "") for k in fields} for r in rows])
    write_json(args.out / "comparison.json", {"samples": details, **environment()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, help="torch CPU threads")
    sub = parser.add_subparsers(dest="command", required=True)
    samples = {"nargs": "+", "choices": list(MEASURED_SAMPLES), "default": list(MEASURED_SAMPLES)}

    rg = sub.add_parser("regions", help="CTF modulation of object, edge and background")
    rg.add_argument("--holograms", type=Path, required=True)
    rg.add_argument("--samples", **samples)
    rg.add_argument("--phases", type=Path, default=RESULTS / "measured" / "b1-p4")
    rg.add_argument("--out", type=Path, default=RESULTS / "objectmodel")
    rg.set_defaults(func=cmd_regions)

    fi = sub.add_parser("fit", help="the texture of each measured hologram")
    fi.add_argument("--holograms", type=Path, required=True)
    fi.add_argument("--samples", **samples)
    fi.add_argument("--n", type=int, default=16, help="test samples per texture")
    fi.add_argument("--out", type=Path, default=RESULTS / "objectmodel")
    fi.set_defaults(func=cmd_fit)

    cf = sub.add_parser("configs", help="the leave-one-out dataset configs")
    cf.add_argument("--samples", **samples)
    cf.add_argument("--out", type=Path, default=CONFIGS)
    cf.add_argument("--results", type=Path, default=RESULTS / "objectmodel")
    cf.set_defaults(func=cmd_configs)

    sp = sub.add_parser("spectra", help="spectral features, measured against simulated")
    sp.add_argument("--holograms", type=Path, required=True)
    sp.add_argument("--samples", **samples)
    sp.add_argument("--limit", type=int, help="only the first N samples of the stored splits")
    sp.add_argument("--data-root", type=Path, default=REPO / "data")
    sp.add_argument("--out", type=Path, default=RESULTS / "objectmodel")
    sp.set_defaults(func=cmd_spectra)

    me = sub.add_parser("measured", help="the leave-one-out CNNs on the measured holograms")
    me.add_argument("--holograms", type=Path, required=True)
    me.add_argument("--samples", **samples)
    me.add_argument("--repeats", type=int, default=20, help="timed estimates")
    me.add_argument("--device", default="cpu", help="cpu, cuda or auto")
    me.add_argument("--learned", type=Path, default=RESULTS / "learned")
    me.add_argument("--measured", type=Path, default=RESULTS / "measured")
    me.add_argument("--out", type=Path, default=RESULTS / "measured" / "cnn-loo")
    me.set_defaults(func=cmd_measured)

    co = sub.add_parser("compare", help="leave-one-out CNN, Stage 5 CNN, [2] and padding 4")
    co.add_argument("--samples", **samples)
    co.add_argument("--cnn", type=Path, default=RESULTS / "measured" / "cnn-loo")
    co.add_argument("--measured", type=Path, default=RESULTS / "measured")
    co.add_argument(
        "--matched-model",
        type=Path,
        default=RESULTS / "objectmodel" / "matched-model",
        help="predictions of the Stage 5 CNNs on the leave-one-out test splits",
    )
    co.add_argument("--out", type=Path, default=RESULTS / "objectmodel")
    co.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
