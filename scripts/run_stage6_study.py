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

Usage::

    python scripts/run_stage6_study.py regions --holograms DIR
    python scripts/run_stage6_study.py --threads 2 fit --holograms DIR --samples spider_hair
    python scripts/run_stage6_study.py configs

``DIR`` is ``Examples/data/holograms`` of the Zenodo record 10.5281/zenodo.8349365.
``regions`` reads the phase files ``<sample>-phase.npy`` of the Stage 3 run
``results/measured/b1-p4`` (not tracked; ``scripts/run_stage3_experiments.py``). ``fit``
continues a record of the same sample, setup and number of samples and writes it after every
texture.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from holofocus.baseline.runner import file_sha256
from holofocus.dataset import DatasetConfig, HologramGenerator
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
    MEASURED_MODELS,
    RadialPower,
    perturbed_config,
    spectral_features,
)
from holofocus.measured import MEASURED_SAMPLES, load_measured
from run_stage5_study import environment, repo_relative, write_csv, write_json

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

    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
