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

Usage::

    python scripts/run_stage6_study.py regions --holograms DIR
    python scripts/run_stage6_study.py --threads 2 fit --holograms DIR --samples spider_hair

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
import time
from pathlib import Path

import numpy as np
import torch

from holofocus.baseline.runner import file_sha256
from holofocus.dataset import DatasetConfig, HologramGenerator
from holofocus.learned.objectmodel import (
    BINNING,
    FIT_SCALES,
    REGION_FRACTIONS,
    Texture,
    fit_changes,
    fit_objective,
    median_features,
    refined_grid,
    region_ctf,
    region_weights,
    texture_grid,
)
from holofocus.learned.sim2real import RadialPower, perturbed_config, spectral_features
from holofocus.measured import MEASURED_SAMPLES, load_measured
from run_stage5_study import environment, repo_relative, write_csv, write_json

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
CONFIGS = REPO / "configs" / "dataset"
SPLIT = "test"


def matched_config(name: str) -> DatasetConfig:
    return DatasetConfig.from_yaml(CONFIGS / f"matched-{name}.yaml")


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

    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
