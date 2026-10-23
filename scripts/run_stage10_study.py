"""Stage 10: beyond the radial power spectrum (Laborbuch 10).

Four families of learned estimates, each with the training seeds 0, 1 and 2 and their
ensemble (:data:`FAMILIES`): the CNN and the NPE on the radial power spectrum of Stages 4
and 7, the CNN on the polar power spectrum (8 sectors of the direction, Laborbuch 10) and the
CNN on the hologram itself, binned 2x. They are trained on cpu-4x (the synthetic baseline
samples) and on the leave-one-out sets of the texture fits with the levels of the spectrum
(``loo2``, :data:`holofocus.learned.objectmodel.TEXTURE_FITS_LEVELS`), and the 95 % interval
of each ensemble bounds Brent's method on the 48 synthetic samples of Stage 3 and on the four
measured holograms.

* ``configs``: ``configs/dataset/loo2-<sample>.yaml`` (6000/250/250) and
  ``loo2-<sample>-large.yaml`` (50 000 training samples, for GPU runs), the ranges and the
  sample's own values (``results/objectmodel/loo2-ranges.json``).
* ``plan``: every job of this stage for ``scripts/run_jobs.py`` (``runs/stage10/plan.json``):
  the leave-one-out sets and their fingerprints, the trainings, predictions and ensembles,
  the estimates on the measured holograms and the bounded searches (``--parts cpu-4x``: only
  the jobs on cpu-4x, which do not need the texture fits).
* ``ensembles``: the seeds of a CNN family on a test split as one equal-weight mixture of
  their Gaussians (calibrated ``sigma``), in the columns of the NPE predictions
  (``results/stage10/cpu-4x/<family>-ensemble/test``,
  ``results/stage10/loo2/<family>/ensemble/loo-<sample>/test``).
* ``measured``: the seeds of a family trained on the leave-one-out set of a measured hologram
  on that hologram, binned 4x, with the 16 phases of the binning grid, and their ensemble
  (``results/measured/<family>-loo2-ensemble/<sample>.json``, the interval of the bounded
  search).
* ``search``: Brent's bounded method in the 95 % interval of a family's ensemble on one of the
  synthetic samples (``runs/stage10/search/<family>/<index>``); ``merge`` joins them
  (``results/stage10/baseline/cpu-4x-test-brent-box95-<family>``).

Usage (Laborbuch 10)::

    python scripts/run_stage10_study.py configs
    python scripts/run_stage10_study.py plan --holograms DIR
    python scripts/run_jobs.py run runs/stage10/plan.json --state runs/stage10
    python scripts/run_stage10_study.py merge

``DIR`` is ``Examples/data/holograms`` of the Zenodo record 10.5281/zenodo.8349365. Run it
from the root of the repository, so the paths in the records stay relative to it.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from holofocus.baseline import BaselineConfig, merge_runs, run_baseline
from holofocus.baseline.runner import Box, file_sha256
from holofocus.dataset import DatasetConfig
from holofocus.learned import read_predictions
from holofocus.learned.objectmodel import (
    LOO_FACTOR,
    TEXTURE_FITS_LEVELS,
    loo_changes,
    loo_ranges,
)
from holofocus.learned.posterior import GaussianMixture, mixture_summaries
from holofocus.learned.sim2real import MEASURED_MODELS, perturbed_config
from holofocus.measured import MEASURED_SAMPLES
from run_stage6_study import LARGE_TRAIN, loo_coverage, matched_config
from run_stage6_study import cmd_measured as cnn_measured
from run_stage7_loo_study import cmd_measured as npe_measured
from run_stage7_loo_study import ensemble_name, run_name
from run_stage7_study import LEVELS, environment, repo_relative, write_json
from run_stage8_study import mixture_record, npe_record, paper_references

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
STAGE10 = RESULTS / "stage10"
MEASURED = RESULTS / "measured"
RUNS = REPO / "runs" / "stage10"
CONFIGS = REPO / "configs" / "dataset"
PROBES = tuple(MEASURED_SAMPLES)
SEEDS = (0, 1, 2)
LOO2_SEEDS = {name: 20261101 + k for k, name in enumerate(PROBES)}
LOO2_LARGE_SEEDS = {name: 20261111 + k for k, name in enumerate(PROBES)}
SPLITS = ("train", "val", "test")
CPU4X_TRAIN = ("data/cpu-4x/train.h5", "data/cpu-4x-extra/train.h5")
CPU4X_VAL = "data/cpu-4x/val.h5"
CPU4X_TEST = "data/cpu-4x/test.h5"
SYNTHETIC = tuple(range(48))
"""The samples of the Stage 3 baseline (``results/baseline/cpu-4x-test``)."""
BRENT = "configs/baseline/paper2-cpu-brent.yaml"
"""Brent's bounded method with the settings of the baseline (Stage 9)."""


@dataclass(frozen=True)
class Family:
    """A family of learned estimates: its config, whether it is an NPE, and the runs on
    cpu-4x of earlier stages that are its seeds (``existing``), with their test predictions
    if ``predicted``."""

    key: str
    label: str
    config: str
    npe: bool = False
    existing: tuple[tuple[int, str], ...] = ()
    predicted: tuple[int, ...] = ()
    limit: int | None = None


FAMILIES = {
    "radial-cnn": Family(
        "radial-cnn",
        "radial spectrum, CNN",
        "configs/learned/spectrum-cnn-wide.yaml",
        existing=((0, "results/learned/spectrum-cnn-wide"),),
        predicted=(0,),
    ),
    "npe": Family(
        "npe",
        "radial spectrum, NPE",
        "configs/learned/npe-nsf.yaml",
        npe=True,
        existing=(
            (0, "results/npe/npe-nsf"),
            (1, "results/npe/npe-nsf-seed1"),
            (2, "results/npe/npe-nsf-seed2"),
        ),
        predicted=(0, 1, 2),
    ),
    "polar-cnn": Family("polar-cnn", "polar spectrum, CNN", "configs/learned/polar-cnn.yaml"),
    "image-cnn": Family(
        "image-cnn",
        "hologram binned 2x, CNN",
        "configs/learned/image-cnn.yaml",
        existing=((0, "results/learned/study/image-n10000"),),
        limit=10000,
    ),
}
"""``limit``: the training samples of cpu-4x of the run of an earlier stage (the image CNN of
Stage 4 took all 10 000 with ``--limit 10000``, which its config keeps)."""


# --- layout --------------------------------------------------------------------------------


def cpu4x_run(family: str, seed: int) -> str:
    return dict(FAMILIES[family].existing).get(seed, f"results/stage10/cpu-4x/{family}/seed-{seed}")


def cpu4x_predictions(family: str, seed: int) -> str:
    if seed in FAMILIES[family].predicted:
        return f"{cpu4x_run(family, seed)}/test"
    return f"results/stage10/cpu-4x/{family}/seed-{seed}/test"


def cpu4x_ensemble(family: str) -> str:
    return f"results/stage10/cpu-4x/{family}-ensemble/test"


def loo2_run(family: str, sample: str, seed: int) -> str:
    """NPE runs in the layout of ``run_stage7_loo_study.py`` (``--npe``), CNN runs in that of
    ``run_stage6_study.py measured --learned``."""
    if FAMILIES[family].npe:
        return f"results/stage10/loo2/npe/{run_name(sample, seed)}"
    return f"results/stage10/loo2/{family}/seed-{seed}/loo-{sample}"


def loo2_ensemble(family: str, sample: str) -> str:
    if FAMILIES[family].npe:
        return f"results/stage10/loo2/npe/{ensemble_name(sample)}/test"
    return f"results/stage10/loo2/{family}/ensemble/loo-{sample}/test"


def measured_ensemble(family: str) -> str:
    return f"results/measured/{family}-loo2-ensemble"


def measured_search(family: str) -> str:
    return f"results/measured/brent-box95-{family}-loo2-ensemble-b1-p4"


def search_dir(family: str, index: int) -> str:
    return f"runs/stage10/search/{family}/{index}"


def merged_search(family: str) -> str:
    return f"results/stage10/baseline/cpu-4x-test-brent-box95-{family}"


def data_dir(sample: str) -> str:
    return f"data/loo2-{sample}"


# --- configs -------------------------------------------------------------------------------


def loo2_config(name: str, large: bool = False) -> DatasetConfig:
    """``configs/dataset/loo2-<name>[-large].yaml``: the matched config of ``name`` with
    :func:`~holofocus.learned.objectmodel.loo_changes` of the texture fits with levels."""
    stem = f"loo2-{name}" + ("-large" if large else "")
    changes = loo_changes(name, TEXTURE_FITS_LEVELS)
    data = perturbed_config(matched_config(name), changes, stem).to_dict()
    data["seed"] = (LOO2_LARGE_SEEDS if large else LOO2_SEEDS)[name]
    if large:
        data["splits"] = {"train": LARGE_TRAIN}
    return DatasetConfig.from_dict(data)


def _g(value: float) -> str:
    return f"{value:.3g}".replace("e+0", "e")


def loo2_header(name: str, large: bool) -> str:
    sample, r = MEASURED_SAMPLES[name], loo_ranges(name, TEXTURE_FITS_LEVELS)
    others = ", ".join(MEASURED_SAMPLES[s].label for s in r.others)
    if large:
        title = f"Large leave-one-out training data for the measured {sample.label} (Stage 10)."
        text = (
            f"loo2-{name}.yaml with {LARGE_TRAIN} training samples and another seed, for GPU "
            "runs (docs/maxwell-gpu.md)."
        )
    else:
        title = f"Leave-one-out training data for the measured {sample.label} (Stage 10)."
        text = (
            f"loo-{name}.yaml of Stage 6 with the texture fits whose objective also holds the "
            "levels of the spectrum (holofocus.learned.objectmodel.TEXTURE_FITS_LEVELS): what "
            f"the measured holograms show comes from the other three ({others}) as ranges, "
            f"widened by a factor of {LOO_FACTOR:g}; the setup and delta/beta = "
            f"{MEASURED_MODELS[name].delta_beta:g} stay. Gaussian detector PSF of "
            f"{_g(r.psf_sigma[0])}-{_g(r.psf_sigma[1])} unbinned pixels, {_g(r.photons[0])}-"
            f"{_g(r.photons[1])} photons per binned pixel (log-uniform), flat-field residual of "
            f"{_g(100 * r.flat_field[0])}-{_g(100 * r.flat_field[1])} % RMS with the corner "
            f"frequency {_g(r.flat_field_cutoff)} per binned pixel; fibers of "
            f"{_g(100 * r.fiber_diameter[0])}-{_g(100 * r.fiber_diameter[1])} % of the field of "
            f"view (log-uniform) with {_g(r.phase_per_pixel[0])}-{_g(r.phase_per_pixel[1])} rad "
            "per binned pixel of projected thickness (log-uniform); a texture with the "
            f"power-law slope {_g(r.texture_slope[0])}-{_g(r.texture_slope[1])} inside the "
            f"fiber and as a film over the field of view, each {_g(r.texture_rms[0])}-"
            f"{_g(r.texture_rms[1])} rad RMS (log-uniform). Own seed."
        )
    lines = [title, ""] + textwrap.wrap(text, 94)
    return "".join(f"# {line}".rstrip() + "\n" for line in lines)


def cmd_configs(args) -> None:
    coverage = {}
    for name in args.samples:
        for large in (False, True):
            config = loo2_config(name, large)
            body = yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=None)
            path = args.out / f"{config.name}.yaml"
            path.write_text(loo2_header(name, large) + body)
            print(f"{repo_relative(path)}: seed {config.seed}, splits {config.splits}")
        coverage[name] = loo_coverage(name, TEXTURE_FITS_LEVELS)
        outside = [k for k, v in coverage[name]["inside"].items() if not v]
        print(f"{name}: own values outside the leave-one-out ranges: {outside or 'none'}")
    fits = {name: dataclasses.asdict(t) for name, t in TEXTURE_FITS_LEVELS.items()}
    write_json(
        args.results / "loo2-ranges.json",
        {"texture_fits": fits, "samples": coverage, **environment()},
    )


# --- plan ----------------------------------------------------------------------------------

GENERATION_SECONDS = {
    "spider_hair": 4.0,
    "tooth": 1.85,
    "cactus_needle": 0.4,
    "magnesium_wire": 0.37,
}
"""Seconds per sample with one thread (Stage 6 and the texture fits)."""
GENERATION_GB = {"spider_hair": 1.8, "tooth": 1.1, "cactus_needle": 0.7, "magnesium_wire": 0.7}
"""Peak memory of a generation (1.65, 1.00 and 0.58 GB for three samples of the spider hair,
the tooth and the cactus needle), with a margin."""
TRAIN_SECONDS = {"radial-cnn": 0.23, "polar-cnn": 0.3, "image-cnn": 0.51, "npe": 0.29}
"""Seconds per training sample with one thread (100 epochs; the image CNN 40)."""
TRAIN_GB = {"radial-cnn": 0.9, "polar-cnn": 1.1, "image-cnn": 2.1, "npe": 0.9}
"""Memory of a training on a leave-one-out set: the inputs of the image CNN (6000 at 256^2 in
float16) peak at 1.8 GB while they are joined; on cpu-4x it takes :data:`IMAGE_CPU4X_GB` (2.8
GB for its 10 000 inputs)."""
IMAGE_CPU4X_GB = 3.1
SEARCH_SECONDS = 480.0
"""A synthetic search in a 95 % interval: about 5.7 reconstructions of 84 s (Stage 9)."""
MEASURED_SEARCH_SECONDS = 4600.0
"""A measured search in a 95 % interval with one thread: about 7 reconstructions of 250 s
with three threads (Stage 9), 2.65 times slower with one."""
MEASURED_SEARCH_GB = 1.3
"""A reconstruction of a measured hologram at binning 1 peaks at 0.9 GB."""
PRIORITY = {
    "gen": 100,
    "predict-cpu4x": 99,
    "predict-loo2": 99,
    "ensemble": 98,
    "measured": 97,
    "fingerprint": 96,
    "merge": 96,
    "train-cpu4x": 92,
    "train-loo2": 90,
    "msearch": 85,
    "search": 10,
}
"""Short jobs that others wait for first, then the trainings on cpu-4x (the synthetic
searches wait for their ensembles) and on the leave-one-out sets; the synthetic searches fill
the free cores."""
LONGEST_FIRST = {"image-cnn": 0.3, "npe": 0.2, "polar-cnn": 0.1, "radial-cnn": 0.0}
PROBE_FIRST = {"spider_hair": 0.03, "tooth": 0.02, "cactus_needle": 0.01, "magnesium_wire": 0.0}
SPLIT_FIRST = {"train": 1.0, "val": 0.5, "test": 0.0}


def job(name: str, command: str, done: list[str], **options) -> dict:
    return {"name": name, "command": command, "done": done, **options}


def train_command(family: str, train: list[str], val: str, out: str, seed: int) -> str:
    f = FAMILIES[family]
    verb = "npe-train" if f.npe else "train"
    limit = f" --limit {f.limit}" if f.limit and train == list(CPU4X_TRAIN) else ""
    return (
        f"holofocus-learn {verb} {f.config} --train {' '.join(train)} --val {val} --out {out} "
        f"--seed {seed}{limit} --threads {{threads}}"
    )


def predict_command(family: str, models: list[str], data: str, out: str) -> str:
    verb = "npe-predict" if FAMILIES[family].npe else "predict"
    return (
        f"holofocus-learn {verb} {' '.join(models)} --data {data} --out {out} --threads {{threads}}"
    )


def generation_jobs() -> list[dict]:
    jobs = []
    for name in PROBES:
        config = loo2_config(name)
        for split in SPLITS:
            n = config.splits[split]
            jobs.append(
                job(
                    f"gen:loo2-{name}/{split}",
                    f"holofocus-dataset generate configs/dataset/loo2-{name}.yaml "
                    f"--out {data_dir(name)} --splits {split}",
                    [f"{data_dir(name)}/{split}.h5"],
                    memory_gb=GENERATION_GB[name],
                    priority=PRIORITY["gen"] + PROBE_FIRST[name] + SPLIT_FIRST[split],
                    seconds=round(n * GENERATION_SECONDS[name] + 60),
                )
            )
            if split != "train":
                jobs.append(
                    job(
                        f"fingerprint:loo2-{name}/{split}",
                        f"python scripts/check_cluster_run.py fingerprint "
                        f"{data_dir(name)}/{split}.h5 "
                        f"--out results/datasets/loo2-{name}-{split}.csv",
                        [f"results/datasets/loo2-{name}-{split}.csv"],
                        needs=[f"gen:loo2-{name}/{split}"],
                        memory_gb=0.5,
                        priority=PRIORITY["fingerprint"],
                        seconds=60,
                    )
                )
    return jobs


def cpu4x_jobs() -> list[dict]:
    jobs = []
    for family, f in FAMILIES.items():
        members = []
        for seed in SEEDS:
            run = cpu4x_run(family, seed)
            needs = []
            if seed not in dict(f.existing):
                memory = IMAGE_CPU4X_GB if family == "image-cnn" else TRAIN_GB[family]
                jobs.append(
                    job(
                        f"train-cpu4x:{family}/seed-{seed}",
                        train_command(family, list(CPU4X_TRAIN), CPU4X_VAL, run, seed),
                        [f"{run}/run.yaml"],
                        clean=[run],
                        max_threads=3,
                        memory_gb=memory,
                        priority=PRIORITY["train-cpu4x"] + LONGEST_FIRST[family],
                        seconds=round(10000 * TRAIN_SECONDS[family] + 100),
                    )
                )
                needs = [f"train-cpu4x:{family}/seed-{seed}"]
            out = cpu4x_predictions(family, seed)
            if seed not in f.predicted:
                jobs.append(
                    job(
                        f"predict-cpu4x:{family}/seed-{seed}",
                        predict_command(family, [run], CPU4X_TEST, out),
                        [f"{out}/run.yaml"],
                        needs=needs,
                        memory_gb=0.8,
                        priority=PRIORITY["predict-cpu4x"],
                        seconds=120,
                    )
                )
                needs = [f"predict-cpu4x:{family}/seed-{seed}"]
            members.append((run, needs))
        out = cpu4x_ensemble(family)
        needs = [n for _, ns in members for n in ns]
        if f.npe:
            command = predict_command(family, [run for run, _ in members], CPU4X_TEST, out)
        else:
            command = (
                f"python scripts/run_stage10_study.py ensembles --families {family} --sets cpu-4x"
            )
        jobs.append(
            job(
                f"ensemble:cpu-4x/{family}",
                command,
                [f"{out}/run.yaml"],
                needs=needs,
                memory_gb=0.8,
                priority=PRIORITY["ensemble"],
                seconds=180 if f.npe else 30,
            )
        )
        for index in SYNTHETIC:
            out = search_dir(family, index)
            jobs.append(
                job(
                    f"search:{family}/{index}",
                    f"python scripts/run_stage10_study.py --threads {{threads}} search "
                    f"--family {family} --index {index}",
                    [f"{out}/results.csv"],
                    needs=[f"ensemble:cpu-4x/{family}"],
                    memory_gb=0.6,
                    priority=PRIORITY["search"],
                    seconds=SEARCH_SECONDS,
                )
            )
        jobs.append(
            job(
                f"merge:{family}",
                f"python scripts/run_stage10_study.py merge --families {family}",
                [f"{merged_search(family)}/summary.json"],
                needs=[f"search:{family}/{index}" for index in SYNTHETIC],
                memory_gb=0.3,
                priority=PRIORITY["merge"],
                seconds=30,
            )
        )
    return jobs


def loo2_jobs(holograms: Path) -> list[dict]:
    jobs = []
    for name in PROBES:
        data = data_dir(name)
        n = loo2_config(name).splits["train"]
        shrink = 0.85 if name == "tooth" else 1.0
        for family, f in FAMILIES.items():
            trained, predicted = [], []
            for seed in SEEDS:
                run = loo2_run(family, name, seed)
                train = f"train-loo2:{family}/seed-{seed}/{name}"
                jobs.append(
                    job(
                        train,
                        train_command(family, [f"{data}/train.h5"], f"{data}/val.h5", run, seed),
                        [f"{run}/run.yaml"],
                        needs=[f"gen:loo2-{name}/train", f"gen:loo2-{name}/val"],
                        clean=[run],
                        max_threads=3,
                        memory_gb=TRAIN_GB[family] * (shrink if family == "image-cnn" else 1),
                        priority=PRIORITY["train-loo2"] + LONGEST_FIRST[family] + PROBE_FIRST[name],
                        seconds=round(n * TRAIN_SECONDS[family] * shrink + 60),
                    )
                )
                trained.append(train)
                predict = f"predict-loo2:{family}/seed-{seed}/{name}"
                jobs.append(
                    job(
                        predict,
                        predict_command(family, [run], f"{data}/test.h5", f"{run}/test"),
                        [f"{run}/test/run.yaml"],
                        needs=[train, f"gen:loo2-{name}/test"],
                        memory_gb=0.8,
                        priority=PRIORITY["predict-loo2"],
                        seconds=90,
                    )
                )
                predicted.append(predict)
            out = loo2_ensemble(family, name)
            runs = [loo2_run(family, name, seed) for seed in SEEDS]
            if f.npe:
                command = predict_command(family, runs, f"{data}/test.h5", out)
                needs = trained + [f"gen:loo2-{name}/test"]
            else:
                command = (
                    f"python scripts/run_stage10_study.py ensembles --families {family} "
                    f"--sets {name}"
                )
                needs = predicted
            jobs.append(
                job(
                    f"ensemble:loo2/{family}/{name}",
                    command,
                    [f"{out}/run.yaml"],
                    needs=needs,
                    memory_gb=0.8,
                    priority=PRIORITY["ensemble"],
                    seconds=180 if f.npe else 30,
                )
            )
            jobs.append(
                job(
                    f"measured:{family}/{name}",
                    f"python scripts/run_stage10_study.py --threads {{threads}} measured "
                    f"--holograms {holograms} --families {family} --samples {name}",
                    [f"{measured_ensemble(family)}/{name}.json"],
                    needs=trained,
                    memory_gb=1.2,
                    priority=PRIORITY["measured"],
                    seconds=300,
                )
            )
            jobs.append(
                job(
                    f"msearch:{family}/{name}",
                    "python scripts/run_stage3_experiments.py --threads {threads} measured "
                    f"--holograms {holograms} --binning 1 --padding 4 --method brent "
                    f"--revisit-tolerance 1e-9 --box {measured_ensemble(family)} "
                    f"--samples {name} --out {measured_search(family)}",
                    [f"{measured_search(family)}/{name}.json"],
                    needs=[f"measured:{family}/{name}"],
                    max_threads=3,
                    memory_gb=MEASURED_SEARCH_GB,
                    priority=PRIORITY["msearch"] + PROBE_FIRST[name],
                    seconds=MEASURED_SEARCH_SECONDS,
                )
            )
    return jobs


def cmd_plan(args) -> None:
    if "loo2" in args.parts and args.holograms is None:
        raise SystemExit("the loo2 part needs --holograms")
    jobs = generation_jobs() if "loo2" in args.parts else []
    if "cpu-4x" in args.parts:
        jobs += cpu4x_jobs()
    if "loo2" in args.parts:
        jobs += loo2_jobs(args.holograms)
    plan = {
        "what": "Stage 10 (Laborbuch 10): scripts/run_stage10_study.py plan",
        "budget": {
            "threads": args.threads_budget,
            "memory_gb": args.memory_gb,
            "reserve_gb": args.reserve_gb,
        },
        "jobs": jobs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(plan, indent=1) + "\n")
    tmp.replace(args.out)
    hours = sum(j["seconds"] * j.get("threads", 1) for j in jobs) / 3600
    groups = {}
    for j in jobs:
        groups[j["name"].split(":", 1)[0]] = groups.get(j["name"].split(":", 1)[0], 0) + 1
    print(f"{repo_relative(args.out)}: {len(jobs)} jobs ({groups}), {hours:.1f} core-hours")


# --- ensembles -----------------------------------------------------------------------------


def write_exact_csv(path: Path, rows: list[dict]) -> None:
    """Rows with full precision: the interval bounds define the bounded searches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mixture_rows(members: list[list[dict]]) -> list[dict]:
    """Per sample the equal-weight mixture of the Gaussians of the CNN seeds (calibrated
    ``sigma``), its median the estimate, in the columns of the NPE predictions that
    ``run_stage7_study.npe_table`` reads; ``seconds`` is the sum over the seeds."""
    first = members[0]
    for rows in members[1:]:
        if [r["index"] for r in rows] != [r["index"] for r in first] or max(
            abs(r["z01_true"] - f["z01_true"]) for r, f in zip(rows, first, strict=True)
        ) > 1e-12:
            raise ValueError("the seeds were not evaluated on the same samples")
    guess = np.array([r["z01_guess"] for r in first])
    z01 = np.array([r["z01_true"] for r in first])
    dz01 = z01 - guess
    mix = GaussianMixture(
        np.array([[r["dz01_est"] for r in rows] for rows in members]).T,
        np.array([[r["sigma"] for r in rows] for rows in members]).T,
    )
    est = mixture_summaries(mix)
    pit, hpd = mix.pit(dz01), mix.hpd_level(dz01)
    log_density = mix.log_density(dz01) - math.log(1e6)
    seconds = np.sum([[r["seconds"] for r in rows] for rows in members], axis=0)
    out = []
    for i, r in enumerate(first):
        z01_est = guess[i] + est["dz01"][i]
        row = {
            "index": r["index"],
            "phantom_type": r["phantom_type"],
            "photons": r["photons"],
            "z01_true": z01[i],
            "z01_guess": guess[i],
            "dz01_true": dz01[i],
            "dz01_est": est["dz01"][i],
            "dz01_mean": est["dz01_mean"][i],
            "sigma": est["sigma"][i],
            "z01_est": z01_est,
            "error_um": (z01_est - z01[i]) * 1e6,
            "abs_error_um": abs(z01_est - z01[i]) * 1e6,
        }
        for key in LEVELS:
            row[f"z01_lower_{key}"] = guess[i] + est[f"lower_{key}"][i]
            row[f"z01_upper_{key}"] = guess[i] + est[f"upper_{key}"][i]
        for key in LEVELS:
            row[f"width_{key}_um"] = (est[f"upper_{key}"][i] - est[f"lower_{key}"][i]) * 1e6
            row[f"hpd_width_{key}_um"] = est[f"hpd_width_{key}"][i] * 1e6
        row.update(
            pit=float(pit[i]),
            hpd_level=float(hpd[i]),
            log_density_um=float(log_density[i]),
            seconds=float(seconds[i]),
        )
        out.append({k: float(v) if isinstance(v, np.floating) else v for k, v in row.items()})
    return out


def ensemble_members(family: str, which: str) -> list[Path]:
    if which == "cpu-4x":
        return [Path(cpu4x_predictions(family, k)) / "predictions.csv" for k in SEEDS]
    return [Path(loo2_run(family, which, k)) / "test" / "predictions.csv" for k in SEEDS]


def cmd_ensembles(args) -> None:
    for family in args.families:
        if FAMILIES[family].npe:
            print(f"{family}: the NPE ensemble comes from holofocus-learn npe-predict")
            continue
        for which in args.sets:
            paths = ensemble_members(family, which)
            missing = [repo_relative(p) for p in paths if not p.exists()]
            if missing:
                print(f"{family} {which}: {missing} missing, skipped")
                continue
            rows = mixture_rows([read_predictions(p) for p in paths])
            out = Path(
                cpu4x_ensemble(family) if which == "cpu-4x" else loo2_ensemble(family, which)
            )
            write_exact_csv(out / "predictions.csv", rows)
            covered = np.mean(
                [r["z01_lower_95"] <= r["z01_true"] <= r["z01_upper_95"] for r in rows]
            )
            info = {
                "what": "equal-weight mixture of the Gaussians (calibrated sigma) of the seeds",
                "members": [
                    {"predictions": repo_relative(p), "sha256": file_sha256(p)} for p in paths
                ],
                "n_samples": len(rows),
                "mae_um": float(np.mean([r["abs_error_um"] for r in rows])),
                "coverage_95": float(covered),
                "predictions_sha256": file_sha256(out / "predictions.csv"),
                **environment(),
            }
            (out / "run.yaml").write_text(yaml.safe_dump(info, sort_keys=False))
            print(
                f"{family} {which}: {len(rows)} samples, MAE {info['mae_um']:.0f} um, "
                f"95 % coverage {covered:.2f}"
            )


# --- measured ------------------------------------------------------------------------------


def measured_references(name: str) -> dict[str, float]:
    """[2], [1], the padding-4 search of Stage 3 and Brent cold (Stage 9)."""
    refs = paper_references(name, MEASURED)
    path = MEASURED / "brent-b1-p4" / f"{name}.json"
    if path.exists():
        refs["brent_b1_p4"] = float(json.loads(path.read_text())["result"]["z01"])
    return refs


def cmd_measured(args) -> None:
    for family in args.families:
        f = FAMILIES[family]
        for name in args.samples:
            if f.npe:
                out = Path("results/stage10/loo2/npe/measured")
                npe_measured(
                    argparse.Namespace(
                        npe=Path("results/stage10/loo2/npe"),
                        seeds=list(SEEDS),
                        samples=[name],
                        holograms=args.holograms,
                        repeats=args.repeats,
                        measured=MEASURED,
                        cnn_measured=MEASURED / "cnn-loo",
                        npe_stage7=RESULTS / "npe" / "measured",
                        out=out,
                        device="cpu",
                    )
                )
                record = npe_record(name, out / f"{name}.json")
            else:
                estimates = []
                for seed in SEEDS:
                    learned = Path(f"results/stage10/loo2/{family}/seed-{seed}")
                    cnn_measured(
                        argparse.Namespace(
                            samples=[name],
                            learned=learned,
                            holograms=args.holograms,
                            measured=MEASURED,
                            repeats=args.repeats,
                            device="cpu",
                            out=learned / "measured",
                        )
                    )
                    estimates.append(learned / "measured" / f"{name}.json")
                record = mixture_record(name, estimates, measured_references(name))
            record["family"] = {"key": family, "label": f.label, "config": f.config}
            out = Path(measured_ensemble(family)) / f"{name}.json"
            write_json(out, {**record, **environment()})
            i95 = record["intervals"]["95"]
            print(
                f"{name} {family} ensemble: {record['z01_est'] * 1e3:.3f} mm, 95 % "
                f"{i95['lower'] * 1e3:.3f}-{i95['upper'] * 1e3:.3f} mm"
            )


# --- searches ------------------------------------------------------------------------------


def cmd_search(args) -> None:
    box = Path(cpu4x_ensemble(args.family)) / "predictions.csv"
    run_baseline(
        BaselineConfig.from_yaml(BRENT),
        Path(CPU4X_TEST),
        Path(search_dir(args.family, args.index)),
        indices=[args.index],
        box=Box.from_csv(box),
        log=lambda msg: print(f"  {msg}", flush=True),
    )


def cmd_merge(args) -> None:
    for family in args.families:
        runs = [Path(search_dir(family, i)) for i in SYNTHETIC]
        done = [run for run in runs if (run / "results.csv").exists()]
        if len(done) < len(runs):
            print(f"{family}: {len(done)} of {len(runs)} searches done, not merged")
            continue
        merge_runs(done, Path(merged_search(family)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, help="torch CPU threads")
    sub = parser.add_subparsers(dest="command", required=True)
    samples = {"nargs": "+", "choices": list(PROBES), "default": list(PROBES)}
    families = {"nargs": "+", "choices": list(FAMILIES), "default": list(FAMILIES)}

    cf = sub.add_parser("configs", help="the leave-one-out configs of the texture fits with levels")
    cf.add_argument("--samples", **samples)
    cf.add_argument("--out", type=Path, default=CONFIGS)
    cf.add_argument("--results", type=Path, default=RESULTS / "objectmodel")
    cf.set_defaults(func=cmd_configs)

    pl = sub.add_parser("plan", help="the jobs of this stage for scripts/run_jobs.py")
    pl.add_argument("--holograms", type=Path, help="the measured holograms (loo2 part)")
    pl.add_argument(
        "--parts",
        nargs="+",
        choices=["cpu-4x", "loo2"],
        default=["cpu-4x", "loo2"],
        help="cpu-4x: trainings and searches on the synthetic samples; loo2: the leave-one-out "
        "sets and the measured holograms",
    )
    pl.add_argument("--threads-budget", type=int, default=4)
    pl.add_argument("--memory-gb", type=float, default=8.0, help="memory of all running jobs")
    pl.add_argument(
        "--reserve-gb", type=float, default=1.0, help="free memory the jobs leave the machine"
    )
    pl.add_argument("--out", type=Path, default=RUNS / "plan.json")
    pl.set_defaults(func=cmd_plan)

    en = sub.add_parser("ensembles", help="the seeds of a CNN family as one mixture")
    en.add_argument("--families", **families)
    en.add_argument("--sets", nargs="+", choices=["cpu-4x", *PROBES], default=["cpu-4x", *PROBES])
    en.set_defaults(func=cmd_ensembles)

    me = sub.add_parser("measured", help="the seeds and the ensemble on the measured holograms")
    me.add_argument("--holograms", type=Path, required=True)
    me.add_argument("--families", **families)
    me.add_argument("--samples", **samples)
    me.add_argument("--repeats", type=int, default=20, help="timed estimates")
    me.set_defaults(func=cmd_measured)

    se = sub.add_parser("search", help="Brent in the 95 % interval of an ensemble, one sample")
    se.add_argument("--family", choices=list(FAMILIES), required=True)
    se.add_argument("--index", type=int, choices=SYNTHETIC, required=True)
    se.set_defaults(func=cmd_search)

    mg = sub.add_parser("merge", help="join the searches of each family")
    mg.add_argument("--families", **families)
    mg.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
