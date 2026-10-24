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
  the estimates on the measured holograms, the bounded searches, ``texture``, ``spectra``
  and, once all others are done, ``compare`` (``--parts cpu-4x``: only the jobs on cpu-4x,
  which do not need the texture fits, and ``compare``).
* ``ensembles``: the seeds of a CNN family on a test split as one equal-weight mixture of
  their Gaussians (calibrated ``sigma``), in the columns of the NPE predictions
  (``results/stage10/cpu-4x/<family>-ensemble/test``,
  ``results/stage10/loo2/<family>/ensemble/loo-<sample>/test``); ``mixture``: the same for
  predictions given by path, e.g. the seeds trained on Maxwell.
* ``measured``: the seeds of a family trained on the leave-one-out set of a measured hologram
  on that hologram, binned 4x, with the 16 phases of the binning grid, and their ensemble
  (``results/measured/<family>-loo2-ensemble/<sample>.json``, the interval of the bounded
  search); ``--runs`` and ``--out`` take the runs of Maxwell instead.
* ``search``: Brent's bounded method in the 95 % interval of a family's ensemble on one of the
  synthetic samples (``runs/stage10/search/<family>/<index>``); ``merge`` joins them
  (``results/stage10/baseline/cpu-4x-test-brent-box95-<family>``).
* ``texture``: the fit set of each measured hologram without texture and with the textures
  of Stages 6 and 10 against its spectrum (``results/stage10/texture-<sample>.json``);
  ``spectra``: the measured hologram among the leave-one-out test splits of Stages 6 and 10
  (``results/stage10/spectra-<sample>.json``).
* ``compare``: the learned estimates of the four families and the searches side by side
  (``results/stage10/compare.json`` and the CSVs next to it); ``evaluate``: the same metrics
  for predictions, searches and estimates on the measured holograms given by path, e.g. the
  runs of Maxwell (``docs/maxwell-gpu.md`` 8.5).

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
import itertools
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy.stats import binomtest

from holofocus.baseline import BaselineConfig, merge_runs, read_rows, run_baseline
from holofocus.baseline.runner import Box, file_sha256
from holofocus.dataset import DatasetConfig, HologramGenerator, read_labels
from holofocus.learned import read_predictions
from holofocus.learned.objectmodel import (
    BINNING,
    LEVEL_NAMES,
    LOO_FACTOR,
    TEXTURE_FITS,
    TEXTURE_FITS_LEVELS,
    Texture,
    fit_changes,
    fit_objective,
    loo_changes,
    loo_ranges,
    median_features,
)
from holofocus.learned.posterior import GaussianMixture, mixture_summaries
from holofocus.learned.sim2real import (
    FEATURE_NAMES,
    MEASURED_MODELS,
    RadialPower,
    percentile_rank,
    perturbed_config,
    spectral_features,
)
from holofocus.measured import MEASURED_SAMPLES, load_measured
from run_stage6_study import LARGE_TRAIN, loo_coverage, matched_config
from run_stage6_study import cmd_measured as cnn_measured
from run_stage7_loo_study import cmd_measured as npe_measured
from run_stage7_loo_study import ensemble_name, guess_table, run_name
from run_stage7_study import (
    LEVELS,
    cnn_table,
    environment,
    npe_table,
    paired,
    repo_relative,
    restrict,
    write_csv,
    write_json,
)
from run_stage8_study import mixture_record, npe_record, paper_references, shifted
from run_stage9_study import CNN_INTERVALS, NPE_PREDICTIONS, ci, search_pair, search_summary

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


def measured_ensemble(family: str, root: str | Path = "results/measured") -> str:
    return f"{root}/{family}-loo2-ensemble"


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
TRAIN_SECONDS = {"radial-cnn": 0.23, "polar-cnn": 0.47, "image-cnn": 0.58, "npe": 0.29}
"""Seconds per training sample with one thread (100 epochs; the image CNN 40): the polar and
the image CNN as their first trainings on cpu-4x took with four jobs on the four cores (46 and
142 s per epoch of 10 000 samples)."""
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
    "spectra": 96,
    "compare": 96,
    "texture": 95,
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


SPECTRA_SECONDS = 0.1
"""Seconds of ``spectra`` per hologram of a test split, with a margin (the test splits of
Stages 6 and 10 are equally large)."""


def texture_jobs(holograms: Path) -> list[dict]:
    jobs = []
    for name in PROBES:
        textures = dict.fromkeys((UNTEXTURED, TEXTURE_FITS[name], TEXTURE_FITS_LEVELS[name]))
        jobs.append(
            job(
                f"texture:{name}",
                f"python scripts/run_stage10_study.py --threads {{threads}} texture "
                f"--holograms {holograms} --samples {name}",
                [f"results/stage10/texture-{name}.json"],
                memory_gb=round(GENERATION_GB[name] + 0.3, 2),
                priority=PRIORITY["texture"] + PROBE_FIRST[name],
                seconds=round(len(textures) * FIT_SAMPLES * GENERATION_SECONDS[name] + 60),
            )
        )
        n = len(SPECTRA_SETS) * loo2_config(name).splits["test"]
        jobs.append(
            job(
                f"spectra:{name}",
                f"python scripts/run_stage10_study.py spectra --holograms {holograms} "
                f"--samples {name}",
                [f"results/stage10/spectra-{name}.json"],
                needs=[f"gen:loo2-{name}/test"],
                memory_gb=0.8,
                priority=PRIORITY["spectra"],
                seconds=round(n * SPECTRA_SECONDS + 60),
            )
        )
    return jobs


def compare_job(jobs: list[dict]) -> dict:
    """``compare`` once every job of the plan that it reads is done."""
    return job(
        "compare",
        "python scripts/run_stage10_study.py compare",
        ["results/stage10/compare.json"],
        needs=[j["name"] for j in jobs if j["name"].split(":", 1)[0] not in ("texture", "spectra")],
        memory_gb=1.0,
        priority=PRIORITY["compare"],
        seconds=300,
    )


def cmd_plan(args) -> None:
    if "loo2" in args.parts and args.holograms is None:
        raise SystemExit("the loo2 part needs --holograms")
    jobs = generation_jobs() if "loo2" in args.parts else []
    if "cpu-4x" in args.parts:
        jobs += cpu4x_jobs()
    if "loo2" in args.parts:
        jobs += loo2_jobs(args.holograms) + texture_jobs(args.holograms)
    jobs.append(compare_job(jobs))
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


def write_mixture(paths: list[Path], out: Path) -> dict:
    """:func:`mixture_rows` of the predictions ``paths`` as ``<out>/predictions.csv``, with
    ``<out>/run.yaml`` naming the members."""
    rows = mixture_rows([read_predictions(p) for p in paths])
    write_exact_csv(out / "predictions.csv", rows)
    covered = np.mean([r["z01_lower_95"] <= r["z01_true"] <= r["z01_upper_95"] for r in rows])
    info = {
        "what": "equal-weight mixture of the Gaussians (calibrated sigma) of the seeds",
        "members": [{"predictions": repo_relative(p), "sha256": file_sha256(p)} for p in paths],
        "n_samples": len(rows),
        "mae_um": float(np.mean([r["abs_error_um"] for r in rows])),
        "coverage_95": float(covered),
        "predictions_sha256": file_sha256(out / "predictions.csv"),
        **environment(),
    }
    (out / "run.yaml").write_text(yaml.safe_dump(info, sort_keys=False))
    return info


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
            out = Path(
                cpu4x_ensemble(family) if which == "cpu-4x" else loo2_ensemble(family, which)
            )
            info = write_mixture(paths, out)
            print(
                f"{family} {which}: {info['n_samples']} samples, MAE {info['mae_um']:.0f} um, "
                f"95 % coverage {info['coverage_95']:.2f}"
            )


def cmd_mixture(args) -> None:
    info = write_mixture(args.predictions, args.out)
    print(
        f"{repo_relative(args.out)}: {info['n_samples']} samples, MAE {info['mae_um']:.0f} um, "
        f"95 % coverage {info['coverage_95']:.2f}"
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
                out = args.runs / "npe" / "measured"
                npe_measured(
                    argparse.Namespace(
                        npe=args.runs / "npe",
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
                    learned = args.runs / family / f"seed-{seed}"
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
            out = Path(measured_ensemble(family, args.out)) / f"{name}.json"
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


# --- the texture at high frequencies -------------------------------------------------------

FIT_SAMPLES = 16
"""Test samples of a fit set, as in the texture fits."""
POWER_FROM_F2 = 0.02
"""Lower end (``f^2``) of the power that ``texture`` sums: the lower end of the mid band."""
PROFILE_BINS = 256
"""``texture`` and ``spectra`` keep the radial log power averaged over this many bins of
``f^2`` (8 of the 2048 of :class:`~holofocus.learned.sim2real.RadialPower` each)."""
UNTEXTURED = Texture(0.0, 0.0, 2.0)
"""The untextured setting of the texture fits (the slope does not matter)."""
QUANTILES = (5, 50, 95)


def binned(log_power: np.ndarray) -> np.ndarray:
    """The last axis of a radial log power averaged over :data:`PROFILE_BINS` bins."""
    v = np.asarray(log_power, dtype=float)
    return v.reshape(*v.shape[:-1], PROFILE_BINS, -1).mean(axis=-1)


def rounded(values) -> list[float]:
    return [round(float(v), 4) for v in values]


def measured_spectrum(holograms: Path, name: str) -> tuple[RadialPower, np.ndarray, dict]:
    """Radial power, radial log power and spectral features (at ``Fr`` of [2]) of the
    measured hologram binned 4x."""
    sample = MEASURED_SAMPLES[name]
    hologram = load_measured(holograms, sample, BINNING).numpy()
    radial = RadialPower(tuple(hologram.shape))
    fresnel_number = sample.geometry(BINNING, sample.z01_nm).fresnel_number
    return radial, radial(hologram), spectral_features(radial, hologram, fresnel_number)


def fit_set(name: str, texture: Texture, radial: RadialPower, n: int):
    """Spectral features and radial log power of the first ``n`` test samples of the fit set
    of ``name`` with ``texture``, the samples of ``run_stage6_study.py fit``."""
    config = perturbed_config(matched_config(name), fit_changes(name, texture), f"fit-{name}")
    generator = HologramGenerator(config, device="cpu")
    features, profiles = [], []
    for i in range(n):
        sample = generator.sample("test", i)
        features.append(spectral_features(radial, sample.hologram, sample.labels["fresnel_number"]))
        profiles.append(radial(sample.hologram))
    return features, np.array(profiles)


def summed_power(log_power: np.ndarray, f2: np.ndarray) -> float:
    return float(np.exp(log_power[f2 >= POWER_FROM_F2]).sum())


def recorded_objective(path: Path, texture: Texture) -> float | None:
    """The objective of ``texture`` in a texture-fit record, if it was tried there."""
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    key = dataclasses.asdict(texture)
    return next((e["objective"] for e in record["settings"] if e["texture"] == key), None)


def cmd_texture(args) -> None:
    """Part 1 of Laborbuch 10: the fit sets of each measured hologram without texture, with
    the texture of Stage 6 and with that of the objective with levels, against the measured
    spectrum (``results/stage10/texture-<sample>.json``)."""
    for name in args.samples:
        radial, measured_power, measured = measured_spectrum(args.holograms, name)
        settings = {
            "untextured": UNTEXTURED,
            "stage6": TEXTURE_FITS[name],
            "stage10": TEXTURE_FITS_LEVELS[name],
        }
        simulated = {t: fit_set(name, t, radial, args.n) for t in dict.fromkeys(settings.values())}
        f2 = radial.f2
        median_power = {t: np.median(p, axis=0) for t, (_, p) in simulated.items()}
        base = summed_power(median_power[UNTEXTURED], f2)
        excess = summed_power(measured_power, f2) - base
        record = {
            "sample": name,
            "n": args.n,
            "what": "median radial log power and spectral features of the first test samples of "
            "the fit sets (detector and bulk object of the measured hologram) with each texture",
            "power_from_f2": POWER_FROM_F2,
            "measured": measured,
            "f2": rounded(binned(f2)),
            "measured_log_power": rounded(binned(measured_power)),
            "measured_excess_over_untextured": excess / base,
            "settings": {},
        }
        for key, texture in settings.items():
            features, _ = simulated[texture]
            median = median_features(features)
            power = summed_power(median_power[texture], f2)
            entry = {
                "texture": dataclasses.asdict(texture),
                "median": median,
                "objective": fit_objective(median, measured),
                "objective_levels": fit_objective(median, measured, levels=True),
                "recorded_objective": recorded_objective(
                    RESULTS / "objectmodel" / f"fit-{name}.json", texture
                ),
                "recorded_objective_levels": recorded_objective(
                    RESULTS / "objectmodel" / "levels" / f"fit-{name}.json", texture
                ),
                "level_differences": {k: median[k] - measured[k] for k in LEVEL_NAMES},
                "power_to_measured": power / summed_power(measured_power, f2),
                "texture_power_to_measured_excess": (power - base) / excess
                if texture.amplitude > 0 and excess > 0
                else None,
                "log_power": rounded(binned(median_power[texture])),
            }
            record["settings"][key] = entry
            d = entry["level_differences"]
            ratio = entry["texture_power_to_measured_excess"]
            print(
                f"{name} {key} {texture}: objective {entry['objective']:.2f}, with levels "
                f"{entry['objective_levels']:.2f}; levels "
                + "/".join(f"{d[k]:+.2f}" for k in LEVEL_NAMES)
                + f" nat; power x{entry['power_to_measured']:.2f} of the measured"
                + ("" if ratio is None else f", texture x{ratio:.1f} of the measured excess")
            )
        write_json(STAGE10 / f"texture-{name}.json", {**record, **environment()})


LEVEL_FEATURES = ("level_mid", "level_upper")
"""The levels of :func:`~holofocus.learned.sim2real.spectral_features` that are not in
:data:`~holofocus.learned.sim2real.FEATURE_NAMES`."""


def split_spectra(path: Path, radial: RadialPower, limit: int | None):
    """Spectral features and radial log power of the holograms of a split file."""
    labels = read_labels(path)
    n = len(labels["dz01"]) if limit is None else min(limit, len(labels["dz01"]))
    features, profiles = [], []
    with h5py.File(path, "r") as fh:
        for i in range(n):
            hologram = np.asarray(fh["hologram"][i])
            features.append(spectral_features(radial, hologram, float(labels["fresnel_number"][i])))
            profiles.append(radial(hologram))
    return features, np.array(profiles)


def split_record(
    features: list[dict], profiles: np.ndarray, measured: dict, measured_power: np.ndarray
) -> dict:
    """5/50/95 % of every feature and of the radial log power over a split, and where the
    measured hologram lies among the split: percentiles of its features and, bin by bin, of
    its radial log power."""
    bands = np.array([f["ctf_bands"] for f in features], dtype=float)
    out = {"n": len(features), "features": {}, "measured_percentile": {}}
    for name in (*FEATURE_NAMES, *LEVEL_FEATURES):
        values = np.array([f[name] for f in features], dtype=float)
        values = values[np.isfinite(values)]
        out["features"][name] = {f"p{q:02d}": float(np.percentile(values, q)) for q in QUANTILES}
        out["measured_percentile"][name] = percentile_rank(values, measured[name])
    out["ctf_bands"] = {f"p{q:02d}": np.nanpercentile(bands, q, axis=0).tolist() for q in QUANTILES}
    out["ctf_bands_percentile"] = [
        percentile_rank(bands[:, k], v) for k, v in enumerate(measured["ctf_bands"])
    ]
    per_bin = binned(profiles)
    out["log_power"] = {f"p{q:02d}": rounded(np.percentile(per_bin, q, axis=0)) for q in QUANTILES}
    out["measured_log_power_percentile"] = [
        round(percentile_rank(column, value), 1)
        for column, value in zip(per_bin.T, binned(measured_power), strict=True)
    ]
    return out


SPECTRA_SETS = {
    "loo": ("data/loo-{}/test.h5", "leave-one-out test split of Stage 6 (TEXTURE_FITS)"),
    "loo2": ("data/loo2-{}/test.h5", "leave-one-out test split of Stage 10 (TEXTURE_FITS_LEVELS)"),
}


def cmd_spectra(args) -> None:
    """The measured hologram among the leave-one-out test splits of Stages 6 and 10, without
    training (``results/stage10/spectra-<sample>.json``)."""
    for name in args.samples:
        radial, measured_power, measured = measured_spectrum(args.holograms, name)
        record = {
            "sample": name,
            "measured": measured,
            "f2": rounded(binned(radial.f2)),
            "measured_log_power": rounded(binned(measured_power)),
            "sets": {},
        }
        for key, (pattern, what) in SPECTRA_SETS.items():
            path = Path(pattern.format(name))
            if not path.exists():
                print(f"{name} {key}: {path} missing, skipped")
                continue
            features, profiles = split_spectra(path, radial, args.limit)
            entry = split_record(features, profiles, measured, measured_power)
            record["sets"][key] = {"what": what, "source": str(path), **entry}
            ranks = entry["measured_percentile"]
            print(
                f"{name} {key}: measured at "
                + ", ".join(f"{k} {v:.0f} %" for k, v in ranks.items())
                + "; CTF bands "
                + ", ".join(f"{v:.0f} %" for v in entry["ctf_bands_percentile"])
            )
        write_json(STAGE10 / f"spectra-{name}.json", {**record, **environment()})


# --- compare -------------------------------------------------------------------------------

WITHIN_UM = 100.0
XATOL_UM = 100.0
"""Tolerance of the searches in ``z01``: results closer than this are the same."""
MFE_APART_PPM = 100.0
"""How far the MFE at the result of a search must lie from that of Brent cold to count as
another minimum (Laborbuch 09)."""
LEARNED_PAIRS = (
    ("polar-cnn", "radial-cnn"),
    ("image-cnn", "radial-cnn"),
    ("polar-cnn", "npe"),
    ("image-cnn", "npe"),
    ("image-cnn", "polar-cnn"),
    ("npe", "radial-cnn"),
)
"""Ensembles that ``compare`` sets against each other sample by sample: each new input against
the CNN and the NPE on the radial spectrum, the two new inputs, and the NPE against the CNN."""
STAGE8_ENSEMBLES = {
    "radial-cnn/stage8": ("cnn-loo-ensemble", "radial spectrum, CNN ensemble of Stage 8 (loo)"),
    "npe/stage8": ("npe-loo-ensemble", "radial spectrum, NPE ensemble of Stage 8 (loo)"),
}
"""The two ensembles of Stage 8 on the measured holograms, trained on the leave-one-out sets of
Stage 6 (``results/measured/<folder>``)."""


def method_key(family: str, seed: int | None) -> str:
    return f"{family}/ensemble" if seed is None else f"{family}/seed-{seed}"


def share(hits) -> dict:
    """Count, share and 95 % Wilson interval."""
    hits = np.asarray(hits, dtype=bool)
    k, n = int(hits.sum()), int(hits.size)
    interval = binomtest(k, n).proportion_ci(method="wilson")
    return {"k": k, "n": n, "share": k / n, "ci": [float(interval.low), float(interval.high)]}


def mcnemar(first, second) -> dict:
    """Paired hits of two methods and the exact McNemar test."""
    a, b = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return {
        "both": int(np.sum(a & b)),
        "only_first": only[0],
        "only_second": only[1],
        "neither": int(np.sum(~a & ~b)),
        "p": float(binomtest(min(only), sum(only)).pvalue) if sum(only) else 1.0,
    }


def learned_metrics(table: dict[str, np.ndarray]) -> dict:
    """Share within 100 um, MAE and the coverage of the 95 % interval, with 95 % intervals
    (Wilson for shares, bootstrap for the MAE); the 68 % coverage, widths and NLL."""
    err = np.abs(table["error_um"])
    out = {
        "n": int(err.size),
        "within_100um": share(err <= WITHIN_UM),
        "mae_um": float(err.mean()),
        "mae_um_ci": ci(err),
        "median_abs_um": float(np.median(err)),
        "bias_um": float(table["error_um"].mean()),
    }
    if "covered_95" in table:
        out.update(
            coverage_95=share(table["covered_95"]),
            coverage_68=share(table["covered_68"]),
            width_95_median_um=float(np.median(table["width_95_um"])),
            width_68_median_um=float(np.median(table["width_68_um"])),
            nll=float(-table["log_density_um"].mean()),
        )
    return out


def prediction_paths(family: str, which: str) -> dict[str, Path]:
    """The predictions of the seeds and the ensemble of ``family`` on cpu-4x or on the
    leave-one-out test split of a probe."""
    paths = {}
    for seed in SEEDS:
        folder = (
            cpu4x_predictions(family, seed)
            if which == "cpu-4x"
            else (f"{loo2_run(family, which, seed)}/test")
        )
        paths[method_key(family, seed)] = Path(folder) / "predictions.csv"
    folder = cpu4x_ensemble(family) if which == "cpu-4x" else loo2_ensemble(family, which)
    paths[method_key(family, None)] = Path(folder) / "predictions.csv"
    return paths


def learned_tables(which: str) -> tuple[dict[str, dict], dict[str, str]]:
    """Every seed and ensemble with predictions on the split, and the guess."""
    tables, sources, rows = {}, {}, None
    for family, f in FAMILIES.items():
        for method, path in prediction_paths(family, which).items():
            if not path.exists():
                continue
            rows = read_predictions(path)
            single_cnn = not f.npe and not method.endswith("/ensemble")
            tables[method] = cnn_table(rows) if single_cnn else npe_table(rows)
            sources[method] = repo_relative(path)
    if rows is not None:
        tables["guess"] = guess_table(rows)
    first = next(iter(tables.values()), None)
    for method, t in tables.items():
        if not np.array_equal(t["index"], first["index"]) or (
            np.abs(t["z01_true"] - first["z01_true"]).max() > 1e-12
        ):
            raise ValueError(f"{which}: {method} was not evaluated on the same samples")
    return tables, sources


def paired_entry(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict:
    """``a`` minus ``b`` on the same samples (Stage 7) with the exact McNemar tests of the hits
    within 100 um and of the 95 % intervals."""
    entry = paired(a, b)
    entry["within_100um_mcnemar"] = mcnemar(
        np.abs(a["error_um"]) <= WITHIN_UM, np.abs(b["error_um"]) <= WITHIN_UM
    )
    entry["coverage_95_mcnemar"] = mcnemar(a["covered_95"], b["covered_95"])
    return entry


def learned_set(tables: dict[str, dict]) -> dict:
    record = {
        "methods": {m: learned_metrics(t) for m, t in tables.items()},
        "seeds": {},
        "paired": {},
    }
    for family in FAMILIES:
        seeds = [record["methods"].get(method_key(family, k)) for k in SEEDS]
        seeds = [s for s in seeds if s is not None]
        if len(seeds) > 1:
            record["seeds"][family] = {
                "n_seeds": len(seeds),
                "within_100um": [min(s["within_100um"]["share"] for s in seeds),
                                 max(s["within_100um"]["share"] for s in seeds)],
                "mae_um": [min(s["mae_um"] for s in seeds), max(s["mae_um"] for s in seeds)],
                "coverage_95": [min(s["coverage_95"]["share"] for s in seeds),
                                max(s["coverage_95"]["share"] for s in seeds)],
            }  # fmt: skip
    comparisons = [(method_key(a, None), method_key(b, None)) for a, b in LEARNED_PAIRS]
    comparisons += [(method_key(f, None), method_key(f, 0)) for f in FAMILIES]
    for first, second in comparisons:
        if first in tables and second in tables:
            record["paired"][f"{first} vs {second}"] = paired_entry(tables[first], tables[second])
    return record


def learned_row(set_name: str, method: str, m: dict) -> dict:
    row = {
        "set": set_name,
        "method": method,
        "n": m["n"],
        "within_100um": m["within_100um"]["share"],
        "within_100um_low": m["within_100um"]["ci"][0],
        "within_100um_high": m["within_100um"]["ci"][1],
        "mae_um": m["mae_um"],
        "mae_um_low": m["mae_um_ci"][0],
        "mae_um_high": m["mae_um_ci"][1],
    }
    if "coverage_95" in m:
        row.update(
            coverage_95=m["coverage_95"]["share"],
            coverage_95_low=m["coverage_95"]["ci"][0],
            coverage_95_high=m["coverage_95"]["ci"][1],
            truth_outside_95=m["n"] - m["coverage_95"]["k"],
            coverage_68=m["coverage_68"]["share"],
            width_95_median_um=m["width_95_median_um"],
            nll=m["nll"],
        )
    return row


def learned_comparison() -> tuple[dict, list[dict]]:
    """The learned estimates of every family on the test split of cpu-4x, on its 48 samples of
    the Stage 3 baseline, on each leave-one-out test split and on all four together."""
    out, rows, pooled = {}, [], {}
    tables, sources = learned_tables("cpu-4x")
    sets = []
    if tables:
        sets.append(("cpu-4x-test", tables, sources))
        baseline = {m: restrict(t, SYNTHETIC) for m, t in tables.items()}
        sets.append(("cpu-4x-baseline", baseline, sources))
    for number, name in enumerate(PROBES):
        tables, sources = learned_tables(name)
        if not tables:
            continue
        sets.append((f"loo2-{name}", tables, sources))
        for method, t in tables.items():
            pooled.setdefault(method, []).append(shifted(t, 100000 * number))
    complete = {m: parts for m, parts in pooled.items() if len(parts) == len(PROBES)}
    if complete:
        tables = {
            m: {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            for m, parts in complete.items()
        }
        sets.append(("loo2-pooled", tables, None))
    for set_name, tables, sources in sets:
        record = learned_set(tables)
        if sources is not None:
            record["predictions"] = sources
        out[set_name] = record
        rows += [learned_row(set_name, m, v) for m, v in record["methods"].items()]
    return out, rows


def nm_cold_z01(name: str) -> float | None:
    path = MEASURED / "b1-p4" / f"{name}.json"
    return float(json.loads(path.read_text())["result"]["z01"]) if path.exists() else None


def measured_records(name: str) -> dict[str, tuple[Path, dict]]:
    """Estimate and 95 % interval of every seed and ensemble on the measured hologram of
    ``name``, and of the two ensembles of Stage 8."""
    found = {}
    for family, f in FAMILIES.items():
        if f.npe:
            path = Path("results/stage10/loo2/npe/measured") / f"{name}.json"
            if path.exists():
                runs = json.loads(path.read_text())["runs"]
                for seed in SEEDS:
                    run = runs.get("npe" if seed == 0 else f"npe-seed{seed}")
                    if run is not None:
                        found[method_key(family, seed)] = (path, run)
        else:
            for seed in SEEDS:
                path = Path(f"results/stage10/loo2/{family}/seed-{seed}/measured/{name}.json")
                if path.exists():
                    found[method_key(family, seed)] = (path, json.loads(path.read_text()))
        path = Path(measured_ensemble(family)) / f"{name}.json"
        if path.exists():
            found[method_key(family, None)] = (path, json.loads(path.read_text()))
    for method, (folder, _) in STAGE8_ENSEMBLES.items():
        path = MEASURED / folder / f"{name}.json"
        if path.exists():
            found[method] = (path, json.loads(path.read_text()))
    return found


def estimate_entry(path: Path, record: dict, refs: dict[str, float]) -> dict:
    i95 = record["intervals"]["95"]
    z01, lower, upper = record["z01_est"], i95["lower"], i95["upper"]
    entry = {
        "record": repo_relative(path),
        "z01_est": z01,
        "lower_95": lower,
        "upper_95": upper,
        "width_95_um": (upper - lower) * 1e6,
    }
    for key, value in refs.items():
        entry[f"minus_{key}_um"] = (z01 - value) * 1e6
        entry[f"{key}_inside_95"] = bool(lower <= value <= upper)
    return entry


def measured_summary(entries: list[dict]) -> dict:
    """Over the probes: share within 100 um of each reference, the mean distance (MAE) and
    how often the 95 % interval holds the reference."""
    out = {"n": len(entries)}
    for key in ("paper2_nm", "nm_cold"):
        if not all(f"minus_{key}_um" in e for e in entries):
            continue
        d = np.abs([e[f"minus_{key}_um"] for e in entries])
        out[key] = {
            "within_100um": share(d <= WITHIN_UM),
            "mae_um": float(d.mean()),
            "inside_95": share([e[f"{key}_inside_95"] for e in entries]),
        }
    out["width_95_median_um"] = float(np.median([e["width_95_um"] for e in entries]))
    return out


def measured_learned() -> tuple[dict, list[dict]]:
    samples, rows, by_method = {}, [], {}
    for name in PROBES:
        refs = {"paper2_nm": MEASURED_SAMPLES[name].z01_nm}
        cold = nm_cold_z01(name)
        if cold is not None:
            refs["nm_cold"] = cold
        entries = {m: estimate_entry(p, r, refs) for m, (p, r) in measured_records(name).items()}
        if not entries:
            continue
        samples[name] = {"references": refs, "methods": entries}
        for method, e in entries.items():
            by_method.setdefault(method, []).append(e)
            rows.append(
                {
                    "sample": name,
                    "method": method,
                    **{k: v for k, v in e.items() if k != "record"},
                }
            )
    summaries = {m: measured_summary(es) for m, es in by_method.items()}
    return {"samples": samples, "summaries": summaries}, rows


SYNTHETIC_SEARCHES = {
    "nm-cold": ("results/baseline/cpu-4x-test", None, "Nelder-Mead in Omega_z (Stage 3)"),
    "brent-cold": ("results/baseline/cpu-4x-test-brent", None, "Brent in Omega_z (Stage 9)"),
    "brent-box95-cnn-stage9": (
        "results/baseline/cpu-4x-test-brent-box95-cnn",
        repo_relative(CNN_INTERVALS),
        "Brent in the 95 % interval of the CNN of Stage 4, one seed (Stage 9)",
    ),
    "brent-box95-npe-stage9": (
        "results/baseline/cpu-4x-test-brent-box95-npe-ensemble",
        repo_relative(NPE_PREDICTIONS),
        "Brent in the 95 % interval of the ensemble of the five NPE seeds of Stage 7 (Stage 9)",
    ),
    **{
        f"brent-box95-{family}": (
            merged_search(family),
            f"{cpu4x_ensemble(family)}/predictions.csv",
            f"Brent in the 95 % interval of the ensemble of three seeds: {f.label}",
        )
        for family, f in FAMILIES.items()
    },
}
"""The searches on the 48 synthetic samples: folder, the CSV of the 95 % intervals (``None``:
``Omega_z``) and what it is."""
SEARCH_PAIRS = (
    *((f"brent-box95-{f}", "brent-cold") for f in FAMILIES),
    *((f"brent-box95-{f}", "nm-cold") for f in FAMILIES),
    ("brent-box95-polar-cnn", "brent-box95-radial-cnn"),
    ("brent-box95-image-cnn", "brent-box95-radial-cnn"),
    ("brent-box95-polar-cnn", "brent-box95-npe"),
    ("brent-box95-image-cnn", "brent-box95-npe"),
    ("brent-box95-radial-cnn", "brent-box95-cnn-stage9"),
    ("brent-box95-npe", "brent-box95-npe-stage9"),
    ("brent-cold", "nm-cold"),
)
"""Each search in an interval against Brent cold and against Nelder-Mead cold, the new inputs
against the radial spectrum, and the ensembles of this stage against the intervals of
Stage 9 (one CNN, five NPE seeds)."""


def search_table(
    runs: dict[str, dict[int, dict]], boxes: dict[str, Box], pairs, indices
) -> tuple[dict, dict, list[dict]]:
    """Summaries of the searches ``runs[method][index]`` (rows of ``results.csv``), for those
    in ``boxes`` with the truth inside the interval cut to ``Omega_z``, and the MFE at each
    result against that of the search named ``brent-cold``; the ``pairs`` of searches both
    present; one row per sample of ``indices``."""
    confidence = BaselineConfig.from_yaml(BRENT).search.confidence
    brent = runs.get("brent-cold", {})
    summaries, inside, bounds = {}, {}, {}
    for method, by_index in runs.items():
        s = search_summary(list(by_index.values()), None)
        if method in boxes:
            bounds[method] = {
                i: boxes[method].bounds(i, r["z01_guess"] - confidence, r["z01_guess"] + confidence)
                for i, r in by_index.items()
            }
            inside[method] = {
                i: lo <= by_index[i]["z01_true"] <= hi for i, (lo, hi) in bounds[method].items()
            }
            s["truth_outside"] = sorted(i for i, v in inside[method].items() if not v)
            s["mae_um_truth_inside"] = float(
                np.mean([by_index[i]["abs_error_um"] for i, v in inside[method].items() if v])
            )
            s["width_um_median"] = float(
                np.median([(hi - lo) * 1e6 for lo, hi in bounds[method].values()])
            )
        if method not in ("nm-cold", "brent-cold") and brent:
            ppm = np.array(
                [(r["mfe_est"] / brent[i]["mfe_est"] - 1.0) * 1e6 for i, r in by_index.items()]
            )
            s["mfe_vs_brent_cold_ppm"] = {
                "median": float(np.median(ppm)),
                "higher": int(np.sum(ppm > MFE_APART_PPM)),
                "lower": int(np.sum(ppm < -MFE_APART_PPM)),
            }
        summaries[method] = s
    pairs = {f"{a} vs {b}": search_pair(runs, a, b) for a, b in pairs if a in runs and b in runs}
    rows = []
    for i in indices:
        first = next((by_index[i] for by_index in runs.values() if i in by_index), None)
        if first is None:
            continue
        row = {"index": i, "z01_true": first["z01_true"], "z01_guess": first["z01_guess"]}
        for method, by_index in runs.items():
            r = by_index.get(i)
            row[f"{method}_error_um"] = None if r is None else r["error_um"]
            row[f"{method}_n_reconstructions"] = None if r is None else r["n_reconstructions"]
            if method in bounds:
                lo, hi = bounds[method].get(i, (None, None))
                row[f"{method}_width_um"] = None if lo is None else (hi - lo) * 1e6
                row[f"{method}_truth_inside"] = None if lo is None else int(inside[method][i])
        rows.append(row)
    return summaries, pairs, rows


def synthetic_searches() -> tuple[dict, list[dict]]:
    runs, boxes, provenance = {}, {}, {}
    for method, (folder, intervals, what) in SYNTHETIC_SEARCHES.items():
        path = Path(folder) / "results.csv"
        if not path.exists():
            continue
        runs[method] = {r["index"]: r for r in read_rows(path) if r["index"] in SYNTHETIC}
        provenance[method] = {"folder": folder, "what": what, "intervals": intervals}
        if intervals is not None:
            boxes[method] = Box.from_csv(intervals)
    summaries, pairs, rows = search_table(runs, boxes, SEARCH_PAIRS, SYNTHETIC)
    record = {"searches": provenance, "summaries": summaries, "pairs": pairs}
    return record, rows


MEASURED_SEARCHES = {
    "nm-cold": ("b1-p4", "Nelder-Mead in Omega_z (Stage 3)"),
    "brent-cold": ("brent-b1-p4", "Brent in Omega_z (Stage 9)"),
    "brent-box95-cnn-stage8": (
        "brent-box95-cnn-loo-ensemble-b1-p4",
        "Brent in the 95 % interval of the CNN ensemble of Stage 8 (Stage 9)",
    ),
    "brent-box95-npe-stage8": (
        "brent-box95-npe-loo-ensemble-b1-p4",
        "Brent in the 95 % interval of the NPE ensemble of Stage 8 (Stage 9)",
    ),
    **{
        f"brent-box95-{family}": (
            Path(measured_search(family)).name,
            f"Brent in the 95 % interval of the ensemble trained on loo2: {f.label}",
        )
        for family, f in FAMILIES.items()
    },
}
"""The searches on the measured holograms with binning 1 and padding 4
(``results/measured/<folder>``)."""


def measured_search_table(folders: dict[str, Path]) -> tuple[dict, list[dict]]:
    """The searches with ``<folder>/<sample>.json`` on each measured hologram against [2] and
    the search named ``nm-cold`` (probes without it are left out), the MFE at each result
    against that of ``brent-cold``, and the totals over the probes."""
    out, rows = {}, []
    for name in PROBES:
        records = {}
        for method, folder in folders.items():
            path = folder / f"{name}.json"
            if path.exists():
                records[method] = (path, json.loads(path.read_text()))
        if "nm-cold" not in records:
            continue
        cold = records["nm-cold"][1]["result"]
        brent = records.get("brent-cold", (None, None))[1]
        paper2 = MEASURED_SAMPLES[name].z01_nm
        entries = {}
        for method, (path, r) in records.items():
            res = r["result"]
            lower, upper = res["bounds"]
            e = {
                "record": repo_relative(path),
                "sha256": file_sha256(path),
                "threads": r["threads"],
                "z01": res["z01"],
                "minus_paper2_nm_um": (res["z01"] - paper2) * 1e6,
                "minus_nm_cold_um": (res["z01"] - cold["z01"]) * 1e6,
                "same_as_nm_cold": bool(abs(res["z01"] - cold["z01"]) * 1e6 <= XATOL_UM),
                "within_100um_of_paper2_nm": bool(abs(res["z01"] - paper2) * 1e6 <= WITHIN_UM),
                "n_reconstructions": res["n_reconstructions"],
                "seconds": res["seconds"],
                "bounds": [lower, upper],
                "width_mm": (upper - lower) * 1e3,
                "paper2_nm_inside": bool(lower <= paper2 <= upper),
                "nm_cold_inside": bool(lower <= cold["z01"] <= upper),
                "mfe": res["mfe"],
            }
            if brent is not None and method != "nm-cold":
                e["mfe_vs_brent_cold_ppm"] = (res["mfe"] / brent["result"]["mfe"] - 1.0) * 1e6
            entries[method] = e
            rows.append(
                {
                    "sample": name,
                    "method": method,
                    **{k: v for k, v in e.items() if k not in ("record", "sha256", "bounds")},
                    "bounds_mm": "-".join(f"{b * 1e3:.4f}" for b in e["bounds"]),
                }
            )
        out[name] = {"paper2_nm": paper2, "searches": entries}
    totals = {}
    for method in folders:
        found = [s["searches"][method] for s in out.values() if method in s["searches"]]
        if not found:
            continue
        d = np.abs([e["minus_paper2_nm_um"] for e in found])
        totals[method] = {
            "samples": [k for k, s in out.items() if method in s["searches"]],
            "n_reconstructions": int(sum(e["n_reconstructions"] for e in found)),
            "within_100um_of_paper2_nm": share(d <= WITHIN_UM),
            "mae_vs_paper2_nm_um": float(d.mean()),
            "same_as_nm_cold": int(sum(e["same_as_nm_cold"] for e in found)),
            "paper2_nm_inside": int(sum(e["paper2_nm_inside"] for e in found)),
        }
    for method, t in totals.items():
        for cold in ("nm-cold", "brent-cold"):
            if method == cold or cold not in totals or t["samples"] != totals[cold]["samples"]:
                continue
            n, c = t["n_reconstructions"], totals[cold]["n_reconstructions"]
            t[f"fewer_than_{cold}"] = {"saved": c - n, "relative": 1.0 - n / c}
    return {"samples": out, "totals": totals}, rows


def measured_searches() -> tuple[dict, list[dict]]:
    return measured_search_table(
        {method: MEASURED / folder for method, (folder, _) in MEASURED_SEARCHES.items()}
    )


def cmd_compare(args) -> None:
    """Parts 2 and 3 of Laborbuch 10: the learned estimates of the four families (seeds and
    ensembles) on the synthetic samples, the leave-one-out test splits and the measured
    holograms, and Brent in the 95 % interval of each ensemble against Nelder-Mead and Brent
    cold (``results/stage10/compare.json`` and the CSVs next to it)."""
    learned, learned_rows = learned_comparison()
    measured, measured_rows = measured_learned()
    synthetic, synthetic_rows = synthetic_searches()
    searches, search_rows = measured_searches()
    record = {
        "what": "learned estimates of four inputs and Brent in their 95 % intervals, on the "
        "synthetic samples, the leave-one-out test splits and the measured holograms",
        "within_um": WITHIN_UM,
        "families": {k: dataclasses.asdict(f) for k, f in FAMILIES.items()},
        "learned": learned,
        "measured": measured,
        "searches": {"synthetic": synthetic, "measured": searches},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "compare.json", {**record, **environment()})
    for file, rows in (
        ("learned.csv", learned_rows),
        ("measured.csv", measured_rows),
        ("searches-synthetic.csv", synthetic_rows),
        ("searches-measured.csv", search_rows),
    ):
        if rows:
            columns = list(dict.fromkeys(k for row in rows for k in row))
            write_csv(args.out / file, [{k: row.get(k) for k in columns} for row in rows])
    for set_name, s in learned.items():
        for method, m in s["methods"].items():
            if not (method.endswith("/ensemble") or method == "radial-cnn/seed-0"):
                continue
            cov = m.get("coverage_95")
            print(
                f"{set_name} {method}: within 100 um {m['within_100um']['share']:.2f}, MAE "
                f"{m['mae_um']:.0f} um"
                + ("" if cov is None else f", 95 % coverage {cov['k']}/{cov['n']}")
            )
    for method, m in measured["summaries"].items():
        p = m["paper2_nm"]
        print(
            f"measured {method}: within 100 um of [2] {p['within_100um']['k']}/{m['n']}, MAE "
            f"{p['mae_um']:.0f} um, [2] in the 95 % interval {p['inside_95']['k']}/{m['n']}"
        )
    for method, s in synthetic["summaries"].items():
        print(
            f"synthetic search {method}: MAE {s['mae_um']:.0f} um, within 100 um "
            f"{s['within_um']['100']:.2f}, {s['reconstructions']:.2f} reconstructions"
            + (f", truth outside {len(s['truth_outside'])}" if "truth_outside" in s else "")
        )
    for method, t in searches["totals"].items():
        within = t["within_100um_of_paper2_nm"]
        print(
            f"measured search {method}: {t['n_reconstructions']} reconstructions on "
            f"{len(t['samples'])} probes, within 100 um of [2] {within['k']}/{within['n']}"
        )


# --- evaluate ------------------------------------------------------------------------------


def named_paths(values: list[str] | None) -> dict[str, list[Path]]:
    """``NAME=PATH[,PATH...]`` arguments; several paths are the seeds of a CNN ensemble."""
    out = {}
    for value in values or []:
        name, sep, paths = value.partition("=")
        if not sep or not name or not paths or name in out:
            raise SystemExit(f"{value}: expected NAME=PATH[,PATH...] with a new NAME")
        out[name] = [Path(p) for p in paths.split(",")]
    return out


def evaluated_table(paths: list[Path]) -> dict[str, np.ndarray]:
    """The table of one predictions file (single CNN or NPE columns), or of the equal-weight
    mixture of the CNN seeds in several."""
    if len(paths) > 1:
        return npe_table(mixture_rows([read_predictions(p) for p in paths]))
    rows = read_predictions(paths[0])
    return npe_table(rows) if "pit" in rows[0] else cnn_table(rows)


def evaluated_estimates(sources: dict[str, list[Path]]) -> dict:
    tables = {m: evaluated_table(paths) for m, paths in sources.items()}
    common = sorted(set.intersection(*({int(i) for i in t["index"]} for t in tables.values())))
    if not common:
        raise SystemExit("the predictions have no sample in common")
    tables = {m: restrict(t, common) for m, t in tables.items()}
    first = next(iter(tables))
    for m, t in tables.items():
        if np.abs(t["z01_true"] - tables[first]["z01_true"]).max() > 1e-12:
            raise SystemExit(f"{m} and {first} are predictions of different test splits")
    return {
        "sources": {m: [repo_relative(p) for p in paths] for m, paths in sources.items()},
        "n_samples": len(common),
        "methods": {m: learned_metrics(t) for m, t in tables.items()},
        "paired": {
            f"{a} vs {b}": paired_entry(tables[a], tables[b])
            for a, b in itertools.combinations(tables, 2)
        },
    }


def search_box(folder: Path, given: Path | None) -> Box | None:
    """The intervals the search in ``folder`` ran in (``box`` of its ``run.yaml``): the CSV
    named there or ``given``, which must have the SHA-256 recorded there."""
    recorded = yaml.safe_load((folder / "run.yaml").read_text()).get("box")
    if recorded is None:
        if given is not None:
            raise SystemExit(f"{folder} did not search in intervals, {given} is not its box")
        return None
    path = given or Path(recorded["source"])
    if not path.exists():
        raise SystemExit(f"{folder} searched in {path}, which is missing: give NAME=DIR,CSV")
    box = Box.from_csv(path, recorded["level"])
    if box.sha256 != recorded["sha256"]:
        raise SystemExit(f"{path} is not the CSV that {folder} searched in (SHA-256)")
    return box


def evaluated_searches(sources: dict[str, list[Path]]) -> dict:
    runs, boxes = {}, {}
    for name, paths in sources.items():
        if len(paths) > 2:
            raise SystemExit(f"{name}: one run directory and the CSV of its intervals at most")
        runs[name] = {r["index"]: r for r in read_rows(paths[0] / "results.csv")}
        box = search_box(paths[0], paths[1] if len(paths) > 1 else None)
        if box is not None:
            boxes[name] = box
    common = sorted(set.intersection(*(set(by_index) for by_index in runs.values())))
    if not common:
        raise SystemExit("the searches have no sample in common")
    for i in common:
        truths = [by_index[i]["z01_true"] for by_index in runs.values()]
        if max(truths) - min(truths) > 1e-12:
            raise ValueError(f"sample {i} differs between the searches")
    runs = {name: {i: by_index[i] for i in common} for name, by_index in runs.items()}
    summaries, pairs, _ = search_table(runs, boxes, itertools.combinations(runs, 2), common)
    return {
        "sources": {name: [repo_relative(p) for p in paths] for name, paths in sources.items()},
        "intervals": {name: box.identity() for name, box in boxes.items()},
        "n_samples": len(common),
        "summaries": summaries,
        "pairs": pairs,
    }


def evaluated_measured_searches(sources: dict[str, list[Path]]) -> dict:
    """The searches of Stages 3 and 9 in ``Omega_z`` as ``nm-cold`` and ``brent-cold`` unless
    given."""
    folders = {"nm-cold": MEASURED / "b1-p4", "brent-cold": MEASURED / "brent-b1-p4"}
    for name, paths in sources.items():
        if len(paths) != 1:
            raise SystemExit(f"{name}: one folder per search")
        folders[name] = paths[0]
    record, _ = measured_search_table(folders)
    return {"sources": {m: repo_relative(f) for m, f in folders.items()}, **record}


def evaluated_measured(sources: dict[str, list[Path]]) -> dict:
    """One folder: the records ``<sample>.json`` of an ensemble or a seed (the NPE ensemble of
    ``run_stage7_loo_study.py measured`` too); several: the records of the CNN seeds of
    ``run_stage6_study.py measured``, as their mixture."""
    samples, by_method = {}, {}
    for name in PROBES:
        refs = {"paper2_nm": MEASURED_SAMPLES[name].z01_nm}
        cold = nm_cold_z01(name)
        if cold is not None:
            refs["nm_cold"] = cold
        entries = {}
        for method, folders in sources.items():
            paths = [folder / f"{name}.json" for folder in folders]
            if not all(p.exists() for p in paths):
                continue
            if len(paths) > 1:
                record = mixture_record(name, paths, measured_references(name))
            else:
                record = json.loads(paths[0].read_text())
                if "runs" in record:
                    record = npe_record(name, paths[0])
            entries[method] = estimate_entry(paths[0], record, refs)
            by_method.setdefault(method, []).append(entries[method])
        if entries:
            samples[name] = {"references": refs, "methods": entries}
    return {
        "sources": {m: [repo_relative(f) for f in folders] for m, folders in sources.items()},
        "samples": samples,
        "summaries": {m: measured_summary(es) for m, es in by_method.items()},
    }


def cmd_evaluate(args) -> None:
    """The metrics of ``compare`` for estimates and searches given by path, e.g. the runs of
    Maxwell (docs/maxwell-gpu.md 8.5): on one test split every predictions file and every two
    paired, on the samples all of them have; every search and every two paired, likewise;
    the estimates and the searches on the measured holograms against [2] and Nelder-Mead
    cold."""
    record = {"what": "metrics of compare for estimates and searches given by path"}
    parts = (
        ("learned", evaluated_estimates, args.predictions),
        ("searches", evaluated_searches, args.searches),
        ("measured", evaluated_measured, args.measured),
        ("measured_searches", evaluated_measured_searches, args.measured_searches),
    )
    for key, evaluate, values in parts:
        if values:
            record[key] = evaluate(named_paths(values))
    write_json(args.out, {**record, **environment()})
    for method, m in record.get("learned", {}).get("methods", {}).items():
        cov = m.get("coverage_95")
        print(
            f"{method}: {m['n']} samples, within 100 um {m['within_100um']['share']:.2f}, MAE "
            f"{m['mae_um']:.0f} um"
            + ("" if cov is None else f", 95 % coverage {cov['k']}/{cov['n']}")
        )
    for method, s in record.get("searches", {}).get("summaries", {}).items():
        print(
            f"search {method}: {s['n_samples']} samples, MAE {s['mae_um']:.0f} um, within "
            f"100 um {s['within_um']['100']:.2f}, {s['reconstructions']:.2f} reconstructions"
            + (
                f", truth outside the interval {len(s['truth_outside'])}/{s['n_samples']}"
                if "truth_outside" in s
                else ""
            )
        )
    for method, m in record.get("measured", {}).get("summaries", {}).items():
        p = m["paper2_nm"]
        print(
            f"measured {method}: within 100 um of [2] {p['within_100um']['k']}/{m['n']}, MAE "
            f"{p['mae_um']:.0f} um, [2] in the 95 % interval {p['inside_95']['k']}/{m['n']}"
        )
    for method, t in record.get("measured_searches", {}).get("totals", {}).items():
        within = t["within_100um_of_paper2_nm"]
        print(
            f"measured search {method}: {t['n_reconstructions']} reconstructions on "
            f"{len(t['samples'])} probes, within 100 um of [2] {within['k']}/{within['n']}"
        )
    print(f"written to {repo_relative(args.out)}")


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

    mx = sub.add_parser("mixture", help="the CNN seeds given by path as one mixture")
    mx.add_argument("--predictions", type=Path, nargs="+", required=True, help="predictions.csv")
    mx.add_argument("--out", type=Path, required=True, help="folder for predictions.csv, run.yaml")
    mx.set_defaults(func=cmd_mixture)

    me = sub.add_parser("measured", help="the seeds and the ensemble on the measured holograms")
    me.add_argument("--holograms", type=Path, required=True)
    me.add_argument("--families", **families)
    me.add_argument("--samples", **samples)
    me.add_argument("--repeats", type=int, default=20, help="timed estimates")
    me.add_argument(
        "--runs",
        type=Path,
        default=Path("results/stage10/loo2"),
        help="<runs>/<family>/seed-<k>/loo-<sample> and <runs>/npe/matched-<sample>[-seed<k>]",
    )
    me.add_argument(
        "--out",
        type=Path,
        default=Path("results/measured"),
        help="<out>/<family>-loo2-ensemble/<sample>.json",
    )
    me.set_defaults(func=cmd_measured)

    se = sub.add_parser("search", help="Brent in the 95 % interval of an ensemble, one sample")
    se.add_argument("--family", choices=list(FAMILIES), required=True)
    se.add_argument("--index", type=int, choices=SYNTHETIC, required=True)
    se.set_defaults(func=cmd_search)

    mg = sub.add_parser("merge", help="join the searches of each family")
    mg.add_argument("--families", **families)
    mg.set_defaults(func=cmd_merge)

    tx = sub.add_parser("texture", help="the fit sets with the textures of Stages 6 and 10")
    tx.add_argument("--holograms", type=Path, required=True)
    tx.add_argument("--samples", **samples)
    tx.add_argument("--n", type=int, default=FIT_SAMPLES, help="test samples per texture")
    tx.set_defaults(func=cmd_texture)

    sp = sub.add_parser("spectra", help="the measured holograms among the leave-one-out splits")
    sp.add_argument("--holograms", type=Path, required=True)
    sp.add_argument("--samples", **samples)
    sp.add_argument("--limit", type=int, help="first holograms of each split only")
    sp.set_defaults(func=cmd_spectra)

    co = sub.add_parser("compare", help="the learned estimates and the searches side by side")
    co.add_argument("--out", type=Path, default=STAGE10)
    co.set_defaults(func=cmd_compare)

    ev = sub.add_parser("evaluate", help="the metrics of compare for runs given by path")
    ev.add_argument(
        "--predictions",
        nargs="+",
        metavar="NAME=CSV[,CSV...]",
        help="predictions.csv on one test split; several CSVs: the seeds of a CNN ensemble",
    )
    ev.add_argument(
        "--searches",
        nargs="+",
        metavar="NAME=DIR[,CSV]",
        help="runs with results.csv; CSV: the intervals a run searched in, if its run.yaml names "
        "a path that is missing here; brent-cold: the reference of the MFE",
    )
    ev.add_argument(
        "--measured",
        nargs="+",
        metavar="NAME=DIR[,DIR...]",
        help="folders with <sample>.json; several folders: the seeds of a CNN ensemble",
    )
    ev.add_argument(
        "--measured-searches",
        nargs="+",
        metavar="NAME=DIR",
        help="folders with <sample>.json of run_stage3_experiments.py measured; nm-cold and "
        "brent-cold are those of Stages 3 and 9 unless given",
    )
    ev.add_argument("--out", type=Path, required=True, help="JSON file")
    ev.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
