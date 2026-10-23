"""Stage 9: the step of the MFE at the tooth, and Brent's bounded method next to Nelder-Mead.

Laborbuch 09. The MFE has the same bits for every thread count since this stage
(:func:`holofocus.device.one_thread`), those of a run with one thread.

* ``sources``: the central 68 and 95 % intervals ``z01_est +- z sigma`` of the Gaussian CNN
  of Stage 4 on the test split of cpu-4x (``results/stage9/intervals-cnn-test.csv``); the NPE
  ensemble of Stage 7 has them in its ``predictions.csv``.
* ``queue``: the searches on the 48 synthetic samples of the Stage 3 baseline (:data:`SEARCHES`):
  Brent cold in ``Omega_z``, and Brent and Nelder-Mead in the 95 % interval of the NPE
  ensemble and of the CNN; and the Stage 3 search itself again on 8 of them, with the
  numerics of this stage (``repeat``). Each lane (``--lane``) takes the next job nobody has
  claimed and writes to ``runs/stage9/cpu-4x-test-<search>/lane-<lane>``; ``merge`` joins the
  lanes into ``results/baseline/cpu-4x-test-<search>``.
* ``scan``: ASRM reconstructions of a measured hologram at given ``z01``, with the settings of
  the searches on the measured holograms (binning 1, padding 4): per point the MFE, the loss of
  every 10th iteration, the losses at the ends of the stages and the statistics of the object
  (``results/stage9/scan-<sample>.json``). The object and the whole loss history go to
  ``results/stage9/scan-<sample>/<z01 in mm>.npz`` (not tracked). Points already in the record
  are skipped, so a scan can be extended. With ``--threads 3 --out results/stage9/threads-3``
  at points of the scans with one thread, it checks that the thread count leaves the MFE as
  it is. ``scan`` also runs with the library of before this stage, whose MFE depends on the
  thread count: copy this script into a checkout of ``4349d6a`` and run it there with
  ``PYTHONPATH=src`` and ``--out results/stage9/threads-3-4349d6a``.
* ``steps``: all reconstructions of the measured holograms with binning 1 and padding 4 (the
  searches of Stages 3, 5, 8 and 9 and the scans) as MFE against ``z01``, per probe and
  numerics, screened for steps; the same ``z01`` with the same numerics, whose MFE must have
  the same bits, and with both numerics; where the loss histories
  of the scans part; the objects of the scans against each other; the Stage 3 search again on
  synthetic samples (``results/stage9/steps.json``, ``mfe-points.csv``,
  ``steps-objects.json``).
* ``compare``: Brent against Nelder-Mead, cold and in the 95 % intervals of the learned
  models: on the four measured holograms against the references of [2], and on the 48
  synthetic samples against the truth; error, share within 100 um, reconstructions, time at
  the rate of a search with 3 threads, and the reconstructions an interval saves per method
  (``results/stage9/compare.json``, ``measured.csv``, ``synthetic.csv``).

Usage (Laborbuch 09)::

    python scripts/run_stage9_study.py sources
    python scripts/run_stage9_study.py --threads 3 queue --lane a   # and --threads 1 --lane b
    python scripts/run_stage9_study.py merge
    python scripts/run_stage9_study.py --threads 1 scan --holograms DIR --sample tooth \\
        --z01-mm 80.695 80.700 80.710
    python scripts/run_stage9_study.py steps
    python scripts/run_stage9_study.py compare

``DIR`` is ``Examples/data/holograms`` of the Zenodo record 10.5281/zenodo.8349365. Run it
from the root of the repository, so the paths in the records stay relative to it.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import norm

from holofocus.asrm import asrm_reconstruct, stage_grid
from holofocus.baseline import BaselineConfig, merge_runs, read_rows, read_traces, run_baseline
from holofocus.baseline.runner import file_sha256
from holofocus.learned import bootstrap_ci, read_predictions
from holofocus.measured import MEASURED_SAMPLES, load_measured
from holofocus.yamlconfig import plain
from run_stage3_experiments import environment, measured_settings, write_json

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
STAGE9 = RESULTS / "stage9"
CNN_PREDICTIONS = RESULTS / "learned" / "spectrum-cnn-wide" / "test" / "predictions.csv"
"""The Gaussian CNN of Stage 4 on the test split of cpu-4x (one network, trained on the 10000
samples of cpu-4x and cpu-4x-extra)."""
NPE_PREDICTIONS = RESULTS / "npe" / "npe-nsf-ensemble" / "test" / "predictions.csv"
"""The ensemble of the five NPE seeds of Stage 7 on the same split, with central intervals."""
CNN_INTERVALS = STAGE9 / "intervals-cnn-test.csv"
LEVELS = ("68", "95")
CONFIGS = REPO / "configs" / "baseline"
RUNS = REPO / "runs" / "stage9"
BASELINE = RESULTS / "baseline"
TEST_SPLIT = REPO / "data" / "cpu-4x" / "test.h5"
SYNTHETIC = tuple(range(48))
"""The samples of the Stage 3 baseline (``results/baseline/cpu-4x-test``)."""
REPEATED = (11, 46, 0, 1, 2, 3, 4, 5)
"""Samples whose Stage 3 search is repeated with the numerics of this stage: the two with
probed MFE values less than 2 ppm apart, and six more."""
SEARCHES = {
    "repeat": (CONFIGS / "paper2-cpu.yaml", None, REPEATED),
    "brent": (CONFIGS / "paper2-cpu-brent.yaml", None, SYNTHETIC),
    "brent-box95-npe-ensemble": (CONFIGS / "paper2-cpu-brent.yaml", NPE_PREDICTIONS, SYNTHETIC),
    "box95-npe-ensemble": (CONFIGS / "paper2-cpu-box.yaml", NPE_PREDICTIONS, SYNTHETIC),
    "brent-box95-cnn": (CONFIGS / "paper2-cpu-brent.yaml", CNN_INTERVALS, SYNTHETIC),
    "box95-cnn": (CONFIGS / "paper2-cpu-box.yaml", CNN_INTERVALS, SYNTHETIC),
}
"""The searches on the synthetic samples (``results/baseline/cpu-4x-test-<name>``): config,
the CSV of the 95 % intervals (``None``: ``Omega_z``) and the samples. ``repeat`` is the
Stage 3 search itself (with the reconstruction at the truth)."""
MEASURED = RESULTS / "measured"
EARLIER = (
    "b1-p4",
    "warm-b1-p4",
    "warm-cnn-loo-ensemble-b1-p4",
    "warm-npe-loo-ensemble-b1-p4",
    "box95-cnn-loo-ensemble-b1-p4",
    "box95-npe-loo-ensemble-b1-p4",
)
"""The searches on the measured holograms with binning 1 and padding 4 before this stage
(Laborbuch 03, 05 and 08); their MFE has the numerics of their thread count."""
MEASURED_SEARCHES = (
    "brent-b1-p4",
    "brent-box95-cnn-loo-ensemble-b1-p4",
    "brent-box95-npe-loo-ensemble-b1-p4",
    "b1-p4-repeat",
)
"""Those of this stage: Brent cold and in the 95 % intervals of the ensembles of Stage 8, and
the full Nelder-Mead search of Stage 3 again."""
BEFORE = "threads-3-4349d6a"
"""``scan`` with the library of before this stage and 3 threads (``results/stage9/<BEFORE>``)."""
AFTER = "threads-3"
"""``scan`` with the library of this stage and 3 threads (``results/stage9/<AFTER>``), at points
the scans with one thread have too."""
STEP = {"spacing_um": 50.0, "jump_ppm": 50.0, "ratio": 3.0}
"""Screening for steps among the neighbouring points of a probe with one numerics: closer than
``spacing_um``, more than ``jump_ppm`` apart and a secant ``ratio`` times as steep as both
neighbouring ones (a jump); or more than ``jump_ppm`` downhill away from the minimum (a
reversal)."""
FIRST_STAGE = "first-stage"
"""Where the scans of the first stage of the plan go (``scan --stages 1``), under
``results/stage9``; those with the library of before this stage under ``first-stage/BEFORE``."""
OMEGA = "omega"
"""Full scans of the tooth over ``Omega_z`` where its first stage changes branch
(``results/stage9/omega``), apart from those about the minimum."""
BRANCH = {"above": 0.05, "window_mm": 1.0}
"""The first stage lies on the upper branch where its loss lies more than ``above`` over the
smallest of the first stage within ``window_mm`` of it (same probe, either numerics: they move
it by less than 0.01 %, and the points with 3 threads before this stage lie too sparsely for a
reference of their own); a switch is a pair of neighbouring points on different branches. The
two branches of the tooth lie 8-11 % apart; on one branch the loss changes by less than 1 % in
1 mm, and from every point of the tooth on the upper branch one on the lower lies at most
0.4 mm away."""
BRANCH_PAIR_UM = 2.0
"""Two neighbouring full reconstructions whose first stage lies on different branches give the
offset of the MFE between the branches where they lie at most this far apart: the MFE on one
branch changes by far less over so short a distance than between the branches. Farther apart,
the bend of the MFE over ``z01`` is as large as the offset."""
BRANCH_SLOPE_UM = 250.0
"""How far from a pair of :data:`BRANCH_PAIR_UM` the full reconstructions on the same branch
may lie that give the slope of the MFE there: up to 42 ppm per um on the tooth, as much over
1 um as the offset far from the focus; over 250 um the slope itself changes by far less."""
PAIR_LOSSES = ("first_stage_loss", "second_stage_loss", "last_stage_smallest", "object_loss")
"""The losses of the full scans that ``branches`` sets upper over lower for every pair too, as
computed (without the slope): at the end of the first and second stage, the smallest of the
last stage and at the returned object."""
BRANCH_SLOPE_POINTS = 2
"""How many of them at most, the nearest: near a switch single reconstructions lie some 50 ppm
off the line through their neighbours (79.15 mm on the tooth, 6 um from the switch), which
over a few um changes the slope by about 10 ppm per um."""
WINDOWS_UM = (250.0, 500.0)
"""Half-widths of the windows about the minimum in which ``steps`` reports how densely the
points lie (:func:`coverage`)."""
LOSS_SPLIT = 1e-4
"""Relative difference of two loss histories from which ``steps`` counts them as split."""
MEASURED_METHODS = {
    "nm-cold": ("b1-p4", "Nelder-Mead in Omega_z (Stage 3)"),
    "nm-cold-repeat": ("b1-p4-repeat", "the same again, with the numerics of this stage"),
    "nm-box95-cnn": (
        "box95-cnn-loo-ensemble-b1-p4",
        "Nelder-Mead in the 95 % interval of the CNN ensemble (Stage 8)",
    ),
    "nm-box95-npe": (
        "box95-npe-loo-ensemble-b1-p4",
        "Nelder-Mead in the 95 % interval of the NPE ensemble (Stage 8)",
    ),
    "brent-cold": ("brent-b1-p4", "Brent in Omega_z"),
    "brent-box95-cnn": (
        "brent-box95-cnn-loo-ensemble-b1-p4",
        "Brent in the 95 % interval of the CNN ensemble",
    ),
    "brent-box95-npe": (
        "brent-box95-npe-loo-ensemble-b1-p4",
        "Brent in the 95 % interval of the NPE ensemble",
    ),
}
"""The searches on the measured holograms that ``compare`` puts side by side
(``results/measured/<folder>``)."""
SYNTHETIC_METHODS = {
    "nm-cold": ("cpu-4x-test", "Nelder-Mead in Omega_z (Stage 3)"),
    "nm-cold-repeat": ("cpu-4x-test-repeat", "the same on 8 samples, numerics of this stage"),
    "brent-cold": ("cpu-4x-test-brent", "Brent in Omega_z"),
    "nm-box95-npe": (
        "cpu-4x-test-box95-npe-ensemble",
        "Nelder-Mead in the 95 % interval of the NPE ensemble",
    ),
    "brent-box95-npe": (
        "cpu-4x-test-brent-box95-npe-ensemble",
        "Brent in the 95 % interval of the NPE ensemble",
    ),
    "nm-box95-cnn": ("cpu-4x-test-box95-cnn", "Nelder-Mead in the 95 % interval of the CNN"),
    "brent-box95-cnn": ("cpu-4x-test-brent-box95-cnn", "Brent in the 95 % interval of the CNN"),
}
"""The same on the synthetic samples (``results/baseline/<folder>``)."""
INTERVALS = {"npe": NPE_PREDICTIONS, "cnn": CNN_INTERVALS}
"""The sources of the 95 % intervals on the synthetic samples."""
PAIRED_ERRORS = (
    ("brent-cold", "nm-cold"),
    ("brent-box95-npe", "nm-box95-npe"),
    ("brent-box95-cnn", "nm-box95-cnn"),
    ("nm-box95-npe", "nm-cold"),
    ("brent-box95-npe", "nm-cold"),
    ("nm-box95-cnn", "nm-cold"),
    ("brent-box95-cnn", "nm-cold"),
    ("nm-cold-repeat", "nm-cold"),
)
"""The searches on the synthetic samples whose errors ``compare`` sets against each other,
sample by sample: Brent against Nelder-Mead in the same interval, each search in an interval
against the full search of [2], and the numerics alone."""
WITHIN_UM = (50, 100, 250, 1000)
MFE_APART_PPM = 100.0
"""How far the MFE at the result of a search must lie from that of the cold Nelder-Mead search
to count as another minimum: far beyond the few ppm of the numerics."""
XATOL_UM = 100.0
"""Tolerance of the searches in ``z01``: results closer than this are the same."""
RESAMPLES = 10000


def write_csv(path: Path, rows: list[dict]) -> None:
    """Rows with full precision (``repr`` of the floats): the bounds define the searches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


LOSS_EVERY = 10
"""Every how many iterations ``scan`` keeps the loss in the record (all of it in the npz)."""


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def point_key(z01: float) -> int:
    """``z01`` in picometres: two scan points closer than that are the same."""
    return round(z01 * 1e12)


def object_statistics(obj: torch.Tensor) -> dict:
    phase, absorption = obj.real.double(), obj.imag.double()
    return {
        "phase_mean": float(phase.mean()),
        "phase_std": float(phase.std()),
        "phase_min": float(phase.min()),
        "phase_max": float(phase.max()),
        "absorption_mean": float(absorption.mean()),
        "absorption_std": float(absorption.std()),
    }


def cmd_sources(args) -> None:
    """Central intervals of the Gaussian CNN on the synthetic test split, for ``--box``."""
    rows = read_predictions(args.cnn)
    levels = {key: float(norm.ppf(0.5 + 0.005 * float(key))) for key in LEVELS}
    out_rows = []
    for row in rows:
        out = {"index": int(row["index"]), "z01_est": row["z01_est"], "sigma": row["sigma"]}
        for key, z in levels.items():
            out[f"z01_lower_{key}"] = row["z01_est"] - z * row["sigma"]
            out[f"z01_upper_{key}"] = row["z01_est"] + z * row["sigma"]
        out_rows.append(out)
    target = args.out / CNN_INTERVALS.name
    write_csv(target, out_rows)
    write_json(
        target.with_suffix(".json"),
        {
            "what": "central intervals z01_est +- z sigma of the Gaussian CNN (calibrated sigma) "
            "on the test split of cpu-4x, the search intervals of the bounded searches",
            "source": repo_relative(args.cnn),
            "source_sha256": file_sha256(args.cnn),
            "z": levels,
            "n_samples": len(out_rows),
            "sha256": file_sha256(target),
        },
    )
    print(f"{target}: {len(out_rows)} samples, z {levels}")


def queue_jobs() -> list[tuple[str, int]]:
    """``(search, index)`` in the order the lanes take them: the repeated Stage 3 searches,
    then per sample the cold Brent and the two searches in the interval of the NPE ensemble,
    then the two in that of the CNN."""
    groups = (
        [name for name in SEARCHES if name == "repeat"],
        [name for name in SEARCHES if name != "repeat" and not name.endswith("cnn")],
        [name for name in SEARCHES if name.endswith("cnn")],
    )
    return [
        (name, index)
        for group in groups
        for index in SYNTHETIC
        for name in group
        if index in SEARCHES[name][2]
    ]


def lane_dir(name: str, lane: str) -> Path:
    return RUNS / f"cpu-4x-test-{name}" / f"lane-{lane}"


def here(path: Path) -> Path:
    """``path`` relative to the working directory: the records keep the paths as given, and
    the source of the intervals is part of the key of a run."""
    return Path(os.path.relpath(path))


def cmd_queue(args) -> None:
    """Work through :func:`queue_jobs` as one lane; a job is claimed by creating its file in
    ``--claims``, so two lanes never take the same one. A lane takes up again the jobs it
    claimed and did not finish."""
    # Not at the top: ``scan`` also runs with the library of before this stage.
    from holofocus.baseline.runner import Box

    args.claims.mkdir(parents=True, exist_ok=True)
    for name, index in queue_jobs():
        claim = args.claims / f"{name}-{index}"
        try:
            with open(claim, "x") as fh:
                fh.write(args.lane)
        except FileExistsError:
            if claim.read_text() != args.lane:
                continue
        config_path, box_path, _ = SEARCHES[name]
        out = here(lane_dir(name, args.lane))
        if index in {row["index"] for row in read_rows(out / "results.csv")}:
            continue
        print(f"lane {args.lane}: {name}, sample {index}", flush=True)
        run_baseline(
            BaselineConfig.from_yaml(config_path),
            here(args.data),
            out,
            indices=[index],
            box=None if box_path is None else Box.from_csv(here(box_path)),
            log=lambda msg: print(f"  {msg}", flush=True),
        )


def cmd_merge(args) -> None:
    """Join the lanes of each search into ``results/baseline/cpu-4x-test-<name>``, and note
    which lane (thread count) took which sample (``results/stage9/lanes.csv``)."""
    rows = []
    for name in SEARCHES:
        lanes = sorted(p for p in (RUNS / f"cpu-4x-test-{name}").glob("lane-*") if p.is_dir())
        if lanes:
            merge_runs([here(lane) for lane in lanes], here(BASELINE / f"cpu-4x-test-{name}"))
        for lane in lanes:
            threads = yaml.safe_load((lane / "run.yaml").read_text())["threads"]
            for row in read_rows(lane / "results.csv"):
                rows.append(
                    {
                        "search": name,
                        "index": row["index"],
                        "lane": lane.name.removeprefix("lane-"),
                        "threads": threads,
                        "n_reconstructions": row["n_reconstructions"],
                        "seconds": row["seconds"],
                    }
                )
    if rows:
        rows.sort(key=lambda r: (r["search"], r["index"]))
        write_csv(args.out / "lanes.csv", rows)


def numerics(threads: int, before: bool) -> int:
    """The thread count whose numerics an MFE has: that of one thread since this stage."""
    return threads if before else 1


def scan_records(stage9: Path):
    """The records of ``scan`` with their source and whether they used the library of before
    this stage: with one thread (also over ``Omega_z``, :data:`OMEGA`), with 3 threads
    (:data:`AFTER`) and before (:data:`BEFORE`)."""
    for directory, source, before in (
        (stage9, "scan", False),
        (stage9 / OMEGA, f"scan-{OMEGA}", False),
        (stage9 / AFTER, f"scan-{AFTER}", False),
        (stage9 / BEFORE, f"scan-{BEFORE}", True),
    ):
        for path in sorted(directory.glob("scan-*.json")):
            yield json.loads(path.read_text()), source, before


def mfe_points(measured: Path, stage9: Path) -> list[dict]:
    """Every reconstruction of the measured holograms with binning 1 and padding 4, from the
    searches and the scans, with its MFE relative to the smallest of the probe and numerics."""
    rows = []

    def add(sample, source, before, z01, mfe, seconds, env, scanned=None, every=None):
        rows.append(
            {
                "sample": sample,
                "source": source,
                "numerics": numerics(env["threads"], before),
                "z01": z01,
                "mfe": mfe,
                "object_loss": scanned.get("object_loss") if scanned else None,
                "first_stage_loss": scanned["stage_losses"][0] if scanned else None,
                "second_stage_loss": scanned["stage_losses"][1] if scanned else None,
                "last_stage_smallest": min(scanned["loss"][scanned["stage_ends"][-2] // every :])
                if scanned
                else None,
                "seconds": seconds,
                "threads": env["threads"],
                "git_commit": env["git_commit"],
            }
        )

    for name in EARLIER + MEASURED_SEARCHES:
        for path in sorted((measured / name).glob("*.json")):
            record = json.loads(path.read_text())
            for e in record["result"]["evaluations"]:
                if e["reconstructed"]:
                    before = name in EARLIER
                    add(record["sample"], name, before, e["z01"], e["mfe"], e["seconds"], record)
    for record, source, before in scan_records(stage9):
        for p in record["points"]:
            add(
                record["sample"],
                source,
                before,
                p["z01"],
                p["mfe"],
                p["seconds"],
                p,
                p,
                record["loss_every"],
            )
    smallest = defaultdict(lambda: np.inf)
    for row in rows:
        key = row["sample"], row["numerics"]
        smallest[key] = min(smallest[key], row["mfe"])
    for row in rows:
        row["ppm"] = (row["mfe"] / smallest[row["sample"], row["numerics"]] - 1.0) * 1e6
    return rows


def distinct(rows: list[dict]) -> tuple[list[dict], float]:
    """The points at different ``z01`` (exactly), sorted, and the largest relative difference
    between reconstructions at the same ``z01``."""
    first, worst = {}, 0.0
    for row in rows:
        if row["z01"] in first:
            worst = max(worst, abs(row["mfe"] / first[row["z01"]]["mfe"] - 1.0))
        else:
            first[row["z01"]] = row
    return sorted(first.values(), key=lambda r: r["z01"]), worst


def screen(points: list[dict]) -> dict:
    """Jumps and reversals (:data:`STEP`) among neighbouring points sorted by ``z01``."""
    z_um = np.array([p["z01"] for p in points]) * 1e6
    ppm = np.array([p["ppm"] for p in points])
    lowest = int(np.argmin(ppm))
    slopes = np.diff(ppm) / np.diff(z_um)
    jumps, reversals = [], []
    for i, slope in enumerate(slopes):
        rise = ppm[i + 1] - ppm[i]
        pair = {
            "z01": [points[i]["z01"], points[i + 1]["z01"]],
            "ppm": [float(ppm[i]), float(ppm[i + 1])],
            "slope_ppm_per_um": float(slope),
            "sources": [points[i]["source"], points[i + 1]["source"]],
        }
        inner = 0 < i < len(slopes) - 1
        if inner and z_um[i + 1] - z_um[i] <= STEP["spacing_um"] and abs(rise) > STEP["jump_ppm"]:
            neighbours = [float(slopes[i - 1]), float(slopes[i + 1])]
            steepest = max(abs(s) for s in neighbours)
            ratio = abs(slope) / steepest if steepest > 0 else float("inf")
            if ratio > STEP["ratio"]:
                jumps.append({**pair, "neighbour_slopes": neighbours, "ratio": ratio})
        if (i >= lowest and rise < -STEP["jump_ppm"]) or (i < lowest and rise > STEP["jump_ppm"]):
            reversals.append(pair)
    return {"jumps": jumps, "reversals": reversals}


def coverage(points: list[dict], centre: float) -> dict:
    """Per window ``centre +- w`` (:data:`WINDOWS_UM`) the share in which neighbouring points
    are at most ``STEP["spacing_um"]`` apart (the ends of the window count as points), and the
    widest gap: where :func:`screen` could not see a jump."""
    z_um = np.array([p["z01"] for p in points]) * 1e6
    out = {}
    for w in WINDOWS_UM:
        lo, hi = centre * 1e6 - w, centre * 1e6 + w
        gaps = np.diff(np.concatenate([[lo], z_um[(z_um > lo) & (z_um < hi)], [hi]]))
        out[f"{w:g}"] = {
            "share": float(gaps[gaps <= STEP["spacing_um"]].sum() / (2 * w)),
            "widest_gap_um": float(gaps.max()),
        }
    return out


def repeats(rows: list[dict]) -> list[dict]:
    """Reconstructions at exactly the same ``z01`` with the same numerics: whether their MFE
    has the same bits across thread counts and versions of the library."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample"], row["numerics"], row["z01"]].append(row)
    return [
        {
            "sample": sample,
            "numerics": n,
            "z01": z01,
            "same_bits": len({r["mfe"] for r in group}) == 1,
            "reconstructions": [
                {k: r[k] for k in ("source", "threads", "git_commit", "mfe")} for r in group
            ],
        }
        for (sample, n, z01), group in sorted(groups.items())
        if len(group) > 1
    ]


def across_numerics(rows: list[dict], losses: dict) -> list[dict]:
    """Reconstructions at exactly the same ``z01`` with both numerics, with the branch of the
    first stage of each where it is known (:func:`first_stage_losses`, ``None`` otherwise)."""
    labels = {
        key: upper_branch(known, probe_losses(losses, key[0])) for key, known in losses.items()
    }
    by_point = defaultdict(dict)
    for row in rows:
        by_point[row["sample"], row["z01"]].setdefault(row["numerics"], row)
    pairs = []
    for (sample, z01), found in sorted(by_point.items()):
        if 1 in found and 3 in found:
            one, three = found[1], found[3]
            pairs.append(
                {
                    "sample": sample,
                    "z01": z01,
                    "mfe_one_thread": one["mfe"],
                    "mfe_3_threads": three["mfe"],
                    "relative_ppm": (three["mfe"] / one["mfe"] - 1.0) * 1e6,
                    "ppm_one_thread": one["ppm"],
                    "ppm_3_threads": three["ppm"],
                    "sources": [one["source"], three["source"]],
                    "upper": [labels.get((sample, n), {}).get(point_key(z01)) for n in (1, 3)],
                }
            )
    return pairs


def loss_split(a: list[float], b: list[float], every: int, ends: list[int]) -> dict:
    """Where two loss histories (every ``every``-th iteration) part: their relative difference
    at the ends of the stages and its largest per stage, the first iteration at which it
    exceeds :data:`LOSS_SPLIT` and the first from which it stays above."""
    rel = np.abs(np.asarray(b) / np.asarray(a) - 1.0)
    above = rel > LOSS_SPLIT
    split = first = None
    if above.any():
        first = int(np.flatnonzero(above)[0] + 1) * every
    if above[-1]:
        below = np.flatnonzero(~above)
        split = int(((below[-1] + 1) if below.size else 0) + 1) * every
    starts = [0] + [e // every for e in ends[:-1]]
    return {
        "at_stage_ends": [float(rel[e // every - 1]) for e in ends],
        "largest_per_stage": [
            float(rel[s : e // every].max()) for s, e in zip(starts, ends, strict=True)
        ],
        "first_above_iteration": first,
        "split_iteration": split,
    }


def scan_points(stage9: Path):
    """``(record, point, numerics)`` of every scan point; one with 3 threads and the library of
    this stage (:data:`AFTER`) has the bits of one thread and counts only where no scan with one
    thread has its ``z01``."""
    seen = set()
    for record, source, before in scan_records(stage9):
        for p in record["points"]:
            key = record["sample"], p["z01"]
            if source == f"scan-{AFTER}" and key in seen:
                continue
            if not before:
                seen.add(key)
            yield record, p, numerics(p["threads"], before)


def scan_pairs(stage9: Path, minima: dict) -> list[dict]:
    """Loss histories of neighbouring scan points and of the same ``z01`` in both numerics."""
    scans = []
    for record, p, n in scan_points(stage9):
        ppm = (p["mfe"] / minima[record["sample"], n] - 1.0) * 1e6
        scans.append(
            {
                **p,
                "sample": record["sample"],
                "numerics": n,
                "ppm": ppm,
                "every": record["loss_every"],
            }
        )
    pairs = []

    def pair(kind, a, b):
        pairs.append(
            {
                "kind": kind,
                "sample": a["sample"],
                "z01": [a["z01"], b["z01"]],
                "numerics": [a["numerics"], b["numerics"]],
                "ppm": [a["ppm"], b["ppm"]],
                "loss": loss_split(a["loss"], b["loss"], a["every"], a["stage_ends"]),
            }
        )

    for sample in sorted({s["sample"] for s in scans}):
        for n in (1, 3):
            line = sorted(
                (s for s in scans if s["sample"] == sample and s["numerics"] == n),
                key=lambda s: s["z01"],
            )
            for a, b in zip(line, line[1:], strict=False):
                pair("neighbours", a, b)
        ones = {s["z01"]: s for s in scans if s["sample"] == sample and s["numerics"] == 1}
        for s in scans:
            if s["sample"] == sample and s["numerics"] == 3 and s["z01"] in ones:
                pair("same z01", ones[s["z01"]], s)
    return pairs


def synthetic_repeat(stage3: Path, repeat: Path) -> list[dict]:
    """The Stage 3 search (3 threads) against itself with the numerics of this stage: the MFE
    at the truth and along the common start of the two paths, and where they part."""
    old_rows = {row["index"]: row for row in read_rows(stage3 / "results.csv")}
    old_traces = {t["index"]: t for t in read_traces(stage3 / "traces.jsonl")}
    new_traces = {t["index"]: t for t in read_traces(repeat / "traces.jsonl")}
    out = []
    for row in read_rows(repeat / "results.csv"):
        i = row["index"]
        old = [e for e in old_traces[i]["evaluations"] if e["reconstructed"]]
        new = [e for e in new_traces[i]["evaluations"] if e["reconstructed"]]
        common = 0
        while common < min(len(old), len(new)) and old[common]["z01"] == new[common]["z01"]:
            common += 1
        values = sorted(e["mfe"] for e in old)
        gaps = np.diff(values) / np.asarray(values[:-1])
        out.append(
            {
                "index": i,
                "mfe_true_relative_ppm": (row["mfe_true"] / old_rows[i]["mfe_true"] - 1.0) * 1e6,
                "common_points": common,
                "common_relative_ppm": [
                    (new[k]["mfe"] / old[k]["mfe"] - 1.0) * 1e6 for k in range(common)
                ],
                "smallest_gap_ppm": float(gaps.min() * 1e6) if gaps.size else None,
                "z01_est": [old_rows[i]["z01_est"], row["z01_est"]],
                "n_reconstructions": [old_rows[i]["n_reconstructions"], row["n_reconstructions"]],
                "abs_error_um": [old_rows[i]["abs_error_um"], row["abs_error_um"]],
            }
        )
    return out


def scan_objects(stage9: Path) -> dict:
    """RMS differences between the objects of all scan points of a probe (both numerics),
    from the npz of the scans."""
    by_sample = defaultdict(list)
    for record, p, n in scan_points(stage9):
        arrays = REPO / p["arrays"]
        if arrays.exists():
            by_sample[record["sample"]].append((p, n, arrays))
    out = {}
    for sample, items in sorted(by_sample.items()):
        items.sort(key=lambda item: (item[0]["z01"], item[1]))
        fields = {}
        for name in ("phase", "absorption"):
            stack = [np.load(arrays)[name].astype(np.float64) for _, _, arrays in items]
            fields[name] = [[float(np.sqrt(np.mean((a - b) ** 2))) for b in stack] for a in stack]
        out[sample] = {
            "points": [
                {"z01": p["z01"], "numerics": n, "mfe": p["mfe"], "arrays": p["arrays"]}
                for p, n, _ in items
            ],
            "phase_rms": fields["phase"],
            "absorption_rms": fields["absorption"],
        }
    return out


def object_difference(record: dict, a: dict, b: dict) -> dict:
    """RMS differences between the objects of two scan points of the first stage: in the field
    of view (also the largest difference there, and the largest absorption of each), and in
    the padded array outside it, where the object is mirrored from the field of view and the
    rest is the momentum of the last update."""
    paths = [REPO / p["arrays"] for p in (a, b)]
    if not all(path.exists() for path in paths):
        return {}
    arrays = [np.load(path) for path in paths]
    settings = record["settings"]
    grid = stage_grid(
        tuple(record["input_shape"]),
        settings["stages"][-1]["down_sampling"],
        settings["padding_factor"],
    )
    outside = np.ones(grid.shape, dtype=bool)
    outside[grid.fov] = False
    out = {}
    for name in ("phase", "absorption"):
        x, y = (arr[name].astype(np.float64) for arr in arrays)
        out[f"{name}_rms_fov"] = float(np.sqrt(np.mean((x - y) ** 2)))
        out[f"{name}_max_abs_fov"] = float(np.abs(x - y).max())
        if name == "absorption":
            out["absorption_largest_fov"] = [float(x.max()), float(y.max())]
        if f"{name}_padded" in arrays[0].files and f"{name}_padded" in arrays[1].files:
            x, y = (arr[f"{name}_padded"].astype(np.float64)[outside] for arr in arrays)
            out[f"{name}_rms_outside"] = float(np.sqrt(np.mean((x - y) ** 2)))
    return out


def first_stage(stage9: Path, losses: dict) -> dict:
    """The scans of the first stage of the plan (:data:`FIRST_STAGE`), per probe and numerics:
    its loss at the lookahead (``loss``, the ``mfe`` of the scan) and at the returned object
    against ``z01``, its branch (:data:`BRANCH`, against all ``losses`` of the probe from
    :func:`first_stage_losses`), its object against that of the point before
    (``to_previous``), the switches between neighbouring points and the largest change between
    neighbours on the same branch."""
    out = defaultdict(dict)
    for directory, before in ((stage9 / FIRST_STAGE, False), (stage9 / FIRST_STAGE / BEFORE, True)):
        for path in sorted(directory.glob("scan-*.json")):
            record = json.loads(path.read_text())
            by_numerics = defaultdict(list)
            for p in record["points"]:
                by_numerics[numerics(p["threads"], before)].append(p)
            for n, points in sorted(by_numerics.items()):
                points.sort(key=lambda p: p["z01"])
                upper = upper_branch(
                    {point_key(p["z01"]): (p["z01"], p["mfe"]) for p in points},
                    probe_losses(losses, record["sample"]),
                )
                on_upper = [upper[point_key(p["z01"])] for p in points]
                summary = [
                    {"z01": p["z01"], "loss": p["mfe"], "object_loss": p["object_loss"], "upper": u}
                    for p, u in zip(points, on_upper, strict=True)
                ]
                switches, same_branch = [], []
                for k, (a, b) in enumerate(zip(points, points[1:], strict=False)):
                    difference = object_difference(record, a, b)
                    summary[k + 1]["to_previous"] = difference
                    change = b["mfe"] / a["mfe"] - 1.0
                    if on_upper[k] == on_upper[k + 1]:
                        same_branch.append(abs(change))
                        continue
                    switches.append(
                        {
                            "z01": [a["z01"], b["z01"]],
                            "loss": [a["mfe"], b["mfe"]],
                            "relative_change": change,
                            "object_loss_relative_change": b["object_loss"] / a["object_loss"]
                            - 1.0,
                            **difference,
                        }
                    )
                out[record["sample"]][str(n)] = {
                    "n_points": len(points),
                    "n_upper": sum(on_upper),
                    "largest_change_same_branch": max(same_branch, default=None),
                    "z01_range": [points[0]["z01"], points[-1]["z01"]],
                    "points": summary,
                    "switches": switches,
                }
    return dict(out)


def first_stage_losses(stage9: Path, rows: list[dict]) -> dict:
    """Per ``(probe, numerics)`` the loss after the first stage at every ``z01`` it is known
    at, ``{point_key: (z01, loss)}``: the scans of the first stage, and the first stage of the
    full scans, which has the same bits."""
    out = defaultdict(dict)
    for directory, before in ((stage9 / FIRST_STAGE, False), (stage9 / FIRST_STAGE / BEFORE, True)):
        for path in sorted(directory.glob("scan-*.json")):
            record = json.loads(path.read_text())
            for p in record["points"]:
                key = record["sample"], numerics(p["threads"], before)
                out[key][point_key(p["z01"])] = (p["z01"], p["mfe"])
    for row in rows:
        if row["first_stage_loss"] is not None:
            key = row["sample"], row["numerics"]
            out[key].setdefault(point_key(row["z01"]), (row["z01"], row["first_stage_loss"]))
    return dict(out)


def probe_losses(losses: dict, sample: str) -> dict:
    """The losses of :func:`first_stage_losses` of ``sample`` with both numerics."""
    return {
        (n, key): value
        for (probe, n), known in losses.items()
        if probe == sample
        for key, value in known.items()
    }


def upper_branch(losses: dict, reference: dict) -> dict:
    """``{point_key: on the upper branch}`` for the losses of one probe and numerics from
    :func:`first_stage_losses` (:data:`BRANCH`), against the smallest of ``reference`` (those
    of the probe with both numerics, :func:`probe_losses`) within the window."""
    z = np.array([z01 for z01, _ in reference.values()]) * 1e3
    loss = np.array([value for _, value in reference.values()])
    out = {}
    for key, (z01, value) in losses.items():
        near = loss[np.abs(z - z01 * 1e3) <= BRANCH["window_mm"]]
        out[key] = bool(value > (1.0 + BRANCH["above"]) * near.min(initial=value))
    return out


def branch_slope(points: list[dict], k: int, step: int) -> float | None:
    """The slope of ``ln`` MFE in ppm per um of the line through ``points[k]`` and the next
    :data:`BRANCH_SLOPE_POINTS` in direction ``step`` on the same branch within
    :data:`BRANCH_SLOPE_UM` (least squares), ``None`` without one."""
    near = [points[k]]
    j = k + step
    while (
        0 <= j < len(points) and abs(points[j]["z01"] - points[k]["z01"]) * 1e6 <= BRANCH_SLOPE_UM
    ):
        if points[j]["upper"] == points[k]["upper"]:
            near.append(points[j])
            if len(near) > BRANCH_SLOPE_POINTS:
                break
        j += step
    if len(near) < 2:
        return None
    z = np.array([p["z01"] for p in near])
    return float(np.polyfit(z - z[0], np.log([p["mfe"] for p in near]), 1)[0])


def branches(losses: dict, rows: list[dict]) -> dict:
    """The full reconstructions (searches and scans) whose first stage is known
    (:func:`first_stage_losses`), per probe and numerics: on which branch the first stage lies
    (:func:`upper_branch`), and the pairs of neighbouring ones on different branches at most
    :data:`BRANCH_PAIR_UM` apart, with the MFE of the one on the upper branch over that of the
    one on the lower (``upper_over_lower_ppm``) and the same at one ``z01`` (``offset_ppm``):
    less the slope on the branches (``slope_ppm_per_um``, :func:`branch_slope`, the mean below
    and above the pair) times the spacing."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample"], row["numerics"]].append(row)
    out = defaultdict(dict)
    for (sample, n), group in sorted(groups.items()):
        known = losses.get((sample, n))
        if not known:
            continue
        upper = upper_branch(known, probe_losses(losses, sample))
        points = [
            {**p, "first_stage": known[point_key(p["z01"])][1], "upper": upper[point_key(p["z01"])]}
            for p in distinct(group)[0]
            if point_key(p["z01"]) in known
        ]
        pairs = []
        for k, (a, b) in enumerate(zip(points, points[1:], strict=False)):
            spacing = (b["z01"] - a["z01"]) * 1e6
            if a["upper"] == b["upper"] or spacing > BRANCH_PAIR_UM + 1e-6:
                continue
            high, low = (a, b) if a["upper"] else (b, a)
            slopes = [branch_slope(points, k, -1), branch_slope(points, k + 1, 1)]
            slopes = [s for s in slopes if s is not None]
            slope = float(np.mean(slopes)) if slopes else None
            offset = np.log(high["mfe"] / low["mfe"]) * 1e6
            pairs.append(
                {
                    "z01": [a["z01"], b["z01"]],
                    "upper": [a["upper"], b["upper"]],
                    "ppm": [a["ppm"], b["ppm"]],
                    "spacing_um": spacing,
                    "upper_over_lower_ppm": (high["mfe"] / low["mfe"] - 1.0) * 1e6,
                    "slope_ppm_per_um": slope,
                    "offset_ppm": None
                    if slope is None
                    else float(offset - slope * (high["z01"] - low["z01"]) * 1e6),
                    "losses_upper_over_lower_ppm": {
                        name: (high[name] / low[name] - 1.0) * 1e6
                        for name in PAIR_LOSSES
                        if high.get(name) is not None and low.get(name) is not None
                    },
                    "sources": [a["source"], b["source"]],
                }
            )
        keep = ("z01", "mfe", "ppm", "source", "first_stage", "upper")
        out[sample][str(n)] = {
            "n_first_stage": len(known),
            "n_first_stage_upper": sum(upper.values()),
            "n_upper": sum(p["upper"] for p in points),
            "n_lower": sum(not p["upper"] for p in points),
            "pairs": pairs,
            "points": [{k: p[k] for k in keep} for p in points],
        }
    return dict(out)


def object_losses(rows: list[dict]) -> dict:
    """The loss at the returned object of the full scans (:func:`object_loss`), per probe and
    numerics, relative to its smallest and screened like the MFE."""
    out = defaultdict(dict)
    groups = defaultdict(list)
    for row in rows:
        if row["object_loss"] is not None:
            groups[row["sample"], row["numerics"]].append(row)
    for (sample, n), group in sorted(groups.items()):
        points, _ = distinct(group)
        smallest = min(p["object_loss"] for p in points)
        points = [{**p, "ppm": (p["object_loss"] / smallest - 1.0) * 1e6} for p in points]
        best = min(points, key=lambda p: p["object_loss"])
        out[sample][str(n)] = {
            "n_points": len(points),
            "minimum": {"z01": best["z01"], "object_loss": best["object_loss"]},
            **screen(points),
        }
    return dict(out)


def last_stage(stage9: Path) -> dict:
    """The loss history of the last stage of the plan in the full scans, per probe and
    numerics: its smallest recorded value and where it lies, how far the MFE (the last value)
    lies above it, and the smallest value against ``z01`` screened like the MFE."""
    groups = defaultdict(list)
    for record, p, n in scan_points(stage9):
        loss = np.asarray(p["loss"])
        every = record["loss_every"]
        start = p["stage_ends"][-2] // every
        i = start + int(np.argmin(loss[start:]))
        groups[record["sample"], n].append(
            {
                "z01": p["z01"],
                "smallest": float(loss[i]),
                "iteration": (i + 1) * every,
                "mfe_above_ppm": (p["mfe"] / loss[i] - 1.0) * 1e6,
                "source": "scan",
            }
        )
    out = defaultdict(dict)
    for (sample, n), points in sorted(groups.items()):
        points.sort(key=lambda p: p["z01"])
        least = min(p["smallest"] for p in points)
        best = min(points, key=lambda p: p["smallest"])
        out[sample][str(n)] = {
            "n_points": len(points),
            "points": points,
            "minimum": {"z01": best["z01"], "smallest": best["smallest"]},
            **screen([{**p, "ppm": (p["smallest"] / least - 1.0) * 1e6} for p in points]),
        }
    return dict(out)


def cmd_steps(args) -> None:
    """Part 1 of Laborbuch 09: the MFE of the measured holograms as a function of ``z01``.

    ``mfe-points.csv``: every reconstruction (searches and scans) with its numerics, for the
    scans also the loss at the returned object, at the end of the first and second stage and
    the smallest of the last stage.
    ``steps.json``: per probe and numerics the smallest MFE, repeated points and the screening
    for steps; the same screening of the loss at the object (``object_loss``) and of the
    smallest loss of the last stage of the plan in the scans (``last_stage``); the scans of the
    first stage and their switches (``first_stage``); the branch of the first stage of every
    reconstruction and the MFE across the switches (``branches``); the reconstructions at the
    same ``z01``
    with the same numerics (``repeats``) and with both numerics (``same_z01``); the loss
    histories of the scans; the Stage 3 search on synthetic samples again (``repeat``).
    ``steps-objects.json``: the objects of the scans against each other (only where their npz
    are present).
    """
    args.out.mkdir(parents=True, exist_ok=True)
    rows = mfe_points(args.measured, args.stage9)
    write_csv(args.out / "mfe-points.csv", rows)
    losses = first_stage_losses(args.stage9, rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample"], row["numerics"]].append(row)
    probes = defaultdict(dict)
    minima = {}
    for (sample, n), group in sorted(groups.items()):
        points, worst = distinct(group)
        best = min(points, key=lambda p: p["mfe"])
        minima[sample, n] = best["mfe"]
        probes[sample][str(n)] = {
            "n_reconstructions": len(group),
            "n_points": len(points),
            "repeated_max_relative": worst,
            "minimum": {"z01": best["z01"], "mfe": best["mfe"], "source": best["source"]},
            "coverage": coverage(points, best["z01"]),
            **screen(points),
        }
    for sample in probes:
        # Both numerics together, each relative to its own minimum, one thread first where
        # both have the point: a solution one numerics reaches and the other does not shows
        # up as a jump.
        pooled = sorted((r for r in rows if r["sample"] == sample), key=lambda r: r["numerics"])
        points, _ = distinct(pooled)
        centre = probes[sample].get("1", probes[sample].get("3"))["minimum"]["z01"]
        probes[sample]["pooled"] = {
            "n_points": len(points),
            "coverage": coverage(points, centre),
            **screen(points),
        }
    record = {
        "what": "MFE of the measured holograms (binning 1, padding 4) against z01, per probe and "
        "numerics (1: one thread, since this stage for every thread count; 3: 3 threads "
        "before this stage), screened for steps",
        "step_screening": STEP,
        "loss_split": LOSS_SPLIT,
        "probes": probes,
        "object_loss": object_losses(rows),
        "last_stage": last_stage(args.stage9),
        "first_stage_branch": BRANCH,
        "first_stage": first_stage(args.stage9, losses),
        "branch_pair_um": BRANCH_PAIR_UM,
        "branches": branches(losses, rows),
        "repeats": repeats(rows),
        "same_z01": across_numerics(rows, losses),
        "scans": scan_pairs(args.stage9, minima),
    }
    repeat = args.baseline / "cpu-4x-test-repeat"
    if (repeat / "results.csv").exists():
        record["synthetic_repeat"] = synthetic_repeat(args.baseline / "cpu-4x-test", repeat)
    write_json(args.out / "steps.json", record)
    objects = scan_objects(args.stage9)
    if objects:
        write_json(args.out / "steps-objects.json", objects)
    for sample, by_numerics in probes.items():
        for n, summary in by_numerics.items():
            share = summary["coverage"][f"{WINDOWS_UM[0]:g}"]["share"]
            print(
                f"{sample} [{n}]: {summary['n_points']} points, {share:.0%} of "
                f"+-{WINDOWS_UM[0]:g} um covered, {len(summary['jumps'])} jumps, "
                f"{len(summary['reversals'])} reversals"
            )
    for sample, by_numerics in record["object_loss"].items():
        for n, summary in by_numerics.items():
            print(
                f"{sample} [{n}] at the object: {summary['n_points']} points, minimum at "
                f"{summary['minimum']['z01'] * 1e3:.4f} mm, {len(summary['jumps'])} jumps, "
                f"{len(summary['reversals'])} reversals"
            )
    for sample, by_numerics in record["last_stage"].items():
        for n, summary in by_numerics.items():
            above = [p["mfe_above_ppm"] for p in summary["points"]]
            print(
                f"{sample} [{n}] last stage: {summary['n_points']} points, smallest loss at "
                f"{summary['minimum']['z01'] * 1e3:.4f} mm, the MFE {min(above):.0f}-"
                f"{max(above):.0f} ppm above it, {len(summary['jumps'])} jumps, "
                f"{len(summary['reversals'])} reversals"
            )
    for sample, by_numerics in record["first_stage"].items():
        for n, summary in by_numerics.items():
            where = ", ".join(
                f"{s['z01'][0] * 1e3:.3f}-{s['z01'][1] * 1e3:.3f} mm ({s['relative_change']:+.1%})"
                for s in summary["switches"]
            )
            largest = summary["largest_change_same_branch"]
            print(
                f"{sample} [{n}] first stage: {summary['n_points']} points, "
                f"{summary['n_upper']} on the upper branch, neighbours on one branch at most "
                f"{'-' if largest is None else f'{largest:.2%}'} apart, "
                f"{len(summary['switches'])} switches{': ' + where if where else ''}"
            )
    for sample, by_numerics in record["branches"].items():
        for n, summary in by_numerics.items():
            where = ", ".join(
                f"{p['z01'][0] * 1e3:.3f}/{p['z01'][1] * 1e3:.3f} mm "
                f"{p['upper_over_lower_ppm']:+.0f} ppm"
                + ("" if p["offset_ppm"] is None else f" ({p['offset_ppm']:+.0f} at one z01)")
                for p in summary["pairs"]
            )
            print(
                f"{sample} [{n}] branches: first stage upper at {summary['n_first_stage_upper']} "
                f"of {summary['n_first_stage']} points; full reconstructions "
                f"{summary['n_upper']} upper, {summary['n_lower']} lower; "
                f"{len(summary['pairs'])} pairs across a switch"
                f"{', upper over lower: ' + where if where else ''}"
            )


def measured_comparison(measured: Path, minima: dict) -> tuple[dict, list[dict]]:
    """Per probe the searches of :data:`MEASURED_METHODS` against the references of [2] and the
    full search of Stage 3; the time at the rate of this stage's searches with 3 threads."""
    out, rows = {}, []
    for sample in MEASURED_SAMPLES:
        records = {}
        for method, (folder, _) in MEASURED_METHODS.items():
            path = measured / folder / f"{sample}.json"
            if path.exists():
                records[method] = (path, json.loads(path.read_text()))
        if "nm-cold" not in records:
            continue
        cold = records["nm-cold"][1]["result"]
        refs = records["nm-cold"][1]["paper2"]
        rates = [
            r["seconds_per_reconstruction"]
            for m, (_, r) in records.items()
            if MEASURED_METHODS[m][0] in MEASURED_SEARCHES and r["threads"] == 3
        ]
        rate = float(np.median(rates)) if rates else cold["seconds"] / cold["n_reconstructions"]
        entries = {}
        for method, (path, r) in records.items():
            res = r["result"]
            n = numerics(r["threads"], MEASURED_METHODS[method][0] in EARLIER)
            lower, upper = res["bounds"]
            e = {
                "record": repo_relative(path),
                "sha256": file_sha256(path),
                "git_commit": r["git_commit"],
                "threads": r["threads"],
                "numerics": n,
                "z01": res["z01"],
                "minus_paper2_nm_um": (res["z01"] - refs["z01_nm"]) * 1e6,
                "minus_paper2_visual_um": None
                if refs.get("z01_visual") is None
                else (res["z01"] - refs["z01_visual"]) * 1e6,
                "minus_nm_cold_um": (res["z01"] - cold["z01"]) * 1e6,
                "same_as_nm_cold": bool(abs(res["z01"] - cold["z01"]) * 1e6 <= XATOL_UM),
                "mfe": res["mfe"],
                "mfe_ppm": (res["mfe"] / minima[sample, n] - 1.0) * 1e6,
                "n_reconstructions": res["n_reconstructions"],
                "n_evaluations": res["n_evaluations"],
                "converged": res["converged"],
                "seconds": res["seconds"],
                "seconds_per_reconstruction": r["seconds_per_reconstruction"],
                "minutes_at_3_threads": res["n_reconstructions"] * rate / 60.0,
                "bounds": [lower, upper],
                "width_mm": (upper - lower) * 1e3,
                "paper2_nm_inside": bool(lower <= refs["z01_nm"] <= upper),
                "nm_cold_inside": bool(lower <= cold["z01"] <= upper),
                "evaluated": [p["z01"] for p in res["evaluations"]],
            }
            entries[method] = e
            rows.append(
                {
                    "sample": sample,
                    "method": method,
                    **{k: v for k, v in e.items() if k not in ("record", "sha256", "evaluated")},
                    "bounds": "-".join(f"{b * 1e3:.4f}" for b in e["bounds"]),
                }
            )
        savings = {}
        for ensemble in ("cnn", "npe"):
            for family, colds in (
                ("nm", ("nm-cold", "nm-cold-repeat")),
                ("brent", ("brent-cold",)),
            ):
                box = f"{family}-box95-{ensemble}"
                for c in colds:
                    if box in entries and c in entries:
                        cn, bn = entries[c]["n_reconstructions"], entries[box]["n_reconstructions"]
                        savings[f"{c} - {box}"] = {"saved": cn - bn, "relative": 1.0 - bn / cn}
        out[sample] = {
            "paper2": {"z01_nm": refs["z01_nm"], "z01_visual": refs.get("z01_visual")},
            "seconds_per_reconstruction_3_threads": rate,
            "searches": entries,
            "savings": savings,
        }
    totals = {}
    for method in MEASURED_METHODS:
        found = [s["searches"][method] for s in out.values() if method in s["searches"]]
        if found:
            totals[method] = {
                "samples": [k for k, s in out.items() if method in s["searches"]],
                "n_reconstructions": sum(e["n_reconstructions"] for e in found),
                "minutes_at_3_threads": sum(e["minutes_at_3_threads"] for e in found),
                "within_100um_of_paper2_nm": sum(
                    abs(e["minus_paper2_nm_um"]) <= 100 for e in found
                ),
                "same_as_nm_cold": sum(e["same_as_nm_cold"] for e in found),
                "max_abs_minus_nm_cold_um": max(abs(e["minus_nm_cold_um"]) for e in found),
            }
    return {"samples": out, "totals": totals}, rows


def lane_rates(path: Path) -> dict:
    """Seconds per reconstruction of the synthetic searches of this stage, per thread count."""
    if not path.exists():
        return {}
    by_threads = defaultdict(lambda: [0.0, 0])
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            entry = by_threads[int(row["threads"])]
            entry[0] += float(row["seconds"])
            entry[1] += int(row["n_reconstructions"])
    return {str(t): s / n for t, (s, n) in sorted(by_threads.items()) if n}


def ci(values, statistic=np.mean) -> list[float]:
    return list(bootstrap_ci(values, statistic, resamples=RESAMPLES))


def search_summary(rows: list[dict], rate: float | None) -> dict:
    err = np.array([r["abs_error_um"] for r in rows])
    recon = np.array([r["n_reconstructions"] for r in rows], dtype=float)
    out = {
        "n_samples": len(rows),
        "mae_um": float(err.mean()),
        "mae_um_ci": ci(err),
        "median_abs_um": float(np.median(err)),
        "p90_abs_um": float(np.percentile(err, 90)),
        "max_abs_um": float(err.max()),
        "within_um": {str(t): float(np.mean(err <= t)) for t in WITHIN_UM},
        "within_100um_ci": ci((err <= 100).astype(float)),
        "reconstructions": float(recon.mean()),
        "reconstructions_ci": ci(recon),
        "reconstructions_range": [int(recon.min()), int(recon.max())],
        "converged": float(np.mean([bool(r["converged"]) for r in rows])),
    }
    if rate is not None:
        out["minutes_at_3_threads"] = float(recon.mean() * rate / 60.0)
    return out


def paired_savings(runs: dict, nm: tuple[str, str], brent: tuple[str, str], idx: list) -> dict:
    """The savings of an interval with Nelder-Mead and with Brent on the same samples: in
    reconstructions per autofocus and relative to the cold search, and their difference."""
    table = np.array(
        [[runs[m][i]["n_reconstructions"] for m in (*nm, *brent)] for i in idx], dtype=float
    )

    def relative(d, cold, box):
        return 1.0 - d[:, box].mean() / d[:, cold].mean()

    out = {"n_samples": len(idx)}
    for name, (cold, box) in (("nm", (0, 1)), ("brent", (2, 3))):
        saved = table[:, cold] - table[:, box]
        out[name] = {
            "cold": float(table[:, cold].mean()),
            "box": float(table[:, box].mean()),
            "saved": float(saved.mean()),
            "saved_ci": ci(saved),
            "relative": relative(table, cold, box),
            "relative_ci": ci(table, lambda d, c=cold, b=box: relative(d, c, b)),
            "fewer": int(np.sum(saved > 0)),
            "same": int(np.sum(saved == 0)),
            "more": int(np.sum(saved < 0)),
        }
    gap = (table[:, 2] - table[:, 3]) - (table[:, 0] - table[:, 1])
    out["brent_minus_nm"] = {
        "saved": float(gap.mean()),
        "saved_ci": ci(gap),
        "relative": relative(table, 2, 3) - relative(table, 0, 1),
        "relative_ci": ci(table, lambda d: relative(d, 2, 3) - relative(d, 0, 1)),
    }
    return out


def width_bins(runs, nm, brent, idx, widths: dict, bins: int = 3) -> list[dict]:
    """:func:`paired_savings` in ``bins`` groups of equal size by the width of the interval."""
    order = sorted(idx, key=lambda i: widths[i])
    out = []
    for group in np.array_split(np.array(order), bins):
        group = [int(i) for i in group]
        s = paired_savings(runs, nm, brent, group)
        out.append(
            {
                "width_um": [widths[group[0]], widths[group[-1]]],
                "n_samples": len(group),
                **{
                    k: {q: s[k][q] for q in ("cold", "box", "saved", "relative")}
                    for k in ("nm", "brent")
                },
            }
        )
    return out


def synthetic_comparison(baseline: Path, lanes: Path) -> tuple[dict, list[dict]]:
    """The searches of :data:`SYNTHETIC_METHODS` on the samples of :data:`SYNTHETIC`: errors
    against the truth, reconstructions, the time at the rate of 3 threads, and what the 95 %
    intervals save with Nelder-Mead and with Brent."""
    from holofocus.baseline.runner import Box

    runs, provenance = {}, {}
    for method, (folder, what) in SYNTHETIC_METHODS.items():
        rows = [r for r in read_rows(baseline / folder / "results.csv") if r["index"] in SYNTHETIC]
        if rows:
            runs[method] = {r["index"]: r for r in rows}
            info = yaml.safe_load((baseline / folder / "run.yaml").read_text())
            provenance[method] = {
                "folder": repo_relative(baseline / folder),
                "what": what,
                "search": info["config"]["search"],
                "box": info.get("box"),
                "threads": sorted(
                    {e["threads"] for e in info.get("merged_from", [])} or {info.get("threads")}
                ),
            }
    rates = lane_rates(lanes)
    rate = rates.get("3")
    boxes = {k: Box.from_csv(path) for k, path in INTERVALS.items() if Path(path).exists()}
    confidence = BaselineConfig.from_yaml(CONFIGS / "paper2-cpu.yaml").search.confidence
    truth = {i: r for i, r in runs.get("nm-cold", {}).items()}
    per_sample, widths, inside = [], {k: {} for k in boxes}, {k: {} for k in boxes}
    for i in SYNTHETIC:
        if i not in truth:
            continue
        r = truth[i]
        row = {"index": i, "z01_true": r["z01_true"], "z01_guess": r["z01_guess"]}
        for key, box in boxes.items():
            lower, upper = box.bounds(i, r["z01_guess"] - confidence, r["z01_guess"] + confidence)
            widths[key][i] = (upper - lower) * 1e6
            inside[key][i] = bool(lower <= r["z01_true"] <= upper)
            row.update(
                {
                    f"{key}_lower": lower,
                    f"{key}_upper": upper,
                    f"{key}_width_um": widths[key][i],
                    f"{key}_truth_inside": int(inside[key][i]),
                }
            )
        for method, by_index in runs.items():
            if i in by_index:
                m = by_index[i]
                row[f"{method}_z01_est"] = m["z01_est"]
                row[f"{method}_error_um"] = m["error_um"]
                row[f"{method}_n_reconstructions"] = m["n_reconstructions"]
                row[f"{method}_mfe_vs_nm_cold_ppm"] = (m["mfe_est"] / r["mfe_est"] - 1.0) * 1e6
            else:
                row[f"{method}_z01_est"] = row[f"{method}_error_um"] = None
                row[f"{method}_n_reconstructions"] = row[f"{method}_mfe_vs_nm_cold_ppm"] = None
        per_sample.append(row)
    summaries = {m: search_summary(list(by_index.values()), rate) for m, by_index in runs.items()}
    for m, by_index in runs.items():
        if m != "nm-cold" and set(by_index) & set(truth):
            ppm = np.array(
                [
                    (by_index[i]["mfe_est"] / truth[i]["mfe_est"] - 1.0) * 1e6
                    for i in by_index
                    if i in truth
                ]
            )
            summaries[m]["mfe_vs_nm_cold_ppm"] = {
                "median": float(np.median(ppm)),
                "higher": int(np.sum(ppm > MFE_APART_PPM)),
                "lower": int(np.sum(ppm < -MFE_APART_PPM)),
            }
    for m, by_index in runs.items():
        if "box" in m:
            key = m.rsplit("-", 1)[1]
            misses = [i for i in by_index if not inside[key][i]]
            summaries[m]["truth_outside"] = len(misses)
            summaries[m]["mae_um_truth_inside"] = float(
                np.mean([by_index[i]["abs_error_um"] for i in by_index if inside[key][i]])
            )
    comparisons = {}
    for key in boxes:
        nm, brent = ("nm-cold", f"nm-box95-{key}"), ("brent-cold", f"brent-box95-{key}")
        if all(m in runs for m in (*nm, *brent)):
            idx = sorted(set.intersection(*(set(runs[m]) for m in (*nm, *brent))))
            comparisons[key] = {
                "searches": [*nm, *brent],
                "width_um": {
                    "median": float(np.median([widths[key][i] for i in idx])),
                    "range": [min(widths[key][i] for i in idx), max(widths[key][i] for i in idx)],
                },
                "truth_inside": int(sum(inside[key][i] for i in idx)),
                **paired_savings(runs, nm, brent, idx),
                "by_width": width_bins(runs, nm, brent, idx, widths[key]),
            }
    accuracy = {}
    for first, second in PAIRED_ERRORS:
        if first not in runs or second not in runs:
            continue
        idx = sorted(set(runs[first]) & set(runs[second]))
        a = np.array([runs[second][i]["abs_error_um"] for i in idx])
        b = np.array([runs[first][i]["abs_error_um"] for i in idx])
        apart = np.abs([runs[first][i]["z01_est"] - runs[second][i]["z01_est"] for i in idx]) * 1e6
        accuracy[f"{first} vs {second}"] = {
            "n_samples": len(idx),
            "abs_error_difference_um": {
                "mean": float((b - a).mean()),
                "mean_ci": ci(b - a),
                "median": float(np.median(b - a)),
            },
            "within_100um": {
                "both": int(np.sum((a <= 100) & (b <= 100))),
                f"only_{second}": int(np.sum((a <= 100) & (b > 100))),
                f"only_{first}": int(np.sum((a > 100) & (b <= 100))),
                "neither": int(np.sum((a > 100) & (b > 100))),
            },
            "results_apart_um": {
                "median": float(np.median(apart)),
                "max": float(apart.max()),
                "same": int(np.sum(apart == 0.0)),
                "within_xatol": int(np.sum(apart <= XATOL_UM)),
            },
        }
    record = {
        "searches": provenance,
        "seconds_per_reconstruction": rates,
        "summaries": summaries,
        "savings": comparisons,
        "accuracy": accuracy,
    }
    return record, per_sample


def cmd_compare(args) -> None:
    """Part 2 of Laborbuch 09: Brent's bounded method next to Nelder-Mead, cold and in the 95 %
    intervals, on the measured holograms (``measured.csv``) and on the synthetic samples
    (``synthetic.csv``); everything with the bootstrap intervals in ``compare.json``."""
    args.out.mkdir(parents=True, exist_ok=True)
    minima = {}
    for row in mfe_points(args.measured, args.stage9):
        key = row["sample"], row["numerics"]
        minima[key] = min(minima.get(key, np.inf), row["mfe"])
    measured, measured_rows = measured_comparison(args.measured, minima)
    synthetic, synthetic_rows = synthetic_comparison(args.baseline, args.lanes)
    record = {
        "what": "Brent's bounded method next to Nelder-Mead, in Omega_z and in the 95 % "
        "intervals of learned models, on the measured holograms and the synthetic samples",
        "xatol_um": XATOL_UM,
        "resamples": RESAMPLES,
        "measured_methods": {m: {"folder": f, "what": w} for m, (f, w) in MEASURED_METHODS.items()},
        "measured": measured,
        "synthetic": synthetic,
    }
    write_json(args.out / "compare.json", record)
    if measured_rows:
        write_csv(args.out / "measured.csv", measured_rows)
    if synthetic_rows:
        write_csv(args.out / "synthetic.csv", synthetic_rows)
    for method, total in measured["totals"].items():
        print(
            f"measured {method}: {total['n_reconstructions']} reconstructions on "
            f"{len(total['samples'])} probes, {total['minutes_at_3_threads']:.0f} min at "
            f"3 threads, {total['same_as_nm_cold']} like the full search"
        )
    for method, s in synthetic["summaries"].items():
        print(
            f"synthetic {method}: {s['n_samples']} samples, MAE {s['mae_um']:.0f} um, "
            f"within 100 um {s['within_um']['100']:.2f}, {s['reconstructions']:.2f} reconstructions"
        )
    for key, s in synthetic["savings"].items():
        print(
            f"interval {key}: NM saves {s['nm']['saved']:.2f} ({s['nm']['relative']:.1%}), "
            f"Brent {s['brent']['saved']:.2f} ({s['brent']['relative']:.1%}) reconstructions"
        )


def object_loss(
    hologram: torch.Tensor, fresnel_number: float, settings, fov: torch.Tensor
) -> float:
    """The fit of the returned object itself: the loss of one more iteration of the last stage,
    started from the object ``fov`` (its field of view) without momentum. The MFE is the loss
    at the lookahead ``obj - momentum * velocity`` of the last iteration instead."""
    last = dataclasses.replace(settings.stages[-1], iterations=1)
    grid = stage_grid(tuple(hologram.shape), last.down_sampling, settings.padding_factor)
    init = torch.zeros(grid.shape, dtype=fov.dtype)
    init[grid.fov] = fov
    one = dataclasses.replace(settings, stages=(last,))
    return float(asrm_reconstruct(hologram, fresnel_number, one, init=init).loss[0])


def cmd_scan(args) -> None:
    sample = MEASURED_SAMPLES[args.sample]
    settings, nominal = measured_settings(sample, args.binning, args.padding)
    if args.stages:
        settings = dataclasses.replace(settings, stages=settings.stages[: args.stages])
    hologram = load_measured(args.holograms, sample, args.binning)
    geometry = sample.geometry(args.binning)
    grids = [
        list(stage_grid(tuple(hologram.shape), s.down_sampling, settings.padding_factor).shape)
        for s in settings.stages
    ]
    header = {
        "sample": args.sample,
        "binning": args.binning,
        "padding_factor": nominal,
        "padding_factor_input": settings.padding_factor,
        "input_shape": list(hologram.shape),
        "stage_arrays": grids,
        "settings": plain(dataclasses.asdict(settings)),
        "geometry_guess": geometry.as_dict(),
        "loss_every": LOSS_EVERY,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"scan-{args.sample}.json"
    record = json.loads(target.read_text()) if target.exists() else {**header, "points": []}
    if any(record.get(k) != v for k, v in header.items()):
        raise SystemExit(f"{target} holds a scan with other settings")
    objects = args.out / f"scan-{args.sample}"
    objects.mkdir(exist_ok=True)
    for point in record["points"]:
        if "object_loss" not in point and Path(point["arrays"]).exists():
            arrays = np.load(point["arrays"])
            fov = torch.complex(
                torch.from_numpy(arrays["phase"]), torch.from_numpy(arrays["absorption"])
            )
            fresnel_number = geometry.with_z01(point["z01"]).fresnel_number
            point["object_loss"] = object_loss(hologram, fresnel_number, settings, fov)
            write_json(target, record)
    done = {point_key(p["z01"]) for p in record["points"]}
    for z01 in args.z01 or [z / 1e3 for z in args.z01_mm]:
        z01_mm = z01 * 1e3
        if point_key(z01) in done:
            print(f"{args.sample}: {z01_mm:.6f} mm scanned already, skipped")
            continue
        fresnel_number = geometry.with_z01(z01).fresnel_number
        start = time.perf_counter()
        result = asrm_reconstruct(hologram, fresnel_number, settings)
        seconds = time.perf_counter() - start
        loss = np.asarray(result.loss, dtype=np.float64)
        path = objects / f"{z01_mm:.6f}.npz"
        padded = {}
        if args.stages:
            padded = {
                "phase_padded": result.padded.real.numpy().astype(np.float32),
                "absorption_padded": result.padded.imag.numpy().astype(np.float32),
            }
        np.savez(
            path,
            phase=result.obj.real.numpy().astype(np.float32),
            absorption=result.obj.imag.numpy().astype(np.float32),
            loss=loss,
            **padded,
        )
        point = {
            "z01": z01,
            "mfe": result.final_loss,
            "object_loss": object_loss(hologram, fresnel_number, settings, result.obj),
            "seconds": seconds,
            "stage_ends": list(result.stage_ends),
            "stage_losses": [float(loss[e - 1]) for e in result.stage_ends],
            "loss": [float(x) for x in loss[LOSS_EVERY - 1 :: LOSS_EVERY]],
            "object": object_statistics(result.obj),
            "arrays": repo_relative(path),
            **environment(),
        }
        record["points"].append(point)
        record["points"].sort(key=lambda p: p["z01"])
        done.add(point_key(z01))
        write_json(target, record)
        print(
            f"{args.sample}: {z01_mm:.6f} mm  MFE {result.final_loss:.10e}  "
            f"at the object {point['object_loss']:.10e}  ({seconds:.0f} s)",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--threads", type=int, help="torch CPU threads")
    sub = parser.add_subparsers(dest="command", required=True)
    sources = sub.add_parser("sources", help="intervals of the CNN on the synthetic test split")
    sources.add_argument("--cnn", type=Path, default=CNN_PREDICTIONS)
    sources.add_argument("--out", type=Path, default=STAGE9)
    sources.set_defaults(func=cmd_sources)
    queue = sub.add_parser("queue", help="the searches on the synthetic samples, as one lane")
    queue.add_argument("--lane", required=True, help="name of the lane, e.g. a or b")
    queue.add_argument("--data", type=Path, default=TEST_SPLIT)
    queue.add_argument("--claims", type=Path, default=RUNS / "claims")
    queue.set_defaults(func=cmd_queue)
    merge = sub.add_parser("merge", help="join the lanes of each search on the synthetic samples")
    merge.add_argument("--out", type=Path, default=STAGE9, help="where lanes.csv goes")
    merge.set_defaults(func=cmd_merge)
    scan = sub.add_parser("scan", help="reconstructions of a measured hologram at given z01")
    scan.add_argument("--holograms", type=Path, required=True)
    scan.add_argument("--sample", choices=list(MEASURED_SAMPLES), required=True)
    points = scan.add_mutually_exclusive_group(required=True)
    points.add_argument("--z01-mm", type=float, nargs="+")
    points.add_argument(
        "--z01", type=float, nargs="+", help="in m, e.g. exactly the z01 of a recorded probe"
    )
    scan.add_argument("--binning", type=int, default=1)
    scan.add_argument("--padding", type=float, default=4.0)
    scan.add_argument(
        "--stages",
        type=int,
        help="only the first STAGES stages of the plan, e.g. 1 with --out "
        "results/stage9/first-stage; 'mfe' is then the loss at the end of the last of them",
    )
    scan.add_argument("--out", type=Path, default=STAGE9)
    scan.set_defaults(func=cmd_scan)
    steps = sub.add_parser("steps", help="the MFE of the measured holograms against z01 (part 1)")
    steps.add_argument("--measured", type=Path, default=MEASURED)
    steps.add_argument("--stage9", type=Path, default=STAGE9, help="where the scans are")
    steps.add_argument("--baseline", type=Path, default=BASELINE)
    steps.add_argument("--out", type=Path, default=STAGE9)
    steps.set_defaults(func=cmd_steps)
    compare = sub.add_parser("compare", help="Brent next to Nelder-Mead (part 2)")
    compare.add_argument("--measured", type=Path, default=MEASURED)
    compare.add_argument("--stage9", type=Path, default=STAGE9, help="where the scans are")
    compare.add_argument("--baseline", type=Path, default=BASELINE)
    compare.add_argument("--lanes", type=Path, default=STAGE9 / "lanes.csv")
    compare.add_argument("--out", type=Path, default=STAGE9)
    compare.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
