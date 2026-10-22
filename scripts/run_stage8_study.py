"""Stage 8: misspecification, the two ensembles and the warm start on the measured holograms.

Laborbuch 08. Everything works on the leave-one-out sets of Stage 6 and the runs of Stage 7 on
them: the CNNs ``results/learned/loo-<sample>`` (seed 0) and
``results/objectmodel/seeds/seed-<k>/loo-<sample>``, the NPEs ``results/npe-loo/matched-<sample>``
and ``matched-<sample>-seed<k>``.

* ``sources``: the two ensembles on the measured holograms, as estimates and intervals for
  the warm start (``results/measured/{cnn,npe}-loo-ensemble/<sample>.json``). The CNN ensemble
  is the equal-weight mixture of the Gaussians of the three CNN seeds
  (:class:`holofocus.learned.posterior.GaussianMixture`, calibrated ``sigma``), summarized
  like the NPE ensemble (median, central intervals, HPD widths, modes, the 16 binning phases);
  the NPE ensemble is copied from ``results/npe-loo/measured/<sample>.json``. Reads only
  committed records.
* ``misspec``: how far each measured hologram lies from its leave-one-out training set, in the
  network input (the standardized log power spectrum, the input of both networks) and in the
  embedding of each NPE seed (:mod:`holofocus.learned.misspecification`): the mean distance to
  the 10 nearest training samples (1 and 50 as checks) and the Mahalanobis distance in the
  principal components of the training samples (input: the first 20, with the RMS of the
  residual; embedding: those holding 99.9 % of the variance). Each is a percentile among the
  test split, for the hologram binned 4x and for its 16 binning phases. On the test split: how
  the distances go with what fails, the absolute error and the 95 % width of the two ensembles
  and the spread of their seeds (Spearman with bootstrap intervals, nearest against farthest
  quarter); whether a neighbourhood predicts them, the Spearman of each test sample's failure
  with its mean over the 25 nearest other test samples (``neighbourhood``); the 25 test
  samples nearest to the hologram in each space, how far the ensembles move from the guess
  there (shrinkage); and the same failures on the hologram, with the padding-4 search of
  Stage 3 as the truth, as percentiles among the test split and among its nearest test samples
  (``results/misspecification/<sample>.json``, ``test-<sample>.csv``, ``summary.csv``). With
  ``--control results/npe``, the same distances to the sets of the Stage 5 simulator
  (``data/matched-<sample>``), in the input and in the embedding of the NPE trained on them,
  as the positive control (``control`` in the records, ``control-*`` rows in the summary).
* ``ensembles``: the CNN ensemble (per test sample the mixture of the Gaussians of the seeds)
  against the NPE ensemble on the leave-one-out test splits, with the seeds and the guess:
  the metrics of Stage 7, paired bootstrap differences, per probe and pooled over the four;
  and the two on the measured holograms from ``sources`` (``results/ensembles/``).
* ``warmstart``: the padding-4 searches on the measured holograms started from an ensemble's
  estimate (``results/measured/warm-{cnn,npe}-loo-ensemble-b1-p4``) or bounded by its 95 %
  interval (``box95-...``), against the full search of Stage 3 (``b1-p4``) and the warm start
  from the CNN of Stage 5 (``warm-b1-p4``): position, MFE, reconstructions and time
  (``results/measured/warmstart.{json,csv}``).

Usage (Laborbuch 08)::

    python scripts/run_stage8_study.py sources
    python scripts/run_stage8_study.py --threads 1 misspec --holograms DIR --control results/npe
    python scripts/run_stage8_study.py ensembles
    python scripts/run_stage3_experiments.py --threads 3 measured --holograms DIR --binning 1 \\
        --padding 4 --revisit-tolerance 1e-9 --warm-start results/measured/npe-loo-ensemble \\
        --out results/measured/warm-npe-loo-ensemble-b1-p4
    python scripts/run_stage3_experiments.py --threads 3 measured --holograms DIR --binning 1 \\
        --padding 4 --revisit-tolerance 1e-9 --box results/measured/npe-loo-ensemble \\
        --out results/measured/box95-npe-loo-ensemble-b1-p4
    python scripts/run_stage8_study.py warmstart

(the searches also with ``cnn-loo-ensemble``). ``DIR`` is ``Examples/data/holograms`` of the
Zenodo record 10.5281/zenodo.8349365; without ``--holograms``, ``misspec`` computes only the
test split. Run it from the root of the repository, so the paths in the records stay relative
to it.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from holofocus.baseline.runner import dataset_identity, file_sha256
from holofocus.learned import read_predictions
from holofocus.learned.data import load_inputs
from holofocus.learned.metrics import bootstrap_ci
from holofocus.learned.misspecification import PrincipalComponents, knn_distance
from holofocus.learned.npe import NPEAutofocus
from holofocus.learned.posterior import GaussianMixture, mixture_summaries
from holofocus.learned.sim2real import binning_phases, percentile_rank
from holofocus.measured import MEASURED_SAMPLES, load_measured
from run_stage7_loo_study import (
    ensemble_name,
    guess_table,
    method_name,
    modes,
    prior_metrics,
    read_run,
    run_name,
    summary_row,
)
from run_stage7_study import (
    BINNING,
    EXAMPLE_POINTS,
    LEVELS,
    cnn_table,
    environment,
    model_sha256,
    npe_table,
    paired,
    paired_mae,
    per_sample_rows,
    repo_relative,
    spearman,
    table_metrics,
    write_csv,
    write_json,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
DATA = REPO / "data"
MEASURED = RESULTS / "measured"
SEEDS = (0, 1, 2)
GRID_POINTS = 2001
"""Points of the grid over ``Omega_z`` for the modes and the plotted density of the CNN
ensemble, as the grid of the NPE (``configs/learned/npe-nsf.yaml``)."""
NEIGHBOURS = (10, 1, 50)
"""``k`` of the distance to the nearest training samples; the first is the main one."""
INPUT_COMPONENTS = 20
EMBEDDING_VARIANCE = 0.999
NEAREST = 25
"""Test samples nearest to a measured hologram for the local look (10 % of a test split)."""
CONTROL = "matched"
"""Prefix of the sets of the Stage 5 simulator (``data/matched-<sample>``) and of the NPE
trained on each (``results/npe/matched-<sample>``, Laborbuch 07, 5.7), the positive control of
``misspec``: with that simulator the learned estimates missed the measured holograms by
millimetres."""
ENSEMBLES = ("cnn-ensemble", "npe-ensemble")
PAIRS = (
    ("npe-ensemble", "cnn-ensemble"),
    ("cnn-ensemble", "cnn"),
    ("npe-ensemble", "npe"),
    ("npe", "cnn"),
)
SEARCHES = (
    ("cold", "b1-p4", "Stage 3: Omega_z = guess +- 5 mm, simplex at its ends"),
    ("warm-stage5", "warm-b1-p4", "Stage 5: from the matched CNN of Stage 5, simplex +- 0.5 mm"),
    ("warm-cnn", "warm-cnn-loo-ensemble-b1-p4", "from the CNN ensemble, simplex +- 0.5 mm"),
    ("warm-npe", "warm-npe-loo-ensemble-b1-p4", "from the NPE ensemble, simplex +- 0.5 mm"),
    ("box95-cnn", "box95-cnn-loo-ensemble-b1-p4", "in the 95 % interval of the CNN ensemble"),
    ("box95-npe", "box95-npe-loo-ensemble-b1-p4", "in the 95 % interval of the NPE ensemble"),
)
XATOL_UM = 100.0
"""Tolerance of the search in ``z01`` (``find_focus``): results closer than this are the same."""
SAME_POINT = 1e-9


def cnn_estimates(args, name: str) -> list[Path]:
    """The records of the leave-one-out CNN seeds on the hologram, seed 0 first
    (``run_stage6_study.py measured``)."""
    paths = [
        args.cnn_measured / f"{name}.json" if k == 0
        else args.cnn_seeds / f"seed-{k}" / "measured" / f"{name}.json"
        for k in args.seeds
    ]  # fmt: skip
    return [p for p in paths if p.exists()]


def paper_references(name: str, measured: Path) -> dict[str, float]:
    """``z01`` of [2] Table 1, [1] and our padding-4 search (Stage 3)."""
    sample = MEASURED_SAMPLES[name]
    refs = {
        "paper2_nm": sample.z01_nm,
        "paper2_visual": sample.z01_visual,
        "paper1_example": sample.z01_example,
    }
    path = measured / "b1-p4" / f"{name}.json"
    if path.exists():
        refs["baseline_b1_p4"] = float(json.loads(path.read_text())["result"]["z01"])
    return refs


def mixture_record(name: str, estimates: list[Path], refs: dict[str, float]) -> dict:
    """The CNN ensemble on a measured hologram, with the fields of an NPE run in
    ``results/npe-loo/measured/<sample>.json``."""
    sample = MEASURED_SAMPLES[name]
    guess = sample.z01_guess
    members = [json.loads(p.read_text()) for p in estimates]
    mix = GaussianMixture(
        np.array([[m["dz01_est"] for m in members]]), np.array([[m["sigma"] for m in members]])
    )
    est = {k: float(v[0]) for k, v in mixture_summaries(mix).items()}
    z01 = guess + est["dz01"]
    intervals = {
        key: {
            "lower": guess + est[f"lower_{key}"],
            "upper": guess + est[f"upper_{key}"],
            "width_um": (est[f"upper_{key}"] - est[f"lower_{key}"]) * 1e6,
            "hpd_width_um": est[f"hpd_width_{key}"] * 1e6,
        }
        for key in LEVELS
    }

    def against(value: float) -> dict:
        t = np.array([value - guess])
        return {
            "z01": value,
            "distance_um": (z01 - value) * 1e6,
            "pit": float(mix.pit(t)[0]),
            "hpd_level": float(mix.hpd_level(t)[0]),
            "inside": {k: bool(i["lower"] <= value <= i["upper"]) for k, i in intervals.items()},
        }

    half = sample.confidence
    on_omega = mix.on_grid(np.linspace(-half, half, GRID_POINTS))
    phase_mix = GaussianMixture(
        np.array([m["binning_phases"]["z01_est"] for m in members]).T - guess,
        np.array([m["binning_phases"]["sigma"] for m in members]).T,
    )
    phase_z01 = guess + phase_mix.median()
    step = max(1, (GRID_POINTS - 1) // (EXAMPLE_POINTS - 1))
    seeds = [m["z01_est"] for m in members]
    return {
        "sample": name,
        "label": sample.label,
        "z01_guess": guess,
        "omega": list(sample.simplex),
        "what": "equal-weight mixture of the Gaussians (calibrated sigma) of the leave-one-out "
        "CNN seeds on the hologram binned 4x",
        "members": [
            {
                "estimate": repo_relative(p),
                "estimate_sha256": file_sha256(p),
                "model": m["model"],
                "model_sha256": m["model_sha256"],
                "z01_est": m["z01_est"],
                "sigma": m["sigma"],
            }
            for p, m in zip(estimates, members, strict=True)
        ],
        "z01_est": z01,
        "dz01_est": est["dz01"],
        "z01_mean": guess + est["dz01_mean"],
        "sigma": est["sigma"],
        "mass_in_omega": float(on_omega.prior_mass[0]),
        "intervals": intervals,
        "hpd_to_central": {k: i["hpd_width_um"] / i["width_um"] for k, i in intervals.items()},
        "guess": against(guess),
        "references": {key: against(value) for key, value in refs.items()},
        "modes": [{"z01": guess + m["t"], **m} for m in modes(on_omega)],
        "binning_phases": {
            "n": len(phase_z01),
            "z01_est": [float(v) for v in phase_z01],
            "std_um": float(np.std(phase_z01, ddof=1) * 1e6),
            "range_um": float((phase_z01.max() - phase_z01.min()) * 1e6),
        },
        "seeds": {
            "z01_est": seeds,
            "std_um": float(np.std(seeds, ddof=1) * 1e6) if len(seeds) > 1 else 0.0,
            "range_um": float((max(seeds) - min(seeds)) * 1e6),
        },
        "posterior": {
            "z01_mm": ((guess + on_omega.theta[::step]) * 1e3).tolist(),
            "density_per_mm": (on_omega.density[0, ::step] * on_omega.prior_mass[0] / 1e3).tolist(),
        },
    }


NPE_FIELDS = (
    "models",
    "model_sha256",
    "z01_est",
    "dz01_est",
    "z01_mean",
    "z01_mode",
    "sigma",
    "prior_mass",
    "intervals",
    "hpd_to_central",
    "guess",
    "references",
    "modes",
    "edges",
    "binning_phases",
    "posterior",
)


def npe_record(name: str, path: Path) -> dict:
    """The NPE ensemble of ``results/npe-loo/measured/<sample>.json`` as a flat record."""
    record = json.loads(path.read_text())
    run = record["runs"]["ensemble"]
    return {
        "sample": name,
        "label": record["label"],
        "z01_guess": record["z01_guess"],
        "omega": record["omega"],
        "what": "mean of the densities of the NPE seeds on the hologram binned 4x (Stage 7)",
        "record": repo_relative(path),
        "record_sha256": file_sha256(path),
        **{k: run[k] for k in NPE_FIELDS if k in run},
        "seeds": record.get("seeds"),
    }


def cmd_sources(args) -> None:
    for name in args.samples:
        refs = paper_references(name, args.measured)
        estimates = cnn_estimates(args, name)
        if len(estimates) > 1:
            record = {**mixture_record(name, estimates, refs), **environment()}
            write_json(args.out / "cnn-loo-ensemble" / f"{name}.json", record)
            i95 = record["intervals"]["95"]
            print(
                f"{name} CNN ensemble ({len(estimates)} seeds): {record['z01_est'] * 1e3:.3f} mm, "
                f"95 % {i95['lower'] * 1e3:.3f}-{i95['upper'] * 1e3:.3f} mm"
            )
        else:
            print(f"{name}: {len(estimates)} CNN estimate(s), no ensemble")
        path = args.npe_measured / f"{name}.json"
        if path.exists() and "ensemble" in json.loads(path.read_text())["runs"]:
            record = {**npe_record(name, path), **environment()}
            write_json(args.out / "npe-loo-ensemble" / f"{name}.json", record)
            i95 = record["intervals"]["95"]
            print(
                f"{name} NPE ensemble: {record['z01_est'] * 1e3:.3f} mm, "
                f"95 % {i95['lower'] * 1e3:.3f}-{i95['upper'] * 1e3:.3f} mm"
            )
        else:
            print(f"{name}: no NPE ensemble in {repo_relative(path)}")


# --- test splits ---------------------------------------------------------------------------


def npe_members(args, name: str) -> list[tuple[int, Path]]:
    """The trained NPE runs of ``name`` among ``--seeds``, seed 0 first."""
    runs = [(k, args.npe / run_name(name, k)) for k in args.seeds]
    return [(k, run) for k, run in runs if (run / "model.pt").exists()]


def cnn_member(args, name: str, seed: int) -> Path:
    return (
        args.cnn / f"loo-{name}" if seed == 0 else args.cnn_seeds / f"seed-{seed}" / f"loo-{name}"
    )


def mixture_table(members: list[list[dict]]) -> dict[str, np.ndarray]:
    """The CNN ensemble on a split: per sample the equal-weight mixture of the Gaussians of the
    seeds (calibrated ``sigma``) in the columns of :func:`npe_table`, its median the estimate;
    ``seconds`` is the sum over the seeds."""
    first = members[0]
    for rows in members[1:]:
        same = [r["index"] for r in rows] == [r["index"] for r in first]
        if (
            not same
            or max(abs(r["z01_true"] - f["z01_true"]) for r, f in zip(rows, first, strict=True))
            > 1e-12
        ):
            raise ValueError("the CNN seeds were not evaluated on the same samples")
    z01 = np.array([r["z01_true"] for r in first])
    dz01 = z01 - np.array([r["z01_guess"] for r in first])
    mix = GaussianMixture(
        np.array([[r["dz01_est"] for r in rows] for rows in members]).T,
        np.array([[r["sigma"] for r in rows] for rows in members]).T,
    )
    est = mixture_summaries(mix)
    table = {
        "index": np.array([r["index"] for r in first]),
        "z01_true": z01,
        "error_um": (est["dz01"] - dz01) * 1e6,
        "sigma_um": est["sigma"] * 1e6,
        "pit": mix.pit(dz01),
        "hpd_level": mix.hpd_level(dz01),
        "log_density_um": mix.log_density(dz01) - math.log(1e6),
        "seconds": np.sum([[r["seconds"] for r in rows] for rows in members], axis=0),
    }
    for key in LEVELS:
        lower, upper = est[f"lower_{key}"], est[f"upper_{key}"]
        table[f"covered_{key}"] = (lower <= dz01) & (dz01 <= upper)
        table[f"width_{key}_um"] = (upper - lower) * 1e6
    return table


def test_tables(args, name: str) -> tuple[dict[str, dict], dict[str, Path], list[dict] | None]:
    """The seeds (``cnn``, ``cnn-seed<k>``, ``npe``, ``npe-seed<k>``), the two ensembles and
    the guess on the leave-one-out test split of ``name``, the paths of the predictions and
    the rows of the first NPE."""
    tables, paths, cnn_rows, npe_rows = {}, {}, [], None
    for k in args.seeds:
        path = cnn_member(args, name, k) / "test" / "predictions.csv"
        if path.exists():
            rows = read_predictions(path)
            tables[method_name("cnn", k)], paths[method_name("cnn", k)] = cnn_table(rows), path
            cnn_rows.append(rows)
    for k in args.seeds:
        path = args.npe / run_name(name, k) / "test" / "predictions.csv"
        if path.exists():
            rows = read_predictions(path)
            tables[method_name("npe", k)], paths[method_name("npe", k)] = npe_table(rows), path
            npe_rows = npe_rows or rows
    if cnn_rows:
        tables["cnn-ensemble"] = mixture_table(cnn_rows)
    path = args.npe / ensemble_name(name) / "test" / "predictions.csv"
    if path.exists():
        tables["npe-ensemble"] = npe_table(read_predictions(path))
        paths["npe-ensemble"] = path
    if npe_rows:
        tables["guess"] = guess_table(npe_rows)
    first = next(iter(tables.values()), None)
    for method, t in tables.items():
        if not np.array_equal(t["index"], first["index"]):
            raise ValueError(f"{name}: {method} was not evaluated on the same test samples")
    return tables, paths, npe_rows


def failures(tables: dict[str, dict]) -> dict[str, np.ndarray]:
    """What fails per test sample: absolute error and 95 % width of each ensemble and the
    standard deviation of the estimates of its seeds (with two seeds or more)."""
    out = {}
    for kind in ("npe", "cnn"):
        ensemble = tables.get(f"{kind}-ensemble")
        if ensemble is None:
            continue
        out[f"{kind}_abs_error_um"] = np.abs(ensemble["error_um"])
        out[f"{kind}_width_95_um"] = ensemble["width_95_um"]
        seeds = [
            t["error_um"] for m, t in tables.items() if m == kind or m.startswith(f"{kind}-seed")
        ]
        if len(seeds) > 1:
            out[f"{kind}_seed_std_um"] = np.std(seeds, axis=0, ddof=1)
    return out


def shrinkage(shift: np.ndarray, offset: np.ndarray) -> float:
    """Least-squares slope of the estimate's shift from the guess against the true offset: 1
    for an estimate that follows the truth, 0 for one that stays at the guess."""
    x = offset - offset.mean()
    return float((x * (shift - shift.mean())).sum() / (x**2).sum()) if (x**2).sum() > 0 else 0.0


# --- misspec -------------------------------------------------------------------------------


def network_inputs(path: Path, representation, scale: float, cache: Path | None):
    """The network inputs ``(N, ...)`` of a split file and its identity; with ``cache``, kept
    as ``.npy`` under the SHA-256 of the file and of the representation."""
    identity = {**dataset_identity(path), "path": repo_relative(path)}
    key = json.dumps(
        [dataclasses.asdict(representation.config), list(representation.shape)], default=str
    )
    file = None
    if cache is not None:
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        file = cache / f"{identity['sha256'][:16]}-{digest}.npy"
        if file.exists():
            return np.load(file), identity
    inputs = load_inputs(path, representation, scale).inputs.float().numpy()
    if file is not None:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(file, inputs)
    return inputs, identity


def flat(x: np.ndarray | None) -> np.ndarray | None:
    return None if x is None else x.reshape(len(x), -1)


def embeddings(autofocus: NPEAutofocus, inputs: np.ndarray, chunk: int = 500) -> list[np.ndarray]:
    """The embedding of the inputs by every NPE of the ensemble, ``(N, features)`` each."""
    out = []
    with torch.inference_mode():
        for estimator in autofocus.estimators:
            parts = [
                estimator.embed(torch.from_numpy(inputs[s : s + chunk]).to(autofocus.device))
                for s in range(0, len(inputs), chunk)
            ]
            out.append(torch.cat(parts).double().cpu().numpy())
    return out


def space_measures(train, test, measured, input_space: bool) -> tuple[dict, dict]:
    """Distances of the test samples and of the measured rows to the training samples in one
    space, ``{measure: (test, measured or None)}``, and the principal components used."""
    pca = (
        PrincipalComponents(train, components=INPUT_COMPONENTS)
        if input_space
        else PrincipalComponents(train, variance=EMBEDDING_VARIANCE)
    )
    measures = {}
    for k in NEIGHBOURS:
        k = min(k, len(train))
        measures[f"knn{k}"] = (
            knn_distance(test, train, k),
            None if measured is None else knn_distance(measured, train, k),
        )
    functions = {"mahalanobis": pca.mahalanobis}
    if input_space:
        functions["residual"] = pca.residual_rms
    for measure, fn in functions.items():
        measures[measure] = (fn(test), None if measured is None else fn(measured))
    info = {
        "dimension": pca.dimension,
        "rank": pca.rank,
        "components": pca.components,
        "explained": pca.explained,
        "components_for": {
            str(v): min(pca.components_for(v), pca.rank) for v in (0.9, 0.99, 0.999)
        },
    }
    return measures, info


def association(distance: np.ndarray, failure: np.ndarray, resamples: int) -> dict:
    """Spearman of a distance and a failure on the test split with a bootstrap interval, and
    the mean failure of the nearest and of the farthest quarter."""
    near = distance <= np.quantile(distance, 0.25)
    far = distance >= np.quantile(distance, 0.75)
    rows = np.column_stack([distance, failure])
    return {
        "spearman": spearman(distance, failure),
        "ci": list(bootstrap_ci(rows, lambda d: spearman(d[:, 0], d[:, 1]), resamples=resamples)),
        "nearest_quarter_mean": float(failure[near].mean()),
        "farthest_quarter_mean": float(failure[far].mean()),
    }


def neighbourhood_check(test: np.ndarray, fail: dict, k: int, resamples: int) -> dict:
    """Whether a neighbourhood predicts the failures on the test split: per test sample, the
    mean of each failure over its ``k`` nearest other test samples (Euclidean, as for the
    hologram's neighbours) and the Spearman of that mean with the sample's own failure. The
    bootstrap resamples the pairs and ignores that neighbourhoods overlap."""
    t = test.astype(np.float64)
    sq = (t**2).sum(axis=1)
    d = sq[:, None] + sq[None, :] - 2.0 * t @ t.T
    np.fill_diagonal(d, np.inf)
    nearest = np.argpartition(d, k - 1, axis=1)[:, :k]
    out = {"k": k}
    for f, v in fail.items():
        local = v[nearest].mean(axis=1)
        rows = np.column_stack([local, v])
        out[f] = {
            "spearman": spearman(local, v),
            "ci": list(
                bootstrap_ci(rows, lambda r: spearman(r[:, 0], r[:, 1]), resamples=resamples)
            ),
        }
    return out


def measure_record(test: np.ndarray, measured: np.ndarray | None) -> dict:
    out = {
        "test": {
            "median": float(np.median(test)),
            "p05": float(np.percentile(test, 5)),
            "p95": float(np.percentile(test, 95)),
            "max": float(test.max()),
        }
    }
    if measured is not None:
        phases = [percentile_rank(test, v) for v in measured[1:]]
        out["measured"] = float(measured[0])
        out["percentile"] = percentile_rank(test, measured[0])
        out["phases"] = {
            "n": len(phases),
            "values": [float(v) for v in measured[1:]],
            "min": float(measured[1:].min()),
            "max": float(measured[1:].max()),
            "percentile_min": float(min(phases)),
            "percentile_median": float(np.median(phases)),
            "percentile_max": float(max(phases)),
        }
    return out


def local_look(tables: dict, fail: dict, shift: dict, offset: np.ndarray, rows: np.ndarray) -> dict:
    """Failures, shrinkage, shift from the guess and coverage of the ensembles on the test
    samples ``rows``."""
    out = {"n": len(rows)}
    for f, v in fail.items():
        out[f"{f}_median"] = float(np.median(v[rows]))
    for kind, s in shift.items():
        table = tables[f"{kind}-ensemble"]
        out[f"{kind}_mae_um"] = float(fail[f"{kind}_abs_error_um"][rows].mean())
        out[f"{kind}_shrinkage"] = shrinkage(s[rows], offset[rows])
        out[f"{kind}_shift_median_um"] = float(np.median(s[rows]))
        out[f"{kind}_coverage"] = {k: float(table[f"covered_{k}"][rows].mean()) for k in LEVELS}
    out["dz01_true_um"] = {
        "min": float(offset[rows].min()),
        "median": float(np.median(offset[rows])),
        "max": float(offset[rows].max()),
    }
    return out


def resolve(path: str) -> Path:
    return Path(path) if Path(path).is_absolute() else REPO / path


def phase_seed_std(phase_z01: list[list[float]]) -> dict:
    """Spread of the seeds' estimates on each of the 16 binning phases, in micrometres."""
    std = np.std(np.array(phase_z01), axis=0, ddof=1) * 1e6
    return {"min": float(std.min()), "median": float(np.median(std)), "max": float(std.max())}


def measured_failures(args, name: str, fail: dict) -> dict:
    """The failures of the two ensembles on the measured hologram (``sources``), with the
    padding-4 search of Stage 3 as the truth, and their percentiles among the test split."""
    out = {}
    for kind, folder in (("npe", "npe-loo-ensemble"), ("cnn", "cnn-loo-ensemble")):
        path = args.sources / folder / f"{name}.json"
        if not path.exists() or f"{kind}_abs_error_um" not in fail:
            continue
        r = json.loads(path.read_text())
        guess, p4 = r["z01_guess"], r["references"].get("baseline_b1_p4")
        entry = {
            "record": repo_relative(path),
            "z01_est": r["z01_est"],
            "shift_um": (r["z01_est"] - guess) * 1e6,
            "width_95_um": r["intervals"]["95"]["width_um"],
            "seed_std_um": r["seeds"]["std_um"],
            "minus_paper2_um": r["references"]["paper2_nm"]["distance_um"],
            "percentiles": {
                "width_95_um": percentile_rank(
                    fail[f"{kind}_width_95_um"], r["intervals"]["95"]["width_um"]
                )
            },
        }
        if f"{kind}_seed_std_um" in fail:
            entry["percentiles"]["seed_std_um"] = percentile_rank(
                fail[f"{kind}_seed_std_um"], entry["seed_std_um"]
            )
        if p4 is not None:
            entry["minus_p4_um"] = p4["distance_um"]
            entry["p4_minus_guess_um"] = (p4["z01"] - guess) * 1e6
            entry["moved_fraction"] = entry["shift_um"] / entry["p4_minus_guess_um"]
            entry["percentiles"]["abs_error_um"] = percentile_rank(
                fail[f"{kind}_abs_error_um"], abs(p4["distance_um"])
            )
        if kind == "npe":
            record = json.loads(resolve(r["record"]).read_text())
            seeds = [run for key, run in record["runs"].items() if key != "ensemble"]
        else:
            seeds = [json.loads(resolve(m["estimate"]).read_text()) for m in r["members"]]
        if len(seeds) > 1:
            entry["phases_seed_std_um"] = phase_seed_std(
                [s["binning_phases"]["z01_est"] for s in seeds]
            )
        out[kind] = entry
    return out


def neighbour_percentiles(fail: dict, shift: dict, measured: dict, rows: np.ndarray) -> dict:
    """Percentiles of the hologram's failures and shift from the guess (``measured_failures``)
    among those of its nearest test samples ``rows``."""
    out = {}
    for kind, m in measured.items():
        values = {f"{kind}_{k}": m[k] for k in ("seed_std_um", "width_95_um") if k in m}
        if "minus_p4_um" in m:
            values[f"{kind}_abs_error_um"] = abs(m["minus_p4_um"])
        out[kind] = {
            key.removeprefix(f"{kind}_"): percentile_rank(fail[key][rows], v)
            for key, v in values.items()
            if key in fail
        }
        if kind in shift:
            out[kind]["shift_um"] = percentile_rank(shift[kind][rows], m["shift_um"])
    return out


def measure_row(name: str, space: str, measure: str, components: int, m: dict) -> dict:
    """A row of ``summary.csv``: the hologram's distance in one space and measure, its
    percentile among the test split and the Spearman of the distance with each failure."""
    phases = m.get("phases", {})
    row = {
        "sample": name,
        "space": space,
        "measure": measure,
        "components": components,
        "measured": m.get("measured", ""),
        "percentile": m.get("percentile", ""),
        "phases_percentile_min": phases.get("percentile_min", ""),
        "phases_percentile_max": phases.get("percentile_max", ""),
        "test_median": m["test"]["median"],
        "test_p95": m["test"]["p95"],
    }
    for f, a in m.get("failures", {}).items():
        row[f"spearman_{f}"] = a["spearman"]
        row[f"spearman_{f}_lo"], row[f"spearman_{f}_hi"] = a["ci"]
    return row


def control_record(args, name: str, autofocus: NPEAutofocus, x_measured: np.ndarray) -> dict | None:
    """The positive control: the same distances of the hologram (and its binning phases) to
    the training set of the Stage 5 simulator, in the network input and in the embedding of
    the NPE trained on it, as percentiles among that set's test split."""
    run, root = args.control / f"{CONTROL}-{name}", args.data / f"{CONTROL}-{name}"
    if not (run / "model.pt").exists() or not (root / "train.h5").exists():
        print(f"{name}: no control ({repo_relative(run)}, {repo_relative(root)})")
        return None
    control = NPEAutofocus.load([run], args.device)
    rep, scale = control.representation, control.config.target_scale
    if (
        dataclasses.asdict(rep.config) != dataclasses.asdict(autofocus.representation.config)
        or tuple(rep.shape) != tuple(autofocus.representation.shape)
        or control.fov_shape != autofocus.fov_shape
    ):
        raise ValueError(f"{name}: the control model {run} sees another input")
    x_train, train_id = network_inputs(root / "train.h5", rep, scale, args.cache)
    x_test, test_id = network_inputs(root / "test.h5", rep, scale, args.cache)
    spaces = {
        "input": tuple(flat(x) for x in (x_train, x_test, x_measured)),
        "embedding": tuple(embeddings(control, x)[0] for x in (x_train, x_test, x_measured)),
    }
    out = {
        "simulator": f"Stage 5 ({CONTROL}-{name}, without texture)",
        "model": repo_relative(run),
        "model_sha256": model_sha256(run),
        "data": {"train": train_id, "test": test_id},
        "spaces": {},
    }
    for space, (train, test, measured) in spaces.items():
        measures, info = space_measures(train, test, measured, space == "input")
        out["spaces"][space] = {
            **info,
            "measures": {m: measure_record(t, v) for m, (t, v) in measures.items()},
        }
        main = out["spaces"][space]["measures"][f"knn{min(NEIGHBOURS[0], len(train))}"]
        print(
            f"{name} control-{space}: knn test median {main['test']['median']:.3f}, hologram "
            f"{main['measured']:.3f} = {main['percentile']:.1f} % (phases "
            f"{main['phases']['percentile_min']:.1f}-{main['phases']['percentile_max']:.1f} %)"
        )
    return out


def cmd_misspec(args) -> None:
    summary = []
    for name in args.samples:
        members = npe_members(args, name)
        if not members:
            print(f"{name}: no trained NPE below {repo_relative(args.npe)}")
            continue
        autofocus = NPEAutofocus.load([run for _, run in members], args.device)
        rep, scale = autofocus.representation, autofocus.config.target_scale
        root = args.data / f"loo-{name}"
        x_train, train_id = network_inputs(root / "train.h5", rep, scale, args.cache)
        x_test, test_id = network_inputs(root / "test.h5", rep, scale, args.cache)
        tables, paths, npe_rows = test_tables(args, name)
        if "npe-ensemble" not in tables or not np.array_equal(
            tables["npe-ensemble"]["index"], np.arange(len(x_test))
        ):
            print(f"{name}: no predictions of the NPE ensemble on the whole test split, skipped")
            continue
        x_measured = None
        if args.holograms is not None:
            sample = MEASURED_SAMPLES[name]
            hologram = load_measured(args.holograms, sample, BINNING)
            if tuple(hologram.shape) != autofocus.fov_shape:
                raise ValueError(
                    f"{name}: hologram {tuple(hologram.shape)}, model {autofocus.fov_shape}"
                )
            native = load_measured(args.holograms, sample, 1).numpy()
            phases = [p.astype(np.float32) for p in binning_phases(native, BINNING)]
            with torch.inference_mode():
                x_measured = (
                    rep(torch.from_numpy(np.stack([hologram.numpy(), *phases]))).cpu().numpy()
                )
        fail = failures(tables)
        offset = np.array([r["z01_true"] - r["z01_guess"] for r in npe_rows]) * 1e6
        shift = {
            kind: tables[f"{kind}-ensemble"]["error_um"] + offset
            for kind in ("npe", "cnn")
            if f"{kind}-ensemble" in tables
        }
        spaces = {"input": tuple(flat(x) for x in (x_train, x_test, x_measured))}
        parts = [embeddings(autofocus, x) for x in (x_train, x_test)]
        parts.append(embeddings(autofocus, x_measured) if x_measured is not None else None)
        for i, (k, _) in enumerate(members):
            measured = None if parts[2] is None else parts[2][i]
            spaces[method_name("embedding", k)] = (parts[0][i], parts[1][i], measured)
        record = {
            "sample": name,
            "label": MEASURED_SAMPLES[name].label,
            "data": {"train": train_id, "test": test_id},
            "models": [repo_relative(run) for _, run in members],
            "model_sha256": [model_sha256(run) for _, run in members],
            "predictions": {m: repo_relative(p) for m, p in paths.items()},
            "measured": None
            if x_measured is None
            else f"the hologram binned {BINNING}x, then its {len(x_measured) - 1} binning phases",
            "settings": {
                "neighbours": list(NEIGHBOURS),
                "input_components": INPUT_COMPONENTS,
                "embedding_variance": EMBEDDING_VARIANCE,
                "nearest": NEAREST,
                "resamples": args.resamples,
            },
            "spaces": {},
            "test": local_look(tables, fail, shift, offset, np.arange(len(x_test))),
            "neighbours": {},
        }
        columns = {
            "index": tables["npe-ensemble"]["index"],
            "phantom_type": np.array([r["phantom_type"] for r in npe_rows]),
            "photons": np.array([r["photons"] for r in npe_rows]),
            "dz01_true_um": offset,
            **fail,
            **{f"{kind}_shift_um": s for kind, s in shift.items()},
        }
        for space, (train, test, measured) in spaces.items():
            measures, info = space_measures(train, test, measured, space == "input")
            entry = {**info, "measures": {}}
            for measure, (d_test, d_measured) in measures.items():
                m = measure_record(d_test, d_measured)
                m["failures"] = {f: association(d_test, v, args.resamples) for f, v in fail.items()}
                entry["measures"][measure] = m
                columns[f"{space}_{measure}"] = d_test
                summary.append(measure_row(name, space, measure, info["components"], m))
            entry["neighbourhood"] = neighbourhood_check(
                test, fail, min(NEAREST, len(test) - 1), args.resamples
            )
            record["spaces"][space] = entry
            if measured is not None:
                nearest = np.argsort(np.linalg.norm(test - measured[0], axis=1))[
                    : min(NEAREST, len(test))
                ]
                record["neighbours"][space] = {
                    **local_look(tables, fail, shift, offset, nearest),
                    "indices": [int(i) for i in nearest],
                }
            main = entry["measures"][f"knn{min(NEIGHBOURS[0], len(train))}"]
            print(
                f"{name} {space} ({info['dimension']} dims, rank {info['rank']}): knn "
                f"test median {main['test']['median']:.3f}"
                + (
                    f", hologram {main['measured']:.3f} = {main['percentile']:.1f} % "
                    f"(phases {main['phases']['percentile_min']:.1f}-"
                    f"{main['phases']['percentile_max']:.1f} %)"
                    if "measured" in main
                    else ""
                )
                + ", Spearman with |error| / seed std of the NPE ensemble "
                + " / ".join(
                    f"{main['failures'][f]['spearman']:+.2f}"
                    for f in ("npe_abs_error_um", "npe_seed_std_um")
                    if f in main["failures"]
                )
                + f", of the {entry['neighbourhood']['k']} nearest test samples' mean with "
                + "the own "
                + " / ".join(
                    f"{entry['neighbourhood'][f]['spearman']:+.2f}"
                    for f in ("npe_abs_error_um", "npe_seed_std_um")
                    if f in entry["neighbourhood"]
                )
            )
        if x_measured is not None:
            record["measured_failures"] = measured_failures(args, name, fail)
            for near in record["neighbours"].values():
                near["measured_percentiles"] = neighbour_percentiles(
                    fail, shift, record["measured_failures"], np.array(near["indices"])
                )
            if args.control is not None:
                control = control_record(args, name, autofocus, x_measured)
                if control is not None:
                    record["control"] = control
                    for space, entry in control["spaces"].items():
                        for measure, m in entry["measures"].items():
                            summary.append(
                                measure_row(
                                    name, f"control-{space}", measure, entry["components"], m
                                )
                            )
        record.update(environment())
        write_json(args.out / f"{name}.json", record)
        n = len(columns["index"])
        write_csv(
            args.out / f"test-{name}.csv",
            [
                {k: v[i].item() if hasattr(v[i], "item") else v[i] for k, v in columns.items()}
                for i in range(n)
            ],
        )
    if not summary:
        raise SystemExit("no sample with NPE runs and predictions")
    write_csv(args.out / "summary.csv", summary)


# --- ensembles -----------------------------------------------------------------------------


def shifted(table: dict[str, np.ndarray], offset: int) -> dict[str, np.ndarray]:
    """The table with indices moved by ``offset``, to pool the test splits of the probes."""
    return {**table, "index": table["index"] + offset}


def pooled_prior(priors: list[dict], sizes: list[int]) -> dict:
    """The intervals of the uniform priors of several test splits, weighted by their sizes."""
    w = np.array(sizes, dtype=float) / sum(sizes)

    def mean(get) -> float:
        return float(sum(wi * get(p) for wi, p in zip(w, priors, strict=True)))

    return {
        "coverage": {k: mean(lambda p, k=k: p["coverage"][k]) for k in LEVELS},
        "width_um": {k: mean(lambda p, k=k: p["width_um"][k]) for k in LEVELS},
        "nll": mean(lambda p: p["nll"]),
    }


def comparison_record(tables: dict[str, dict]) -> dict:
    record = {"methods": {m: table_metrics(t) for m, t in tables.items()}, "paired": {}}
    for a, b in PAIRS:
        if a in tables and b in tables:
            record["paired"][f"{a}_vs_{b}"] = paired(tables[a], tables[b])
    for m in ENSEMBLES:
        if m in tables and "guess" in tables:
            record["paired"][f"{m}_vs_guess"] = {"mae_um": paired_mae(tables[m], tables["guess"])}
    return record


def measured_comparison(args, name: str, tables: dict[str, dict]) -> dict | None:
    """The two ensembles on the measured hologram (``sources``), against [2] and the padding-4
    search of Stage 3, with the width relative to the test split."""
    out = {}
    for kind, folder in (
        ("cnn-ensemble", "cnn-loo-ensemble"),
        ("npe-ensemble", "npe-loo-ensemble"),
    ):
        path = args.sources / folder / f"{name}.json"
        if not path.exists() or kind not in tables:
            continue
        r = json.loads(path.read_text())
        test = tables[kind]
        entry = {
            "record": repo_relative(path),
            "record_sha256": file_sha256(path),
            "z01_est": r["z01_est"],
            "minus_guess_um": (r["z01_est"] - r["z01_guess"]) * 1e6,
            "width_um": {k: r["intervals"][k]["width_um"] for k in LEVELS},
            "width_68_to_test_median": r["intervals"]["68"]["width_um"]
            / float(np.median(test["width_68_um"])),
            "seeds": r["seeds"],
            "phases_std_um": r["binning_phases"]["std_um"],
            "references": {},
        }
        for key in ("paper2_nm", "baseline_b1_p4"):
            if key in r["references"]:
                ref = r["references"][key]
                entry["references"][key] = {
                    k: ref[k] for k in ("z01", "distance_um", "pit", "hpd_level", "inside")
                }
                entry["references"][key]["abs_distance_percentile"] = percentile_rank(
                    np.abs(test["error_um"]), abs(ref["distance_um"])
                )
        out[kind] = entry
    return out or None


def cmd_ensembles(args) -> None:
    summary, pooled, priors, sizes, names, measured, measured_rows = [], {}, [], [], [], {}, []
    for number, name in enumerate(args.samples):
        tables, paths, npe_rows = test_tables(args, name)
        if not all(m in tables for m in (*ENSEMBLES, "npe", "cnn")):
            print(f"{name}: an ensemble or a seed-0 run is missing, skipped")
            continue
        info = read_run(paths["npe"].parent)
        prior = prior_metrics(npe_rows, info["prior"], info["config"]["target_scale"])
        record = {
            "sample": name,
            "n_samples": len(tables["npe"]["index"]),
            "predictions": {m: repo_relative(p) for m, p in paths.items()},
            "cnn-ensemble": "per sample the equal-weight mixture of the Gaussians of "
            + ", ".join(m for m in tables if m.startswith("cnn") and m != "cnn-ensemble"),
            "prior": prior,
            **comparison_record(tables),
        }
        write_json(args.out / f"{name}.json", record)
        write_csv(args.out / f"{name}.csv", per_sample_rows(tables, None))
        for m, t in tables.items():
            pooled.setdefault(m, []).append(shifted(t, 100000 * number))
            summary.append(
                summary_row(name, m, record["methods"][m], prior if m == "guess" else None)
            )
        priors.append(prior)
        sizes.append(record["n_samples"])
        names.append(name)
        d = record["paired"]["npe-ensemble_vs_cnn-ensemble"]
        print(
            f"{name}: MAE CNN ensemble {record['methods']['cnn-ensemble']['error_um']['mae']:.0f}, "
            f"NPE ensemble {record['methods']['npe-ensemble']['error_um']['mae']:.0f} um, "
            f"difference {d['mae_um']['difference']:+.0f} [{d['mae_um']['ci'][0]:+.0f}, "
            f"{d['mae_um']['ci'][1]:+.0f}]; coverage 68 % "
            f"{record['methods']['cnn-ensemble']['coverage']['68']:.3f} / "
            f"{record['methods']['npe-ensemble']['coverage']['68']:.3f}"
        )
        entry = measured_comparison(args, name, tables)
        if entry:
            measured[name] = entry
            for kind, e in entry.items():
                row = {"sample": name, "method": kind, "z01_est_mm": e["z01_est"] * 1e3}
                row["minus_guess_um"] = e["minus_guess_um"]
                for key, ref in e["references"].items():
                    row[f"minus_{key}_um"] = ref["distance_um"]
                    row[f"{key}_in_68"], row[f"{key}_in_95"] = (
                        int(ref["inside"][k]) for k in LEVELS
                    )
                    row[f"{key}_hpd_level"] = ref["hpd_level"]
                    row[f"{key}_abs_distance_percentile"] = ref["abs_distance_percentile"]
                row["width_68_um"], row["width_95_um"] = (e["width_um"][k] for k in LEVELS)
                row["width_68_to_test_median"] = e["width_68_to_test_median"]
                row["seed_std_um"] = e["seeds"]["std_um"]
                row["phases_std_um"] = e["phases_std_um"]
                measured_rows.append(row)
    if not summary:
        raise SystemExit("no sample with both ensembles")
    if len(sizes) > 1:
        common = [m for m, parts in pooled.items() if len(parts) == len(sizes)]
        tables = {
            m: {k: np.concatenate([p[k] for p in pooled[m]]) for k in pooled[m][0]} for m in common
        }
        prior = pooled_prior(priors, sizes)
        record = {"samples": names, "prior": prior}
        record.update(comparison_record(tables))
        write_json(args.out / "pooled.json", record)
        for m in tables:
            summary.append(
                summary_row("pooled", m, record["methods"][m], prior if m == "guess" else None)
            )
    write_csv(args.out / "comparison.csv", summary)
    if measured:
        write_json(args.out / "measured.json", {"samples": measured, **environment()})
        write_csv(args.out / "measured.csv", measured_rows)


# --- warmstart -----------------------------------------------------------------------------


def distinct_points(evaluations: list[dict]) -> int:
    """Reconstructed points, counting points closer than :data:`SAME_POINT` once."""
    seen: list[float] = []
    for e in evaluations:
        if e["reconstructed"] and not any(abs(e["z01"] - s) <= SAME_POINT for s in seen):
            seen.append(e["z01"])
    return len(seen)


def search_entry(path: Path, cold: dict | None) -> dict:
    r = json.loads(path.read_text())
    res = r["result"]
    points = distinct_points(res["evaluations"])
    entry = {
        "record": repo_relative(path),
        "sha256": file_sha256(path),
        "git_commit": r.get("git_commit"),
        "threads": r.get("threads"),
        "z01": res["z01"],
        "mfe": res["mfe"],
        "converged": res["converged"],
        "nm_iterations": res["nm_iterations"],
        "n_reconstructions": res["n_reconstructions"],
        "n_evaluations": res["n_evaluations"],
        "distinct_points": points,
        "duplicates": res["n_reconstructions"] - points,
        "seconds": res["seconds"],
        "seconds_per_reconstruction": r["seconds_per_reconstruction"],
        "bounds": res["bounds"],
        "initial_simplex": res.get("initial_simplex"),
        "revisit_tolerance": r.get("revisit_tolerance", 0.0),
        "minus_paper2_um": r["z01_minus_paper2_nm_um"],
    }
    if "warm_start" in r:
        entry["start"] = {k: r["warm_start"][k] for k in ("source", "sha256", "z01_est")}
    if "box" in r:
        entry["box"] = r["box"]
    if cold is not None:
        lower, upper = res["bounds"]
        entry.update(
            {
                "minus_cold_um": (res["z01"] - cold["z01"]) * 1e6,
                "same_as_cold": bool(abs(res["z01"] - cold["z01"]) * 1e6 <= XATOL_UM),
                "mfe_relative_to_cold": res["mfe"] / cold["mfe"] - 1.0,
                "reconstructions_to_cold": res["n_reconstructions"] / cold["n_reconstructions"],
                "seconds_to_cold": res["seconds"] / cold["seconds"],
                "cold_inside_bounds": bool(lower <= cold["z01"] <= upper),
            }
        )
        if "start" in entry:
            entry["start_minus_cold_um"] = (entry["start"]["z01_est"] - cold["z01"]) * 1e6
    return entry


def cmd_warmstart(args) -> None:
    samples, rows = {}, []
    for name in args.samples:
        cold_path = args.measured / "b1-p4" / f"{name}.json"
        if not cold_path.exists():
            print(f"{name}: no full search {repo_relative(cold_path)}, skipped")
            continue
        cold = json.loads(cold_path.read_text())["result"]
        entries = {}
        for variant, folder, _ in SEARCHES:
            path = args.measured / folder / f"{name}.json"
            if path.exists():
                entries[variant] = search_entry(path, None if variant == "cold" else cold)
        best = min(e["mfe"] for e in entries.values())
        for variant, e in entries.items():
            e["mfe_relative_to_best"] = e["mfe"] / best - 1.0
            row = {"sample": name, "variant": variant, "z01_mm": e["z01"] * 1e3}
            for key in (
                "minus_cold_um", "same_as_cold", "minus_paper2_um", "mfe", "mfe_relative_to_best",
                "n_reconstructions", "distinct_points", "duplicates", "n_evaluations",
                "nm_iterations", "converged", "seconds", "seconds_per_reconstruction", "threads",
                "reconstructions_to_cold", "seconds_to_cold", "start_minus_cold_um",
                "cold_inside_bounds",
            ):  # fmt: skip
                value = e.get(key, "")
                row[key] = int(value) if isinstance(value, bool) else value
            row["bounds_mm"] = "-".join(f"{b * 1e3:.3f}" for b in e["bounds"])
            rows.append(row)
            print(
                f"{name} {variant}: {e['z01'] * 1e3:.4f} mm"
                + (f" ({e['minus_cold_um']:+.0f} um from cold)" if "minus_cold_um" in e else "")
                + f", {e['n_reconstructions']} reconstructions ({e['distinct_points']} points), "
                f"{e['seconds']:.0f} s at {e['threads']} threads"
            )
        samples[name] = entries
    if not samples:
        raise SystemExit("no full search found")
    totals = {}
    for variant, _, what in SEARCHES:
        found = [s[variant] for s in samples.values() if variant in s]
        if not found:
            continue
        totals[variant] = {
            "what": what,
            "samples": [n for n, s in samples.items() if variant in s],
            "n_reconstructions": sum(e["n_reconstructions"] for e in found),
            "distinct_points": sum(e["distinct_points"] for e in found),
            "seconds": sum(e["seconds"] for e in found),
            "threads": sorted({e["threads"] for e in found}),
        }
        if variant != "cold":
            totals[variant]["max_abs_minus_cold_um"] = max(abs(e["minus_cold_um"]) for e in found)
            totals[variant]["all_same_as_cold"] = all(e["same_as_cold"] for e in found)
            cold = [samples[n]["cold"] for n in totals[variant]["samples"]]
            totals[variant]["cold_reconstructions"] = sum(e["n_reconstructions"] for e in cold)
            totals[variant]["cold_seconds"] = sum(e["seconds"] for e in cold)
    record = {
        "xatol_um": XATOL_UM,
        "same_point_m": SAME_POINT,
        "searches": {v: {"folder": f, "what": w} for v, f, w in SEARCHES},
        "samples": samples,
        "totals": totals,
    }
    write_json(args.out / "warmstart.json", record)
    write_csv(args.out / "warmstart.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, help="torch CPU threads")
    sub = parser.add_subparsers(dest="command", required=True)
    samples = {"nargs": "+", "choices": list(MEASURED_SAMPLES), "default": list(MEASURED_SAMPLES)}

    so = sub.add_parser("sources", help="the two ensembles on the measured holograms")
    so.add_argument("--samples", **samples)
    so.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    so.add_argument("--measured", type=Path, default=MEASURED, help="Stage 3 records (b1-p4)")
    so.add_argument("--cnn-measured", type=Path, default=MEASURED / "cnn-loo", help="CNN seed 0")
    so.add_argument(
        "--cnn-seeds",
        type=Path,
        default=RESULTS / "objectmodel" / "seeds",
        help="<cnn-seeds>/seed-<k>/measured/<sample>.json",
    )
    so.add_argument(
        "--npe-measured", type=Path, default=RESULTS / "npe-loo" / "measured", help="Stage 7"
    )
    so.add_argument("--out", type=Path, default=MEASURED)
    so.set_defaults(func=cmd_sources)

    def runs(p):
        p.add_argument("--samples", **samples)
        p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
        p.add_argument("--npe", type=Path, default=RESULTS / "npe-loo", help="Stage 7 NPE runs")
        p.add_argument("--cnn", type=Path, default=RESULTS / "learned", help="<cnn>/loo-<sample>")
        p.add_argument(
            "--cnn-seeds",
            type=Path,
            default=RESULTS / "objectmodel" / "seeds",
            help="<cnn-seeds>/seed-<k>/loo-<sample>",
        )
        p.add_argument(
            "--sources", type=Path, default=MEASURED, help="<sources>/{cnn,npe}-loo-ensemble"
        )

    mi = sub.add_parser("misspec", help="distances of the measured holograms to the training sets")
    runs(mi)
    mi.add_argument("--data", type=Path, default=DATA, help="<data>/loo-<sample>/{train,test}.h5")
    mi.add_argument("--holograms", type=Path, help="without it, only the test split")
    mi.add_argument("--cache", type=Path, help="keep the network inputs of the splits here")
    mi.add_argument(
        "--control",
        type=Path,
        help=f"Stage 7 NPE runs <control>/{CONTROL}-<sample> on the Stage 5 sets "
        f"<data>/{CONTROL}-<sample> (positive control, e.g. results/npe)",
    )
    mi.add_argument("--resamples", type=int, default=2000, help="bootstrap of the Spearman")
    mi.add_argument("--device", default="cpu")
    mi.add_argument("--out", type=Path, default=RESULTS / "misspecification")
    mi.set_defaults(func=cmd_misspec)

    en = sub.add_parser("ensembles", help="CNN ensemble against NPE ensemble")
    runs(en)
    en.add_argument("--out", type=Path, default=RESULTS / "ensembles")
    en.set_defaults(func=cmd_ensembles)

    wa = sub.add_parser("warmstart", help="warm and bounded searches against the full search")
    wa.add_argument("--samples", **samples)
    wa.add_argument("--measured", type=Path, default=MEASURED, help="<measured>/<search>/")
    wa.add_argument("--out", type=Path, default=MEASURED)
    wa.set_defaults(func=cmd_warmstart)

    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
