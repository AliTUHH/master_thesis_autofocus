"""Stage 7: neural posterior estimation of dz01 against the Gaussian CNN of Stage 4 (plan step 13).

* ``predict``: the NPE runs below ``results/npe`` on the validation, test and calibration
  splits of cpu-4x (matched runs: their validation and test splits), the ensemble of the five
  seeds of the main NPE (``npe-nsf-ensemble``) and the Stage 4 CNN on the calibration split
  (``results/learned/spectrum-cnn-wide/calibration``).
* ``sbc``: sbi's simulation-based calibration, sample-based expected coverage and TARP of an
  NPE on the calibration split (``<run>/calibration/sbi-*``).
* ``compare``: NPE, its variants and the CNN on the full test split, the 48 test samples of
  the Stage 3 baseline (with its raw and offset-corrected errors) and the calibration split:
  errors of the point estimate, coverage and width of the central 68/95 % intervals, NLL,
  whether wider intervals go with larger errors (rank correlation), PIT (SBC) and HPD
  (expected coverage) statistics, paired bootstrap differences
  (``results/npe/comparison/<set>.{json,csv}``, ``comparison.csv``).
* ``timing``: time per estimate of the CNN and the NPEs at the same thread count on the same
  test holograms, alternating, with the parts of one NPE estimate (``results/npe/timing.json``).
* ``examples``: posterior densities of the NPEs and the CNN for a few test samples
  (``results/npe/comparison/posteriors.json``, for the figures).
* ``perturb``: the matched NPEs (``results/npe/matched-<sample>``) on the matched test split
  and on the perturbed test splits of Stage 5 (``data/sim2real/<sample>/<name>.h5``, generated
  if missing; predictions in ``results/npe/sim2real/<sample>/<name>/``); widths and coverage
  against the matched CNN on the same sets, paired per sample, and the paired differences to
  the CNN with bootstrap intervals (``results/npe/sim2real/<sample>.{json,csv}``).
* ``measured``: the matched NPEs on the four measured holograms, binned 4x as in Stages 3 and
  5, with the 16 phases of the binning grid; reported for completeness, nothing is tuned on
  them (``results/npe/measured/<sample>.json``).

Usage (the commands of Laborbuch 07)::

    python scripts/run_stage7_study.py --threads 3 predict
    python scripts/run_stage7_study.py --threads 3 sbc --runs npe-nsf npe-mdn
    python scripts/run_stage7_study.py compare
    python scripts/run_stage7_study.py --threads 1 timing
    python scripts/run_stage7_study.py examples
    python scripts/run_stage7_study.py --threads 3 perturb
    python scripts/run_stage7_study.py --threads 1 measured --holograms DIR

``DIR`` is ``Examples/data/holograms`` of the Zenodo record 10.5281/zenodo.8349365. Other data
(the object model of Stage 6, runs of docs/maxwell-gpu.md): ``--npe``, ``--cnn`` and the split
options of each command; the outputs go below ``--npe`` unless ``--out`` is given. Predictions
that still match model and data are reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from holofocus.baseline.runner import file_sha256
from holofocus.dataset import DatasetConfig, read_config, read_labels
from holofocus.dataset.storage import _git_commit, write_split
from holofocus.learned import LearnedAutofocus, load_inputs, predict_split, read_predictions
from holofocus.learned.compare import read_comparison
from holofocus.learned.metrics import bootstrap_ci, interval_quantile, point_metrics
from holofocus.learned.npe import NPEAutofocus, check_npe_split, predict_npe_split
from holofocus.learned.posterior import (
    GridPosterior,
    calibration_summary,
    gaussian_hpd_level,
    gaussian_pit,
    posterior_summaries,
)
from holofocus.learned.sim2real import (
    PERTURBATIONS,
    binning_phases,
    perturbation_changes,
    perturbed_config,
)
from holofocus.measured import MEASURED_SAMPLES, load_measured

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
DATA = REPO / "data"
MAIN = "npe-nsf"
SEEDS = (MAIN, *(f"{MAIN}-seed{k}" for k in range(1, 5)))
ENSEMBLE = f"{MAIN}-ensemble"
VARIANTS = ("npe-mdn",)
CNN = RESULTS / "learned" / "spectrum-cnn-wide"
SPLITS = {
    "val": DATA / "cpu-4x" / "val.h5",
    "test": DATA / "cpu-4x" / "test.h5",
    "calibration": DATA / "cpu-4x-calibration" / "calibration.h5",
}
BASELINE_COMPARISON = RESULTS / "comparison" / "cpu-4x-test"
BINNING = 4
LEVELS = ("68", "95")
EXAMPLE_POINTS = 401


def environment() -> dict:
    import sbi

    return {
        "git_commit": _git_commit(),
        "torch_version": str(torch.__version__),
        "sbi_version": str(sbi.__version__),
        "threads": torch.get_num_threads(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
    }


def repo_relative(path: Path) -> str:
    """``path`` relative to the repository if it lies inside, so committed records stay valid."""
    path = Path(path).resolve()
    return str(path.relative_to(REPO) if path.is_relative_to(REPO) else path)


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


def model_sha256(run: Path) -> str:
    return file_sha256(run / "model.pt")


def is_current(out: Path, models: list[str], data: Path) -> bool:
    """Whether ``out`` holds the predictions of the models (SHA-256) on ``data``."""
    info = out / "run.yaml"
    if not (out / "predictions.csv").exists() or not info.exists():
        return False
    run = yaml.safe_load(info.read_text())
    stored = run.get("model_sha256")
    stored = [stored] if isinstance(stored, str) else stored
    return stored == models and (run.get("dataset") or {}).get("sha256") == file_sha256(data)


def trained(npe: Path, names) -> list[Path]:
    return [npe / name for name in names if (npe / name / "model.pt").exists()]


# --- predict -------------------------------------------------------------------------------


def npe_predict(models: list[Path], data: Path, out: Path, device: str, limit=None) -> None:
    shas = [model_sha256(m) for m in models]
    if limit is None and is_current(out, shas, data):
        print(f"{repo_relative(out)}: current")
        return
    autofocus = NPEAutofocus.load(models, device)
    predict_npe_split(autofocus, data, out, limit=limit, model_paths=models, log=print)


def cmd_predict(args) -> None:
    runs = sorted(p.parent for p in args.npe.glob("*/model.pt"))
    if args.runs:
        runs = [args.npe / r for r in args.runs]
    for run in runs:
        name = run.name
        if name.startswith("matched-"):
            root = args.data / name
            splits = {"val": root / "val.h5", "test": root / "test.h5"}
        else:
            splits = {"val": args.val, "test": args.test, "calibration": args.calibration}
        for split, data in splits.items():
            if data.exists():
                npe_predict([run], data, run / split, args.device, args.limit)
    members = trained(args.npe, SEEDS)
    if len(members) == len(SEEDS) and not args.runs:
        for split, data in (("test", args.test), ("calibration", args.calibration)):
            npe_predict(members, data, args.npe / ENSEMBLE / split, args.device, args.limit)
    elif not args.runs:
        print(f"ensemble: {len(members)} of {len(SEEDS)} seeds trained, skipped")
    out, data = args.cnn / "calibration", args.calibration
    sha = model_sha256(args.cnn)
    if args.limit is None and is_current(out, [sha], data):
        print(f"{repo_relative(out)}: current")
    else:
        cnn = LearnedAutofocus.load(args.cnn, args.device)
        predict_split(cnn, data, out, limit=args.limit, model_path=args.cnn, log=print)


def cmd_sbc(args) -> None:
    for name in args.runs:
        run = args.npe / name
        autofocus = NPEAutofocus.load(run, args.device)
        check_npe_split(
            autofocus,
            args.calibration,
            run / "calibration",
            limit=args.limit,
            num_posterior_samples=args.samples,
            model_path=run,
            log=print,
        )


# --- compare -------------------------------------------------------------------------------


def cnn_table(rows: list[dict]) -> dict[str, np.ndarray]:
    """Per-sample errors, intervals and calibration statistics of the Gaussian CNN."""
    err = np.array([r["error_um"] for r in rows])
    sigma = np.array([r["sigma"] for r in rows]) * 1e6
    table = {
        "index": np.array([r["index"] for r in rows]),
        "z01_true": np.array([r["z01_true"] for r in rows]),
        "error_um": err,
        "sigma_um": sigma,
        "pit": gaussian_pit(err, sigma),
        "hpd_level": gaussian_hpd_level(err, sigma),
        "log_density_um": -0.5 * np.log(2 * math.pi * sigma**2) - 0.5 * (err / sigma) ** 2,
        "seconds": np.array([r["seconds"] for r in rows]),
    }
    for key in LEVELS:
        half = interval_quantile(int(key) / 100) * sigma
        table[f"covered_{key}"] = np.abs(err) <= half
        table[f"width_{key}_um"] = 2 * half
    return table


def npe_table(rows: list[dict]) -> dict[str, np.ndarray]:
    """The same for NPE predictions (central intervals of the posterior)."""
    z01 = np.array([r["z01_true"] for r in rows])
    table = {
        "index": np.array([r["index"] for r in rows]),
        "z01_true": z01,
        "error_um": np.array([r["error_um"] for r in rows]),
        "sigma_um": np.array([r["sigma"] for r in rows]) * 1e6,
        "pit": np.array([r["pit"] for r in rows]),
        "hpd_level": np.array([r["hpd_level"] for r in rows]),
        "log_density_um": np.array([r["log_density_um"] for r in rows]),
        "seconds": np.array([r["seconds"] for r in rows]),
    }
    for key in LEVELS:
        lower = np.array([r[f"z01_lower_{key}"] for r in rows])
        upper = np.array([r[f"z01_upper_{key}"] for r in rows])
        table[f"covered_{key}"] = (lower <= z01) & (z01 <= upper)
        table[f"width_{key}_um"] = np.array([r[f"width_{key}_um"] for r in rows])
    return table


def restrict(table: dict[str, np.ndarray], indices) -> dict[str, np.ndarray]:
    position = {int(i): k for k, i in enumerate(table["index"])}
    missing = [i for i in indices if int(i) not in position]
    if missing:
        raise ValueError(f"samples {missing[:5]} missing")
    rows = np.array([position[int(i)] for i in indices])
    return {k: v[rows] for k, v in table.items()}


def midranks(v: np.ndarray) -> np.ndarray:
    """Ranks with tied values at the mean of their ranks; bootstrap resamples repeat rows, and
    widths tie where posteriors fill the prior box."""
    _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation (Pearson correlation of the midranks); NaN if all x or all y tie."""
    rx, ry = midranks(x), midranks(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def width_error(table: dict[str, np.ndarray]) -> dict:
    """How well the 68 % width orders the samples by their error: rank correlation of width
    and absolute error, and the MAE of the quarters with the narrowest and widest intervals."""
    width, error = table["width_68_um"], np.abs(table["error_um"])
    narrow, wide = width <= np.quantile(width, 0.25), width >= np.quantile(width, 0.75)
    return {
        "spearman": spearman(width, error),
        "mae_narrowest_quarter_um": float(error[narrow].mean()),
        "mae_widest_quarter_um": float(error[wide].mean()),
    }


def table_metrics(table: dict[str, np.ndarray]) -> dict:
    """Metrics of a method table; point metrics and cost only for methods without intervals
    (the baseline)."""
    out = {"n": len(table["index"]), "error_um": point_metrics(table["error_um"])}
    if "width_68_um" not in table:
        return {**out, "seconds_median": float(np.median(table["seconds"]))}
    out = {
        **out,
        "coverage": {k: float(table[f"covered_{k}"].mean()) for k in LEVELS},
        "width_um": {
            k: {
                "mean": float(table[f"width_{k}_um"].mean()),
                "median": float(np.median(table[f"width_{k}_um"])),
            }
            for k in LEVELS
        },
        "sigma_median_um": float(np.median(table["sigma_um"])),
        "nll": float(-table["log_density_um"].mean()),
        "width_error": width_error(table),
        "calibration": calibration_summary(table["pit"], table["hpd_level"]),
        "seconds_median": float(np.median(table["seconds"])),
    }
    return out


def paired(a: dict[str, np.ndarray], b: dict[str, np.ndarray], seed: int = 0) -> dict:
    """``a`` minus ``b`` on the same samples, with paired percentile bootstrap intervals."""
    same = np.array_equal(a["index"], b["index"])
    if not same or np.abs(a["z01_true"] - b["z01_true"]).max() > 1e-12:
        raise ValueError("the methods were not evaluated on the same samples")
    abs_a, abs_b = np.abs(a["error_um"]), np.abs(b["error_um"])
    columns = {
        "mae_um": (abs_a, abs_b),
        "within_100": (abs_a <= 100, abs_b <= 100),
        "coverage_68": (a["covered_68"], b["covered_68"]),
        "coverage_95": (a["covered_95"], b["covered_95"]),
        "nll": (-a["log_density_um"], -b["log_density_um"]),
    }
    out = {}
    for name, (x, y) in columns.items():
        pairs = np.column_stack([x, y]).astype(float)
        out[name] = {
            "difference": float(pairs[:, 0].mean() - pairs[:, 1].mean()),
            "ci": list(bootstrap_ci(pairs, lambda d: d[:, 0].mean() - d[:, 1].mean(), seed=seed)),
        }
    width = np.log(a["width_68_um"] / b["width_68_um"])
    out["width_68_ratio_median"] = float(np.exp(np.median(width)))
    out["fraction_first_better"] = float(np.mean(abs_a < abs_b))
    rows = np.column_stack([a["width_68_um"], abs_a, b["width_68_um"], abs_b])

    def gain(d: np.ndarray) -> float:
        return spearman(d[:, 0], d[:, 1]) - spearman(d[:, 2], d[:, 3])

    out["width_error_spearman"] = {
        "difference": gain(rows),
        "ci": list(bootstrap_ci(rows, gain, resamples=2000, seed=seed)),
    }
    return out


def per_sample_rows(tables: dict[str, dict], labels: dict | None) -> list[dict]:
    first = next(iter(tables.values()))
    rows = []
    for k, index in enumerate(first["index"]):
        row = {"index": int(index), "z01_true": float(first["z01_true"][k])}
        if labels is not None:
            row["phantom_type"] = int(labels["phantom_type"][index])
            row["photons"] = float(labels["photons"][index])
            row["dz01_true_um"] = float(labels["dz01"][index] * 1e6)
        for name, t in tables.items():
            row[f"{name}_error_um"] = float(t["error_um"][k])
            if "width_68_um" in t:
                for key in LEVELS:
                    row[f"{name}_width_{key}_um"] = float(t[f"width_{key}_um"][k])
                    row[f"{name}_covered_{key}"] = int(t[f"covered_{key}"][k])
                row[f"{name}_pit"] = float(t["pit"][k])
                row[f"{name}_hpd_level"] = float(t["hpd_level"][k])
        rows.append(row)
    return rows


def method_tables(npe: Path, cnn: Path, split: str) -> dict[str, dict]:
    tables = {}
    path = cnn / split / "predictions.csv"
    if path.exists():
        tables["cnn"] = cnn_table(read_predictions(path))
    for name in (MAIN, ENSEMBLE, *VARIANTS):
        path = npe / name / split / "predictions.csv"
        if path.exists():
            tables[name] = npe_table(read_predictions(path))
    return tables


def comparison_set(name: str, tables: dict[str, dict], labels: dict | None, out: Path) -> dict:
    if "cnn" not in tables or MAIN not in tables:
        raise FileNotFoundError(f"{name}: predictions of the CNN or of {MAIN} missing")
    record = {
        "set": name,
        "n_samples": len(tables[MAIN]["index"]),
        "methods": {method: table_metrics(t) for method, t in tables.items()},
        "paired": {},
    }
    for method, t in tables.items():
        if method != "cnn" and "width_68_um" in t:
            record["paired"][f"{method}_vs_cnn"] = paired(t, tables["cnn"])
        elif method.startswith("baseline"):
            record["paired"][f"{MAIN}_vs_{method}"] = {"mae_um": paired_mae(tables[MAIN], t)}
    write_json(out / f"{name}.json", record)
    write_csv(out / f"{name}.csv", per_sample_rows(tables, labels))
    return record


def paired_mae(a: dict, b: dict) -> dict:
    pairs = np.column_stack([np.abs(a["error_um"]), np.abs(b["error_um"])])
    return {
        "difference": float(pairs[:, 0].mean() - pairs[:, 1].mean()),
        "ci": list(bootstrap_ci(pairs, lambda d: d[:, 0].mean() - d[:, 1].mean())),
        "fraction_first_better": float(np.mean(pairs[:, 0] < pairs[:, 1])),
    }


def baseline_tables(path: Path, z01_true: dict[int, float]) -> dict[str, dict]:
    """Raw and offset-corrected errors of the Stage 3 baseline (``comparison.csv`` of Stage 4)."""
    rows = read_comparison(path)
    index = np.array([r["index"] for r in rows])
    z01 = np.array([z01_true[int(i)] for i in index])
    seconds = np.array([r["raw_seconds"] for r in rows], dtype=float)
    out = {}
    for name, key in (
        ("baseline_raw", "raw_error_um"),
        ("baseline_corrected", "corrected_error_um"),
    ):
        out[name] = {
            "index": index,
            "z01_true": z01,
            "error_um": np.array([r[key] for r in rows], dtype=float),
            "seconds": seconds,
        }
    return out


def summary_row(set_name: str, method: str, m: dict) -> dict:
    row = {
        "set": set_name,
        "method": method,
        "n": m["n"],
        "mae_um": m["error_um"]["mae"],
        "median_abs_um": m["error_um"]["median_abs"],
        "within_100": m["error_um"]["within"]["100"],
        "bias_um": m["error_um"]["bias"],
    }
    if "coverage" in m:
        hpd = m["calibration"]["hpd"]
        row.update(
            {
                "coverage_68": m["coverage"]["68"],
                "coverage_95": m["coverage"]["95"],
                "width_68_median_um": m["width_um"]["68"]["median"],
                "width_95_median_um": m["width_um"]["95"]["median"],
                "expected_coverage_68": hpd["coverage"]["68"],
                "expected_coverage_95": hpd["coverage"]["95"],
                "coverage_curve_max_dev": hpd["curve"]["max_abs_deviation"],
                "pit_ks_p": m["calibration"]["pit"]["ks"]["p_value"],
                "nll": m["nll"],
                "width_error_spearman": m["width_error"]["spearman"],
            }
        )
    row["seconds_median"] = m["seconds_median"]
    return row


def cmd_compare(args) -> None:
    summary_rows = []
    labels = {"test": read_labels(args.test), "calibration": read_labels(args.calibration)}
    sets = {
        "test": method_tables(args.npe, args.cnn, "test"),
        "calibration": method_tables(args.npe, args.cnn, "calibration"),
    }
    baseline_csv = args.baseline / "comparison.csv"
    if baseline_csv.exists():
        indices = json.loads((args.baseline / "comparison.json").read_text())["indices"]
        z01 = dict(enumerate(labels["test"]["z01"]))
        restricted = {m: restrict(t, indices) for m, t in sets["test"].items()}
        sets[f"baseline-{len(indices)}"] = {**restricted, **baseline_tables(baseline_csv, z01)}
    records = {}
    for name, tables in sets.items():
        if not tables:
            continue
        split_labels = labels["calibration" if name == "calibration" else "test"]
        records[name] = comparison_set(name, tables, split_labels, args.out)
        for method, metrics in records[name]["methods"].items():
            summary_rows.append(summary_row(name, method, metrics))
        main, cnn = records[name]["methods"][MAIN], records[name]["methods"]["cnn"]
        d = records[name]["paired"][f"{MAIN}_vs_cnn"]
        print(
            f"{name}: NPE MAE {main['error_um']['mae']:.0f} um (CNN {cnn['error_um']['mae']:.0f}; "
            f"difference {d['mae_um']['difference']:+.0f} [{d['mae_um']['ci'][0]:+.0f}, "
            f"{d['mae_um']['ci'][1]:+.0f}]), within 100 um {main['error_um']['within']['100']:.3f} "
            f"({cnn['error_um']['within']['100']:.3f}), coverage 68/95 % "
            f"{main['coverage']['68']:.3f}/{main['coverage']['95']:.3f} "
            f"({cnn['coverage']['68']:.3f}/{cnn['coverage']['95']:.3f}), NLL {main['nll']:.3f} "
            f"({cnn['nll']:.3f})"
        )
    write_csv(args.out / "comparison.csv", summary_rows)
    write_json(
        args.out / "sources.json",
        {
            "npe": repo_relative(args.npe),
            "cnn": repo_relative(args.cnn),
            "baseline": repo_relative(args.baseline),
            "test": repo_relative(args.test),
            "calibration": repo_relative(args.calibration),
            "sets": {name: sorted(tables) for name, tables in sets.items()},
            **environment(),
        },
    )


# --- timing --------------------------------------------------------------------------------


def _seconds(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def npe_parts(autofocus: NPEAutofocus, hologram: torch.Tensor) -> dict[str, float]:
    """Seconds of the parts of one NPE estimate (single model)."""
    estimator = autofocus.estimators[0]
    with torch.inference_mode():
        start = time.perf_counter()
        x = autofocus.representation(hologram)
        t1 = time.perf_counter()
        e = estimator.embed(x.to(torch.float32))
        t2 = time.perf_counter()
        grid = autofocus.theta[:, None, None].expand(len(autofocus.theta), 1, 1)
        log_q = estimator.net.log_prob(grid, e).T.double().cpu().numpy()
        t3 = time.perf_counter()
        post = GridPosterior.from_log_density(autofocus.theta_grid, log_q)
        posterior_summaries(post, autofocus.config.target_scale)
        t4 = time.perf_counter()
    return {
        "representation": t1 - start,
        "embedding": t2 - t1,
        "density_on_grid": t3 - t2,
        "summaries": t4 - t3,
    }


def cmd_timing(args) -> None:
    with h5py.File(args.test, "r") as fh:
        holograms = [torch.from_numpy(np.asarray(fh["hologram"][i])) for i in range(args.samples)]
    methods = {"cnn": LearnedAutofocus.load(args.cnn, args.device)}
    for name, members in ((MAIN, [args.npe / MAIN]), (ENSEMBLE, trained(args.npe, SEEDS))):
        if members and (name != ENSEMBLE or len(members) == len(SEEDS)):
            methods[name] = NPEAutofocus.load(members, args.device)
    for name in VARIANTS:
        if (args.npe / name / "model.pt").exists():
            methods[name] = NPEAutofocus.load(args.npe / name, args.device)
    for autofocus in methods.values():
        autofocus.estimate(holograms[0])
    seconds = {name: [] for name in methods}
    parts = {k: [] for k in ("representation", "embedding", "density_on_grid", "summaries")}
    order = list(methods)
    for repeat in range(args.repeats):
        for k, hologram in enumerate(holograms):
            shift = (repeat + k) % len(order)
            for name in order[shift:] + order[:shift]:
                seconds[name].append(_seconds(methods[name].estimate, hologram))
            for part, value in npe_parts(methods[MAIN], hologram).items():
                parts[part].append(value)
    record = {
        "data": repo_relative(args.test),
        "samples": args.samples,
        "repeats": args.repeats,
        "device": args.device,
        "what": "one estimate from the hologram in memory, batch of one: representation, "
        "network, for the NPE the density on the grid, median and intervals",
        "methods": {},
        "npe_parts_median": {part: float(np.median(v)) for part, v in parts.items()},
        "load_average": list(os.getloadavg()),
        **environment(),
    }
    for name, values in seconds.items():
        v = np.array(values)
        m = record["methods"][name] = {
            "median": float(np.median(v)),
            "mean": float(v.mean()),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
            "ratio_to_cnn": float(np.median(v) / np.median(seconds["cnn"])),
        }
        print(
            f"{name}: {1e3 * m['median']:.2f} ms per estimate (p10 {1e3 * m['p10']:.2f}, "
            f"p90 {1e3 * m['p90']:.2f}), {m['ratio_to_cnn']:.2f} x CNN"
        )
    print(
        "NPE parts (ms): "
        + ", ".join(f"{k} {1e3 * v:.2f}" for k, v in record["npe_parts_median"].items())
    )
    write_json(args.out, record)


# --- examples ------------------------------------------------------------------------------


def cmd_examples(args) -> None:
    """Posterior densities (per mm of dz01) of the NPEs and the CNN Gaussian on the test split:
    the samples ``--indices`` (chosen before looking at the results) and the ``--worst`` largest
    errors of the main NPE."""
    cnn_rows = {r["index"]: r for r in read_predictions(args.cnn / "test" / "predictions.csv")}
    main_csv = args.npe / MAIN / "test" / "predictions.csv"
    main_rows = {r["index"]: r for r in read_predictions(main_csv)}
    worst = sorted(main_rows, key=lambda i: -main_rows[i]["abs_error_um"])
    extra = [i for i in worst if i not in args.indices][: args.worst]
    chosen = [*args.indices, *extra]
    models = {MAIN: [args.npe / MAIN]}
    members = trained(args.npe, SEEDS)
    if len(members) == len(SEEDS):
        models[ENSEMBLE] = members
    labels = read_labels(args.test)
    samples = {}
    for i in chosen:
        cnn = cnn_rows[i]
        samples[str(i)] = {
            "dz01_true_mm": float(labels["dz01"][i] * 1e3),
            "phantom_type": int(labels["phantom_type"][i]),
            "photons": float(labels["photons"][i]),
            "cnn": {"dz01_mm": cnn["dz01_est"] * 1e3, "sigma_mm": cnn["sigma"] * 1e3},
        }
    for name, paths in models.items():
        autofocus = NPEAutofocus.load(paths, args.device)
        scale = autofocus.config.target_scale
        inputs = load_inputs(args.test, autofocus.representation, scale, indices=chosen).inputs
        post = autofocus.posterior_of_inputs(inputs)
        median = posterior_summaries(post, scale)["dz01"]
        step = max(1, (len(autofocus.theta_grid) - 1) // (EXAMPLE_POINTS - 1))
        dz01_mm = (autofocus.theta_grid[::step] * scale * 1e3).tolist()
        for k, i in enumerate(chosen):
            samples[str(i)][name] = {
                "dz01_median_mm": float(median[k] * 1e3),
                "density_per_mm": (post.density[k, ::step] / (scale * 1e3)).tolist(),
            }
    record = {
        "data": repo_relative(args.test),
        "a_priori": list(args.indices),
        "largest_error": extra,
        "dz01_mm": dz01_mm,
        "samples": samples,
    }
    write_json(args.out, record)
    print(f"posteriors of {len(chosen)} test samples -> {repo_relative(args.out)}")


# --- perturb -------------------------------------------------------------------------------


def _run_info(directory: Path) -> dict:
    path = directory / "run.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def set_is_current(path: Path, config: DatasetConfig, n: int) -> bool:
    return (
        path.exists()
        and read_config(path).to_dict() == config.to_dict()
        and len(read_labels(path)["dz01"]) >= n
    )


def widening(reference: dict, perturbed: dict) -> dict:
    """Per-sample change of the 68 % width and of the coverage (perturbed against reference)."""
    position = {int(i): k for k, i in enumerate(reference["index"])}
    pairs = [(position[int(i)], k) for k, i in enumerate(perturbed["index"]) if int(i) in position]
    ref, pert = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    ratio = perturbed["width_68_um"][pert] / reference["width_68_um"][ref]
    return {
        "n": len(pairs),
        "width_68_ratio": {
            "median": float(np.median(ratio)),
            "geometric_mean": float(np.exp(np.mean(np.log(ratio)))),
            "geometric_mean_ci": [float(np.exp(v)) for v in bootstrap_ci(np.log(ratio), np.mean)],
            "fraction_wider": float(np.mean(ratio > 1)),
        },
        "coverage_change": {
            k: float(perturbed[f"covered_{k}"][pert].mean() - reference[f"covered_{k}"][ref].mean())
            for k in LEVELS
        },
    }


def paired_common(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict:
    """:func:`paired` on the samples that both tables hold."""
    common = np.intersect1d(a["index"], b["index"])
    return paired(restrict(a, common), restrict(b, common))


def robustness_row(sample: str, pert: str, method: str, m: dict, w: dict | None) -> dict:
    row = {
        "sample": sample,
        "perturbation": pert,
        "method": method,
        "n": m["n"],
        "mae_um": m["error_um"]["mae"],
        "within_100": m["error_um"]["within"]["100"],
        "coverage_68": m["coverage"]["68"],
        "coverage_95": m["coverage"]["95"],
        "width_68_median_um": m["width_um"]["68"]["median"],
        "width_95_median_um": m["width_um"]["95"]["median"],
        "expected_coverage_68": m["calibration"]["hpd"]["coverage"]["68"],
        "nll": m["nll"],
        "width_68_ratio_median": 1.0 if w is None else w["width_68_ratio"]["median"],
        "fraction_wider": 0.0 if w is None else w["width_68_ratio"]["fraction_wider"],
    }
    return row


def cmd_perturb(args) -> None:
    for name in args.samples:
        model = args.npe / f"matched-{name}"
        data = args.data / f"matched-{name}" / "test.h5"
        config = read_config(data)
        n = len(read_labels(data)["dz01"]) if args.limit is None else args.limit
        npe_predict([model], data, model / "test", args.device, args.limit)
        reference = npe_table(read_predictions(model / "test" / "predictions.csv"))
        cnn_reference_path = args.cnn_learned / f"matched-{name}" / "test" / "predictions.csv"
        cnn_reference = cnn_table(
            [r for r in read_predictions(cnn_reference_path) if r["index"] < n]
        )
        cnn_model = _run_info(cnn_reference_path.parent).get("model_sha256")
        record = {
            "sample": name,
            "model": repo_relative(model),
            "model_sha256": model_sha256(model),
            "data": repo_relative(data),
            "cnn_sim2real": repo_relative(args.cnn_sim2real),
            "reference": {MAIN: table_metrics(reference), "cnn": table_metrics(cnn_reference)},
            "perturbations": {},
            "paired_vs_cnn": {"none": paired_common(reference, cnn_reference)},
            **environment(),
        }
        rows = [
            robustness_row(name, "none", "npe", record["reference"][MAIN], None),
            robustness_row(name, "none", "cnn", record["reference"]["cnn"], None),
        ]
        for pert in args.perturbations:
            sets = args.sets / name / f"{pert}.h5"
            changes = perturbation_changes(pert, name)
            cfg = perturbed_config(config, changes, f"{config.name}-{pert}")
            if not set_is_current(sets, cfg, n):
                start = time.perf_counter()
                write_split(cfg, "test", sets, n_samples=n, device="cpu", log=None)
                print(f"{name} {pert}: {n} samples in {time.perf_counter() - start:.0f} s")
            out = args.out / name / pert
            npe_predict([model], sets, out, args.device, args.limit)
            perturbed = npe_table(read_predictions(out / "predictions.csv"))
            entry = {
                "description": PERTURBATIONS[pert],
                "changes": changes,
                "data_sha256": file_sha256(sets),
                MAIN: {**table_metrics(perturbed), "widening": widening(reference, perturbed)},
            }
            rows.append(robustness_row(name, pert, "npe", entry[MAIN], entry[MAIN]["widening"]))
            cnn_out = args.cnn_sim2real / name / pert
            cnn_info = _run_info(cnn_out)
            same_data = (cnn_info.get("dataset") or {}).get("sha256") == entry["data_sha256"]
            if same_data and cnn_info.get("model_sha256") == cnn_model:
                cnn_rows = [
                    r for r in read_predictions(cnn_out / "predictions.csv") if r["index"] < n
                ]
                cnn_pert = cnn_table(cnn_rows)
                entry["cnn"] = {
                    **table_metrics(cnn_pert),
                    "widening": widening(cnn_reference, cnn_pert),
                }
                record["paired_vs_cnn"][pert] = paired_common(perturbed, cnn_pert)
                rows.append(
                    robustness_row(name, pert, "cnn", entry["cnn"], entry["cnn"]["widening"])
                )
            else:
                entry["cnn"] = None
                print(
                    f"{name} {pert}: no predictions of the reference CNN on this set in "
                    f"{repo_relative(cnn_out)}"
                )
            record["perturbations"][pert] = entry
            parts = []
            for method in (MAIN, "cnn"):
                m = entry[method]
                if m is not None:
                    ratio = m["widening"]["width_68_ratio"]["median"]
                    parts.append(
                        f"{method} coverage 68/95 {m['coverage']['68']:.2f}/"
                        f"{m['coverage']['95']:.2f}, width68 x{ratio:.2f}"
                    )
            print(f"{name} {pert}: " + "; ".join(parts))
        write_json(args.out / f"{name}.json", record)
        write_csv(args.out / f"{name}.csv", rows)


# --- measured ------------------------------------------------------------------------------


def measured_references(name: str, measured: Path) -> dict[str, float]:
    sample = MEASURED_SAMPLES[name]
    refs = {
        "paper2_nm": sample.z01_nm,
        "paper2_visual": sample.z01_visual,
        "paper1_example": sample.z01_example,
    }
    for key, run in (("baseline_b1_p4", "b1-p4"), ("baseline_b1_p2", "b1-p2"), ("cnn", "cnn")):
        path = measured / run / f"{name}.json"
        if path.exists():
            record = json.loads(path.read_text())
            refs[key] = float(record["z01_est"] if key == "cnn" else record["result"]["z01"])
    return refs


def cmd_measured(args) -> None:
    for name in args.samples:
        sample = MEASURED_SAMPLES[name]
        run = args.npe / f"matched-{name}"
        autofocus = NPEAutofocus.load(run, args.device)
        hologram = load_measured(args.holograms, sample, BINNING)
        if tuple(hologram.shape) != autofocus.fov_shape:
            raise ValueError(
                f"{name}: hologram {tuple(hologram.shape)}, model {autofocus.fov_shape}"
            )
        scale = autofocus.config.target_scale
        autofocus.estimate(hologram)
        seconds = [_seconds(autofocus.estimate, hologram) for _ in range(args.repeats)]
        post = autofocus.posterior(hologram)
        est = {k: float(v[0]) for k, v in posterior_summaries(post, scale).items()}
        z01 = sample.z01_guess + est["dz01"]
        intervals = {
            key: {
                "lower": sample.z01_guess + est[f"lower_{key}"],
                "upper": sample.z01_guess + est[f"upper_{key}"],
                "width_um": (est[f"upper_{key}"] - est[f"lower_{key}"]) * 1e6,
                "hpd_width_um": est[f"hpd_width_{key}"] * 1e6,
            }
            for key in LEVELS
        }
        references = {}
        for ref, value in measured_references(name, args.measured).items():
            t = np.array([(value - sample.z01_guess) / scale])
            references[ref] = {
                "z01": value,
                "distance_um": (z01 - value) * 1e6,
                "pit": float(post.pit(t)[0]),
                "hpd_level": float(post.hpd_level(t)[0]),
                "inside": {
                    k: bool(i["lower"] <= value <= i["upper"]) for k, i in intervals.items()
                },
            }
        native = load_measured(args.holograms, sample, 1).numpy()
        phases = torch.from_numpy(
            np.stack([p.astype(np.float32) for p in binning_phases(native, BINNING)])
        )
        phase_post = autofocus.posterior(phases)
        phase_z01 = sample.z01_guess + posterior_summaries(phase_post, scale)["dz01"]
        step = max(1, (len(autofocus.theta_grid) - 1) // (EXAMPLE_POINTS - 1))
        lower, upper = sample.simplex
        record = {
            "sample": name,
            "label": sample.label,
            "model": repo_relative(run),
            "model_sha256": model_sha256(run),
            "binning": BINNING,
            "z01_guess": sample.z01_guess,
            "z01_est": z01,
            "dz01_est": est["dz01"],
            "z01_mean": sample.z01_guess + est["dz01_mean"],
            "z01_mode": sample.z01_guess + est["dz01_mode"],
            "sigma": est["sigma"],
            "prior_mass": est["prior_mass"],
            "inside_omega": bool(lower <= z01 <= upper),
            "intervals": intervals,
            "references": references,
            "binning_phases": {
                "n": len(phase_z01),
                "z01_est": [float(v) for v in phase_z01],
                "std_um": float(np.std(phase_z01, ddof=1) * 1e6),
                "range_um": float((phase_z01.max() - phase_z01.min()) * 1e6),
            },
            "posterior": {
                "z01_mm": (
                    (sample.z01_guess + autofocus.theta_grid[::step] * scale) * 1e3
                ).tolist(),
                "density_per_mm": (post.density[0, ::step] / (scale * 1e3)).tolist(),
            },
            "seconds_per_estimate": {
                "median": float(np.median(seconds)),
                "min": float(np.min(seconds)),
                "repeats": args.repeats,
            },
            "note": "for completeness only: no model, prior or setting was chosen on this hologram",
            **environment(),
        }
        write_json(args.out / f"{name}.json", record)
        ref = references["paper2_nm"]
        print(
            f"{name}: NPE z01 = {z01 * 1e3:.3f} mm, 68 % [{intervals['68']['lower'] * 1e3:.3f}, "
            f"{intervals['68']['upper'] * 1e3:.3f}] mm, minus [2] {ref['distance_um']:+.0f} um "
            f"(HPD level of [2] {ref['hpd_level']:.2f}), phases std "
            f"{record['binning_phases']['std_um']:.0f} um"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, help="torch CPU threads")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or auto")
    parser.add_argument("--npe", type=Path, default=RESULTS / "npe", help="NPE runs")
    parser.add_argument("--cnn", type=Path, default=CNN, help="Gaussian CNN run of Stage 4")
    sub = parser.add_subparsers(dest="command", required=True)
    samples = {"nargs": "+", "choices": list(MEASURED_SAMPLES), "default": list(MEASURED_SAMPLES)}

    def splits(p, *names):
        for name in names:
            p.add_argument(f"--{name}", type=Path, default=SPLITS[name], help=f"{name} split")

    pr = sub.add_parser("predict", help="NPEs, ensemble and CNN on the splits")
    pr.add_argument("--runs", nargs="+", help="run names below --npe (default: all)")
    pr.add_argument("--data", type=Path, default=DATA, help="root of the matched datasets")
    pr.add_argument("--limit", type=int, help="only the first N samples (not stored as current)")
    splits(pr, "val", "test", "calibration")
    pr.set_defaults(func=cmd_predict)

    sb = sub.add_parser(
        "sbc", help="sbi's SBC, expected coverage and TARP on the calibration split"
    )
    sb.add_argument("--runs", nargs="+", default=[MAIN])
    sb.add_argument("--samples", type=int, default=1000, help="posterior samples per input")
    sb.add_argument("--limit", type=int)
    splits(sb, "calibration")
    sb.set_defaults(func=cmd_sbc)

    def output(p, default: str):
        p.add_argument("--out", type=Path, help=f"default: <npe>/{default}")
        p.set_defaults(out_default=default)

    co = sub.add_parser("compare", help="NPE against the CNN (and the baseline on its samples)")
    co.add_argument("--baseline", type=Path, default=BASELINE_COMPARISON, help="Stage 4 comparison")
    output(co, "comparison")
    splits(co, "test", "calibration")
    co.set_defaults(func=cmd_compare)

    ti = sub.add_parser("timing", help="time per estimate, CNN against NPE")
    ti.add_argument("--samples", type=int, default=50, help="test holograms")
    ti.add_argument("--repeats", type=int, default=4)
    output(ti, "timing.json")
    splits(ti, "test")
    ti.set_defaults(func=cmd_timing)

    ex = sub.add_parser("examples", help="posterior densities of a few test samples")
    ex.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ex.add_argument("--worst", type=int, default=2, help="plus the N largest NPE errors")
    output(ex, "comparison/posteriors.json")
    splits(ex, "test")
    ex.set_defaults(func=cmd_examples)

    pe = sub.add_parser("perturb", help="matched NPEs on the perturbed test splits of Stage 5")
    pe.add_argument("--samples", **samples)
    pe.add_argument(
        "--perturbations", nargs="+", choices=list(PERTURBATIONS), default=list(PERTURBATIONS)
    )
    pe.add_argument("--limit", type=int, help="only the first N test samples")
    pe.add_argument("--data", type=Path, default=DATA, help="root of the matched datasets")
    pe.add_argument("--sets", type=Path, default=DATA / "sim2real", help="perturbed splits")
    pe.add_argument("--cnn-learned", type=Path, default=RESULTS / "learned", help="matched CNNs")
    pe.add_argument(
        "--cnn-sim2real", type=Path, default=RESULTS / "sim2real", help="CNN on the sets"
    )
    output(pe, "sim2real")
    pe.set_defaults(func=cmd_perturb)

    me = sub.add_parser("measured", help="matched NPEs on the measured holograms")
    me.add_argument("--holograms", type=Path, required=True)
    me.add_argument("--samples", **samples)
    me.add_argument("--repeats", type=int, default=20, help="timed estimates")
    me.add_argument("--measured", type=Path, default=RESULTS / "measured", help="Stage 3/5 records")
    output(me, "measured")
    me.set_defaults(func=cmd_measured)

    args = parser.parse_args()
    if getattr(args, "out_default", None) and args.out is None:
        args.out = args.npe / args.out_default
    if args.threads:
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
