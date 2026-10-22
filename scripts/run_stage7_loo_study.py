"""Stage 7 on the object model of Stage 6: the NPE of Laborbuch 07 on the leave-one-out sets.

The runs are ``holofocus-learn npe-train configs/learned/npe-nsf.yaml`` on
``data/loo-<sample>/{train,val}.h5``, below ``results/npe-loo``: ``matched-<sample>`` with the
seed of the config (0) and ``matched-<sample>-seed<k>`` with ``--seed k``, the seeds of the
leave-one-out CNNs of Stage 6 (the names of ``run_stage7_study.py`` and of the job array of
``cluster/maxwell/jobs/npe.sh``).

* ``predict``: every run on the validation and test split of its leave-one-out set, and the
  ensemble of the seeds of a sample (the mean of their densities) on the test split
  (``matched-<sample>-ensemble/test``).
* ``compare``: on the leave-one-out test splits, the NPE seeds, their ensemble, the
  leave-one-out CNNs of Stage 6 (``results/learned/loo-<sample>``, seeds in
  ``results/objectmodel/seeds``) and the guess, whose intervals are those of the uniform prior:
  errors, coverage and width of the central 68/95 % intervals, NLL, PIT and HPD statistics and
  paired differences with bootstrap intervals (``comparison/<sample>.{json,csv}``,
  ``comparison/comparison.csv``). Reads only predictions, so it runs on the committed results.
* ``measured``: the seeds and the ensemble on the four measured holograms, binned 4x as in
  Stages 3, 5 and 6, with the 16 phases of the binning grid: estimate, central intervals, [2]
  Table 1, our padding-4 search and the leave-one-out CNN (``results/measured/cnn-loo``)
  against them, the distance from the guess and the modes of the posterior
  (``measured/<sample>.json``). Nothing is tuned on them.
* ``timing``: time per estimate of the leave-one-out CNN, one NPE and the ensemble on the same
  test holograms of every sample, alternating, at the same thread count (``timing.json``).

Usage (Laborbuch 07, Abschnitt 9)::

    python scripts/run_stage7_loo_study.py --threads 3 predict
    python scripts/run_stage7_loo_study.py compare
    python scripts/run_stage7_loo_study.py --threads 1 measured --holograms DIR
    python scripts/run_stage7_loo_study.py --threads 1 timing

``DIR`` is ``Examples/data/holograms`` of the Zenodo record 10.5281/zenodo.8349365. Other runs
(the large sets of ``docs/maxwell-gpu.md``, 8.2): ``--npe``, ``--cnn`` and ``--cnn-seeds``, and
``--seeds`` and ``--samples`` after the command; the outputs go below ``--npe`` unless ``--out``
is given. Run it from the root of the repository, so the paths in the records stay relative to
it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from holofocus.learned import LearnedAutofocus, read_predictions
from holofocus.learned.npe import NPEAutofocus
from holofocus.learned.posterior import GridPosterior, posterior_summaries
from holofocus.learned.sim2real import binning_phases
from holofocus.measured import MEASURED_SAMPLES, load_measured
from run_stage7_study import (
    BINNING,
    EXAMPLE_POINTS,
    LEVELS,
    cnn_table,
    environment,
    model_sha256,
    npe_parts,
    npe_table,
    paired,
    paired_mae,
    per_sample_rows,
    repo_relative,
    table_metrics,
    write_csv,
    write_json,
)
from run_stage7_study import npe_predict as stage7_npe_predict

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
DATA = REPO / "data"
NPE = RESULTS / "npe-loo"
CNN = RESULTS / "learned"
CNN_SEEDS = RESULTS / "objectmodel" / "seeds"
SEEDS = (0, 1, 2)
MODE_FLOOR = 0.1
EDGE_MM = 0.5


def run_name(sample: str, seed: int) -> str:
    return f"matched-{sample}" if seed == 0 else f"matched-{sample}-seed{seed}"


def ensemble_name(sample: str) -> str:
    return f"matched-{sample}-ensemble"


def method_name(kind: str, seed: int) -> str:
    """``npe``/``cnn`` for seed 0, ``npe-seed<k>``/``cnn-seed<k>`` for the others."""
    return kind if seed == 0 else f"{kind}-seed{seed}"


def cnn_run(args, sample: str, seed: int) -> Path:
    if seed == 0:
        return args.cnn / f"loo-{sample}"
    return args.cnn_seeds / f"seed-{seed}" / f"loo-{sample}"


def members(args, sample: str) -> list[Path]:
    """The trained runs of ``sample`` among ``--seeds``, seed 0 first."""
    runs = [args.npe / run_name(sample, k) for k in args.seeds]
    return [run for run in runs if (run / "model.pt").exists()]


def recorded(path: Path) -> Path:
    """``path`` relative to the repository when run from its root, so the records stay valid."""
    return Path(repo_relative(path)) if Path.cwd().resolve() == REPO else Path(path)


def seconds_of(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def read_run(directory: Path) -> dict:
    path = directory / "run.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}


# --- predict -------------------------------------------------------------------------------


def npe_predict(models: list[Path], data: Path, out: Path, device: str, limit=None) -> None:
    stage7_npe_predict([recorded(m) for m in models], recorded(data), recorded(out), device, limit)


def cmd_predict(args) -> None:
    for sample in args.samples:
        root = args.data / f"loo-{sample}"
        runs = members(args, sample)
        if not runs:
            print(f"{sample}: no trained run below {repo_relative(args.npe)}")
            continue
        for run in runs:
            for split in ("val", "test"):
                npe_predict([run], root / f"{split}.h5", run / split, args.device, args.limit)
        if len(runs) > 1:
            out = args.npe / ensemble_name(sample) / "test"
            npe_predict(runs, root / "test.h5", out, args.device, args.limit)
        else:
            print(f"{sample}: ensemble skipped, {len(runs)} of {len(args.seeds)} seeds trained")


# --- compare -------------------------------------------------------------------------------


def guess_table(rows: list[dict]) -> dict[str, np.ndarray]:
    """The guess as the estimate: the median of the uniform prior around it."""
    z01 = np.array([r["z01_true"] for r in rows])
    return {
        "index": np.array([r["index"] for r in rows]),
        "z01_true": z01,
        "error_um": (np.array([r["z01_guess"] for r in rows]) - z01) * 1e6,
        "seconds": np.zeros(len(rows)),
    }


def prior_metrics(rows: list[dict], prior, scale: float) -> dict:
    """Coverage and width of the central intervals of the uniform prior of the NPE (centred on
    the guess) and its NLL: what a method knows without looking at the hologram."""
    lo, hi = (float(v) * scale for v in prior)
    dz01 = np.array([r["z01_true"] - r["z01_guess"] for r in rows])
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    out = {"coverage": {}, "width_um": {}, "nll": math.log((hi - lo) * 1e6)}
    for key in LEVELS:
        level = int(key) / 100
        out["coverage"][key] = float(np.mean(np.abs(dz01 - centre) <= level * half))
        out["width_um"][key] = 2 * level * half * 1e6
    return out


def dataset_sha256(predictions: Path) -> str | None:
    return (read_run(predictions.parent).get("dataset") or {}).get("sha256")


def sample_tables(args, sample: str) -> tuple[dict, dict, list[dict]]:
    """Tables of every method found for ``sample`` on its leave-one-out test split, the paths
    of their predictions and the rows of the first NPE."""
    paths = {}
    for k in args.seeds:
        paths[method_name("cnn", k)] = cnn_run(args, sample, k) / "test" / "predictions.csv"
    for k in args.seeds:
        paths[method_name("npe", k)] = args.npe / run_name(sample, k) / "test" / "predictions.csv"
    paths["ensemble"] = args.npe / ensemble_name(sample) / "test" / "predictions.csv"
    paths = {name: path for name, path in paths.items() if path.exists()}
    tables, npe_rows = {}, None
    for name, path in paths.items():
        rows = read_predictions(path)
        if name.startswith("cnn"):
            tables[name] = cnn_table(rows)
        else:
            tables[name] = npe_table(rows)
            npe_rows = npe_rows or rows
    return tables, paths, npe_rows


def summary_row(sample: str, method: str, m: dict, prior: dict | None = None) -> dict:
    row = {
        "sample": sample,
        "method": method,
        "n": m["n"],
        "mae_um": m["error_um"]["mae"],
        "median_abs_um": m["error_um"]["median_abs"],
        "within_100": m["error_um"]["within"]["100"],
        "within_1000": m["error_um"]["within"]["1000"],
        "bias_um": m["error_um"]["bias"],
    }
    source = m if "coverage" in m else prior
    row.update(
        {
            "coverage_68": source["coverage"]["68"],
            "coverage_95": source["coverage"]["95"],
            "width_68_median_um": (
                m["width_um"]["68"]["median"] if "coverage" in m else prior["width_um"]["68"]
            ),
            "width_95_median_um": (
                m["width_um"]["95"]["median"] if "coverage" in m else prior["width_um"]["95"]
            ),
            "nll": source["nll"],
        }
    )
    if "coverage" in m:
        row["expected_coverage_68"] = m["calibration"]["hpd"]["coverage"]["68"]
        row["pit_ks_p"] = m["calibration"]["pit"]["ks"]["p_value"]
        row["width_error_spearman"] = m["width_error"]["spearman"]
    else:
        row["expected_coverage_68"] = row["pit_ks_p"] = row["width_error_spearman"] = ""
    return row


def seed_spread(metrics: dict, kind: str) -> dict | None:
    """Range over the seeds of a method: MAE, coverage and median 68 % width."""
    names = [n for n in metrics if n == kind or n.startswith(f"{kind}-seed")]
    if len(names) < 2:
        return None
    values = {
        "mae_um": [metrics[n]["error_um"]["mae"] for n in names],
        "coverage_68": [metrics[n]["coverage"]["68"] for n in names],
        "coverage_95": [metrics[n]["coverage"]["95"] for n in names],
        "width_68_median_um": [metrics[n]["width_um"]["68"]["median"] for n in names],
    }
    return {
        "methods": names,
        **{k: {"values": v, "min": min(v), "max": max(v)} for k, v in values.items()},
    }


def cmd_compare(args) -> None:
    summary = []
    sources = {}
    for sample in args.samples:
        tables, paths, npe_rows = sample_tables(args, sample)
        if "npe" not in tables or "cnn" not in tables:
            print(f"{sample}: predictions of the NPE (seed 0) or of the CNN missing, skipped")
            continue
        info = read_run(paths["npe"].parent)
        prior = prior_metrics(npe_rows, info["prior"], info["config"]["target_scale"])
        tables["guess"] = guess_table(npe_rows)
        metrics = {name: table_metrics(t) for name, t in tables.items()}
        shas = {name: dataset_sha256(path) for name, path in paths.items()}
        if len(set(shas.values())) > 1:
            print(f"{sample}: predictions on files with different SHA-256 (same labels): {shas}")
        record = {
            "sample": sample,
            "n_samples": len(tables["npe"]["index"]),
            "dataset_sha256": shas,
            "predictions": {name: repo_relative(path) for name, path in paths.items()},
            "prior": prior,
            "methods": metrics,
            "paired": {},
            "seeds": {kind: seed_spread(metrics, kind) for kind in ("npe", "cnn")},
        }
        for name in tables:
            if name in ("guess", "cnn") or name.startswith("cnn-seed"):
                continue
            seed = name.removeprefix("npe-seed") if name.startswith("npe-seed") else None
            cnn = "cnn" if seed is None else f"cnn-seed{seed}"
            if cnn in tables:
                record["paired"][f"{name}_vs_{cnn}"] = paired(tables[name], tables[cnn])
        for name in ("npe", "ensemble", "cnn"):
            if name in tables:
                record["paired"][f"{name}_vs_guess"] = {
                    "mae_um": paired_mae(tables[name], tables["guess"])
                }
        write_json(args.out / f"{sample}.json", record)
        write_csv(args.out / f"{sample}.csv", per_sample_rows(tables, None))
        sources[sample] = record["predictions"]
        for name, m in metrics.items():
            summary.append(summary_row(sample, name, m, prior if name == "guess" else None))
            cov = m.get("coverage", prior["coverage"])
            print(
                f"{sample} {name}: MAE {m['error_um']['mae']:.0f} um, coverage 68/95 % "
                f"{cov['68']:.3f}/{cov['95']:.3f}"
            )
        for name in ("npe", "ensemble"):
            d = record["paired"].get(f"{name}_vs_cnn")
            if d:
                print(
                    f"{sample} {name} - cnn: MAE {d['mae_um']['difference']:+.0f} um "
                    f"[{d['mae_um']['ci'][0]:+.0f}, {d['mae_um']['ci'][1]:+.0f}]"
                )
    if not summary:
        raise SystemExit("no sample with predictions of the NPE and the CNN")
    write_csv(args.out / "comparison.csv", summary)
    write_json(args.out / "sources.json", {"samples": sources})


# --- measured ------------------------------------------------------------------------------


def references(name: str, args) -> dict[str, float]:
    """``z01`` of [2] Table 1, [1], our padding-4 search, the leave-one-out CNN of Stage 6, the
    matched CNN of Stage 5 and the matched NPE of Stage 7, where their records exist."""
    sample = MEASURED_SAMPLES[name]
    refs = {
        "paper2_nm": sample.z01_nm,
        "paper2_visual": sample.z01_visual,
        "paper1_example": sample.z01_example,
    }
    for key, path, field in (
        ("baseline_b1_p4", args.measured / "b1-p4" / f"{name}.json", ("result", "z01")),
        ("cnn_loo", args.cnn_measured / f"{name}.json", ("z01_est",)),
        ("cnn_stage5", args.measured / "cnn" / f"{name}.json", ("z01_est",)),
        ("npe_stage7", args.npe_stage7 / f"{name}.json", ("z01_est",)),
    ):
        if path.exists():
            value = json.loads(path.read_text())
            for part in field:
                value = value[part]
            refs[key] = float(value)
    return refs


def modes(post: GridPosterior, floor: float = MODE_FLOOR) -> list[dict]:
    """Local maxima of the density of the first row whose prominence is at least ``floor``
    times the highest density, highest first; ``t`` units. The density is taken as zero outside
    the prior box, so density that still rises at an edge counts as a mode there."""
    d = np.concatenate([[0.0], post.density[0], [0.0]])
    peak = d.max()
    found = []
    for i in range(1, len(d) - 1):
        if not (d[i] > d[i - 1] and d[i] >= d[i + 1]):
            continue
        left = i
        while left > 0 and d[left - 1] <= d[i]:
            left -= 1
        right = i
        while right < len(d) - 1 and d[right + 1] <= d[i]:
            right += 1
        base = max(d[left : i + 1].min(), d[i : right + 1].min())
        if d[i] - base >= floor * peak:
            found.append(
                {
                    "t": float(post.theta[i - 1]),
                    "relative_height": float(d[i] / peak),
                    "relative_prominence": float((d[i] - base) / peak),
                }
            )
    return sorted(found, key=lambda m: -m["relative_height"])


def posterior_record(
    autofocus: NPEAutofocus,
    runs: list[Path],
    hologram: torch.Tensor,
    phases: torch.Tensor,
    name: str,
    refs: dict[str, float],
    repeats: int,
) -> dict:
    sample = MEASURED_SAMPLES[name]
    guess, scale = sample.z01_guess, autofocus.config.target_scale
    autofocus.estimate(hologram)
    seconds = [seconds_of(autofocus.estimate, hologram) for _ in range(repeats)]
    post = autofocus.posterior(hologram)
    est = {k: float(v[0]) for k, v in posterior_summaries(post, scale).items()}
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
        t = np.array([(value - guess) / scale])
        return {
            "z01": value,
            "distance_um": (z01 - value) * 1e6,
            "pit": float(post.pit(t)[0]),
            "hpd_level": float(post.hpd_level(t)[0]),
            "inside": {k: bool(i["lower"] <= value <= i["upper"]) for k, i in intervals.items()},
        }

    lo, hi = autofocus.prior
    edge = EDGE_MM * 1e-3 / scale
    peak = float(post.density[0].max())
    phase_z01 = guess + posterior_summaries(autofocus.posterior(phases), scale)["dz01"]
    step = max(1, (len(autofocus.theta_grid) - 1) // (EXAMPLE_POINTS - 1))
    found = modes(post)
    return {
        "models": [repo_relative(r) for r in runs],
        "model_sha256": [model_sha256(r) for r in runs],
        "z01_est": z01,
        "dz01_est": est["dz01"],
        "z01_mean": guess + est["dz01_mean"],
        "z01_mode": guess + est["dz01_mode"],
        "sigma": est["sigma"],
        "prior_mass": est["prior_mass"],
        "intervals": intervals,
        "hpd_to_central": {k: i["hpd_width_um"] / i["width_um"] for k, i in intervals.items()},
        "guess": against(guess),
        "references": {key: against(value) for key, value in refs.items()},
        "modes": [{"z01": guess + m["t"] * scale, **m} for m in found],
        "edges": {
            "lower": {
                "relative_density": float(post.density[0, 0] / peak),
                f"mass_within_{EDGE_MM}_mm": float(post.pit(np.array([lo + edge]))[0]),
            },
            "upper": {
                "relative_density": float(post.density[0, -1] / peak),
                f"mass_within_{EDGE_MM}_mm": float(1.0 - post.pit(np.array([hi - edge]))[0]),
            },
        },
        "binning_phases": {
            "n": len(phase_z01),
            "z01_est": [float(v) for v in phase_z01],
            "std_um": float(np.std(phase_z01, ddof=1) * 1e6),
            "range_um": float((phase_z01.max() - phase_z01.min()) * 1e6),
        },
        "posterior": {
            "z01_mm": ((guess + autofocus.theta_grid[::step] * scale) * 1e3).tolist(),
            "density_per_mm": (post.density[0, ::step] / (scale * 1e3)).tolist(),
        },
        "seconds_per_estimate": {
            "median": float(np.median(seconds)),
            "min": float(np.min(seconds)),
            "repeats": repeats,
        },
    }


def cnn_record(path: Path, guess: float) -> dict | None:
    """The leave-one-out CNN on the hologram (``run_stage6_study.py measured``)."""
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    intervals = record["intervals"]
    return {
        "estimate": repo_relative(path),
        "model_sha256": record.get("model_sha256"),
        "z01_est": record["z01_est"],
        "dz01_est": record["dz01_est"],
        "sigma": record["sigma"],
        "intervals": intervals,
        "guess_inside": {k: bool(i["lower"] <= guess <= i["upper"]) for k, i in intervals.items()},
    }


def cmd_measured(args) -> None:
    for name in args.samples:
        sample = MEASURED_SAMPLES[name]
        runs = members(args, name)
        if not runs:
            print(f"{name}: no trained run below {repo_relative(args.npe)}")
            continue
        variants = {method_name("npe", k): [args.npe / run_name(name, k)] for k in args.seeds}
        variants = {v: r for v, r in variants.items() if (r[0] / "model.pt").exists()}
        if len(runs) > 1:
            variants["ensemble"] = runs
        hologram = load_measured(args.holograms, sample, BINNING)
        native = load_measured(args.holograms, sample, 1).numpy()
        phases = torch.from_numpy(
            np.stack([p.astype(np.float32) for p in binning_phases(native, BINNING)])
        )
        refs = references(name, args)
        record = {
            "sample": name,
            "label": sample.label,
            "binning": BINNING,
            "z01_guess": sample.z01_guess,
            "omega": list(sample.simplex),
            "references": refs,
            "cnn_loo": cnn_record(args.cnn_measured / f"{name}.json", sample.z01_guess),
            "runs": {},
        }
        for variant, paths in variants.items():
            autofocus = NPEAutofocus.load(paths, args.device)
            if tuple(hologram.shape) != autofocus.fov_shape:
                raise ValueError(
                    f"{name}: hologram {tuple(hologram.shape)}, model {autofocus.fov_shape}"
                )
            rec = posterior_record(autofocus, paths, hologram, phases, name, refs, args.repeats)
            record["runs"][variant] = rec
            paper2, p4 = rec["references"]["paper2_nm"], rec["references"].get("baseline_b1_p4")
            print(
                f"{name} {variant}: z01 = {rec['z01_est'] * 1e3:.3f} mm, 68 % "
                f"[{rec['intervals']['68']['lower'] * 1e3:.3f}, "
                f"{rec['intervals']['68']['upper'] * 1e3:.3f}] mm, from the guess "
                f"{rec['dz01_est'] * 1e6:+.0f} um, minus [2] {paper2['distance_um']:+.0f} um "
                f"(in 68/95 {paper2['inside']['68']}/{paper2['inside']['95']})"
                + (
                    f", minus p4 {p4['distance_um']:+.0f} um (in 68/95 "
                    f"{p4['inside']['68']}/{p4['inside']['95']})"
                    if p4
                    else ""
                )
                + f", {len(rec['modes'])} mode(s), "
                f"{1e3 * rec['seconds_per_estimate']['median']:.1f} ms"
            )
        seeds = [record["runs"][v]["z01_est"] for v in variants if v != "ensemble"]
        if len(seeds) > 1:
            record["seeds"] = {
                "z01_est": seeds,
                "std_um": float(np.std(seeds, ddof=1) * 1e6),
                "range_um": float((max(seeds) - min(seeds)) * 1e6),
            }
        record["note"] = "nothing was chosen on this hologram: model, prior and settings as on the"
        record["note"] += " simulated leave-one-out sets"
        record.update(environment())
        write_json(args.out / f"{name}.json", record)


# --- timing --------------------------------------------------------------------------------


def cmd_timing(args) -> None:
    record = {
        "per_sample": args.per_sample,
        "repeats": args.repeats,
        "device": args.device,
        "what": "one estimate from the hologram in memory, batch of one: representation, "
        "network, for the NPE the density on the grid, median and intervals",
        "samples": {},
    }
    for name in args.samples:
        runs = members(args, name)
        if not runs or runs[0].name != run_name(name, 0):
            print(f"{name}: no NPE with seed 0 below {repo_relative(args.npe)}, skipped")
            continue
        data = args.data / f"loo-{name}" / "test.h5"
        with h5py.File(data, "r") as fh:
            holograms = [
                torch.from_numpy(np.asarray(fh["hologram"][i])) for i in range(args.per_sample)
            ]
        methods = {
            "cnn": LearnedAutofocus.load(cnn_run(args, name, 0), args.device),
            "npe": NPEAutofocus.load(runs[0], args.device),
        }
        if len(runs) > 1:
            methods["ensemble"] = NPEAutofocus.load(runs, args.device)
        for autofocus in methods.values():
            autofocus.estimate(holograms[0])
        seconds = {method: [] for method in methods}
        parts = {k: [] for k in ("representation", "embedding", "density_on_grid", "summaries")}
        order = list(methods)
        for repeat in range(args.repeats):
            for k, hologram in enumerate(holograms):
                shift = (repeat + k) % len(order)
                for method in order[shift:] + order[:shift]:
                    seconds[method].append(seconds_of(methods[method].estimate, hologram))
                for part, value in npe_parts(methods["npe"], hologram).items():
                    parts[part].append(value)
        entry = {
            "data": repo_relative(data),
            "fov_shape": list(methods["npe"].fov_shape),
            "ensemble_size": len(runs),
            "methods": {},
            "npe_parts_median": {part: float(np.median(v)) for part, v in parts.items()},
        }
        for method, values in seconds.items():
            v = np.array(values)
            entry["methods"][method] = {
                "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)),
                "ratio_to_cnn": float(np.median(v) / np.median(seconds["cnn"])),
            }
        record["samples"][name] = entry
        print(
            f"{name} ({entry['fov_shape'][0]}x{entry['fov_shape'][1]}): "
            + ", ".join(
                f"{m} {1e3 * e['median']:.2f} ms ({e['ratio_to_cnn']:.2f} x CNN)"
                for m, e in entry["methods"].items()
            )
        )
    record["load_average"] = list(os.getloadavg())
    record.update(environment())
    write_json(args.out, record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, help="torch CPU threads")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or auto")
    parser.add_argument("--npe", type=Path, default=NPE, help="NPE runs on the leave-one-out sets")
    parser.add_argument(
        "--cnn", type=Path, default=CNN, help="<cnn>/loo-<sample>: the CNNs, seed 0"
    )
    parser.add_argument(
        "--cnn-seeds", type=Path, default=CNN_SEEDS, help="<cnn-seeds>/seed-<k>/loo-<sample>"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def selection(p):
        p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
        p.add_argument(
            "--samples", nargs="+", choices=list(MEASURED_SAMPLES), default=list(MEASURED_SAMPLES)
        )

    def output(p, default: str):
        selection(p)
        p.add_argument("--out", type=Path, help=f"default: <npe>/{default}")
        p.set_defaults(out_default=default)

    pr = sub.add_parser("predict", help="the runs on val and test, the ensemble on test")
    pr.add_argument("--data", type=Path, default=DATA, help="<data>/loo-<sample>/{val,test}.h5")
    pr.add_argument("--limit", type=int, help="only the first N samples (not stored as current)")
    selection(pr)
    pr.set_defaults(func=cmd_predict)

    co = sub.add_parser("compare", help="NPE, ensemble, CNN and guess on the test splits")
    output(co, "comparison")
    co.set_defaults(func=cmd_compare)

    me = sub.add_parser("measured", help="seeds and ensemble on the measured holograms")
    me.add_argument("--holograms", type=Path, required=True)
    me.add_argument("--repeats", type=int, default=20, help="timed estimates")
    me.add_argument("--measured", type=Path, default=RESULTS / "measured", help="Stage 3/5 records")
    me.add_argument(
        "--cnn-measured",
        type=Path,
        default=RESULTS / "measured" / "cnn-loo",
        help="the leave-one-out CNNs on the holograms (Stage 6)",
    )
    me.add_argument(
        "--npe-stage7",
        type=Path,
        default=RESULTS / "npe" / "measured",
        help="the matched NPEs of Stage 7 on the holograms",
    )
    output(me, "measured")
    me.set_defaults(func=cmd_measured)

    ti = sub.add_parser("timing", help="time per estimate, CNN against NPE and ensemble")
    ti.add_argument("--data", type=Path, default=DATA, help="<data>/loo-<sample>/test.h5")
    ti.add_argument("--per-sample", type=int, default=20, help="test holograms per sample")
    ti.add_argument("--repeats", type=int, default=3)
    output(ti, "timing.json")
    ti.set_defaults(func=cmd_timing)

    args = parser.parse_args()
    if getattr(args, "out_default", None) and args.out is None:
        args.out = args.npe / args.out_default
    if args.threads:
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
