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

Usage (Laborbuch 08)::

    python scripts/run_stage8_study.py sources

Run it from the root of the repository, so the paths in the records stay relative to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from holofocus.baseline.runner import file_sha256
from holofocus.learned.posterior import GaussianMixture, mixture_summaries
from holofocus.measured import MEASURED_SAMPLES
from run_stage7_loo_study import modes
from run_stage7_study import EXAMPLE_POINTS, LEVELS, environment, repo_relative, write_json

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
MEASURED = RESULTS / "measured"
SEEDS = (0, 1, 2)
GRID_POINTS = 2001
"""Points of the grid over ``Omega_z`` for the modes and the plotted density of the CNN
ensemble, as the grid of the NPE (``configs/learned/npe-nsf.yaml``)."""


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

    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
