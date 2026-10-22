"""Command line interface ``holofocus-learn`` (also ``python -m holofocus.learned``).

::

    holofocus-learn train configs/learned/spectrum-cnn.yaml --train data/cpu-4x/train.h5 \\
        --val data/cpu-4x/val.h5 --out results/learned/spectrum-cnn
    holofocus-learn predict results/learned/spectrum-cnn --data data/cpu-4x/test.h5 \\
        --out results/learned/spectrum-cnn/test
    holofocus-learn compare --learned results/learned/spectrum-cnn/test/predictions.csv \\
        --baseline results/baseline/cpu-4x-test --baseline-val results/baseline/cpu-4x-val \\
        --out results/comparison/cpu-4x-test
    holofocus-learn summary results/learned/spectrum-cnn/test

NPE of Stage 7 (needs the optional extra ``npe``, i.e. sbi)::

    holofocus-learn npe-train configs/learned/npe-nsf.yaml --train data/cpu-4x/train.h5 \\
        --val data/cpu-4x/val.h5 --out results/npe/npe-nsf
    holofocus-learn npe-predict results/npe/npe-nsf --data data/cpu-4x/test.h5 \\
        --out results/npe/npe-nsf/test
    holofocus-learn npe-check results/npe/npe-nsf \\
        --data data/cpu-4x-calibration/calibration.h5 --out results/npe/npe-nsf/calibration
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from ..baseline.results import read_rows
from .compare import compare_methods, offset_estimate, write_comparison
from .config import LearnedConfig, NPEConfig
from .inference import LearnedAutofocus, predict_split, read_predictions
from .training import train_model


def _log(msg: str) -> None:
    print(msg, flush=True)


def _with_training(config, limit: int | None, seed: int | None):
    overrides = {k: v for k, v in (("limit", limit), ("seed", seed)) if v is not None}
    if not overrides:
        return config
    return dataclasses.replace(config, training=dataclasses.replace(config.training, **overrides))


def _npe():
    """The NPE module, which needs sbi (optional extra ``npe``)."""
    try:
        from . import npe
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "sbi":
            raise
        raise SystemExit(f"error: the NPE needs sbi ({exc}); pip install -e '.[npe]'") from None
    return npe


def train(args: argparse.Namespace) -> int:
    config = _with_training(LearnedConfig.from_yaml(args.config), args.limit, args.seed)
    try:
        summary = train_model(
            config, args.train, args.val, args.out, args.device, args.overwrite, _log
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary["validation"], indent=2))
    return 0


def predict(args: argparse.Namespace) -> int:
    try:
        autofocus = LearnedAutofocus.load(args.model, args.device)
        predict_split(
            autofocus,
            args.data,
            args.out,
            indices=args.indices,
            limit=args.limit,
            model_path=args.model,
            log=_log,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def compare(args: argparse.Namespace) -> int:
    baseline = read_rows(Path(args.baseline) / "results.csv")
    validation = read_rows(Path(args.baseline_val) / "results.csv")
    warm = None if args.warm is None else read_rows(Path(args.warm) / "results.csv")
    try:
        offset = offset_estimate([r["error_um"] for r in validation])
        rows, summary = compare_methods(read_predictions(args.learned), baseline, offset, warm)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary["sources"] = {
        "learned": str(args.learned),
        "baseline": str(args.baseline),
        "baseline_val": str(args.baseline_val),
        "warm": None if args.warm is None else str(args.warm),
    }
    write_comparison(args.out, rows, summary)
    print(
        json.dumps({k: summary[k] for k in ("n_samples", "offset_um", "speedup_median")}, indent=2)
    )
    return 0


def npe_train(args: argparse.Namespace) -> int:
    npe = _npe()
    config = _with_training(NPEConfig.from_yaml(args.config), args.limit, args.seed)
    try:
        summary = npe.train_npe(
            config, args.train, args.val, args.out, args.device, args.overwrite, _log
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    validation = summary["validation"]
    print(json.dumps({k: validation[k] for k in ("error_um", "interval")}, indent=2))
    return 0


def npe_predict(args: argparse.Namespace) -> int:
    npe = _npe()
    try:
        autofocus = npe.NPEAutofocus.load(args.models, args.device)
        npe.predict_npe_split(
            autofocus,
            args.data,
            args.out,
            indices=args.indices,
            limit=args.limit,
            model_paths=args.models,
            log=_log,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def npe_check(args: argparse.Namespace) -> int:
    npe = _npe()
    try:
        autofocus = npe.NPEAutofocus.load(args.model, args.device)
        npe.check_npe_split(
            autofocus,
            args.data,
            args.out,
            limit=args.limit,
            num_posterior_samples=args.samples,
            seed=args.seed,
            model_path=args.model,
            log=_log,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def summary(args: argparse.Namespace) -> int:
    path = Path(args.run) / "summary.json"
    if not path.exists():
        print(f"error: no summary.json in {args.run}", file=sys.stderr)
        return 1
    print(path.read_text(), end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="holofocus-learn",
        description=(
            "Learned autofocus: a CNN estimates dz01 with an uncertainty (Stage 4), an NPE "
            "its posterior (Stage 7)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="train on a split, select and calibrate on another")
    tr.add_argument("config", type=Path, help="YAML file, e.g. configs/learned/spectrum-cnn.yaml")
    tr.add_argument("--train", type=Path, nargs="+", required=True, help="training split file(s)")
    tr.add_argument("--val", type=Path, required=True, help="validation split file")
    tr.add_argument("--out", type=Path, required=True, help="run directory")
    tr.add_argument("--limit", type=int, help="only the first N training samples")
    tr.add_argument("--seed", type=int, help="training seed (default: from the config)")
    tr.add_argument("--device", help="cpu, cuda, cuda:1 or auto (default: from the config)")
    tr.add_argument("--overwrite", action="store_true", help="replace a trained model in --out")
    tr.add_argument("--threads", type=int, help="torch CPU threads")
    tr.set_defaults(func=train)
    pr = sub.add_parser("predict", help="estimate dz01 for the samples of a split")
    pr.add_argument("model", type=Path, help="training run directory or model.pt")
    pr.add_argument("--data", type=Path, required=True, help="split file, e.g. .../test.h5")
    pr.add_argument("--out", type=Path, required=True, help="output directory")
    pr.add_argument("--limit", type=int, help="only the first N selected samples")
    pr.add_argument("--indices", type=int, nargs="+", help="sample indices (default: all)")
    pr.add_argument("--device", help="cpu, cuda, cuda:1 or auto (default: from the config)")
    pr.add_argument("--threads", type=int, help="torch CPU threads")
    pr.set_defaults(func=predict)
    co = sub.add_parser("compare", help="learned model vs. baseline (raw and offset-corrected)")
    co.add_argument("--learned", type=Path, required=True, help="predictions.csv")
    co.add_argument("--baseline", type=Path, required=True, help="baseline run directory")
    co.add_argument(
        "--baseline-val", type=Path, required=True, help="baseline run on validation samples"
    )
    co.add_argument("--warm", type=Path, help="warm-started baseline run directory")
    co.add_argument("--out", type=Path, required=True, help="output directory")
    co.set_defaults(func=compare)
    nt = sub.add_parser("npe-train", help="train an NPE of dz01 (Stage 7, needs sbi)")
    nt.add_argument("config", type=Path, help="YAML file, e.g. configs/learned/npe-nsf.yaml")
    nt.add_argument("--train", type=Path, nargs="+", required=True, help="training split file(s)")
    nt.add_argument("--val", type=Path, required=True, help="validation split file")
    nt.add_argument("--out", type=Path, required=True, help="run directory")
    nt.add_argument("--limit", type=int, help="only the first N training samples")
    nt.add_argument("--seed", type=int, help="training seed (default: from the config)")
    nt.add_argument("--device", help="cpu, cuda, cuda:1 or auto (default: from the config)")
    nt.add_argument("--overwrite", action="store_true", help="replace a trained model in --out")
    nt.add_argument("--threads", type=int, help="torch CPU threads")
    nt.set_defaults(func=npe_train)
    np_ = sub.add_parser("npe-predict", help="posterior of dz01 for the samples of a split")
    np_.add_argument(
        "models", type=Path, nargs="+", help="NPE run directory or model.pt (several: ensemble)"
    )
    np_.add_argument("--data", type=Path, required=True, help="split file, e.g. .../test.h5")
    np_.add_argument("--out", type=Path, required=True, help="output directory")
    np_.add_argument("--limit", type=int, help="only the first N selected samples")
    np_.add_argument("--indices", type=int, nargs="+", help="sample indices (default: all)")
    np_.add_argument("--device", help="cpu, cuda, cuda:1 or auto (default: from the config)")
    np_.add_argument("--threads", type=int, help="torch CPU threads")
    np_.set_defaults(func=npe_predict)
    nc = sub.add_parser("npe-check", help="SBC, expected coverage and TARP of sbi on a split")
    nc.add_argument("model", type=Path, help="NPE run directory or model.pt")
    nc.add_argument("--data", type=Path, required=True, help="split file (calibration split)")
    nc.add_argument("--out", type=Path, required=True, help="output directory")
    nc.add_argument("--limit", type=int, help="only the first N samples")
    nc.add_argument("--samples", type=int, default=1000, help="posterior samples per input")
    nc.add_argument("--seed", type=int, default=0, help="seed of the posterior samples")
    nc.add_argument("--device", help="cpu, cuda, cuda:1 or auto (default: from the config)")
    nc.add_argument("--threads", type=int, help="torch CPU threads")
    nc.set_defaults(func=npe_check)
    show = sub.add_parser("summary", help="print summary.json of a run or prediction directory")
    show.add_argument("run", type=Path)
    show.set_defaults(func=summary)
    args = parser.parse_args(argv)
    if getattr(args, "threads", None):
        torch.set_num_threads(args.threads)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
