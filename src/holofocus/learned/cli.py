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
from .config import LearnedConfig
from .inference import LearnedAutofocus, predict_split, read_predictions
from .training import train_model


def _log(msg: str) -> None:
    print(msg, flush=True)


def train(args: argparse.Namespace) -> int:
    config = LearnedConfig.from_yaml(args.config)
    if args.limit is not None:
        config = dataclasses.replace(
            config, training=dataclasses.replace(config.training, limit=args.limit)
        )
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
        description="Learned autofocus: a CNN estimates dz01 with an uncertainty (Stage 4).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="train on a split, select and calibrate on another")
    tr.add_argument("config", type=Path, help="YAML file, e.g. configs/learned/spectrum-cnn.yaml")
    tr.add_argument("--train", type=Path, nargs="+", required=True, help="training split file(s)")
    tr.add_argument("--val", type=Path, required=True, help="validation split file")
    tr.add_argument("--out", type=Path, required=True, help="run directory")
    tr.add_argument("--limit", type=int, help="only the first N training samples")
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
    show = sub.add_parser("summary", help="print summary.json of a run or prediction directory")
    show.add_argument("run", type=Path)
    show.set_defaults(func=summary)
    args = parser.parse_args(argv)
    if getattr(args, "threads", None):
        torch.set_num_threads(args.threads)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
