"""Command line interface ``holofocus-baseline`` (also ``python -m holofocus.baseline``).

::

    holofocus-baseline run configs/baseline/paper2-cpu.yaml --data data/cpu-4x/test.h5 \\
        --out runs/baseline/cpu-4x-test --limit 48
    holofocus-baseline summary runs/baseline/cpu-4x-test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from .config import BaselineConfig
from .results import read_rows, summarize
from .runner import run_baseline


def run(args: argparse.Namespace) -> int:
    if args.threads:
        torch.set_num_threads(args.threads)
    config = BaselineConfig.from_yaml(args.config)
    try:
        summary = run_baseline(
            config,
            args.data,
            args.out,
            indices=args.indices,
            limit=args.limit,
            device=args.device,
            log=lambda msg: print(msg, flush=True),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


def summary(args: argparse.Namespace) -> int:
    rows = read_rows(Path(args.run) / "results.csv")
    if not rows:
        print(f"error: no results in {args.run}", file=sys.stderr)
        return 1
    print(json.dumps(summarize(rows), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="holofocus-baseline",
        description="Classical autofocus of [2] (ASRM + MFE + Nelder-Mead) on a dataset split.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run", help="autofocus the samples of an HDF5 split")
    go.add_argument("config", type=Path, help="YAML file, e.g. configs/baseline/paper2-cpu.yaml")
    go.add_argument("--data", type=Path, required=True, help="split file, e.g. data/.../test.h5")
    go.add_argument("--out", type=Path, required=True, help="run directory (resumable)")
    go.add_argument("--limit", type=int, help="only the first N selected samples")
    go.add_argument("--indices", type=int, nargs="+", help="sample indices (default: all)")
    go.add_argument("--device", help="cpu, cuda, cuda:1 or auto (default: from the config)")
    go.add_argument("--threads", type=int, help="torch CPU threads")
    go.set_defaults(func=run)
    show = sub.add_parser("summary", help="print the summary of a run directory")
    show.add_argument("run", type=Path)
    show.set_defaults(func=summary)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
