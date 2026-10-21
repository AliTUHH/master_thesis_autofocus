"""Command line interface ``holofocus-baseline`` (also ``python -m holofocus.baseline``).

::

    holofocus-baseline run configs/baseline/paper2-cpu.yaml --data data/cpu-4x/test.h5 \\
        --out runs/baseline/cpu-4x-test --limit 48
    holofocus-baseline run configs/baseline/paper2-cpu-warm.yaml --data data/cpu-4x/test.h5 \\
        --out runs/baseline/cpu-4x-test-warm --limit 48 \\
        --warm-start results/learned/spectrum-cnn/test/predictions.csv --half-width 5e-4
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
from .runner import WarmStart, run_baseline


def run(args: argparse.Namespace) -> int:
    if args.threads:
        torch.set_num_threads(args.threads)
    config = BaselineConfig.from_yaml(args.config)
    try:
        warm_start = None
        if args.warm_start is not None:
            warm_start = WarmStart.from_csv(
                args.warm_start, args.half_width, args.sigmas, args.anchor
            )
        summary = run_baseline(
            config,
            args.data,
            args.out,
            indices=args.indices,
            limit=args.limit,
            device=args.device,
            warm_start=warm_start,
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
    go.add_argument(
        "--warm-start",
        type=Path,
        help="CSV with index, z01_est (and sigma) per sample: start the simplex around it",
    )
    go.add_argument(
        "--half-width", type=float, default=5e-4, help="warm-start half width in m (5e-4)"
    )
    go.add_argument(
        "--sigmas", type=float, default=0.0, help="widen the half width to this many sigma (0)"
    )
    go.add_argument(
        "--anchor",
        choices=("center", "vertex"),
        default="center",
        help="warm-start simplex c +- w (center, default) or (c, c + w) (vertex)",
    )
    go.set_defaults(func=run)
    show = sub.add_parser("summary", help="print the summary of a run directory")
    show.add_argument("run", type=Path)
    show.set_defaults(func=summary)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
