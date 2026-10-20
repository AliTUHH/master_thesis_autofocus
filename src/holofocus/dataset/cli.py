"""Command line interface ``holofocus-dataset`` (also ``python -m holofocus.dataset``).

::

    holofocus-dataset generate configs/dataset/cpu-4x.yaml --out data/cpu-4x
    holofocus-dataset generate configs/dataset/ci.yaml --out /tmp/ci --splits val --limit 4
    holofocus-dataset inspect data/cpu-4x/train.h5
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import h5py

from .config import DatasetConfig
from .storage import read_config, read_labels, write_split


def _existing_matches(path: Path, config: DatasetConfig, n_samples: int) -> bool:
    try:
        with h5py.File(path, "r") as fh:
            n_stored = int(fh.attrs["n_samples"])
        return n_stored == n_samples and read_config(path) == config
    except (OSError, KeyError, ValueError):
        return False


def generate(args: argparse.Namespace) -> int:
    config = DatasetConfig.from_yaml(args.config)
    splits = args.splits or list(config.splits)
    unknown = [s for s in splits if s not in config.splits]
    if unknown:
        print(f"error: splits {unknown} not in config {list(config.splits)}", file=sys.stderr)
        return 2
    print(
        f"{config.name}: fov {config.geometry.fov_shape}, simulation {config.simulation_shape}, "
        f"pixel {config.geometry.pixel_size * 1e6:.1f} um, z01 in "
        f"[{config.z01_bounds[0] * 1e3:.2f}, {config.z01_bounds[1] * 1e3:.2f}] mm"
    )
    out = Path(args.out)
    todo = []
    for split in splits:
        n = config.splits[split]
        if args.limit is not None:
            n = min(n, args.limit)
        path = out / f"{split}.h5"
        if path.exists() and not args.overwrite:
            if _existing_matches(path, config, n):
                print(f"{path}: up to date, skipped")
                continue
            print(f"error: {path} exists with another config, use --overwrite", file=sys.stderr)
            return 1
        todo.append((split, n, path))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(config.to_yaml())
    for split, n, path in todo:
        write_split(config, split, path, n_samples=n, device=args.device)
        print(f"wrote {path}")
    return 0


def inspect(args: argparse.Namespace) -> int:
    for path in args.files:
        with h5py.File(path, "r") as fh:
            a = fh.attrs
            fov = tuple(int(v) for v in a["fov_shape"])
            sim = tuple(int(v) for v in a["simulation_shape"])
            print(
                f"{path}: split {a['split']}, {a['n_samples']} samples, "
                f"fov {fov}, simulation {sim}, "
                f"pixel {a['pixel_size'] * 1e6:.1f} um, seed {a['seed']}, "
                f"commit {a['git_commit']}, device {a['device']}, {a['seconds']:.1f} s"
            )
            for name in ("hologram", "hologram_clean", "phase"):
                if name in fh:
                    ds = fh[name]
                    print(f"  {name}: {ds.shape} {ds.dtype}, compression {ds.compression}")
        for key, v in read_labels(path).items():
            print(f"  {key}: min {v.min():.6g}, mean {v.mean():.6g}, max {v.max():.6g}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="holofocus-dataset", description="Synthetic hologram datasets for learning dz01."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="simulate the splits of a YAML config into HDF5 files")
    gen.add_argument("config", type=Path, help="YAML file, e.g. configs/dataset/cpu-4x.yaml")
    gen.add_argument("--out", type=Path, required=True, help="output directory")
    gen.add_argument("--splits", nargs="+", help="subset of the splits in the config")
    gen.add_argument("--limit", type=int, help="at most this many samples per split")
    gen.add_argument("--device", help="cpu, cuda, cuda:1 or auto (default: from the config)")
    gen.add_argument("--overwrite", action="store_true", help="replace existing split files")
    gen.set_defaults(func=generate)
    ins = sub.add_parser("inspect", help="print attributes and label ranges of HDF5 files")
    ins.add_argument("files", nargs="+", type=Path)
    ins.set_defaults(func=inspect)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
