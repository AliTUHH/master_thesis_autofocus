"""Generate train/val/test hologram datasets with HoloForge from a YAML description.

Usage::

    python -m src.data.generate_data --config configs/data_small.yaml [--out data/processed/<name>] [--overwrite]

For every split a HoloForge JSON config is assembled from the YAML (geometry, phantom, probe and
noise options), a :class:`~src.data.forge_setup.NFHRandomDistSetup` with a split-specific seed is
installed, and HoloForge's ``DataGenerator`` writes ``<split>.hdf5`` plus the fully merged Forge
config ``<split>.json``. Finally ``meta.json`` records the configuration, seeds, timings, package
versions and a label verification (stored Fr vs. recomputation from z01/z02/E/px).

HDF5 layout written by HoloForge (``HDF5Labeller``)::

    images/hologram           (N, H, W) float32   detector-plane amplitude |psi| incl. noise
    images/gt_hologram        optional, noise-free amplitude
    images/phantoms           optional, (N, H, W) complex64 (phase shift + i*absorption)
    metadata/setup/{z01, z02, Fr, energy, detector_px_size, detector_size, padding_factor,
                    downsample_factor, probe_size}   (N,) float32, one entry per sample
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

import holowizard.forge.generators as forge_generators
import holowizard.forge.utils.torch_settings as torch_settings
from holowizard.forge.configs.parse_config import ConfigParser
from holowizard.forge.generators import DataGenerator, HologramGenerator, PhantomGenerator, ProbeGenerator

from src.data.forge_setup import NFHRandomDistSetup, register_forge_setup, reset_global_forge_setup
from src.utils.config import PROJECT_ROOT, deep_update, load_yaml, require_keys, resolve_path
from src.utils.physics import fresnel_number

__all__ = [
    "SPLITS",
    "build_forge_config",
    "propagator_sampling_check",
    "generate_split",
    "generate_dataset",
    "verify_split_labels",
    "main",
]

SPLITS: tuple[str, ...] = ("train", "val", "test")

HOLOGRAM_CONVENTION = (
    "images/hologram stores the detector-plane amplitude |psi_det| as simulated by HoloForge "
    "(NFHSimulation uses torch.abs, not abs**2); intensity = hologram**2. Noise is added to the amplitude."
)


def _noise_block(spec: dict[str, Any] | None, where: str) -> dict[str, Any] | None:
    """Translate ``{type: Gaussian, mean: .., std: .., intensity: ..}`` into a Forge ``{type, args}`` block."""
    if spec is None:
        return None
    require_keys(spec, ("type",), where)
    args = {key: value for key, value in spec.items() if key != "type"}
    return {"type": spec["type"], "args": args}


def _range_pair(value: Any, where: str) -> tuple[Any, Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"'{where}' must be a [min, max] pair, got {value!r}")
    return value[0], value[1]


def build_forge_config(cfg: dict[str, Any], split: str, seed: int) -> dict[str, Any]:
    """Assemble the HoloForge JSON config for one split from the YAML mapping ``cfg``.

    The result is a plain ``dict`` that HoloForge merges over its own ``default.json``.
    """
    require_keys(cfg, ("setup", "phantom", "probe", "num_samples", "seeds"), "<root>")
    setup = cfg["setup"]
    require_keys(
        setup,
        ("detector_size", "detector_px_size_mm", "energy_kev", "padding_factor", "downsample_factor", "z01_mm", "z02_mm"),
        "setup",
    )
    z01 = setup["z01_mm"]
    require_keys(z01, ("min", "max"), "setup.z01_mm")
    z02 = setup["z02_mm"]
    if isinstance(z02, dict):
        require_keys(z02, ("min", "max"), "setup.z02_mm")
        z02_min, z02_max = float(z02["min"]), float(z02["max"])
    else:
        z02_min, z02_max = float(z02), None

    setup_args: dict[str, Any] = {
        "detector_size": int(setup["detector_size"]),
        "detector_px_size": float(setup["detector_px_size_mm"]),
        "padding_factor": int(setup["padding_factor"]),
        "downsample_factor": int(setup["downsample_factor"]),
        "energy": float(setup["energy_kev"]),
        "z01_min": float(z01["min"]),
        "z01_max": float(z01["max"]),
        "z02_min": z02_min,
        "z02_max": z02_max,
        "z01_distribution": str(z01.get("distribution", "uniform")),
        "seed": int(seed),
    }

    phantom = cfg["phantom"]
    require_keys(
        phantom,
        ("materials", "num_shapes", "thickness_um", "shapes", "radius_range_px", "size_range_px", "smoothing"),
        "phantom",
    )
    shapes_min, shapes_max = _range_pair(phantom["num_shapes"], "phantom.num_shapes")
    thick_min, thick_max = _range_pair(phantom["thickness_um"], "phantom.thickness_um")
    smoothing = phantom["smoothing"]
    require_keys(smoothing, ("kernel_size", "sigma"), "phantom.smoothing")
    phantom_args: dict[str, Any] = {
        "position": str(phantom.get("position", "random")),
        "materials": list(phantom["materials"]),
        "num_shapes_min": int(shapes_min),
        "num_shapes_max": int(shapes_max),
        "thickness_min": int(thick_min),
        "thickness_max": int(thick_max),
        "smoothing_filter": {
            "type": "Gaussian",
            "args": {"kernel_size": int(smoothing["kernel_size"]), "sigma": float(smoothing["sigma"])},
        },
    }
    for key in ("absorption_min", "absorption_max", "phaseshift_min", "phaseshift_max"):
        if phantom.get(key) is not None:
            phantom_args[key] = float(phantom[key])

    shape_sampler_args: dict[str, Any] = {
        "shapes": list(phantom["shapes"]),
        "radius_range": list(_range_pair(phantom["radius_range_px"], "phantom.radius_range_px")),
        "size_range": list(_range_pair(phantom["size_range_px"], "phantom.size_range_px")),
        "rotate": bool(phantom.get("rotate", True)),
        "polygon_max_corners": int(phantom.get("polygon_max_corners", 4)),
        "ellipse_max_cut": float(phantom.get("ellipse_max_cut", 0)),
    }

    probe = cfg["probe"]
    require_keys(probe, ("constant", "linear", "square", "center_beam"), "probe")
    probe_args = {
        "constant": float(probe["constant"]),
        "linear": float(probe["linear"]),
        "square": float(probe["square"]),
        "center_beam": bool(probe["center_beam"]),
    }

    store = cfg.get("store", {})
    labeller_args = {
        "store_hologram": bool(store.get("hologram", True)),
        "store_gt_hologram": bool(store.get("gt_hologram", False)),
        "store_phantom": bool(store.get("phantom", False)),
        "store_probe": bool(store.get("probe", False)),
        "store_flatfield": bool(store.get("flatfield", False)),
        "store_polynomial": bool(store.get("polynomial", False)),
        "store_setup": True,
        "cache_size": int(cfg.get("cache_size", 10)),
        "num_conditions": 0,
    }

    forge_cfg: dict[str, Any] = {
        "name": split,
        "labeller": {"type": "HDF5Labeller", "args": labeller_args},
        "data_generator": {"type": "DataGenerator", "args": {"crop_probe": True}},
        "phantom_generator": {"type": "PhantomGenerator", "args": phantom_args},
        "probe_generator": {"type": "ProbeGenerator", "args": probe_args},
        "probe_noise": _noise_block(probe.get("noise"), "probe.noise"),
        "shape_sampler": {"type": "ShapeSampler", "args": shape_sampler_args},
        "flatfield_generator": {"type": "FlatFieldGenerator", "args": {}},
        "flatfield_dataset": None,
        "hologram_generator": {"type": "HologramGenerator", "args": {}},
        "hologram_noise": _noise_block(cfg.get("hologram_noise"), "hologram_noise"),
        "simulation": {"type": "NFHSimulation", "args": {}},
        "setup": {"type": NFHRandomDistSetup.__name__, "args": setup_args},
    }
    overrides = cfg.get("forge_overrides") or {}
    return deep_update(forge_cfg, overrides)


def propagator_sampling_check(cfg: dict[str, Any]) -> dict[str, Any]:
    """Check the transfer-function propagator's sampling criterion ``Fr >= 1 / grid_size``.

    HoloForge applies ``exp(-i*pi/Fr*(xi^2+eta^2))`` on a grid of ``detector_size/downsample * padding``
    pixels; for smaller Fresnel numbers the chirp is aliased and holograms show wrap-around artefacts.
    Returns the minimal Fr of the configured z01/z02 range, the limit and an ``ok`` flag.
    """
    setup = cfg["setup"]
    px_eff = float(setup["detector_px_size_mm"]) * int(setup["downsample_factor"])
    grid = int(setup["detector_size"]) // int(setup["downsample_factor"]) * int(setup["padding_factor"])
    z02 = setup["z02_mm"]
    z02_max = float(z02["max"]) if isinstance(z02, dict) else float(z02)
    fr_min = float(fresnel_number(float(setup["z01_mm"]["min"]), z02_max, float(setup["energy_kev"]), px_eff))
    limit = 1.0 / grid
    return {"fr_min": fr_min, "fr_limit": limit, "grid_size": grid, "ok": fr_min >= limit}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch_settings.set_reproducibility(seed)


def generate_split(forge_cfg: dict[str, Any], out_dir: Path, num_samples: int, seed: int, overwrite: bool) -> dict[str, Any]:
    """Run HoloForge for one split and return ``{'file', 'num_samples', 'seed', 'duration_s', 'seconds_per_sample'}``.

    Raises ``FileExistsError`` if ``<out_dir>/<name>.hdf5`` exists and ``overwrite`` is False.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{forge_cfg['name']}.hdf5"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite to replace it")

    register_forge_setup()
    reset_global_forge_setup()
    _seed_everything(seed)

    with tempfile.TemporaryDirectory(prefix="forge_cfg_") as tmp:
        cfg_path = Path(tmp) / f"{forge_cfg['name']}.json"
        cfg_path.write_text(json.dumps(forge_cfg, indent=2), encoding="utf-8")
        config = ConfigParser(cfg_path)

    hologram_generator = HologramGenerator(config)
    phantom_generator = PhantomGenerator(config)
    probe_generator = ProbeGenerator(config)
    flatfield_generator = config.init_obj("flatfield_generator", forge_generators, config=config)
    data_generator = DataGenerator(
        output=str(out_dir),
        num_samples=num_samples,
        config=config,
        override=True,
        hologram_generator=hologram_generator,
        phantom_generator=phantom_generator,
        probe_generator=probe_generator,
        flatfield_generator=flatfield_generator,
    )
    start = time.perf_counter()
    data_generator.generate_data()
    duration = time.perf_counter() - start
    return {
        "file": target.name,
        "num_samples": num_samples,
        "seed": seed,
        "duration_s": round(duration, 3),
        "seconds_per_sample": round(duration / num_samples, 4),
    }


def verify_split_labels(h5_path: Path, z01_bounds: tuple[float, float] | None = None) -> dict[str, Any]:
    """Recompute Fr from the stored per-sample z01/z02/energy/pixel size and compare with the stored Fr.

    Returns max/mean relative deviation (float32 storage limits agreement to ~1e-7), the z01 range,
    and whether all z01 lie inside ``z01_bounds`` (if given).
    """
    with h5py.File(h5_path, "r") as handle:
        setup = handle["metadata/setup"]
        z01 = setup["z01"][...].astype(np.float64)
        z02 = setup["z02"][...].astype(np.float64)
        energy = setup["energy"][...].astype(np.float64)
        px = setup["detector_px_size"][...].astype(np.float64)
        fr_stored = setup["Fr"][...].astype(np.float64)
        num_holograms = handle["images/hologram"].shape[0]
    fr_recomputed = np.asarray(fresnel_number(z01, z02, energy, px), dtype=np.float64)
    rel = np.abs(fr_stored - fr_recomputed) / fr_recomputed
    result: dict[str, Any] = {
        "num_samples": int(num_holograms),
        "fr_rel_deviation_max": float(rel.max()),
        "fr_rel_deviation_mean": float(rel.mean()),
        "z01_mm_min": float(z01.min()),
        "z01_mm_max": float(z01.max()),
        "z01_mm_unique": int(np.unique(z01).size),
        "fr_min": float(fr_stored.min()),
        "fr_max": float(fr_stored.max()),
    }
    if z01_bounds is not None:
        lo, hi = z01_bounds
        result["z01_within_bounds"] = bool(np.all((z01 >= lo) & (z01 <= hi)))
    return result


def generate_dataset(cfg: dict[str, Any], out_dir: Path, overwrite: bool = False, config_path: Path | None = None) -> dict[str, Any]:
    """Generate all splits defined in ``cfg['num_samples']`` into ``out_dir`` and write ``meta.json``."""
    require_keys(cfg, ("num_samples", "seeds"), "<root>")
    splits = [split for split in SPLITS if cfg["num_samples"].get(split, 0)]
    if not splits:
        raise ValueError("num_samples must request at least one of train/val/test")
    seeds = cfg["seeds"]
    require_keys(seeds, splits, "seeds")
    if len({int(seeds[split]) for split in splits}) != len(splits):
        raise ValueError(f"seeds must be disjoint across splits, got {seeds}")

    sampling = propagator_sampling_check(cfg)
    if not sampling["ok"]:
        warnings.warn(
            f"minimal Fresnel number {sampling['fr_min']:.3e} is below the propagator sampling limit "
            f"1/{sampling['grid_size']} = {sampling['fr_limit']:.3e}; increase z01_mm.min, downsample_factor or "
            "padding_factor to avoid aliasing artefacts",
            stacklevel=2,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "name": cfg.get("name", out_dir.name),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": str(config_path) if config_path else None,
        "config": cfg,
        "propagator_sampling": sampling,
        "versions": {
            "holowizard": _package_version("holowizard"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "device": str(torch_settings.get_torch_device()),
        "hologram_convention": HOLOGRAM_CONVENTION,
        "units": {"z01": "mm", "z02": "mm", "detector_px_size": "mm", "energy": "keV", "Fr": "dimensionless"},
        "splits": {},
    }
    z01_bounds = (float(cfg["setup"]["z01_mm"]["min"]), float(cfg["setup"]["z01_mm"]["max"]))
    for split in splits:
        num_samples = int(cfg["num_samples"][split])
        seed = int(seeds[split])
        forge_cfg = build_forge_config(cfg, split, seed)
        print(f"[{split}] generating {num_samples} holograms (seed={seed}) -> {out_dir / (split + '.hdf5')}", flush=True)
        info = generate_split(forge_cfg, out_dir, num_samples, seed, overwrite)
        info["verification"] = verify_split_labels(out_dir / info["file"], z01_bounds)
        meta["splits"][split] = info
        print(
            f"[{split}] done: {info['duration_s']} s total, {info['seconds_per_sample']} s/sample, "
            f"max rel. Fr deviation {info['verification']['fr_rel_deviation_max']:.2e}, "
            f"z01 in [{info['verification']['z01_mm_min']:.3f}, {info['verification']['z01_mm_max']:.3f}] mm",
            flush=True,
        )
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {meta_path}")
    return meta


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Generate NFH hologram datasets with HoloForge.")
    parser.add_argument("--config", required=True, help="YAML dataset description, e.g. configs/data_small.yaml")
    parser.add_argument("--out", default=None, help="Output directory (default: <output_dir>/<name> from the YAML)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing split files")
    args = parser.parse_args(argv)

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    if args.out is not None:
        out_dir = resolve_path(args.out)
    else:
        require_keys(cfg, ("name", "output_dir"), "<root>")
        out_dir = resolve_path(cfg["output_dir"]) / str(cfg["name"])
    return generate_dataset(cfg, out_dir, overwrite=args.overwrite, config_path=config_path.relative_to(PROJECT_ROOT) if config_path.is_relative_to(PROJECT_ROOT) else config_path)


if __name__ == "__main__":
    main()
