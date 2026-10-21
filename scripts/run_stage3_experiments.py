"""Stage 3 experiments with many ASRM reconstructions; results go to ``results/``.

* ``measured``: the classical autofocus of [2] (ASRM + MFE + Nelder-Mead) on the measured
  holograms of the Zenodo record of [1], binned 4x, compared with [2] Table 1. Writes
  ``<sample>.json`` and the phase at the estimate (``<sample>-phase.npy``, not tracked). With
  ``--warm-start DIR`` the search starts from ``z01_est +- half width`` of ``DIR/<sample>.json``
  (e.g. the learned estimates of ``scripts/run_stage5_study.py measured``) instead of the ends
  of ``Omega_z``.
* ``curves``: focus curves with one ASRM reconstruction per sampling point ([2] Fig. 8), for
  samples of a synthetic split (``synthetic-<index>.json``, with the phase error against the
  ground truth) or for measured holograms centred on the visually determined focus of [2]
  Table 1 (``measured-<sample>.json``).
* ``bias``: a fine MFE scan around the true ``z01`` of synthetic samples for several data
  models (:data:`BIAS_VARIANTS`), to locate the minimum of the objective without the
  Nelder-Mead tolerance (``bias-<index>.json``).

All commands skip outputs that exist already. Usage::

    python scripts/run_stage3_experiments.py measured --holograms DIR --out results/measured/p2
    python scripts/run_stage3_experiments.py measured --holograms DIR --paper-padding \\
        --samples spider_hair tooth --out results/measured/paper-padding
    python scripts/run_stage3_experiments.py --threads 3 measured --holograms DIR --binning 1 \\
        --padding 4 --warm-start results/measured/cnn --out results/measured/warm-b1-p4
    python scripts/run_stage3_experiments.py curves --data data/cpu-4x/test.h5 --indices 1 5 \\
        --out results/focus-curves
    python scripts/run_stage3_experiments.py curves --holograms DIR --samples spider_hair \\
        --out results/focus-curves
    python scripts/run_stage3_experiments.py bias --data data/cpu-4x/test.h5 --indices 2 \\
        --out results/bias
    python scripts/run_stage3_experiments.py --l2-scale 0.6667 bias --data data/cpu-4x/test.h5 \\
        --indices 2 --variants noisy --out results/padding/bias-p2-l2

``DIR`` is ``Examples/data/holograms`` of the Zenodo record 10.5281/zenodo.8349365.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from holofocus.asrm import ASRMSettings, stage_grid
from holofocus.autofocus import find_focus, reconstruction_focus_curve, symmetric_offsets
from holofocus.baseline import BaselineConfig
from holofocus.baseline.runner import WarmStart, file_sha256
from holofocus.dataset import HologramGenerator, read_config, read_labels
from holofocus.dataset.config import DatasetConfig
from holofocus.dataset.storage import _git_commit
from holofocus.forward import object_from_phase, simulate_hologram
from holofocus.measured import MEASURED_SAMPLES, MeasuredSample, load_measured
from holofocus.yamlconfig import plain

REPO = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = REPO / "configs" / "baseline" / "paper2-cpu.yaml"

BIAS_VARIANTS = {
    "noisy": "hologram of the split: intensity integrated over the oversampled sub-pixels, "
    "detector noise",
    "clean": "the same hologram without noise",
    "grid": "noise-free hologram simulated on the reconstruction grid from the binned phase "
    "(the forward model of the reconstruction)",
    "grid-nofilter": "as grid, reconstructed without the object filters of [2] Table 3",
}


def environment() -> dict:
    return {
        "git_commit": _git_commit(),
        "torch_version": str(torch.__version__),
        "threads": torch.get_num_threads(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
    }


def scaled(settings: ASRMSettings, scale: float, l2_scale: float = 1.0) -> ASRMSettings:
    """Scale the iterations (``scale``) and the L2 weight on ``mu`` (``l2_scale``) of every stage.

    The L2 term is normalized by the norm of ``mu`` over the whole padded array, so its
    weight per pixel depends on the padding; ``l2_scale`` compensates that in experiments.
    """
    if scale == 1.0 and l2_scale == 1.0:
        return settings
    stages = tuple(
        dataclasses.replace(
            s, iterations=max(1, round(s.iterations * scale)), l2_weight=s.l2_weight * l2_scale
        )
        for s in settings.stages
    )
    return dataclasses.replace(settings, stages=stages)


def measured_settings(
    sample: MeasuredSample,
    binning: int,
    padding: float | None,
    scale: float = 1.0,
    l2_scale: float = 1.0,
) -> tuple[ASRMSettings, float]:
    """[2] Table 3 on the binned detector with the flat-field offset ``A0`` of the sample.

    ``padding=None`` takes the array size of [2] Table 1. Returns the settings and the
    nominal padding factor (relative to the uncut detector).
    """
    nominal = sample.padding_factor if padding is None else padding
    settings = BaselineConfig.from_yaml(BASELINE_CONFIG).reconstruction.settings(binning)
    settings = dataclasses.replace(
        settings, padding_factor=sample.padding_for(nominal), a0=sample.a0
    )
    return scaled(settings, scale, l2_scale), nominal


def synthetic_settings(
    config: DatasetConfig, padding: float, scale: float = 1.0, l2_scale: float = 1.0
) -> ASRMSettings:
    """The settings of the baseline run for a synthetic split, with ``padding``."""
    settings = BaselineConfig.from_yaml(BASELINE_CONFIG).reconstruction.settings(
        config.geometry.binning
    )
    return scaled(dataclasses.replace(settings, padding_factor=padding), scale, l2_scale)


def write_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")


def cmd_measured(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.samples:
        target = args.out / f"{name}.json"
        if target.exists():
            print(f"{name}: {target} exists, skipped")
            continue
        sample = MEASURED_SAMPLES[name]
        padding = None if args.paper_padding else args.padding
        settings, nominal = measured_settings(
            sample, args.binning, padding, args.iteration_scale, args.l2_scale
        )
        hologram = load_measured(args.holograms, sample, args.binning)
        geometry = sample.geometry(args.binning)
        grids = [
            stage_grid(tuple(hologram.shape), s.down_sampling, settings.padding_factor).shape
            for s in settings.stages
        ]
        print(
            f"{name}: {tuple(hologram.shape)} px, arrays {grids}, Fr {geometry.fresnel_number:.4e}"
        )
        warm, initial = None, None
        if args.warm_start is not None:
            source = args.warm_start / f"{name}.json"
            estimate = float(json.loads(source.read_text())["z01_est"])
            warm = WarmStart(
                str(source), file_sha256(source), args.half_width, 0.0, {0: estimate}, {}
            )
            initial = warm.simplex(0, *sample.simplex)
            print(
                f"  warm start from {estimate * 1e3:.3f} mm: "
                f"simplex {initial[0] * 1e3:.3f} / {initial[1] * 1e3:.3f} mm"
            )
        result = find_focus(
            hologram,
            geometry,
            settings,
            confidence=sample.confidence,
            initial_simplex=initial,
            log=lambda msg: print(f"  {msg}", flush=True),
        )
        np.save(args.out / f"{name}-phase.npy", result.reconstruction.obj.real.numpy())
        record = {
            "sample": name,
            "label": sample.label,
            "binning": args.binning,
            "cutting_band": sample.cutting_band,
            "input_shape": list(hologram.shape),
            "padding_factor": nominal,
            "padding_factor_input": settings.padding_factor,
            "stage_arrays": [list(g) for g in grids],
            "settings": plain(dataclasses.asdict(settings)),
            "geometry_guess": geometry.as_dict(),
            "paper2": {
                "simplex": list(sample.simplex),
                "z01_nm": sample.z01_nm,
                "z01_visual": sample.z01_visual,
                "samples": sample.nm_samples,
                "padding_factor": sample.padding_factor,
            },
            "result": result.to_dict(),
            "z01_minus_paper2_nm_um": (result.z01 - sample.z01_nm) * 1e6,
            "z01_minus_visual_um": (result.z01 - sample.z01_visual) * 1e6,
            "seconds_per_reconstruction": float(
                np.mean([e.seconds for e in result.evaluations if e.reconstructed])
            ),
            **environment(),
        }
        if warm is not None:
            record["warm_start"] = {
                **warm.identity(),
                "z01_est": warm.centers[0],
                "initial_simplex": list(initial),
            }
        write_json(target, record)
        print(
            f"{name}: z01 = {result.z01 * 1e3:.3f} mm ([2]: {sample.z01_nm * 1e3:.1f} mm), "
            f"{result.n_reconstructions} reconstructions, {result.seconds:.0f} s"
        )


def curve_record(curve: dict, meta: dict, settings: ASRMSettings, seconds: float) -> dict:
    return {
        **meta,
        "settings": plain(dataclasses.asdict(settings)),
        "curve": {k: [float(x) for x in v] for k, v in curve.items()},
        "seconds": seconds,
        **environment(),
    }


def cmd_curves(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    dz = symmetric_offsets(args.points, args.half_range)
    log = (lambda msg: print(f"  {msg}", flush=True)) if args.verbose else None
    if args.data is not None:
        config = read_config(args.data)
        labels = read_labels(args.data)
        settings = synthetic_settings(config, args.padding, args.iteration_scale, args.l2_scale)
        for index in args.indices:
            target = args.out / f"synthetic-{index}.json"
            if target.exists():
                print(f"sample {index}: {target} exists, skipped")
                continue
            with h5py.File(args.data, "r") as fh:
                hologram = torch.from_numpy(np.asarray(fh["hologram"][index], dtype=np.float32))
                phase = torch.from_numpy(np.asarray(fh["phase"][index], dtype=np.float32))
            z01 = float(labels["z01"][index])
            start = time.perf_counter()
            curve = reconstruction_focus_curve(
                hologram, config.geometry.at(z01), dz, settings, phase_truth=phase, log=log
            )
            meta = {
                "kind": "synthetic",
                "name": f"test sample {index}",
                "data": str(args.data),
                "index": index,
                "phantom_type": int(labels["phantom_type"][index]),
                "photons": float(labels["photons"][index]),
                "z01_reference": z01,
                "reference": "true z01",
            }
            write_json(target, curve_record(curve, meta, settings, time.perf_counter() - start))
            print(f"sample {index}: {len(dz)} reconstructions, {time.perf_counter() - start:.0f} s")
    for name in args.samples or []:
        target = args.out / f"measured-{name}.json"
        if target.exists():
            print(f"{name}: {target} exists, skipped")
            continue
        sample = MEASURED_SAMPLES[name]
        settings, nominal = measured_settings(
            sample, args.binning, args.padding, args.iteration_scale, args.l2_scale
        )
        hologram = load_measured(args.holograms, sample, args.binning)
        start = time.perf_counter()
        curve = reconstruction_focus_curve(
            hologram, sample.geometry(args.binning, sample.z01_visual), dz, settings, log=log
        )
        meta = {
            "kind": "measured",
            "name": sample.label,
            "sample": name,
            "binning": args.binning,
            "padding_factor": nominal,
            "z01_reference": sample.z01_visual,
            "reference": "visually determined z01 of [2] Table 1",
        }
        write_json(target, curve_record(curve, meta, settings, time.perf_counter() - start))
        print(f"{name}: {len(dz)} reconstructions, {time.perf_counter() - start:.0f} s")


def bias_holograms(path: Path, index: int) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Sample ``index`` of a split under the data models of :data:`BIAS_VARIANTS`.

    The sample is regenerated from its seed; the stored phase must match. Returns the
    holograms, the binned phase and the regeneration check.
    """
    config = read_config(path)
    with h5py.File(path, "r") as fh:
        split = str(fh.attrs["split"])
        noisy = np.asarray(fh["hologram"][index], dtype=np.float32)
        phase = np.asarray(fh["phase"][index], dtype=np.float32)
    generator = HologramGenerator(config, device="cpu")
    sample = generator.sample(split, index)
    phase_difference = float(np.abs(sample.phase - phase).max())
    if phase_difference > 1e-6:
        raise RuntimeError(f"sample {index} does not regenerate (phase off by {phase_difference})")
    s = config.simulation.oversample
    obj = object_from_phase(
        torch.from_numpy(phase).to(generator.real_dtype), config.phantoms.delta_beta
    )
    grid = simulate_hologram(
        obj,
        config.geometry.at(sample.labels["z01"]).fresnel_number,
        padded_shape=tuple(n // s for n in generator.simulation_shape),
        a0=sample.meta["a0"],
    ).numpy()
    holograms = {"noisy": noisy, "clean": sample.clean, "grid": grid, "grid-nofilter": grid}
    check = {
        "split": split,
        "phase_max_difference": phase_difference,
        "noisy_max_difference": float(np.abs(sample.hologram - noisy).max()),
    }
    return holograms, phase, check


def without_object_filters(settings: ASRMSettings) -> ASRMSettings:
    stages = tuple(
        dataclasses.replace(s, phase_fwhm=0.0, absorption_fwhm=0.0) for s in settings.stages
    )
    return dataclasses.replace(settings, stages=stages)


def parabola_minimum(dz: np.ndarray, values: np.ndarray, half_width: int = 2) -> float:
    """Vertex of the parabola through the points around the smallest value (``nan`` if none)."""
    k = int(np.argmin(values))
    lo, hi = max(0, k - half_width), min(len(dz), k + half_width + 1)
    a, b, _ = np.polyfit(dz[lo:hi], values[lo:hi], 2)
    return float(-b / (2 * a)) if a > 0 else float("nan")


def cmd_bias(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    config = read_config(args.data)
    labels = read_labels(args.data)
    settings = synthetic_settings(config, args.padding, args.iteration_scale, args.l2_scale)
    dz = symmetric_offsets(args.points, args.half_range)
    log = (lambda msg: print(f"  {msg}", flush=True)) if args.verbose else None
    for index in args.indices:
        target = args.out / f"bias-{index}.json"
        record = json.loads(target.read_text()) if target.exists() else None
        todo = [v for v in args.variants if record is None or v not in record["curves"]]
        if not todo:
            print(f"sample {index}: {target} complete, skipped")
            continue
        holograms, phase, check = bias_holograms(args.data, index)
        z01 = float(labels["z01"][index])
        if record is None:
            record = {
                "kind": "bias",
                "data": str(args.data),
                "index": index,
                "phantom_type": int(labels["phantom_type"][index]),
                "photons": float(labels["photons"][index]),
                "phase_shift": float(labels["phase_shift"][index]),
                "z01": z01,
                "regeneration": check,
                "settings": plain(dataclasses.asdict(settings)),
                "variants": {},
                "curves": {},
                "minimum_um": {},
                **environment(),
            }
        for variant in todo:
            start = time.perf_counter()
            curve = reconstruction_focus_curve(
                torch.from_numpy(holograms[variant]),
                config.geometry.at(z01),
                dz,
                without_object_filters(settings) if variant == "grid-nofilter" else settings,
                phase_truth=torch.from_numpy(phase),
                log=log,
            )
            record["variants"][variant] = BIAS_VARIANTS[variant]
            record["curves"][variant] = {k: [float(x) for x in v] for k, v in curve.items()}
            record["minimum_um"][variant] = {
                "argmin": float(dz[np.argmin(curve["MFE"])] * 1e6),
                "parabola": parabola_minimum(dz, curve["MFE"]) * 1e6,
            }
            write_json(target, record)
            print(
                f"sample {index} {variant}: MFE minimum at "
                f"{record['minimum_um'][variant]['parabola']:+.0f} um (parabola), "
                f"{time.perf_counter() - start:.0f} s",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, help="torch CPU threads")
    parser.add_argument(
        "--iteration-scale", type=float, default=1.0, help="scale the ASRM iterations (smoke test)"
    )
    parser.add_argument(
        "--l2-scale",
        type=float,
        default=1.0,
        help="scale the L2 weight on the absorption of every stage (padding experiments)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    measured = sub.add_parser("measured", help="autofocus of the measured holograms")
    measured.add_argument("--holograms", type=Path, required=True)
    measured.add_argument("--samples", nargs="+", default=list(MEASURED_SAMPLES))
    measured.add_argument("--binning", type=int, default=4)
    measured.add_argument("--padding", type=float, default=2.0)
    measured.add_argument(
        "--paper-padding", action="store_true", help="array sizes of [2] Table 1 (p = 7 or 4)"
    )
    measured.add_argument(
        "--warm-start", type=Path, help="directory with <sample>.json holding z01_est (metres)"
    )
    measured.add_argument(
        "--half-width", type=float, default=5e-4, help="half width of the warm start (metres)"
    )
    measured.add_argument("--out", type=Path, required=True)
    measured.set_defaults(func=cmd_measured)

    curves = sub.add_parser("curves", help="focus curves with one reconstruction per point")
    curves.add_argument("--data", type=Path, help="synthetic split (HDF5 with phase)")
    curves.add_argument("--indices", type=int, nargs="+", default=[])
    curves.add_argument("--holograms", type=Path)
    curves.add_argument("--samples", nargs="+", choices=list(MEASURED_SAMPLES))
    curves.add_argument("--binning", type=int, default=4)
    curves.add_argument("--padding", type=float, default=2.0)
    curves.add_argument("--points", type=int, default=41)
    curves.add_argument("--half-range", type=float, default=10e-3, help="metres")
    curves.add_argument("--verbose", action="store_true")
    curves.add_argument("--out", type=Path, required=True)
    curves.set_defaults(func=cmd_curves)

    bias = sub.add_parser("bias", help="MFE near the true z01 under several data models")
    bias.add_argument("--data", type=Path, required=True, help="synthetic split with phase")
    bias.add_argument("--indices", type=int, nargs="+", required=True)
    bias.add_argument(
        "--variants", nargs="+", choices=list(BIAS_VARIANTS), default=list(BIAS_VARIANTS)
    )
    bias.add_argument("--padding", type=float, default=2.0)
    bias.add_argument("--points", type=int, default=7)
    bias.add_argument("--half-range", type=float, default=0.6e-3, help="metres")
    bias.add_argument("--verbose", action="store_true")
    bias.add_argument("--out", type=Path, required=True)
    bias.set_defaults(func=cmd_bias)

    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.command == "curves" and args.samples and args.holograms is None:
        parser.error("--samples needs --holograms")
    args.func(args)


if __name__ == "__main__":
    main()
