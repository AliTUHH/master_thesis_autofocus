"""Cross-check the ASRM port and the nested autofocus against the paper code (holowizard 3.0.6).

``reconstruct`` runs one multi-stage reconstruction with both codes on the same hologram and
the same Fresnel number and compares the loss of every iteration and the reconstructed
object. ``focus`` runs both autofocus searches (holowizard ``find_focus`` and
``holofocus.autofocus.find_focus``) and compares the probing points.

Runs in the holowizard environment (holowizard pins Python 3.11) with this repository
installed without dependencies::

    uv venv --python 3.11 /tmp/hw/venv311
    VIRTUAL_ENV=/tmp/hw/venv311 uv pip install holowizard==3.0.6 h5py
    VIRTUAL_ENV=/tmp/hw/venv311 uv pip install --no-deps -e .
    /tmp/hw/venv311/bin/python scripts/crosscheck_asrm_holowizard.py reconstruct \\
        --data data/cpu-4x/test.h5 --index 0 --out /tmp/xcheck
    /tmp/hw/venv311/bin/python scripts/crosscheck_asrm_holowizard.py focus \\
        --data data/cpu-4x/test.h5 --index 0 --iteration-scale 0.1 --out /tmp/xcheck

holowizard rotates the hologram by 90 degrees inside ``process_image``; it is given the
hologram rotated by -90 degrees so that both codes work on the same array.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch
from holowizard.core.api.functions.find_focus.find_focus import find_focus as hw_find_focus
from holowizard.core.api.functions.single_projection.reconstruction import (
    reconstruct as hw_reconstruct,
)
from holowizard.core.api.parameters import (
    BeamSetup,
    DataDimensions,
    Measurement,
    Options,
    Padding,
    RecoParams,
    Regularization,
)
from holowizard.core.logging.logger import Logger
from holowizard.core.models.cone_beam import ConeBeam

from holofocus.asrm import ASRMSettings, asrm_reconstruct
from holofocus.autofocus.search import find_focus
from holofocus.dataset.storage import read_config, read_labels


def hw_options(settings: ASRMSettings, z01_tol_mm: float) -> list[Options]:
    options = []
    for stage in settings.stages:
        padding = Padding(
            padding_mode=Padding.PaddingMode.MIRROR_ALL,
            padding_factor=settings.padding_factor,
            down_sampling_factor=stage.down_sampling,
            cutting_band=0,
            a0=settings.a0,
        )
        options.append(
            Options(
                regularization_object=Regularization(
                    iterations=stage.iterations,
                    update_rate=stage.step,
                    l2_weight=complex(0.0, stage.l2_weight),
                    gaussian_filter_fwhm=complex(stage.phase_fwhm, stage.absorption_fwhm),
                ),
                nesterov_object=Regularization(
                    update_rate=stage.momentum,
                    gaussian_filter_fwhm=complex(stage.momentum_fwhm, stage.momentum_fwhm),
                ),
                verbose_interval=10**9,
                padding=deepcopy(padding),
                z01_tol=z01_tol_mm,
            )
        )
    return options


def hw_params(hologram, setup, z01_mm, confidence_mm, settings, z01_tol_mm) -> RecoParams:
    rotated = np.ascontiguousarray(np.rot90(hologram, -1)).astype(np.float32)
    shape = rotated.shape
    width = settings.fade_width
    return RecoParams(
        beam_setup=deepcopy(setup),
        output_path="",
        measurements=[
            Measurement(data=rotated, z01=z01_mm, z01_confidence=confidence_mm, data_path="")
        ],
        reco_options=hw_options(settings, z01_tol_mm),
        data_dimensions=DataDimensions(
            total_size=shape,
            fov_size=shape,
            window_type="blackman",
            fading_width=[(width, width), (width, width)],
        ),
    )


def load(args):
    config = read_config(args.data)
    labels = read_labels(args.data)
    with h5py.File(args.data, "r") as fh:
        hologram = np.asarray(fh["hologram"][args.index], dtype=np.float32)
    geometry = config.geometry
    setup = BeamSetup(
        energy=geometry.energy_kev, px_size=geometry.pixel_size * 1e3, z02=geometry.z02 * 1e3
    )
    settings = ASRMSettings.for_binning(geometry.binning, padding_factor=args.padding_factor)
    if args.iteration_scale != 1.0:
        stages = tuple(
            replace(s, iterations=max(1, round(s.iterations * args.iteration_scale)))
            for s in settings.stages
        )
        settings = replace(settings, stages=stages)
    return config, labels, hologram, setup, settings


def cmd_reconstruct(args) -> dict:
    config, labels, hologram, setup, settings = load(args)
    z01_mm = float(labels["z01"][args.index]) * 1e3
    fr = ConeBeam.get_fr(setup, Measurement(z01=z01_mm), verbose=False)
    params = hw_params(hologram, setup, z01_mm, 5.0, settings, 0.1)
    start = time.perf_counter()
    hw_obj, hw_loss = hw_reconstruct(params, viewer=[])
    hw_seconds = time.perf_counter() - start
    hw_obj = hw_obj.cpu()
    hw_loss = hw_loss.cpu().numpy().astype(np.float64)

    start = time.perf_counter()
    ours = asrm_reconstruct(torch.from_numpy(hologram), fr, settings)
    our_seconds = time.perf_counter() - start
    our_loss = np.asarray(ours.loss)
    rel = np.abs(our_loss / hw_loss - 1)
    diff = ours.obj - hw_obj
    report = {
        "index": args.index,
        "z01_mm": z01_mm,
        "fresnel_number": fr,
        "settings": str(settings),
        "holowizard_seconds": hw_seconds,
        "holofocus_seconds": our_seconds,
        "final_loss": {"holowizard": float(hw_loss[-1]), "holofocus": float(our_loss[-1])},
        "final_loss_rel_diff": float(rel[-1]),
        "loss_rel_diff_max": float(rel.max()),
        "loss_rel_diff_median": float(np.median(rel)),
        "stage_end_loss": {
            "holowizard": [float(hw_loss[e - 1]) for e in ours.stage_ends],
            "holofocus": [float(our_loss[e - 1]) for e in ours.stage_ends],
        },
        "phase_max_abs_diff": float(diff.real.abs().max()),
        "phase_rms_diff": float(diff.real.square().mean().sqrt()),
        "phase_rms": float(hw_obj.real.square().mean().sqrt()),
        "absorption_max_abs_diff": float(diff.imag.abs().max()),
        "stage_ends": list(ours.stage_ends),
        "loss": {"holowizard": hw_loss.tolist(), "holofocus": our_loss.tolist()},
    }
    np.savez_compressed(
        Path(args.out) / f"reconstruct_{args.index}.npz",
        holowizard_loss=hw_loss,
        holofocus_loss=our_loss,
        holowizard_phase=hw_obj.real.numpy(),
        holofocus_phase=ours.obj.real.numpy(),
        stage_ends=np.asarray(ours.stage_ends),
    )
    return report


def cmd_focus(args) -> dict:
    config, labels, hologram, setup, settings = load(args)
    z01_true = float(labels["z01"][args.index])
    z01_guess = float(labels["z01_guess"][args.index])
    confidence, xatol = 5e-3, 1e-4

    params = hw_params(hologram, setup, z01_guess * 1e3, confidence * 1e3, settings, xatol * 1e3)
    start = time.perf_counter()
    hw_z01, hw_points, hw_values = hw_find_focus(params, viewer=[], plotter=None)
    hw_seconds = time.perf_counter() - start

    geometry = config.geometry.at(z01_guess)
    fr_hw = ConeBeam.get_fr(setup, Measurement(z01=z01_guess * 1e3), verbose=False)
    start = time.perf_counter()
    ours = find_focus(
        torch.from_numpy(hologram), geometry, settings, confidence=confidence, xatol=xatol
    )
    our_seconds = time.perf_counter() - start
    our_points = [e.z01 for e in ours.evaluations if e.reconstructed]
    our_values = [e.mfe for e in ours.evaluations if e.reconstructed]
    n = min(len(hw_points), len(our_points))
    return {
        "index": args.index,
        "z01_true_mm": z01_true * 1e3,
        "z01_guess_mm": z01_guess * 1e3,
        "fresnel_number_rel_diff": geometry.fresnel_number / fr_hw - 1,
        "settings": str(settings),
        "holowizard": {
            "z01_mm": float(hw_z01),
            "points_mm": [float(z) for z in hw_points],
            "mfe": [float(v) for v in hw_values],
            "reconstructions": len(hw_points),
            "seconds": hw_seconds,
        },
        "holofocus": {
            "z01_mm": ours.z01 * 1e3,
            "points_mm": [z * 1e3 for z in our_points],
            "mfe": our_values,
            "reconstructions": ours.n_reconstructions,
            "evaluations": ours.n_evaluations,
            "seconds": our_seconds,
        },
        "estimate_diff_um": (ours.z01 * 1e3 - float(hw_z01)) * 1e3,
        "same_points": bool(
            len(hw_points) == len(our_points)
            and all(
                math.isclose(a, b * 1e3, rel_tol=0, abs_tol=1e-9)
                for a, b in zip(hw_points, our_points, strict=False)
            )
        ),
        "points_max_abs_diff_um": float(
            max(abs(hw_points[i] - our_points[i] * 1e3) for i in range(n)) * 1e3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("reconstruct", "focus"))
    parser.add_argument("--data", type=Path, required=True, help="HDF5 split file")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--padding-factor", type=float, default=2.0)
    parser.add_argument("--iteration-scale", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("/tmp/xcheck"))
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp/holowizard-logs"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    Logger.current_log_level = Logger.level_num_loss
    Logger.configure(session_name="xcheck", working_dir=str(args.log_dir) + "/")
    report = cmd_reconstruct(args) if args.command == "reconstruct" else cmd_focus(args)
    path = args.out / f"{args.command}_{args.index}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
