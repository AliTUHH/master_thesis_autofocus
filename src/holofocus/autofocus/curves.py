"""Focus curves: criteria sampled over the assumed de-focus ``dz01 = z01_hat - z01``.

* :func:`simulated_focus_curve` follows [2] Sec. 3.5 / Fig. 7: no reconstruction. The true
  exit wave is propagated to every de-focus position ([2] Eq. (11)), projected onto
  ``Omega_P`` ([2] Eq. (12)-(13)), and the result feeds the MFE (with a noise-free hologram)
  and the image criteria (on its phase).
* :func:`reconstruction_focus_curve` follows [2] Sec. 3.6 / Fig. 8: one ASRM reconstruction
  per sampling point; the MFE is its final loss, the image criteria use its phase.

Here ``dz01`` is the error of the assumed distance as on the axes of [2] Fig. 7-8, i.e.
minus the dataset label ``z01 - z01_guess``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch

from ..arrays import center_slices, pad_center
from ..asrm import ASRMSettings, asrm_reconstruct
from ..constraints import project_wavefield
from ..forward import exit_wave
from ..geometry import ConeBeamGeometry
from ..propagation import FresnelPropagator
from .criteria import IMAGE_CRITERIA, image_criteria, model_fit_error


def simulated_focus_curve(
    obj: torch.Tensor,
    geometry: ConeBeamGeometry,
    offsets: Sequence[float],
    padded_shape: tuple[int, int],
    fwhm: float = 4.0,
) -> dict[str, np.ndarray]:
    """MFE and image criteria of the projected de-focused exit wave ([2] Fig. 7).

    ``obj`` is the ground-truth ``O~`` on the detector field of view of ``geometry``; it is
    embedded in free space of ``padded_shape``. Returns arrays over ``offsets`` (metres) for
    ``"MFE"`` and every name in :data:`IMAGE_CRITERIA`, plus ``"dz01"``.
    """
    shape = tuple(obj.shape[-2:])
    fov = center_slices(tuple(padded_shape), shape)
    psi_exit = exit_wave(pad_center(obj, tuple(padded_shape), 0.0))
    detector = FresnelPropagator(padded_shape, geometry.fresnel_number, obj.device, psi_exit.dtype)
    sqrt_hologram = detector(psi_exit).abs()
    out: dict[str, list[float]] = {"dz01": [], "MFE": [], **{k: [] for k in IMAGE_CRITERIA}}
    for dz in offsets:
        defocus = FresnelPropagator(
            padded_shape, geometry.fresnel_number_at(dz), obj.device, psi_exit.dtype
        )
        projected = project_wavefield(defocus(psi_exit))
        out["dz01"].append(float(dz))
        out["MFE"].append(
            model_fit_error(projected, sqrt_hologram, geometry.defocused(dz).fresnel_number, fov)
        )
        phase = torch.angle(projected[fov]).to(obj.real.dtype)
        for name, value in image_criteria(phase, fwhm).items():
            out[name].append(value)
    return {k: np.asarray(v) for k, v in out.items()}


def reconstruction_focus_curve(
    hologram: torch.Tensor,
    geometry: ConeBeamGeometry,
    offsets: Sequence[float],
    settings: ASRMSettings | None = None,
    fwhm: float = 4.0,
    phase_truth: torch.Tensor | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray]:
    """MFE and image criteria of ASRM reconstructions at ``geometry.z01 + dz`` ([2] Fig. 8).

    ``geometry.z01`` is the reference distance (truth or visual focus). With ``phase_truth``
    the RMS error of the reconstructed phase is returned as ``"phase_rmse"``.
    """
    settings = ASRMSettings() if settings is None else settings
    keys = ["dz01", "MFE", "seconds", *IMAGE_CRITERIA]
    if phase_truth is not None:
        keys.append("phase_rmse")
    out: dict[str, list[float]] = {k: [] for k in keys}
    for dz in offsets:
        start = time.perf_counter()
        result = asrm_reconstruct(hologram, geometry.defocused(dz).fresnel_number, settings)
        phase = result.obj.real
        out["dz01"].append(float(dz))
        out["MFE"].append(result.final_loss)
        out["seconds"].append(time.perf_counter() - start)
        for name, value in image_criteria(phase, fwhm).items():
            out[name].append(value)
        if phase_truth is not None:
            truth = phase_truth.to(phase.device, phase.dtype)
            if truth.shape != phase.shape:
                raise ValueError("phase_truth must match the field of view of the last stage")
            out["phase_rmse"].append(float((phase - truth).square().mean().sqrt()))
        if log is not None:
            log(f"dz01 = {dz * 1e3:+.3f} mm  MFE = {result.final_loss:.6e}")
    return {k: np.asarray(v) for k, v in out.items()}
