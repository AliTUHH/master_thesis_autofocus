"""Model-based autofocus baseline (HoloWizard ``find_focus``) and a classical sharpness-metric fallback.

Usage::

    python -m src.baseline.model_based_autofocus --data data/processed/small/test.hdf5 [--n 8] [--search-width 50]
        [--preset p05filter_dsf2_1] [--method holowizard|classical|both] [--iterations-scale 1.0]
        [--out reports/baseline_model_based/<name>]

Writes ``samples.csv`` (common schema of :mod:`src.baseline.results`), ``summary.json``, ``histories.json``
(objective values of every evaluation), ``scatter.png`` and ``objective_curves.png``.

Model-based autofocus (Dora et al., Opt. Express 33(4), 6641 (2025); HoloWizard 3.0.6)
---------------------------------------------------------------------------------------
``holowizard.core.api.functions.find_focus.find_focus(reco_params)`` runs a 1-D Nelder-Mead search (scipy)
over the focus-object distance ``z01`` inside ``Measurement.z01_bounds = (z01 - z01_confidence, z01 +
z01_confidence)``; the start simplex is the pair of interval bounds and the search stops when the simplex is
narrower than ``Options.z01_tol`` (P05 default 0.1 mm).  Every objective evaluation is a complete multi-stage
ASRM/PGD reconstruction with the stages given in ``RecoParams.reco_options``; the objective is the data
residual (mean squared amplitude error inside the FOV) of the last iteration.  The Fresnel number is
derived from ``z01`` inside the core (``ConeBeam.get_fr``, identical to :func:`src.utils.physics.fresnel_number`).

Conventions that this wrapper takes care of (see ``reports/baseline_model_based.md``):

1. HoloForge stores the detector **amplitude** ``|psi|``; the core API expects **intensities** and takes the
   square root itself -> :func:`model_based_autofocus` squares the hologram.
2. The core rotates the measurement by 90 degrees (``torch.rot90``) during preprocessing and does not rotate
   the result back; irrelevant for Fr, but reconstructions must be rotated back with ``torch.rot90(x, k=-1)``
   before they are compared with ``images/phantoms`` (:mod:`src.eval.downstream`).
3. ``Logger.configure`` (with ``current_log_level = level_num_header``) must precede the first API call; the
   core registers custom logging levels and writes a geometry file per reconstruction into the log
   directory.  Importing ``holowizard.core`` probes ``nvidia-smi`` and sets ``tempfile.tempdir``; the import
   is therefore lazy (:func:`_import_core`) and the module stays importable without HoloWizard.
4. ``find_focus_z01`` keeps its evaluation history in module globals: never call the model-based autofocus
   concurrently inside one process (parallelise over processes instead).
5. The search runs in ``z01`` [mm]; Fr bounds are converted with ``z01 = Fr lambda z02^2 / (px^2 + Fr lambda z02)``.
6. Accuracy requirement: a relative Fr error ``e`` blurs the reconstruction by ``b = sqrt(|e| / Fr)`` detector
   pixels (:func:`src.utils.fresnel.defocus_blur_px`); ``b <= 1-2`` px is the target.

The classical fallback (:func:`classical_autofocus`) is **not** part of HoloWizard: it back-propagates the
amplitude with ``D_{-Fr}`` and minimises a sharpness criterion (total variation, variance or Laplacian energy
of the back-propagated amplitude, minimal in focus for weak objects) over a logarithmic Fr grid with Brent
refinement.  It is three orders of magnitude faster but fails on holograms with few Fresnel fringes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from src.baseline.results import make_result_row, plot_scatter, summarize_by_method, summary_markdown_table, write_samples_csv
from src.data.forge_samples import ForgeSample, expand_paths, read_forge_samples
from src.utils.config import PROJECT_ROOT, resolve_path
from src.utils.fresnel import fresnel_kernel
from src.utils.physics import fresnel_number, z01_from_fresnel_number

__all__ = [
    "Geometry",
    "AutofocusResult",
    "StageSpec",
    "RecoPreset",
    "PRESETS",
    "DEFAULT_FOCUS_PRESET",
    "DEFAULT_QUALITY_PRESET",
    "SMOKE_PRESET",
    "get_preset",
    "configure_holowizard",
    "default_fading_width_px",
    "build_reco_params",
    "model_based_autofocus",
    "ClassicalFocus",
    "CLASSICAL_METRICS",
    "classical_autofocus",
    "fr_search_bounds",
    "run_baseline",
    "main",
]

METHOD_HOLOWIZARD = "holowizard_find_focus"
CLASSICAL_METRICS: tuple[str, ...] = ("tv", "var", "lap")


# --------------------------------------------------------------------------------------------------
# geometry and result containers
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Geometry:
    """Cone-beam constants in HoloWizard units: photon energy in keV, detector pixel size and z02 in mm.

    ``px_mm`` is the *effective* pixel size of the hologram grid (binned pixel for downsampled data, as
    stored by HoloForge in ``metadata/setup/detector_px_size``).  Conversions delegate to
    :mod:`src.utils.physics` so that there is a single implementation of the Fr <-> z01 relation.
    """

    energy_kev: float
    px_mm: float
    z02_mm: float

    def fresnel(self, z01_mm: float) -> float:
        """Pixel Fresnel number for the focus-object distance ``z01_mm`` (mm)."""
        return float(fresnel_number(z01_mm, self.z02_mm, self.energy_kev, self.px_mm))

    def z01_from_fresnel(self, fr: float) -> float:
        """Focus-object distance in mm for the pixel Fresnel number ``fr``."""
        return float(z01_from_fresnel_number(fr, self.z02_mm, self.energy_kev, self.px_mm))

    def z01_interval(self, fr_bounds: tuple[float, float]) -> tuple[float, float, float, float]:
        """``(z01_lo, z01_hi, centre, half width)`` in mm of a Fresnel-number interval (centre = HoloWizard's
        ``Measurement.z01`` start value, half width = ``z01_confidence``)."""
        fr_lo, fr_hi = _check_bounds(fr_bounds)
        z_lo, z_hi = self.z01_from_fresnel(fr_lo), self.z01_from_fresnel(fr_hi)
        return z_lo, z_hi, 0.5 * (z_lo + z_hi), 0.5 * (z_hi - z_lo)

    @classmethod
    def from_sample(cls, sample: ForgeSample) -> Geometry:
        return cls(energy_kev=sample.energy_kev, px_mm=sample.px_mm, z02_mm=sample.z02_mm)


@dataclass
class AutofocusResult:
    """Outcome of one autofocus run."""

    fr_est: float
    """Estimated pixel Fresnel number."""
    z01_est_mm: float
    """Estimated focus-object distance in mm (``nan`` if no geometry was given to the classical method)."""
    n_evals: int
    """Objective evaluations = reconstructions (model-based) or back-propagations (classical)."""
    runtime_s: float
    z01_history: list[float]
    """Evaluated distances in mm, in evaluation order (``nan`` entries without geometry)."""
    loss_history: list[float]
    """Objective value per evaluation (data residual or sharpness criterion)."""
    method: str
    fr_history: list[float] = field(default_factory=list)
    """Evaluated Fresnel numbers, aligned with ``loss_history``."""
    fr_bounds: tuple[float, float] = (float("nan"), float("nan"))
    preset: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_bounds(fr_bounds: tuple[float, float]) -> tuple[float, float]:
    fr_lo, fr_hi = float(fr_bounds[0]), float(fr_bounds[1])
    if not 0.0 < fr_lo < fr_hi:
        raise ValueError(f"fr_bounds must satisfy 0 < lo < hi, got {fr_bounds}")
    return fr_lo, fr_hi


def fr_search_bounds(fr_true: float, search_width_pct: float) -> tuple[float, float]:
    """Symmetric search interval ``Fr_true (1 -/+ w/100)`` used by the baseline CLIs."""
    width = float(search_width_pct) / 100.0
    if not 0.0 < width < 1.0:
        raise ValueError("search_width_pct must lie in (0, 100)")
    return float(fr_true) * (1.0 - width), float(fr_true) * (1.0 + width)


# --------------------------------------------------------------------------------------------------
# reconstruction stage presets
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StageSpec:
    """One ASRM/PGD stage (``holowizard.core.api.parameters.Options``).

    Filter widths are in pixels of the *stage grid* (hologram downsampled by ``down_sampling_factor``);
    the complex ``fwhm`` values follow HoloWizard: real part = phase channel, imaginary part = absorption.
    """

    iterations: int
    update_rate: float
    l2_absorption: float
    """Imaginary part of ``l2_weight`` (L2 penalty on the absorption channel; the phase is not penalised)."""
    fwhm_object: complex
    fwhm_nesterov: complex
    down_sampling_factor: int
    nesterov_rate: float = 1.0


@dataclass(frozen=True)
class RecoPreset:
    """Named multi-stage configuration; ``purpose`` is ``"focus"`` (autofocus objective) or ``"quality"``."""

    name: str
    purpose: str
    padding_factor: float
    stages: tuple[StageSpec, ...]
    description: str

    def scaled(self, iterations_scale: float) -> RecoPreset:
        """Copy with all iteration counts multiplied by ``iterations_scale`` (at least one iteration)."""
        if iterations_scale <= 0:
            raise ValueError("iterations_scale must be positive")
        if iterations_scale == 1.0:
            return self
        stages = tuple(replace(s, iterations=max(1, int(round(s.iterations * iterations_scale)))) for s in self.stages)
        return replace(self, stages=stages)

    def grid_sizes(self, n_pixels: int) -> list[int]:
        """Computational grid (padded, downsampled) of every stage for an ``n_pixels`` hologram."""
        return [int(round(n_pixels * self.padding_factor / s.down_sampling_factor)) for s in self.stages]

    @property
    def iterations(self) -> list[int]:
        return [s.iterations for s in self.stages]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for s in d["stages"]:
            s["fwhm_object"] = str(s["fwhm_object"])
            s["fwhm_nesterov"] = str(s["fwhm_nesterov"])
        return d


DEFAULT_FOCUS_PRESET = "p05filter_dsf2_1"
DEFAULT_QUALITY_PRESET = "quality_256"
SMOKE_PRESET = "smoke"

PRESETS: dict[str, RecoPreset] = {
    "p05filter_dsf2_1": RecoPreset(
        name="p05filter_dsf2_1",
        purpose="focus",
        padding_factor=2.0,
        stages=(
            StageSpec(300, 0.9, 1.0, 2 + 8j, 8 + 8j, down_sampling_factor=2, nesterov_rate=1.0),
            StageSpec(300, 1.1, 1.0, 1 + 8j, 4 + 4j, down_sampling_factor=1, nesterov_rate=1.0),
        ),
        description=(
            "Autofocus objective for 256-px holograms (default): P05 regularisation (l2 1j on the absorption, "
            "fwhm 2+8j / 1+8j, Nesterov 1.0) but stages on dsf 2 (128 px FOV, 256^2 grid) and dsf 1 (512^2 grid) "
            "instead of the P05 dsf 16/4. Only variant with a smooth objective at 256 px (see scan)."
        ),
    ),
    "p05scaled_dsf4_2": RecoPreset(
        name="p05scaled_dsf4_2",
        purpose="focus",
        padding_factor=2.0,
        stages=(
            StageSpec(300, 0.9, 10.0, 2 + 0j, 8 + 8j, down_sampling_factor=4, nesterov_rate=1.0),
            StageSpec(200, 1.1, 1.0, 2 + 8j, 16 + 16j, down_sampling_factor=2, nesterov_rate=1.0),
        ),
        description=(
            "P05 find_focus stages scaled 1:1 from 2048 to 256 px (dsf 4 -> 64 px, dsf 2 -> 128 px). The data "
            "residual has NO minimum near the true Fr on 256-px HoloForge data (monotonic towards small Fr); kept "
            "for reproducing the negative result."
        ),
    ),
    "p05_default_2048": RecoPreset(
        name="p05_default_2048",
        purpose="focus",
        padding_factor=4.0,
        stages=(
            StageSpec(700, 0.9, 10.0, 2 + 0j, 8 + 8j, down_sampling_factor=16, nesterov_rate=1.0),
            StageSpec(300, 1.1, 10.0, 2 + 8j, 16 + 16j, down_sampling_factor=4, nesterov_rate=1.0),
            StageSpec(500, 1.1, 1.0, 2 + 8j, 16 + 16j, down_sampling_factor=4, nesterov_rate=1.0),
        ),
        description=(
            "holowizard/pipe/scripts/config/find_focus/defaults.yaml (P05 production autofocus for 2048-px "
            "detectors): padding 4, dsf 16/4/4, 700/300/500 iterations, l2 10j/10j/1j, z01_tol 0.1 mm. "
            "About 7.7 min per evaluation and 2.7 h per hologram on 2 CPU threads (GPU required in practice)."
        ),
    ),
    "quality_256": RecoPreset(
        name="quality_256",
        purpose="quality",
        padding_factor=2.0,
        stages=(
            StageSpec(300, 0.9, 10.0, 2 + 0j, 4 + 4j, down_sampling_factor=4, nesterov_rate=0.9),
            StageSpec(200, 1.1, 1.0, 2 + 8j, 8 + 8j, down_sampling_factor=2, nesterov_rate=0.9),
            StageSpec(300, 1.1, 0.1, 0 + 2j, 2 + 2j, down_sampling_factor=1, nesterov_rate=0.9),
        ),
        description=(
            "Downstream reconstruction at 256 px (pipe reconstruction/defaults_thick.yaml scaled to 256 px): "
            "warm-up on dsf 4, dsf 2, full resolution with weak filters (fwhm 0+2j; the P05 widths designed "
            "for 2048 px double the phase at 256 px). Nesterov 0.9 instead of 1.0: with 1.0 some phantoms "
            "diverge (phase -> -600 rad, loss ~ 1, NaN). Not suitable as autofocus objective: without the "
            "strong absorption regularisation the residual decreases monotonically towards larger Fr."
        ),
    ),
    "smoke": RecoPreset(
        name="smoke",
        purpose="focus",
        padding_factor=2.0,
        stages=(
            StageSpec(20, 0.9, 1.0, 2 + 4j, 4 + 4j, down_sampling_factor=2, nesterov_rate=0.9),
            StageSpec(20, 1.1, 1.0, 1 + 4j, 2 + 2j, down_sampling_factor=1, nesterov_rate=0.9),
        ),
        description=(
            "Tiny two-stage configuration for tests and smoke runs (about 0.1 s per evaluation at 64 px). "
            "Exercises the whole pipeline but its objective is not accurate."
        ),
    ),
}
"""Stage presets.

Rationale and limits (prototype study on 256-px HoloForge data, ``reports/baseline_model_based.md``):

* At 256 px only ``p05filter_dsf2_1`` (full 256 px in the last stage + P05 absorption regularisation) gives a
  smooth objective with a minimum near ``Fr_true``.  The P05 stages scaled 1:1 (``p05scaled_dsf4_2``: 64/128 px
  grids) have no usable minimum; weakly regularised stages (quality presets) give a noisy or monotonic
  objective because the absorption channel fits the data for too large Fr.
* ``p05filter_dsf2_1`` is biased by -5 ... -10 % for ``Fr >= 1.5e-2`` in the 256-px setting (asymmetric valley
  of the objective); for ``Fr <= 6e-3`` the error stays within about 2 %.  The achievable accuracy is also bounded
  by ``z01_tol`` (0.1 mm ~ 0.04-0.2 % Fr) in the small-Fr regime.
* Nesterov momentum 1.0 (P05 default) diverged for some phantoms with weak regularisation; the autofocus
  presets keep 1.0 (strong regularisation, no divergence observed), ``quality_256`` uses 0.9.
* Iteration counts scale with ``iterations_scale`` (:meth:`RecoPreset.scaled`); 150/150 iterations roughly halve
  the runtime of ``p05filter_dsf2_1`` (about 1.8 min per hologram on 2 CPU threads instead of 3.5 min).
"""


def get_preset(preset: str | RecoPreset, iterations_scale: float = 1.0) -> RecoPreset:
    """Resolve a preset name (or pass a :class:`RecoPreset` through) and apply ``iterations_scale``."""
    if isinstance(preset, RecoPreset):
        return preset.scaled(iterations_scale)
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; available: {sorted(PRESETS)}")
    return PRESETS[preset].scaled(iterations_scale)


# --------------------------------------------------------------------------------------------------
# lazy HoloWizard import and one-time configuration
# --------------------------------------------------------------------------------------------------
_CORE: SimpleNamespace | None = None
_LOGGER_STATE: dict[str, Any] = {"configured": False, "working_dir": None}


def _import_core() -> SimpleNamespace:
    """Import ``holowizard.core`` lazily (it probes ``nvidia-smi`` and sets ``tempfile.tempdir`` on import)."""
    global _CORE
    if _CORE is not None:
        return _CORE
    try:
        import holowizard.core as hw_core
        from holowizard.core.api.functions.find_focus.find_focus import find_focus
        from holowizard.core.api.functions.single_projection.reconstruction import reconstruct
        from holowizard.core.api.parameters import BeamSetup, DataDimensions, Measurement, Options, Padding, RecoParams, Regularization
        from holowizard.core.logging.logger import Logger
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "holowizard.core is required for the model-based autofocus and the downstream reconstruction "
            "(pip install holowizard==3.0.6, Python 3.11; see requirements.txt). The CTF ring fit and the "
            "classical fallback work without it."
        ) from exc
    _CORE = SimpleNamespace(
        find_focus=find_focus,
        reconstruct=reconstruct,
        BeamSetup=BeamSetup,
        DataDimensions=DataDimensions,
        Measurement=Measurement,
        Options=Options,
        Padding=Padding,
        RecoParams=RecoParams,
        Regularization=Regularization,
        Logger=Logger,
        device=hw_core.torch_running_device,
    )
    return _CORE


def configure_holowizard(
    working_dir: str | Path | None = None,
    session_name: str = "autofocus",
    threads: int | None = 2,
    verbose: bool = False,
) -> Path:
    """Configure HoloWizard's logger once per process and limit the PyTorch CPU threads.

    HoloWizard registers custom logging levels in ``Logger.configure`` and writes one small geometry text
    file per reconstruction into ``<working_dir>/<session_name>/``; without the call the first API use fails.
    ``working_dir`` defaults to ``$HOLOWIZARD_LOG_DIR`` or a fresh temporary directory.  With
    ``verbose=False`` the console output (header, per-iteration losses) is suppressed.  Repeated calls only
    update the thread count and return the directory chosen first.
    """
    core = _import_core()
    if threads is not None:
        torch.set_num_threads(int(threads))
    if _LOGGER_STATE["configured"]:
        return _LOGGER_STATE["working_dir"]
    if working_dir is None:
        working_dir = os.environ.get("HOLOWIZARD_LOG_DIR") or tempfile.mkdtemp(prefix="holowizard_logs_")
    log_dir = Path(working_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    logger_cls = core.Logger
    logger_cls.current_log_level = logger_cls.level_num_header if verbose else logger_cls.level_num_header + 1
    logger_cls.configure(working_dir=str(log_dir), session_name=session_name)
    logging.getLogger(session_name).setLevel(logger_cls.current_log_level)
    _LOGGER_STATE.update(configured=True, working_dir=log_dir)
    return log_dir


def default_fading_width_px(n_pixels: int) -> int:
    """Blackman fading width of HoloWizard's FOV window: the 80-px default (P05, 2048 px) is kept for holograms
    with >= 256 px and scaled down proportionally (minimum 4 px) for smaller test grids."""
    return 80 if n_pixels >= 256 else max(4, int(round(80 * n_pixels / 256)))


def build_reco_params(
    hologram_intensity: np.ndarray,
    geom: Geometry,
    z01_mm: float,
    preset: RecoPreset,
    z01_confidence_mm: float = 5.0,
    z01_tol_mm: float = 0.1,
    a0: float = 1.0,
    fading_width_px: int | None = None,
    window_type: str = "blackman",
) -> Any:
    """``RecoParams`` for one hologram.  ``hologram_intensity`` must be ``|psi|^2`` (the API applies ``sqrt``).

    Each :class:`StageSpec` becomes an ``Options`` object with MIRROR_ALL padding, the preset's padding
    factor and the stage's downsampling factor; ``z01_tol_mm`` is the Nelder-Mead termination tolerance of
    ``find_focus`` and ``z01_confidence_mm`` the half width of its search interval.
    """
    core = _import_core()
    intensity = np.ascontiguousarray(hologram_intensity, dtype=np.float32)
    if intensity.ndim != 2:
        raise ValueError(f"expected a 2-D hologram, got shape {intensity.shape}")
    n_rows, n_cols = intensity.shape
    stages = [
        core.Options(
            regularization_object=core.Regularization(
                iterations=int(s.iterations),
                update_rate=float(s.update_rate),
                l2_weight=complex(0.0, float(s.l2_absorption)),
                gaussian_filter_fwhm=complex(s.fwhm_object),
            ),
            nesterov_object=core.Regularization(update_rate=float(s.nesterov_rate), gaussian_filter_fwhm=complex(s.fwhm_nesterov)),
            verbose_interval=10**6,
            z01_tol=float(z01_tol_mm),
            padding=core.Padding(
                padding_mode=core.Padding.PaddingMode.MIRROR_ALL,
                padding_factor=preset.padding_factor,
                down_sampling_factor=int(s.down_sampling_factor),
                cutting_band=0,
                a0=float(a0),
            ),
        )
        for s in preset.stages
    ]
    fading = default_fading_width_px(min(n_rows, n_cols)) if fading_width_px is None else int(fading_width_px)
    data_dimensions = core.DataDimensions(
        total_size=(n_rows, n_cols),
        fov_size=(n_rows, n_cols),
        window_type=window_type,
        fading_width=[(fading, fading), (fading, fading)],
    )
    measurement = core.Measurement(data=intensity.copy(), z01=float(z01_mm), z01_confidence=float(z01_confidence_mm))
    return core.RecoParams(
        beam_setup=core.BeamSetup(energy=geom.energy_kev, px_size=geom.px_mm, z02=geom.z02_mm),
        measurements=[measurement],
        reco_options=stages,
        data_dimensions=data_dimensions,
        output_path="",
    )


# --------------------------------------------------------------------------------------------------
# model-based autofocus
# --------------------------------------------------------------------------------------------------
def model_based_autofocus(
    hologram_amplitude: np.ndarray,
    geom: Geometry,
    fr_bounds: tuple[float, float],
    preset: str | RecoPreset = DEFAULT_FOCUS_PRESET,
    z01_tol_mm: float = 0.1,
    iterations_scale: float = 1.0,
    threads: int | None = 2,
    a0: float = 1.0,
    fading_width_px: int | None = None,
) -> AutofocusResult:
    """HoloWizard model-based autofocus of one amplitude hologram inside the Fresnel-number interval ``fr_bounds``.

    Args:
        hologram_amplitude: Stored HoloForge hologram ``|psi|`` (2-D); it is squared before it is handed to
            the core API, which expects intensities.  Pass ``sqrt(I)`` for measured intensities ``I``.
        geom: Energy, effective pixel size and z02 (:class:`Geometry`).
        fr_bounds: ``(fr_lo, fr_hi)``; converted to z01 and passed as ``Measurement(z01=centre,
            z01_confidence=half width)``, i.e. the start value is the interval centre, not a guess.
        preset: Name in :data:`PRESETS` (purpose ``focus``) or a :class:`RecoPreset`.
        z01_tol_mm: Nelder-Mead termination tolerance on z01 (P05 default 0.1 mm).
        iterations_scale: Multiplier for all iteration counts of the preset.
        threads: ``torch.set_num_threads`` (``None`` = leave unchanged).
        a0: Flat-field level of the measurement (HoloForge holograms are normalised to 1).
        fading_width_px: Blackman fading width of the FOV window (default :func:`default_fading_width_px`).

    Returns:
        :class:`AutofocusResult` with ``n_evals = len(loss_history)`` (unique reconstructions; Nelder-Mead
        re-evaluations of an already visited z01 are served from HoloWizard's cache).
    """
    configure_holowizard(threads=threads)
    core = _import_core()
    preset_obj = get_preset(preset, iterations_scale)
    fr_lo, fr_hi = _check_bounds(fr_bounds)
    _, _, z_centre, z_half = geom.z01_interval((fr_lo, fr_hi))
    intensity = np.ascontiguousarray(hologram_amplitude, dtype=np.float32) ** 2
    params = build_reco_params(
        intensity, geom, z_centre, preset_obj, z01_confidence_mm=z_half, z01_tol_mm=z01_tol_mm, a0=a0, fading_width_px=fading_width_px
    )
    start = time.perf_counter()
    z01_found, z01_hist, loss_hist = core.find_focus(params, viewer=None, plotter=None)
    runtime = time.perf_counter() - start
    z01_history = [float(z) for z in z01_hist]
    loss_history = [float(np.asarray(value).reshape(-1)[0]) for value in loss_hist]
    z01_est = float(z01_found)
    return AutofocusResult(
        fr_est=geom.fresnel(z01_est),
        z01_est_mm=z01_est,
        n_evals=len(loss_history),
        runtime_s=runtime,
        z01_history=z01_history,
        loss_history=loss_history,
        method=METHOD_HOLOWIZARD,
        fr_history=[geom.fresnel(z) for z in z01_history],
        fr_bounds=(fr_lo, fr_hi),
        preset=preset_obj.name,
    )


# --------------------------------------------------------------------------------------------------
# classical sharpness-metric fallback (no HoloWizard model)
# --------------------------------------------------------------------------------------------------
class ClassicalFocus:
    """Learning-free sharpness autofocus: back-propagate the amplitude and minimise a focus criterion.

    This is **not** the HoloWizard method.  The amplitude is mirror-padded (``pad_factor``), transformed
    once, back-propagated with ``D_{-Fr}`` (kernel from :mod:`src.utils.fresnel`) and cropped; the criterion
    is evaluated on the cropped amplitude:

    * ``"tv"``  total variation (``holowizard.core.find_focus.focus_loss_metrics.get_gra`` equivalent) --
      minimal in focus because the amplitude of a phase object becomes flat;
    * ``"var"`` variance of the amplitude;
    * ``"lap"`` Laplacian energy.

    Works for weak objects with many Fresnel fringes (p05bin8 regime: median error < 1 %, but 25 % gross
    failures at secondary minima); fails completely in the thesis regime ``Fr ~ 1e-4`` at 256 px.  Reason:
    back-propagating the *amplitude* of a weak phase object does not give a flat field in focus but its
    twin image at ``Fr/2`` with half the contrast, so the criterion only has a minimum at the true focus if
    the object spectrum extends well beyond the first CTF zero ``|u|^2 = Fr`` (fine structures, many fringes).
    Smooth objects or ``Fr`` below ``1/N`` (less than one fringe in the field of view) have no usable minimum.
    """

    def __init__(self, hologram_amplitude: np.ndarray, metric: str = "tv", pad_factor: int = 2) -> None:
        if metric not in CLASSICAL_METRICS:
            raise ValueError(f"metric must be one of {CLASSICAL_METRICS}, got {metric!r}")
        amplitude = torch.as_tensor(np.ascontiguousarray(hologram_amplitude, dtype=np.float32))
        if amplitude.ndim != 2:
            raise ValueError(f"expected a 2-D hologram, got shape {tuple(amplitude.shape)}")
        self.shape = (int(amplitude.shape[0]), int(amplitude.shape[1]))
        self.pad = ((self.shape[0] * pad_factor - self.shape[0]) // 2, (self.shape[1] * pad_factor - self.shape[1]) // 2)
        padded = torch.nn.functional.pad(amplitude[None, None], [self.pad[1], self.pad[1], self.pad[0], self.pad[0]], mode="reflect")[0, 0]
        self.field_fft = torch.fft.fft2(padded.to(torch.complex64))
        self.metric = metric
        self.n_evals = 0
        self.history: list[tuple[float, float]] = []

    def backpropagated_amplitude(self, fr: float) -> torch.Tensor:
        kernel = fresnel_kernel((self.field_fft.shape[0], self.field_fft.shape[1]), -float(fr))
        psi = torch.fft.ifft2(self.field_fft * kernel)
        return torch.abs(psi)[self.pad[0] : self.pad[0] + self.shape[0], self.pad[1] : self.pad[1] + self.shape[1]]

    def criterion(self, fr: float) -> float:
        self.n_evals += 1
        a = self.backpropagated_amplitude(fr)
        if self.metric == "tv":
            gx = a - torch.roll(a, 1, 0)
            gy = a - torch.roll(a, 1, 1)
            value = torch.sqrt(gx * gx + gy * gy + 1e-12).mean().item()
        elif self.metric == "var":
            value = torch.var(a).item()
        else:
            sx = torch.roll(a, 1, 0) + torch.roll(a, -1, 0)
            sy = torch.roll(a, 1, 1) + torch.roll(a, -1, 1)
            value = ((sx + sy - 4 * a) ** 2).mean().item()
        self.history.append((float(fr), float(value)))
        return float(value)

    def run(self, fr_lo: float, fr_hi: float, n_grid: int = 21, xtol_rel: float = 1e-4) -> float:
        """Coarse logarithmic grid search followed by bounded Brent refinement in ``log Fr``."""
        fr_lo, fr_hi = _check_bounds((fr_lo, fr_hi))
        grid = np.geomspace(fr_lo, fr_hi, max(int(n_grid), 3))
        values = np.array([self.criterion(float(fr)) for fr in grid])
        k = int(np.argmin(values))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        if lo >= hi:
            return float(grid[k])
        result = minimize_scalar(
            lambda u: self.criterion(float(np.exp(u))), bounds=(np.log(lo), np.log(hi)), method="bounded", options={"xatol": xtol_rel}
        )
        return float(np.exp(result.x))


def classical_autofocus(
    hologram_amplitude: np.ndarray,
    fr_bounds: tuple[float, float],
    metric: str = "tv",
    n_grid: int = 21,
    geom: Geometry | None = None,
    pad_factor: int = 2,
    threads: int | None = 2,
) -> AutofocusResult:
    """Classical (learning-free, model-free) sharpness-metric autofocus; see :class:`ClassicalFocus`.

    ``geom`` is only needed to report the estimate in z01 (``nan`` otherwise).  The method label is
    ``classical_<metric>``.
    """
    if threads is not None:
        torch.set_num_threads(int(threads))
    fr_lo, fr_hi = _check_bounds(fr_bounds)
    start = time.perf_counter()
    focus = ClassicalFocus(hologram_amplitude, metric=metric, pad_factor=pad_factor)
    fr_est = focus.run(fr_lo, fr_hi, n_grid=n_grid)
    runtime = time.perf_counter() - start
    fr_history = [h[0] for h in focus.history]
    return AutofocusResult(
        fr_est=fr_est,
        z01_est_mm=geom.z01_from_fresnel(fr_est) if geom is not None else float("nan"),
        n_evals=focus.n_evals,
        runtime_s=runtime,
        z01_history=[geom.z01_from_fresnel(fr) for fr in fr_history] if geom is not None else [float("nan")] * len(fr_history),
        loss_history=[h[1] for h in focus.history],
        method=f"classical_{metric}",
        fr_history=fr_history,
        fr_bounds=(fr_lo, fr_hi),
        preset=None,
    )


# --------------------------------------------------------------------------------------------------
# evaluation on HoloForge files (CLI)
# --------------------------------------------------------------------------------------------------
def _plot_objectives(histories: dict[str, dict[str, list[float]]], samples: list[ForgeSample], path: Path, max_samples: int = 4) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for i, sample in enumerate(samples[:max_samples]):
        key = f"{METHOD_HOLOWIZARD}_{i}"
        if key in histories:
            h = histories[key]
            order = np.argsort(h["z01"])
            axes[0].plot(np.array(h["z01"])[order] / sample.z01_mm, np.array(h["loss"])[order], "o-", ms=3, label=f"S{i} Fr={sample.fr:.1e}")
        classical = [k for k in histories if k.startswith("classical_") and k.endswith(f"_{i}")]
        for key in classical:
            h = histories[key]
            order = np.argsort(h["fr"])
            axes[1].plot(np.array(h["fr"])[order] / sample.fr, np.array(h["loss"])[order], "o-", ms=3, label=f"S{i} Fr={sample.fr:.1e}")
    axes[0].set_title("HoloWizard find_focus: data residual vs. z01 / z01_true")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("z01 / z01_true")
    axes[1].set_title("classical fallback: criterion vs. Fr / Fr_true")
    axes[1].set_xlabel("Fr / Fr_true")
    for ax in axes:
        ax.axvline(1.0, color="gray", lw=0.8)
        ax.grid(True, alpha=0.3)
        if ax.get_lines():
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_baseline(
    data: list[str | Path],
    out_dir: str | Path,
    n_per_file: int | None = None,
    search_width_pct: float = 50.0,
    preset: str = DEFAULT_FOCUS_PRESET,
    method: str = "both",
    iterations_scale: float = 1.0,
    z01_tol_mm: float = 0.1,
    classical_metric: str = "tv",
    classical_grid: int = 21,
    threads: int = 2,
    log_dir: str | Path | None = None,
    hologram_key: str = "images/hologram",
) -> dict[str, Any]:
    """Run the model-based and/or classical autofocus over HoloForge files and write the report files.

    The search interval of every hologram is ``Fr_true (1 -/+ search_width_pct/100)`` (default +-50 %, much
    wider than the P05 default of +-5 mm ~ +-2 %).  Returns the summary dictionary (also written to
    ``summary.json``).
    """
    if method not in ("holowizard", "classical", "both"):
        raise ValueError("method must be holowizard, classical or both")
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = expand_paths([str(resolve_path(p)) for p in data])
    samples = read_forge_samples(paths, n_per_file=n_per_file, hologram_key=hologram_key)
    if not samples:
        raise ValueError("no samples found")
    preset_obj = get_preset(preset, iterations_scale)
    if method in ("holowizard", "both"):
        log_path = configure_holowizard(working_dir=log_dir, threads=threads)
    else:
        log_path = None
        torch.set_num_threads(int(threads))
    n_px = min(samples[0].shape)
    print(
        f"{len(samples)} holograms from {len(paths)} file(s), {samples[0].shape[0]}x{samples[0].shape[1]} px; search Fr_true*(1 -/+ {search_width_pct:g} %); "
        f"method {method}; preset {preset_obj.name} iterations {preset_obj.iterations} on grids {preset_obj.grid_sizes(n_px)}; "
        f"z01_tol {z01_tol_mm} mm; {torch.get_num_threads()} torch threads",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, list[float]]] = {}
    start_all = time.perf_counter()
    for i, sample in enumerate(samples):
        geom = Geometry.from_sample(sample)
        bounds = fr_search_bounds(sample.fr, search_width_pct)
        results: list[AutofocusResult] = []
        if method in ("holowizard", "both"):
            results.append(
                model_based_autofocus(sample.hologram_amplitude, geom, bounds, preset=preset_obj, z01_tol_mm=z01_tol_mm, threads=None)
            )
        if method in ("classical", "both"):
            results.append(classical_autofocus(sample.hologram_amplitude, bounds, metric=classical_metric, n_grid=classical_grid, geom=geom, threads=None))
        for res in results:
            row = make_result_row(
                index=sample.index,
                source=sample.source,
                fr_true=sample.fr,
                z01_true_mm=sample.z01_mm,
                fr_est=res.fr_est,
                z01_est_mm=res.z01_est_mm,
                runtime_s=res.runtime_s,
                n_evals=res.n_evals,
                method=res.method,
                fr_lo=bounds[0],
                fr_hi=bounds[1],
                preset=res.preset or "",
            )
            rows.append(row)
            histories[f"{res.method}_{i}"] = {"z01": res.z01_history, "fr": res.fr_history, "loss": res.loss_history}
            print(
                f"[{i}] {res.method:22s} Fr_true={sample.fr:.4e} z01_true={sample.z01_mm:8.3f} | Fr_est={res.fr_est:.4e} "
                f"z01_est={res.z01_est_mm:8.3f} rel.err={row['rel_err_fr_pct']:+7.3f} % dz01={row['dz01_mm']:+8.3f} mm "
                f"b={row['blur_px']:.2f} px evals={res.n_evals} t={res.runtime_s:.2f} s",
                flush=True,
            )
        # A full run takes minutes per hologram; keep the partial table on disk so an interrupted run is not lost.
        write_samples_csv(rows, out / "samples.csv")
        (out / "histories.json").write_text(json.dumps(histories), encoding="utf-8")

    per_method = summarize_by_method(rows)
    summary: dict[str, Any] = {
        "data": [_portable(p) for p in paths],
        "hologram_key": hologram_key,
        "n_samples": len(samples),
        "hologram_shape": list(samples[0].shape),
        "search_width_pct": float(search_width_pct),
        "method": method,
        "preset": preset_obj.to_dict(),
        "iterations_scale": float(iterations_scale),
        "z01_tol_mm": float(z01_tol_mm),
        "classical": {"metric": classical_metric, "n_grid": int(classical_grid)},
        "torch_num_threads": torch.get_num_threads(),
        "holowizard_log_dir": str(log_path) if log_path else None,
        "total_runtime_s": time.perf_counter() - start_all,
        "per_method": per_method,
    }
    write_samples_csv(rows, out / "samples.csv")
    (out / "histories.json").write_text(json.dumps(histories), encoding="utf-8")
    figures = {
        "scatter": plot_scatter(rows, out / "scatter.png", title=f"autofocus baseline, search Fr_true*(1 -/+ {search_width_pct:g} %), {len(samples)} holograms").name,
        "objective_curves": _plot_objectives(histories, samples, out / "objective_curves.png").name,
    }
    summary["figures"] = figures
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary_table.md").write_text(summary_markdown_table(per_method), encoding="utf-8")
    for s in per_method:
        print(
            f"{s['method']:22s} n={s['n']} MAE {s['mae_rel_fr_pct']:.2f} % median {s['median_rel_fr_pct']:.2f} % p95 {s['p95_rel_fr_pct']:.2f} % | "
            f"z01 MAE {s['mae_z01_mm']:.2f} mm | blur median {s['blur_px_median']:.2f} px | {s['median_runtime_s']:.3g} s/hologram, "
            f"{s['mean_n_evals']:.1f} evaluations",
            flush=True,
        )
    print(f"report written to {_portable(out)}")
    return summary


def _portable(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Model-based (HoloWizard find_focus) and classical autofocus baselines on HoloForge HDF5 files.")
    parser.add_argument("--data", nargs="+", required=True, help="HDF5 file(s) or glob patterns")
    parser.add_argument("--n", type=int, default=None, help="holograms per file (default: all)")
    parser.add_argument("--search-width", type=float, default=50.0, help="half width of the search interval in %% of Fr_true")
    parser.add_argument("--preset", default=DEFAULT_FOCUS_PRESET, choices=sorted(PRESETS), help="reconstruction stages of the autofocus objective")
    parser.add_argument("--method", default="both", choices=("holowizard", "classical", "both"))
    parser.add_argument("--iterations-scale", type=float, default=1.0, help="multiplier for the iteration counts of the preset")
    parser.add_argument("--z01-tol", type=float, default=0.1, help="Nelder-Mead tolerance on z01 in mm (P05: 0.1)")
    parser.add_argument("--classical-metric", default="tv", choices=CLASSICAL_METRICS)
    parser.add_argument("--classical-grid", type=int, default=21)
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    parser.add_argument("--log-dir", default=None, help="HoloWizard log directory (default: $HOLOWIZARD_LOG_DIR or a temp dir)")
    parser.add_argument("--hologram-key", default="images/hologram")
    parser.add_argument("--out", default=None, help="output directory (default: reports/baseline_model_based/<first data stem>)")
    args = parser.parse_args(argv)
    out = args.out or str(Path("reports/baseline_model_based") / Path(args.data[0]).stem.replace("*", "all"))
    return run_baseline(
        args.data,
        out,
        n_per_file=args.n,
        search_width_pct=args.search_width,
        preset=args.preset,
        method=args.method,
        iterations_scale=args.iterations_scale,
        z01_tol_mm=args.z01_tol,
        classical_metric=args.classical_metric,
        classical_grid=args.classical_grid,
        threads=args.threads,
        log_dir=args.log_dir,
        hologram_key=args.hologram_key,
    )


if __name__ == "__main__":
    main()
