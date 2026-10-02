"""Downstream reconstruction test: how much does a Fresnel-number error degrade the phase reconstruction?

Every hologram is reconstructed with HoloWizard's multi-stage ASRM/PGD (``holowizard.core``) for several
Fresnel-number candidates -- either a relative error curve ``Fr_true (1 + e)`` or the estimates of autofocus
methods read from their ``samples.csv`` -- and compared with the ground-truth phantom (``images/phantoms``,
real part = phase shift).  The wrong Fresnel number enters the core API through ``z01 = z01(Fr)``.

Usage::

    python -m src.eval.downstream --data data/processed/small/test.hdf5 --n 2 \
        --errors -20 -10 -5 -1 0 1 5 10 20 [--preset quality_256] [--iterations-scale 1.0] \
        --out reports/downstream/<name>
    python -m src.eval.downstream --data data/processed/small/test.hdf5 --n 8 \
        --candidates-csv reports/baseline_model_based/<name>/samples.csv [more samples.csv ...] \
        --out reports/downstream/<name>_methods

Outputs: ``downstream_results.csv`` (one row per hologram and candidate), ``downstream_summary.json``,
``downstream_table.md``, ``downstream_error_vs_fr_error.png`` and ``downstream_image_grid.png``.

Conventions (``reports/baseline_model_based.md``): the stored hologram is an amplitude and is squared for the
core API; the core rotates the measurement by 90 degrees during preprocessing, so the reconstruction is
rotated back with ``torch.rot90(x, k=-1)`` before it is compared with the phantom; the phase of a single-
distance hologram is only determined up to a constant and low-frequency background, hence the offset-free
and gradient NRMSE next to the plain NRMSE.  The defocus blur ``b = sqrt(|e| / Fr)`` px is reported for
every candidate (:func:`src.utils.fresnel.defocus_blur_px`).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.baseline.model_based_autofocus import (
    DEFAULT_QUALITY_PRESET,
    Geometry,
    RecoPreset,
    _import_core,
    build_reco_params,
    configure_holowizard,
    get_preset,
)
from src.baseline.results import read_samples_csv
from src.data.forge_samples import ForgeSample, expand_paths, read_forge_samples
from src.utils.config import PROJECT_ROOT, resolve_path
from src.utils.fresnel import blur_px_from_fresnel_numbers
from src.utils.fresnel import hologram_amplitude as forward_hologram_amplitude

__all__ = [
    "RecoResult",
    "DEFAULT_ERRORS_PCT",
    "METRIC_KEYS",
    "reconstruct",
    "reconstruction_metrics",
    "data_residual",
    "downstream_eval",
    "downstream_curve",
    "candidates_from_csv",
    "summarize_downstream",
    "downstream_markdown_table",
    "run_downstream",
    "main",
]

DEFAULT_ERRORS_PCT: tuple[float, ...] = (-20.0, -10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 20.0)
TRUE_CANDIDATE = "true"
METRIC_KEYS: tuple[str, ...] = ("nrmse", "nrmse_offset_free", "nrmse_grad", "pearson", "ssim")
"""Phase-quality metrics of :func:`reconstruction_metrics` (``ssim`` is ``None`` without scikit-image)."""
_SUMMARY_METRICS: tuple[str, ...] = METRIC_KEYS + ("core_loss_final", "data_residual_true_fr", "runtime_s", "blur_px")
_RELATIVE_METRICS: tuple[str, ...] = ("nrmse", "nrmse_offset_free", "nrmse_grad", "core_loss_final", "data_residual_true_fr")
_SSIM_WARNED = False


# --------------------------------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------------------------------
@dataclass
class RecoResult:
    """One HoloWizard reconstruction (field of view, rotated back into the hologram orientation)."""

    phase: np.ndarray
    """Reconstructed phase shift in rad (``<= 0`` for HoloForge phantoms), float32 ``(H, W)``."""
    absorption: np.ndarray
    """Reconstructed absorption (imaginary part of the object), float32 ``(H, W)``."""
    losses: np.ndarray
    """Data residual (mean squared amplitude error inside the FOV) per iteration, all stages concatenated."""
    runtime_s: float
    z01_mm: float
    """Distance handed to the core (``geom.z01_from_fresnel(fr)``)."""
    fr: float
    """Fresnel number used for the reconstruction."""
    preset: str

    @property
    def obj(self) -> np.ndarray:
        """Complex object ``phase + i absorption`` (complex64)."""
        return (self.phase + 1j * self.absorption).astype(np.complex64)

    @property
    def loss_final(self) -> float:
        return float(self.losses[-1]) if self.losses.size else float("nan")


def reconstruct(
    hologram_amplitude: np.ndarray,
    geom: Geometry,
    fr: float,
    preset: str | RecoPreset = DEFAULT_QUALITY_PRESET,
    iterations_scale: float = 1.0,
    threads: int | None = 2,
    a0: float = 1.0,
    fading_width_px: int | None = None,
) -> RecoResult:
    """Reconstruct one amplitude hologram with HoloWizard's multi-stage ASRM/PGD at the Fresnel number ``fr``.

    ``fr`` is converted to ``z01`` (:meth:`Geometry.z01_from_fresnel`) because the core derives the Fresnel
    number from the geometry; the stored amplitude is squared (the API applies ``sqrt``); the result is
    rotated back (``torch.rot90(k=-1)``) into the orientation of the hologram and the phantom.
    """
    configure_holowizard(threads=threads)
    core = _import_core()
    preset_obj = get_preset(preset, iterations_scale)
    if not fr > 0:
        raise ValueError(f"fr must be positive, got {fr}")
    z01 = geom.z01_from_fresnel(float(fr))
    intensity = np.ascontiguousarray(hologram_amplitude, dtype=np.float32) ** 2
    params = build_reco_params(intensity, geom, z01, preset_obj, a0=a0, fading_width_px=fading_width_px)
    start = time.perf_counter()
    result, losses = core.reconstruct(params, viewer=[])
    runtime = time.perf_counter() - start
    result = torch.rot90(result, k=-1)
    return RecoResult(
        phase=result.real.detach().cpu().numpy().astype(np.float32),
        absorption=result.imag.detach().cpu().numpy().astype(np.float32),
        losses=torch.as_tensor(losses).detach().cpu().numpy().astype(np.float64).reshape(-1),
        runtime_s=runtime,
        z01_mm=float(z01),
        fr=float(fr),
        preset=preset_obj.name,
    )


# --------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------
def _crop_border(x: np.ndarray, border: int) -> np.ndarray:
    border = int(border)
    if border <= 0 or 2 * border >= min(x.shape[-2:]):
        return x
    return x[..., border:-border, border:-border]


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _ssim(a: np.ndarray, b: np.ndarray) -> float | None:
    global _SSIM_WARNED
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        if not _SSIM_WARNED:
            warnings.warn("scikit-image is not installed; SSIM is skipped (None)", RuntimeWarning, stacklevel=3)
            _SSIM_WARNED = True
        return None
    data_range = float(b.max() - b.min())
    if not data_range > 0:
        return None
    return float(structural_similarity(a, b, data_range=data_range))


def reconstruction_metrics(phase_hat: np.ndarray, phase_gt: np.ndarray, border: int = 16) -> dict[str, float | None]:
    """Compare a reconstructed phase with the ground truth after cropping ``border`` pixels on every side.

    * ``nrmse``: ``||hat - gt|| / ||gt||``;
    * ``nrmse_offset_free``: NRMSE after removing the mean of both images (the phase of a single-distance
      hologram is only determined up to a constant);
    * ``nrmse_grad``: NRMSE of the image gradients (sensitive to defocus blur and edge ringing, insensitive to
      low-frequency background errors);
    * ``pearson``: Pearson correlation coefficient;
    * ``ssim``: structural similarity (scikit-image, ``data_range`` = range of the ground truth); ``None`` if
      scikit-image is not installed (a ``RuntimeWarning`` is emitted once) or the ground truth is constant.
    """
    r = _crop_border(np.asarray(phase_hat), border).astype(np.float64)
    g = _crop_border(np.asarray(phase_gt), border).astype(np.float64)
    if r.shape != g.shape:
        raise ValueError(f"shape mismatch {r.shape} vs {g.shape}")
    grad_r = np.stack(np.gradient(r))
    grad_g = np.stack(np.gradient(g))
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = float(np.corrcoef(r.ravel(), g.ravel())[0, 1]) if r.std() > 0 and g.std() > 0 else float("nan")
    return {
        "nrmse": _nrmse(r, g),
        "nrmse_offset_free": _nrmse(r - r.mean(), g - g.mean()),
        "nrmse_grad": _nrmse(grad_r, grad_g),
        "pearson": pearson,
        "ssim": _ssim(r, g),
    }


def data_residual(obj_hat: np.ndarray, hologram_amplitude: np.ndarray, fr_true: float, padding_factor: float = 2.0) -> float:
    """RMSE between the forward-modelled amplitude ``|D_Fr_true(exp(i obj_hat))|`` and the measured amplitude.

    Uses the HoloForge forward model (:func:`src.utils.fresnel.hologram_amplitude`: zero-padded object,
    unit probe, transfer-function propagator); for the true object the residual equals the noise level of
    the hologram (0.05 for the data sets of this repository).  Independent of HoloWizard.
    """
    obj = torch.as_tensor(np.asarray(obj_hat, dtype=np.complex64))
    model = forward_hologram_amplitude(obj, float(fr_true), pad_factor=float(padding_factor)).cpu().numpy()
    measured = np.asarray(hologram_amplitude, dtype=np.float32)
    if model.shape != measured.shape:
        raise ValueError(f"shape mismatch {model.shape} vs {measured.shape}")
    return float(np.sqrt(np.mean((model.astype(np.float64) - measured.astype(np.float64)) ** 2)))


def is_diverged(core_loss_final: float, nrmse: float) -> bool:
    """Divergence of the ASRM iteration: NaN or a data residual ~1 instead of ~1e-3 (or an absurd NRMSE)."""
    return bool(not np.isfinite(core_loss_final) or core_loss_final > 0.1 or not np.isfinite(nrmse) or nrmse > 20.0)


# --------------------------------------------------------------------------------------------------
# per-sample evaluation
# --------------------------------------------------------------------------------------------------
def downstream_eval(
    sample: ForgeSample,
    fr_candidates: dict[str, float],
    preset: str | RecoPreset = DEFAULT_QUALITY_PRESET,
    iterations_scale: float = 1.0,
    border: int = 16,
    threads: int | None = 2,
    include_true: bool = True,
    images: dict[str, np.ndarray] | None = None,
    fr_err_pct_nominal: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct ``sample`` for every Fresnel-number candidate and compute all quality metrics.

    Args:
        sample: HoloForge sample with phantom (``images/phantoms`` must be stored).
        fr_candidates: ``{label: Fr}``; the label ``"true"`` denotes the reference.  With ``include_true``
            the true Fresnel number is added under that label if it is missing.
        preset, iterations_scale: Reconstruction stages (:data:`src.baseline.model_based_autofocus.PRESETS`).
        border: Pixels ignored at every edge for the phase metrics.
        images: If given, the reconstructed phases are stored here (``label -> phase``).
        fr_err_pct_nominal: Optional nominal relative errors per label (error-curve mode), stored in the
            ``fr_err_pct_nominal`` column; otherwise the actual error is used.

    Returns:
        One row per candidate with the metrics of :func:`reconstruction_metrics`, ``blur_px``,
        ``core_loss_final`` (HoloWizard's residual at the candidate Fr), ``data_residual_true_fr`` (forward
        model of the reconstruction evaluated at ``Fr_true``), ``runtime_s``, ``diverged`` and the values
        relative to the ``"true"`` candidate (``<metric>_rel_true``).
    """
    if sample.phantom is None:
        raise ValueError(f"sample {sample.source}#{sample.index} has no phantom (images/phantoms); cannot evaluate")
    candidates = dict(fr_candidates)
    if include_true and TRUE_CANDIDATE not in candidates:
        candidates = {TRUE_CANDIDATE: float(sample.fr), **candidates}
    geom = Geometry.from_sample(sample)
    gt_phase = sample.gt_phase
    rows: list[dict[str, Any]] = []
    for label, fr in candidates.items():
        fr = float(fr)
        row: dict[str, Any] = {
            "source": sample.source,
            "index": int(sample.index),
            "fr_true": float(sample.fr),
            "z01_true_mm": float(sample.z01_mm),
            "candidate": str(label),
            "fr_err_pct_nominal": float(fr_err_pct_nominal[label]) if fr_err_pct_nominal and label in fr_err_pct_nominal else (fr / sample.fr - 1.0) * 100.0,
            "fr_used": fr,
            "rel_err_fr_pct": (fr / sample.fr - 1.0) * 100.0 if np.isfinite(fr) else float("nan"),
            "blur_px": float(blur_px_from_fresnel_numbers(fr, sample.fr)) if np.isfinite(fr) else float("nan"),
        }
        if not (np.isfinite(fr) and fr > 0):
            row.update(_failed_row(geom))
            rows.append(row)
            continue
        reco = reconstruct(sample.hologram_amplitude, geom, fr, preset=preset, iterations_scale=iterations_scale, threads=threads)
        metrics = reconstruction_metrics(reco.phase, gt_phase, border=border)
        row.update(
            z01_used_mm=reco.z01_mm,
            dz01_mm=reco.z01_mm - sample.z01_mm,
            **metrics,
            core_loss_final=reco.loss_final,
            data_residual_true_fr=data_residual(reco.obj, sample.hologram_amplitude, sample.fr, padding_factor=sample.padding_factor),
            runtime_s=reco.runtime_s,
            phase_min=float(reco.phase.min()),
            preset=reco.preset,
        )
        row["diverged"] = is_diverged(row["core_loss_final"], row["nrmse"])
        if images is not None:
            images[str(label)] = reco.phase
        rows.append(row)
    _add_relative_to_true(rows)
    return rows


def _failed_row(geom: Geometry) -> dict[str, Any]:
    nan = float("nan")
    return {
        "z01_used_mm": nan,
        "dz01_mm": nan,
        **{key: nan for key in METRIC_KEYS},
        "core_loss_final": nan,
        "data_residual_true_fr": nan,
        "runtime_s": 0.0,
        "phase_min": nan,
        "preset": "",
        "diverged": True,
    }


def _add_relative_to_true(rows: list[dict[str, Any]]) -> None:
    ref = next((row for row in rows if row["candidate"] == TRUE_CANDIDATE), None)
    for row in rows:
        for key in _RELATIVE_METRICS:
            value = row.get(key)
            ref_value = ref.get(key) if ref is not None else None
            ok = value is not None and ref_value not in (None, 0) and np.isfinite(value) and np.isfinite(ref_value)
            row[f"{key}_rel_true"] = float(value / ref_value) if ok else float("nan")


def downstream_curve(
    sample: ForgeSample,
    errors_pct: Sequence[float] = DEFAULT_ERRORS_PCT,
    preset: str | RecoPreset = DEFAULT_QUALITY_PRESET,
    iterations_scale: float = 1.0,
    border: int = 16,
    threads: int | None = 2,
    images: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct with ``Fr_true (1 + e/100)`` for every relative error ``e`` (``0`` is the ``"true"`` candidate)."""
    candidates: dict[str, float] = {}
    nominal: dict[str, float] = {}
    for e in errors_pct:
        label = candidate_label(e)
        candidates[label] = float(sample.fr) * (1.0 + float(e) / 100.0)
        nominal[label] = float(e)
    return downstream_eval(
        sample, candidates, preset=preset, iterations_scale=iterations_scale, border=border, threads=threads,
        include_true=True, images=images, fr_err_pct_nominal=nominal,
    )


def candidate_label(error_pct: float) -> str:
    """Label of an error-curve candidate: ``"true"`` for 0, otherwise e.g. ``"-5%"``/``"+10%"``."""
    return TRUE_CANDIDATE if float(error_pct) == 0.0 else f"{float(error_pct):+g}%"


def candidates_from_csv(paths: Iterable[str | Path], samples: Sequence[ForgeSample]) -> dict[tuple[str, int], dict[str, float]]:
    """Fresnel-number estimates per sample (``(source, index) -> {method: fr_est}``) from ``samples.csv`` files.

    Rows are matched to ``samples`` by file name and index; methods of all files are merged.  Raises if a
    file contributes nothing for the given samples.
    """
    keys = {(s.source, int(s.index)) for s in samples}
    out: dict[tuple[str, int], dict[str, float]] = {key: {} for key in keys}
    for path in paths:
        rows = read_samples_csv(path)
        matched = 0
        for row in rows:
            key = (str(row.get("source")), int(row["index"]))
            if key in out and row.get("fr_est") is not None:
                out[key][str(row["method"])] = float(row["fr_est"])
                matched += 1
        if matched == 0:
            raise ValueError(f"{path}: no rows match the selected samples (sources {sorted({k[0] for k in keys})})")
    return out


# --------------------------------------------------------------------------------------------------
# aggregation and report files
# --------------------------------------------------------------------------------------------------
def summarize_downstream(rows: Sequence[dict[str, Any]], candidates: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Per candidate: mean/std/median of every metric over the non-diverged holograms, number of divergences,
    and the paired medians of ``<metric>_rel_true`` (value / value at the true Fr of the same hologram)."""
    labels = list(candidates) if candidates is not None else list(dict.fromkeys(row["candidate"] for row in rows))
    summary: list[dict[str, Any]] = []
    for label in labels:
        sel = [row for row in rows if row["candidate"] == label]
        ok = [row for row in sel if not row.get("diverged", False)]
        nominal = [row["fr_err_pct_nominal"] for row in sel if np.isfinite(row.get("fr_err_pct_nominal", np.nan))]
        actual = [row["rel_err_fr_pct"] for row in sel if np.isfinite(row.get("rel_err_fr_pct", np.nan))]
        entry: dict[str, Any] = {
            "candidate": label,
            "fr_err_pct_nominal": float(np.median(nominal)) if nominal and len(set(nominal)) == 1 else None,
            "rel_err_fr_pct_median_abs": float(np.median(np.abs(actual))) if actual else None,
            "rel_err_fr_pct_mean": float(np.mean(actual)) if actual else None,
            "n": len(sel),
            "n_diverged": len(sel) - len(ok),
        }
        for key in _SUMMARY_METRICS:
            values = np.array([row[key] for row in ok if row.get(key) is not None], dtype=np.float64)
            values = values[np.isfinite(values)]
            entry[f"{key}_mean"] = float(values.mean()) if values.size else None
            entry[f"{key}_std"] = float(values.std()) if values.size else None
            entry[f"{key}_median"] = float(np.median(values)) if values.size else None
        for key in _RELATIVE_METRICS:
            paired = np.array([row.get(f"{key}_rel_true", np.nan) for row in ok], dtype=np.float64)
            paired = paired[np.isfinite(paired)]
            entry[f"{key}_rel_true_median"] = float(np.median(paired)) if paired.size else None
        summary.append(entry)
    return summary


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return format(value, spec)


def downstream_markdown_table(summary: Sequence[dict[str, Any]]) -> str:
    """Markdown table with one row per candidate (error value or method)."""
    header = (
        "| Candidate | Fr error [%] | b = sqrt(abs(e)/Fr) [px] (mean) | n div. | NRMSE phase (median) | NRMSE offset-free (median) | "
        "NRMSE gradient (median) | Pearson (median) | SSIM (median) | Core loss (median) | Data residual at Fr_true (median) | "
        "NRMSE rel. true (paired median) | NRMSE grad rel. true | Residual rel. true |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for e in summary:
        err = e["fr_err_pct_nominal"] if e["fr_err_pct_nominal"] is not None else e["rel_err_fr_pct_mean"]
        lines.append(
            f"| {e['candidate']} | {_fmt(err, '+.2f')} | {_fmt(e['blur_px_mean'], '.1f')} | {e['n_diverged']}/{e['n']} | "
            f"{_fmt(e['nrmse_median'])} | {_fmt(e['nrmse_offset_free_median'])} | {_fmt(e['nrmse_grad_median'])} | "
            f"{_fmt(e['pearson_median'])} | {_fmt(e['ssim_median'])} | {_fmt(e['core_loss_final_median'], '.3e')} | "
            f"{_fmt(e['data_residual_true_fr_median'], '.4f')} | {_fmt(e['nrmse_rel_true_median'], '.2f')} | "
            f"{_fmt(e['nrmse_grad_rel_true_median'], '.2f')} | {_fmt(e['data_residual_true_fr_rel_true_median'], '.2f')} |"
        )
    return header + "\n".join(lines) + "\n"


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    import csv

    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in fieldnames})
    return path


def _plot_error_curves(rows: Sequence[dict[str, Any]], summary: Sequence[dict[str, Any]], path: Path, title: str, curve_mode: bool) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    sample_keys = list(dict.fromkeys((row["source"], row["index"]) for row in rows))
    for key in sample_keys:
        sel = sorted((row for row in rows if (row["source"], row["index"]) == key and np.isfinite(row["rel_err_fr_pct"])), key=lambda r: r["rel_err_fr_pct"])
        if not sel:
            continue
        label = f"{key[0]}#{key[1]} Fr={sel[0]['fr_true']:.1e}"
        style = "o-" if curve_mode else "o"
        x = [row["rel_err_fr_pct"] for row in sel]
        axes[0].plot(x, [row["nrmse"] for row in sel], style, alpha=0.45, lw=1, label=label)
        axes[1].plot(x, [row["nrmse_grad"] for row in sel], style, alpha=0.45, lw=1)
        axes[2].semilogy(x, [row["core_loss_final"] for row in sel], style, alpha=0.45, lw=1)
        axes[3].plot([row["blur_px"] for row in sel], [row["nrmse_grad_rel_true"] for row in sel], "o", alpha=0.6)
    if curve_mode:
        ordered = sorted((e for e in summary if e["fr_err_pct_nominal"] is not None), key=lambda e: e["fr_err_pct_nominal"])
        errs = [e["fr_err_pct_nominal"] for e in ordered]
        axes[0].plot(errs, [e["nrmse_median"] for e in ordered], "k-", lw=2.5, label="median")
        axes[1].plot(errs, [e["nrmse_grad_median"] for e in ordered], "k-", lw=2.5)
        axes[2].semilogy(errs, [e["core_loss_final_median"] for e in ordered], "k-", lw=2.5)
    else:
        for e in summary:
            sel = [row for row in rows if row["candidate"] == e["candidate"] and np.isfinite(row["rel_err_fr_pct"])]
            if sel:
                axes[0].plot([r["rel_err_fr_pct"] for r in sel], [r["nrmse"] for r in sel], "s", ms=8, mfc="none", label=f"{e['candidate']}")
    axes[0].set_ylim(0, min(3.0, axes[0].get_ylim()[1]))
    axes[1].set_ylim(0, min(4.0, axes[1].get_ylim()[1]))
    axes[0].set_ylabel("NRMSE phase (vs. GT)")
    axes[1].set_ylabel("NRMSE phase gradient (vs. GT)")
    axes[2].set_ylabel("data residual (core loss, MSE amplitudes)")
    axes[3].set_ylabel("NRMSE gradient rel. to Fr_true")
    axes[3].set_xlabel("defocus blur b = sqrt(|e|/Fr) [px]")
    axes[3].axhline(1.0, color="gray", lw=0.8)
    for ax in axes[:3]:
        ax.set_xlabel("relative Fr error [%]")
        ax.axvline(0.0, color="gray", lw=0.8)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _plot_image_grid(sample: ForgeSample, images: dict[str, np.ndarray], rows: Sequence[dict[str, Any]], path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(images)
    n = len(labels) + 2
    ncol = (n + 1) // 2
    fig, axes = plt.subplots(2, ncol, figsize=(2.9 * ncol, 6.2))
    axes = np.atleast_1d(axes).ravel()
    gt_phase = sample.gt_phase
    vmin, vmax = float(gt_phase.min()), max(0.0, float(gt_phase.max()))
    axes[0].imshow(sample.hologram_amplitude, cmap="gray")
    axes[0].set_title("hologram (amplitude)", fontsize=9)
    axes[1].imshow(gt_phase, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"GT phase [{vmin:.1f}, {vmax:.1f}] rad", fontsize=9)
    for j, label in enumerate(labels):
        row = next((r for r in rows if r["candidate"] == label and r["source"] == sample.source and r["index"] == sample.index), None)
        axes[j + 2].imshow(images[label], cmap="gray", vmin=vmin, vmax=vmax)
        sub = f"\nFr {row['rel_err_fr_pct']:+.1f} %, NRMSE {row['nrmse']:.2f}" if row is not None else ""
        axes[j + 2].set_title(f"{label}{sub}", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _portable(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_downstream(
    data: Sequence[str | Path],
    out_dir: str | Path,
    n_per_file: int | None = 2,
    errors_pct: Sequence[float] = DEFAULT_ERRORS_PCT,
    candidates_csv: Sequence[str | Path] | None = None,
    preset: str = DEFAULT_QUALITY_PRESET,
    iterations_scale: float = 1.0,
    border: int = 16,
    threads: int = 2,
    grid_sample: int = 0,
    log_dir: str | Path | None = None,
    hologram_key: str = "images/hologram",
) -> dict[str, Any]:
    """Run the downstream test over HoloForge files and write all report files; returns the summary dict.

    Without ``candidates_csv`` the error curve ``errors_pct`` is evaluated; with it, the Fresnel-number
    estimates of the given ``samples.csv`` files (one or more methods) are used as candidates in addition to
    the true value.
    """
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = expand_paths([str(resolve_path(p)) for p in data])
    samples = read_forge_samples(paths, n_per_file=n_per_file, hologram_key=hologram_key)
    if not samples:
        raise ValueError("no samples found")
    if any(s.phantom is None for s in samples):
        raise ValueError("the downstream test needs images/phantoms in the HDF5 file (store.phantom: true)")
    preset_obj = get_preset(preset, iterations_scale)
    log_path = configure_holowizard(working_dir=log_dir, threads=threads)
    curve_mode = not candidates_csv
    per_sample_candidates = None if curve_mode else candidates_from_csv([resolve_path(p) for p in candidates_csv], samples)
    n_px = min(samples[0].shape)
    # on small test grids a 16-px border would crop away most of the phantom (empty reference -> NRMSE blows up)
    border = min(int(border), n_px // 8)
    print(
        f"{len(samples)} holograms from {len(paths)} file(s), {samples[0].shape[0]}x{samples[0].shape[1]} px; "
        f"{'Fr errors ' + str(list(errors_pct)) + ' %' if curve_mode else 'candidates from ' + ', '.join(str(p) for p in candidates_csv)}; "
        f"preset {preset_obj.name} iterations {preset_obj.iterations} on grids {preset_obj.grid_sizes(n_px)}; metric border {border} px; "
        f"{torch.get_num_threads()} torch threads",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    grid_images: dict[str, np.ndarray] = {}
    start_all = time.perf_counter()
    for si, sample in enumerate(samples):
        images = grid_images if si == grid_sample else None
        if curve_mode:
            sample_rows = downstream_curve(sample, errors_pct, preset=preset_obj, border=border, threads=None, images=images)
        else:
            sample_rows = downstream_eval(sample, per_sample_candidates[(sample.source, int(sample.index))], preset=preset_obj, border=border, threads=None, images=images)
        for row in sample_rows:
            row = {"sample": si, **row}
            rows.append(row)
            print(
                f"[{si}] {row['candidate']:>22s} Fr {row['rel_err_fr_pct']:+7.2f} % b={row['blur_px']:5.1f} px | nrmse {row['nrmse']:.3f} "
                f"off {row['nrmse_offset_free']:.3f} grad {row['nrmse_grad']:.3f} pearson {row['pearson']:.3f} ssim {_fmt(row['ssim'])} | "
                f"core {row['core_loss_final']:.3e} resid(Fr_true) {row['data_residual_true_fr']:.4f} | {row['runtime_s']:.1f} s"
                + ("  DIVERGED" if row["diverged"] else ""),
                flush=True,
            )

    labels = [candidate_label(e) for e in errors_pct] if curve_mode else list(dict.fromkeys(row["candidate"] for row in rows))
    if curve_mode:
        labels = list(dict.fromkeys(labels))
    summary_rows = summarize_downstream(rows, labels)
    _write_csv(rows, out / "downstream_results.csv")
    (out / "downstream_table.md").write_text(downstream_markdown_table(summary_rows), encoding="utf-8")
    title_data = ", ".join(sorted({s.source for s in samples}))
    figures = {
        "error_vs_fr_error": _plot_error_curves(
            rows, summary_rows, out / "downstream_error_vs_fr_error.png",
            f"downstream: reconstruction error vs. Fr error ({len(samples)} holograms, {title_data}, preset {preset_obj.name})", curve_mode,
        ).name
    }
    if grid_images:
        s0 = samples[grid_sample]
        figures["image_grid"] = _plot_image_grid(
            s0, grid_images, rows, out / "downstream_image_grid.png",
            f"HoloWizard reconstruction ({preset_obj.name}, {n_px} px) at wrong Fr; {s0.source} #{s0.index}, Fr={s0.fr:.2e}",
        ).name
    summary: dict[str, Any] = {
        "data": [_portable(p) for p in paths],
        "hologram_key": hologram_key,
        "n_samples": len(samples),
        "hologram_shape": list(samples[0].shape),
        "mode": "error_curve" if curve_mode else "candidates",
        "errors_pct": [float(e) for e in errors_pct] if curve_mode else None,
        "candidates_csv": None if curve_mode else [_portable(resolve_path(p)) for p in candidates_csv],
        "preset": preset_obj.to_dict(),
        "iterations_scale": float(iterations_scale),
        "border_px": int(border),
        "torch_num_threads": torch.get_num_threads(),
        "holowizard_log_dir": str(log_path),
        "total_runtime_s": time.perf_counter() - start_all,
        "figures": figures,
        "summary": summary_rows,
    }
    (out / "downstream_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(downstream_markdown_table(summary_rows))
    print(f"report written to {_portable(out)} ({summary['total_runtime_s']:.0f} s)")
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Downstream reconstruction test: HoloWizard reconstruction quality vs. Fresnel-number error.")
    parser.add_argument("--data", nargs="+", required=True, help="HDF5 file(s) or glob patterns (must contain images/phantoms)")
    parser.add_argument("--n", type=int, default=2, help="holograms per file (default 2; 0 = all)")
    parser.add_argument("--errors", type=float, nargs="+", default=list(DEFAULT_ERRORS_PCT), help="relative Fr errors in %% (error-curve mode)")
    parser.add_argument("--candidates-csv", nargs="+", default=None, help="samples.csv of one or more methods; reconstruct with their fr_est instead of the error curve")
    parser.add_argument("--preset", default=DEFAULT_QUALITY_PRESET, help="reconstruction preset (src.baseline.model_based_autofocus.PRESETS)")
    parser.add_argument("--iterations-scale", type=float, default=1.0)
    parser.add_argument("--border", type=int, default=16, help="pixels ignored at every edge for the phase metrics (clamped to 1/8 of the hologram size)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--grid-sample", type=int, default=0, help="sample index (in the loaded list) for the image grid")
    parser.add_argument("--log-dir", default=None, help="HoloWizard log directory (default: $HOLOWIZARD_LOG_DIR or a temp dir)")
    parser.add_argument("--hologram-key", default="images/hologram")
    parser.add_argument("--out", default=None, help="output directory (default: reports/downstream/<first data stem>)")
    args = parser.parse_args(argv)
    out = args.out or str(Path("reports/downstream") / Path(args.data[0]).stem.replace("*", "all"))
    return run_downstream(
        args.data,
        out,
        n_per_file=None if args.n == 0 else args.n,
        errors_pct=args.errors,
        candidates_csv=args.candidates_csv,
        preset=args.preset,
        iterations_scale=args.iterations_scale,
        border=args.border,
        threads=args.threads,
        grid_sample=args.grid_sample,
        log_dir=args.log_dir,
        hologram_key=args.hologram_key,
    )


if __name__ == "__main__":
    main()
