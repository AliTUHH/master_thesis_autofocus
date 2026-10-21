"""Inference of a trained learned autofocus on a dataset split, with per-sample timing.

``predictions.csv`` has one row per sample, keyed by ``index`` like the baseline results:

* ``dz01_est``, ``z01_est = z01_guess + dz01_est`` and ``z01_true`` in metres,
* ``sigma`` (calibrated, ``sigma_scale * sigma_raw``) and ``sigma_raw`` in metres,
* ``error_um = (z01_est - z01_true) * 1e6`` and ``z_score = error / sigma``,
* ``fresnel_error_rel = Fr(z01_est) / Fr(z01_true) - 1``,
* ``seconds``: one estimate from the hologram in memory (representation and network, batch
  of one, after one warm-up call).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from .. import __version__
from ..baseline.runner import dataset_identity, file_sha256
from ..dataset.storage import _git_commit, read_config, read_labels
from ..device import get_device
from ..phantoms import PHANTOM_TYPES
from .config import LearnedConfig
from .data import choose_indices
from .metrics import (
    grouped_point_metrics,
    interval_metrics,
    photon_bin_labels,
    photon_bin_names,
    point_metrics,
)
from .models import build_model
from .representations import Representation

PREDICTION_FIELDS = (
    "index",
    "phantom_type",
    "photons",
    "z01_true",
    "z01_guess",
    "dz01_true",
    "dz01_est",
    "sigma",
    "sigma_raw",
    "z01_est",
    "error_um",
    "abs_error_um",
    "z_score",
    "fresnel_number_true",
    "fresnel_number_est",
    "fresnel_error_rel",
    "seconds",
)
INT_FIELDS = {"index", "phantom_type"}


class LearnedAutofocus:
    """A trained network with its representation: hologram -> ``dz01`` and ``sigma`` (m)."""

    def __init__(
        self,
        config: LearnedConfig,
        model: torch.nn.Module,
        fov_shape: tuple[int, int],
        sigma_scale: float = 1.0,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = get_device(device or config.device)
        self.model = model.to(self.device).eval()
        self.fov_shape = (int(fov_shape[0]), int(fov_shape[1]))
        self.sigma_scale = float(sigma_scale)
        self.representation = Representation(config.representation, self.fov_shape, self.device)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> LearnedAutofocus:
        """From ``model.pt`` or a training run directory containing it."""
        path = Path(path)
        checkpoint = torch.load(
            path / "model.pt" if path.is_dir() else path, map_location="cpu", weights_only=True
        )
        config = LearnedConfig.from_dict(checkpoint["config"])
        model = build_model(config)
        model.load_state_dict(checkpoint["state_dict"])
        return cls(config, model, checkpoint["fov_shape"], checkpoint["sigma_scale"], device)

    @torch.inference_mode()
    def estimate(self, holograms: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(dz01, sigma, sigma_raw)`` in metres for holograms ``(B, H, W)`` or ``(H, W)``."""
        mean, var = self.model(self.representation(holograms))
        scale = self.config.target_scale
        sigma_raw = var.double().sqrt().cpu().numpy() * scale
        return mean.double().cpu().numpy() * scale, self.sigma_scale * sigma_raw, sigma_raw


def predict_split(
    autofocus: LearnedAutofocus,
    data: str | Path,
    out: str | Path,
    indices: Sequence[int] | None = None,
    limit: int | None = None,
    model_path: str | Path | None = None,
    log: Callable[[str], None] | None = print,
) -> dict:
    """Estimate every selected sample of the split file ``data``; returns the summary."""
    data, out = Path(data), Path(out)
    dataset_config = read_config(data)
    if dataset_config.geometry.fov_shape != autofocus.fov_shape:
        raise ValueError(
            f"{data} has a field of view {dataset_config.geometry.fov_shape}, the model "
            f"was trained on {autofocus.fov_shape}"
        )
    labels = read_labels(data)
    chosen = choose_indices(len(labels["dz01"]), indices, limit)
    rows = []
    with h5py.File(data, "r") as fh:
        holograms = fh["hologram"]
        autofocus.estimate(torch.from_numpy(np.asarray(holograms[chosen[0]])))
        for index in chosen:
            hologram = torch.from_numpy(np.asarray(holograms[index]))
            start = time.perf_counter()
            dz01_est, sigma, sigma_raw = (float(v[0]) for v in autofocus.estimate(hologram))
            seconds = time.perf_counter() - start
            z01_true, z01_guess = float(labels["z01"][index]), float(labels["z01_guess"][index])
            z01_est = z01_guess + dz01_est
            error_um = (z01_est - z01_true) * 1e6
            fr_true = float(labels["fresnel_number"][index])
            fr_est = dataset_config.geometry.at(z01_est).fresnel_number
            rows.append(
                {
                    "index": index,
                    "phantom_type": int(labels["phantom_type"][index]),
                    "photons": float(labels["photons"][index]),
                    "z01_true": z01_true,
                    "z01_guess": z01_guess,
                    "dz01_true": float(labels["dz01"][index]),
                    "dz01_est": dz01_est,
                    "sigma": sigma,
                    "sigma_raw": sigma_raw,
                    "z01_est": z01_est,
                    "error_um": error_um,
                    "abs_error_um": abs(error_um),
                    "z_score": error_um * 1e-6 / sigma,
                    "fresnel_number_true": fr_true,
                    "fresnel_number_est": fr_est,
                    "fresnel_error_rel": fr_est / fr_true - 1.0,
                    "seconds": seconds,
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    write_predictions(out / "predictions.csv", rows)
    summary = summarize_predictions(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    meta = {
        "model": None if model_path is None else str(model_path),
        "model_sha256": None if model_path is None else file_sha256(_model_file(model_path)),
        "config": autofocus.config.to_dict(),
        "sigma_scale": autofocus.sigma_scale,
        "dataset": dataset_identity(data),
        "holofocus_version": __version__,
        "git_commit": _git_commit(),
        "torch_version": str(torch.__version__),
        "device": str(autofocus.device),
        "threads": torch.get_num_threads(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "run.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    if log is not None:
        err = summary["error_um"]
        log(
            f"{len(rows)} samples of {data}: MAE {err['mae']:.0f} um, RMSE {err['rmse']:.0f} um, "
            f"bias {err['bias']:+.0f} um, coverage 68/95 % "
            f"{summary['interval']['coverage']['68']:.2f}/"
            f"{summary['interval']['coverage']['95']:.2f}, "
            f"{1e3 * summary['seconds_per_estimate']['median']:.1f} ms per estimate"
        )
    return summary


def _model_file(path: str | Path) -> Path:
    path = Path(path)
    return path / "model.pt" if path.is_dir() else path


def write_predictions(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: repr(float(v)) if isinstance(v, float) else v for k, v in row.items()}
            )


def read_predictions(path: str | Path) -> list[dict]:
    with open(path, newline="") as fh:
        return [
            {k: int(v) if k in INT_FIELDS else float(v) for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]


def summarize_predictions(rows: list[dict]) -> dict:
    """Accuracy, calibration, Fresnel-number error, cost and breakdowns of predictions."""
    err = np.array([r["error_um"] for r in rows])
    sigma = np.array([r["sigma"] for r in rows]) * 1e6
    sigma_raw = np.array([r["sigma_raw"] for r in rows]) * 1e6
    fr = np.array([r["fresnel_error_rel"] for r in rows])
    seconds = np.array([r["seconds"] for r in rows])
    names = [PHANTOM_TYPES[r["phantom_type"]] for r in rows]
    photons = [r["photons"] for r in rows]
    return {
        "n_samples": len(rows),
        "error_um": point_metrics(err),
        "interval": interval_metrics(err, sigma),
        "interval_uncalibrated": interval_metrics(err, sigma_raw),
        "fresnel_error_rel": {
            "bias": float(fr.mean()),
            "mae": float(np.abs(fr).mean()),
            "rmse": float(np.sqrt(np.mean(fr**2))),
            "median_abs": float(np.median(np.abs(fr))),
        },
        "guess_error_um": {"mae": float(np.mean([abs(r["dz01_true"]) for r in rows]) * 1e6)},
        "seconds_per_estimate": {
            "mean": float(seconds.mean()),
            "median": float(np.median(seconds)),
            "max": float(seconds.max()),
        },
        "by_phantom_type": grouped_point_metrics(err, names, order=PHANTOM_TYPES),
        "by_photons": grouped_point_metrics(
            err, photon_bin_labels(photons), order=photon_bin_names()
        ),
    }
