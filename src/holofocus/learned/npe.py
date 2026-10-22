"""Neural posterior estimation (NPE) of ``dz01`` with sbi (Stage 7, [6]).

``q(t | x)`` approximates the posterior of ``t = dz01 / target_scale`` given the network input
``x`` of Stage 4 (the radial power spectrum or the binned image of one hologram). It is a
conditional density estimator of sbi (``posterior_nn``: neural spline flow or mixture density
network, :class:`~holofocus.learned.config.DensityConfig`) on the output ``e = E(x)`` of an
embedding network: the convolution stack of the Stage 4 CNN with an MLP head
(:class:`Embedding`). Both are trained together on ``-log q(t | x)`` of simulated pairs, and
the epoch with the smallest validation loss is kept ([6], Sec. 3.3 and A1.1: the validation
loss bounds the KL divergence from the true posterior).

The prior is uniform on the ``dz01_range`` of the training data (:func:`prior_box`), the box
from which the simulator draws ``dz01`` (``Omega_z`` of [2]). sbi's estimators have unbounded
support; the posterior summaries use the density on ``grid_points`` values of ``t`` over the
prior box, normalized there (:class:`~holofocus.learned.posterior.GridPosterior`), as sbi's
``DirectPosterior`` truncates to the prior support.

The embedding runs outside the sbi estimator: an sbi estimator repeats its condition for every
value of ``t`` in ``log_prob``, so the grid would run the CNN once per grid point.
:class:`EmbeddedEstimator` joins both into one sbi estimator of ``x`` for ``DirectPosterior``
and the diagnostics of sbi (:func:`sbi_checks`).

:func:`train_npe` writes to the run directory ``config.yaml``, ``history.csv`` (training and
validation loss; MAE of the posterior median and coverage of the central intervals on the
validation split, errors in micrometres), ``model.pt`` (weights of the selected epoch, config,
field of view, prior), ``run.yaml`` (datasets, software, device, timing) and ``summary.json``
(validation metrics of the selected epoch). :func:`predict_npe_split` writes the per-sample
``predictions.csv`` (:data:`NPE_PREDICTION_FIELDS`), ``summary.json`` and ``run.yaml``; several
runs form an ensemble whose posterior is the mean of their densities ([6], Sec. 4.3).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from torch import nn

from .. import __version__
from ..baseline.runner import dataset_identity, file_sha256
from ..dataset.storage import _git_commit, read_config, read_labels
from ..device import get_device
from .config import EmbeddingConfig, NPEConfig
from .data import choose_indices, fov_shape, load_inputs, load_many
from .models import ImageCNN, SpectrumCNN, count_parameters
from .posterior import (
    GridPosterior,
    calibration_summary,
    log_density_um,
    posterior_metrics,
    posterior_summaries,
    summarize_npe_predictions,
    truth_statistics,
    write_npe_predictions,
)
from .representations import Representation
from .training import warmup_cosine

NPE_FORMAT_VERSION = 1
HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_mae_um",
    "val_bias_um",
    "val_coverage_68",
    "val_coverage_95",
    "val_width_68_um",
    "learning_rate",
    "seconds",
)


def prior_box(paths: Sequence[str | Path], target_scale: float) -> tuple[float, float]:
    """Uniform prior of ``t``: the ``dz01_range`` of the split files (which must agree)."""
    ranges = {tuple(read_config(path).sampling.dz01_range) for path in paths}
    if len(ranges) != 1:
        raise ValueError(f"the splits draw dz01 from different ranges {sorted(ranges)}")
    lo, hi = ranges.pop()
    return lo / target_scale, hi / target_scale


class Embedding(nn.Module):
    """Convolution stack of the Stage 4 CNN for the representation ``kind`` (global average
    pooling for images, adaptive pooling for spectra) and an MLP to ``features`` outputs."""

    def __init__(self, config: EmbeddingConfig, kind: str) -> None:
        super().__init__()
        model = config.model_config()
        self.image = kind == "image"
        self.features = (ImageCNN(model) if self.image else SpectrumCNN(model)).features
        n_in = model.channels[-1] * (1 if self.image else model.pool_to)
        self.head = nn.Sequential(
            nn.Linear(n_in, config.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.head(h.mean(dim=(-2, -1)) if self.image else h.flatten(1))


class EmbeddedEstimator(ConditionalDensityEstimator):
    """sbi density estimator ``q(t | E(x))`` of the network input ``x``: the embedding
    network ``E`` followed by a density estimator conditioned on its output."""

    def __init__(
        self,
        embedding: Embedding,
        density: ConditionalDensityEstimator,
        condition_shape: Sequence[int],
    ) -> None:
        super().__init__(density, density.input_shape, torch.Size(condition_shape))
        self.embedding = embedding

    @property
    def embedding_net(self) -> nn.Module:
        return self.embedding

    def embed(self, condition: torch.Tensor) -> torch.Tensor:
        """``(*batch, *condition_shape)`` -> ``(*batch, features)``."""
        batch = condition.shape[: condition.dim() - len(self.condition_shape)]
        e = self.embedding(condition.reshape(-1, *self.condition_shape))
        return e.reshape(*batch, e.shape[-1])

    def log_prob(self, input: torch.Tensor, condition: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net.log_prob(input, self.embed(condition))

    def loss(self, input: torch.Tensor, condition: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net.loss(input, self.embed(condition))

    def sample(self, sample_shape, condition: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net.sample(sample_shape, self.embed(condition))


def build_estimator(
    config: NPEConfig, prior: tuple[float, float], input_shape: Sequence[int]
) -> EmbeddedEstimator:
    """Embedding and density estimator with fresh weights (from torch's global generator).

    ``t`` is z-scored with the moments of the uniform prior (a grid over the box), so the
    estimator depends on the config and the prior only.
    """
    embedding = Embedding(config.embedding, config.representation.kind)
    d = config.density
    build = posterior_nn(
        model=d.estimator,
        z_score_theta="independent",
        z_score_x="none",
        hidden_features=d.hidden_features,
        num_transforms=d.num_transforms,
        num_bins=d.num_bins,
        num_components=d.num_components,
    )
    theta = torch.linspace(prior[0], prior[1], 1001, dtype=torch.float32)[:, None]
    density = build(theta, torch.zeros((2, config.embedding.features)))
    return EmbeddedEstimator(embedding, density, input_shape)


@torch.inference_mode()
def grid_log_density(
    estimator: EmbeddedEstimator,
    inputs: torch.Tensor,
    theta: torch.Tensor,
    batch_size: int = 64,
) -> np.ndarray:
    """``log q(t | x)`` ``(B, G)`` at the grid ``theta`` ``(G,)`` for the inputs ``(B, ...)``;
    the embedding runs once per input."""
    estimator.eval()
    device = theta.device
    out = []
    for start in range(0, len(inputs), batch_size):
        x = inputs[start : start + batch_size].to(device, torch.float32)
        e = estimator.embed(x)
        grid = theta[:, None, None].expand(len(theta), len(x), 1)
        out.append(estimator.net.log_prob(grid, e).T.double().cpu())
    return torch.cat(out).numpy()


class NPEAutofocus:
    """Trained NPE run(s) with the representation: hologram -> posterior of ``dz01``.

    Several runs with the same representation, ``target_scale``, field of view, prior and grid
    form an ensemble: their densities are averaged.
    """

    def __init__(
        self,
        config: NPEConfig,
        estimators: Sequence[EmbeddedEstimator],
        fov_shape: tuple[int, int],
        prior: tuple[float, float],
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = get_device(device or config.device)
        self.estimators = [e.to(self.device).eval() for e in estimators]
        self.fov_shape = (int(fov_shape[0]), int(fov_shape[1]))
        self.prior = (float(prior[0]), float(prior[1]))
        self.representation = Representation(config.representation, self.fov_shape, self.device)
        self.theta = torch.linspace(
            *self.prior, config.grid_points, dtype=torch.float32, device=self.device
        )
        self.theta_grid = np.linspace(*self.prior, config.grid_points)

    @classmethod
    def load(
        cls, paths: str | Path | Sequence[str | Path], device: str | None = None
    ) -> NPEAutofocus:
        """From ``model.pt`` files or run directories (one, or several for an ensemble)."""
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        if not paths:
            raise ValueError("no model to load")
        estimators, first = [], None
        for path in paths:
            checkpoint = torch.load(_model_file(path), map_location="cpu", weights_only=True)
            if checkpoint.get("kind") != "npe":
                raise ValueError(f"{path} is not an NPE model")
            config = NPEConfig.from_dict(checkpoint["config"])
            fov = tuple(int(v) for v in checkpoint["fov_shape"])
            prior = tuple(float(v) for v in checkpoint["prior"])
            key = (
                config.representation,
                config.target_scale,
                config.grid_points,
                fov,
                prior,
            )
            if first is None:
                first = (config, key)
            elif key != first[1]:
                raise ValueError(f"{path} does not share representation, prior and grid")
            input_shape = Representation(config.representation, fov).input_shape
            estimator = build_estimator(config, prior, input_shape)
            estimator.load_state_dict(checkpoint["state_dict"])
            estimators.append(estimator)
        config, (_, _, _, fov, prior) = first
        return cls(config, estimators, fov, prior, device)

    def posterior_of_inputs(self, inputs: torch.Tensor) -> GridPosterior:
        members = [
            GridPosterior.from_log_density(self.theta_grid, grid_log_density(e, inputs, self.theta))
            for e in self.estimators
        ]
        return members[0] if len(members) == 1 else GridPosterior.mixture(members)

    @torch.inference_mode()
    def posterior(self, holograms: torch.Tensor) -> GridPosterior:
        """Posterior of ``t`` for holograms ``(B, H, W)`` or ``(H, W)``."""
        return self.posterior_of_inputs(self.representation(holograms))

    def estimate(self, holograms: torch.Tensor) -> dict[str, np.ndarray]:
        """:func:`posterior_summaries` (metres) for holograms ``(B, H, W)`` or ``(H, W)``."""
        return posterior_summaries(self.posterior(holograms), self.config.target_scale)


def _model_file(path: str | Path) -> Path:
    path = Path(path)
    return path / "model.pt" if path.is_dir() else path


def _evaluate(
    estimator: EmbeddedEstimator,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    theta: torch.Tensor,
    theta_grid: np.ndarray,
    scale: float,
) -> tuple[float, dict]:
    """Validation loss and the posterior statistics of the selected samples."""
    estimator.eval()
    device = theta.device
    losses = []
    with torch.inference_mode():
        for start in range(0, len(inputs), 256):
            x = inputs[start : start + 256].to(device, torch.float32)
            t = targets[start : start + 256, None].to(device)
            losses.append(estimator.loss(t, x).double().cpu())
    post = GridPosterior.from_log_density(theta_grid, grid_log_density(estimator, inputs, theta))
    return float(torch.cat(losses).mean()), truth_statistics(post, targets.double().numpy(), scale)


def _append_history(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row[k] for k in HISTORY_FIELDS})


def train_npe(
    config: NPEConfig,
    train: str | Path | Sequence[str | Path],
    val: str | Path,
    out: str | Path,
    device: str | None = None,
    overwrite: bool = False,
    log: Callable[[str], None] | None = print,
) -> dict:
    """Train on the split file(s) ``train`` and select the epoch on ``val``; returns the
    summary. Several training files are concatenated in the given order."""
    trains = [Path(train)] if isinstance(train, (str, Path)) else [Path(p) for p in train]
    val, out = Path(val), Path(out)
    if (out / "model.pt").exists() and not overwrite:
        raise ValueError(f"{out} already holds a trained model (use overwrite)")
    shape = fov_shape(val)
    if any(fov_shape(path) != shape for path in trains):
        raise ValueError(f"{[str(p) for p in trains]} and {val} have different fields of view")
    prior = prior_box([*trains, val], config.target_scale)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("history.csv", "summary.json"):
        (out / name).unlink(missing_ok=True)
    (out / "config.yaml").write_text(config.to_yaml())

    cfg = config.training
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    torch_device = get_device(device or config.device)
    representation = Representation(config.representation, shape, torch_device)
    start = time.perf_counter()
    train_data = load_many(trains, representation, config.target_scale, limit=cfg.limit)
    val_data = load_inputs(val, representation, config.target_scale)
    load_seconds = time.perf_counter() - start
    estimator = build_estimator(config, prior, representation.input_shape).to(torch_device)
    theta = torch.linspace(*prior, config.grid_points, dtype=torch.float32, device=torch_device)
    theta_grid = np.linspace(*prior, config.grid_points)
    val_dz01 = val_data.labels["dz01"]
    if log is not None:
        log(
            f"{config.name}: {len(train_data)} training / {len(val_data)} validation samples, "
            f"input {representation.input_shape}, {config.density.estimator}, "
            f"{count_parameters(estimator)} parameters, prior t in [{prior[0]:g}, {prior[1]:g}], "
            f"{torch_device}, {torch.get_num_threads()} threads (inputs in {load_seconds:.1f} s)"
        )

    batch = min(cfg.batch_size, len(train_data))
    steps_per_epoch = max(1, len(train_data) // batch)
    optimizer = torch.optim.AdamW(
        estimator.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine(step, cfg.warmup_epochs * steps_per_epoch, total_steps),
    )
    best = {"loss": math.inf, "epoch": 0, "state": None}
    train_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        estimator.train()
        order = rng.permutation(len(train_data))
        losses = []
        for step in range(steps_per_epoch):
            pick = torch.from_numpy(order[step * batch : (step + 1) * batch])
            x = train_data.inputs[pick].to(torch_device, torch.float32)
            t = train_data.targets[pick, None].to(torch_device)
            loss = estimator.loss(t, x).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss in epoch {epoch}, step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(estimator.parameters(), cfg.clip_max_norm)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        val_loss, stats = _evaluate(
            estimator, val_data.inputs, val_data.targets, theta, theta_grid, config.target_scale
        )
        metrics = posterior_metrics(stats, val_dz01)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_mae_um": metrics["error_um"]["mae"],
            "val_bias_um": metrics["error_um"]["bias"],
            "val_coverage_68": metrics["interval"]["coverage"]["68"],
            "val_coverage_95": metrics["interval"]["coverage"]["95"],
            "val_width_68_um": metrics["interval"]["width_um"]["68"]["median"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        _append_history(out / "history.csv", row)
        if val_loss < best["loss"]:
            best = {"loss": val_loss, "epoch": epoch, "state": _copy_state(estimator)}
        if log is not None:
            log(
                f"epoch {epoch:3d}: train {row['train_loss']:+.4f}, val {val_loss:+.4f}, "
                f"MAE {row['val_mae_um']:.0f} um, bias {row['val_bias_um']:+.0f} um, "
                f"cov68/95 {row['val_coverage_68']:.2f}/{row['val_coverage_95']:.2f}, "
                f"width68 {row['val_width_68_um']:.0f} um ({row['seconds']:.1f} s)"
            )
        if cfg.patience is not None and epoch - best["epoch"] >= cfg.patience:
            break
    train_seconds = time.perf_counter() - train_start

    estimator.load_state_dict(best["state"])
    val_loss, stats = _evaluate(
        estimator, val_data.inputs, val_data.targets, theta, theta_grid, config.target_scale
    )
    summary = {
        "name": config.name,
        "best_epoch": best["epoch"],
        "epochs_run": epoch,
        "val_loss": val_loss,
        "prior": list(prior),
        "validation": posterior_metrics(stats, val_dz01),
        "train_seconds": train_seconds,
    }
    torch.save(
        {
            "kind": "npe",
            "format_version": NPE_FORMAT_VERSION,
            "config": config.to_dict(),
            "fov_shape": list(shape),
            "prior": list(prior),
            "best_epoch": best["epoch"],
            "state_dict": best["state"],
        },
        out / "model.pt",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    meta = {
        "config": config.to_dict(),
        "datasets": {
            "train": [dataset_identity(path) for path in trains],
            "val": dataset_identity(val),
        },
        "n_train": len(train_data),
        "n_val": len(val_data),
        "input_shape": list(representation.input_shape),
        "prior": list(prior),
        "n_parameters": count_parameters(estimator),
        "n_parameters_density": count_parameters(estimator.net),
        "best_epoch": best["epoch"],
        "epochs_run": epoch,
        "load_seconds": load_seconds,
        "train_seconds": train_seconds,
        "holofocus_version": __version__,
        "git_commit": _git_commit(),
        "torch_version": str(torch.__version__),
        "sbi_version": _sbi_version(),
        "device": str(torch_device),
        "threads": torch.get_num_threads(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "run.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    if log is not None:
        v = summary["validation"]
        log(
            f"best epoch {best['epoch']}: validation loss {val_loss:+.4f}, MAE "
            f"{v['error_um']['mae']:.0f} um, coverage 68/95 % {v['interval']['coverage']['68']:.2f}"
            f"/{v['interval']['coverage']['95']:.2f} ({train_seconds:.0f} s)"
        )
    return summary


def _copy_state(module: nn.Module) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}


def _sbi_version() -> str:
    import sbi

    return str(sbi.__version__)


def predict_npe_split(
    autofocus: NPEAutofocus,
    data: str | Path,
    out: str | Path,
    indices: Sequence[int] | None = None,
    limit: int | None = None,
    model_paths: Sequence[str | Path] | None = None,
    log: Callable[[str], None] | None = print,
) -> dict:
    """Posterior of every selected sample of the split file ``data``; returns the summary.

    ``seconds`` is one estimate (representation, embedding, density on the grid, median and
    intervals) from the hologram in memory, batch of one, after one warm-up call.
    """
    data, out = Path(data), Path(out)
    dataset_config = read_config(data)
    if dataset_config.geometry.fov_shape != autofocus.fov_shape:
        raise ValueError(
            f"{data} has a field of view {dataset_config.geometry.fov_shape}, the model "
            f"was trained on {autofocus.fov_shape}"
        )
    scale = autofocus.config.target_scale
    labels = read_labels(data)
    chosen = choose_indices(len(labels["dz01"]), indices, limit)
    rows = []
    with h5py.File(data, "r") as fh:
        holograms = fh["hologram"]
        autofocus.estimate(torch.from_numpy(np.asarray(holograms[chosen[0]])))
        for index in chosen:
            hologram = torch.from_numpy(np.asarray(holograms[index]))
            start = time.perf_counter()
            post = autofocus.posterior(hologram)
            est = {k: float(v[0]) for k, v in posterior_summaries(post, scale).items()}
            seconds = time.perf_counter() - start
            dz01_true = float(labels["dz01"][index])
            truth = np.array([dz01_true / scale])
            z01_true, z01_guess = float(labels["z01"][index]), float(labels["z01_guess"][index])
            z01_est = z01_guess + est["dz01"]
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
                    "dz01_true": dz01_true,
                    "dz01_est": est["dz01"],
                    "dz01_mean": est["dz01_mean"],
                    "dz01_mode": est["dz01_mode"],
                    "sigma": est["sigma"],
                    "z01_est": z01_est,
                    "error_um": error_um,
                    "abs_error_um": abs(error_um),
                    "z_score": error_um * 1e-6 / est["sigma"],
                    "z01_lower_68": z01_guess + est["lower_68"],
                    "z01_upper_68": z01_guess + est["upper_68"],
                    "z01_lower_95": z01_guess + est["lower_95"],
                    "z01_upper_95": z01_guess + est["upper_95"],
                    "width_68_um": (est["upper_68"] - est["lower_68"]) * 1e6,
                    "width_95_um": (est["upper_95"] - est["lower_95"]) * 1e6,
                    "hpd_width_68_um": est["hpd_width_68"] * 1e6,
                    "hpd_width_95_um": est["hpd_width_95"] * 1e6,
                    "pit": float(post.pit(truth)[0]),
                    "hpd_level": float(post.hpd_level(truth)[0]),
                    "log_density_um": float(log_density_um(post, truth, scale)[0]),
                    "prior_mass": est["prior_mass"],
                    "fresnel_number_true": fr_true,
                    "fresnel_number_est": fr_est,
                    "fresnel_error_rel": fr_est / fr_true - 1.0,
                    "seconds": seconds,
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    write_npe_predictions(out / "predictions.csv", rows)
    summary = summarize_npe_predictions(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    paths = [] if model_paths is None else [Path(p) for p in model_paths]
    meta = {
        "models": [str(p) for p in paths],
        "model_sha256": [file_sha256(_model_file(p)) for p in paths],
        "ensemble_size": len(autofocus.estimators),
        "config": autofocus.config.to_dict(),
        "prior": list(autofocus.prior),
        "dataset": dataset_identity(data),
        "holofocus_version": __version__,
        "git_commit": _git_commit(),
        "torch_version": str(torch.__version__),
        "sbi_version": _sbi_version(),
        "device": str(autofocus.device),
        "threads": torch.get_num_threads(),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "run.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    if log is not None:
        err, interval = summary["error_um"], summary["interval"]
        log(
            f"{len(rows)} samples of {data}: MAE {err['mae']:.0f} um, within 100 um "
            f"{err['within']['100']:.2f}, coverage 68/95 % {interval['coverage']['68']:.2f}/"
            f"{interval['coverage']['95']:.2f}, median width68 "
            f"{interval['width_um']['68']['median']:.0f} um, "
            f"{1e3 * summary['seconds_per_estimate']['median']:.1f} ms per estimate"
        )
    return summary


def sbi_checks(
    autofocus: NPEAutofocus,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_posterior_samples: int = 1000,
    chunk: int = 250,
    seed: int = 0,
) -> dict:
    """SBC, sample-based expected coverage and TARP of sbi ([6], Appendix A5) for network
    inputs and targets ``t`` of a calibration split: a cross-check of the grid statistics
    with sbi's own sampling (``DirectPosterior``, rejection at the prior box).

    The posterior is sampled once per chunk of ``chunk`` inputs; the samples give the SBC
    rank of ``t`` and the rank of ``log q(t | x)`` (expected coverage with the HPD regions of
    ``q``), and all of them together the TARP coverage (sbi's ``run_sbc`` and ``run_tarp``
    without holding the samples of every input in one sampling call).

    sbi's C2ST of the ranks (``c2st_ranks``) sets the integer ranks against continuous uniform
    samples, and its classifier tells the two apart even for exactly uniform ranks (accuracy
    0.70 with 1000 posterior samples and 2000 inputs). ``c2st_ranks_jittered`` spreads each
    rank ``r`` uniformly to ``(r + u) L / (L + 1)``, ``u ~ U(0, 1)``, which is uniform on
    ``[0, L]`` for a calibrated posterior; on these, an accuracy of 0.5 means uniform ranks.
    """
    from sbi.diagnostics import check_sbc, check_tarp
    from sbi.diagnostics.sbc import _run_sbc, check_uniformity_c2st
    from sbi.diagnostics.tarp import _run_tarp, get_tarp_references
    from sbi.inference import DirectPosterior
    from sbi.utils import BoxUniform
    from sbi.utils.metrics import l2

    if len(autofocus.estimators) != 1:
        raise ValueError("the sbi checks take a single NPE, not an ensemble")
    torch.manual_seed(seed)
    device = autofocus.device
    lo, hi = autofocus.prior
    prior = BoxUniform(
        low=torch.tensor([lo], device=device), high=torch.tensor([hi], device=device)
    )
    posterior = DirectPosterior(posterior_estimator=autofocus.estimators[0], prior=prior)

    def marginal(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return theta[:, 0]

    def log_q(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return posterior.log_prob(theta, x=x, norm_posterior=False)

    thetas = targets.to(device, torch.float32)[:, None]
    xs = inputs.to(device, torch.float32)
    start = time.perf_counter()
    ranks, samples = [], []
    with torch.inference_mode():
        for s in range(0, len(xs), chunk):
            draw = posterior.sample_batched(
                (num_posterior_samples,), x=xs[s : s + chunk], show_progress_bars=False
            )
            ranks.append(
                _run_sbc(
                    thetas[s : s + chunk],
                    xs[s : s + chunk],
                    draw,
                    [marginal, log_q],
                    show_progress_bar=False,
                )
            )
            samples.append(draw.cpu())
        ranks, samples = torch.cat(ranks), torch.cat(samples, dim=1)
        checks = check_sbc(
            ranks[:, :1], prior.sample((len(thetas),)).cpu(), samples[0], num_posterior_samples
        )
        truth = thetas.cpu()
        bins = max(1, len(truth) // 10)
        ecp, alpha = _run_tarp(samples, truth, get_tarp_references(truth), l2, bins, True)
        atc, tarp_p = check_tarp(ecp, alpha)
        spread = ranks[:, :1].double() + torch.rand(len(ranks), 1, dtype=torch.float64)
        spread = spread * num_posterior_samples / (num_posterior_samples + 1)
        c2st_jittered = check_uniformity_c2st(spread.float(), num_posterior_samples)
    rank = ranks[:, 0].double().numpy() / num_posterior_samples
    hpd = 1.0 - ranks[:, 1].double().numpy() / num_posterior_samples
    return {
        "n": len(thetas),
        "num_posterior_samples": num_posterior_samples,
        "sbc": {
            "ks_p_value": float(checks["ks_pvals"][0]),
            "c2st_ranks": float(checks["c2st_ranks"][0]),
            "c2st_ranks_jittered": float(c2st_jittered[0]),
            "c2st_dap": float(np.asarray(checks["c2st_dap"]).ravel()[0]),
        },
        "calibration": calibration_summary(rank, hpd),
        "tarp": {"atc": float(atc), "ks_p_value": float(tarp_p)},
        "seconds": time.perf_counter() - start,
        "sbi_version": _sbi_version(),
        "rank": rank.tolist(),
        "hpd_level": hpd.tolist(),
    }


def check_npe_split(
    autofocus: NPEAutofocus,
    data: str | Path,
    out: str | Path,
    limit: int | None = None,
    num_posterior_samples: int = 1000,
    seed: int = 0,
    model_path: str | Path | None = None,
    log: Callable[[str], None] | None = print,
) -> dict:
    """:func:`sbi_checks` on the samples of the split file ``data``; writes ``sbi-checks.json``
    and the normalized ranks per sample, ``sbi-ranks.csv``, to ``out``."""
    data, out = Path(data), Path(out)
    if read_config(data).geometry.fov_shape != autofocus.fov_shape:
        raise ValueError(f"{data} does not have the field of view {autofocus.fov_shape}")
    split = load_inputs(data, autofocus.representation, autofocus.config.target_scale, limit=limit)
    result = sbi_checks(autofocus, split.inputs, split.targets, num_posterior_samples, seed=seed)
    rank, hpd = result.pop("rank"), result.pop("hpd_level")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sbi-ranks.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "rank", "hpd_level"])
        writer.writerows(zip(split.indices.tolist(), rank, hpd, strict=True))
    result["dataset"] = dataset_identity(data)
    if model_path is not None:
        result["model_sha256"] = file_sha256(_model_file(model_path))
    (out / "sbi-checks.json").write_text(json.dumps(result, indent=2) + "\n")
    if log is not None:
        sbc, hpd_cov = result["sbc"], result["calibration"]["hpd"]["coverage"]
        log(
            f"sbi checks on {len(split)} samples of {data}: SBC KS p {sbc['ks_p_value']:.3f}, "
            f"C2ST ranks {sbc['c2st_ranks_jittered']:.3f} (jittered; sbi {sbc['c2st_ranks']:.3f}), "
            f"expected coverage 68/95 % "
            f"{hpd_cov['68']:.3f}/{hpd_cov['95']:.3f}, TARP ATC {result['tarp']['atc']:+.3f} "
            f"({result['seconds']:.0f} s)"
        )
    return result
