"""Neural posterior estimation (NPE) of the focus-to-object distance ``z01`` from a single hologram.

Design
------
* **Parameter** ``theta = z01`` in mm (1-D).  The prior is ``Uniform(z01_min, z01_max)`` with the bounds
  read from the ``meta.json`` of the data set, i.e. exactly the distribution
  :class:`src.data.forge_setup.NFHRandomDistSetup` used to simulate the training holograms.  Working in
  ``z01`` (not ``log Fr``) keeps the prior identical to the data-generating process (a uniform prior in
  ``log Fr`` would be a *different* prior and the posterior would differ by the Jacobian), and the result
  is directly the quantity the beamline needs (``Measurement(z01=..., z01_confidence=...)`` of HoloWizard).
  ``Fr`` follows deterministically from ``z01`` and the fixed geometry (:mod:`src.utils.physics`).
* **Observation** ``x`` = network input of :class:`src.data.dataset.HologramHDF5Dataset`: either the
  1-D radial log-power profile (compact, flip/rotation invariant, cheap -- main variant) or the
  standardised hologram (CNN embedding).  An **embedding network** maps ``x`` to a low-dimensional
  summary that conditions the density estimator.
* **Density estimator**: neural spline flow (``nsf``, default), masked autoregressive flow (``maf``) or a
  mixture density network (``mdn``) via ``sbi.neural_nets.posterior_nn``.
* **Training**: a single round of NPE on the pre-simulated ``(theta, x)`` pairs of the HDF5 files
  (amortised posterior; no online simulation).  By default the files ``train.hdf5``/``val.hdf5`` are used
  as training/validation set (``data.validation_split: files``), which mirrors the point-estimator runs;
  ``random`` lets ``sbi`` split the pooled data with ``train.validation_fraction``.  Early stopping is
  ``sbi``'s own (``stop_after_epochs`` epochs without improvement of the validation log-probability).

The resulting :class:`sbi.inference.posteriors.direct_posterior.DirectPosterior` is stored with
``torch.save`` together with the config, prior, geometry and training summary (:func:`save_posterior`).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sbi.inference import NPE, EnsemblePosterior
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform
from torch.utils import data as torch_data
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.data.dataset import HologramHDF5Dataset, dataset_kwargs_from_config
from src.models.cnn import AutofocusCNN, RadialProfileMLP
from src.sbi.calibration import central_interval
from src.utils.config import require_keys, resolve_path
from src.utils.torch_utils import count_parameters, seed_everything, select_device

__all__ = [
    "PARAMETERS",
    "EMBEDDINGS",
    "DENSITY_ESTIMATORS",
    "VALIDATION_SPLITS",
    "PriorSpec",
    "read_meta",
    "prior_from_meta",
    "prior_from_config",
    "RadialProfileEmbedding",
    "HologramCNNEmbedding",
    "build_embedding_net",
    "build_density_estimator",
    "load_simulations",
    "FixedSplitNPE",
    "NPETrainingResult",
    "fit_npe",
    "save_posterior",
    "load_posterior",
    "posterior_members",
    "embedding_net_of",
    "sample_posterior",
    "log_prob_grid",
    "parameter_grid",
    "point_estimates",
    "leakage_acceptance",
    "posterior_to_find_focus_init",
    "find_focus_bounds",
]

PARAMETERS: tuple[str, ...] = ("z01_mm",)
"""Supported inference parameters (``theta``)."""
EMBEDDINGS: tuple[str, ...] = ("mlp", "cnn", "identity")
"""Embedding network types: ``mlp`` for the radial profile, ``cnn`` for image inputs, ``identity`` for low-dimensional x."""
DENSITY_ESTIMATORS: tuple[str, ...] = ("nsf", "maf", "mdn")
VALIDATION_SPLITS: tuple[str, ...] = ("files", "random")

_META_FILE = "meta.json"


# --------------------------------------------------------------------------------------------------
# prior
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PriorSpec:
    """Box prior ``Uniform(low, high)`` over one parameter (``z01_mm``)."""

    low: float
    high: float
    parameter: str = "z01_mm"
    source: str = "config"
    """Where the bounds came from: ``meta.json`` (data set) or ``config`` (explicit)."""

    def __post_init__(self) -> None:
        if self.parameter not in PARAMETERS:
            raise ValueError(f"parameter must be one of {PARAMETERS}, got {self.parameter!r}")
        if not np.isfinite(self.low) or not np.isfinite(self.high) or not self.low < self.high:
            raise ValueError(f"require finite low < high, got low={self.low}, high={self.high}")

    def build(self, device: str | torch.device = "cpu") -> BoxUniform:
        """``sbi.utils.BoxUniform`` over ``[low, high]`` (event shape ``[1]``)."""
        return BoxUniform(
            low=torch.tensor([self.low], dtype=torch.float32), high=torch.tensor([self.high], dtype=torch.float32), device=str(device)
        )

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        return (arr >= self.low) & (arr <= self.high)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"distribution": "uniform"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriorSpec:
        return cls(low=float(data["low"]), high=float(data["high"]), parameter=str(data.get("parameter", "z01_mm")), source=str(data.get("source", "config")))


def read_meta(data_dir: str | Path) -> dict[str, Any]:
    """Load ``<data_dir>/meta.json`` written by :mod:`src.data.generate_data`."""
    path = resolve_path(data_dir) / _META_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; the prior bounds are read from the data set's meta.json")
    return json.loads(path.read_text(encoding="utf-8"))


def prior_from_meta(meta: dict[str, Any], parameter: str = "z01_mm") -> PriorSpec:
    """Prior bounds from ``meta['config']['setup']['z01_mm']`` (the sampling range of the simulation).

    Only a uniform z01 distribution is supported: the NPE prior must equal the distribution the training
    holograms were simulated from, and ``BoxUniform`` is the only prior used here.
    """
    if parameter != "z01_mm":
        raise ValueError(f"parameter must be 'z01_mm', got {parameter!r}")
    try:
        z01 = meta["config"]["setup"]["z01_mm"]
    except (KeyError, TypeError) as exc:
        raise KeyError("meta.json lacks config.setup.z01_mm (min/max)") from exc
    distribution = str(z01.get("distribution", "uniform"))
    if distribution != "uniform":
        raise NotImplementedError(
            f"z01 was sampled {distribution!r}; only a uniform prior is implemented (use a TransformedDistribution for log-uniform)"
        )
    return PriorSpec(low=float(z01["min"]), high=float(z01["max"]), parameter=parameter, source="meta.json")


def prior_from_config(prior_cfg: dict[str, Any], meta: dict[str, Any] | None) -> PriorSpec:
    """``prior.bounds: meta`` (default, bounds from the data set) or an explicit ``[low, high]`` pair in mm."""
    parameter = str(prior_cfg.get("parameter", "z01_mm"))
    bounds = prior_cfg.get("bounds", "meta")
    if isinstance(bounds, str):
        if bounds != "meta":
            raise ValueError(f"prior.bounds must be 'meta' or a [low, high] pair, got {bounds!r}")
        if meta is None:
            raise ValueError("prior.bounds is 'meta' but no meta.json was given")
        return prior_from_meta(meta, parameter)
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"prior.bounds must be 'meta' or a [low, high] pair, got {bounds!r}")
    return PriorSpec(low=float(bounds[0]), high=float(bounds[1]), parameter=parameter, source="config")


# --------------------------------------------------------------------------------------------------
# embedding networks
# --------------------------------------------------------------------------------------------------
class RadialProfileEmbedding(nn.Module):
    """MLP summary network for the radial log-power profile: ``[B, n_bins] -> [B, output_dim]``.

    Reuses the trunk of :class:`src.models.cnn.RadialProfileMLP` (LayerNorm -> hidden layers) with the
    regression output replaced by ``output_dim`` features.
    """

    def __init__(self, n_bins: int, hidden_units: Sequence[int] = (256, 128), output_dim: int = 32, dropout_rate: float = 0.0) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        self.mlp = RadialProfileMLP(n_bins=n_bins, hidden_units=hidden_units, dropout_rate=dropout_rate)
        self.mlp.net[-1] = nn.Linear(int(hidden_units[-1]), int(output_dim))
        self.n_bins = int(n_bins)
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 2 or x.shape[-1] != self.n_bins:
            raise ValueError(f"expected input [B, {self.n_bins}], got {tuple(x.shape)}")
        return self.mlp.net(x)


class HologramCNNEmbedding(nn.Module):
    """CNN summary network for image inputs ``[B, C, H, W] -> [B, output_dim]``.

    Same convolutional trunk as the point-estimator :class:`src.models.cnn.AutofocusCNN` (Conv-BN-ReLU-
    MaxPool blocks, adaptive average pooling, one hidden FC layer) so that the comparison between the
    regressor and the NPE embedding is architecture-matched; only the last layer outputs ``output_dim``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: Sequence[int] = (16, 32, 64, 128),
        fc_units: int = 128,
        pool_size: int = 4,
        output_dim: int = 32,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        self.cnn = AutofocusCNN(conv_channels=conv_channels, fc_units=fc_units, dropout_rate=dropout_rate, pool_size=pool_size, in_channels=in_channels)
        self.cnn.head[-1] = nn.Linear(int(fc_units), int(output_dim))
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected input [B, C, H, W], got {tuple(x.shape)}")
        return self.cnn.head(self.cnn.pool(self.cnn.features(x)))


def build_embedding_net(embedding_cfg: dict[str, Any], input_shape: Sequence[int]) -> nn.Module:
    """Instantiate the embedding described by the ``embedding`` config section for the given ``x`` shape.

    ``type: mlp`` needs a 1-D input ``(n_bins,)``, ``type: cnn`` an image ``(C, H, W)``; ``identity`` passes
    ``x`` straight to the density estimator (only sensible for a handful of features).
    """
    kind = str(embedding_cfg.get("type", "mlp"))
    shape = tuple(int(s) for s in input_shape)
    if kind == "mlp":
        if len(shape) != 1:
            raise ValueError(f"embedding type 'mlp' expects a 1-D input such as the radial profile, got shape {shape}")
        return RadialProfileEmbedding(
            n_bins=shape[0],
            hidden_units=embedding_cfg.get("hidden_units", (256, 128)),
            output_dim=int(embedding_cfg.get("output_dim", 32)),
            dropout_rate=float(embedding_cfg.get("dropout_rate", 0.0)),
        )
    if kind == "cnn":
        if len(shape) != 3:
            raise ValueError(f"embedding type 'cnn' expects an image input (C, H, W), got shape {shape}")
        return HologramCNNEmbedding(
            in_channels=shape[0],
            conv_channels=embedding_cfg.get("conv_channels", (16, 32, 64, 128)),
            fc_units=int(embedding_cfg.get("fc_units", 128)),
            pool_size=int(embedding_cfg.get("pool_size", 4)),
            output_dim=int(embedding_cfg.get("output_dim", 32)),
            dropout_rate=float(embedding_cfg.get("dropout_rate", 0.0)),
        )
    if kind == "identity":
        return nn.Identity()
    raise ValueError(f"unknown embedding.type {kind!r}; expected one of {EMBEDDINGS}")


def build_density_estimator(de_cfg: dict[str, Any], embedding_net: nn.Module) -> Callable[..., Any]:
    """``sbi.neural_nets.posterior_nn`` builder for ``density_estimator.model`` (``nsf`` | ``maf`` | ``mdn``).

    ``z_score_x`` is applied to the raw ``x`` *before* the embedding (per-dimension statistics of the
    training set); ``none`` is recommended for images that are already standardised per sample.
    """
    model = str(de_cfg.get("model", "nsf"))
    if model not in DENSITY_ESTIMATORS:
        raise ValueError(f"unknown density_estimator.model {model!r}; expected one of {DENSITY_ESTIMATORS}")
    kwargs: dict[str, Any] = {
        "model": model,
        "z_score_theta": de_cfg.get("z_score_theta", "independent"),
        "z_score_x": de_cfg.get("z_score_x", "independent"),
        "hidden_features": int(de_cfg.get("hidden_features", 50)),
        "num_transforms": int(de_cfg.get("num_transforms", 5)),
        "num_bins": int(de_cfg.get("num_bins", 10)),
        "num_components": int(de_cfg.get("num_components", 10)),
        "embedding_net": embedding_net,
    }
    return posterior_nn(**kwargs)


# --------------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def load_simulations(dataset: HologramHDF5Dataset, batch_size: int = 64, num_workers: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """All ``(theta, x)`` pairs of a dataset as float32 tensors: ``theta`` ``[N, 1]`` (z01 in mm), ``x`` ``[N, *output_shape]``.

    The holograms are run through the dataset's preprocessing (crop, pooling, normalisation,
    representation), so ``x`` is exactly what the embedding network sees at inference.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    xs = [batch_x for batch_x, _ in loader]
    x = torch.cat(xs, dim=0).float()
    theta = torch.from_numpy(np.asarray(dataset.z01_mm, dtype=np.float32)).reshape(-1, 1)
    if theta.shape[0] != x.shape[0]:
        raise RuntimeError(f"inconsistent sample counts: {theta.shape[0]} labels vs {x.shape[0]} inputs")
    return theta, x


class FixedSplitNPE(NPE):
    """``sbi`` NPE whose validation set can be pinned to given sample indices.

    ``sbi`` draws a random train/validation split (``validation_fraction``) from all appended simulations.
    With ``val_indices`` the samples at these indices form the validation set and all others the training
    set -- used to keep the ``train.hdf5`` / ``val.hdf5`` files of a data set as training/validation data,
    like the point-estimator runs do.  Everything else (loss, early stopping, batching) is unchanged; the
    loader construction mirrors ``sbi`` 0.27 ``NeuralInference.get_dataloaders``.
    """

    def __init__(self, *args: Any, val_indices: torch.Tensor | Sequence[int] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fixed_val_indices = None if val_indices is None else torch.as_tensor(val_indices, dtype=torch.long).reshape(-1)

    def get_dataloaders(
        self,
        starting_round: int = 0,
        training_batch_size: int = 200,
        validation_fraction: float = 0.1,
        resume_training: bool = False,
        dataloader_kwargs: dict[str, Any] | None = None,
    ) -> tuple[DataLoader, DataLoader]:
        if self._fixed_val_indices is None:
            return super().get_dataloaders(starting_round, training_batch_size, validation_fraction, resume_training, dataloader_kwargs)
        theta, x, prior_masks = self.get_simulations(starting_round)
        dataset = torch_data.TensorDataset(theta, x, prior_masks)
        num_examples = theta.shape[0]
        val_indices = self._fixed_val_indices
        if val_indices.numel() == 0 or int(val_indices.min()) < 0 or int(val_indices.max()) >= num_examples:
            raise ValueError(f"val_indices must be non-empty and within [0, {num_examples})")
        if not resume_training:
            is_val = torch.zeros(num_examples, dtype=torch.bool)
            is_val[val_indices] = True
            self.train_indices = torch.arange(num_examples)[~is_val]
            self.val_indices = torch.arange(num_examples)[is_val]
        if self.train_indices.numel() == 0:
            raise ValueError("val_indices cover all samples; nothing left for training")
        train_loader_kwargs: dict[str, Any] = {
            "batch_size": min(training_batch_size, int(self.train_indices.numel())),
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.train_indices.tolist()),
        }
        val_loader_kwargs: dict[str, Any] = {
            "batch_size": min(training_batch_size, int(self.val_indices.numel())),
            "shuffle": False,
            "drop_last": True,
            "sampler": SubsetRandomSampler(self.val_indices.tolist()),
        }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
            val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)
        return DataLoader(dataset, **train_loader_kwargs), DataLoader(dataset, **val_loader_kwargs)


# --------------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------------
@dataclass
class NPETrainingResult:
    """Outcome of :func:`fit_npe`."""

    posterior: DirectPosterior | EnsemblePosterior
    """Amortised posterior; an ``EnsemblePosterior`` (equal-weight mixture) when ``train.ensemble_size > 1``."""
    members: list[DirectPosterior]
    density_estimator: nn.Module
    inference: NPE
    prior: PriorSpec
    summary: dict[str, Any]
    """Training summary: ``epochs_trained``/``best_validation_loss`` (lists, one entry per member), per-epoch
    ``training_loss``/``validation_loss``/``epoch_durations_sec`` of every member under ``members``."""
    train_time_s: float
    num_simulations: dict[str, int]
    setup_constants: dict[str, Any]
    dataset_kwargs: dict[str, Any]
    input_shape: tuple[int, ...]
    num_parameters: dict[str, int]
    validation_split: str
    ensemble_size: int
    datasets: dict[str, HologramHDF5Dataset] = field(repr=False)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def fit_npe(
    config: dict[str, Any],
    data_dir: str | Path | None = None,
    summary_writer: Any | None = None,
) -> NPETrainingResult:
    """Train an amortised NPE posterior ``q(z01 | x)`` on the ``train``/``val`` splits of a data set.

    Config sections: ``data`` (as in the regression configs plus ``validation_split``), ``prior``
    (``parameter``, ``bounds``), ``embedding`` (``type`` + architecture), ``density_estimator``
    (``model`` + flow options) and ``train`` (``training_batch_size``, ``learning_rate``,
    ``stop_after_epochs``, ``max_num_epochs``, ``clip_max_norm``, ``validation_fraction``, ``seed``,
    ``num_threads``, ``device``).
    """
    require_keys(config, ("data", "prior", "embedding", "density_estimator", "train"), "<root>")
    data_cfg, train_cfg = config["data"], config["train"]
    seed = int(train_cfg.get("seed", 0))
    if train_cfg.get("num_threads"):
        torch.set_num_threads(int(train_cfg["num_threads"]))
    seed_everything(seed, deterministic=bool(train_cfg.get("deterministic", False)))
    device = select_device(train_cfg.get("device", "cpu"))

    directory = resolve_path(data_dir if data_dir is not None else data_cfg.get("dir", ""))
    if not directory.is_dir():
        raise FileNotFoundError(f"data directory not found: {directory}")
    meta = read_meta(directory) if (directory / _META_FILE).is_file() else None
    prior = prior_from_config(config["prior"], meta)

    kwargs = dataset_kwargs_from_config(data_cfg) | {"target_mode": "z01_mm"}
    datasets = {split: HologramHDF5Dataset(directory / f"{split}.hdf5", **kwargs) for split in ("train", "val")}
    batch_size = int(data_cfg.get("batch_size", 64))
    theta_train, x_train = load_simulations(datasets["train"], batch_size)
    theta_val, x_val = load_simulations(datasets["val"], batch_size)
    for name, theta in (("train", theta_train), ("val", theta_val)):
        outside = ~prior.contains(theta.numpy())
        if outside.any():
            raise ValueError(f"{int(outside.sum())} {name} samples lie outside the prior [{prior.low}, {prior.high}] mm")

    split = str(data_cfg.get("validation_split", "files"))
    if split not in VALIDATION_SPLITS:
        raise ValueError(f"data.validation_split must be one of {VALIDATION_SPLITS}, got {split!r}")
    theta = torch.cat([theta_train, theta_val], dim=0)
    x = torch.cat([x_train, x_val], dim=0)
    val_indices = torch.arange(theta_train.shape[0], theta.shape[0]) if split == "files" else None
    validation_fraction = float(train_cfg.get("validation_fraction", 0.2))

    input_shape = tuple(datasets["train"].output_shape)
    ensemble_size = int(train_cfg.get("ensemble_size", 1))
    if ensemble_size < 1:
        raise ValueError(f"train.ensemble_size must be >= 1, got {ensemble_size}")

    members: list[DirectPosterior] = []
    member_summaries: list[dict[str, Any]] = []
    inference: FixedSplitNPE | None = None
    density_estimator: nn.Module | None = None
    embedding_net: nn.Module | None = None
    start = time.perf_counter()
    for member in range(ensemble_size):
        # every member gets its own initialisation (and, for a random split, its own split)
        seed_everything(seed + member, deterministic=bool(train_cfg.get("deterministic", False)))
        embedding_net = build_embedding_net(config["embedding"], input_shape)
        builder = build_density_estimator(config["density_estimator"], embedding_net)
        inference = FixedSplitNPE(
            prior=prior.build(device),
            density_estimator=builder,
            device=str(device),
            summary_writer=summary_writer if member == 0 else None,
            show_progress_bars=bool(train_cfg.get("show_progress_bars", False)),
            val_indices=val_indices,
        )
        inference.append_simulations(theta, x)
        density_estimator = inference.train(
            training_batch_size=int(train_cfg.get("training_batch_size", 50)),
            learning_rate=float(train_cfg.get("learning_rate", 5e-4)),
            validation_fraction=validation_fraction,
            stop_after_epochs=int(train_cfg.get("stop_after_epochs", 20)),
            max_num_epochs=int(train_cfg.get("max_num_epochs", 500)),
            clip_max_norm=train_cfg.get("clip_max_norm", 5.0),
            show_train_summary=False,
        )
        density_estimator.eval()
        members.append(inference.build_posterior(density_estimator))
        member_summary = {key: _to_builtin(value) for key, value in inference.summary.items()}
        for key in ("epochs_trained", "best_validation_loss"):
            if isinstance(member_summary.get(key), list) and len(member_summary[key]) == 1:
                member_summary[key] = member_summary[key][0]
        member_summaries.append(member_summary)
    train_time = time.perf_counter() - start
    assert inference is not None and density_estimator is not None and embedding_net is not None

    posterior: DirectPosterior | EnsemblePosterior = members[0] if ensemble_size == 1 else EnsemblePosterior(members)
    summary: dict[str, Any] = {
        "ensemble_size": ensemble_size,
        "epochs_trained": [s.get("epochs_trained") for s in member_summaries],
        "best_validation_loss": [s.get("best_validation_loss") for s in member_summaries],
        "members": member_summaries,
        # per-epoch curves of the first member for backwards-compatible consumers
        "training_loss": member_summaries[0].get("training_loss", []),
        "validation_loss": member_summaries[0].get("validation_loss", []),
        "epoch_durations_sec": member_summaries[0].get("epoch_durations_sec", []),
    }
    return NPETrainingResult(
        posterior=posterior,
        members=members,
        density_estimator=density_estimator,
        inference=inference,
        prior=prior,
        summary=summary,
        train_time_s=train_time,
        num_simulations={"train": int(inference.train_indices.numel()), "val": int(inference.val_indices.numel())},
        setup_constants=datasets["train"].setup_constants,
        dataset_kwargs=kwargs,
        input_shape=input_shape,
        num_parameters={
            "embedding": count_parameters(embedding_net),
            "per_member": count_parameters(density_estimator),
            "total": count_parameters(density_estimator) * ensemble_size,
        },
        validation_split=split,
        ensemble_size=ensemble_size,
        datasets=datasets,
    )


# --------------------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------------------
def save_posterior(path: str | Path, result: NPETrainingResult, config: dict[str, Any]) -> Path:
    """Pickle the posterior with everything needed to use it: config, prior, geometry, preprocessing."""
    import sbi

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "posterior": result.posterior,
        "config": config,
        "prior": result.prior.to_dict(),
        "setup_constants": result.setup_constants,
        "dataset_kwargs": result.dataset_kwargs,
        "input_shape": list(result.input_shape),
        "training_summary": result.summary,
        "train_time_s": result.train_time_s,
        "num_simulations": result.num_simulations,
        "num_parameters": result.num_parameters,
        "validation_split": result.validation_split,
        "ensemble_size": result.ensemble_size,
        "versions": {"sbi": sbi.__version__, "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
    }
    torch.save(bundle, path)
    return path


def posterior_members(posterior: DirectPosterior | EnsemblePosterior) -> list[DirectPosterior]:
    """The ``DirectPosterior`` members of an ensemble, or ``[posterior]`` for a single posterior."""
    if isinstance(posterior, EnsemblePosterior):
        return list(posterior.posteriors)
    return [posterior]


def embedding_net_of(posterior: DirectPosterior | EnsemblePosterior) -> nn.Module:
    """Embedding network of the (first) density estimator."""
    return posterior_members(posterior)[0].posterior_estimator.embedding_net


def load_posterior(path: str | Path) -> dict[str, Any]:
    """Load a bundle written by :func:`save_posterior`; ``bundle['posterior']`` is the (ensemble) posterior."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"posterior file not found: {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("posterior", "config", "prior", "dataset_kwargs"):
        if key not in bundle:
            raise KeyError(f"{path} lacks {key!r}; was it written by src.sbi.npe.save_posterior?")
    for member in posterior_members(bundle["posterior"]):
        member.posterior_estimator.eval()
    return bundle


# --------------------------------------------------------------------------------------------------
# inference helpers
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_posterior(
    posterior: DirectPosterior | EnsemblePosterior,
    x: torch.Tensor,
    n_samples: int = 1000,
    chunk_size: int = 25,
    seed: int | None = None,
) -> np.ndarray:
    """Posterior samples for a batch of observations: ``x`` ``[N, *shape]`` -> ``[N, n_samples]`` (mm).

    Uses ``sample_batched`` in chunks of observations; samples outside the prior are rejected by ``sbi``
    (leakage correction), so all returned values lie inside the prior bounds.  For an ensemble the
    samples are drawn from the equal-weight mixture of the members.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    if x.ndim < 2:
        raise ValueError(f"x must be a batch [N, *shape], got {tuple(x.shape)}")
    if seed is not None:
        torch.manual_seed(seed)
    chunks: list[np.ndarray] = []
    for start in range(0, x.shape[0], max(1, chunk_size)):
        batch = x[start : start + chunk_size]
        samples = posterior.sample_batched((n_samples,), x=batch, show_progress_bars=False)
        chunks.append(samples[..., 0].T.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def parameter_grid(prior: PriorSpec, num_points: int = 1001) -> np.ndarray:
    """Equidistant grid over the prior support (used for MAP search and density plots)."""
    return np.linspace(prior.low, prior.high, int(num_points))


@torch.no_grad()
def log_prob_grid(
    posterior: DirectPosterior | EnsemblePosterior,
    x: torch.Tensor,
    grid: np.ndarray,
    normalize: bool | None = None,
) -> np.ndarray:
    """Posterior log-density on ``grid`` for every observation: ``[N, len(grid)]``.

    For a single posterior the leakage normalisation is skipped by default (``norm_posterior=False``): it
    does not change the shape of the density, and MAP and plotted curves are normalised numerically on the
    grid.  For an ensemble the members are normalised (``norm_posterior=True``) before mixing, otherwise
    members with different leakage would be weighted inconsistently.
    """
    if normalize is None:
        normalize = isinstance(posterior, EnsemblePosterior)
    theta = torch.as_tensor(np.asarray(grid, dtype=np.float32)).reshape(-1, 1)
    rows = []
    for i in range(x.shape[0]):
        logp = posterior.log_prob(theta, x=x[i : i + 1], norm_posterior=bool(normalize))
        rows.append(logp.cpu().numpy().astype(np.float64))
    return np.stack(rows, axis=0)


def _refine_argmax(grid: np.ndarray, logp: np.ndarray) -> np.ndarray:
    """Parabolic interpolation of the grid maximum (sub-grid MAP), one value per row."""
    idx = np.argmax(logp, axis=1)
    out = grid[idx].astype(np.float64)
    step = float(grid[1] - grid[0]) if grid.size > 1 else 0.0
    for row, k in enumerate(idx):
        if 0 < k < grid.size - 1 and step > 0:
            y0, y1, y2 = logp[row, k - 1], logp[row, k], logp[row, k + 1]
            denominator = y0 - 2.0 * y1 + y2
            if np.isfinite(denominator) and denominator < 0:
                out[row] = grid[k] + 0.5 * (y0 - y2) / denominator * step
    return out


def point_estimates(samples: np.ndarray, grid: np.ndarray | None = None, logp: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-observation summaries of posterior samples ``[N, S]``: ``mean``, ``median``, ``std``, central
    68 %/95 % interval bounds and widths and -- if ``grid``/``logp`` are given -- the ``map``."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"samples must be [N, S], got {arr.shape}")
    out: dict[str, np.ndarray] = {
        "mean": arr.mean(axis=1),
        "median": np.median(arr, axis=1),
        "std": arr.std(axis=1, ddof=1),
    }
    for level in (0.68, 0.95):
        lo, hi = central_interval(arr, level)
        tag = f"{int(round(level * 100))}"
        out[f"lo_{tag}"], out[f"hi_{tag}"], out[f"width_{tag}"] = lo, hi, hi - lo
    if grid is not None and logp is not None:
        out["map"] = _refine_argmax(np.asarray(grid, dtype=np.float64), np.asarray(logp, dtype=np.float64))
    return out


@torch.no_grad()
def leakage_acceptance(posterior: DirectPosterior | EnsemblePosterior, x: torch.Tensor, num_samples: int = 10_000) -> np.ndarray:
    """Fraction of flow samples inside the prior support per observation (1 = no leakage; ensemble: mean over members).

    A low acceptance means the flow places mass outside the prior, which for a well-trained estimator
    happens mainly for observations unlike the training data (misspecification / out-of-range z01).
    """
    members = posterior_members(posterior)
    values = []
    for i in range(x.shape[0]):
        acceptances = [
            float(torch.as_tensor(member.leakage_correction(x[i : i + 1], num_rejection_samples=int(num_samples), force_update=True)).reshape(-1)[0])
            for member in members
        ]
        values.append(float(np.mean(acceptances)))
    return np.asarray(values, dtype=np.float64)


# --------------------------------------------------------------------------------------------------
# hybrid interface to HoloWizard's model-based autofocus
# --------------------------------------------------------------------------------------------------
def posterior_to_find_focus_init(
    samples: Any,
    level: float = 0.95,
    centre: str = "median",
    min_confidence_mm: float = 0.0,
) -> dict[str, float]:
    """Turn posterior samples of one hologram into the start value + search interval of HoloWizard.

    HoloWizard's ``Measurement(z01=..., z01_confidence=...)`` defines the Nelder-Mead search range of
    ``find_focus`` as ``z01_bounds = (z01 - z01_confidence, z01 + z01_confidence)``.  Here ``z01`` is the
    posterior median (or mean) and ``z01_confidence`` the half-width that covers the central ``level``
    credible interval (the larger of the two distances from the centre to the interval bounds, so an
    asymmetric posterior is fully enclosed), optionally floored at ``min_confidence_mm``.

    Returns ``{"z01": ..., "z01_confidence": ...}`` in mm.
    """
    arr = np.asarray(samples, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError("need at least two posterior samples")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if centre not in ("median", "mean"):
        raise ValueError(f"centre must be 'median' or 'mean', got {centre!r}")
    if min_confidence_mm < 0:
        raise ValueError("min_confidence_mm must be non-negative")
    alpha = 1.0 - level
    lo, hi = np.quantile(arr, [alpha / 2.0, 1.0 - alpha / 2.0])
    z01 = float(np.median(arr) if centre == "median" else arr.mean())
    confidence = max(hi - z01, z01 - lo, min_confidence_mm)
    return {"z01": z01, "z01_confidence": float(confidence)}


def find_focus_bounds(init: dict[str, float]) -> tuple[float, float]:
    """``(z01 - z01_confidence, z01 + z01_confidence)`` exactly as ``Measurement.z01_bounds`` in HoloWizard."""
    return (init["z01"] - init["z01_confidence"], init["z01"] + init["z01_confidence"])
