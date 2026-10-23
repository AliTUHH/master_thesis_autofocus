"""YAML configuration of the learned autofocus (Stage 4) and of its NPE (Stage 7).

Example (``configs/learned/spectrum-cnn.yaml``)::

    name: spectrum-cnn
    target_scale: 5.0e-3      # dz01 is learned in units of the +-5 mm of Omega_z ([2])
    representation:
      kind: spectrum          # radial power spectrum on an f^2 axis -> 1-D CNN
      bins: 1024
      f2_max: 0.125
    model:
      channels: [16, 32, 64, 64]
    training:
      epochs: 60

:class:`NPEConfig` (``configs/learned/npe-*.yaml``) has the same ``representation``, an
``embedding`` in place of ``model``, a ``density`` section for the density estimator of sbi
and its own ``training``. Lengths in metres. Unknown keys are rejected.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..yamlconfig import build, plain

REPRESENTATIONS = ("image", "spectrum", "polar")


@dataclass(frozen=True)
class RepresentationConfig:
    """Network input computed from one flat-field corrected hologram.

    ``image``: the hologram binned by ``binning`` and standardized (2-D CNN).
    ``spectrum``: radially averaged power spectrum in ``bins`` bins of ``f^2`` from 0 to
    ``f2_max`` (cycles^2 per pixel^2 of the stored grid), logarithmic and standardized
    (1-D CNN).
    ``polar``: the power spectrum in ``angles`` sectors of the direction of the spatial
    frequency (modulo 180 degrees) times the same ``bins`` of ``f^2``, logarithmic and
    standardized (Stage 10). ``angles`` is even, so a rotation by 90 degrees shifts the
    sectors by ``angles / 2``.
    """

    kind: str = "spectrum"
    binning: int = 2
    bins: int = 1024
    f2_max: float = 0.125
    angles: int = 8

    def __post_init__(self) -> None:
        if self.kind not in REPRESENTATIONS:
            raise ValueError(f"representation {self.kind!r}, choose from {REPRESENTATIONS}")
        if self.binning < 1 or self.bins < 8:
            raise ValueError("binning must be >= 1 and bins >= 8")
        if not 0 < self.f2_max <= 0.5:
            raise ValueError("f2_max must lie in (0, 0.5] (Nyquist corner)")
        if self.angles < 4 or self.angles % 2:
            raise ValueError("angles must be even and at least 4")


@dataclass(frozen=True)
class ModelConfig:
    """CNN with a Gaussian head (mean and variance of ``dz01 / target_scale``).

    ``channels`` per stage; every stage halves the resolution. The image CNN uses a 5x5
    stem and 3x3 convolutions, the spectrum CNN ``kernel_size`` and adaptive average pooling
    to ``pool_to`` positions before the head. The polar spectrum CNN has a first stage over
    angle and ``f^2`` and the spectrum CNN after it. ``min_sigma`` bounds the predicted
    standard deviation from below (units of ``target_scale``).
    """

    channels: tuple[int, ...] = (16, 32, 64, 64)
    kernel_size: int = 7
    pool_to: int = 16
    hidden: int = 128
    dropout: float = 0.1
    min_sigma: float = 1.0e-3

    def __post_init__(self) -> None:
        if not self.channels or min(self.channels) < 1:
            raise ValueError("channels must be a non-empty list of positive integers")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if self.pool_to < 1 or self.hidden < 1 or not 0 <= self.dropout < 1:
            raise ValueError("pool_to and hidden must be positive, dropout in [0, 1)")
        if self.min_sigma <= 0:
            raise ValueError("min_sigma must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    """AdamW with linear warm-up and cosine decay; the epoch with the lowest validation MAE
    is kept.

    ``beta`` weights the Gaussian NLL per sample by ``sigma^(2 beta)`` (beta-NLL; 0 is the
    plain NLL, 1 gives the mean the gradient of the squared error). ``augment`` applies
    random rotations by 90 degrees and flips (image input). ``limit`` uses only the first N
    training samples (learning curves); ``patience`` stops after that many epochs without
    improvement.
    """

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    warmup_epochs: int = 2
    beta: float = 0.5
    augment: bool = True
    limit: int | None = None
    patience: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("epochs must be >= 1 and batch_size >= 2")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.warmup_epochs < 0:
            raise ValueError("learning_rate must be positive, weight_decay and warm-up >= 0")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must lie in [0, 1]")
        if self.limit is not None and self.limit < 2:
            raise ValueError("limit must be at least 2")


@dataclass(frozen=True)
class LearnedConfig:
    """Representation, model and training of one learned autofocus."""

    name: str = "learned"
    target_scale: float = 5.0e-3
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.target_scale <= 0:
            raise ValueError("target_scale must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedConfig:
        return build(cls, data, "config", _SECTIONS)

    @classmethod
    def from_yaml(cls, path: str | Path) -> LearnedConfig:
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        return plain(dataclasses.asdict(self))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


_SECTIONS = {
    "representation": RepresentationConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
}

DENSITY_ESTIMATORS = ("nsf", "mdn")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding network of the NPE: the convolution stack of the Stage 4 CNN for the
    representation (``channels``, ``kernel_size``, ``pool_to`` as in :class:`ModelConfig`) and
    an MLP with one hidden layer of ``hidden`` units and ``dropout`` to ``features`` outputs,
    the condition of the density estimator (a learned summary statistic, [6] Sec. 3.2).
    """

    channels: tuple[int, ...] = (16, 32, 64, 64)
    kernel_size: int = 7
    pool_to: int = 16
    hidden: int = 128
    dropout: float = 0.1
    features: int = 32

    def __post_init__(self) -> None:
        self.model_config()
        if self.features < 1:
            raise ValueError("features must be positive")

    def model_config(self) -> ModelConfig:
        """The convolution stack and head width as a :class:`ModelConfig` (validated)."""
        return ModelConfig(
            channels=self.channels,
            kernel_size=self.kernel_size,
            pool_to=self.pool_to,
            hidden=self.hidden,
            dropout=self.dropout,
        )


@dataclass(frozen=True)
class DensityConfig:
    """Conditional density estimator of sbi (``posterior_nn``) for ``t`` given the embedding.

    ``nsf``: neural spline flow of ``num_transforms`` rational-quadratic splines with
    ``num_bins`` bins, whose parameters an MLP with ``hidden_features`` units computes from
    the embedding (for one parameter they depend on the embedding only). ``mdn``: mixture of
    ``num_components`` Gaussians from an MLP with ``hidden_features`` units. ``t`` enters
    z-scored with the mean and standard deviation of the uniform prior.
    """

    estimator: str = "nsf"
    hidden_features: int = 50
    num_transforms: int = 5
    num_bins: int = 10
    num_components: int = 10

    def __post_init__(self) -> None:
        if self.estimator not in DENSITY_ESTIMATORS:
            raise ValueError(f"estimator {self.estimator!r}, choose from {DENSITY_ESTIMATORS}")
        if min(self.hidden_features, self.num_transforms, self.num_bins) < 1:
            raise ValueError("hidden_features, num_transforms and num_bins must be positive")
        if self.num_components < 1:
            raise ValueError("num_components must be positive")


@dataclass(frozen=True)
class NPETrainingConfig:
    """AdamW with linear warm-up and cosine decay (the schedule of the Stage 4 CNN) on the
    NPE loss ``-log q(t | x)``; the epoch with the smallest validation loss is kept ([6],
    A1.1). ``clip_max_norm`` clips the gradient norm (sbi's default 5; ``null``: off).
    ``limit``, ``patience`` and ``seed`` as in :class:`TrainingConfig`.
    """

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    warmup_epochs: int = 2
    clip_max_norm: float | None = 5.0
    limit: int | None = None
    patience: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("epochs must be >= 1 and batch_size >= 2")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.warmup_epochs < 0:
            raise ValueError("learning_rate must be positive, weight_decay and warm-up >= 0")
        if self.clip_max_norm is not None and self.clip_max_norm <= 0:
            raise ValueError("clip_max_norm must be positive or null")
        if self.limit is not None and self.limit < 2:
            raise ValueError("limit must be at least 2")


@dataclass(frozen=True)
class NPEConfig:
    """Neural posterior estimation of ``t = dz01 / target_scale`` (Stage 7).

    ``grid_points``: values of ``t`` over the prior box at which the posterior density is
    evaluated for its median, intervals and calibration statistics.
    """

    name: str = "npe"
    target_scale: float = 5.0e-3
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    training: NPETrainingConfig = field(default_factory=NPETrainingConfig)
    grid_points: int = 2001
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.target_scale <= 0:
            raise ValueError("target_scale must be positive")
        if self.grid_points < 101:
            raise ValueError("grid_points must be at least 101")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NPEConfig:
        return build(cls, data, "config", _NPE_SECTIONS)

    @classmethod
    def from_yaml(cls, path: str | Path) -> NPEConfig:
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        return plain(dataclasses.asdict(self))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


_NPE_SECTIONS = {
    "representation": RepresentationConfig,
    "embedding": EmbeddingConfig,
    "density": DensityConfig,
    "training": NPETrainingConfig,
}
