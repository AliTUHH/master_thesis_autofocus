"""CNNs with a Gaussian output for ``t = dz01 / target_scale`` and the beta-NLL loss.

Both networks return ``(mean, var)`` of shape ``(B,)``. The variance is
``softplus(raw) + min_sigma^2``, so it is positive and bounded away from zero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import LearnedConfig, ModelConfig


class GaussianHead(nn.Module):
    """MLP from pooled features to the mean and variance of the target."""

    def __init__(self, n_in: int, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, config.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, 2),
        )
        self.min_var = config.min_sigma**2

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(features)
        return out[:, 0], F.softplus(out[:, 1]) + self.min_var


def _conv_bn(conv: nn.Module, channels: int, two_d: bool) -> list[nn.Module]:
    norm = nn.BatchNorm2d(channels) if two_d else nn.BatchNorm1d(channels)
    return [conv, norm, nn.ReLU(inplace=True)]


class ImageCNN(nn.Module):
    """2-D CNN on the binned hologram.

    Stage 0 is a strided 5x5 stem; every further stage is a strided 3x3 convolution followed
    by a 3x3 convolution. Global average pooling makes the features independent of the
    position of the object; the fringe spacing that encodes ``Fr`` is a local property.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for i, c in enumerate(config.channels):
            k = 5 if i == 0 else 3
            layers += _conv_bn(nn.Conv2d(c_in, c, k, 2, k // 2, bias=False), c, True)
            if i > 0:
                layers += _conv_bn(nn.Conv2d(c, c, 3, 1, 1, bias=False), c, True)
            c_in = c
        self.features = nn.Sequential(*layers)
        self.n_features = c_in
        self.head = GaussianHead(self.n_features, config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.features(x).mean(dim=(-2, -1)))


class SpectrumCNN(nn.Module):
    """1-D CNN on the radial power spectrum.

    Every stage is a convolution with ``kernel_size`` and a max-pooling by 2. Adaptive
    average pooling to ``pool_to`` positions keeps coarse information on where in ``f^2``
    a feature occurs (the CTF minima fade into noise at high ``f^2``) at a small head.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in, k = 1, config.kernel_size
        for c in config.channels:
            layers += _conv_bn(nn.Conv1d(c_in, c, k, 1, k // 2, bias=False), c, False)
            layers.append(nn.MaxPool1d(2))
            c_in = c
        layers.append(nn.AdaptiveAvgPool1d(config.pool_to))
        self.features = nn.Sequential(*layers)
        self.n_features = c_in * config.pool_to
        self.head = GaussianHead(self.n_features, config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.features(x).flatten(1))


class AngularConv(nn.Module):
    """Convolution over ``(angle, f^2)`` with a kernel of 3 sectors times ``kernel_size`` bins
    and ``stride`` along ``f^2``: cyclic along the angle, zero padded along ``f^2`` and
    symmetric in the angle, so it commutes with every cyclic shift and every reversal of the
    sectors."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, (3, kernel_size), bias=False)
        self.pad = kernel_size // 2
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x[:, :, -1:], x, x[:, :, :1]], dim=2)
        weight = 0.5 * (self.conv.weight + self.conv.weight.flip(2))
        return F.conv2d(x, weight, stride=(1, self.stride), padding=(0, self.pad))


class AnglePool(nn.Module):
    """``(B, C, angles, F)`` -> ``(B, 2 C, F)``: mean and maximum over the sectors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)


class PolarSpectrumCNN(nn.Module):
    """CNN on the polar power spectrum ``(1, angles, bins)`` (Stage 10).

    Stage 0 is an :class:`AngularConv` with ``channels[0]`` outputs and stride 2 along
    ``f^2`` (in place of the max-pooling of :class:`SpectrumCNN`, which on all sectors costs
    twice the time of the whole spectrum CNN); the mean and the maximum over the sectors
    (:class:`AnglePool`) give ``2 channels[0]`` channels on the ``f^2`` axis, on which the
    stages ``channels[1:]`` of :class:`SpectrumCNN` follow. The maximum keeps the sectors
    across a fiber, the mean the radial average. The output does not change under rotations
    of the hologram by 90 degrees and flips (cyclic shifts and reversals of the sectors).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        c, k = config.channels[0], config.kernel_size
        layers: list[nn.Module] = [*_conv_bn(AngularConv(1, c, k, stride=2), c, True), AnglePool()]
        c_in = 2 * c
        for c in config.channels[1:]:
            layers += _conv_bn(nn.Conv1d(c_in, c, k, 1, k // 2, bias=False), c, False)
            layers.append(nn.MaxPool1d(2))
            c_in = c
        layers.append(nn.AdaptiveAvgPool1d(config.pool_to))
        self.features = nn.Sequential(*layers)
        self.n_features = c_in * config.pool_to
        self.head = GaussianHead(self.n_features, config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.features(x).flatten(1))


def build_cnn(kind: str, config: ModelConfig) -> nn.Module:
    """The network for the representation ``kind``."""
    if kind == "image":
        return ImageCNN(config)
    if kind == "polar":
        return PolarSpectrumCNN(config)
    return SpectrumCNN(config)


def build_model(config: LearnedConfig) -> nn.Module:
    """The network for ``config.representation.kind``."""
    return build_cnn(config.representation.kind, config.model)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def gaussian_nll(
    mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor, beta: float = 0.0
) -> torch.Tensor:
    """Mean Gaussian negative log-likelihood without the constant ``log(2 pi) / 2``.

    ``beta > 0`` multiplies each term by the detached ``var^beta`` (beta-NLL, Seitzer et
    al., ICLR 2022), which keeps the NLL from neglecting the mean of hard samples.
    """
    nll = 0.5 * (torch.log(var) + (target - mean).square() / var)
    if beta > 0:
        nll = nll * var.detach() ** beta
    return nll.mean()
