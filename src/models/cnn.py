"""Regression networks mapping a single-channel hologram to one scalar (log Fr, Fr or z01)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

__all__ = ["AutofocusCNN", "build_model", "ARCHITECTURES"]

ARCHITECTURES: tuple[str, ...] = ("cnn", "resnet18")


class AutofocusCNN(nn.Module):
    """Stack of Conv3x3-BatchNorm-ReLU-MaxPool blocks followed by adaptive pooling and an MLP head.

    ``nn.AdaptiveAvgPool2d`` makes the head independent of the input resolution, so 128 px, 256 px
    or cropped/downsampled holograms all work with the same weights.

    Args:
        conv_channels: Output channels of each block, e.g. ``[16, 32, 64, 128]`` (one block per entry).
        fc_units: Hidden units of the fully connected head.
        dropout_rate: Dropout probability in the head.
        pool_size: Spatial size after adaptive average pooling (``pool_size x pool_size``).
        in_channels: Number of input channels (1 for a hologram).
    """

    def __init__(
        self,
        conv_channels: Sequence[int] = (16, 32, 64, 128),
        fc_units: int = 256,
        dropout_rate: float = 0.3,
        pool_size: int = 4,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        if len(conv_channels) == 0:
            raise ValueError("conv_channels must contain at least one entry")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        blocks: list[nn.Module] = []
        channels_in = in_channels
        for channels_out in conv_channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            channels_in = channels_out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels_in * pool_size * pool_size, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 1, H, W] -> [B]``; H and W must be at least ``2 ** len(conv_channels)``."""
        return self.head(self.pool(self.features(x))).squeeze(-1)


class _ResNet18Regressor(nn.Module):
    """torchvision ResNet-18 with a single-channel stem and one regression output (no pretrained weights)."""

    def __init__(self, in_channels: int = 1, dropout_rate: float = 0.0) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18
        except ImportError as exc:
            raise ImportError("arch 'resnet18' requires torchvision; install it or use arch 'cnn'") from exc
        self.backbone = resnet18(weights=None, num_classes=1)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 1, H, W] -> [B]``."""
        return self.backbone(x).squeeze(-1)


def build_model(config: dict[str, Any]) -> nn.Module:
    """Instantiate the architecture described by ``config['model']`` (``arch: cnn | resnet18``)."""
    model_cfg = config.get("model", config)
    arch = str(model_cfg.get("arch", "cnn"))
    if arch == "cnn":
        return AutofocusCNN(
            conv_channels=model_cfg.get("conv_channels", (16, 32, 64, 128)),
            fc_units=int(model_cfg.get("fc_units", 256)),
            dropout_rate=float(model_cfg.get("dropout_rate", 0.3)),
            pool_size=int(model_cfg.get("pool_size", 4)),
            in_channels=int(model_cfg.get("in_channels", 1)),
        )
    if arch == "resnet18":
        return _ResNet18Regressor(
            in_channels=int(model_cfg.get("in_channels", 1)),
            dropout_rate=float(model_cfg.get("dropout_rate", 0.0)),
        )
    raise ValueError(f"unknown model.arch {arch!r}; expected one of {ARCHITECTURES}")
