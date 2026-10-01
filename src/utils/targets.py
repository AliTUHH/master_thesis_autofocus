"""Affine standardisation of regression targets (the network predicts ``(t - mean) / std``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

__all__ = ["TargetScaler"]


@dataclass(frozen=True)
class TargetScaler:
    """Maps targets to zero mean / unit variance and back; ``mean=0, std=1`` is the identity."""

    mean: float = 0.0
    std: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std <= 0:
            raise ValueError(f"invalid scaler parameters mean={self.mean}, std={self.std}")

    @classmethod
    def from_values(cls, values: ArrayLike) -> TargetScaler:
        """Fit mean and standard deviation on (training) targets; falls back to std=1 for constant targets."""
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError("cannot fit TargetScaler on empty values")
        std = float(arr.std())
        return cls(mean=float(arr.mean()), std=std if std > 0 else 1.0)

    @classmethod
    def identity(cls) -> TargetScaler:
        return cls(0.0, 1.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetScaler:
        return cls(mean=float(data["mean"]), std=float(data["std"]))

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    def transform(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.mean) / self.std

    def inverse(self, scaled: torch.Tensor) -> torch.Tensor:
        return scaled * self.std + self.mean
