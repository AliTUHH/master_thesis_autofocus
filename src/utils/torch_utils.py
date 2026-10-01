"""Seeding and device selection helpers shared by training and evaluation."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything", "select_device", "count_parameters"]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed ``random``, NumPy and PyTorch; optionally request deterministic kernels (warn-only)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(preference: str | None = "auto") -> torch.device:
    """Resolve ``auto | cpu | cuda | cuda:N | mps`` to an available device (auto = cuda > mps > cpu)."""
    name = (preference or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {preference!r} requested but CUDA is not available")
    if name == "mps" and not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        raise RuntimeError("device 'mps' requested but MPS is not available")
    return torch.device(name)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters of ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
