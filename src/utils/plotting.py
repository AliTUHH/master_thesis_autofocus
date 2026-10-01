"""Headless matplotlib figures for training and evaluation; every function saves to disk and returns the path."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected before importing pyplot)
import numpy as np  # noqa: E402
from numpy.typing import ArrayLike  # noqa: E402

__all__ = [
    "plot_loss_curves",
    "plot_true_vs_pred",
    "plot_error_vs_z01",
    "plot_example_holograms",
    "plot_radial_profiles",
]

_DPI = 150


def _finish(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    path: str | Path,
    learning_rates: Sequence[float] | None = None,
    loss_name: str = "loss",
) -> Path:
    """Train/validation loss per epoch (log scale) with an optional learning-rate axis."""
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel(loss_name)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    if learning_rates is not None and len(learning_rates) == len(train_losses):
        ax_lr = ax.twinx()
        ax_lr.plot(epochs, learning_rates, color="gray", linestyle="--", label="learning rate")
        ax_lr.set_ylabel("learning rate")
        ax_lr.set_yscale("log")
        lines = ax.get_lines() + ax_lr.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
    else:
        ax.legend(loc="upper right")
    ax.set_title("Training history")
    return _finish(fig, path)


def plot_true_vs_pred(
    true: ArrayLike,
    pred: ArrayLike,
    path: str | Path,
    quantity: str = "target",
    unit: str = "",
    title: str | None = None,
) -> Path:
    """Scatter of predictions against ground truth with the identity line."""
    true_arr = np.asarray(true, dtype=np.float64).reshape(-1)
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    label = f"{quantity} [{unit}]" if unit else quantity
    lo = float(min(true_arr.min(), pred_arr.min()))
    hi = float(max(true_arr.max(), pred_arr.max()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_arr, pred_arr, s=12, alpha=0.5, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="identity")
    ax.set_xlabel(f"true {label}")
    ax.set_ylabel(f"predicted {label}")
    ax.set_title(title or f"True vs. predicted {quantity} (n={true_arr.size})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="datalim")
    return _finish(fig, path)


def plot_error_vs_z01(
    z01_true_mm: ArrayLike,
    z01_pred_mm: ArrayLike,
    path: str | Path,
    num_bins: int = 10,
) -> Path:
    """Signed z01 error (pred - true, mm) per z01 bin as box plots with the mean ± std overlay."""
    true_arr = np.asarray(z01_true_mm, dtype=np.float64).reshape(-1)
    err = np.asarray(z01_pred_mm, dtype=np.float64).reshape(-1) - true_arr
    edges = np.linspace(true_arr.min(), true_arr.max(), num_bins + 1)
    bin_idx = np.clip(np.digitize(true_arr, edges[1:-1]), 0, num_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    groups = [err[bin_idx == b] for b in range(num_bins)]
    means = np.array([g.mean() if g.size else np.nan for g in groups])
    stds = np.array([g.std() if g.size else np.nan for g in groups])
    width = 0.6 * (edges[1] - edges[0]) if num_bins > 1 else 1.0

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(
        [g if g.size else np.array([np.nan]) for g in groups],
        positions=centers,
        widths=width,
        showfliers=False,
        manage_ticks=False,
    )
    ax.errorbar(centers, means, yerr=stds, fmt="o-", color="tab:red", capsize=3, label="mean ± std")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("true z01 [mm]")
    ax.set_ylabel("z01 error (pred − true) [mm]")
    ax.set_title("z01 error over the sampled distance range")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    return _finish(fig, path)


def plot_example_holograms(
    images: ArrayLike,
    labels: Sequence[str],
    path: str | Path,
    ncols: int = 4,
    suptitle: str | None = None,
) -> Path:
    """Grid of hologram images (``[N, H, W]`` or ``[N, 1, H, W]``) with one title string per image."""
    arr = np.asarray(images, dtype=np.float64)
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"expected images of shape [N, H, W] or [N, 1, H, W], got {arr.shape}")
    if len(labels) != arr.shape[0]:
        raise ValueError(f"got {len(labels)} labels for {arr.shape[0]} images")
    n = arr.shape[0]
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    for idx, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if idx >= n:
            continue
        img = arr[idx]
        vmin, vmax = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax if vmax > vmin else None)
        ax.set_title(labels[idx], fontsize=9)
    if suptitle:
        fig.suptitle(suptitle)
    return _finish(fig, path)


def plot_radial_profiles(
    profiles: ArrayLike,
    labels: Sequence[str],
    path: str | Path,
    title: str | None = None,
    xlabel: str = "radial bin (uniform in |xi|^2)",
) -> Path:
    """Overlay of 1-D radial profiles (``[N, n_bins]``), one curve per sample, legend from ``labels``."""
    arr = np.asarray(profiles, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 2:
        raise ValueError(f"expected profiles of shape [N, n_bins], got {arr.shape}")
    if len(labels) != arr.shape[0]:
        raise ValueError(f"got {len(labels)} labels for {arr.shape[0]} profiles")
    fig, ax = plt.subplots(figsize=(9, 5))
    for profile, label in zip(arr, labels, strict=True):
        ax.plot(profile, lw=0.9, label=label.replace("\n", ", "))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("standardised log power")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    if title:
        ax.set_title(title)
    return _finish(fig, path)
