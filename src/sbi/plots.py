"""Headless figures for NPE training and posterior evaluation (Agg backend, every function saves and returns the path)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import ArrayLike  # noqa: E402

__all__ = [
    "plot_npe_loss_curves",
    "plot_posterior_examples",
    "plot_coverage_curve",
    "plot_sbc_histogram",
    "plot_uncertainty_vs_error",
    "plot_point_estimate_scatter",
    "plot_interval_plot",
]

_DPI = 150
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 renamed trapz


def _finish(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_npe_loss_curves(
    train_loss: Sequence[float] | Sequence[Sequence[float]],
    val_loss: Sequence[float] | Sequence[Sequence[float]],
    path: str | Path,
    best_epoch: int | Sequence[int | None] | None = None,
) -> Path:
    """Negative log-probability per epoch (linear scale: NPE losses may be negative).

    Accepts one curve pair or a list of curve pairs (ensemble members, drawn in matching colours).
    """
    train_curves = [list(c) for c in train_loss] if train_loss and isinstance(train_loss[0], (list, tuple)) else [list(train_loss)]
    val_curves = [list(c) for c in val_loss] if val_loss and isinstance(val_loss[0], (list, tuple)) else [list(val_loss)]
    best_epochs = list(best_epoch) if isinstance(best_epoch, (list, tuple)) else [best_epoch] * len(train_curves)
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, (tr, va) in enumerate(zip(train_curves, val_curves, strict=True)):
        colour = colours[k % len(colours)]
        suffix = f" (member {k})" if len(train_curves) > 1 else ""
        ax.plot(np.arange(1, len(tr) + 1), tr, color=colour, alpha=0.8, label=f"train{suffix}")
        ax.plot(np.arange(1, len(va) + 1), va, color=colour, linestyle="--", label=f"validation{suffix}")
        if k < len(best_epochs) and best_epochs[k] is not None:
            ax.axvline(best_epochs[k], color=colour, linestyle=":", linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("−log q(θ | x)  (z-scored θ)")
    ax.set_title("NPE training history (dotted: best validation epoch)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    return _finish(fig, path)


def plot_posterior_examples(
    grid: ArrayLike,
    log_prob: ArrayLike,
    samples: ArrayLike,
    truth: ArrayLike,
    estimates: dict[str, ArrayLike],
    prior_bounds: tuple[float, float],
    path: str | Path,
    holograms: ArrayLike | None = None,
    labels: Sequence[str] | None = None,
) -> Path:
    """One column per example: hologram thumbnail (optional, top) and posterior density with sample histogram,
    true z01 (red) and point estimates (bottom)."""
    grid_arr = np.asarray(grid, dtype=np.float64)
    logp = np.asarray(log_prob, dtype=np.float64)
    smp = np.asarray(samples, dtype=np.float64)
    true = np.asarray(truth, dtype=np.float64).reshape(-1)
    n = true.size
    rows = 2 if holograms is not None else 1
    fig, axes = plt.subplots(rows, n, figsize=(3.4 * n, 3.2 * rows), squeeze=False)
    for k in range(n):
        if holograms is not None:
            img = np.asarray(holograms)[k]
            img = img[0] if img.ndim == 3 else img
            vmin, vmax = np.percentile(img, [1, 99])
            axes[0, k].imshow(img, cmap="gray", vmin=vmin, vmax=vmax if vmax > vmin else None)
            axes[0, k].axis("off")
            if labels is not None:
                axes[0, k].set_title(labels[k], fontsize=8)
        ax = axes[rows - 1, k]
        density = np.exp(logp[k] - logp[k].max())
        area = _trapezoid(density, grid_arr)
        density = density / area if area > 0 else density
        ax.hist(smp[k], bins=40, range=prior_bounds, density=True, color="tab:blue", alpha=0.35, label="samples")
        ax.plot(grid_arr, density, color="tab:blue", lw=1.2, label="q(z01 | x)")
        ax.axvline(true[k], color="red", lw=1.4, label="true z01")
        for name, style in (("mean", ":"), ("median", "--"), ("map", "-.")):
            if name in estimates:
                ax.axvline(np.asarray(estimates[name])[k], color="k", lw=0.9, linestyle=style, label=name)
        ax.set_xlim(*prior_bounds)
        ax.set_xlabel("z01 [mm]")
        if k == 0:
            ax.set_ylabel("posterior density [1/mm]")
            ax.legend(fontsize=6, loc="upper right")
        if labels is not None and holograms is None:
            ax.set_title(labels[k], fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Posterior examples (histogram: samples, line: flow density)")
    return _finish(fig, path)


def plot_coverage_curve(
    curve: dict[str, np.ndarray],
    path: str | Path,
    points: Sequence[dict[str, Any]] | None = None,
    title: str = "Calibration: empirical vs. nominal coverage of central credible intervals",
) -> Path:
    """Empirical coverage against the nominal level with the Wilson band and the diagonal of perfect calibration."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.fill_between(curve["nominal"], curve["ci_low"], curve["ci_high"], color="tab:blue", alpha=0.2, label="95 % Wilson interval")
    ax.plot(curve["nominal"], curve["empirical"], "o-", color="tab:blue", ms=4, label="empirical coverage")
    ax.plot([0, 1], [0, 1], "r--", lw=1, label="perfect calibration")
    if points:
        ax.scatter([p["level"] for p in points], [p["coverage"] for p in points], color="k", zorder=5, s=25, label="reported levels")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal credible level")
    ax.set_ylabel("empirical coverage")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    return _finish(fig, path)


def plot_sbc_histogram(ranks: ArrayLike, n_samples: int, path: str | Path, num_bins: int = 20, ks_pvalue: float | None = None) -> Path:
    """Histogram of SBC ranks with the 99 % band expected for uniform ranks (binomial)."""
    arr = np.asarray(ranks, dtype=np.float64).reshape(-1)
    n = arr.size
    edges = np.linspace(0, n_samples, num_bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    expected = n / num_bins
    from scipy import stats

    lo, hi = stats.binom.ppf([0.005, 0.995], n, 1.0 / num_bins)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhspan(lo, hi, color="gray", alpha=0.2, label="99 % band (uniform)")
    ax.axhline(expected, color="gray", lw=0.8)
    ax.bar(0.5 * (edges[:-1] + edges[1:]), counts, width=edges[1] - edges[0], color="tab:blue", alpha=0.7, edgecolor="k", lw=0.5, label="ranks")
    ax.set_xlabel(f"rank of true z01 among {n_samples} posterior samples")
    ax.set_ylabel("count")
    title = f"SBC rank histogram (n={n})"
    if ks_pvalue is not None:
        title += f", KS p = {ks_pvalue:.3f}"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    return _finish(fig, path)


def plot_uncertainty_vs_error(
    std: ArrayLike,
    abs_error: ArrayLike,
    path: str | Path,
    correlation: dict[str, float] | None = None,
    risk_curve: Sequence[dict[str, Any]] | None = None,
) -> Path:
    """Scatter of posterior std versus |error| of the posterior mean plus the risk-coverage curve."""
    s = np.asarray(std, dtype=np.float64).reshape(-1)
    e = np.asarray(abs_error, dtype=np.float64).reshape(-1)
    fig, axes = plt.subplots(1, 2 if risk_curve else 1, figsize=(12 if risk_curve else 6, 5), squeeze=False)
    ax = axes[0, 0]
    ax.scatter(s, e, s=14, alpha=0.6, edgecolors="none")
    lim = max(s.max(), e.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=0.8, label="|error| = std")
    ax.set_xlabel("posterior std [mm]")
    ax.set_ylabel("|posterior mean − true z01| [mm]")
    title = "Is the uncertainty informative?"
    if correlation:
        title += f"\nPearson r = {correlation['pearson_r']:.2f}, Spearman ρ = {correlation['spearman_rho']:.2f}"
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    if risk_curve:
        ax = axes[0, 1]
        fractions = [r["keep_fraction"] for r in risk_curve]
        ax.plot(fractions, [r["mae"] for r in risk_curve], "o-", label="MAE")
        ax.plot(fractions, [r["rmse"] for r in risk_curve], "s--", label="RMSE")
        ax.plot(fractions, [r["p95"] for r in risk_curve], "^:", label="p95")
        ax.invert_xaxis()
        ax.set_xlabel("fraction of holograms kept (most certain first)")
        ax.set_ylabel("error of the posterior mean [mm]")
        ax.set_title("Risk-coverage curve (discarding the most uncertain)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_point_estimate_scatter(
    truth: ArrayLike,
    estimates: dict[str, ArrayLike],
    path: str | Path,
    std: ArrayLike | None = None,
    prior_bounds: tuple[float, float] | None = None,
) -> Path:
    """Posterior mean (coloured by posterior std) and other point estimates against the true z01."""
    true = np.asarray(truth, dtype=np.float64).reshape(-1)
    names = [name for name in ("mean", "median", "map") if name in estimates]
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 5), squeeze=False)
    lo = min(true.min(), *(np.asarray(estimates[n]).min() for n in names))
    hi = max(true.max(), *(np.asarray(estimates[n]).max() for n in names))
    if prior_bounds is not None:
        lo, hi = min(lo, prior_bounds[0]), max(hi, prior_bounds[1])
    for ax, name in zip(axes[0], names, strict=True):
        pred = np.asarray(estimates[name], dtype=np.float64)
        if name == "mean" and std is not None:
            sc = ax.scatter(true, pred, c=np.asarray(std), cmap="viridis", s=16, alpha=0.85)
            fig.colorbar(sc, ax=ax, label="posterior std [mm]")
        else:
            ax.scatter(true, pred, s=16, alpha=0.7, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        if prior_bounds is not None:
            for b in prior_bounds:
                ax.axhline(b, color="gray", lw=0.6, linestyle=":")
                ax.axvline(b, color="gray", lw=0.6, linestyle=":")
        mae = float(np.mean(np.abs(pred - true)))
        ax.set_xlabel("true z01 [mm]")
        ax.set_ylabel(f"posterior {name} [mm]")
        ax.set_title(f"posterior {name}: MAE {mae:.2f} mm (n={true.size})", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    return _finish(fig, path)


def plot_interval_plot(
    truth: ArrayLike,
    median: ArrayLike,
    lo: ArrayLike,
    hi: ArrayLike,
    path: str | Path,
    level: float = 0.95,
    prior_bounds: tuple[float, float] | None = None,
) -> Path:
    """Observations sorted by true z01 with the posterior median and the central credible interval."""
    true = np.asarray(truth, dtype=np.float64).reshape(-1)
    order = np.argsort(true)
    med = np.asarray(median, dtype=np.float64)[order]
    low = np.asarray(lo, dtype=np.float64)[order]
    high = np.asarray(hi, dtype=np.float64)[order]
    covered = (true[order] >= low) & (true[order] <= high)
    idx = np.arange(true.size)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(idx, med, yerr=np.vstack([med - low, high - med]), fmt="o", ms=3, lw=0.8, color="tab:blue", alpha=0.7, label=f"median ± {int(level * 100)} % interval")
    ax.scatter(idx[~covered], true[order][~covered], color="red", s=18, zorder=5, label="true z01 (outside interval)")
    ax.plot(idx, true[order], color="k", lw=1, label="true z01")
    if prior_bounds is not None:
        for b in prior_bounds:
            ax.axhline(b, color="gray", lw=0.6, linestyle=":")
    ax.set_xlabel("test hologram (sorted by true z01)")
    ax.set_ylabel("z01 [mm]")
    ax.set_title(f"Posterior intervals: {covered.mean() * 100:.0f} % of true values inside the {int(level * 100)} % interval")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    return _finish(fig, path)
