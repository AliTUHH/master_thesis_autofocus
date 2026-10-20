"""Stage 2 figures for the lab notebook (``docs/laborbuch/figures``).

* ``stage2-phantoms.png``: four random phantoms of each class (phase in rad).
* ``stage2-samples.png``: dataset samples: phase, noise-free and noisy hologram, labels.
* ``stage2-labels.png``: label and metadata distributions of a generated split.
* ``stage2-noise.png``: one sample at several photon counts; defocus signal versus noise.
* ``stage2-binning.png``: what detector binning costs and saves: array size and run time,
  error of the direct coarse simulation, Cramer-Rao bound of ``z01``.
* ``stage2-measured-noise.png`` (only with ``--holograms``): noise level of the measured
  holograms of the Zenodo record of [1], used to calibrate the photon counts.
* ``stage2-metrics.json``: the numbers quoted in the lab notebook.

Usage::

    python scripts/make_stage2_figures.py --holograms DIR   # about 4 min, needs about 8 GB RAM
    python scripts/make_stage2_figures.py --quick --out /tmp/figures   # smoke test (CI)

``DIR`` is ``Examples/data/holograms`` of the Zenodo record 10.5281/zenodo.8349365.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from holofocus.dataset import (  # noqa: E402
    DatasetConfig,
    HologramGenerator,
    read_labels,
    target_values,
    write_split,
)
from holofocus.forward import object_from_phase, simulate_hologram  # noqa: E402
from holofocus.noise import add_detector_noise  # noqa: E402
from holofocus.phantoms import PHANTOM_TYPES, random_phantom  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "laborbuch" / "figures"
# White-noise level of the measured P05 holograms per 6.5 um pixel, see stage2-measured-noise.
PHOTONS_NATIVE = 2500.0
CRB_STEP = 1e-4  # m, central difference for dI/dz01
PHASE_SHIFT = 2.0  # rad, the largest phase shift of the phantoms in [2] Fig. 5
NM_STOP = 100e-6  # m, simplex length at which Nelder-Mead stops in [2] Table 1


@dataclass
class Settings:
    config: str
    label_samples: int
    fine_binning: int
    binnings: list[int]
    timing_samples: int
    dpi: int


FULL = Settings("configs/dataset/cpu-4x.yaml", 250, 1, [1, 2, 4, 8, 16], 3, 120)
QUICK = Settings("configs/dataset/ci.yaml", 24, 8, [8, 16, 32], 1, 60)


def bin_mean(a: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return a
    h, w = a.shape
    return a.reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def simulate(config: DatasetConfig, phase: np.ndarray, z01: float) -> np.ndarray:
    """Noise-free hologram of ``phase`` (on the detector grid of ``config``) at ``z01``."""
    obj = object_from_phase(torch.from_numpy(phase).float(), config.phantoms.delta_beta)
    fr = config.geometry.at(z01).fresnel_number
    return simulate_hologram(obj, fr, padded_shape=config.simulation_shape).double().numpy()


def single_type(config: DatasetConfig, kind: str) -> DatasetConfig:
    return replace(config, phantoms=replace(config.phantoms, types={kind: 1.0}))


def no_ticks(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def phantom_figure(config: DatasetConfig, settings: Settings, out: Path) -> dict:
    fig, axes = plt.subplots(len(PHANTOM_TYPES), 4, figsize=(8.6, 9.0), layout="constrained")
    coverage = {}
    for row, kind in enumerate(PHANTOM_TYPES):
        gen = HologramGenerator(single_type(config, kind), device="cpu")
        fractions = []
        for col in range(4):
            s = gen.sample("train", 4 * row + col)
            fractions.append(float((s.phase < 0).mean()))
            im = axes[row, col].imshow(s.phase, cmap="gray", vmin=-PHASE_SHIFT, vmax=0.0)
            axes[row, col].set_title(f"{kind}, shift {s.meta['phase_shift']:.2f} rad", fontsize=9)
            no_ticks(axes[row, col])
        coverage[kind] = float(np.mean(fractions))
    fig.colorbar(im, ax=axes, shrink=0.5, label=r"phase $\varphi$ / rad")
    g = config.geometry
    sample_pixel = g.at(g.z01_guess).effective_pixel_size * 1e9
    fig.suptitle(
        f"Random phantoms ({g.fov_shape[0]}$^2$ px of {sample_pixel:.0f} nm in the sample plane), "
        rf"single material, $\delta/\beta$ = {config.phantoms.delta_beta:g}"
    )
    fig.savefig(out / "stage2-phantoms.png", dpi=settings.dpi)
    plt.close(fig)
    return {"mean_area_fraction": coverage}


def samples_figure(config: DatasetConfig, settings: Settings, out: Path) -> dict:
    gen = HologramGenerator(config, device="cpu")
    samples = [gen.sample("train", i) for i in range(4)]
    fig, axes = plt.subplots(3, 4, figsize=(11.6, 9.4), layout="constrained")
    for col, s in enumerate(samples):
        lab, meta = s.labels, s.meta
        lo, hi = np.percentile(s.clean, [0.5, 99.5])
        kind = PHANTOM_TYPES[int(meta["phantom_type"])]
        panels = (
            (s.phase, (-PHASE_SHIFT, 0.0), f"{kind}: phase, shift {meta['phase_shift']:.2f} rad"),
            (
                s.clean,
                (lo, hi),
                f"noise-free, $\\Delta z_{{01}}$ = {lab['dz01'] * 1e3:+.2f} mm\n"
                f"Fr = {lab['fresnel_number']:.4e}",
            ),
            (s.hologram, (lo, hi), f"noisy, N = {meta['photons']:.0f} counts/px"),
        )
        for row, (image, (vmin, vmax), title) in enumerate(panels):
            axes[row, col].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            axes[row, col].set_title(title, fontsize=9)
            no_ticks(axes[row, col])
    g = config.geometry
    fig.suptitle(
        f"Dataset '{config.name}': {g.energy_kev:g} keV, z02 = {g.z02:g} m, "
        f"z01 = {g.z01_guess * 1e3:g} mm + $\\Delta z_{{01}}$, detector {g.binning}x binned "
        f"({g.fov_shape[0]}$^2$ px of {g.pixel_size * 1e6:g} um)"
    )
    fig.savefig(out / "stage2-samples.png", dpi=settings.dpi)
    plt.close(fig)
    return {"labels": [s.labels for s in samples], "meta": [s.meta for s in samples]}


def labels_figure(config: DatasetConfig, settings: Settings, out: Path) -> dict:
    n = settings.label_samples
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "val.h5"
        start = time.perf_counter()
        write_split(config, "val", path, n_samples=n, device="cpu", log=None)
        seconds = time.perf_counter() - start
        size = path.stat().st_size
        labels = read_labels(path)
    dz = labels["dz01"] * 1e3
    ratio = target_values(labels, "fresnel_ratio") * 100
    lo, hi = (v * 1e3 for v in config.sampling.dz01_range)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), layout="constrained")
    ax = axes[0, 0]
    ax.hist(dz, bins=20, range=(lo, hi), color="C0")
    ax.axhline(n / 20, color="k", ls="--", lw=1, label="uniform")
    ax.set(xlabel=r"$\Delta z_{01}$ = $z_{01} - z_{01}^\mathrm{guess}$ / mm", ylabel="samples")
    ax.legend()
    ax = axes[0, 1]
    ax.hist(labels["fresnel_number"] * 1e3, bins=20, color="C1")
    guess_fr = labels["fresnel_number_guess"][0] * 1e3
    ax.axvline(guess_fr, color="k", ls="--", lw=1, label="Fr of guess")
    ax.set(xlabel=r"Fresnel number Fr / $10^{-3}$ (true z01)", ylabel="samples")
    ax.legend()
    ax = axes[0, 2]
    ax.plot(dz, ratio, ".", ms=3, label="samples")
    g = config.geometry
    grid = np.linspace(lo, hi, 101) * 1e-3 + g.z01_guess
    exact = grid * (g.z02 - g.z01_guess) / (g.z01_guess * (g.z02 - grid)) - 1
    ax.plot((grid - g.z01_guess) * 1e3, exact * 100, "k-", lw=1, label="[1] Eq. (9)")
    ax.set(xlabel=r"$\Delta z_{01}$ / mm", ylabel=r"Fr / Fr$^\mathrm{guess}$ - 1 / %")
    ax.legend()
    ax = axes[1, 0]
    finite = np.isfinite(labels["photons"])
    if finite.any():
        ax.hist(np.log10(labels["photons"][finite]), bins=20, color="C2")
    ax.set(xlabel=r"$\log_{10}$ N (flat-field counts per pixel)", ylabel="samples")
    ax = axes[1, 1]
    counts = np.bincount(labels["phantom_type"].astype(int), minlength=len(PHANTOM_TYPES))
    ax.bar(PHANTOM_TYPES, counts, color="C3")
    ax.set(ylabel="samples")
    ax = axes[1, 2]
    ax.hist(labels["phase_shift"], bins=20, color="C4")
    ax.set(xlabel="maximal phase shift / rad", ylabel="samples")
    fig.suptitle(f"Labels and metadata of {n} generated samples ('{config.name}', split val)")
    fig.savefig(out / "stage2-labels.png", dpi=settings.dpi)
    plt.close(fig)
    return {
        "samples": n,
        "seconds_per_sample": seconds / n,
        "megabytes_per_sample": size / n / 1e6,
        "dz01_mm_range": [float(dz.min()), float(dz.max())],
        "fresnel_ratio_percent_range": [float(ratio.min()), float(ratio.max())],
        "phantom_type_counts": dict(zip(PHANTOM_TYPES, counts.tolist(), strict=True)),
    }


def noise_figure(config: DatasetConfig, settings: Settings, out: Path) -> dict:
    cfg = single_type(config, "cell")
    s = HologramGenerator(cfg, device="cpu").sample("train", 0)
    guess = s.labels["z01_guess"]
    phase = s.phase.astype(np.float32)
    clean = simulate(cfg, phase, guess)
    rows, cols = np.nonzero(phase < -1e-3)
    cy, cx = int(rows.mean()), int(cols.mean())
    n = phase.shape[0]
    half = max(int(0.75 * max(np.ptp(rows), np.ptp(cols))), n // 6)
    r0, c0 = max(cy - half, 0), max(cx - half, 0)
    crop = np.s_[r0 : min(cy + half, n), c0 : min(cx + half, n)]
    levels = [None, 1e5, 1e4, 1e3]
    lo, hi = np.percentile(clean[crop], [0.5, 99.5])
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8), layout="constrained")
    noisy = {}
    for k, photons in enumerate(levels):
        image = clean
        if photons is not None:
            image = add_detector_noise(clean, np.random.default_rng(k), "poisson", photons)
            noisy[photons] = image
        axes[0, k].imshow(image[crop], cmap="gray", vmin=lo, vmax=hi)
        title = "noise-free" if photons is None else f"N = {photons:.0e} counts/px"
        axes[0, k].set_title(title)
        no_ticks(axes[0, k])
    ax = axes[1, 0]
    x = np.arange(crop[1].start, crop[1].stop)
    for photons in (1e3, 1e4):
        ax.plot(x, noisy[photons][cy, crop[1]], lw=0.7, label=f"N = {photons:.0e}")
    ax.plot(x, clean[cy, crop[1]], "k", lw=1.2, label="noise-free")
    ax.set(xlabel="pixel", ylabel="intensity", title=f"row {cy}")
    ax.legend(fontsize=8)
    plus = simulate(cfg, phase, guess + 5e-3)
    minus = simulate(cfg, phase, guess - 5e-3)
    diff = plus - minus
    lim = np.abs(diff[crop]).max()
    ax = axes[1, 1]
    im = ax.imshow(diff[crop], cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title(r"$I(z_{01} + 5\,\mathrm{mm}) - I(z_{01} - 5\,\mathrm{mm})$")
    no_ticks(ax)
    fig.colorbar(im, ax=ax, shrink=0.8, location="bottom")
    offsets = np.linspace(-5e-3, 5e-3, 11)
    reference = simulate(cfg, phase, guess)
    change = [rms(simulate(cfg, phase, guess + d) - reference) for d in offsets]
    ax = axes[1, 2]
    ax.plot(offsets * 1e3, change, "o-", label="RMS of hologram change")
    for photons, style in ((1e3, ":"), (1e4, "--"), (1e5, "-.")):
        sigma = 1 / math.sqrt(photons)
        ax.axhline(sigma, color="gray", ls=style, label=f"noise, N = {photons:.0e}")
    ax.set(xlabel=r"$\Delta z_{01}$ / mm", ylabel="per-pixel RMS", yscale="log")
    ax.legend(fontsize=8)
    ax = axes[1, 3]
    ks = np.fft.fftfreq(n)
    k = np.sqrt(ks[:, None] ** 2 + ks[None, :] ** 2)
    edges = np.linspace(0, 0.5, 41)
    which = np.digitize(k.ravel(), edges)
    for name, image in (("signal 5 mm", diff / 2), ("noise N = 1e4", noisy[1e4] - clean)):
        power = np.abs(np.fft.fft2(image)) ** 2 / image.size
        radial = np.bincount(which, power.ravel(), len(edges) + 1)[1:-1]
        radial /= np.maximum(np.bincount(which, minlength=len(edges) + 1)[1:-1], 1)
        ax.semilogy(0.5 * (edges[1:] + edges[:-1]), radial, label=name)
    ax.set(xlabel="spatial frequency / cycles per pixel", ylabel="power", title="spectra")
    ax.legend(fontsize=8)
    fig.suptitle(
        f"Noise model (Poisson, flat-field corrected) and defocus signal; cell phantom, "
        f"z01 = {guess * 1e3:g} mm, binning {config.geometry.binning}x"
    )
    fig.savefig(out / "stage2-noise.png", dpi=settings.dpi)
    plt.close(fig)
    return {
        "rms_change_offsets_mm": (offsets * 1e3).tolist(),
        "rms_change": change,
        "noise_sigma": {f"{p:.0e}": 1 / math.sqrt(p) for p in (1e3, 1e4, 1e5)},
    }


def crb(holograms: dict[float, np.ndarray], photons: float) -> float:
    """Cramer-Rao bound of z01 for Poisson data with a known object (optimistic)."""
    derivative = (holograms[CRB_STEP] - holograms[-CRB_STEP]) / (2 * CRB_STEP)
    fisher = photons * np.sum(derivative**2 / np.clip(holograms[0.0], 1e-6, None))
    return 1 / math.sqrt(fisher)


def relative_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    return rms(estimate - reference) / rms(reference - reference.mean())


def binning_figure(settings: Settings, out: Path) -> dict:
    z01 = 0.100
    fine_b = settings.fine_binning
    needed = set(settings.binnings) | {b // 2 for b in settings.binnings if b // 2 > fine_b}
    configs = {
        b: DatasetConfig.from_dict({"geometry": {"binning": b}, "simulation": {"device": "cpu"}})
        for b in sorted(needed | {fine_b})
    }
    fine = configs[fine_b]
    offsets = (-CRB_STEP, 0.0, CRB_STEP, 5e-3)
    kinds = ("triangle", "cell", "ball")
    study = {kind: {} for kind in kinds}
    crops = {}
    for k, kind in enumerate(kinds):
        t = random_phantom(
            np.random.default_rng(100 + k), kind, fine.geometry.fov_shape, size_range=(0.45, 0.45)
        )
        phase_fine = (-PHASE_SHIFT * t).astype(np.float32)
        holo_fine = {dz: simulate(fine, phase_fine, z01 + dz) for dz in offsets}
        for b in settings.binnings:
            f = b // fine_b
            physical = {dz: bin_mean(h, f) for dz, h in holo_fine.items()}
            direct = physical
            if f > 1:
                phase_b = bin_mean(phase_fine, f)
                direct = {dz: simulate(configs[b], phase_b, z01 + dz) for dz in offsets}
            photons = PHOTONS_NATIVE * b**2
            entry = {
                "photons_per_pixel": photons,
                "crb_physical_um": crb(physical, photons) * 1e6,
                "crb_direct_um": crb(direct, photons) * 1e6,
                "contrast_rms": rms(physical[0.0] - physical[0.0].mean()),
                "direct_error": relative_error(direct[0.0], physical[0.0]),
                "signal_rms_5mm": rms(physical[5e-3] - physical[0.0]),
                "noise_sigma": 1 / math.sqrt(photons),
            }
            if f > 2:
                half = configs[b // 2]
                coarse = simulate(half, bin_mean(phase_fine, f // 2), z01)
                entry["oversample2_error"] = relative_error(bin_mean(coarse, 2), physical[0.0])
            study[kind][b] = entry
            if kind == "cell":
                crops[b] = (physical[0.0], np.nonzero(bin_mean(phase_fine, f) < -1e-3))
        del holo_fine

    timing = {}
    for b in settings.binnings:
        gen = HologramGenerator(configs[b], device="cpu")
        repeats = 1 if b == 1 else settings.timing_samples
        times = []
        for i in range(repeats):
            start = time.perf_counter()
            gen.sample("train", i)
            times.append(time.perf_counter() - start)
        n_sim = configs[b].simulation_shape[0]
        timing[b] = {
            "fov": configs[b].geometry.fov_shape[0],
            "simulation_shape": n_sim,
            "fresnel_number": configs[b].geometry.at(z01).fresnel_number,
            "seconds_per_sample": float(np.median(times)),
            "complex64_array_mb": n_sim**2 * 8 / 1e6,
        }

    shown = [settings.binnings[0], settings.binnings[len(settings.binnings) // 2]]
    shown.append(settings.binnings[-1])
    fig = plt.figure(figsize=(13.5, 8.6), layout="constrained")
    top = fig.subfigures(2, 1, height_ratios=[1, 1.05])
    axes_top = top[0].subplots(1, 3)
    axes_bottom = top[1].subplots(1, 3)
    for ax, b in zip(axes_top, shown, strict=True):
        image, (rows, cols) = crops[b]
        n = image.shape[0]
        size = max(n // 4, 8)
        cy, cx = int(rows.mean()), min(int(cols.max()), n - size // 2)
        r0 = int(np.clip(cy - size // 2, 0, n - size))
        c0 = int(np.clip(cx - size // 2, 0, n - size))
        part = image[r0 : r0 + size, c0 : c0 + size]
        lo, hi = np.percentile(part, [0.5, 99.5])
        px = configs[b].geometry.pixel_size * 1e6
        ax.imshow(part, cmap="gray", vmin=lo, vmax=hi, extent=(0, size * px, size * px, 0))
        fr = timing[b]["fresnel_number"]
        ax.set_title(f"binning {b}x: {n}$^2$ px of {px:g} um, Fr = {fr:.2e}")
        ax.set_xlabel("detector / um")
    top[0].suptitle("Cell phantom, same detector region (physical binning of the fine hologram)")
    bs = np.array(settings.binnings)
    ax = axes_bottom[0]
    ax.loglog(bs, [timing[b]["simulation_shape"] for b in bs], "o-", label="simulation array N")
    ax.loglog(bs, [timing[b]["fov"] for b in bs], "s--", label="field of view")
    ax.set(xlabel="binning b", ylabel="pixels per side")
    ax2 = ax.twinx()
    ax2.loglog(bs, [timing[b]["seconds_per_sample"] for b in bs], "^-", color="C3")
    ax2.set_ylabel("seconds per sample (CPU)", color="C3")
    ax.set_xscale("log", base=2)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("cost")
    ax = axes_bottom[1]
    for k, kind in enumerate(kinds):
        coarse = [b for b in bs if b > fine_b]
        ax.semilogy(coarse, [study[kind][b]["direct_error"] for b in coarse], "x--", color=f"C{k}")
        over = [b for b in bs if "oversample2_error" in study[kind][b]]
        ax.semilogy(over, [study[kind][b]["oversample2_error"] for b in over], "d:", color=f"C{k}")
        ax.plot([], [], "s", color=f"C{k}", label=kind)
    ax.plot([], [], "kx--", label="direct on binned grid")
    ax.plot([], [], "kd:", label="2x finer grid, then binned")
    ax.set_xscale("log", base=2)
    ax.set(xlabel="binning b", ylabel="relative RMS error", title="coarse vs physical binning")
    ax.legend(fontsize=8)
    ax = axes_bottom[2]
    for k, kind in enumerate(kinds):
        ax.loglog(bs, [study[kind][b]["crb_physical_um"] for b in bs], "o-", color=f"C{k}")
        ax.loglog(bs, [study[kind][b]["crb_direct_um"] for b in bs], "x--", color=f"C{k}")
        ax.plot([], [], "s", color=f"C{k}", label=kind)
    ax.plot([], [], "ko-", label="physical binning")
    ax.plot([], [], "kx--", label="direct on binned grid")
    ax.axhline(NM_STOP * 1e6, color="gray", ls="-.", label="[2] Nelder-Mead stop")
    ax.set_xscale("log", base=2)
    ax.set(
        xlabel="binning b",
        ylabel=r"CRB of $z_{01}$ / um (known object)",
        title=f"information, {PHOTONS_NATIVE:g} counts per native pixel",
    )
    ax.legend(fontsize=8)
    top[1].suptitle(
        f"Detector binning at 11 keV, z01 = 100 mm, z02 = 20 m; reference grid {fine_b}x"
    )
    fig.savefig(out / "stage2-binning.png", dpi=settings.dpi)
    plt.close(fig)
    return {
        "reference_binning": fine_b,
        "photons_native": PHOTONS_NATIVE,
        "phase_shift": PHASE_SHIFT,
        "timing": timing,
        "study": study,
    }


def noise_sigma(image: np.ndarray) -> float:
    """White-noise standard deviation from the second difference along rows (robust MAD)."""
    d = image[:, 2:] - 2 * image[:, 1:-1] + image[:, :-2]
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(6))


def measured_noise_figure(directory: Path, settings: Settings, out: Path) -> dict:
    from PIL import Image

    corners = {"top left": np.s_[:256, :256], "top right": np.s_[:256, -256:]}
    corners["bottom right"] = np.s_[-256:, -256:]
    table = {}
    images = {}
    for path in sorted(directory.glob("*.tif*")):
        image = np.asarray(Image.open(path), dtype=np.float64)
        images[path.stem] = image
        rows = {}
        for name, region in corners.items():
            part = image[region]
            s1, s4 = noise_sigma(part), noise_sigma(bin_mean(part, 4))
            rows[name] = {
                "mean": float(part.mean()),
                "sigma_native": s1,
                "sigma_binned4": s4,
                "photons_native": float(part.mean() / s1**2),
                "photons_binned4_if_white": float(16 * part.mean() / s1**2),
                "photons_binned4_measured": float(part.mean() / s4**2),
            }
        table[path.stem] = rows
    if "spider_hair" not in images:
        return {"holograms": table}
    part = images["spider_hair"][corners["top left"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    lo, hi = np.percentile(part, [0.5, 99.5])
    axes[0].imshow(part, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("spider hair, top-left corner, 256$^2$ native px")
    axes[1].imshow(bin_mean(part, 4), cmap="gray", vmin=lo, vmax=hi)
    axes[1].set_title("same region, binned 4x (64$^2$ px)")
    for ax in axes[:2]:
        no_ticks(ax)
    ks = np.fft.fftfreq(part.shape[0])
    k = np.sqrt(ks[:, None] ** 2 + ks[None, :] ** 2)
    edges = np.linspace(0, 0.5, 33)
    which = np.digitize(k.ravel(), edges)
    power = np.abs(np.fft.fft2(part - part.mean())) ** 2 / part.size
    radial = np.bincount(which, power.ravel(), len(edges) + 1)[1:-1]
    radial /= np.maximum(np.bincount(which, minlength=len(edges) + 1)[1:-1], 1)
    s1 = table["spider_hair"]["top left"]["sigma_native"]
    axes[2].semilogy(0.5 * (edges[1:] + edges[:-1]), radial, "o-", ms=3, label="measured")
    axes[2].axhline(s1**2, color="k", ls="--", label=rf"white noise $\sigma^2$ ({s1:.3f}$^2$)")
    axes[2].axvline(0.125, color="gray", ls=":", label="Nyquist after 4x binning")
    axes[2].set(xlabel="spatial frequency / cycles per native pixel", ylabel="power")
    axes[2].legend(fontsize=8)
    axes[2].set_title("radially averaged power spectrum")
    fig.suptitle("Measured flat-field corrected hologram of [1] (Zenodo), background region")
    fig.savefig(out / "stage2-measured-noise.png", dpi=settings.dpi)
    plt.close(fig)
    return {"holograms": table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="16x binned config, small study")
    parser.add_argument("--holograms", type=Path, help="measured holograms of the [1] record")
    args = parser.parse_args()
    settings = QUICK if args.quick else FULL
    config = DatasetConfig.from_yaml(REPO / settings.config)
    config = replace(config, simulation=replace(config.simulation, device="cpu"))
    args.out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"settings": asdict(settings), "quick": args.quick}
    metrics["config"] = {
        "name": config.name,
        "fov_shape": config.geometry.fov_shape,
        "simulation_shape": config.simulation_shape,
        "pixel_size": config.geometry.pixel_size,
        "fresnel_number_guess": config.geometry.at(config.geometry.z01_guess).fresnel_number,
        "fresnel_number_range": [config.geometry.at(z).fresnel_number for z in config.z01_bounds],
    }
    steps = [
        ("phantoms", lambda: phantom_figure(config, settings, args.out)),
        ("samples", lambda: samples_figure(config, settings, args.out)),
        ("labels", lambda: labels_figure(config, settings, args.out)),
        ("noise", lambda: noise_figure(config, settings, args.out)),
        ("binning", lambda: binning_figure(settings, args.out)),
    ]
    if args.holograms is not None:
        steps.append(
            ("measured_noise", lambda: measured_noise_figure(args.holograms, settings, args.out))
        )
    for name, fn in steps:
        start = time.perf_counter()
        metrics[name] = fn()
        metrics[name]["seconds"] = time.perf_counter() - start
        print(f"{name}: {metrics[name]['seconds']:.1f} s")
    with open(args.out / "stage2-metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
