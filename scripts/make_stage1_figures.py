"""Stage 1 figures for the lab notebook (``docs/laborbuch/figures``).

* ``stage1-defocus.png``: one phantom, holograms for z01 and z01 +- 5 mm, and
  back-propagation / PGD reconstructions of the true hologram with the assumed z01.
* ``stage1-focus-curve.png``: reconstruction error and data misfit versus the assumed z01.
* ``stage1-fresnel-sensitivity.png``: relative Fresnel-number error ([2] Fig. 3), the four
  samples of [1] Table 3, and the de-focus blur width for several detector binnings.
* ``stage1-preprocessing.png``: mirroring, A0 padding and fade window of [1] Alg. 1.
* ``stage1-metrics.json``: the numbers quoted in the lab notebook.

Usage::

    python scripts/make_stage1_figures.py                 # about 3 min on 4 CPU threads
    python scripts/make_stage1_figures.py --quick --out /tmp/figures   # smoke test (CI)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from holofocus.forward import object_from_phase, simulate_hologram  # noqa: E402
from holofocus.geometry import ConeBeamGeometry, relative_fresnel_error  # noqa: E402
from holofocus.phantoms import ball, demo_object, rasterize, union_max  # noqa: E402
from holofocus.preprocessing import preprocess_hologram  # noqa: E402
from holofocus.reconstruction import backpropagate, pgd_reconstruct  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "laborbuch" / "figures"

# [2] Table 2, simulation row (11 keV, z01 = 100 mm, z02 = 20 m) on the P05 detector.
SIMULATION = ConeBeamGeometry(energy_kev=11.0, z01=0.100, z02=20.0)
# [1] Table 3
PAPER1_SAMPLES = {
    "spider hair": ConeBeamGeometry(11.0, 79.95e-3, 19.661),
    "cactus needle": ConeBeamGeometry(17.0, 285.41e-3, 19.652),
    "tooth": ConeBeamGeometry(17.0, 81.708e-3, 19.652),
    "magnesium wire": ConeBeamGeometry(11.0, 470.51e-3, 19.661),
}
PHASE_MAX = -1.0  # rad; [2] Fig. 5 uses phi in [-2, 0]
DELTA_BETA = 200.0  # [2] Sec. 3.5: mu in [0, 0.01] for phi in [-2, 0]
OFFSETS = (-5e-3, 0.0, 5e-3)


@dataclass
class Settings:
    binning: int
    fov: int
    sim_shape: int
    rec_shape: int
    iterations: int
    pgd_offsets_mm: list[float]
    bp_offsets_mm: list[float]
    dpi: int


FULL = Settings(
    binning=4,
    fov=512,
    sim_shape=2048,
    rec_shape=1024,
    iterations=300,
    pgd_offsets_mm=[float(v) for v in np.arange(-5.0, 5.01, 1.0)],
    bp_offsets_mm=[float(v) for v in np.arange(-5.0, 5.01, 0.25)],
    dpi=120,
)
QUICK = Settings(
    binning=16,
    fov=128,
    sim_shape=512,
    rec_shape=256,
    iterations=20,
    pgd_offsets_mm=[-5.0, 0.0, 5.0],
    bp_offsets_mm=[-5.0, -2.5, 0.0, 2.5, 5.0],
    dpi=60,
)


def highpass_error(estimate: torch.Tensor, truth: torch.Tensor, cutoff: float = 0.03) -> float:
    """Relative error of the spatial frequencies above ``cutoff`` cycles/pixel."""
    ky = torch.fft.fftfreq(truth.shape[-2], dtype=torch.float64)[:, None]
    kx = torch.fft.fftfreq(truth.shape[-1], dtype=torch.float64)[None, :]
    mask = (kx**2 + ky**2).sqrt() > cutoff
    diff = torch.fft.fft2((estimate - truth).double()) * mask
    return float(diff.abs().norm() / (torch.fft.fft2(truth.double()) * mask).abs().norm())


def label(dz: float) -> str:
    if dz == 0:
        return r"$\hat z_{01} = z_{01}$"
    sign = "+" if dz > 0 else "-"
    return rf"$\hat z_{{01}} = z_{{01}} {sign} {abs(dz) * 1e3:g}$ mm"


SMALL_BALLS = ((-0.30, 0.006), (-0.25, 0.009), (-0.19, 0.013))  # (x / n, radius / n)
SMALL_BALLS_Y = 0.33  # y / n of the row of small balls (below the triangle)


def figure_phantom(n: int) -> np.ndarray:
    """:func:`demo_object` plus a row of small balls (radius 3-7 px for n = 512)."""
    small = union_max(*(ball(x * n, SMALL_BALLS_Y * n, r * n) for x, r in SMALL_BALLS))
    return np.maximum(demo_object((n, n)), rasterize(small, (n, n), supersample=4))


def simulate(settings: Settings):
    geometry = SIMULATION.binned(settings.binning)
    thickness = torch.from_numpy(figure_phantom(settings.fov)).float()
    obj = object_from_phase(PHASE_MAX * thickness, DELTA_BETA)
    sim = (settings.sim_shape, settings.sim_shape)
    holograms = {
        dz: simulate_hologram(obj, geometry.defocused(dz).fresnel_number, padded_shape=sim)
        for dz in OFFSETS
    }
    return geometry, obj, holograms


def reconstruct(hologram, geometry, dz, settings: Settings):
    fr = geometry.defocused(dz).fresnel_number
    rec = (settings.rec_shape, settings.rec_shape)
    bp = torch.angle(backpropagate(hologram, fr, padded_shape=rec))
    result = pgd_reconstruct(
        hologram, fr, iterations=settings.iterations, step=1.0, momentum=0.9, padded_shape=rec
    )
    return bp, result


def defocus_figure(settings: Settings, out: Path) -> dict:
    geometry, obj, holograms = simulate(settings)
    truth = obj.real
    f = settings.fov
    zoom = (slice(int(0.66 * f), int(0.90 * f)), slice(int(0.06 * f), int(0.37 * f)))
    row = round(f / 2 + SMALL_BALLS_Y * f - 0.5)
    cols = slice(int(0.15 * f), int(0.36 * f))
    px_um = geometry.effective_pixel_size * 1e6

    recs = {dz: reconstruct(holograms[0.0], geometry, dz, settings) for dz in OFFSETS}

    fig, axes = plt.subplots(3, 4, figsize=(15, 11.2), layout="constrained")
    extent = [0, f * px_um, f * px_um, 0]
    zoom_extent = [
        zoom[1].start * px_um,
        zoom[1].stop * px_um,
        zoom[0].stop * px_um,
        zoom[0].start * px_um,
    ]

    ax = axes[0, 0]
    im = ax.imshow(truth, cmap="gray", extent=extent, vmin=PHASE_MAX, vmax=0)
    ax.add_patch(
        plt.Rectangle(
            (zoom_extent[0], zoom_extent[3]),
            zoom_extent[1] - zoom_extent[0],
            zoom_extent[2] - zoom_extent[3],
            fill=False,
            ec="tab:orange",
            lw=1.5,
        )
    )
    ax.plot([cols.start * px_um, cols.stop * px_um], [row * px_um] * 2, color="tab:cyan", lw=1)
    ax.set_title(r"Phantom $\varphi$ / rad ($\delta/\beta = 200$)")
    ax.set_xlabel("x / µm")
    ax.set_ylabel("y / µm")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    im = ax.imshow(holograms[0.0], cmap="gray", extent=extent)
    ax.set_title(rf"Hologram $I$ at $z_{{01}}$, Fr = {geometry.fresnel_number:.3e}")
    ax.set_xlabel("x / µm")
    fig.colorbar(im, ax=ax, shrink=0.8)

    diffs = {dz: holograms[dz] - holograms[0.0] for dz in (-5e-3, 5e-3)}
    vmax = max(float(d.abs().max()) for d in diffs.values())
    rel_rms = {}
    for ax, dz in zip(axes[0, 2:], (-5e-3, 5e-3), strict=True):
        d = diffs[dz]
        rel_rms[dz] = float(d.pow(2).mean().sqrt() / (holograms[0.0] - 1).pow(2).mean().sqrt())
        im = ax.imshow(d, cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=extent)
        ax.set_title(
            rf"$I(z_{{01}} {'+' if dz > 0 else '-'} 5\,$mm$) - I(z_{{01}})$"
            + f"\nrel. RMS {100 * rel_rms[dz]:.0f} % of the fringe contrast"
        )
        ax.set_xlabel("x / µm")
        fig.colorbar(im, ax=ax, shrink=0.8)

    profiles = {"truth": truth[row, cols].numpy()}
    metrics = {}
    for r, kind in enumerate(("bp", "pgd"), start=1):
        images = {dz: (recs[dz][0] if kind == "bp" else recs[dz][1].obj.real) for dz in OFFSETS}
        if kind == "bp":
            lo, hi = np.percentile(images[0.0][zoom].numpy(), [0.5, 99.5])
        else:
            lo, hi = 1.3 * PHASE_MAX, 0.0
        for ax, dz in zip(axes[r, :3], OFFSETS, strict=True):
            img = images[dz]
            im = ax.imshow(img[zoom], cmap="gray", vmin=lo, vmax=hi, extent=zoom_extent)
            err = highpass_error(img, truth)
            metrics[f"{kind}_highpass_error_{dz * 1e3:+.0f}mm"] = err
            if kind == "bp":
                fr_hat = geometry.defocused(dz).fresnel_number
                title = rf"Back-propagation, $\hat{{Fr}}$ = {fr_hat:.3e}"
            else:
                title = f"PGD, {settings.iterations} iterations"
            ax.set_title(f"{title}\n{label(dz)}")
            ax.text(
                0.03,
                0.04,
                f"high-pass error {err:.2f}",
                transform=ax.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            ax.set_xlabel("x / µm")
            if dz == OFFSETS[0]:
                ax.set_ylabel("y / µm")
            fig.colorbar(im, ax=ax, shrink=0.8)
            profiles[f"{kind} {dz * 1e3:+.0f} mm"] = img[row, cols].numpy()

    x_um = np.arange(cols.start, cols.stop) * px_um
    colors = {-5e-3: "tab:blue", 0.0: "black", 5e-3: "tab:red"}
    for ax, kind in zip(axes[1:, 3], ("bp", "pgd"), strict=True):
        if kind == "pgd":
            ax.plot(x_um, profiles["truth"], color="0.6", lw=3, label="truth")
        for dz in OFFSETS:
            ax.plot(
                x_um,
                profiles[f"{kind} {dz * 1e3:+.0f} mm"],
                color=colors[dz],
                lw=1,
                label=f"{dz * 1e3:+.0f} mm" if dz else "0 mm (true z01)",
            )
        ax.set_xlabel("x / µm (cyan line)")
        ax.set_ylabel("phase / rad")
        ax.set_title("Line profile, " + ("back-propagation" if kind == "bp" else "PGD"))
        ax.legend(fontsize=8)
    fig.suptitle(
        f"De-focus of ±5 mm in z01 ([2] simulation geometry, {settings.binning}x binned "
        f"detector: {f}² px of {geometry.pixel_size * 1e6:.0f} µm, sample pixel "
        f"{geometry.effective_pixel_size * 1e9:.0f} nm)",
        fontsize=12,
    )
    fig.savefig(out / "stage1-defocus.png", dpi=settings.dpi)
    plt.close(fig)

    metrics.update(
        {
            "geometry": geometry.as_dict(),
            "fov": f,
            "sim_shape": settings.sim_shape,
            "rec_shape": settings.rec_shape,
            "hologram_rel_rms_change_-5mm": rel_rms[-5e-3],
            "hologram_rel_rms_change_+5mm": rel_rms[5e-3],
            "fresnel_number_assumed": {
                f"{dz * 1e3:+.0f}mm": geometry.defocused(dz).fresnel_number for dz in OFFSETS
            },
            "defocus_blur_px_5mm": 1.0 / abs(geometry.fresnel_number_at(5e-3)) ** 0.5,
        }
    )
    return metrics


def focus_curve_figure(settings: Settings, out: Path) -> dict:
    geometry, obj, holograms = simulate(settings)
    hologram, truth = holograms[0.0], obj.real
    rec = (settings.rec_shape, settings.rec_shape)
    bp_err = []
    for dz_mm in settings.bp_offsets_mm:
        fr = geometry.defocused(dz_mm * 1e-3).fresnel_number
        bp_err.append(
            highpass_error(torch.angle(backpropagate(hologram, fr, padded_shape=rec)), truth)
        )
    pgd_err, pgd_loss = [], []
    for dz_mm in settings.pgd_offsets_mm:
        fr = geometry.defocused(dz_mm * 1e-3).fresnel_number
        result = pgd_reconstruct(
            hologram, fr, iterations=settings.iterations, step=1.0, momentum=0.9, padded_shape=rec
        )
        pgd_err.append(highpass_error(result.obj.real, truth))
        pgd_loss.append(result.loss[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ax1.plot(settings.bp_offsets_mm, bp_err, ".-", label="back-propagation")
    ax1.plot(settings.pgd_offsets_mm, pgd_err, "o-", label=f"PGD ({settings.iterations} it.)")
    ax1.set_xlabel(r"assumed $\hat z_{01} - z_{01}$ / mm")
    ax1.set_ylabel("high-pass phase error vs. ground truth")
    ax1.set_title("Reference-based: error above 0.03 cycles/px")
    ax1.legend()
    ax2.semilogy(settings.pgd_offsets_mm, pgd_loss, "o-", color="tab:orange")
    ax2.set_xlabel(r"assumed $\hat z_{01} - z_{01}$ / mm")
    ax2.set_ylabel("PGD data misfit per pixel (last iteration)")
    ax2.set_title(r"Reference-free: $\frac{1}{2}\| |D_{\hat{Fr}}(\psi)| - \sqrt{I} \|^2$")
    for ax in (ax1, ax2):
        ax.axvline(0, color="0.7", lw=0.8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Focus curves, {settings.binning}x binned [2] simulation geometry (noise-free)",
        fontsize=11,
    )
    fig.savefig(out / "stage1-focus-curve.png", dpi=settings.dpi)
    plt.close(fig)
    return {
        "bp_offsets_mm": settings.bp_offsets_mm,
        "bp_highpass_error": bp_err,
        "pgd_offsets_mm": settings.pgd_offsets_mm,
        "pgd_highpass_error": pgd_err,
        "pgd_final_loss": pgd_loss,
    }


def sensitivity_figure(settings: Settings, out: Path) -> dict:
    dz = np.linspace(-10e-3, 10e-3, 401)
    dz = dz[dz != 0]
    g = SIMULATION
    err01 = [relative_fresnel_error(g, dz01=v) for v in dz]
    err02 = [relative_fresnel_error(g, dz02=v) for v in dz]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    ax = axes[0]
    ax.semilogy(dz * 1e3, err01, label=r"error in $z_{01}$")
    ax.semilogy(dz * 1e3, err02, label=r"error in $z_{02}$")
    ax.axvspan(-5, 5, color="0.9", zorder=0, label=r"$\Omega_z$ = ±5 mm")
    ax.set_xlabel("distance error / mm")
    ax.set_ylabel(r"$|\hat{Fr}/Fr - 1|$")
    ax.set_title("[2] Eq. (8), simulation geometry (cf. [2] Fig. 3)")
    ax.legend()

    ax = axes[1]
    for name, geom in PAPER1_SAMPLES.items():
        ax.plot(
            dz * 1e3,
            [100 * (geom.defocused(v).fresnel_number / geom.fresnel_number - 1) for v in dz],
            label=f"{name} (z01 = {geom.z01 * 1e3:.1f} mm)",
        )
    ax.axvspan(-5, 5, color="0.9", zorder=0)
    ax.set_xlabel(r"$\Delta z_{01}$ / mm")
    ax.set_ylabel(r"$\hat{Fr}/Fr - 1$ / %")
    ax.set_title("Samples of [1] Table 3")
    ax.legend(fontsize=8)

    ax = axes[2]
    dz_pos = np.linspace(0.1e-3, 5e-3, 200)
    blur = {}
    for b in (1, 2, 4, 8, 16):
        gb = SIMULATION.binned(b)
        w = [1.0 / abs(gb.fresnel_number_at(v)) ** 0.5 for v in dz_pos]
        blur[b] = w[-1]
        ax.plot(dz_pos * 1e3, w, label=f"{b}x binning, Fr = {gb.fresnel_number:.1e}")
    ax.axhline(1, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel(r"$|\Delta z_{01}|$ / mm")
    ax.set_ylabel(r"blur width $1/\sqrt{|Fr(\Delta z_{01})|}$ / px")
    ax.set_title("De-focus blur on the detector grid")
    ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.savefig(out / "stage1-fresnel-sensitivity.png", dpi=settings.dpi)
    plt.close(fig)
    return {
        "relative_fr_error_z01_+5mm": relative_fresnel_error(g, dz01=5e-3),
        "relative_fr_error_z02_+5mm": relative_fresnel_error(g, dz02=5e-3),
        "blur_px_at_5mm_by_binning": {str(b): w for b, w in blur.items()},
        "paper1_relative_fr_change_+5mm": {
            name: geom.defocused(5e-3).fresnel_number / geom.fresnel_number - 1
            for name, geom in PAPER1_SAMPLES.items()
        },
    }


def preprocessing_figure(settings: Settings, out: Path) -> dict:
    geometry, _, holograms = simulate(settings)
    hologram = holograms[0.0]
    f = settings.fov
    big = (4 * f, 4 * f)
    pre = preprocess_hologram(hologram, big, a0=1.0)
    raw = preprocess_hologram(hologram, big, a0=1.0, fade_width=0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), layout="constrained")
    vmin, vmax = float(hologram.min()), float(hologram.max())
    axes[0].imshow(hologram, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Hologram (FOV {f}² px)")
    axes[1].imshow(raw.data, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(r"mirrored + padded with $A_0$")
    axes[2].plot(pre.window[2 * f].numpy(), color="black")
    for s in (pre.fov[1].start, pre.fov[1].stop):
        axes[2].axvline(s, color="tab:orange", lw=1.2, ls="--")
    axes[2].set_title("fade window, central row ([1] Fig. 4)")
    axes[2].set_xlabel("pixel")
    axes[2].set_box_aspect(1)
    axes[3].imshow(pre.data, cmap="gray", vmin=vmin, vmax=vmax)
    axes[3].set_title(r"$(I - A_0) W + A_0$ ([1] Alg. 1)")
    for ax in (axes[1], axes[3]):
        r, c = pre.fov
        ax.add_patch(
            plt.Rectangle((c.start, r.start), f, f, fill=False, ec="tab:orange", lw=1.0, ls="--")
        )
    fig.suptitle(
        f"Preprocessing of [1] Sec. 4.1 on a {big[0]}² array "
        f"(1/Fr = {1 / geometry.fresnel_number:.0f} px)",
        fontsize=11,
    )
    fig.savefig(out / "stage1-preprocessing.png", dpi=settings.dpi)
    plt.close(fig)
    return {"preprocessing_fov_unchanged": bool(torch.equal(pre.crop(pre.data), hologram))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="small grid, few iterations")
    args = parser.parse_args()
    settings = QUICK if args.quick else FULL
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    metrics: dict = {"settings": asdict(settings), "quick": args.quick}
    for name, fn in (
        ("sensitivity", sensitivity_figure),
        ("preprocessing", preprocessing_figure),
        ("defocus", defocus_figure),
        ("focus_curve", focus_curve_figure),
    ):
        start = time.perf_counter()
        metrics[name] = fn(settings, args.out)
        metrics[name]["seconds"] = time.perf_counter() - start
        print(f"{name}: {metrics[name]['seconds']:.1f} s")
    with open(args.out / "stage1-metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
