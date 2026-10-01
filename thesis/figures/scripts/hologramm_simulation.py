"""Simulierte Hologramme bei zwei Fresnel-Zahlen mit Leistungsspektrum und
radialem Spektralprofil.

Erzeugt zwei Abbildungen:
  * thesis/figures/grundlagen/hologramm_spektrum.pdf          (fig:grundlagen:hologramm)
      je Fresnel-Zahl: Hologramm einer Mg-Kugel, Hologramm eines schwachen
      Phasenobjekts mit glatter Zufallstextur und dessen log-Leistungsspektrum
      |F[I/<I> - 1]|^2 mit den Kreisen |xi|^2 = n Fr (Nullstellen der CTF)
  * thesis/figures/stand_der_technik/radialprofil_spektrum.pdf (fig:stand:radialprofil)
      radial gemitteltes Leistungsspektrum des Zufallsobjekts über u = |xi|^2;
      Minima bei u = n Fr (Analogie zu Thon-Ringen in der Kryo-EM)

Vorwärtsmodell (identisch zu HoloWizard/HoloForge in Pixeleinheiten):
    psi_det = IFFT[ exp(-i pi (xi^2 + eta^2) / Fr) * FFT[ exp(i O) * P ] ],  I = |psi_det|^2
mit O = phi + i mu, phi = -k delta T, mu = k beta T, P = 1 (ebene Beleuchtung).

Objekte:
  * Kugel mit analytischer projizierter Dicke T(r) = 2 sqrt(R^2 - r^2),
    Durchmesser 9 um Magnesium bei 11 keV (delta = 2.97e-6, beta = 2.48e-8,
    xraylib; vgl. docs/01, Abschnitt 2.2): Phasenschub im Zentrum -1.5 rad,
    Intensitätsabsorption 2.5 %. Radius 60 px.
  * Reines Phasenobjekt mit zufälliger glatter Textur (gefiltertes Gauß-Rauschen,
    Phasen-Standardabweichung 0.3 rad, weiche kreisförmige Einhüllende); der
    konstante Phasenoffset ist für die Intensität irrelevant. Dient als
    Analogon zum amorphen Präparat der Kryo-EM.

Die Fresnel-Zahlen (4e-3 und 1.2e-2) liegen im Regime der um Faktor 8
gebinnten P05-Geometrie (Pixel 52 um; z01 ~ 67 mm bzw. 198 mm bei z02 = 20 m,
11 keV), damit die Abtastbedingung N_pad >= 1/Fr auf dem 1024er-Gitter erfüllt
ist (vgl. docs/01, Abschnitt 3.5). Rechenaufwand: wenige FFTs der Größe 1024^2.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from thesis_style import OI, save, use_thesis_style

N = 512               # Detektorausschnitt in Pixeln
PAD = 2               # Padding-Faktor -> Rechengitter 1024
FR_VALUES = (4.0e-3, 1.2e-2)
SPHERE_RADIUS_PX = 60
SPHERE_DIAMETER_UM = 9.0
DELTA, BETA = 2.97e-6, 2.48e-8   # Mg, 11 keV
LAMBDA_NM = 1.2398 / 11.0
PHASE_STD_RANDOM = 0.3           # rad
SEED = 0


def grids(n: int):
    y, x = np.mgrid[-n // 2 : n // 2, -n // 2 : n // 2].astype(float)
    return x, y


def sphere_exit_wave(n: int) -> np.ndarray:
    x, y = grids(n)
    r2 = x**2 + y**2
    thickness = np.zeros((n, n))
    inside = r2 < SPHERE_RADIUS_PX**2
    um_per_px = SPHERE_DIAMETER_UM / (2 * SPHERE_RADIUS_PX)
    thickness[inside] = 2.0 * np.sqrt(SPHERE_RADIUS_PX**2 - r2[inside]) * um_per_px
    k_per_um = 2.0 * np.pi / (LAMBDA_NM * 1e-3)
    phase = -k_per_um * DELTA * thickness
    absorption = k_per_um * BETA * thickness
    return np.exp(1j * (phase + 1j * absorption))


def random_phase_exit_wave(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal((n, n))
    xi = np.fft.fftfreq(n)
    rho = np.sqrt(xi[:, None] ** 2 + xi[None, :] ** 2)
    amplitude_filter = 1.0 / (1.0 + (rho / 0.03) ** 2)  # glattes, langsam abfallendes Spektrum
    field = np.fft.ifft2(np.fft.fft2(noise) * amplitude_filter).real
    x, y = grids(n)
    envelope = np.exp(-((np.sqrt(x**2 + y**2) / (0.42 * N)) ** 10))  # weiche Einhüllende im Ausschnitt
    core = envelope > 0.5
    field = field - field[core].mean()
    field *= PHASE_STD_RANDOM / field[core].std()
    phase = field * envelope
    return np.exp(1j * phase)


def fresnel_propagate(psi: np.ndarray, fr: float) -> np.ndarray:
    n = psi.shape[0]
    xi = np.fft.fftfreq(n)
    xi2 = xi[:, None] ** 2 + xi[None, :] ** 2
    kernel = np.exp(-1j * np.pi * xi2 / fr)
    return np.fft.ifft2(np.fft.fft2(psi) * kernel)


def hologram(psi: np.ndarray, fr: float) -> np.ndarray:
    n_pad = psi.shape[0]
    assert n_pad * fr >= 1.0, "Abtastbedingung N >= 1/Fr verletzt"
    intensity = np.abs(fresnel_propagate(psi, fr)) ** 2
    lo, hi = (n_pad - N) // 2, (n_pad + N) // 2
    return intensity[lo:hi, lo:hi]


def power_spectrum(intensity: np.ndarray) -> np.ndarray:
    contrast = intensity / intensity.mean() - 1.0
    spec = np.abs(np.fft.fftshift(np.fft.fft2(contrast))) ** 2
    return spec / spec.max()


def radial_profile(spec: np.ndarray, du: float):
    n = spec.shape[0]
    xi = np.fft.fftshift(np.fft.fftfreq(n))
    u = xi[:, None] ** 2 + xi[None, :] ** 2
    edges = np.arange(0.0, 0.25 + du, du)
    idx = np.digitize(u.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    sums = np.bincount(idx[valid], weights=spec.ravel()[valid], minlength=len(edges) - 1)
    counts = np.bincount(idx[valid], minlength=len(edges) - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = sums / counts
    return centers, prof


def fr_label(fr: float) -> str:
    return rf"$\mathrm{{Fr}} = {fr * 1e3:.0f}\cdot10^{{-3}}$"


def main() -> None:
    use_thesis_style()
    n_pad = N * PAD
    psi_sphere = sphere_exit_wave(n_pad)
    psi_random = random_phase_exit_wave(n_pad)

    holo_sphere = [hologram(psi_sphere, fr) for fr in FR_VALUES]
    holo_random = [hologram(psi_random, fr) for fr in FR_VALUES]
    spectra = [power_spectrum(h) for h in holo_random]

    # --- Abbildung 1: Hologramme und Spektren (2 Zeilen = 2 Fresnel-Zahlen) ----
    fig, axes = plt.subplots(2, 3, figsize=(5.9, 4.2))
    panel = iter("abcdef")
    crop = 128  # zentraler Ausschnitt des Spektrums, |xi| <= 0.25
    extent_px = (-N / 2, N / 2, -N / 2, N / 2)
    for row, fr in enumerate(FR_VALUES):
        ax = axes[row, 0]
        im = ax.imshow(holo_sphere[row], cmap="gray", vmin=0.5, vmax=1.5, extent=extent_px)
        ax.set_ylabel(fr_label(fr) + "\nPixel")
        ax = axes[row, 1]
        ax.imshow(holo_random[row], cmap="gray", vmin=0.5, vmax=1.5, extent=extent_px)
        ax = axes[row, 2]
        c = N // 2
        sub = spectra[row][c - crop : c + crop, c - crop : c + crop]
        ax.imshow(sub, norm=LogNorm(vmin=1e-8, vmax=1e-1), cmap="viridis", extent=(-0.25, 0.25, -0.25, 0.25))
        theta = np.linspace(0, 2 * np.pi, 400)
        for n_ring in range(1, 4):
            rho = np.sqrt(n_ring * fr)
            ax.plot(rho * np.cos(theta), rho * np.sin(theta), color=OI["vermillion"], lw=0.6, ls="--", alpha=0.9)
        ax.set_xticks([-0.2, 0, 0.2])
        ax.set_yticks([-0.2, 0, 0.2])
        for col in range(3):
            axes[row, col].grid(False)
            axes[row, col].text(
                0.03, 0.96, f"({next(panel)})", transform=axes[row, col].transAxes, va="top", color="w", fontweight="bold"
            )
            if row == 0:
                axes[row, col].set_xticklabels([])
    axes[0, 0].set_title("Mg-Kugel", fontsize=9)
    axes[0, 1].set_title("schwaches Zufallsobjekt", fontsize=9)
    axes[0, 2].set_title(r"Leistungsspektrum (log), $|\xi|^2 = n\,\mathrm{Fr}$", fontsize=9)
    axes[1, 0].set_xlabel("Pixel")
    axes[1, 1].set_xlabel("Pixel")
    axes[1, 2].set_xlabel(r"$\xi$ in Zyklen/Pixel")
    axes[1, 2].set_ylabel(r"$\eta$ in Zyklen/Pixel")
    fig.colorbar(im, ax=axes[:, :2].ravel().tolist(), shrink=0.6, label=r"$I/\langle I\rangle$", pad=0.01, location="bottom")
    save(fig, "grundlagen", "hologramm_spektrum")

    # --- Abbildung 2: radiale Profile über u = |xi|^2 ---------------------------
    fig2, ax = plt.subplots(figsize=(5.9, 3.0))
    colors = (OI["blue"], OI["vermillion"])
    u_max = 0.06
    for fr, spec, color in zip(FR_VALUES, spectra, colors):
        centers, prof = radial_profile(spec, du=FR_VALUES[0] / 10)
        sel = centers <= u_max
        ax.plot(centers[sel], prof[sel] / np.nanmax(prof[sel]), color=color, lw=1.1, label=fr_label(fr))
        for n_ring in range(1, int(u_max / fr) + 1):
            ax.axvline(n_ring * fr, color=color, lw=0.5, ls=":", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlim(0, u_max)
    ax.set_xlabel(r"$u = \xi^2 + \eta^2$ in (Zyklen/Pixel)$^2$")
    ax.set_ylabel("radial gemitteltes\nLeistungsspektrum (norm.)")
    ax.legend(loc="upper right", fontsize=8)
    ax.text(
        0.02,
        0.06,
        r"gepunktet: $u = n\,\mathrm{Fr}$ (Nullstellen des Phasenkontrasts)",
        transform=ax.transAxes,
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
    )
    save(fig2, "stand_der_technik", "radialprofil_spektrum")


if __name__ == "__main__":
    main()
