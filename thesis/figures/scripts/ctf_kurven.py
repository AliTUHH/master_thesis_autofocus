"""Kontrastübertragungsfunktion schwacher Objekte (Abbildung fig:grundlagen:ctf).

Phasenkontrast  ~ sin(pi u / Fr),  Absorptionskontrast ~ cos(pi u / Fr),
mit u = xi^2 + eta^2 in (Zyklen/Pixel)^2. Nullstellen des Phasenkontrasts bei
u = n Fr, des Absorptionskontrasts bei u = (n + 1/2) Fr. Gezeigt für zwei
Fresnel-Zahlen der P05-Standardgeometrie (11 keV, z02 = 20 m, 6.5 um Pixel):
z01 = 100 mm (Fr = 9.42e-5) und z01 = 250 mm (Fr = 2.37e-4).

Ausgabe: thesis/figures/grundlagen/ctf_kurven.pdf
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from thesis_style import OI, P05, fresnel_number, save, use_thesis_style


def sci(value: float, digits: int = 2) -> str:
    """Zahl als LaTeX-Mantisse mit Zehnerpotenz, z.B. 9.42\\cdot10^{-5}."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{mantissa}\cdot10^{{{int(exponent)}}}"


def main() -> None:
    use_thesis_style()
    z01_values = (100.0, 250.0)
    colors = (OI["blue"], OI["vermillion"])

    fig, axes = plt.subplots(2, 1, figsize=(5.9, 3.9), sharex=True)
    u_max = 1.0e-3  # (Zyklen/Pixel)^2, entspricht |xi| <= 0.032
    u = np.linspace(0.0, u_max, 4000)

    for z01, color in zip(z01_values, colors):
        fr = float(fresnel_number(P05["energy_keV"], z01, P05["z02_mm"], P05["px_mm"]))
        label = rf"$z_{{01}} = {z01:.0f}\,$mm, $\mathrm{{Fr}} = {sci(fr)}$"

        axes[0].plot(u, np.sin(np.pi * u / fr), color=color, label=label)
        axes[1].plot(u, np.cos(np.pi * u / fr), color=color, label=label)

        n = np.arange(1, int(u_max / fr) + 1)
        axes[0].plot(n * fr, np.zeros_like(n, dtype=float), "o", color=color, ms=3.5, mfc="white", zorder=5)
        m = np.arange(0, int(u_max / fr - 0.5) + 1)
        axes[1].plot((m + 0.5) * fr, np.zeros_like(m, dtype=float), "o", color=color, ms=3.5, mfc="white", zorder=5)

    axes[0].axhline(0, color="k", lw=0.5)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel(r"Phasenkontrast $\sin(\pi u/\mathrm{Fr})$")
    axes[1].set_ylabel(r"Absorption $\cos(\pi u/\mathrm{Fr})$")
    axes[1].set_xlabel(r"$u = \xi^2 + \eta^2$ in (Zyklen/Pixel)$^2$")
    axes[0].set_ylim(-1.15, 1.15)
    axes[1].set_ylim(-1.15, 1.15)
    axes[1].set_xlim(0, u_max)
    axes[0].legend(loc="upper right", ncol=1, fontsize=8, framealpha=0.9, frameon=True)
    axes[0].text(0.01, 0.95, r"Nullstellen bei $u = n\,\mathrm{Fr}$", transform=axes[0].transAxes, va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    axes[1].text(
        0.01, 0.95, r"Nullstellen bei $u = (n+\frac{1}{2})\,\mathrm{Fr}$", transform=axes[1].transAxes, va="top", fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5)
    )

    # Sekundärachse: |xi| in Zyklen/Pixel
    sec = axes[0].secondary_xaxis("top", functions=(np.sqrt, np.square))
    sec.set_xlabel(r"$|\xi|$ in Zyklen/Pixel")
    sec.set_xticks([0.005, 0.01, 0.015, 0.02, 0.025, 0.03])

    save(fig, "grundlagen", "ctf_kurven")


if __name__ == "__main__":
    main()
