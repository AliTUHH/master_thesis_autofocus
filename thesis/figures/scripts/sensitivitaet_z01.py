"""Sensitivität der Fresnel-Zahl gegenüber z01 und z02 (Abbildung fig:grundlagen:sensitivitaet).

Links: Fr(z01) für die P05-Standardgeometrie (z02 = 20 m, 6.5 um Pixel) bei
11 keV und 17 keV. Rechts: logarithmische Sensitivitäten
    d ln Fr / d z01 =  1/z01 + 1/(z02 - z01)
    d ln Fr / d z02 = -1/z02 - 1/(z02 - z01)
in Prozent pro Millimeter sowie die z01-Abweichung, die einem relativen
Fr-Fehler von 0.04 % entspricht (Zielgenauigkeit 0.1 mm bei z01 = 250 mm).

Ausgabe: thesis/figures/grundlagen/sensitivitaet_z01.pdf
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from thesis_style import OI, P05, dlnfr_dz01, dlnfr_dz02, fresnel_number, save, use_thesis_style


def main() -> None:
    use_thesis_style()
    z01 = np.linspace(20.0, 500.0, 500)
    z02 = P05["z02_mm"]
    px = P05["px_mm"]

    fig, (ax_fr, ax_sens) = plt.subplots(1, 2, figsize=(5.9, 2.9))

    for energy, color, ls in ((11.0, OI["blue"], "-"), (17.0, OI["vermillion"], "--")):
        fr = fresnel_number(energy, z01, z02, px)
        ax_fr.plot(z01, fr, color=color, ls=ls, label=rf"$E = {energy:.0f}\,$keV")
    ax_fr.set_yscale("log")
    ax_fr.set_xlabel(r"$z_{01}$ in mm")
    ax_fr.set_ylabel(r"Fresnel-Zahl $\mathrm{Fr}$")
    ax_fr.set_xlim(0, 500)
    ax_fr.text(0.02, 0.97, "(a)", transform=ax_fr.transAxes, va="top", fontweight="bold")

    # Fr-Werte der fünf Messobjekte aus Dora et al. 2025, Tab. 2 (z01 in mm, z02 in mm, E in keV)
    objects = [
        ("Spinnenhaar", 79.4, 19661.0, 11.0),
        ("Zahn", 81.0, 19661.0, 17.0),
        ("Kaktusnadel", 284.1, 19661.0, 17.0),
        ("Mg-Draht", 470.8, 19661.0, 11.0),
        ("Mg-Draht (Flusszelle)", 329.3, 19914.0, 11.0),
    ]
    for _, z, zz, e in objects:
        ax_fr.plot(z, fresnel_number(e, z, zz, px), "o", color="k", ms=3, mfc="white", zorder=5)
    ax_fr.plot([], [], "o", color="k", ms=3, mfc="white", label="Messobjekte in Dora et al. 2025")
    ax_fr.legend(loc="lower right", fontsize=7.5, handlelength=1.6)

    s01 = 100.0 * dlnfr_dz01(z01, z02)  # %/mm
    s02 = 100.0 * np.abs(dlnfr_dz02(z01, z02))
    ax_sens.plot(z01, s01, color=OI["blue"], label=r"$|\partial \ln\mathrm{Fr}/\partial z_{01}|$")
    ax_sens.plot(z01, s02, color=OI["green"], ls="--", label=r"$|\partial \ln\mathrm{Fr}/\partial z_{02}|$")
    ax_sens.set_yscale("log")
    ax_sens.set_xlabel(r"$z_{01}$ in mm")
    ax_sens.set_ylabel(r"Sensitivität in %/mm")
    ax_sens.set_xlim(0, 500)
    ax_sens.set_ylim(5e-3, 10)
    ax_sens.legend(loc="upper right", fontsize=8)
    ax_sens.text(0.02, 0.97, "(b)", transform=ax_sens.transAxes, va="top", fontweight="bold")

    # Markierung z01 = 250 mm: 0.405 %/mm
    z_ref = 250.0
    s_ref = 100.0 * float(dlnfr_dz01(z_ref, z02))
    ax_sens.plot(z_ref, s_ref, "o", color=OI["blue"], ms=4, zorder=5)
    ax_sens.annotate(
        rf"{s_ref:.2f} %/mm bei {z_ref:.0f} mm",
        xy=(z_ref, s_ref),
        xytext=(120, 2.2),
        fontsize=8,
        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"),
    )

    save(fig, "grundlagen", "sensitivitaet_z01")


if __name__ == "__main__":
    main()
