"""Schema der Kegelstrahlgeometrie der Nahfeldholographie (Abbildung fig:grundlagen:kegelstrahl).

Zeigt Fokus (virtuelle Punktquelle), Objektebene bei z01, Detektorebene bei z02,
den Kegelstrahl mit Vergrößerung M = z02/z01 sowie das äquivalente
Parallelstrahl-Bild nach dem Fresnel-Skalierungstheorem (effektive Pixelgröße
dx/M, effektive Distanz z12/M). Rein schematisch, nicht maßstäblich.

Ausgabe: thesis/figures/grundlagen/kegelstrahl_geometrie.pdf
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from thesis_style import OI, save, use_thesis_style

WHITE_BOX = dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85)


def dimension(ax, x0, x1, y, label, above=True):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="<|-|>", mutation_scale=8, lw=0.8, color="k"))
    dy = 0.1 if above else -0.1
    ax.text(0.5 * (x0 + x1), y + dy, label, ha="center", va="bottom" if above else "top", fontsize=9)


def draw_cone_beam(ax):
    ax.set_xlim(-0.8, 10.6)
    ax.set_ylim(-2.7, 2.9)
    ax.set_aspect("equal")
    ax.axis("off")

    x_src, x_obj, x_det = 0.0, 2.5, 9.5
    h_det = 1.6
    h_obj = h_det * x_obj / x_det

    cone = Polygon(
        [[x_src, 0.0], [x_det, h_det], [x_det, -h_det]],
        closed=True,
        facecolor=OI["skyblue"],
        alpha=0.18,
        edgecolor="none",
    )
    ax.add_patch(cone)
    ax.plot([x_src, x_det], [0, 0], color=OI["black"], lw=0.6, ls=":")
    ax.plot([x_src, x_det], [0, h_det], color=OI["blue"], lw=0.9)
    ax.plot([x_src, x_det], [0, -h_det], color=OI["blue"], lw=0.9)

    # Fokus der Fresnel-Zonenplatte = virtuelle Punktquelle
    ax.plot(x_src, 0, "o", color=OI["vermillion"], ms=5, zorder=5)
    ax.text(x_src, 0.35, "FZP-Fokus\n(Quelle)", ha="center", va="bottom", fontsize=8)

    # Objektebene und Objekt
    ax.plot([x_obj, x_obj], [-h_obj - 0.25, h_obj + 0.25], color="0.3", lw=0.8, ls="--")
    ax.add_patch(plt.Circle((x_obj, 0.0), 0.16, facecolor=OI["orange"], edgecolor="k", lw=0.6, zorder=4))
    ax.text(x_obj, h_obj + 0.4, "Objekt", ha="center", va="bottom", fontsize=9)
    ax.text(x_obj, -h_obj - 0.4, r"$\psi = \mathrm{e}^{\mathrm{i}\tilde O}\,P$", ha="center", va="top", fontsize=9)

    # Detektor
    ax.add_patch(Rectangle((x_det, -h_det), 0.18, 2 * h_det, facecolor="0.25", edgecolor="none"))
    ax.text(x_det + 0.09, h_det + 0.15, "Detektor", ha="center", va="bottom", fontsize=9)
    ax.text(x_det + 0.09, -h_det - 0.15, r"$I = |\mathcal{D}(\psi)|^2$, Pixel $\Delta x$", ha="center", va="top", fontsize=9)

    # Abstände
    dimension(ax, x_src, x_obj, -2.35, r"$z_{01}$", above=False)
    dimension(ax, x_obj, x_det, -2.35, r"$z_{12} = z_{02} - z_{01}$", above=False)
    dimension(ax, x_src, x_det, 2.3, r"$z_{02}$", above=True)

    # Vergrößerung
    x_m = x_det - 0.6
    ax.annotate(
        "",
        xy=(x_m, h_det * x_m / x_det),
        xytext=(x_m, -h_det * x_m / x_det),
        arrowprops=dict(arrowstyle="<->", lw=0.7, color=OI["green"]),
    )
    ax.text(x_m - 0.2, 0.0, r"$M = z_{02}/z_{01}$", ha="right", va="center", fontsize=9, color=OI["green"], bbox=WHITE_BOX)


def draw_parallel_equivalent(ax):
    ax.set_xlim(-0.8, 10.6)
    ax.set_ylim(-2.7, 2.9)
    ax.set_aspect("equal")
    ax.axis("off")

    x_obj, x_det = 2.5, 9.5
    h = 1.0
    ax.add_patch(Rectangle((x_obj, -h), x_det - x_obj, 2 * h, facecolor=OI["skyblue"], alpha=0.18, edgecolor="none"))
    for y in (-h, h):
        ax.plot([x_obj - 1.8, x_det], [y, y], color=OI["blue"], lw=0.9)
    for y in (-0.6, -0.2, 0.2, 0.6):
        ax.add_patch(
            FancyArrowPatch((x_obj - 1.8, y), (x_obj - 0.45, y), arrowstyle="-|>", mutation_scale=7, lw=0.6, color=OI["blue"])
        )
    ax.text(x_obj - 1.1, -h - 0.15, "ebene Welle", ha="center", va="top", fontsize=8)

    ax.add_patch(plt.Circle((x_obj, 0.0), 0.16, facecolor=OI["orange"], edgecolor="k", lw=0.6, zorder=4))
    ax.text(x_obj, h + 0.15, "Objekt", ha="center", va="bottom", fontsize=9)
    ax.add_patch(Rectangle((x_det, -h), 0.18, 2 * h, facecolor="0.25", edgecolor="none"))
    ax.text(x_det + 0.09, h + 0.15, "Detektor", ha="center", va="bottom", fontsize=9)
    ax.text(x_det + 0.09, -h - 0.15, r"Pixel $\Delta x_\mathrm{eff} = \Delta x / M$", ha="center", va="top", fontsize=9)

    dimension(ax, x_obj, x_det, -2.35, r"$z_\mathrm{eff} = z_{12}/M$", above=False)
    ax.text(
        0.5 * (x_obj + x_det),
        2.3,
        r"$\mathrm{Fr} = \Delta x_\mathrm{eff}^2 / (\lambda\, z_\mathrm{eff})$",
        ha="center",
        va="center",
        fontsize=9,
    )


def main() -> None:
    use_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    draw_cone_beam(axes[0])
    draw_parallel_equivalent(axes[1])
    axes[0].set_title("(a) Kegelstrahl", loc="left", fontsize=10, fontweight="bold")
    axes[1].set_title("(b) äquivalenter Parallelstrahl", loc="left", fontsize=10, fontweight="bold")
    save(fig, "grundlagen", "kegelstrahl_geometrie")


if __name__ == "__main__":
    main()
