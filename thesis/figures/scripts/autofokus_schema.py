"""Schema: modellbasierter Autofokus (Dora et al. 2025) gegenüber lernbasierter Schätzung.

Erzeugt ``thesis/figures/stand_der_technik/autofokus_schema.pdf``.

Oben: die geschachtelte Schleife aus Nelder--Mead-Vorschlag, ASRM-Rekonstruktion
und MFE-Auswertung (9--13 Rekonstruktionen à 6--16 s laut Dora et al. 2025,
Abbruch bei Simplexlänge 0.1 mm). Unten: ein lernbasierter Schätzer liefert in
einem Vorwärtsdurchlauf eine Punktschätzung mit Unsicherheit, die entweder direkt
verwendet oder als verengtes Startintervall an die modellbasierte Verfeinerung
übergeben wird (Hybrid). Rein schematisch, keine Daten.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from thesis_style import OI, save, use_thesis_style

BOX_STYLE = "round,pad=0.02,rounding_size=0.08"


def box(ax, x, y, w, h, text, fc="white", ec="black", lw=0.8, fontsize=8):
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=BOX_STYLE, fc=fc, ec=ec, lw=lw)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, linespacing=1.25)
    return patch


def arrow(ax, p0, p1, text=None, color="black", rad=0.0, text_offset=(0, 0.12), fontsize=7.5, ls="-"):
    arr = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=9, lw=0.8, color=color,
        connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)
    if text:
        xm = (p0[0] + p1[0]) / 2 + text_offset[0]
        ym = (p0[1] + p1[1]) / 2 + text_offset[1]
        ax.text(xm, ym, text, ha="center", va="center", fontsize=fontsize, color=color,
                bbox=dict(fc="white", ec="none", pad=1.0))


def main() -> None:
    use_thesis_style()
    fig, ax = plt.subplots(figsize=(5.9, 3.6))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 6.1)
    ax.axis("off")

    x1, x2, x3, x4 = 1.1, 3.65, 6.35, 8.95
    w1, w2, w3, w4 = 1.8, 2.0, 2.2, 1.9

    # ---------------- oben: modellbasierte Schleife ----------------
    ax.text(0.15, 5.85, "(a) modellbasiert (Dora et al. 2025)", fontsize=8.5, fontweight="bold", va="center")
    y = 4.6
    box(ax, x1, y, w1, 1.0, "Hologramm $I$\nStartintervall\n$z^{\\mathrm{est}}_{01} \\pm 5\\,$mm", fc="#F2F2F2")
    box(ax, x2, y, w2, 0.85, "Nelder–Mead\nschlägt $\\hat z_{01}$ vor", fc=OI["skyblue"] + "40")
    box(ax, x3, y, w3, 0.95, "ASRM-Rekonstruktion\nMultigrid $16\\times/4\\times/4\\times$\n6–16 s", fc=OI["orange"] + "40")
    box(ax, x4, y, w4, 0.85, "MFE$(\\hat z_{01})$\nDatenterm", fc=OI["vermillion"] + "30")

    arrow(ax, (x1 + w1 / 2, y), (x2 - w2 / 2, y))
    arrow(ax, (x2 + w2 / 2, y), (x3 - w3 / 2, y), text="$\\mathrm{Fr}(\\hat z_{01})$", text_offset=(0, 0.3))
    arrow(ax, (x3 + w3 / 2, y), (x4 - w4 / 2, y), text="$\\hat\\psi_P$", text_offset=(0, 0.3))
    # Rückkopplung zum Optimierer
    arrow(ax, (x4, y - 0.43), (x2, y - 0.43), rad=-0.35, color=OI["vermillion"],
          text="9–13 Iterationen, Abbruch bei Simplexlänge 0.1 mm", text_offset=(0, -0.7))
    arrow(ax, (x4 + 0.3, y - 0.43), (x4 + 1.0, y - 0.95), color="black")
    ax.text(x4 + 1.05, y - 1.05, "$z_{01}^\\ast$", fontsize=9, ha="left", va="center")

    # ---------------- unten: lernbasiert / Hybrid ----------------
    ax.text(0.15, 2.3, "(b) lernbasiert (diese Arbeit)", fontsize=8.5, fontweight="bold", va="center")
    y2 = 1.25
    box(ax, x1, y2, w1, 0.85, "Hologramm $I$\n(oder Spektrum)", fc="#F2F2F2")
    box(ax, x2, y2, w2, 0.95, "CNN / NPE\nein Vorwärtsdurchlauf\n$\\ll 1$ s", fc=OI["green"] + "40")
    box(ax, x3, y2, w3, 0.95, "$\\hat z_{01}$ und Unsicherheit $\\hat\\sigma$\n(bzw. Posterior $q(z_{01}\\mid I)$)", fc=OI["blue"] + "30")
    box(ax, x4, y2, w4, 1.05, "Hybrid:\nmodellbasierte\nVerfeinerung in\n$\\hat z_{01} \\pm c\\,\\hat\\sigma$", fc=OI["orange"] + "40")

    arrow(ax, (x1 + w1 / 2, y2), (x2 - w2 / 2, y2))
    arrow(ax, (x2 + w2 / 2, y2), (x3 - w3 / 2, y2))
    arrow(ax, (x3 + w3 / 2, y2), (x4 - w4 / 2, y2), text="optional", text_offset=(0, 0.36), fontsize=7)
    ax.text(x3, y2 - 0.75, "direkt verwendbar, wenn $\\hat\\sigma$ klein", fontsize=7.5, ha="center", va="center")

    # Hybrid ersetzt das Startintervall der Schleife (gestrichelt, über Eckpunkte geführt)
    y_mid = 2.95
    col = OI["orange"]
    ax.add_line(Line2D([x4, x4], [y2 + 1.05 / 2 + 0.04, y_mid], color=col, lw=0.9, ls="--"))
    ax.add_line(Line2D([x4, x1], [y_mid, y_mid], color=col, lw=0.9, ls="--"))
    arrow(ax, (x1, y_mid), (x1, y - 0.5 - 0.02), color=col, ls="--")
    ax.text((x1 + x4) / 2, y_mid + 0.17, "ersetzt das Startintervall $\\pm 5\\,$mm durch $\\hat z_{01} \\pm c\\,\\hat\\sigma$",
            fontsize=7.5, color=col, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=1.0))

    save(fig, "stand_der_technik", "autofokus_schema")


if __name__ == "__main__":
    main()
