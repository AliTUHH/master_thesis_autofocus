"""Gemeinsame Hilfsfunktionen für die analytischen Abbildungen der Thesis.

Alle Skripte in diesem Ordner erzeugen ihre PDFs direkt in
``thesis/figures/<kapitel>/`` und verwenden ``thesis/thesis.mplstyle``.
Aufruf (aus beliebigem Verzeichnis)::

    /tmp/holo311/bin/python thesis/figures/scripts/<skript>.py

Physikalische Konstanten und die Fresnel-Zahl-Formel folgen HoloWizard
(``holowizard.forge.utils.calc_Fr``): Längen in mm, Energie in keV,
Wellenlänge in nm.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: Skripte laufen ohne Display (CI, Server)

import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
THESIS_DIR = HERE.parents[1]
FIGURES_DIR = THESIS_DIR / "figures"

# Okabe-Ito-Palette (identisch zu thesis.mplstyle, hier als benannte Farben)
OI = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "skyblue": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

# Standardgeometrie der Nano-Imaging-Endstation P05 (vgl. docs/01, Abschnitt 4.3)
P05 = {
    "energy_keV": 11.0,
    "z02_mm": 20000.0,
    "px_mm": 0.0065,
}

HC_KEV_NM = 1.2398  # h*c in keV*nm, wie in HoloWizard (lam = 1.2398 / energy)


def use_thesis_style() -> None:
    """Aktiviert thesis.mplstyle und registriert Latin Modern aus TeX Live, falls vorhanden."""
    for pattern in (
        "/usr/share/texmf/fonts/opentype/public/lm/lmroman*.otf",
        "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman*.otf",
    ):
        for path in glob.glob(pattern):
            try:
                fm.fontManager.addfont(path)
            except Exception:  # pragma: no cover - Font-Registrierung ist optional
                pass
    plt.style.use(str(THESIS_DIR / "thesis.mplstyle"))
    matplotlib.rcParams["axes.unicode_minus"] = False


def wavelength_nm(energy_keV: float) -> float:
    return HC_KEV_NM / energy_keV


def fresnel_number(energy_keV: float, z01_mm, z02_mm: float, px_mm: float):
    """Pixel-Fresnel-Zahl Fr = px^2 z01 / (lambda z02 (z02 - z01)), alle Längen in mm."""
    lam_mm = wavelength_nm(energy_keV) * 1e-6
    z01_mm = np.asarray(z01_mm, dtype=float)
    return px_mm**2 * z01_mm / (lam_mm * z02_mm * (z02_mm - z01_mm))


def dlnfr_dz01(z01_mm, z02_mm: float):
    """Logarithmische Sensitivität d ln Fr / d z01 = 1/z01 + 1/(z02 - z01) in 1/mm."""
    z01_mm = np.asarray(z01_mm, dtype=float)
    return 1.0 / z01_mm + 1.0 / (z02_mm - z01_mm)


def dlnfr_dz02(z01_mm, z02_mm: float):
    """Logarithmische Sensitivität d ln Fr / d z02 = -1/z02 - 1/(z02 - z01) in 1/mm."""
    z01_mm = np.asarray(z01_mm, dtype=float)
    return -1.0 / z02_mm - 1.0 / (z02_mm - z01_mm)


def save(fig, chapter: str, name: str) -> Path:
    out_dir = FIGURES_DIR / chapter
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.pdf"
    fig.savefig(out)
    print(f"geschrieben: {out.relative_to(THESIS_DIR.parent)}")
    return out
