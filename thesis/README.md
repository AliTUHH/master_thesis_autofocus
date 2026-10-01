# thesis/ – LaTeX-Skelett der Masterarbeit

Kompilierbares Gerüst für die Masterarbeit *Learning-based Autofocus for Holography* (TUHH, Institut für Biomedizinische Bildgebung). Es ist so gebaut, dass dieselbe Quelle lokal (tectonic oder TeX Live/latexmk) und auf Overleaf (pdfLaTeX) ohne Änderung kompiliert.

## Struktur

| Pfad | Inhalt |
| --- | --- |
| `main.tex` | Präambel (Pakete, Notation, Metadaten), Sprachschalter, Kapitel-Includes |
| `frontmatter/titlepage.tex` | Titelseite mit Platzhaltern (TUHH, Institut, Prüfer, Betreuer) |
| `frontmatter/declaration.tex` | Eidesstattliche Erklärung (Platzhaltertext – offiziellen TUHH-Wortlaut einsetzen) |
| `frontmatter/abstract.tex` | Kurzfassung (DE) und Abstract (EN) |
| `frontmatter/acronyms.tex` | Abkürzungen für das Paket `acro` (`\ac{nfh}`, `\acs{cnn}`, …) |
| `chapters/01_einleitung.tex` … `07_fazit.tex`, `anhang.tex` | Kapitel mit Kommentar-Stichpunkten, was hineingehört, plus je Beispiele für Zitat, Formel, Abbildung, Tabelle |
| `references.bib` | Literaturdatenbank (biblatex/biber); Citekey-Schema `autorJahrKurzwort` |
| `figures/<kapitel>/` | Abbildungen + `.tex`-Snippets, erzeugt von `tools/export_figures.py` (siehe `figures/README.md`) |
| `thesis.mplstyle` | Matplotlib-Stil für alle Thesis-Abbildungen |
| `.latexmkrc` | Build-Konfiguration für latexmk (pdflatex + biber) |

LaTeX-Build-Artefakte (`thesis/*.pdf`, `*.aux`, `*.bbl`, …) sind in der Haupt-`.gitignore` des Repos eingetragen; `thesis/figures/**/*.pdf` bleibt versioniert.

## Kompilieren

### Variante A: tectonic (eine Binary, lädt Pakete bei Bedarf)

```bash
# Installation (Linux/macOS): https://tectonic-typesetting.github.io
curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
mv tectonic ~/.local/bin/

cd thesis
tectonic -X compile main.tex        # erzeugt main.pdf
```

tectonic ruft `biber` extern auf – `biber` muss installiert sein **und zur biblatex-Version des tectonic-Bundles passen** (Kompatibilitätsmatrix in der biblatex-Doku). Getestet am 2026-10-01: tectonic 0.17.0 (Bundle mit TeX Live 2022, biblatex 3.17) + biber 2.17 → 22 Seiten, alle Zitate aufgelöst. Mit biber 2.19 (Ubuntu 24.04, TeX Live 2023) meldet biber eine Versionsinkompatibilität; dann Variante B nutzen oder das passende biber-Binary von SourceForge (`biblatex-biber/2.17/binaries/Linux`) in den `PATH` legen.

### Variante B: TeX Live + latexmk (empfohlen, identisch zu Overleaf)

```bash
# Ubuntu/Debian (ca. 1 GB):
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-bibtex-extra \
     texlive-lang-german texlive-science texlive-fonts-recommended biber latexmk

cd thesis
latexmk            # liest .latexmkrc: pdflatex + biber, so oft wie nötig
latexmk -pvc       # Dauerbetrieb: baut bei jedem Speichern neu
latexmk -C         # Build-Artefakte löschen
```

Getestet am 2026-10-01 mit TeX Live 2023 (Ubuntu 24.04), pdfTeX 1.40.25, biber 2.19 → 22 Seiten, 0 unaufgelöste Zitate, Exit-Code 0.

### Variante C: Overleaf

1. Repo-Ordner `thesis/` als ZIP packen (nur Quellen, keine Build-Artefakte) und in Overleaf unter *New Project → Upload Project* hochladen. Alternativ bei Overleaf-Premium: *New Project → Import from GitHub* und das Repo wählen – Overleaf nimmt dann den Repo-Root; setze in *Menu → Main document* die Datei `thesis/main.tex`.
2. *Menu → Compiler*: **pdfLaTeX** (Standard), *Main document*: `main.tex`. Overleaf nutzt latexmk und biber automatisch; die `.latexmkrc` wird gelesen.
3. Free-Plan: Compile-Timeout 10 s (Stand 2026-10-01, Overleaf-Doku). Das Skelett kompiliert in wenigen Sekunden; bei vielen großen PNG-Abbildungen auf PDF-Grafiken umstellen oder `\includeonly` für einzelne Kapitel nutzen.
4. GitHub-Sync und Zotero-Integration sind in Overleaf **Premium-Funktionen** (Student-Plan). Im Free-Plan: `.bib` und Abbildungen manuell hochladen bzw. die lokale Variante B als Hauptweg nutzen und Overleaf nur zum Korrekturlesen mit Betreuern.

## Sprache umschalten

In `main.tex`, Abschnitt 0:

```latex
\newcommand{\thesislanguage}{german}   % oder: english
```

Der Schalter steuert babel (Hauptsprache zuletzt), siunitx-Locale (Dezimalkomma/-punkt), cleveref-Namen, PDF-Metadaten und alle Textbausteine der Form `\lang{Deutsch}{English}`. Kurzfassung und Abstract erscheinen immer in beiden Sprachen (Hauptsprache zuerst). Für reine Textpassagen eines Kapitels brauchst du `\lang` nicht – schreibe direkt in der gewählten Sprache; `\lang` ist nur für Überschriften/Bausteine gedacht, die im Skelett beide Sprachen abdecken sollen.

## Literatur: Zotero → `references.bib`

* Zotero-Sammlung „Masterarbeit Autofokus“ anlegen, Paper per Zotero Connector (Browser) sammeln.
* Better BibTeX installieren; Citekey-Formel in den BBT-Einstellungen: `auth.lower + year + shorttitle(1,0).lower` (ergibt `dora2025model`). Für sprechende Kurzwörter (`dora2025autofocus`) den Key pinnen: im Zotero-Feld *Extra* die Zeile `Citation Key: dora2025autofocus` eintragen.
* Rechtsklick auf die Sammlung → *Export Collection…* → Format **Better BibLaTeX**, Haken **Keep updated**, Ziel: `<Repo>/thesis/references.bib`. Ab dann aktualisiert Zotero die Datei bei jeder Änderung automatisch.
* Die bereits vorhandenen 10 Einträge (6 aus der Aufgabenstellung + Software) in Zotero importieren (*File → Import* der aktuellen `references.bib`), damit der Auto-Export sie nicht überschreibt. Die Quellen aus `docs/02_literatur.md` kommen auf demselben Weg hinzu.
* Zitierstil: `biblatex` mit `style=ieee` (numerisch, Reihenfolge nach erstem Zitat). Umstellen auf Autor-Jahr: in `main.tex` `style=authoryear` setzen.

## Konventionen

* Notation (Makros in `main.tex`): `\Fr`, `\zOne`, `\zTwo`, `\magn`, `\wl`, `\dx`, `\prop{\Fr}` – konsistent mit HoloWizard/Dora et al. 2025.
* Einheiten nur mit `siunitx`: `\qty{6.5}{\micro\metre}`, `\unit{\milli\metre}`, Tabellenspalten als `S[table-format=1.2]`.
* Abkürzungen nur über `acro`: erste Nennung `\ac{nfh}`, in Tabellen/Überschriften `\acs{nfh}`.
* Labels: `ch:`, `sec:<kapitel>:<name>`, `fig:<kapitel>:<name>`, `tab:<kapitel>:<name>`, `eq:<kapitel>:<name>`; Verweise mit `\cref`.
* Offene Stellen mit `\todo{...}` markieren (rot im PDF); vor Abgabe `grep -rn "\\\\todo" thesis/` muss leer sein.
* Jede Abbildung hat eine reproduzierende Quelle (Run-Name + Commit-SHA) – steht im Snippet-Kommentar aus `tools/export_figures.py`.
