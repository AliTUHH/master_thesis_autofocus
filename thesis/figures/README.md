# thesis/figures/ – Abbildungen der Thesis

## Konvention

* Ein Unterordner pro Kapitel: `einleitung/`, `grundlagen/`, `stand/`, `methoden/`, `experimente/`, `diskussion/`, `anhang/`.
* Pro Abbildung zwei Dateien mit gleichem Stamm: Grafik (`<name>.pdf`, Fallback `.png`) und Snippet `<name>.tex` mit `figure`-Umgebung, Caption und Label `fig:<kapitel>:<name>`.
* Einbinden im Kapitel ausschließlich über `\input{figures/<kapitel>/<name>}` – so bleibt Caption/Label an einem Ort.
* **Jede Abbildung hat eine reproduzierende Quelle**: Run-Name und Commit-SHA stehen als Kommentar im Kopf des Snippets (automatisch gesetzt), zusätzlich im Logbuch `docs/notizen/experimente.md`. Ohne Quelle keine Abbildung in der Thesis.
* Grafiken werden mit `thesis/thesis.mplstyle` erzeugt (Schriftgrößen für 11-pt-Text, farbenblind-sichere Okabe-Ito-Palette, PDF-Export). Vektor (PDF) vor Raster (PNG); Bilder/Hologramme als PNG mit ≥ 300 dpi.
* Dateinamen: Kleinbuchstaben, Unterstriche, sprechend, keine Leerzeichen, keine Run-Namen im Dateinamen (der steht im Kommentar). Das Export-Skript behält den Dateistamm der Pipeline bei – aus `src.train`: `loss_curves`, `test_scatter_target`, `test_scatter_z01_mm`, `test_error_vs_z01`, `test_examples`; aus `src.evaluate` (Ordner `runs/<run>/eval_<split>/`): `scatter_target`, `scatter_z01_mm`, `error_vs_z01`, `examples`. Eigene Abbildungen (Skripte) sprechend benennen (`ablation_targets`, `fresnelzahl`).

## Export aus einem Trainingslauf

```bash
# verfügbare Figuren eines Runs anzeigen
python tools/export_figures.py --run runs/2026-10-15_cnn-baseline_s0 --list

# ausgewählte Figuren nach thesis/figures/experimente/ kopieren + Snippets schreiben
python tools/export_figures.py --run runs/2026-10-15_cnn-baseline_s0 \
    --chapter experimente --names loss_curves test_scatter_z01_mm

# alle Test-Plots eines Runs in den Anhang, Breite anpassen
python tools/export_figures.py --run runs/2026-10-15_cnn-baseline_s0 \
    --chapter anhang --names "test_*" --width "0.7\textwidth"

# Plots einer einzelnen Evaluation (z. B. OOD-Testset), falls mehrere eval_*-Ordner existieren
python tools/export_figures.py --run runs/2026-10-15_cnn-baseline_s0/eval_ood \
    --chapter anhang --names "*"
```

Bestehende `.tex`-Snippets werden **nicht** überschrieben (die Caption darin ist Handarbeit); nur die Grafikdateien werden aktualisiert. Mit `--force` wird auch das Snippet neu geschrieben.

## Beispiel

`grundlagen/fresnelzahl.{pdf,tex}` zeigt die Konvention: Beispielabbildung im `thesis.mplstyle`-Stil, Snippet mit Quellkommentar, bilinguale Caption über `\lang{..}{..}`, Label `fig:grundlagen:fresnelzahl`. Vor der Abgabe durch eine echte Abbildung ersetzen oder löschen.
