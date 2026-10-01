# 04 – Werkzeuge und Workflow der Masterarbeit

Stand: 2026-10-01. Dieses Dokument legt fest, **womit** und **wie** in dieser Arbeit gearbeitet wird, damit das Zusammenschreiben am Ende ein Zusammensetzen ist und kein Rekonstruieren. Alles, was im Repo eingerichtet werden konnte, ist eingerichtet und getestet (Abschnitt 10 listet, was du selbst anlegen musst). Fakten zu externen Diensten sind mit Datum und Quelle versehen; Unverifiziertes ist als solches markiert.

---

## 1. Überblick der Werkzeugkette

| Zweck | Werkzeug | Warum | Wo im Repo |
| --- | --- | --- | --- |
| Code, Notizen, Thesis-Quellen versionieren | **Git + GitHub** (`AliTUHH/master_thesis_autofocus`) | Ein Ort für alles Textuelle; Verlauf = Nachweis für die Thesis; kostenlos; Betreuer können mitlesen | gesamtes Repo |
| Aufgaben, Experimente, Literatur verwalten | **GitHub Issues + Milestones + Projects** | Direkt am Code; Issue-Vorlagen erzwingen Hypothese/Ergebnis/Entscheidung; Milestones = Phasen aus `docs/03_projektplan.md` | `.github/ISSUE_TEMPLATE/` |
| Qualitätssicherung | **GitHub Actions** (`pytest`, Python 3.11, CPU) | Jeder Push/PR wird getestet; verhindert kaputte Datenerzeugung vor teuren GPU-Läufen | `.github/workflows/tests.yml` |
| Python-Umgebung | **uv** (`uv venv --python 3.11`) + `requirements.txt` (alternativ conda `environment.yml`) | holowizard braucht Python 3.11 (<3.12); uv liefert exakt diese Version überall – lokal, CI, Colab | `requirements.txt`, `environment.yml` |
| Rechnen klein (Smoke-Tests) | **lokal, CPU** | sofort verfügbar; `configs/data_small.yaml` | `src/`, `configs/` |
| Rechnen mittel (GPU, Stunden) | **Google Colab** (T4, Free/Pro), Alternative **Kaggle** (30 GPU-h/Woche) | kostenlos/günstig; Notebook ist fertig | `notebooks/colab_training.ipynb` |
| Rechnen groß (Datensätze `data_p05`) | **DESY Maxwell** / **TUHH-HPC** über Betreuer | einzige realistische Option für viele große Hologramme; Daten bleiben bei DESY | `docs/notizen/fragen_an_betreuer.md` |
| Experiment-Tracking | **TensorBoard** (im Projekt) + **Logbuch** `docs/notizen/experimente.md` | offline, ohne Account, Logs liegen im Run-Ordner; Logbuch ist die zitierfähige Wahrheit | `runs/<run>/`, `docs/notizen/experimente.md` |
| Literatur sammeln & zitieren | **Zotero 7 + Better BibTeX** (Auto-Export) + **biblatex/biber** | stabile Citekeys, PDF-Verwaltung, ein `.bib` als Single Source of Truth | `thesis/references.bib` |
| Schreiben | **LaTeX-Skelett** (KOMA `scrbook`, biblatex-ieee, acro, siunitx, cleveref), lokal mit `latexmk`/`tectonic`, optional **Overleaf** zum Korrekturlesen | kompiliert geprüft; DE/EN per Schalter; identische Quelle lokal und auf Overleaf | `thesis/` |
| Abbildungen | **Matplotlib-Stil** `thesis.mplstyle` + **Export-Skript** `tools/export_figures.py` | einheitliche Schriftgrößen/Farben; jede Abbildung trägt Run + Commit als Quelle | `thesis/thesis.mplstyle`, `tools/export_figures.py`, `thesis/figures/` |
| Notizen | **Markdown im Repo** (`docs/notizen/`), optional Obsidian als Oberfläche | versioniert, durchsuchbar, Citekeys direkt übertragbar | `docs/notizen/` |

Grundprinzip: **Alles, was in die Thesis einfließt, hat eine Spur im Repo** – ein Commit, ein Run-Name, ein Citekey.

---

## 2. Code & Versionierung mit GitHub

### 2.1 Branch-Strategie

- `main` ist **immer lauffähig** (CI grün, `python -m pytest -q` besteht). Von `main` aus muss jederzeit ein Smoke-Test (`configs/data_small.yaml` → `src.train` → `src.evaluate`) funktionieren.
- Arbeit passiert auf kurzlebigen Branches: `feature/<thema>` (Code), `exp/<kurzname>` (Experiment-Serie, wenn Code-Änderungen nötig sind), `docs/<thema>` (Text/Notizen), `thesis/<kapitel>` (Thesis-Kapitel).
- **Auch allein per Pull Request mergen.** Gründe: (1) die CI läuft auf dem PR und du siehst Fehler, bevor sie auf `main` sind; (2) der PR beschreibt *warum* etwas geändert wurde – diese Begründungen brauchst du im Methoden- und Diskussionskapitel; (3) die PR-Vorlage fragt gezielt nach Auswirkungen auf Ergebnisse und Thesis-Kapitel. Squash-Merge halten die `main`-Historie lesbar.
- Betreuer-Feedback zu Code läuft als PR-Review-Kommentar (GitHub-Account der Betreuer als Collaborator mit Read/Triage reicht).

### 2.2 Commit-Konventionen

Kurzform von Conventional Commits, auf Deutsch oder Englisch – aber konsistent:

```
<typ>: <was, imperativ, max. 72 Zeichen>

<optional: warum; Bezug auf Issue #12; Run-Name, falls Ergebnis>
```

Typen: `feat` (neue Funktion), `fix`, `exp` (Experiment-Config/Ergebnisse), `data` (Datenerzeugung), `docs`, `thesis`, `ci`, `refactor`, `test`. Beispiele: `feat: HDF5-Dataset mit Lazy Loading`, `exp: Ablation Zielgröße logFr (runs 2026-11-02_*)`, `thesis: Kapitel 2.3 Fresnel-Skalierungstheorem`.

Regeln: ein Commit = eine logische Änderung; keine Checkpoints, HDF5-Dateien oder Rohdaten committen (siehe `.gitignore`); `results.json`, `eval_<split>/eval_results.json` und Figuren von Läufen, die in die Thesis sollen, **schon** committen (klein, zitierfähig). Da `runs/` komplett in `.gitignore` steht, geht das nur gezielt mit `git add -f runs/<run>/results.json runs/<run>/eval_test/eval_results.json runs/<run>/*.png` – Checkpoints (`*.pt`), `*.npz` und TensorBoard-Events bleiben auch dann ignoriert.

### 2.3 Tags für Experiment-Meilensteine

Wenn ein Ergebnis in die Thesis eingeht, den Stand taggen, damit „der Code, mit dem Tabelle 5.1 entstand“ eine Adresse hat:

```bash
git tag -a exp/v1-cnn-baseline -m "CNN-Baseline, Tabelle 5.1, runs 2026-10-15_cnn-baseline_s{0,1,2}"
git push origin exp/v1-cnn-baseline
```

Schema: `exp/v<N>-<kurzname>`. Im Logbuch und im Figur-Snippet steht zusätzlich der Commit-SHA (das Export-Skript schreibt ihn automatisch).

### 2.4 Issues, Milestones, Projects

- **Milestones = Phasen** aus `docs/03_projektplan.md` (z. B. „Phase 1 – Einarbeitung“, „Phase 2 – Datenerzeugung“, …) mit Fälligkeitsdatum. Jedes Issue gehört zu genau einem Milestone.
- **Issue-Vorlagen** (`.github/ISSUE_TEMPLATE/`):
  - *Experiment* – Hypothese, Config/Daten, erwartete Metriken, Ergebnis, Entscheidung. Ein Issue pro Experiment-Serie (mehrere Seeds/Varianten), geschlossen erst mit Logbuch-Eintrag.
  - *Aufgabe* – Ziel, Teilschritte, Definition of Done.
  - *Paper-Notiz* – Citekey, Kernaussage, Relevanz; verweist auf `docs/notizen/paper/<citekey>.md`.
- **Labels**: `experiment`, `aufgabe`, `literatur` (werden von den Vorlagen gesetzt), zusätzlich `blocker`, `frage-betreuer`, `thesis`.
- **Projects-Board** (GitHub Projects, „Board“-Layout) mit Spalten *Backlog → Diese Woche → In Arbeit → Review/Warten auf Betreuer → Fertig*. Automatisierung: neue Issues → Backlog, geschlossene → Fertig. Wochenplanung = Karten in „Diese Woche“ ziehen (max. 3–5).

### 2.5 Release und Zenodo-DOI am Ende

Zum Abgabetermin ein GitHub-Release `v1.0-thesis` erstellen (Release-Notes: Zusammenfassung, Befehle zur Reproduktion, Verweis auf Thesis). Mit der GitHub–Zenodo-Integration (zenodo.org → *GitHub* → Repo aktivieren, **vor** dem Release) erzeugt jedes Release automatisch einen DOI; der Code wird damit in der Thesis zitierbar (`@software`-Eintrag in `references.bib`, analog zu HoloWizard). Lizenz vorher klären (MIT passt zu HoloWizard; Frage an Betreuer, ob der Code öffentlich bleiben darf – steht in `fragen_an_betreuer.md`).

---

## 3. Rechnen: lokal → Colab/Kaggle → Cluster

### 3.1 Stufenmodell

| Stufe | Wofür | Grenzen | Befehl/Einstieg |
| --- | --- | --- | --- |
| **Lokal, CPU** (hier: 4 Kerne, 15 GB RAM, keine GPU) | Tests, Smoke-Runs, Debugging, kleine Datensätze (`data_small`), Plots | Training mit 256²-Hologrammen (`data_small`: 2048 px um Faktor 8 gebinnt) und einigen hundert Samples in Minuten (Smoke-Run 400/100/100, 30 Epochen: ≈ 13 min); alles Größere unpraktisch | `uv venv --python 3.11 .venv && uv pip install -r requirements.txt`, dann CLI (Abschnitt 3.4) |
| **Google Colab** | mittlere Experimente (Stunden), Ablationen mit `data_small`/mittleren Datensätzen, Hyperparameter-Suche | Free: ≤ 12 h/Session, GPU (meist T4) nicht garantiert, Idle-Timeout unveröffentlicht; Pro 9,99 $/Monat (100 Compute Units) | `notebooks/colab_training.ipynb` |
| **Kaggle Notebooks** | wie Colab, planbarer | 30 GPU-h/Woche (schwankend nach oben), 12 h/Session, P100 oder 2×T4, 9 h für TPU | gleiches Notebook importieren |
| **DESY Maxwell / TUHH-HPC** | große Datensätze (`data_p05.yaml`), Messdaten (bleiben bei DESY), lange Läufe, finale Modelle | Zugang über Betreuer (J. Dora bzw. TUHH); SLURM-Jobs statt Notebook | Frage **A** in `fragen_an_betreuer.md`; Job-Skript später unter `tools/` |

Verifizierte Fakten (Stand 2026-10-01):

- **Colab-Python-Version:** Runtime 2026.07 = Python **3.12.13** (auch 2026.04 und 2026.01 sind 3.12.x). Quelle: Colab „Runtime version FAQ“ (research.google.com/colaboratory/runtime-version-faq.html) und github.com/googlecolab/backend-info. Colab Enterprise defaultet ebenfalls auf 3.12. → `pip install holowizard` scheitert im Standard-Kernel, weil holowizard `python >=3.11,<3.12` verlangt (PyPI-Metadaten holowizard 3.0.6, geprüft per `pypi.org/pypi/holowizard/json`).
- **Workaround (im Notebook umgesetzt):** eigenes Python-3.11-venv mit `uv` im Projektordner, alle Repo-Befehle über `.venv/bin/python`. Colab bringt `uv` mit, setzt aber Umgebungsvariablen (`UV_CONSTRAINT`, `UV_BUILD_CONSTRAINT`, `UV_PRERELEASE`, `UV_SYSTEM_PYTHON=true`), die `uv`-Aufrufe stören (googlecolab/colabtools Issue #5237). Das Notebook entfernt sie und setzt `UV_PYTHON_DOWNLOADS=automatic`, damit uv ein 3.11 herunterlädt. Alternative: `condacolab` (installiert Miniconda mit Python 3.11 und startet den Kernel neu; nur `base`-Environment nutzbar). Die uv-Variante ist schneller und braucht keinen Kernel-Neustart. **Nicht in einer echten Colab-Session getestet** (hier kein Browser-Zugang zu Colab) – die einzelnen Bausteine (uv-venv mit 3.11, Installation von holowizard+torch, Ausführung) sind lokal verifiziert.
- **Torch/GPU in Colab:** holowizard pinnt `torch==2.10`; das PyPI-Wheel bringt CUDA 12.8 mit (lokal installiert: `torch 2.10.0+cu128`), nutzt also die Colab-GPU ohne Zusatzschritt. Für CPU-only (CI, lokal) den PyTorch-CPU-Index verwenden (`--extra-index-url https://download.pytorch.org/whl/cpu` → `torch-2.10.0+cpu`, Auflösung mit `pip --dry-run` geprüft).
- **Colab-Limits:** Free: „notebooks can run for at most 12 hours“, GPU-Typen und Idle-Timeout werden nicht veröffentlicht, keine Garantie für GPU; Pro+ erlaubt „continuous code execution for up to 24 hours“ (Colab-FAQ, research.google.com/colaboratory/faq.html, 2026-10-01). Preise laut Drittquellen mit Stand 2026-09 (gpuperhour.com, freetier.co): Pro 9,99 $/Monat mit 100 Compute Units, Pro+ 49,99 $/Monat mit 600 Units, Pay-as-you-go 9,99 $/100 Units (90 Tage gültig) – vor dem Kauf auf colab.research.google.com/signup prüfen.
- **Kaggle:** 30 GPU-h/Woche („floating“, manchmal mehr), Reset samstags 00:00 UTC, 12 h pro CPU/GPU-Session, 9 h TPU; Beschleuniger P100 (16 GB) oder T4×2. Quelle: kaggle.com/docs/notebooks, kaggle.com/docs/efficient-gpu-usage.
- **Lightning AI Studio:** Free-Tier seit 2026-09 ein **einmaliges** Guthaben (bis 30 Credits ≈ 75 T4-Stunden, 12 Monate gültig; lightning.ai-Preisseite laut usagepricing.com), keine monatliche Auffrischung mehr. **Paperspace Gradient (DigitalOcean):** Free-Plan mit kostenloser M4000-GPU für Notebooks in privaten Workspaces (Verfügbarkeit schwankt, Auto-Shutdown-Limit; docs.digitalocean.com/products/paperspace/pricing), Pro 8 $/Monat mit schnelleren Free-GPUs. Beide nur als Reserve; nicht getestet.

### 3.2 Colab-Notebook – Ablauf

`notebooks/colab_training.ipynb` (Badge in `notebooks/README.md`; öffnet direkt aus GitHub):

1. Laufzeit → Laufzeittyp ändern → **T4 GPU**.
2. `!nvidia-smi` – GPU prüfen (läuft auch ohne GPU weiter).
3. Google Drive mounten (`USE_DRIVE = True`) → Ablage `MyDrive/masterarbeit_autofocus/`.
4. Repo klonen (`BRANCH` wählbar), `%cd`, aktuellen Commit anzeigen.
5. `uv venv --python 3.11 .venv` + `uv pip install -r requirements.txt` (Fallback-Zelle 5b installiert `holowizard pytest pyyaml matplotlib tensorboard tqdm h5py`); Zelle 5c prüft torch/CUDA/holowizard im venv.
6. Datenerzeugung: `python -m src.data.generate_data --config configs/data_small.yaml --out data/processed/small` (optional `data_p05.yaml`); ein vorhandener Datensatz (`meta.json`) wird übersprungen, `OVERWRITE = True` erzwingt `--overwrite`; gibt den `meta.json`-Hash fürs Logbuch aus.
7. Training: `python -m src.train --config configs/base.yaml --data-dir … --run-name <JJJJ-MM-TT>_<kurzname>_s<seed>` → `runs/<run>/` (Dateien siehe Abschnitt 3.4).
8. `%tensorboard --logdir runs` (liest `runs/<run>/tb/`).
9. Evaluation: `python -m src.evaluate --checkpoint runs/<run>/best.pt --data data/processed/small/test.hdf5 --out runs/<run>/eval_test` (entspricht dem Default `<Checkpoint-Ordner>/eval_<Dateistamm>/`); liest danach MAE z01 und rel. Fr-Fehler aus `results.json`/`eval_results.json`.
10. Sicherung nach Drive: `results.json`, `history.json`, `config.yaml`, `eval_test/eval_results.json`, Figuren, `*.npz`, `tb/`, optional `best.pt`; druckt die fertige Logbuch-Zeile (mit MAE/rel. Fr-Fehler).
11. Hinweise zu Session-Limits (kein Resume in `src.train`; `--epochs` zum Kürzen) und Hygiene.

Alle Shell-Zellen prüfen vorher, ob `.venv/bin/python` existiert, und brechen mit klarer Meldung ab.

### 3.3 Daten und Ergebnisse: was wohin

| Artefakt | Ort | Versioniert? |
| --- | --- | --- |
| Rohdaten / HDF5-Datensätze (`data/processed/<name>/*.hdf5`, dazu `train/val/test.json` = HoloForge-Config je Split) | lokal, Drive, Cluster-Scratch | **nein** (`.gitignore`); nur `meta.json`-Hash im Logbuch |
| `data/processed/<name>/meta.json` | neben den Daten; Kopie in Drive | optional ja (klein) – empfohlen, sobald ein Datensatz in der Thesis verwendet wird (`git add -f`, da `data/processed/` ignoriert ist) |
| Checkpoints `runs/<run>/best.pt`, `*.npz` | Drive / Cluster | **nein** |
| `runs/<run>/results.json`, `runs/<run>/eval_<split>/eval_results.json`, Figuren (`*.png`) | Repo (per PR, `git add -f`, da `runs/` ignoriert ist), Drive | **ja** |
| TensorBoard-Logs `runs/<run>/tb/` | Drive; lokal zum Anschauen | nein (optional für finale Runs) |
| Thesis-Abbildungen | `thesis/figures/<kapitel>/` | ja |

Datenaustausch Drive ↔ Repo: Ergebnisse aus Drive in `runs/` kopieren, dann `tools/export_figures.py` für die Thesis-Figuren und Commit `exp: …`.

### 3.4 CLI-Referenz (Stand des Codes, geprüft gegen `src/*.py` und einen Smoke-Run)

```bash
# Daten: Default-Ausgabe <output_dir>/<name> aus der YAML (data/processed/small); bricht ohne --overwrite ab, wenn Splits existieren
python -m src.data.generate_data --config configs/data_small.yaml [--out data/processed/<name>] [--overwrite]
#   -> train.hdf5, val.hdf5, test.hdf5, train.json/val.json/test.json (HoloForge-Config je Split), meta.json
#      (Config, Seeds, Zeiten, Versionen, Abtastprüfung "propagator_sampling", Label-Verifikation)

# Training: Run-Ordner <paths.runs_dir>/<run-name> (Default-Name: <JJJJMMTT_HHMMSS>_<arch>_<target>)
python -m src.train --config configs/base.yaml [--data-dir data/processed/<name>] [--run-name …] [--epochs N] [--device auto|cpu|cuda|mps]
#   -> best.pt, config.yaml, history.json, results.json (Test-Metriken des besten Checkpoints, Zeiten, Inferenzzeit),
#      loss_curves.png, test_scatter_target.png, test_scatter_z01_mm.png, test_error_vs_z01.png, test_examples.png,
#      test_predictions.npz, tb/ (TensorBoard)

# Evaluation: beliebige HDF5-Datei; Ausgabe <Checkpoint-Ordner>/eval_<Dateistamm>/ (z. B. runs/<run>/eval_test/) oder --out
python -m src.evaluate --checkpoint runs/<run>/best.pt --data data/processed/<name>/test.hdf5 [--out DIR] [--batch-size N] [--device …] [--num-workers N] [--threads N]
#   -> eval_results.json, predictions.npz, scatter_target.png, scatter_z01_mm.png, error_vs_z01.png, examples.png

python -m pytest -q          # ~60 Tests, erzeugt einen winzigen HoloForge-Datensatz im Temp-Verzeichnis (~1 min CPU)
```

Metrik-Schlüssel: in `results.json` unter `test_metrics.physical` (`z01_mae_mm`, `z01_rmse_mm`, `z01_p95_abs_err_mm`, `z01_bias_mm`, `fr_rel_err_mean_pct`, `fr_rel_err_median_pct`, …) und `test_metrics.target_space` (`mae`, `rmse`, `r2`, …); in `eval_results.json` dieselben Blöcke unter `metrics.physical` / `metrics.target_space`; Inferenzzeit unter `inference_time.ms_per_hologram_batch_1` (auf CPU von `--threads` abhängig).

---

## 4. Experimente protokollieren

### 4.1 Drei Ebenen

1. **TensorBoard** (automatisch, pro Run): Verlustkurven, Metriken je Epoche, ggf. Bilder. Anschauen: `tensorboard --logdir runs` lokal oder inline in Colab. Logs bleiben im Run-Ordner; kein Account, offline, reproduzierbar aus `results.json`.
2. **Logbuch** `docs/notizen/experimente.md` (manuell, 2 Minuten pro Run): die zitierfähige Tabelle. Spalten: Run · Datum · Commit-SHA · Daten-Version/`meta.json`-Hash · Config-Änderung · MAE z01 [mm] · rel. Fr-Fehler [%] · Beobachtung · nächster Schritt. Ein Beispieleintrag steht drin (als Beispiel markiert).
3. **Experiment-Issue** (pro Serie): Hypothese vorher, Entscheidung nachher. Verweist auf die Logbuch-Zeilen.

### 4.2 Konventionen

- **`run_name` = `JJJJ-MM-TT_<kurzname>_s<seed>`**, z. B. `2026-10-15_cnn-baseline_s0`. Datum zuerst (sortierbar), Kurzname kleingeschrieben mit Bindestrichen, Seed am Ende. Mehrere Seeds (mindestens 3 für Zahlen in der Thesis) → Mittelwert ± Std.
- **Immer notieren:** Seed, Config-Datei + Abweichungen, Commit-SHA (`git rev-parse --short=12 HEAD`; nicht mit uncommitteten Änderungen trainieren – sonst `+dirty` vermerken), Datensatz-Hash (`sha256sum data/processed/<name>/meta.json | cut -c1-16`), Hardware (CPU/T4/Cluster) und Laufzeit.
- **Metriken** auf dem Testsplit: MAE von z01 in mm (`z01_mae_mm`); relativer Fr-Fehler in % (`fr_rel_err_mean_pct`, Umrechnung über Fr = Δx²/(λ (z02 − z01) M), M = z02/z01 – gleiche Formel wie HoloWizard, siehe `docs/01_thesis_erklaerung.md`); für den Vergleich mit dem modellbasierten Autofokus zusätzlich Inferenzzeit pro Hologramm (`inference_time.ms_per_hologram_batch_1`) mit Hardware- und Thread-Angabe.
- Nach jedem Lauf, der in die Thesis könnte: `results.json` + Figuren committen, Logbuch-Zeile, ggf. Tag.

### 4.3 Warum nicht Weights & Biases oder MLflow?

- **W&B**: für Studierende kostenlos („Academic“-Plan: Pro-Funktionen, 200 GB Speicher, Antrag mit Hochschul-E-Mail über die Pricing-Seite; der normale Free-Plan hat 5 GB) – verifiziert auf wandb.ai/site/pricing, 2026-10-01. Vorteile: Vergleich vieler Runs im Browser, Hyperparameter-Sweeps. Nachteile: Account und Internet nötig (auf dem DESY-Cluster evtl. eingeschränkt), Daten liegen extern, und am Ende zählt ohnehin die Tabelle im Logbuch. → **Optional ab Phase „Ablationen“**, wenn > 30 Runs verglichen werden; dann `wandb` nur zusätzlich zu TensorBoard loggen (ein `--wandb`-Flag in `src.train`), nie als einzige Quelle.
- **MLflow**: lokal/offline möglich (`mlflow ui`), aber zusätzliche Infrastruktur für wenig Mehrwert bei einer Person. Nicht vorgesehen.
- **Entscheidung**: TensorBoard + Logbuch + Issues. Einfach, offline-fähig, reproduzierbar, nichts zu betreiben.

---

## 5. Literatur & Zitieren

### 5.1 Zotero-Workflow (Schritt für Schritt)

1. **Zotero 7** installieren (zotero.org, kostenlos; Metadaten-Sync unbegrenzt, Cloud-Speicher für PDFs im Free-Account begrenzt – aktuelle Grenze auf zotero.org/storage prüfen; PDFs können alternativ lokal bleiben) und den **Zotero Connector** für den Browser.
2. **Better BibTeX (BBT)** installieren (retorque.re/zotero-better-bibtex → `.xpi` herunterladen → Zotero: *Tools → Plugins → Zahnrad → Install Plugin From File*).
3. BBT-Einstellungen (*Edit → Settings → Better BibTeX*):
   - *Citation key formula*: `auth.lower + year + shorttitle(1,0).lower` → `dora2025model`. Für sprechende Kurzwörter den Key **pinnen**: im Zotero-Feld *Extra* des Eintrags die Zeile `Citation Key: dora2025autofocus` eintragen (BBT übernimmt gepinnte Keys unverändert; ungepinnte Keys erhalten bei Kollision automatisch Suffixe a, b, …).
   - *Export → Fields*: `abstract, file, keywords` abwählen (hält die `.bib` klein).
4. **Sammlung** „Masterarbeit Autofokus“ anlegen, Unterordner nach Kapitel (Grundlagen NFH, Autofokus, Deep Learning, SBI). Paper über den Connector (DOI-Seite oder PDF) hinzufügen; Zotero holt Metadaten und PDF.
5. **Auto-Export**: Rechtsklick auf die Sammlung → *Export Collection…* → Format **Better BibLaTeX** → Haken **Keep updated** → Ziel `<Repo>/thesis/references.bib`. Ab jetzt schreibt Zotero die Datei bei jeder Änderung neu (verwalten unter *Settings → Better BibTeX → Automatic export*).
6. **Bestand übernehmen**: die vorhandene `thesis/references.bib` (10 Einträge, DOIs geprüft) in Zotero importieren (*File → Import*), Citekeys als gepinnt übernehmen (BBT liest `@article{dora2025autofocus,` als Key). Danach die ~28 weiteren Quellen aus `docs/02_literatur.md` nach und nach ergänzen (dort sind sie bisher mit [1]–[34] nummeriert; beim Übertragen Citekeys vergeben und in `02_literatur.md` dazuschreiben).
7. **Alternativen** (falls Zotero nicht gefällt): *JabRef* (reine `.bib`-Pflege, Open Source, gut für LaTeX-Nutzer), *Mendeley* (Elsevier, Cloud-zentriert, BibTeX-Export weniger kontrollierbar). Empfehlung bleibt Zotero + BBT wegen stabiler Keys und Auto-Export.

### 5.2 Volltexte

- *Optics Express* (Dora et al. 2024, 2025) ist vollständig Open Access (CC BY) – direkt über die DOI.
- *Journal of Microscopy* (Paganin 2002), *Applied Optics* (Fienup 1982) und die meisten Quellen in `docs/02_literatur.md` sind lizenzpflichtig → über **TUHH-Bibliothek/VPN** (tub.tuhh.de, Zugriff auf Wiley/Optica je nach Lizenz – nicht geprüft) oder die Autoren anfragen; viele Autoren legen Postprints auf arXiv/Uni-Servern ab (Google Scholar „Alle Versionen“).
- DOI → BibTeX ohne Zotero: `curl -sL -H "Accept: application/x-bibtex" https://doi.org/10.1364/OE.544573` (so wurden alle Einträge in `references.bib` erzeugt/geprüft).

### 5.3 Zitieren in LaTeX

- `biblatex` mit `backend=biber`, `style=ieee`, `sorting=none` (Nummern in Reihenfolge des ersten Zitats – wie in den Optics-Express-Papern der Gruppe). Umschalten auf Autor-Jahr: `style=authoryear` in `thesis/main.tex`.
- Befehle: `\cite{dora2025autofocus}`, mehrere `\cite{a,b}`, mit Seite `\cite[S.~6645]{dora2025autofocus}`, im Fließtext `\textcite{…}`.
- Software zitieren: HoloWizard (`dora2026holowizard`, Zenodo-Versions-DOI 10.5281/zenodo.19560147 für v3.0.6; Konzept-DOI 10.5281/zenodo.16275927; die ältere Referenz [5] der Aufgabenstellung ist `dora2024framework`, 10.5281/zenodo.14024980 = Version 1.3.1 vom 2024-11-01 – beides per Zenodo-API geprüft), PyTorch (`paszke2019pytorch`), sbi (`tejerocantero2020sbi`, `boelts2025sbireloaded` = JOSS 10(108):7754, DOI 10.21105/joss.07754), SBI-Leitfaden (`deistler2025sbi` = arXiv:2508.12939).

### 5.4 Zitieren in Markdown-Notizen

Citekey in eckigen Klammern: „Der modellbasierte Autofokus [dora2025autofocus] braucht pro Hologramm …“. Seitenzahlen als `[dora2025autofocus, S. 6645]`. Jede Paper-Notiz heißt `docs/notizen/paper/<citekey>.md` (Vorlage: `docs/notizen/vorlage_paper_notiz.md`). So lassen sich Notizen per Suchen/Ersetzen in `\cite{}` überführen.

---

## 6. Schreiben

### 6.1 LaTeX-Skelett `thesis/`

Struktur und Bedienung stehen in `thesis/README.md`. Kurz:

- `main.tex`: KOMA-Script `scrbook`, 11 pt, A4; Pakete `babel` (Sprachschalter), `csquotes`, `amsmath/mathtools`, `siunitx`, `graphicx`, `booktabs`, `caption/subcaption`, `microtype`, `biblatex` (ieee, biber), `acro` (Abkürzungen), `hyperref` + `cleveref`. Notationsmakros `\Fr`, `\zOne`, `\zTwo`, `\magn`, `\wl`, `\dx`, `\prop{\Fr}` – identisch mit HoloWizard und Dora et al. 2025. `\todo{…}` markiert offene Stellen rot.
- `frontmatter/`: Titelseite (Platzhalter TUHH/Institut/Prüfer/Betreuer), Eidesstattliche Erklärung (Platzhaltertext → offiziellen TUHH-Wortlaut einsetzen), Kurzfassung + Abstract, Abkürzungen.
- `chapters/01_einleitung` … `07_fazit`, `anhang`: jedes Kapitel beginnt mit einem Kommentarblock „Was hinein gehört“ und enthält Beispiele für Zitat, Formel (u. a. Fr-Formel), Abbildung (per Snippet) und Tabelle (booktabs + siunitx).
- **Sprache**: `\newcommand{\thesislanguage}{german}` ↔ `{english}` in `main.tex`. Steuert babel, siunitx-Locale, cleveref, PDF-Metadaten und alle `\lang{DE}{EN}`-Bausteine; Kurzfassung/Abstract immer beide Sprachen.
- **Kompilieren** (beides am 2026-10-01 geprüft, 22 Seiten, 0 unaufgelöste Zitate): `latexmk` (TeX Live 2023: pdflatex + biber 2.19, `.latexmkrc` liegt bei) oder `tectonic -X compile main.tex` (tectonic 0.17 mit TeX-Live-2022-Bundle; braucht ein zur biblatex-Version passendes `biber` 2.17 – Details in `thesis/README.md`). Empfehlung lokal: TeX Live + `latexmk -pvc` im Editor (VS Code mit *LaTeX Workshop* oder TeXstudio).
- **Build-Artefakte** nicht committen: die LaTeX-Muster (`thesis/*.pdf`, `*.aux`, `*.bbl`, …) stehen in der Haupt-`.gitignore`; `thesis/figures/**/*.pdf` bleibt bewusst versioniert.

### 6.2 Overleaf

- Nutzen: Korrekturlesen durch Betreuer mit Kommentarfunktion, Schreiben ohne lokale TeX-Installation.
- **Free-Plan (Stand 2026-10-01, docs.overleaf.com):** Compile-Timeout **10 s** (Premium 240 s), **1 Kollaborator** pro Projekt (Student/Standard 10, Pro unbegrenzt), 2000 Dateien, 7 MB editierbarer Text pro Projekt. **GitHub-Synchronisation, Git-Zugriff, Zotero/Mendeley-Integration und Track Changes sind Premium-Funktionen.** Wenn der Projekt-Eigentümer Premium hat, können eingeladene Free-Nutzer die projektbezogenen Funktionen mitnutzen.
- Empfohlener Weg ohne Premium: lokal mit Git arbeiten (Quelle der Wahrheit = Repo), für Review-Runden den Ordner `thesis/` als ZIP hochladen (*New Project → Upload*). Kapitelweise Reviews über GitHub-PRs (`thesis/<kapitel>`-Branch) sind die Alternative ohne Overleaf.
- Das Skelett braucht lokal ca. 9 s für einen kompletten Build (drei pdflatex-Durchläufe + biber); Overleaf nutzt Zwischenergebnisse, sodass einzelne Compiles kürzer sind. Wenn das 10-s-Limit später trotzdem greift: PDF- statt PNG-Grafiken (die Pipeline schreibt PNG; das Export-Skript bevorzugt PDF, wenn beide vorliegen), `\includeonly{chapters/05_experimente}` während des Schreibens, oder Student-Plan.
- TUHH-Vorlagen: Es gibt **keine zentrale TUHH- oder IBI-LaTeX-Vorlage** (gefunden am 2026-10-01: Overleaf-Vorlage des Instituts für Data Engineering „TUHH IDE Thesis Template“ von D. Schallmoser, CC BY 4.0, overleaf.com/latex/templates/tuhh-ide-thesis-template/rmdsqzgknkpr; TUHH-GitLab-Vorlage des Instituts M-21 collaborating.tuhh.de/m21/public/theses/itt-latex-template; ältere Vorlage collaborating.tuhh.de/cst9446/latex-header; IBBD stellt nur Word-Formatvorlagen bereit). Die IBI-Seite tuhh.de/ibi/thesis nennt keine Vorlage. → Frage **B** an die Betreuer; falls das Institut eine Vorlage vorgibt, übernimm deren Titelseite/Erklärung und behalte Kapitelstruktur, Makros und `references.bib` aus `thesis/`.

### 6.3 Konventionen im Text

- **Einheiten** immer mit `siunitx`: `\qty{6.5}{\micro\metre}`, `\qty{11}{\kilo\electronvolt}`, `\qty{0.85}{\milli\metre}`, Bereiche `\qtyrange{20}{500}{\milli\metre}`; in Tabellen `S`-Spalten mit `table-format`.
- **Abkürzungen** über `acro`: definieren in `frontmatter/acronyms.tex`, erste Verwendung `\ac{nfh}` (Langform + Kurzform), danach automatisch nur Kurzform; in Tabellen/Überschriften `\acs{}`; Verzeichnis per `\printacronyms` (nur verwendete).
- **Notation** konsistent mit HoloWizard: Fr (Pixel-Fresnel-Zahl), z01 (Fokus–Objekt), z02 (Fokus–Detektor), M = z02/z01, λ, Δx; Propagator D_Fr. Nicht mischen (z. B. kein „d“ für Abstände).
- **Labels/Verweise**: `ch:`, `sec:<kap>:<name>`, `fig:<kap>:<name>`, `tab:<kap>:<name>`, `eq:<kap>:<name>`; immer `\cref`.
- **Zahlen** aus `results.json` übernehmen, nie abtippen aus Konsolenausgaben; Rundung auf signifikante Stellen (MAE in mm mit 2 Dezimalen).

---

## 7. Abbildungen

### 7.1 Stil `thesis/thesis.mplstyle`

- Figurbreite 5.9 in (≈ Textbreite bei A4/11 pt), halbe Breite `figsize=(2.9, 2.2)`; Schriftgrößen 9–10 pt (Achsen, Legende, Ticks), damit nichts größer wirkt als der Fließtext.
- Serifenschrift passend zu Latin Modern (Fallback-Kette `Latin Modern Roman → CMU Serif → TeX Gyre Termes → Times New Roman → DejaVu Serif`), Mathe in Computer Modern (`mathtext.fontset: cm`), kein `usetex` (läuft so auch in Colab).
- Farbpalette **Okabe-Ito** (farbenblind-sicher), Colormap `viridis`.
- Export: `savefig.format: pdf`, `bbox: tight`, `pdf.fonttype: 42` (editierbare Schriften), 300 dpi für Raster.
- Verwendung in eigenen Skripten: `plt.style.use("thesis/thesis.mplstyle")`. Die Pipeline-Plots aus `src/utils/plotting.py` (PNG, Standard-Matplotlib-Stil) nutzen den Stil bisher **nicht**; für Thesis-Abbildungen entweder dort einbinden (Code-Änderung) oder die Daten aus `history.json`/`predictions.npz` mit einem eigenen Skript im Thesis-Stil neu zeichnen.

### 7.2 Export-Skript `tools/export_figures.py`

```bash
python tools/export_figures.py --run runs/<run> --list
python tools/export_figures.py --run runs/<run> --chapter experimente --names loss_curves test_scatter_z01_mm
python tools/export_figures.py --run runs/<run> --chapter anhang --names "test_*" --width "0.7\textwidth"
python tools/export_figures.py --run runs/<run>/eval_ood --chapter anhang --names "*"     # ein einzelner eval_*-Ordner
```

- Figurnamen = Dateistamm der Pipeline-Plots: aus `src.train` `loss_curves`, `test_scatter_target`, `test_scatter_z01_mm`, `test_error_vs_z01`, `test_examples` (direkt in `runs/<run>/`); aus `src.evaluate` `scatter_target`, `scatter_z01_mm`, `error_vs_z01`, `examples` (in `runs/<run>/eval_<split>/`). `--list` zeigt, was ein Run tatsächlich enthält.
- Sucht rekursiv im Run-Ordner nach `<name>.{pdf,png,jpg,eps}`, kopiert nach `thesis/figures/<kapitel>/`, schreibt `<name>.tex` mit `\begin{figure}`, `\includegraphics`, Caption-Platzhalter, Label `fig:<kapitel>:<name>` und einem **Kommentar mit Quelle**: Run-Pfad, Commit-SHA (`git rev-parse HEAD`, nur lesend; `+dirty`-Hinweis bei uncommitteten Änderungen), Datum, Dateien.
- Einbinden im Kapitel: `\input{figures/experimente/test_scatter_z01_mm}`.
- Bestehende Snippets werden nicht überschrieben (Caption bleibt erhalten), Grafiken schon; `--force` überschreibt, `--dry-run` zeigt nur, `--list` listet. Liegt derselbe Name in mehreren `eval_*`-Ordnern, wird er übersprungen (Warnung) – dann `--run` auf den Unterordner setzen. Nur Standardbibliothek; getestet mit dem Smoke-Run `runs/smoke_cnn_logfr` (fehlende Namen → Warnung + Exit-Code 2; Glob-Muster; Schutz bestehender Snippets; Mehrdeutigkeit).

### 7.3 Regel

**Jede Abbildung in der Thesis hat eine reproduzierende Quelle**: Run-Name + Commit-SHA (im Snippet-Kommentar und im Logbuch) oder – bei schematischen Grafiken – das erzeugende Skript. Abbildungen ohne Quelle fliegen raus. Beispiel im Skelett: `thesis/figures/grundlagen/fresnelzahl.{pdf,tex}`.

---

## 8. Notizen

`docs/notizen/` (Regeln in `docs/notizen/README.md`):

| Datei | Zweck |
| --- | --- |
| `vorlage_wochennotiz.md` | Freitags 15 min: erledigt, Erkenntnisse (mit Run/Citekey), Blocker, Entscheidungen, Plan, Zeitaufwand → `wochen/JJJJ-WW.md` |
| `vorlage_betreuer_meeting.md` | Datum, Teilnehmer, Status seit letztem Mal, gezeigte Ergebnisse, offene Fragen, Antworten, Entscheidungen (mit verworfenen Alternativen), To-dos mit Owner → `meetings/JJJJ-MM-TT_betreuer.md` |
| `vorlage_paper_notiz.md` | Citekey, Kernaussage in drei Sätzen, Methode, Ergebnisse, Relevanz, Zitate mit Seitenzahl, offene Fragen → `paper/<citekey>.md` |
| `experimente.md` | Experiment-Logbuch (Abschnitt 4) |
| `fragen_an_betreuer.md` | priorisierte Fragenliste (A/B/C) mit Startfragen zu Messdaten, Referenz-Fr, Rechenressourcen, Sprache, Vorlage, Umfang, Formalia, Meeting-Rhythmus |

Optional **Obsidian**: Vault auf `docs/` zeigen lassen; `[[dateiname]]`-Links und Graph-Ansicht funktionieren über die Markdown-Dateien; `.obsidian/` in `.gitignore`. Kein Zwang – GitHub rendert die Dateien ebenso.

---

## 9. Reproduzierbarkeits-Checkliste (pro Ergebnis in der Thesis)

- [ ] Commit-SHA (und ggf. Tag `exp/…`) notiert; Code zu diesem Stand ist auf `main` oder getaggt
- [ ] Config-Datei + Abweichungen dokumentiert (Logbuch-Spalte „Config-Änderung“)
- [ ] Datensatz identifiziert: Name + `meta.json`-Hash; `meta.json` committet
- [ ] Seeds angegeben; Zahl ist Mittelwert ± Std über ≥ 3 Seeds (oder begründet, warum nicht)
- [ ] Metriken aus `results.json`/`eval_results.json`, Dateien committet
- [ ] Hardware und Laufzeit angegeben (für Laufzeitvergleiche Pflicht)
- [ ] Abbildung über `tools/export_figures.py` exportiert (Quelle im Snippet-Kommentar)
- [ ] Logbuch-Zeile vorhanden; Experiment-Issue geschlossen mit Entscheidung
- [ ] Software-Versionen bekannt (`requirements.txt`-Stand des Commits; holowizard 3.0.6, torch 2.10)
- [ ] Befehl zur Reproduktion in einer Zeile aufschreibbar (Anhang „Reproduktion der Ergebnisse“)

---

## 10. Was hier nicht eingerichtet werden konnte – Klick-Anleitung

Diese Schritte brauchen deine Accounts oder die GitHub-Weboberfläche (das Repo-Setup selbst ist fertig):

1. **GitHub – Labels, Milestones, Projects-Board**
   - Repo → *Issues → Labels → New label*: `experiment`, `aufgabe`, `literatur`, `blocker`, `frage-betreuer`, `thesis`.
   - *Issues → Milestones → New milestone*: eine pro Phase aus `docs/03_projektplan.md` mit Due date.
   - Profil → *Projects → New project → Board* → Name „Masterarbeit Autofokus“ → *Add item* → Repo verknüpfen; Spalten *Backlog, Diese Woche, In Arbeit, Review/Betreuer, Fertig*; *Workflows*: „Item added → Backlog“, „Item closed → Fertig“.
   - *Settings → Collaborators*: Betreuer mit *Read* (oder *Triage*) einladen, falls sie mitlesen sollen.
   - *Settings → Branches → Add rule* für `main`: „Require status checks to pass“ → `pytest (py3.11, CPU)` auswählen (erst sichtbar, nachdem der Workflow einmal gelaufen ist).
2. **Zotero**: zotero.org → Download Zotero 7 + Connector; Account für Sync (optional); Better BibTeX `.xpi` installieren; Einstellungen und Auto-Export wie in Abschnitt 5.1; `thesis/references.bib` importieren.
3. **Overleaf** (optional): overleaf.com → Account mit TUHH-Mail (prüfen, ob die TUHH eine Overleaf-Commons-Lizenz hat – dann wären Premium-Funktionen inklusive; nicht verifiziert) → *New Project → Upload Project* mit ZIP von `thesis/` → Compiler pdfLaTeX, Main document `main.tex` → Betreuer per *Share* einladen (Free-Plan: 1 Person).
4. **Google Colab / Drive**: Google-Konto; Notebook über den Badge in `notebooks/README.md` öffnen; beim ersten Drive-Mount Zugriff bestätigen; Ordner `MyDrive/masterarbeit_autofocus/` entsteht automatisch. Colab Pro (9,99 $/Monat) erst kaufen, wenn Free-Sessions tatsächlich zu kurz sind oder keine GPU zugeteilt wird. Kaggle: kaggle.com → Account verifizieren (Telefon) für GPU-Zugang → *Code → New Notebook → File → Import Notebook → GitHub*.
5. **Weights & Biases** (optional, Abschnitt 4.3): wandb.ai → Sign up mit TUHH-Mail → Pricing-Seite → *Academic* beantragen → API-Key in Colab als Secret hinterlegen (`🔑`-Seitenleiste), nie ins Repo.
6. **Zenodo** (zum Ende): zenodo.org → Login mit GitHub → *GitHub* → Repo auf „ON“ → GitHub-Release `v1.0-thesis` erstellen → DOI erscheint im Zenodo-Dashboard → `@software`-Eintrag in `references.bib`.
7. **Rechenzugang DESY/TUHH**: Fragen A aus `docs/notizen/fragen_an_betreuer.md` im ersten Meeting klären; Account-Antrag läuft über die Betreuer.

---

## Quellen der verifizierten Fakten (abgerufen 2026-10-01)

- Colab Runtime-Versionen: research.google.com/colaboratory/runtime-version-faq.html; github.com/googlecolab/backend-info; Colab-FAQ research.google.com/colaboratory/faq.html (12-h-Limit, 24 h Pro+, keine veröffentlichten GPU-/Idle-Grenzen); Colab-Preise nur aus Drittquellen (gpuperhour.com, freetier.co; Stand 2026-09) – Original: colab.research.google.com/signup.
- Colab + uv-Umgebungsvariablen: github.com/googlecolab/colabtools/issues/5237; condacolab: pypi.org/project/condacolab.
- Kaggle: kaggle.com/docs/notebooks; kaggle.com/docs/efficient-gpu-usage; kaggle.com/product-feedback/173129.
- Overleaf: docs.overleaf.com/getting-started/free-and-premium-plans/plan-limits; …/premium-features; …/github-synchronization; overleaf.com/user/subscription/plans.
- Weights & Biases: wandb.ai/site/pricing; docs.wandb.ai (Academic-Plan).
- Zotero Better BibTeX: retorque.re/zotero-better-bibtex/citing/ (Formel, Pinning), …/exporting/auto/ (Keep updated).
- holowizard-Metadaten: pypi.org/pypi/holowizard/json (3.0.6, `>=3.11,<3.12`, `torch==2.10`); Zenodo-API zenodo.org/api/records/14024980 und /19560147 (Konzept 16275927).
- DOIs/BibTeX: doi.org Content-Negotiation (`Accept: application/x-bibtex`) für alle Einträge in `thesis/references.bib`; arXiv-API für 2508.12939.
- TUHH-Vorlagen: overleaf.com/latex/templates/tuhh-ide-thesis-template/rmdsqzgknkpr; collaborating.tuhh.de/m21/public/theses/itt-latex-template; collaborating.tuhh.de/cst9446/latex-header; tuhh.de/ibi/thesis.
- Lightning AI / Paperspace: lightning.ai Preisseite (Stand 2026-09 laut usagepricing.com); docs.digitalocean.com/products/paperspace/pricing.
