# Learning-based Autofocus for Holography (Masterarbeit)

Masterarbeit an der TUHH, Institut für Biomedizinische Bildgebung, in Zusammenarbeit mit DESY
(Betreuung: Daniel Hernández Durán, Johannes Dora, Tobias Knopp).

## Ziel der Arbeit

In der Röntgen-Nahfeldholographie (NFH) wird die Austrittswelle `ψ = exp(iO)·P` eines Objekts mit dem
Fresnel-Propagator `D_Fr` zum Detektor propagiert, `I = |D_Fr(ψ)|²`. Die Fresnel-Zahl `Fr` fasst die
Geometrie (Abstände `z01`, `z02`, Energie, Pixelgröße) zusammen und muss für die Phasenrekonstruktion
exakt bekannt sein; ihre Bestimmung ist das Autofokus-Problem. Der Stand der Technik ist eine
modellbasierte Optimierung (Dora et al., Opt. Express 33(4), 2025), die interpretierbar, aber langsam
ist. Diese Arbeit untersucht, ob `Fr` bzw. `z01` direkt und schnell aus einem einzelnen Hologramm mit
lernbasierten Modellen geschätzt werden kann. Die Trainingsdaten werden mit **HoloForge** (Teil von
[HoloWizard](https://github.com/DESY-FS-PETRA/holowizard), DESY) simuliert.

## Projektstruktur

```
configs/
  base.yaml            Experiment (data / model / train / paths) für Training und Evaluation
  data_small.yaml      Datensatz für CPU-Smoke-Tests (2048 px um 8 gebinnt -> 256 px, 400/100/100 Samples)
  data_p05.yaml        Realistischer P05-Datensatz (2048 px, Padding 4, 10000/1000/1000) – nur für GPU
src/
  data/forge_setup.py  NFHRandomDistSetup: HoloForge-Setup mit zufälligem z01 (und optional z02) pro Hologramm
  data/generate_data.py  CLI: YAML -> HoloForge-Konfiguration -> train/val/test.hdf5 + meta.json
  data/dataset.py      HologramHDF5Dataset (Target-Modi, Normalisierung, Crop/Downsample), make_dataloaders
  data/representations.py  Eingaberepräsentationen: Hologramm, log-Leistungsspektrum, 2-Kanal, Radialprofil (data.representation)
  baseline/ctf_ringfit.py  Lernfreie Baseline: Fr aus den CTF-Ringen im Radialspektrum (CLI, Bericht in reports/ringfit/)
  baseline/model_based_autofocus.py  Modellbasierter Autofokus (HoloWizard find_focus) + klassischer Schärfemetrik-Fallback (CLI, reports/baseline_model_based.md)
  baseline/results.py  Gemeinsames Ergebnisschema (samples.csv) und Statistiken für alle Autofokus-Methoden
  eval/downstream.py   Downstream-Test: Rekonstruktionsqualität bei falschem/geschätztem Fr (CLI, reports/downstream/)
  models/cnn.py        AutofocusCNN (beliebige Auflösung via AdaptiveAvgPool2d), ResNet-18-Variante, RadialProfileMLP, build_model
  train.py             CLI: Training mit Early Stopping, bestem Checkpoint, TensorBoard, Test-Evaluation
  evaluate.py          CLI: Checkpoint auf beliebiger HDF5-Datei auswerten (Metriken, Plots, Inferenzzeit)
  utils/physics.py     Fresnel-Zahl, Umkehrung nach z01, effektive Geometrie (identisch zu holowizard calc_Fr)
  utils/fresnel.py     Torch-Vorwärtsmodell (Fresnel-Propagator in HoloForge-Konvention), Defokus-Unschärfe b = sqrt(|e|/Fr)
  utils/metrics.py     Regressionsmetriken im Zielraum und in physikalischen Einheiten (z01 in mm, Fr in %, Unschärfe in px)
  utils/plotting.py    Headless-Plots (Loss, Scatter, Fehler über z01, Beispiel-Hologramme)
  utils/config.py, targets.py, torch_utils.py   YAML/Pfade, Ziel-Standardisierung, Seeds/Device
tests/                 pytest-Suite (Physik, Random-Setup + Labels, Dataset, Modelle, Trainings-Smoke-Test, Fresnel-Modell, Baselines, Downstream)
tools/build_project_documentation.py   ReportLab-Dokumentation (unabhängig von der Pipeline)
docs/                  Begleitdokumente (01_thesis_erklaerung.md, 02_literatur.md, 03_projektplan.md; entstehen parallel)
```

## Installation

HoloWizard 3.0.6 verlangt Python 3.11 (`>=3.11,<3.12`). Empfohlen ist [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Alternativ mit conda (`environment.yml` installiert dieselben gepinnten Pakete über pip):

```bash
conda env create -f environment.yml
conda activate holophd
```

Alle Befehle werden aus der Projektwurzel als Module gestartet (`python -m ...`); Pfade in den YAML-Dateien sind
relativ zur Projektwurzel. Eine GPU ist nicht erforderlich (`train.device: auto` wählt cuda > mps > cpu).

## Datenerzeugung (HoloForge)

```bash
python -m src.data.generate_data --config configs/data_small.yaml          # -> data/processed/small/
python -m src.data.generate_data --config configs/data_small.yaml --out data/processed/mein_name --overwrite
```

Pro Split (`train`, `val`, `test`) wird aus der YAML eine HoloForge-JSON-Konfiguration gebaut, das Setup
`NFHRandomDistSetup` (eigene Klasse, in `holowizard.forge.experiment.setup` registriert) mit split-eigenem
Seed installiert und HoloForges `DataGenerator`/`HDF5Labeller` ausgeführt. Für **jedes** Hologramm wird
`z01` neu gezogen (uniform oder log-uniform); der Labeller schreibt die tatsächlich verwendeten Werte
`z01`, `z02`, `Fr` pro Sample nach `metadata/setup/`. `meta.json` enthält Konfiguration, Seeds, Zeiten,
Versionen und eine Verifikation (gespeichertes `Fr` gegen Neuberechnung aus `z01`, `z02`, `E`, `px`).

HDF5-Layout: `images/hologram (N,H,W) float32`, optional `images/gt_hologram`, `images/phantoms (complex64)`,
`metadata/setup/{z01,z02,Fr,energy,detector_px_size,detector_size,padding_factor,downsample_factor,probe_size}`.

## Training

```bash
python -m src.train --config configs/base.yaml --data-dir data/processed/small --run-name smoke
tensorboard --logdir runs
```

Ergebnisse landen in `runs/<run_name>/`: `best.pt` (Gewichte + Konfiguration + Target-Modus + Ziel-Skalierung +
Setup-Konstanten), `config.yaml`, `history.json`, `results.json` (Test-Metriken des besten Checkpoints,
Trainings- und Inferenzzeiten), Loss-Kurven, Scatter-Plots (Zielraum und `z01` in mm), Fehler über `z01`,
Beispiel-Hologramme und das TensorBoard-Verzeichnis `tb/`. Konfigurierbar: Ziel (`log_fr` | `fr` | `z01_mm`),
Eingangsnormalisierung, Crop/Downsample, Architektur (`cnn` | `resnet18`), Loss (`mse` | `huber`), Scheduler
(`cosine` | `plateau` | `none`), Early Stopping, Seed, Device.

## Evaluation

```bash
python -m src.evaluate --checkpoint runs/smoke/best.pt --data data/processed/small/test.hdf5
```

Schreibt `eval_results.json` (Metriken im Zielraum und physikalisch: MAE/RMSE/p95/Bias von `z01` in mm,
relativer `Fr`-Fehler in %, Defokus-Unschärfe `blur_px_*`, Inferenzzeit pro Hologramm für Batch 1 und Batch N),
`predictions.npz`, `samples.csv` (gemeinsames Schema aller Methoden) und Plots nach `<Checkpoint-Ordner>/eval_<datei>/`
(oder `--out`). Auf CPU hängt die Batch-1-Latenz stark von der Thread-Zahl ab (Oversubscription auf
kleinen/geteilten Maschinen); `--threads 1` liefert reproduzierbare Werte.

## Baselines und Downstream-Test (HoloWizard-Rekonstruktion)

```bash
python -m src.baseline.model_based_autofocus --data data/processed/small/test.hdf5 --n 8 --method both --out reports/baseline_model_based/small_test
python -m src.eval.downstream --data data/processed/small/test.hdf5 --n 2 --errors -20 -10 -5 -1 0 1 5 10 20 --out reports/downstream/small_test
python -m src.eval.downstream --data data/processed/small/test.hdf5 --n 8 --candidates-csv runs/smoke/eval_test/samples.csv --out reports/downstream/small_test_ml
```

Modellbasierter Autofokus (`holowizard.core` `find_focus`, ≈ 3,5 min pro 256-px-Hologramm auf 2 CPU-Threads) und
klassischer Fallback mit gemeinsamem Ergebnisschema; der Downstream-Test rekonstruiert mit falschem bzw. geschätztem Fr
(`--candidates-csv` liest `samples.csv` beliebiger Methoden) und vergleicht mit `images/phantoms` (benötigt `store.phantom: true`).
Ergebnisse und Konventionen: `reports/baseline_model_based.md`.

## Tests

```bash
python -m pytest
```

Die Suite erzeugt einmalig einen winzigen HoloForge-Datensatz (64 px) im Temp-Verzeichnis und prüft u. a.,
dass `src.utils.physics.fresnel_number` mit `holowizard.forge.utils.calc_Fr` übereinstimmt, dass die gespeicherten
Labels exakt zu den simulierten Hologrammen gehören und dass Training + Evaluation durchlaufen (~1 min auf CPU).

## Einheiten-Konventionen

Wie in HoloWizard: `z01`, `z02` und Pixelgröße in **mm** (6.5 µm = 0.0065 mm), Energie in **keV**,
Wellenlänge `λ = 1.2398 / E` in nm, Dicken der Phantome in µm, `Fr` dimensionslos. Die effektive (Pixel-)
Fresnel-Zahl der Kegelstrahlgeometrie folgt aus dem Fresnel-Skalierungstheorem mit `M = z02/z01`:

```
Fr = (px/M)² / (λ · (z02 − z01)/M) = px² · z01 / (λ · z02 · (z02 − z01))
z01 = Fr · λ · z02² / (px² + Fr · λ · z02)
```

HoloWizard rundet Abstände intern auf 1 µm; `fresnel_number` tut dasselbe und ist dadurch bit-identisch zu
`calc_Fr`. Bei `downsample_factor > 1` speichert HoloForge die *effektive* Pixelgröße (`px · downsample_factor`).

## Bekannte Einschränkungen

- HoloForge speichert in `images/hologram` die **Amplitude** `|ψ|` am Detektor (nicht `|ψ|²`); Rauschen wird auf die
  Amplitude addiert. Die Modelle werden auf dieser Darstellung trainiert; Messdaten (Intensitäten) müssten vor der
  Inferenz radiziert werden (`sqrt(I)`), so wie es HoloWizards `find_focus` ebenfalls tut.
- Der Transferfunktions-Propagator von HoloForge ist nur für `Fr ≥ 1/(Gittergröße)` korrekt abgetastet
  (`Gittergröße = detector_size/downsample_factor · padding_factor`). `generate_data` warnt bei Verletzung; darum
  bindet `data_small.yaml` den 2048-px-Detektor um Faktor 8 und `data_p05.yaml` beginnt bei `z01 = 150 mm`.
- Pro Hologramm wird ein neuer Propagationskernel berechnet; bei 2048 px und Padding 4 sind das 8192² komplexe
  Werte (≈ 0.5 GB) – realistische Datensätze erfordern eine GPU.
- Noch nicht enthalten: Vergleich mit dem modellbasierten Autofokus von HoloWizard
  (`holowizard.core.api.functions.find_focus.find_focus`), simulation-based inference (sbi/NPE), Lader für
  gemessene P05-Daten, gemessene Flatfields (`flatfield_dataset` kann über `forge_overrides` eingebunden werden).
- In HoloForge 3.0.6 lassen sich die Phantom-Glättung nicht deaktivieren und `probe.constant` nicht als Bereich
  angeben (beides führt dort zu Fehlern); die YAML-Schnittstelle erzwingt daher Skalar bzw. Glättung.

## ملخص

يهدف هذا المشروع إلى تقدير عدد فرينل `Fr` (أو المسافة `z01`) مباشرةً من هولوغرام واحد للأشعة السينية
باستخدام شبكات عصبية، كبديل سريع لطرق التحسين النموذجية البطيئة. تُولَّد بيانات التدريب بمحاكي HoloForge
من حزمة HoloWizard، مع مسافة `z01` عشوائية لكل هولوغرام وتخزين القيم الحقيقية كتسميات. يوفر المستودع
سلسلة كاملة لتوليد البيانات والتدريب والتقييم مع اختبارات آلية، وتُشغَّل جميع الأوامر من جذر المشروع
بصيغة `python -m ...`.
