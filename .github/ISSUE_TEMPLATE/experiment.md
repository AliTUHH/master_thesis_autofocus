---
name: Experiment
about: Ein geplantes Experiment mit Hypothese, Setup, Erwartung, Ergebnis und Entscheidung
title: "exp: <kurzer Name>"
labels: experiment
assignees: ''
---

## Hypothese
<!-- Was erwartest du und warum? Ein Satz, falsifizierbar. -->
Wenn …, dann …, weil ….

## Bezug
- Forschungsfrage: F?
- Thesis-Kapitel/Abschnitt: 5.?
- Vorgänger-Experiment / Issue: #

## Setup
- Config: `configs/....yaml` (Abweichungen vom Default hier auflisten)
- Daten: `data/processed/<name>` – `meta.json`-Hash: `sha256:…`
- Seeds: 0, 1, 2
- Hardware: lokal CPU / Colab T4 / Kaggle / Cluster
- Geplante Run-Namen: `YYYY-MM-DD_<kurzname>_s<seed>`

## Erwartete Metriken
| Metrik | Baseline | Erwartung |
| --- | --- | --- |
| MAE z01 [mm] | | |
| rel. Fr-Fehler [%] | | |
| Inferenzzeit [ms/Hologramm] | | |

## Ergebnis
<!-- Nach dem Lauf ausfüllen. Zahlen aus runs/<run>/results.json (test_metrics.physical) bzw.
     runs/<run>/eval_<split>/eval_results.json (metrics.physical), Figuren verlinken. -->
- Commit: `<sha>`
- Runs: 
- Ergebnis-Tabelle:

| Run | MAE z01 [mm] | rel. Fr-Fehler [%] | Bemerkung |
| --- | --- | --- | --- |
| | | | |

- Figuren: `runs/<run>/loss_curves.png`, `runs/<run>/test_*.png`, `runs/<run>/eval_<split>/*.png` (Export in Thesis: `python tools/export_figures.py --run runs/<run> --chapter … --names …`)

## Beobachtungen
<!-- Was war unerwartet? Fehlerfälle? Trainingsverlauf? -->

## Entscheidung
- [ ] Hypothese bestätigt
- [ ] Hypothese widerlegt
- [ ] Unklar, Folgeexperiment nötig: #
- Nächster Schritt:
- Logbuch-Eintrag in `docs/notizen/experimente.md` ergänzt: [ ]
