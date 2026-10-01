# Experiment-Logbuch

Eine Zeile pro Trainingslauf. Ausfüllen **direkt nach dem Lauf** (Zahlen aus `runs/<run>/results.json`, Block `test_metrics.physical`, bzw. aus `runs/<run>/eval_<split>/eval_results.json`, Block `metrics.physical`: `z01_mae_mm`, `fr_rel_err_mean_pct`). Ergänzend gibt es pro Experiment ein GitHub-Issue (Vorlage „Experiment“) mit Hypothese und Entscheidung; hier steht die kompakte, durchsuchbare Historie.

## Konventionen

- **Run** = `JJJJ-MM-TT_<kurzname>_s<seed>` (identisch mit `--run-name` und dem Ordner `runs/…`). Kurzname kleingeschrieben mit Bindestrichen (`cnn-baseline`, `resnet18-logfr`, `npe-flow`).
- **Commit** = `git rev-parse --short=12 HEAD` zum Zeitpunkt des Trainings. Lauf nur mit committetem Stand starten; sonst „+dirty“ anhängen.
- **Daten** = Datensatzname (`small`, `p05`, …) + gekürzter SHA-256 der `meta.json` (`sha256sum data/processed/<name>/meta.json | cut -c1-16`). Gleicher Hash = identischer Datensatz.
- **Config-Änderung** = nur die Abweichung vom Default in `configs/base.yaml` (z. B. `lr=3e-4, target=logfr`), nicht die ganze Config.
- **Metriken** = MAE z01 in mm und relativer Fr-Fehler in % auf dem **Testsplit**; bei mehreren Seeds Mittelwert ± Std in der Zeile des letzten Seeds oder separate Zusammenfassungszeile.
- **Beobachtung** = ein Satz: was ist aufgefallen (Overfitting, Ausreißer, Laufzeit, Abbruch)?
- **Nächster Schritt** = konkrete Folgeaktion oder „abgeschlossen“.
- Jeder Figur in der Thesis muss eine Zeile hier entsprechen (Quelle = Run + Commit).

## Logbuch

| Run | Datum | Commit | Daten / meta.json-Hash | Config-Änderung | MAE z01 [mm] | rel. Fr-Fehler [%] | Beobachtung | Nächster Schritt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *BEISPIEL* `2026-10-15_cnn-baseline_s0` | 2026-10-15 | `0123abcd4567` | `small` / `9f3b2c1d8e7a6f50` | – (Default `base.yaml`, 50 Epochen) | 0.85 | 1.7 | *Beispielzeile, keine echten Werte.* Val-Loss ab Epoche 30 flach; 3 Ausreißer bei kleinem z01. | Seeds 1, 2 nachziehen; Ausreißer in Scatter-Plot markieren (#12) |

## Zusammenfassungen (pro Experiment-Issue)

<!-- Wenn ein Experiment mit mehreren Runs abgeschlossen ist: eine Zusammenfassungszeile mit Mittelwert ± Std über Seeds und Link zum Issue. -->

| Experiment (Issue) | Runs | MAE z01 [mm] | rel. Fr-Fehler [%] | Entscheidung |
| --- | --- | --- | --- | --- |
| *BEISPIEL* CNN-Baseline (#12) | `…_s0`, `…_s1`, `…_s2` | 0.86 ± 0.04 | 1.7 ± 0.1 | *Beispiel.* Als Baseline für Kapitel 5.2 festgelegt. |
