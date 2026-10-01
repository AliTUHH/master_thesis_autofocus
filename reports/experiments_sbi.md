# Simulation-based inference: amortisierte Posterior q(z01 | Hologramm) mit NPE (data_small, CPU)

Branch `cursor/sbi-npe-z01-6175` (aufgesetzt auf `cursor/spectral-input-ringfit-6175`). Alle Zahlen stammen aus
Läufen in diesem Branch auf `data/processed/small` (HoloForge, 2048 px um 8 gebinnt → 256 px, 400/100/100
Samples, Mg 30 µm, z01 ∈ [50, 300] mm → Fr ∈ [3.0e-3, 1.8e-2], 5 % Gauß-Rauschen auf der Amplitude), CPU mit
2 Threads, `sbi==0.27.0`, `torch 2.10.0`. Rohdaten (`results.json`, `eval_results.json`, Sweeps, Abbildungen)
liegen in `reports/experiments_sbi/`. `docs/notizen/experimente.md` existiert in diesem Branch nicht, deshalb
diese Notiz. Referenzzahlen der Punktschätzer stammen aus `reports/experiments_spectral.md` (gleicher Datensatz,
gleicher Test-Split; die `test_predictions.npz` der Läufe wurden für die Risk-Coverage-Vergleiche nachgeladen).

## 1. Design

* **Parameter** θ = z01 in mm (1-D). **Prior** `Uniform(50, 300) mm`, gelesen aus `meta.json`
  (`config.setup.z01_mm`), also exakt die Verteilung, mit der `NFHRandomDistSetup` die Trainingshologramme
  simuliert hat (`src/sbi/npe.py: prior_from_meta`). Bewusst nicht `log Fr`: ein Uniform-Prior in `log Fr` wäre
  ein anderer Prior (Jacobi-Faktor), und das Beamline-Interface (`Measurement(z01=..., z01_confidence=...)`)
  braucht ohnehin mm. Fr folgt deterministisch aus der Geometrie (`src/utils/physics.py`).
* **Beobachtung x**: Hauptvariante das azimutal gemittelte log-Leistungsspektrum (`radial_profile`, 256 Bins in
  |ξ|², Hann-Fenster) – kompakt, dreh-/spiegelinvariant, 1.2 ms pro Hologramm; Variante 2 das standardisierte
  Hologramm `[1, 256, 256]`.
* **Embedding-Netz**: `RadialProfileEmbedding` (LayerNorm → 128 → 64 → 16, Dropout 0.1; Rumpf identisch zum
  Punktschätzer `RadialProfileMLP`) bzw. `HologramCNNEmbedding` (Faltungsrumpf des `AutofocusCNN`, 32 Merkmale).
* **Dichteschätzer**: Neural Spline Flow (`sbi.neural_nets.posterior_nn("nsf")`, 50 Hidden, 5 Transformationen,
  10 Spline-Bins, z-Scoring von θ und x). MAF und MDN wurden im Sweep mitgetestet (schlechter, Abschnitt 2).
* **Training**: eine NPE-Runde auf den vorsimulierten (θ, x)-Paaren (amortisiert, keine Online-Simulation).
  `train.hdf5` = Training, `val.hdf5` = Validierung/Early-Stopping (`FixedSplitNPE` überschreibt sbis zufälligen
  Split, damit die 400/100-Aufteilung der Punktschätzer erhalten bleibt), Adam 5e-4, Batch 50, Early Stopping
  nach 20 Epochen ohne Verbesserung. **Deep Ensemble**: `train.ensemble_size: 5` trainiert fünf unabhängig
  initialisierte Flows (Seeds 42–46) und mischt sie gleichgewichtet (`sbi EnsemblePosterior`); Grund siehe
  Abschnitt 2.
* **Auswertung** (`src/sbi/evaluate_npe.py`): 1000 Posterior-Samples pro Hologramm; Punktschätzer Mittelwert,
  Median und MAP (Dichte auf einem 1001-Punkt-Gitter über dem Prior, parabolisch verfeinert); zentrale
  Kredibilitätsintervalle; Coverage 50/68/90/95 % mit 95-%-Wilson-Intervallen; Coverage-Kurve + erwarteter
  Coverage-Fehler (ECE); SBC-Ränge mit KS-Test (eigene Implementierung und `sbi.diagnostics.run_sbc`); TARP
  (`sbi.diagnostics.run_tarp`); Spearman/Pearson zwischen Posterior-Std und |Fehler|; Risk-Coverage-Kurve
  (MAE, wenn die unsichersten Hologramme verworfen werden); Misspezifikations-Indikatoren (Leakage-Akzeptanz,
  Posterior-Masse in den äußeren 5 % des Priors); Laufzeit.
* **Beamline-Ausgabe**: `posterior_to_find_focus_init(samples)` → `{"z01": Median, "z01_confidence": halbe
  Breite des zentralen 95-%-Intervalls}`; `find_focus_bounds(init)` liefert `(z01 − c, z01 + c)` wie
  HoloWizards `Measurement.z01_bounds`.

## 2. Hyperparameter-Sweep (Validierungs-Split, 100 Hologramme)

Alle Varianten wurden *nur* auf `val.hdf5` verglichen, der Test-Split blieb unberührt
(`reports/experiments_sbi/sweep_single_flow_val.json`, `sweep_ensemble_val.json`). Einzelne Flows (Ausgangspunkt:
Embedding [256,128]→32 ohne Dropout, NSF 50/5/10, lr 5e-4):

| Variante (einzelner Flow) | Val-MAE Mittel / Median / MAP [mm] | mittlere Std [mm] | Coverage 68 / 95 % | ECE | SBC-KS-p |
|---|---|---|---|---|---|
| Basis (Seed 42) | 15.4 / 15.5 / 15.9 | 24.3 | 0.65 / 0.92 | 0.041 | 0.09 |
| Basis, Seed 1 | 21.6 / 20.1 / 22.0 | 39.4 | 0.85 / 0.92 | 0.108 | 0.16 |
| Basis, Seed 2 | 17.7 / 17.8 / 17.7 | 27.5 | 0.65 / 0.92 | 0.021 | 0.08 |
| Basis, anderer RNG-Strom (Ensemble-Code) | 19.5 / 19.0 / 19.3 | 22.9 | 0.52 / 0.89 | 0.074 | 0.001 |
| lr 1e-3 | 13.4 / 12.6 / 12.6 | 21.0 | 0.60 / 0.93 | 0.045 | 0.35 |
| lr 2e-4 | 21.2 / 21.5 / 21.3 | 35.3 | 0.74 / 0.89 | 0.069 | 0.08 |
| Embedding [128,64]→16, Dropout 0.1 | 14.2 / 13.0 / 13.9 | 21.2 | 0.57 / 0.90 | 0.057 | 0.05 |
| Embedding Dropout 0.2 | 15.7 / 14.7 / 13.4 | 20.4 | 0.64 / 0.88 | 0.037 | 0.35 |
| NSF klein (32/3/8) | 18.5 / 18.8 / 20.1 | 24.3 | 0.69 / 0.88 | 0.025 | 0.007 |
| MAF | 20.8 / 20.7 / 20.5 | 18.4 | 0.58 / 0.85 | 0.072 | 0.06 |
| MDN (5 Komponenten) | 19.9 / 20.0 / 20.1 | 22.9 | 0.67 / 0.87 | 0.038 | 0.23 |
| kein z-Scoring von x | 17.3 / 17.9 / 19.8 | 23.9 | 0.64 / 0.97 | 0.033 | 0.005 |
| Batch 25 | 17.1 / 16.3 / 17.5 | 27.0 | 0.70 / 0.93 | 0.017 | 0.02 |

Die **Seed-Streuung dominiert** (Val-MAE 15–22 mm, Std 23–39 mm bei identischer Konfiguration): mit 400
Trainingsbeispielen ist der Flow stark von der Initialisierung abhängig. Deshalb Deep Ensembles:

| Variante | Val-MAE Mittel / Median / MAP [mm] | mittlere Std [mm] | Coverage 50 / 68 / 90 / 95 % | ECE | SBC-KS-p | ρ(Std, Fehler) |
|---|---|---|---|---|---|---|
| 1 Flow (Basis) | 19.5 / 19.0 / 19.3 | 22.9 | 0.42 / 0.52 / 0.86 / 0.89 | 0.074 | 0.001 | 0.19 |
| Ensemble 5 (Basis) | 16.1 / 15.3 / 15.1 | 27.5 | 0.64 / 0.78 / 0.88 / 0.93 | 0.047 | 0.10 | 0.35 |
| Ensemble 5, lr 1e-3 | 14.0 / 13.0 / 12.8 | 24.1 | 0.55 / 0.74 / 0.88 / 0.93 | 0.031 | 0.02 | 0.29 |
| Ensemble 5, Dropout 0.2 | 14.1 / 13.2 / 12.7 | 23.1 | 0.56 / 0.72 / 0.90 / 0.93 | 0.035 | 0.03 | 0.49 |
| **Ensemble 5, Embedding [128,64]→16, Dropout 0.1** | **13.7 / 12.4 / 11.9** | **21.0** | 0.51 / 0.72 / 0.87 / 0.91 | **0.020** | 0.19 | 0.49 |
| Ensemble 10 (Basis) | 16.3 / 15.0 / 15.1 | 30.2 | 0.65 / 0.80 / 0.90 / 0.94 | 0.073 | 0.04 | 0.46 |

Gewählt (jetzt `configs/sbi_radial.yaml`): Ensemble aus 5 Flows mit kleinerem Embedding + Dropout 0.1 – bester
Val-MAE, kleinste Std, kleinster ECE, unauffällige SBC-Ränge. Ein Ensemble aus 10 Basis-Flows bringt gegenüber 5
nichts mehr (die zusätzlichen Seeds waren schlechter), ein kleineres Netz regularisiert besser als mehr Mitglieder.
Nicht in den Tabellen: `validation_split: random` (sbi-Split über train+val gepoolt) erreicht scheinbar 9.7 mm, ist
aber unfair, weil Validierungshologramme im Training landen – deshalb bleibt der Datei-Split die Voreinstellung.

## 3. Ergebnistabelle (Test-Split, 100 Hologramme)

| Methode | z01 MAE / RMSE / p95 [mm] | mittlere Posterior-Std [mm] | Coverage 68 / 95 % | SBC-KS-p | Inferenz [ms/Hologramm, 2 Threads] | Training |
|---|---|---|---|---|---|---|
| **NPE Radialprofil, Ensemble 5** – Median | **11.05 / 14.68 / 27.4** | 19.6 (Breite 68 %: 28.3, 95 %: 74.2) | 0.58 [0.48, 0.67] / 0.94 [0.88, 0.97] | 0.21 (sbi: 0.19) | 1.2 Profil + 28.2 Sampling (1000) = 29.5; MAP-Gitter +22.6; gebatcht 5.1 | 24.5 s (5 × 51–72 Ep.) |
| NPE Radialprofil, Ensemble 5 – Mittelwert | 11.65 / 15.51 / 30.4 | " | " | " | " | " |
| NPE Radialprofil, Ensemble 5 – MAP | 11.44 / 16.01 / 33.4 | " | " | " | " | " |
| NPE Radialprofil, 1 Flow (Seed 42) – Mittelwert | 13.22 / 17.21 / 29.9 | 19.0 | 0.48 [0.38, 0.58] / 0.87 [0.79, 0.92] | 0.025 | 1.2 + 8.0 = 9.2 | 5.6 s (65 Ep.) |
| NPE Hologramm-CNN, 1 Flow – Median | 17.77 / 22.87 / 42.2 | 31.1 (Breite 95 %: 128) | 0.74 [0.65, 0.82] / 0.96 [0.90, 0.98] | 0.43 | 0.1 + 20.5 (davon CNN 6.2) = 20.6 | 220 s (21 Ep., beste 13) |
| NPE Hologramm-CNN, 1 Flow – Mittelwert | 18.31 / 23.37 / 42.3 | " | " | " | " | " |
| Radialprofil-MLP (Punktschätzer, `log Fr`) | 17.38 / 22.19 / 44.3 | – | – | – | 1.0 Profil + 0.04 | 70 s (30 Ep.) |
| Hologramm-CNN (Punktschätzer, `log Fr`) | 13.04 / 19.17 / 34.6 | – | – | – | 6.8 | 419 s (30 Ep.) |
| 2-Kanal-CNN (Hologramm + Spektrum, Punktschätzer) | 11.42 / 18.55 / 33.1 | – | – | – | 6.3 + 1.4 | 416 s (30 Ep.) |

Weitere Kennzahlen des Haupt-Laufs (`sbi_radial_eval_test.json`): Bias +0.8 mm, Median-|Fehler| 8.3 mm (Median-
Schätzer), relativer Fr-Fehler Median 6.7 % / MAE 9.4 %; Coverage 50 % = 0.44 [0.35, 0.54], 90 % = 0.86 [0.78,
0.91]; ECE 0.076; TARP ATC −0.04 (KS-p 1.0); Leakage-Akzeptanz im Mittel 0.98 (min 0.64); 7 % der Posterior-Masse in
den äußeren 5 % des Priors; `find_focus`-Startwert: z01_confidence im Mittel 44 mm (Median 46 mm), in 99 % der
Testfälle liegt das wahre z01 innerhalb der Grenzen. Parameter: 42 704 (Embedding) + 24 395 (Flow) = 67 099 pro
Mitglied, 335 495 gesamt.

Interpretation:

* Die NPE auf dem Radialprofil ist **deutlich besser als der Punktschätzer auf derselben Eingabe** (Median-MAE
  11.1 vs. 17.4 mm, −36 %; p95 27 vs. 44 mm) und auch besser als das Hologramm-CNN (13.0 mm), bei 17× kürzerem
  Training (25 s vs. 419 s) und einer Latenz von 29 ms pro Hologramm inklusive 1000 Posterior-Samples (gebatcht
  5 ms; CNN-Punktschätzer 6.8 ms). Gegenüber dem
  2-Kanal-CNN (11.4 mm) ist der Unterschied nicht signifikant (ein Seed, 100 Testsamples). Ein Teil des Gewinns
  kommt vom Ensemble (1 Flow: 13.2 mm); der Rest vermutlich daher, dass der Flow die volle (mehrgipflige)
  Verteilung lernt und der Median daraus robuster ist als eine MSE-Regression in `log Fr`, die zwischen Moden
  mittelt.
* Nach z01-Bereichen (Median-Schätzer, MAE in mm; NPE / MLP / CNN-Punktschätzer): 50–100 mm 14.1 / 14.1 / 5.4,
  100–150 mm 10.4 / 16.8 / 10.8, 150–200 mm 10.7 / 18.0 / 9.8, 200–250 mm 12.5 / 13.9 / 17.9, 250–300 mm
  **6.5** / 25.6 / 24.6. Die NPE gewinnt bei großen Abständen (kleines Fr, wenige, weit auseinander liegende
  CTF-Ringe, die das Radialprofil direkt abbildet); bei kleinen Abständen (viele Ringe, Fr-Information in
  hohen Frequenzen) ist das Hologramm-CNN besser. Das spricht für ein kombiniertes Embedding (Abschnitt 7).
* **Mittelwert vs. Median vs. MAP**: Die Posteriors sind häufig mehrgipflig (Abb.
  `sbi_radial_test_posterior_examples.png`: Nebenmoden an Ring-Mehrdeutigkeiten und am Prior-Rand); der
  Mittelwert liegt dann zwischen den Moden. Der Median ist der robusteste Punktschätzer (bester MAE, RMSE und
  p95), der MAP hat den kleinsten Median-|Fehler| (7.9 mm), aber mehr Ausreißer (p95 33 mm).
* **NPE mit CNN-Embedding** (ein Lauf, 21 Epochen): gut kalibriert (ECE 0.023, SBC-KS-p 0.43, TARP-KS-p 1.0),
  aber breit (Std 31 mm = 43 % der Prior-Std) und mit 17.8 mm schlechter als der CNN-Punktschätzer (13.0 mm).
  Mit 400 Hologrammen lernt der Flow über das 363 k-Parameter-Embedding nur eine grobe Posterior; die
  Risk-Coverage-Kurve ist flach (Std kaum informativ, ρ = 0.17, p = 0.10). Ein Ensemble war im CPU-Budget nicht
  drin (220 s pro Mitglied).

## 4. Kalibrierung

Coverage der zentralen Kredibilitätsintervalle (Test, n = 100, 95-%-Wilson-Intervalle):

| Niveau | NPE radial, Ensemble 5 | NPE radial, 1 Flow | NPE CNN |
|---|---|---|---|
| 50 % | 0.44 [0.35, 0.54], Breite 18.5 mm | 0.34 [0.25, 0.44] | 0.53 [0.43, 0.62] |
| 68 % | 0.58 [0.48, 0.67], Breite 28.3 mm | 0.48 [0.38, 0.58] | 0.74 [0.65, 0.82] |
| 90 % | 0.86 [0.78, 0.91], Breite 54.9 mm | 0.80 [0.71, 0.87] | 0.92 [0.85, 0.96] |
| 95 % | 0.94 [0.88, 0.97], Breite 74.2 mm | 0.87 [0.79, 0.92] | 0.96 [0.90, 0.98] |
| ECE | 0.076 | 0.143 | 0.023 |
| SBC KS-p (eigene / sbi) | 0.21 / 0.19 | 0.025 / 0.036 | 0.43 / 0.46 |
| TARP ATC / KS-p | −0.04 / 1.00 | −0.09 / 0.83 | −0.02 / 1.00 |

* Der **einzelne Flow ist überkonfident** (Coverage 68 % → 48 %, 95 % → 87 %; SBC-Ränge nicht gleichverteilt,
  KS-p 0.025, 20 % der Ränge in den äußersten 5-%-Bins statt erwarteter 10 %). Das Ensemble korrigiert das
  weitgehend (14 % in den Randbins): SBC-Ränge sind mit der
  Gleichverteilung verträglich (p 0.21; Histogramm `sbi_radial_test_sbc_histogram.png`), die 95-%-Intervalle
  decken 94 % ab. Bei den mittleren Niveaus bleibt eine leichte Überkonfidenz (68 % → 58 %, Wilson-Intervall
  schließt 0.68 knapp aus; Coverage-Kurve `sbi_radial_test_coverage_curve.png` liegt durchgehend 5–10
  Prozentpunkte unter der Diagonale). Für die Beamline heißt das: das 95-%-Intervall (`z01_confidence` ≈ 45 mm)
  ist verlässlich, ein 1σ-Intervall (Std 20 mm) sollte man um ~20 % aufweiten oder die Konformal-Korrektur
  aus Abschnitt 7 nutzen.
* TARP ist in allen Fällen unauffällig (ATC ≈ 0) – TARP testet mit zufälligen Referenzpunkten und ist bei 1-D
  weniger sensitiv als die SBC-Ränge.
* Der Leakage-Anteil (Masse des Flows außerhalb [50, 300] mm) ist gering (Akzeptanz 0.98 im Mittel), nur für
  Hologramme nahe am Prior-Rand fällt sie auf 0.64; die Normierung ist in `log_prob_grid` für Ensembles
  berücksichtigt.

## 5. Informativität der Unsicherheit (Experiment 2)

* Posterior-Std vs. |Fehler des Mittelwerts|: Spearman ρ = 0.32 (p = 0.001), Pearson r = 0.31
  (`sbi_radial_test_std_vs_abs_error.png`). Die Unsicherheit ist also informativ, aber nicht stark: der eine
  grobe Ausreißer (Fehler 75 mm) hat eine mittlere Std (24 mm).
* **Risk-Coverage** (Hologramme mit der größten Posterior-Std verwerfen; MAE in mm, Median-Schätzer):
  100 % behalten 11.05 → 90 % 10.75 → **80 % 10.61** → 70 % 9.64 → 60 % 8.66 → 50 % 7.95 (Mittelwert-Schätzer:
  11.65 → 11.11 bei 80 % → 8.33 bei 50 %; p95 fällt von 30 auf 17 mm). Beim Verwerfen der unsichersten 20 % sinkt
  der MAE nur um 4 %, erst ab 30–50 % deutlich; die Std trennt vor allem die besten Fälle ab (Std < 15 mm bei
  32 Hologrammen ↔ mittlerer Fehler 7.3 mm, maximal 16 mm).
* Dieselbe Auswahl (NPE-Std ≤ 25.1 mm, 80 % der Hologramme) auf die Punktschätzer angewandt: MLP 17.38 → 17.32 mm,
  CNN 13.04 → 13.48 mm; bei 50 % MLP 15.15, CNN 13.94 mm. Die NPE-Unsicherheit beschreibt also die eigene
  Fehlerverteilung, nicht die Schwierigkeit des Hologramms an sich (Spearman zwischen NPE-Std und CNN-Fehler
  −0.04). Für ein Gating des Hologramm-CNN taugt sie nicht; für den eigenen Schätzer ja.
* Für `find_focus`: Startwert = Median, Suchbereich = 95-%-Intervall (im Mittel ±44 mm, Minimum ±10 mm). Das wahre
  z01 liegt in 99 % der Testfälle im Suchbereich; gegenüber dem Prior (±125 mm) ist der Suchbereich im Mittel
  2.8× kleiner, im besten Viertel der Fälle ≥ 4.4×, in den besten 10 % ≥ 6×.

## 6. Out-of-range-Verhalten (Experiment 4)

`configs/data_small_oor.yaml` erzeugt 30 Testhologramme mit z01 ∈ [300, 350] mm (sonst identisches Setup, Seed
4000); das Ensemble wurde ohne Änderung ausgewertet (`sbi_radial_eval_oor.json`,
`sbi_radial_oor_posterior_examples.png`, `sbi_radial_oor_interval_plot.png`):

| | in-range Test (n = 100) | out-of-range (n = 30, Ensemble 5) | out-of-range, NPE CNN |
|---|---|---|---|
| MAE Median / MAP [mm] | 11.1 / 11.4 | 67.0 / 61.7 (Bias −67 mm) | 75.6 / 49.3 |
| mittlere Std [mm] | 19.6 (Median 20.7) | 23.5 (Median 21.7) | 33.6 |
| Coverage 68 / 95 % | 0.58 / 0.94 | **0.00 / 0.00** | 0.00 / 0.00 |
| SBC-Ränge | gleichverteilt (KS-p 0.21) | alle = 1000 (KS-p 0) | alle = 1000 |
| Masse in den oberen 5 % des Priors | 0.031 | **0.177** | 0.136 |
| Leakage-Akzeptanz | 0.98 | 0.97 | 0.96 |
| ρ(Std, Fehler) | 0.32 | 0.73 | 0.70 |
| wahres z01 in `find_focus`-Grenzen | 0.99 | 0.33 | 0.73 |

* Die Posterior **bleibt im Prior** (Flow-Masse außerhalb [50, 300] mm wird verworfen) und schiebt sich an den
  oberen Rand: 73 % der Mediane > 250 mm, 63 % der MAPs > 280 mm, Moden bei 285–295 mm mit abgeschnittener
  rechter Flanke, meist eine zweite Mode bei 250–260 mm (Ring-Mehrdeutigkeit). Alle 30 Posteriors verfehlen das
  wahre z01 vollständig (Coverage 0, alle Ränge maximal). Die **Posterior-Std verrät das nicht** (21.7 vs. 20.7 mm
  Median); eine NPE ist kein Out-of-distribution-Detektor – sie beantwortet „welches z01 im Prior passt am
  besten", nicht „passt überhaupt ein z01".
* Brauchbarer Indikator ist die **Randmasse**: Anteil der Samples in den oberen 5 % des Priors (> 287.5 mm). Mit
  der Schwelle beim 95-%-Quantil der in-range-Testdaten (0.10) werden 50 % der OOR-Hologramme erkannt bei 5 %
  Fehlalarmen – und die fünf Fehlalarme haben wahre z01 von 287–293 mm, liegen also tatsächlich am Rand.
  Oberes 95-%-Quantil ≥ 292 mm erkennt 47 %, Std > 29 mm nur 30 %, Leakage-Akzeptanz 7 % (nutzlos). Etwa ein
  Viertel der OOR-Fälle (Mediane 155–250 mm) ist von in-range-Posteriors nicht unterscheidbar.
* Konsequenz für die Beamline: der Prior muss den gesamten physikalisch möglichen Bereich abdecken (Prior breiter
  als der Arbeitsbereich wählen, z. B. data_p05: 150–1000 mm), und eine Posterior mit MAP/Median in den äußeren
  5 % des Priors sollte als „außerhalb des Trainingsbereichs" markiert werden, statt als Startwert zu dienen.
  Der kombinierte Fehler des 95-%-Intervalls ist dann die Prior-Grenze selbst.

## 7. Laufzeit, Speicher, Kompatibilität

* Inferenz pro Hologramm (CPU, 2 Threads, Median über 10 Hologramme): Radialprofil 1.2 ms; Ensemble-Sampling
  1000 Samples 28 ms (1 Flow: 8 ms; alle 100 Testhologramme gebatcht: 5 ms/Hologramm); MAP-Gitter 1001 Punkte
  23 ms (1 Flow: 4.6 ms). Weniger Samples lohnen kaum: 250 Samples kosten noch 22 ms (der Ensemble-Overhead
  dominiert), und bei den mehrgipfligen Posteriors schwankt der Median mit 250 Samples im Mittel um 4 mm gegenüber
  1000 Samples. NPE-CNN: Embedding 6.2 ms, Sampling 20 ms.
* Training: Ensemble 5 × ~5 s (51–72 Epochen à 0.07 s) = 25 s; CNN-Variante 220 s (21 Epochen à ~10 s).
  Prozessspeicher: Radialprofil < 1 GB; CNN-Training ≈ 2.1 GB RSS (Stichproben), Evaluation < 1 GB.
* **sbi 0.27.0 / torch 2.10.0**: Training, Sampling (`sample_batched`), `EnsemblePosterior`, Pickling, `run_sbc`,
  `run_tarp` funktionieren ohne Patches. Stolpersteine, die in `src/sbi/npe.py` umgangen sind:
  (i) `DirectPosterior.map()` ist deprecated und wirft → eigener Gitter-MAP; (ii) `log_prob(theta, x)` vervielfacht
  die *rohe* Beobachtung pro θ, bevor das Embedding läuft (`_broadcast_and_align`): beim CNN-Embedding 1001 × 256²
  Aktivierungen ≈ 5 GB RSS → `member_log_prob` berechnet das Embedding einmal und wertet den nflows-Flow direkt
  aus (numerisch identisch, Test `test_npe_ensemble_smoke`; auch für das MLP 19× schneller); (iii) `run_sbc` liegt
  in `sbi.diagnostics`, nicht mehr in `sbi.analysis`; (iv) `summary_writer=` ist deprecated (FutureWarning,
  `tracker` ab 0.27), funktioniert aber; (v) sbis z-Scoring warnt bei Profil-Bins mit Ausreißern (harmlos);
  (vi) sbi zieht `nflows 0.14`, `zuko 1.6.0`, `skorch 1.4.0`, `tabulate` nach – keine Versionskonflikte mit den
  gepinnten Paketen.

## 8. Dateien

* `configs/sbi_radial.yaml` – Hauptkonfiguration (Radialprofil, NSF, Ensemble 5); `configs/sbi_cnn.yaml` –
  CNN-Embedding; `configs/data_small_oor.yaml` – Out-of-range-Testsatz.
* `src/sbi/npe.py` – Prior aus `meta.json`, Embeddings, `FixedSplitNPE`, `fit_npe` (Ensemble), Speichern/Laden,
  Sampling, Gitter-Dichte/MAP, Punktschätzer, `posterior_to_find_focus_init`.
* `src/sbi/calibration.py` – Intervalle, Wilson-Intervalle, Coverage-Kurve/ECE, SBC-Ränge + KS, Risk-Coverage.
* `src/sbi/train_npe.py`, `src/sbi/evaluate_npe.py` – CLIs; `src/sbi/plots.py` – Abbildungen;
  `src/sbi/simulator.py` – `HoloForgeSimulator` (θ → Hologramm) für Online-SBI/SNPE (Schnittstelle, getestet,
  im Training noch nicht benutzt).
* `reports/experiments_sbi/` – `sbi_*_results.json`, `sbi_*_eval_test.json`, `sbi_*_eval_oor.json`,
  `sweep_*_val.json`, Abbildungen (Loss, Posterior-Beispiele, Coverage, SBC, Std vs. Fehler, Intervalle).
* `tests/test_sbi.py` – 14 Tests (< 10 s).

## 9. Offene Punkte / nächste Schritte

* **Mehr Simulationen** sind der größte Hebel: 400 Beispiele erzeugen die Seed-Streuung und die
  Mehrgipfligkeit. `HoloForgeSimulator` erlaubt Online-Simulation (0.4 s pro 256²-Hologramm auf CPU) oder
  sequentielles NPE (SNPE/TSNPE um eine gemessene Beobachtung) ohne neuen Datensatz; auf GPU mit `data_p05`.
* **Konformale Nachkalibrierung**: Die 68-%-Intervalle sind 10 Prozentpunkte zu eng; ein Skalierungsfaktor der
  Intervallbreite aus dem Validierungs-Split (Quantil der normierten Fehler) macht alle Niveaus kalibriert,
  ohne neu zu trainieren. Alternativ Ensemble-Mitglieder mit verschiedenen Architekturen.
* **Kombiniertes Embedding** (Radialprofil-MLP + Hologramm-CNN, ggf. + Ring-Fit-Schätzung als Merkmal), da NPE
  bei großen z01 und CNN bei kleinen z01 gewinnt; 3–5 Seeds für signifikante Vergleiche.
* **OOD-Markierung** in `posterior_to_find_focus_init` (Randmasse-Schwelle) und ein Prior, der breiter ist als der
  Arbeitsbereich; Test der Posterior auf gemessenen P05-Hologrammen (Intensität → Amplitude, Flatfield).
* **Integration in HoloWizard**: `Measurement(z01=init["z01"], z01_confidence=init["z01_confidence"])` als
  Startpunkt von `find_focus`; Vergleich der Konvergenz gegen den heutigen festen Startwert und Suchbereich.
* **Realistische Datensätze** (2048 px, Padding 4): Das Radialprofil skaliert gutmütig (eine FFT + Binning);
  die Zahl der Bins sollte an `N·Fr/2` angepasst werden (siehe `experiments_spectral.md`, Abschnitt Auflösung).
