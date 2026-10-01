# Spektrale Eingaberepräsentationen und CTF-Ring-Fit-Baseline (data_small, CPU)

Branch `cursor/spectral-input-ringfit-6175`. Alle Zahlen stammen aus Läufen in diesem Branch auf
`data/processed/small` (HoloForge, 2048 px um 8 gebinnt → 256 px, 400/100/100 Samples, Mg 30 µm,
z01 ∈ [50, 300] mm → Fr ∈ [3.0e-3, 1.8e-2], 5 % Gauß-Rauschen auf der Amplitude). Trainiert wurde mit der
Referenzkonfiguration (`configs/base.yaml`: AutofocusCNN, `log_fr`, MSE, Adam 1e-3 + Cosine, 30 Epochen,
Early Stopping 10, Seed 42, CPU mit 2 Threads). Die Rohdaten liegen in `reports/experiments_spectral/`
(`repr_*_results.json`) und `reports/ringfit/*/` (`summary.json`, `samples.csv`, Abbildungen).
`docs/notizen/experimente.md` existiert in diesem Branch nicht, deshalb diese Notiz.

## 1. Ergebnistabelle (Test-Split, 100 Hologramme)

| Methode | z01 MAE [mm] | RMSE [mm] | p95 [mm] | Bias [mm] | Fr-Fehler Median / MAE / p95 [%] | Inferenz [ms/Hologramm] | Training |
|---|---|---|---|---|---|---|---|
| Hologramm-CNN (Referenz, Code-Agent) | 13.06 | 19.11 | 33.9 | −5.4 | 6.16 / 7.85 / 18.66 | 11 (1 Thread) | 30 Ep. |
| Hologramm-CNN (Kontrolle, dieser Branch) | 13.04 | 19.17 | 34.6 | −5.52 | 6.06 / 7.82 / 18.60 | 6.8 (b=1) / 11.9 (b=32), 2 Threads | 419 s, 30 Ep. (beste 26) |
| Spektrum-CNN (log\|FFT\|², 256²) | 56.86 | 66.81 | 117.4 | −3.27 | 34.57 / 41.17 / 108.6 | 6.5 + 1.0 FFT | 584 s, Early Stop nach 26 Ep. (beste 16) |
| Spektrum-CNN (pool 2, 128²; Diagnose) | 38.42 | 50.31 | 113.2 | −11.78 | 20.54 / 24.55 / 54.1 | 2.0 + 1.0 FFT | 103 s, 30 Ep. (beste 29, noch fallend) |
| 2-Kanal-CNN (Hologramm + Spektrum) | **11.42** | **18.55** | **33.1** | −3.56 | **5.95 / 7.10 / 17.22** | 6.3 + 1.4 FFT | 416 s, 30 Ep. (beste 28) |
| Radialprofil-MLP (256 Bins in \|ξ\|², 99 k Param.) | 17.38 | 22.19 | 44.3 | −5.38 | 10.07 / 12.82 / 38.5 | 0.04 + 1.0 FFT | 70 s, 30 Ep. (beste 23) |
| Ring-Fit (cos-Template + Brent, Suchbereich z01 ∈ [50, 300] mm) | 38.22 (Median 6.5) | 65.65 | 150.3 | −17.6 | 5.58 / 26.80 / 103.9 | 21 (numpy, 1 Kern) | – |
| Ring-Fit (comb-Template) | 46.32 | 69.02 | 150.0 | +2.1 | 17.08 / 32.16 / 92.5 | 10 | – |

Inferenzzeiten: Modell-Forward aus `results.json` (torch, 2 Threads) plus gemessene Kosten der
Repräsentation (`build_representation(...)(x)` auf 256², Batch 1: Spektrum 0.98 ms, 2-Kanal 1.38 ms,
Radialprofil 1.04 ms). Die Kontrolle reproduziert die Referenz (13.04 vs. 13.06 mm), d. h. Konfiguration,
Daten und Seeds sind äquivalent.

## 2. Befunde

**Hilft das Spektrum als Eingabe?** Nur als Zusatzkanal. Das 2-Kanal-CNN ist in allen Metriken etwas
besser als das Hologramm-CNN (z01-MAE −12 %, p95 −4 %, Fr-MAE −9 %), aber bei einem Seed und 100
Testsamples liegt der Abstand innerhalb der Epoche-zu-Epoche-Schwankung des Validierungs-MAE (±1.5 mm);
ohne 3–5 Seeds ist das nicht signifikant. Das reine Spektrum-CNN lernt praktisch nichts (Val-MAE bleibt
bei 54–60 mm, das entspricht der Vorhersage des Mittelwerts). Ursachen: (i) das Periodogramm ist pro
Pixel exponentialverteilt (Standardabweichung = Mittelwert), nach `log` und Standardisierung dominiert
diese Rauschtextur das Bild; (ii) die Ringe liegen bei 5 % Rauschen nur bis u ≈ 0.03 (|ξ| ≈ 0.17, Radius
≈ 44 px im 256²-Spektrum) über dem Rauschboden, ihr Abstand in Pixeln beträgt Δ|ξ|·N = N·Fr/(2|ξ|) ≈ 4–13 px
und wird vom ersten 4×4-Pooling des AutofocusCNN weitgehend zerstört; (iii) das Spektrum verwirft die
Ortsinformation (Kantenlage, Fresnel-Säume), die das Hologramm-CNN nutzt. Mittelt man das Spektrum vor dem
Logarithmus 2×2 (Rauschvarianz /4), halbiert sich der Fehler (38.4 mm) und die Kurve fällt nach 30 Epochen
noch – das Rauschargument (i) trägt also, reicht aber nicht an das Hologramm heran. Das Radialprofil
(azimutale Mittelung über ~100–800 Pixel pro Bin) ist deutlich besser als das 2-D-Spektrum (17.4 mm bei
99 k Parametern, 2 s/Epoche), bleibt aber hinter dem Hologramm-CNN zurück: die Objektstatistik der
starken Mg-Phantome moduliert das Profil stärker als die CTF-Ringe (siehe Ring-Fit).

**Ring-Fit: wo er funktioniert und wo nicht.** Die Methode ist eine Schwachobjekt-Methode. Regime-Studie
(je 100 Hologramme, Suchbereich z01 ∈ [50, 300] mm, `reports/ringfit/*`):

| Regime | Objekt | Rauschen | Fr-Fehler Median / MAE / p95 [%] | innerhalb 2 / 5 / 10 % | z01 MAE [mm] |
|---|---|---|---|---|---|
| `weak_clean_cos` | Mg 1–2 µm (φ ≈ 0.16–0.33 rad) | – | 0.36 / 0.44 / 0.97 | 100 / 100 / 100 % | **0.85** (Bias −0.80) |
| `weak_noisy_comb` | Mg 1–2 µm | 5 % | 3.03 / 7.85 / 29.4 | 36 / 64 / 78 % | 14.99 |
| `weak_noisy_cos` | Mg 1–2 µm | 5 % | 4.67 / 14.60 / 71.6 | 25 / 53 / 67 % | 27.52 |
| `strong_clean_cos` | Mg 30 µm (φ ≈ 4.9 rad) | – | 2.07 / 25.22 / 77.9 | 50 / 55 / 62 % | 37.21 |
| `small_test_cos` (= data_small) | Mg 30 µm | 5 % | 5.58 / 26.80 / 103.9 | 43 / 49 / 55 % | 38.22 |
| `small_test_comb` | Mg 30 µm | 5 % | 17.08 / 32.16 / 92.5 | 17 / 36 / 45 % | 46.32 |

* *Starke Objekte* (data_small, φ ≈ 4.9 rad): etwa die Hälfte der Hologramme wird auf < 2 % genau
  geschätzt, die andere Hälfte grob falsch – unabhängig vom Rauschen (ohne Rauschen 55 % statt 49 % innerhalb
  5 %). Die Fehlschläge sind keine Harmonischen (nur 9 von 51 bei Faktor 2, 1/2 oder 3), sondern breit
  verteilt (Faktor 0.24–4): Das Leistungsspektrum eines starken Objekts ist kein `S(u)·sin²(πu/Fr)` mehr,
  die Nichtlinearität (`cos φ − 1` wirkt wie ein Absorptionsterm und mischt Objektfrequenzen) füllt die
  CTF-Nullstellen und verschiebt die Phase (gefittete Phase der Erfolge 1.45 ± 1.2 rad statt π). Der
  Template-Score taugt hier nicht als Konfidenz (Erfolg und Fehlschlag haben gleichen Mittelwert ≈ 0.33);
  Übereinstimmung von Template- und Minima-Methode ist ein schwacher Indikator (53 % vs. 27 % Erfolg).
  Keine Fr-Abhängigkeit innerhalb [3e-3, 1.8e-2] (Erfolgsquote 44 / 50 / 56 % in den Terzilen).
* *Schwache Objekte ohne Rauschen*: Median 0.36 %, alle 100 innerhalb 1 %; z01 MAE 0.85 mm. Es bleibt ein
  systematischer Offset von −0.38 %, der mit Fr wächst (−0.13 % → −0.37 % → −0.65 % in den Fr-Terzilen,
  Korrelation −0.64), identisch für Template- und Minima-Methode. Mein eigener Fresnel-Simulator
  (`tests/test_ctf_ringfit.py::simulate_weak_phase_hologram`, gleiche Kernel-Konvention wie HoloForge) zeigt
  keinen solchen Offset (Fehler ±0.5 %, mittelwertfrei). Für das 0.1-mm-Ziel ist das relevant (0.4 % von
  z01 = 0.2–1.2 mm) und muss geklärt werden (HoloForge-Simulationsgitter/Padding/Binning, s. offene Punkte).
* *Schwache Objekte mit 5 % Rauschen*: Nur die ersten Ringe (u ≲ 0.03) liegen über dem Rauschboden
  σ² = 2.5e-3, die Minima-Methode findet in 84 % der Fälle keine ≥ 3 Minima mehr. Das comb-Template
  (log sin² – betont die Nullstellen) ist hier robuster als das cos-Template (64 % vs. 53 % innerhalb 5 %),
  bei starken Objekten umgekehrt (36 % vs. 49 %).
* *Laufzeit*: 10–22 ms pro 256²-Hologramm (numpy/scipy, 1 Kern; 45 ms bei rauschfreien Profilen wegen
  mehr Minima), vergleichbar mit der CNN-Inferenz; kein Training.

**Vergleich mit dem CNN.** Auf data_small ist der Ring-Fit im Median gleich gut wie das CNN (5.6 % vs.
6.1 % Fr-Fehler), aber durch die ~50 % Fehlschläge im Mittel dreimal schlechter (z01 MAE 38 vs. 13 mm). Für
schwache Objekte ohne Rauschen ist er um Größenordnungen besser als jedes trainierte Modell dieser Studie
(0.85 mm) und als einziges Verfahren nahe am Thesis-Ziel von 0.1 mm.

**Ring-Auflösungskriterium (Herleitung + Empirie).** Mit u = |ξ|² (Zyklen²/px²) liegen die CTF-Nullstellen
bei u_n = n·Fr, der Ringabstand in u ist konstant Fr (deshalb wird radial gleichmäßig in u gebinnt: gleiche
Pixelzahl π·Δu·N² pro Bin, äquidistante Ringe). Das diskrete Spektrum eines N-px-Bildes hat die
Frequenzauflösung Δ|ξ| = 1/N (Tukey α = 0.2 ≈ 1.1/N, Hann 2/N), in u also Δu = 2|ξ|/N. Ringe sind auflösbar,
solange Fr > 2|ξ|/N, d. h. für |ξ| < N·Fr/2; die Zahl auflösbarer Ringe ist min(u_max, (N·Fr/2)²)/Fr =
N²·Fr/4, wenn die Fensterauflösung limitiert (`resolvable_frequency_limit`, `max_resolvable_rings`).
Empirisch (synthetische schwache Phasenobjekte, 0–1 % Rauschen, Padding ≥ 1/Fr, 4 Seeds) ist der Fit ab
≈ 10–13 auflösbaren Ringen zuverlässig (N = 192 / Fr = 2e-3: 18 Ringe → Fehler ≤ 1.7 %; N = 128 /
Fr = 3.2e-3: 13 Ringe → ≤ 5 %), bei 6–8 Ringen grenzwertig (N = 128 / Fr = 2e-3, 8 Ringe: je nach Seed
2 % bis 140 % Fehler) und bei ≤ 4 Ringen falsch. Faustregel: N²·Fr ≳ 40. Für data_small (N = 256, Fr ≥ 3e-3: ≥ 49 Ringe) spielt
die Auflösung also keine Rolle – die Fehlschläge dort sind Objekt-, nicht Auflösungsgrenzen.
Zwei Implementierungsfallen wurden dabei gefunden und behoben: rauschfreie Profile mit 10⁻¹⁴-tiefen
Nullstellen kippen den log-Hüllkurvenfit (jetzt Dynamikbegrenzung auf 60 dB), und Kandidatenperioden mit
nur 1–2 Bins pro Periode aliasieren zu Schein-Korrelationen (jetzt ≥ 8 Bins pro Periode bei fr_min).

**Implikationen für das reale P05-Regime (N = 2048, Fr ≈ 1e-4 … 5e-4).** Bei Fr = 1e-4 sind Ringe bis
|ξ| = N·Fr/2 = 0.1024 (u = 0.0105) auflösbar: 104 Ringe, der erste bei |ξ| = √Fr = 0.01 (20 px Radius im
2048²-Spektrum). Bei Fr = 5e-4 ist |ξ|_lim = 0.512 > 0.5, alle 500 Ringe bis u_max = 0.25 sind auflösbar.
Mit Hann-Fenster halbiert sich |ξ|_lim (26 Ringe bei Fr = 1e-4, noch ausreichend). Die Auflösung ist in
diesem Regime also kein Hindernis; limitierend werden Rauschboden (nur Ringe mit S(u) > σ²), Objektstärke
und die Binnung: mit 4N = 8192 Bins hätte Fr = 1e-4 nur 3 Bins pro Periode, `default_n_bins` wählt deshalb
20 000 Bins (bzw. u_max sollte auf ≈ (N·Fr/2)² begrenzt werden). Ein 2048²-Hologramm kostet im Ring-Fit
≈ eine FFT plus Binnung (geschätzt 0.3–0.5 s pro Hologramm in numpy). Für die CNN-Eingabe gilt: Binnen um
s skaliert Fr_eff = s²·Fr, die Labels im Dataset werden nicht umskaliert (das Netz lernt den konstanten
log-Offset, der Ring-Fit teilt durch s²); Crop lässt Fr unverändert und senkt nur die Auflösung auf 1/N_c;
Zoom/Resize ist verboten; Flips/rot90 sind exakt invariant (Test).

## 3. Reproduktion

```bash
export OMP_NUM_THREADS=2
for exp in hologram spectrum spectrum_pool2 hologram_spectrum radial_mlp; do
  python -m src.train --config configs/exp_repr_$exp.yaml --data-dir data/processed/small --run-name repr_$exp
done
python -m src.baseline.ctf_ringfit --data data/processed/small/test.hdf5 --z01-range 50 300 --out reports/ringfit/small_test_cos
python -m src.baseline.ctf_ringfit --data data/processed/small/test.hdf5 --z01-range 50 300 --template comb --out reports/ringfit/small_test_comb
# Regime-Studie: schwache Objekte (configs/data_small_weak.yaml, store.gt_hologram: true) und rauschfreie starke Objekte
python -m src.data.generate_data --config configs/data_small_weak.yaml   # -> data/processed/small_weak/test.hdf5
python -m src.baseline.ctf_ringfit --data data/processed/small_weak/test.hdf5 --hologram-key images/gt_hologram --z01-range 50 300 --out reports/ringfit/weak_clean_cos
python -m src.baseline.ctf_ringfit --data data/processed/small_weak/test.hdf5 --z01-range 50 300 --template comb --out reports/ringfit/weak_noisy_comb
```

## 4. Offene Punkte

* 0.4 %-Offset zwischen Ringperiode und HoloForge-Label bei schwachen Objekten (wächst ∝ Fr): HoloForge-
  Simulationsgitter (`probe_size`, `padding_factor`, Binning um 8) gegen einen direkten Fresnel-Simulator
  mit identischen Labels prüfen; bis dahin ist 0.4 % die Untergrenze jeder physikbasierten Methode auf diesen Daten.
* Mehr Seeds (≥ 3) für Hologramm- vs. 2-Kanal-CNN; Spektrum-CNN mit pool 4 oder ohne frühes 4×4-Pooling.
* Hybrid: Ring-Fit (bzw. Minima-Regression) als Initialisierung/Prior für modellbasiertes Autofokus
  (Fokus-Metrik-Suche um Fr̂ ± 10 %) und als Plausibilitätscheck der CNN-Vorhersage bei schwachen Objekten.
* Radialprofil/Spektrum als Embedding für SBI/NPE: das 1-D-Profil ist kompakt (256 Werte), rotations- und
  flip-invariant und erlaubt einen schnellen Embedding-Netzzweig neben dem Hologramm-CNN.
