# Laborbuch 05 – Etappe 5: Gelernter Autofokus auf den Messhologrammen

| | |
|---|---|
| Datum | 21.10.2026 |
| Branch | `cursor/stage5-messdaten-1091` |
| Code | `src/holofocus/learned/sim2real.py`; Simulatoroptionen für Messdaten in `src/holofocus/dataset/`, `forward.py` (`detector_blur`), `noise.py` (`flat_field_residual`) und `phantoms.py` (Fasern); `configs/dataset/matched-*.yaml`, `configs/learned/spectrum-cnn-wide-hf.yaml`; Warmstart auf Messdaten in `scripts/run_stage3_experiments.py`; `scripts/run_stage5_study.py`, `scripts/make_stage5_figures.py`; `tests/test_sim2real.py` und Erweiterungen der Tests von Datensatz, Phantomen, Rauschen und Vorwärtsmodell |
| Ergebnisse | `results/learned/matched-*` (Modelle, Vorhersagen pro Sample), `results/datasets/matched-*` (Fingerabdrücke der Val- und Test-Splits), `results/measured/cnn`, `b1-p4`, `warm-b1-p4`, `comparison.{csv,json}` (Messdaten), `results/sim2real/` (Störungen, Spektren, Rauschen) |
| Abbildungen | `docs/laborbuch/figures/stage5-*.png`, Zahlen in `stage5-metrics.json` |
| Quellen | [1] Dora et al., Opt. Express 32(7), 10801 (2024), mit dem Zenodo-Archiv 10.5281/zenodo.8349365; [2] Dora et al., Opt. Express 33(4), 6641 (2025) |

## 1 Ziel

Plan-Schritt 8 (`thesis-plan.md`, Phase B), „Messdaten klären und aufbereiten“: Zweck ist die
„Generalisierung und Thesis-Validierung (‚synthetic **and** measurement data‘)“, mit
derselben Vorverarbeitung wie [2]. Messdaten sind die vier Hologramme aus dem Zenodo-Archiv
von [1] (`Examples/data/holograms`), auf denen Etappe 3 die Baseline gerechnet hat:
Spinnenhaar, Zahn, Kaktusnadel und Mg-Draht. Die Aufgabe:

- **Trainingsdaten:** pro Probe ein simulierter Datensatz mit ihrer Energie, ihrer Geometrie
  und dem Detektorpfad der Messläufe aus Etappe 3, $z_{01}$ = Schätzwert ± 5 mm, mit denselben
  Schätzwerten wie dort. Hier in CPU-tauglicher Größe, die großen Versionen als Maxwell-Jobs.
- **Modell:** ein Spektrum-CNN pro Geometrie oder ein geometrie-konditioniertes Modell, wenn
  es deutlich besser ist und auf die CPU passt.
- **Auswertung:** die Schätzungen des Netzes auf den Messhologrammen gegen [2] Table 1 und
  gegen unsere Baseline mit Padding 4: Abstand zu beiden, Intervallbreite, Zeit. Wenn die
  CPU-Zeit reicht, auch der Warmstart von der Netzschätzung.
- **Sim-to-Real-Lücke:** der Fehler des Netzes auf passenden simulierten Testdaten gegen seine
  Abweichung auf den Messdaten; welche Annahmen des Simulators den Unterschied erklären
  könnten: Rauschen, Flatfield, Detektorunschärfe, Phantomtypen.

NPE mit sbi folgt in Etappe 7.

## 2 Ausgangslage

Etappe 4 ist in `main` gemergt
([PR #10](https://github.com/AliTUHH/master_thesis_autofocus/pull/10), Merge-Commit `0f45a7a`),
ebenso das Maxwell-Kit
([PR #11](https://github.com/AliTUHH/master_thesis_autofocus/pull/11), `2dc11b5`); CI grün.
Laborbuch 04, Abschnitt 6 hat festgehalten, was ein Trainingssatz für die Messhologramme
bräuchte: ihre Geometrie, Fasern und dichte Objekte statt der Phantome aus Etappe 2, ein
Modell für die korrelierten Reste des Flatfields und den Vergleich mit [2] Table 1 bei
Padding 4. Die vier Aufbauten (4×-Gitter, 26 µm Pixel, $Fr$ am Schätzwert und über
$\Omega_z$ = Schätzwert ± 5 mm):

| Probe | Energie | $z_{02}$ | Schätzwert | $Fr$ | $Fr$ in $\Omega_z$ | [2] NM / visuell | Padding [2] | Proben [2] |
|---|---|---|---|---|---|---|---|---|
| Spinnenhaar | 11 keV | 19,661 m | 79,8 mm | 1,243·10⁻³ | ±6,3 % | 79,4 / 79,4 mm | 7 | 13 |
| Zahn | 17 keV | 19,652 m | 82,9 mm | 1,998·10⁻³ | ±6,1 % | 80,9 / 81,0 mm | 4 | 11 |
| Kaktusnadel | 17 keV | 19,652 m | 286,1 mm | 6,968·10⁻³ | ±1,8 % | 284,2 / 284,2 mm | 7 | 12 |
| Mg-Draht | 11 keV | 19,661 m | 472,4 mm | 7,510·10⁻³ | ±1,1 % | 470,8 / 470,7 mm | 4 | 9 |

Der Schätzwert ist die Mitte des Anfangssimplex von [2] Table 1, wie in den Messläufen von
Etappe 3 (Laborbuch 03, 3.6). Etappe 3 hat auf allen vier Proben mit Padding 2 gesucht und
auf Spinnenhaar und Zahn zusätzlich mit Padding 4 (`b1-p4`: ungebinntes Hologramm mit dem
4×-Down-Sampling von [2] Table 3 in ASRM, Arrays 2048²). Mit Padding 4 trifft die Suche [2]
beim Spinnenhaar auf 88 µm (79,488 mm, 13 Rekonstruktionen, 45 min mit 3 Threads); beim Zahn
bleibt sie 266 µm darunter (80,634 mm, 11 Rekonstruktionen, 36 min). Für Kaktusnadel und
Mg-Draht fehlte der Lauf mit Padding 4. Bei beiden ist der MFE über $\Omega_z$ fast flach
(0,9–1,2 %), und die Werte der Autoren selbst streuen: Die Beispielskripte von [1]
rekonstruieren dieselben Hologramme bis zu 1,2 mm neben [2] Table 1. Eine Referenz, die
genauer als etwa 1 mm ist, gibt es für die Messdaten nicht (Laborbuch 03, 5.7).

## 3 Was gemacht wurde

### 3.1 Simulatoroptionen für Messdaten (`88a32a6`)

Drei Annahmen der Daten aus Etappe 2 passen nicht zu den Messhologrammen (Laborbuch 04,
Abschnitt 6). Für jede gibt es jetzt eine Option; ohne sie entstehen bitgleich dieselben
Samples wie vorher (geprüft an den Samples 0–2 des Test-Splits von `cpu-4x` aus Etappe 3,
mit dem Code dieser Etappe neu gerechnet):

| Option | Config | Modell |
|---|---|---|
| Detektor-PSF | `simulation.psf_sigma_range` (unbinnte Pixel) | Gauß auf der Intensität vor dem Binning, Transferfunktion $\exp(-2\pi^2\sigma^2 f^2)$ (`forward.detector_blur`); weder [1] noch [2] nennen die PSF |
| Flatfield-Rest | `noise.flat_field_range`, `noise.flat_field_cutoff` | multiplikatives Gaußfeld $r$ mit Spektrum $1/(1 + (f/f_c)^2)$ auf dem 4×-Gitter und RMS $\rho$, nach dem Rauschen: $I\,(1 + r)$ (`noise.flat_field_residual`) |
| Fasern | `phantoms.types: {fiber: …}`, `phantoms.fiber_diameter_range` | eine (70 %) oder zwei gerade Zylinder durch das ganze Sichtfeld, Durchmesser log-gleichverteilt relativ zum Sichtfeld, die Hälfte verjüngt, einige bis zu einer Spitze im Bild; auf dem ganzen Simulationsgitter gerechnet |

PSF-Breite und Flatfield-Rest ziehen aus einem eigenen Zufallsstrom (Spawn-Key
`(crc32(split), i, 1)`). Ein Sample behält deshalb Aufbau, Objekt und Photonenzahl, wenn man
sie einschaltet (Test `test_detector_effects_leave_the_sample_unchanged`). Darauf beruhen die
gepaarten Störungen in 3.7.

### 3.2 Passende Datensätze (`e90171d`, `d9a575b`)

`configs/dataset/matched-<probe>.yaml` übernimmt Energie, $z_{02}$ und Schätzwert aus
`MEASURED_SAMPLES`, $\Delta z_{01}$ gleichverteilt in ±5 mm, und den Detektorpfad der
Messläufe: den P05-Detektor (2048² Pixel zu 6,5 µm) im Blockmittel 4× gebinnt auf 512² Pixel
zu 26 µm; beim Zahn ohne den Schneidrand von 80 Pixeln, also 1888² → 472². Phantome, Rauschen
und Numerik sind die von `cpu-4x`; nur der Aufbau unterscheidet sich von den Daten der Etappe
4 (Test `test_matched_datasets_follow_the_measured_setup`). Die neuen Optionen aus 3.1 sind
hier ausgeschaltet: Die Datensätze sind der Simulator von Etappe 2 und 4 im Aufbau der
Messung.

| Datensatz | Seed | Sichtfeld | Simulationsgitter | Samples (Train/Val/Test) | Erzeugung, 3 Threads | Größe |
|---|---|---|---|---|---|---|
| `matched-spider_hair` | 20261031 | 512² | 4480² | 6000/250/250 | 72 min (0,66 s/Sample) | 4,4 GB |
| `matched-tooth` | 20261032 | 472² | 3087² | 6000/250/250 | 34 min (0,31 s/Sample) | 3,8 GB |
| `matched-cactus_needle` | 20261033 | 512² | 1620² | 6000/250/250 | 16 min (0,14 s/Sample) | 4,3 GB |
| `matched-magnesium_wire` | 20261034 | 512² | 1568² | 6000/250/250 | 13 min (0,12 s/Sample) | 4,3 GB |

Das Simulationsgitter wächst mit dem Fresnel-Kern, also mit $1/Fr$: beim Spinnenhaar 4480²,
beim Mg-Draht 1568². Erzeugt habe ich die Daten mit dem Stand `2dc11b5` in einem eigenen
Worktree, damit sich der Code während der Läufe nicht ändert. Der Simulator von Etappe 5
erzeugt für diese Configs dieselben Samples: Sample 0 jedes Test-Splits, mit dem aktuellen
Code neu gerechnet, ist bitgleich mit dem gespeicherten (`regeneration` in
`results/sim2real/<probe>.json`), ebenso die Samples 0–2 des Test-Splits des Zahns und des
Val-Splits der Kaktusnadel. Die Fingerabdrücke der Val- und
Test-Splits liegen in `results/datasets/matched-<probe>-{val,test}.csv`; damit prüft
`check_cluster_run.py data` auf Maxwell neu erzeugte Splits.

`matched-<probe>-large.yaml` legt für Maxwell 50 000 Trainings-Samples mit eigenem Seed und
derselben Physik dazu (Test `test_large_matched_datasets_have_the_physics_of_the_cpu_sets`;
Anleitung in `docs/maxwell-gpu.md`, 7.1).

### 3.3 Modelle

Pro Probe das Hauptmodell von Etappe 4 (`spectrum-cnn-wide`: Spektrum mit 1024 Intervallen bis
$f^2 = 0{,}125$, 449 826 Parameter, 100 Epochen, β-NLL, Kalibrierung von $\sigma$ auf dem
Val-Split; Laborbuch 04, 3.4 und 3.5), trainiert auf den 6000 Samples des passenden
Datensatzes mit einem Thread. Code und Konfiguration sind dieselben wie in Etappe 4. Die
Modelle auf den passenden Datensätzen sind mit dem Stand `e90171d` trainiert, die
realistischen (3.9) mit `3f7f87e`; dazwischen kamen nur die Config `-hf` und `sim2real.py`
dazu, der Trainingscode ist derselbe.

Für Kaktusnadel und Mg-Draht zusätzlich `spectrum-cnn-wide-hf`: dasselbe Modell mit 2048
Intervallen bis $f^2 = 0{,}25$, also gleicher Intervallbreite. Bei $Fr \approx 7\cdot10^{-3}$
passen bis $f^2 = 0{,}125$ nur 17–18 Perioden der CTF ins Spektrum, beim Spinnenhaar 100 und
in `cpu-4x` 83; bis 0,25 sind es 33–36. Welche Variante für eine Probe gilt, entscheidet
allein der MAE auf ihrem simulierten Val-Split (E4).

### 3.4 Schätzung auf den Messhologrammen

`run_stage5_study.py measured` lädt jedes Hologramm wie in Etappe 3 (`load_measured`:
Flatfield-korrigiert, beim Zahn ohne Schneidrand, Blockmittel 4×) und schätzt mit jedem
trainierten Modell der Probe $\hat z_{01} = z_{01}^\text{guess} + \Delta\hat z_{01}$ mit dem
kalibrierten $\hat\sigma$ und den 68-/95-%-Intervallen. Für jede Referenz (Nelder–Mead und
visueller Wert aus [2] Table 1, das Beispiel aus [1], unsere Suchen mit Padding 4 und 2) stehen
Abstand, $z$-Wert und ob sie in den Intervallen liegt in `results/measured/cnn/<probe>.json`.
Dazu kommen zwei Prüfungen:

- **Binning-Phasen:** dasselbe Hologramm auf allen 16 Verschiebungen des 4×-Blockgitters
  gebinnt. Die Streuung der 16 Schätzungen zeigt, wie stark die Schätzung an der Abtastung
  derselben Messung hängt.
- **Zeit:** Median von 20 Schätzungen (Batch 1, 1 Thread, nach einem Aufwärmaufruf),
  einschließlich Spektrum.

### 3.5 Baseline mit Padding 4 und Warmstart (`62a8606`)

Die Suche mit Padding 4 habe ich für Kaktusnadel und Mg-Draht mit denselben Einstellungen wie
in Etappe 3 nachgeholt (`results/measured/b1-p4/`, 3 Threads). Der Warmstart
(`run_stage3_experiments.py measured --warm-start results/measured/cnn`) startet Nelder–Mead
von $\hat z_{01} \pm 0{,}5$ mm statt von den Enden des Anfangssimplex von [2], verschoben, falls
der Simplex sonst aus $\Omega_z$ ragt; Grenzen, Projektion und Abbruch bei 100 µm Simplexlänge
bleiben wie in [2], gerechnet wird mit Padding 4 wie `b1-p4`. Der Lauf hält Quelle und SHA-256
der Netzschätzung fest. Zeitmessungen laufen mit 3 Threads und höchstens einem weiteren Job
mit einem Thread auf der VM (4 Kerne), wie in Etappe 3.

### 3.6 Rauschen der Messhologramme (`6fc7065`)

`run_stage5_study.py noise` misst das Rauschleistungsspektrum (NPS) in den vier Ecken (256²
native Pixel) jedes Hologramms, jede Ecke durch ihren Mittelwert geteilt, mit Hann-Fenster und
radialem Mittel:

- **volle Auflösung:** $A\,e^{-4\pi^2\sigma^2 f^2} + B$. Für Quantenrauschen, das vor dem
  Szintillator entsteht, ist das NPS $\propto \lvert\text{MTF}\rvert^2$; der erste Term ist
  also das durch eine Gauß-PSF der Breite $\sigma$ verschmierte Rauschen, $B$ der weiße Teil
  (`fit_blurred_white`).
- **4×-Gitter:** $\rho^2 L(f)/\overline L + 1/N$ mit $L = 1/(1 + (f/f_c)^2)$, das Modell des
  Flatfield-Rests aus 3.1 plus weißem Rauschen (`fit_lorentzian`).

Aus den Medianen der vier Ecken folgen die Konstanten in `sim2real.py`: `PSF_SIGMA` = 1,6 px,
`FLAT_FIELD_CUTOFF` = 0,18 und der Rest $\rho$ pro Probe in `MEASURED_MODELS`. Die
Photonenzahl dort ist der weiße Teil aus Laborbuch 02, 5.4 ($16 N_1$ pro 4×-Pixel); die Werte
passen zum weißen Teil der Fits bei voller Auflösung (Spinnenhaar $16/B$ = 6,6·10⁴ gegen
6,8·10⁴). Die Objekte in `MEASURED_MODELS` (Phasenhub, Breite, $\delta/\beta$) stammen aus den
Phasenrekonstruktionen von Etappe 3 und der Zusammensetzung der Proben (Protein,
Hydroxylapatit, Zellulose, Mg) bei ihrer Energie.

### 3.7 Gepaarte Störungen (`d2f9c52`, `df75fb9`)

`run_stage5_study.py perturb` erzeugt den Test-Split einer Probe neu, mit genau einer
geänderten Annahme (`PERTURBATIONS`), und lässt das Netz darauf schätzen:

| Störung | Änderung |
|---|---|
| `psf` | Gauß-PSF mit $\sigma$ = 1,6 unbinnten Pixeln |
| `flat-field` | Flatfield-Rest der Messung ($\rho$ der Probe, $f_c$ = 0,18) |
| `noise` | wie `flat-field`, dazu die Photonenzahl der Messung fest |
| `fiber` | Fasern statt der kompakten Phantome, sonst wie in Etappe 2 (0,5–2 rad, $\delta/\beta$ = 200) |
| `object` | das Objekt der Messung: Fasern mit dessen Phasenhub, Breite und $\delta/\beta$ |
| `combined` | `psf`, `noise` und `object` zusammen |

Seed und Split bleiben; jedes gestörte Sample hat dasselbe $z_{01}$ und, außer bei `noise`
und `combined`, dieselbe Photonenzahl wie das Original, bei `psf`, `flat-field` und `noise`
auch dasselbe Objekt (Test `test_perturbed_samples_pair_with_the_original`). Die
Verschiebung der Schätzung (gestört minus ungestört) ist deshalb eine gepaarte Differenz;
berichtet werden ihr Mittel mit Bootstrap-Intervall, ihr Median und MAE, Bias, Anteil innerhalb
100 µm und Überdeckung auf dem gestörten Split. Die gestörten Splits liegen in
`data/sim2real/` (nicht versioniert), die Vorhersagen in `results/sim2real/<probe>/<störung>/`.

### 3.8 Spektrale Merkmale

`run_stage5_study.py spectra` stellt das Messhologramm in die Verteilung der simulierten
Splits: das radiale Leistungsspektrum bis $f^2 = 0{,}25$ ohne Standardisierung, daraus Boden
(höchste Frequenzen), Abstand des oberen CNN-Bands zum Boden, Überschuss der tiefsten
Frequenzen, Steigung über $f^2$ (Unschärfe macht sie steiler) und die CTF-Modulation
(Korrelation des trendbereinigten Spektrums mit $\sin^2(\pi f^2/Fr)$, über das CNN-Band und
pro Band). Für jedes Merkmal steht das Perzentil des Messhologramms im passenden Test-Split
und in jedem gestörten Split in `results/sim2real/spectra-<probe>.json`, dazu die
Netzeingaben (5/50/95 % pro Intervall).

### 3.9 Realistische Trainingssätze (`3f7f87e`)

`realistic_changes` macht aus der Störung `combined` Bereiche um die gemessenen Werte: PSF
1,2–2,0 px, Photonen und Rest $\rho$ im Faktor 2 um die Werte aus `MEASURED_MODELS`
(Photonen log-gleichverteilt), $f_c$ = 0,18 und das Objekt von `object`. Die Configs
`matched-cactus_needle-realistic.yaml` und `matched-magnesium_wire-realistic.yaml` sind die
passenden Datensätze mit diesen Änderungen und eigenem Seed (Test
`test_realistic_datasets_are_the_matched_sets_with_realistic_changes`). Darauf läuft dasselbe
Modell wie in 3.3 (`spectrum-cnn-wide`), damit sich nur die Trainingsdaten unterscheiden. Der
Lauf heißt `matched-<probe>-realistic` und wird auf seinem eigenen Test-Split, auf dem
passenden Test-Split, auf allen gestörten Splits und auf dem Messhologramm ausgewertet.

### 3.10 Tests

277 Tests bestehen, einer wird übersprungen (nur CUDA); `ruff` ist sauber. In `main` (`2dc11b5`)
waren es 217. Neu sind 61:

| Datei | Neu | Prüft |
|---|---|---|
| `test_sim2real.py` | 23 | Messmodelle für alle vier Proben; jede Störung ändert nur ihre Schlüssel, `combined` ist `psf`, `noise` und `object` zusammen; die realistischen Bereiche enthalten `combined`, die realistischen Configs sind die passenden mit `realistic_changes`; gestörte Samples paaren sich mit dem Original; gepaarte Verschiebungen; das radiale Spektrum von weißem Rauschen ist seine Varianz; das CTF-Merkmal findet $Fr$, Unschärfe macht die Steigung steiler; Perzentilrang; Binning-Phasen; Schätzung eines Hologramms mit Intervallen und Referenzabständen; die Rauschfits finden exakte Spektren wieder, die Eckspektren eine bekannte Unschärfe, weißes Rauschen und Rest |
| `test_dataset_config.py` | 24 | passende Datensätze folgen dem Messaufbau, die großen haben die Physik der CPU-Sätze, jeder Datensatz im Repository hat einen eigenen Seed; Validierung der neuen Schlüssel; die zehn neuen Configs im Test aller Configs |
| `test_phantoms.py` | 8 | Faser als Pixelmittel eines Zylinders (vier Winkel und Radien), verjüngte Faser mit Spitze, Fasern bis in die Leinwand, kompakte Typen auf der Leinwand mit Nullen gepaddet, stabile Typindizes |
| `test_dataset_generator.py` | 3 | Detektoreffekte lassen Aufbau, Objekt und Photonenzahl eines Samples unverändert; der Flatfield-Rest multipliziert das Hologramm; Faser-Samples kreuzen das Sichtfeld |
| `test_forward.py` | 2 | `detector_blur` ist eine normierte Gaußfunktion; die PSF wirkt auf die gepaddete Intensität vor dem Zuschnitt |
| `test_noise.py` | 1 | RMS und Spektrum des Flatfield-Rests |

Die CI rechnet zusätzlich `run_stage5_study.py perturb` mit allen sechs Störungen auf 4 Samples
des Demo-Datensatzes mit dem CI-Modell und erzeugt die Stage-5-Abbildungen im Schnellmodus,
einmal aus diesem Lauf und einmal aus den versionierten Ergebnissen. Messhologramme und
passende Datensätze gibt es in der CI nicht; Abbildungen, die sie brauchen, entfallen dort.

## 4 Entscheidungen und Begründung

**E1 – Ein Modell pro Geometrie, kein geometrie-konditioniertes Modell.** Laborbuch 04,
Abschnitt 6 hatte ein Netz auf der Achse $u = f^2/Fr^\text{guess}$ vorgeschlagen, das alle
Aufbauten kennt. Ich habe mich für vier einzelne Modelle entschieden:

1. *Gleiche CPU-Kosten.* Auch ein konditioniertes Modell braucht Trainingsdaten in allen vier
   Aufbauten, also dieselben 4 × 6000 Samples, eher mehr, damit es zwischen den Aufbauten
   interpoliert. Die Rechenzeit steckt im Simulator, nicht im Training (3.2).
2. *Keine neue Repräsentation.* Die Modelle pro Geometrie nutzen Repräsentation, Architektur
   und Training von Etappe 4 unverändert; deren Ergebnisse (Laborbuch 04, 5.2–5.4) gelten
   weiter. Die $u$-Achse wäre ein neues Modell, das erst synthetisch geprüft werden müsste.
3. *Die Lücke liegt am Simulator.* Die Abweichung auf den Messdaten ist 1,7- bis 6,5-mal so
   groß wie der simulierte Testfehler (5.4), und die gepaarten Störungen verschieben die
   Schätzung um bis zu 2 mm, vor allem über die Objekte (5.5). Ein konditioniertes Modell auf
   denselben Daten erbt diese Lücke.

Für den Einsatz an der Beamline, mit vielen Aufbauten, bleibt das konditionierte Modell der
richtige Weg (Abschnitt 7, Punkt 5).

**E2 – Schätzwert und $\Omega_z$ wie in Etappe 3.** $z_{01}^\text{guess}$ ist die Mitte des
Anfangssimplex von [2] Table 1, $\Omega_z$ = ±5 mm darum. Damit starten Netz, Suche und
Warmstart vom selben Wissen, und die Messläufe von Etappe 3 bleiben vergleichbar.

**E3 – Detektorpfad der Messläufe.** Die Trainingsdaten durchlaufen denselben Pfad wie die
Messhologramme in Etappe 3: Blockmittel 4× auf 512² (Zahn: 472² ohne Schneidrand). Das Netz
sieht also dieselbe Abtastung wie die Baseline mit `p2`. Die Baseline mit Padding 4 rechnet
dagegen auf dem ungebinnten Hologramm (`b1`), wie [2].

**E4 – Auswahl nur auf simulierten Daten.** Welche Variante (Standard oder `-hf`) für eine
Probe gilt, entscheidet der MAE auf ihrem simulierten Val-Split. Die Messhologramme haben
kein wahres $z_{01}$; eine Auswahl nach dem Abstand zu [2] würde [2] als Wahrheit behandeln und
das Ergebnis auf vier Hologrammen schönen. Alle Varianten stehen trotzdem in den
Ergebnisdateien.

**E5 – Zwei Referenzen, keine Wahrheit.** Verglichen wird mit dem Nelder–Mead-Ergebnis von [2]
Table 1 (Hauptreferenz) und mit unserer Suche mit Padding 4. Beide sind Schätzungen desselben
Verfahrens; sie liegen beim Spinnenhaar 88 µm, beim Zahn 266 µm auseinander (Abschnitt 2), und
die Werte der Autoren streuen um bis zu 1,2 mm. Die Erfolgsschwelle von 100 µm (Laborbuch 04,
Abschnitt 2) lässt sich auf Messdaten deshalb nicht prüfen; ich berichte Abstände und
$z$-Werte, und den Anteil innerhalb 100 µm nur auf simulierten Daten.

**E6 – Konstanten aus dem Median der Ecken.** Der Median der vier Ecken ist robuster als der
Fit des mittleren Spektrums: Beim Zahn liegt eine Ecke bei 6,5 % Rest, und das mittlere
Spektrum ergibt 4,9 % statt des Medians 3,9 %. Beim Mg-Draht liegt der Draht in den Ecken
(Mittelwerte 1,06–1,52), seine Fits taugen nicht; sein Rest ist auf 3 % gesetzt, zwischen
Kaktusnadel (2,6 %) und Zahn (3,9 %). Die PSF-Breite ist für alle Proben dieselbe, da alle mit
demselben Detektor gemessen sind.

**E7 – Eine Annahme nach der anderen, gepaart.** Jede Störung ändert genau eine Annahme, und
alle teilen Seed, $z_{01}$ und, wo möglich, Objekt und Photonenzahl mit dem Original. So ist
die Verschiebung der Schätzung der Effekt dieser einen Annahme, ohne das Rauschen eines neuen
Splits. `combined` zeigt, ob sich die Effekte addieren.

**E8 – Das realistische Modell wird nie ausgewählt.** Sein Val-Split hat andere Annahmen, sein
MAE ist mit dem der anderen Varianten nicht vergleichbar. Außerdem nutzen seine
Trainingsdaten Eigenschaften, die ich an den Messhologrammen selbst gemessen habe (Rauschen,
Objekt), nicht aber deren $z_{01}$-Referenzen. Es steht deshalb neben dem ausgewählten Modell,
als Antwort auf die Frage, ob realistischere Trainingsdaten die Lücke schließen.

**E9 – Eingefrorener Code für lange Läufe.** Datensätze, Training und die Suchen mit Padding 4
liefen aus Worktrees mit festem Stand (`2dc11b5` für die passenden Datensätze, `e90171d` für
deren Training, `3f7f87e` für die realistischen Sätze, ihr Training und die Suchen), damit
Änderungen während der Läufe nichts verschieben. Schätzungen und Störungen liefen aus dem
Arbeitsverzeichnis mit `e032094`, dessen `src/` dasselbe ist wie in `3f7f87e`; nur die
Rauschfits, die gestörten Splits von Kaktusnadel und Mg-Draht und deren Vorhersagen stammen
von `bd262bc` und `44f9631`, deren `src/` sich davon nur durch das Fehlen von
`realistic_changes` unterscheidet. Die Spektren habe ich mit `902756e` neu geschrieben, weil
ihr Protokoll vorher keinen Commit enthielt; die Merkmale sind dieselben. Jeder Lauf hält
seinen Commit fest.

## 5 Ergebnisse

Kurz: Auf passenden simulierten Daten liest das Netz $z_{01}$ in allen vier Aufbauten auf
0,25–0,30 % genau, in 2–4 ms pro Hologramm (5.1, 5.2). Auf den Messhologrammen liegen alle vier
Schätzungen über [2] Table 1, um 0,42 bis 4,75 mm, und nur beim Zahn ist das Netz näher an [2]
als der Schätzwert (5.2). Die Abweichungen sind 1,7- bis 6,5-mal so groß wie der simulierte
Testfehler; mit den gemessenen Eigenschaften im Simulator wären sie gewöhnlich (5.4). Den
größten Teil der Lücke erklären die Objekte, einen kleineren der Flatfield-Rest, die PSF nicht
(5.5). Die simulierten Objekte der Messung löschen dabei die CTF-Modulation aus, die echten
nicht (5.6); auf realistischen Trainingssätzen lernt das Netz deshalb wenig oder nichts (5.7).
Der Warmstart vom Netz endet bei allen vier Proben höchstens 38 µm neben der kalten Suche mit
Padding 4; Rekonstruktionen spart er nur, wo die Netzschätzung nahe am Minimum liegt, beim
Spinnenhaar 5 von 13 (5.3).

### 5.1 Passende simulierte Test-Splits

Je 250 Samples (`results/learned/matched-<probe>/test/`), Fehler in µm; zum Vergleich das
Hauptmodell von Etappe 4 auf `cpu-4x` ($z_{01}$ = 100 mm, 10 000 Trainings-Samples):

| Probe | Modell | Val-MAE | Test-MAE | Median | innerhalb 100 µm | Überdeckung 68/95 % | Test-MAE / $z_{01}$ | beste Epoche |
|---|---|---|---|---|---|---|---|---|
| Spinnenhaar | Standard | 253 | 241 | 168 | 30 % | 77/97 % | 0,30 % | 44 |
| Zahn | Standard | 237 | 209 | 132 | 41 % | 88/96 % | 0,25 % | 53 |
| Kaktusnadel | Standard | 731 | 735 | 470 | 10 % | 74/96 % | 0,26 % | 24 |
| Kaktusnadel | `-hf` | 811 | 798 | 478 | 14 % | 78/94 % | 0,28 % | 59 |
| Mg-Draht | Standard | 1204 | 1273 | 760 | 11 % | 66/95 % | 0,27 % | 7 |
| Mg-Draht | `-hf` | 1261 | 1219 | 711 | 10 % | 74/92 % | 0,26 % | 16 |
| Etappe 4, `cpu-4x` | Standard | 190 | 240 | 158 | 34 % | 70/91 % | 0,24 % | 67 |

Der Schätzwert allein hat einen MAE von 2500 µm (gleichverteilt in ±5 mm).

1. **Der Fehler wächst mit $z_{01}$, relativ ist er überall gleich.** In allen fünf Geometrien
   liegt der Test-MAE bei 0,25–0,30 % von $z_{01}$. Wegen
   $Fr \propto z_{01}/(z_{02}\,z_{12})$ mit $z_{12} = z_{02} - z_{01}$ ist bei festem $z_{02}$
   $\mathrm{d}\ln Fr = \frac{z_{02}}{z_{12}}\,\mathrm{d}\ln z_{01} \approx \mathrm{d}\ln z_{01}$
   (Faktor 1,004–1,025); das Netz liest also die Fresnelzahl auf etwa 0,3 % genau, unabhängig
   davon, wie viele CTF-Perioden ins Spektrum passen (17 beim Mg-Draht, 100 beim
   Spinnenhaar). 100 µm sind beim Spinnenhaar 0,13 % von $z_{01}$, beim Mg-Draht 0,02 %;
   deshalb liegen bei Kaktusnadel und Mg-Draht nur 10–11 % der Samples innerhalb 100 µm.
2. **Mehr Frequenzen helfen nicht.** Das Spektrum bis $f^2 = 0{,}25$ (`-hf`, doppelt so viele
   CTF-Perioden) ist bei der Kaktusnadel auf Val- und Test-Split schlechter (811 gegen 731 µm
   bzw. 798 gegen 735 µm). Beim Mg-Draht ist es auf dem Val-Split schlechter (1261 gegen
   1204 µm), auf dem Test-Split etwas besser (1219 gegen 1273 µm); beide Unterschiede liegen
   in der Streuung zwischen Seeds aus Etappe 4 (etwa ±6 %, Laborbuch 04, 5.2). Nach E4 gilt
   für alle Proben das Standardmodell.
3. **Bei großem $z_{01}$ reichen 6000 Samples nicht.** Bei Mg-Draht und Kaktusnadel liegt die
   beste Epoche bei 7 bzw. 24 von 100. Danach wird das Netz auf dem Val-Split überkonfident:
   Bis Epoche 100 steigt die Val-NLL von 8,6 auf 19,7 (Mg-Draht) bzw. von 8,2 auf 15,3
   (Kaktusnadel), die 68-%-Intervalle decken nur noch 21 bzw. 33 %, und der Val-MAE wird
   schlechter (1204 → 1258 bzw. 731 → 798 µm; Abb. `stage5-training.png`). Bei Spinnenhaar
   und Zahn bleibt der MAE nach der besten Epoche fast gleich. Auswahl der Epoche nach Val-MAE
   und Kalibrierung von $\sigma$ fangen das ab (Faktor 1,0–1,6). In Etappe 4 fiel der Val-MAE
   mit der Datenmenge wie $n^{-0{,}34}$ (Laborbuch 04, 5.2); gilt das auch hier, halbieren
   56 000 Samples den MAE etwa. Dafür sind die großen Datensätze auf Maxwell da
   (`docs/maxwell-gpu.md`, 7.1).
4. **Die Intervalle stimmen auf simulierten Daten.** Die 68-%-Intervalle decken 66–88 %, die
   95-%-Intervalle 92–97 % der Test-Samples.

### 5.2 Schätzungen auf den Messhologrammen

`results/measured/cnn/<probe>.json`, ausgewähltes Modell (Standard, E4; Abb.
`stage5-measured.png`). Abstände in µm als Schätzung minus Referenz, in Klammern
$z$ = Abstand/$\hat\sigma$; Binning-Phasen als Standardabweichung (Spannweite) der 16
Schätzungen in µm; Zeit pro Schätzung mit einem Thread:

| Probe | Schätzung | $\hat\sigma$ | Breite 68/95 % | − [2] NM | − Padding 4 | − Padding 2 | − [1] | Binning-Phasen | Zeit |
|---|---|---|---|---|---|---|---|---|---|
| Spinnenhaar | 79,818 mm | 211 | 420/827 | +418 (1,98) | +330 (1,56) | −373 (−1,77) | −132 (−0,63) | 8 (32) | 2,3 ms |
| Zahn | 81,643 mm | 601 | 1195/2355 | +743 (1,24) | +1009 (1,68) | +1009 (1,68) | −65 (−0,11) | 30 (101) | 4,0 ms |
| Kaktusnadel | 288,953 mm | 1545 | 3073/6057 | +4753 (3,08) | +4024 (2,60) | +4415 (2,86) | +3537 (2,29) | 91 (292) | 2,4 ms |
| Mg-Draht | 473,613 mm | 2052 | 4081/8044 | +2813 (1,37) | +2385 (1,16) | +3166 (1,54) | +3098 (1,51) | 77 (261) | 2,4 ms |

1. **Alle vier Schätzungen liegen über [2]**, um 0,42, 0,74, 4,75 und 2,81 mm. [2] liegt
   beim Zahn und beim Mg-Draht im 95-%-Intervall des Netzes, beim Spinnenhaar knapp außerhalb
   ($z$ = 1,98) und bei der Kaktusnadel deutlich außerhalb ($z$ = 3,08). Unsere Suche mit
   Padding 4 liegt nur bei der Kaktusnadel außerhalb des 95-%-Intervalls.
2. **Gegenüber dem Schätzwert hilft das Netz nur beim Zahn.** Der Schätzwert, die Mitte des
   Anfangssimplex von [2], liegt 0,4, 2,0, 1,9 und 1,6 mm über [2]. Beim Zahn rückt das Netz
   1,26 mm näher an [2]; beim Spinnenhaar bleibt es beim Schätzwert (+18 µm), bei Kaktusnadel
   und Mg-Draht entfernt es sich um weitere 2,85 bzw. 1,21 mm. Auf den simulierten
   Test-Splits ist es dagegen 2- bis 12-mal genauer als der Schätzwert (5.1).
3. **Das Netz zeigt seine Unsicherheit meist an.** Bei Zahn, Kaktusnadel und Mg-Draht ist
   $\hat\sigma$ auf dem Messhologramm 1,7- bis 1,9-mal so groß wie der Median auf dem
   Test-Split. Beim Spinnenhaar ist es das 0,6-Fache; dort ist das Netz sicherer als auf einem
   typischen simulierten Sample, und [2] liegt knapp außerhalb des 95-%-Intervalls.
4. **Die Referenzen streuen selbst.** Das Beispiel aus [1] liegt 0,55, 0,81, 1,22 und
   −0,28 mm neben [2]. Beim Spinnenhaar und beim Zahn liegt das Netz näher an [1] als an [2]
   (−132 und −65 µm). Unsere Suchen mit Padding 2 und 4 liegen bei Kaktusnadel und Mg-Draht
   0,39 bzw. 0,78 mm auseinander, weil der MFE dort fast flach ist (5.3).
5. **Die Schätzung hängt kaum an der Abtastung.** Über die 16 Binning-Phasen streut sie um
   8–91 µm, 4–6 % von $\hat\sigma$. Die Abweichung zu [2] ist also keine Frage, wie dieselbe
   Messung abgetastet wird.
6. **Varianten:** `-hf` schätzt bei der Kaktusnadel fast dasselbe (+5 µm), beim Mg-Draht
   1,7 mm mehr (475,323 mm). Die realistischen Modelle stehen in 5.7.
7. **Zeit:** 2,3–4,0 ms pro Schätzung mit einem Thread, einschließlich Spektrum. Der Zahn
   braucht länger, zum größeren Teil wegen der FFT auf 472² statt 512² Pixeln (allein 2,4
   statt 1,4 ms).

### 5.3 Baseline mit Padding 4 und Warmstart

Kalt (`results/measured/b1-p4/`, Anfangssimplex von [2]) und warm (`warm-b1-p4/`, Netzschätzung
± 0,5 mm), beide auf dem ungebinnten Hologramm mit Padding 4 und 3 Threads; $z_{01}$ in mm, in
Klammern der Abstand zu [2] in µm, Zeit für den ganzen Autofokus:

| Probe | [2] NM | kalt | Rekonstruktionen | Zeit | Warmstart | Rekonstruktionen | Zeit | warm − kalt |
|---|---|---|---|---|---|---|---|---|
| Spinnenhaar | 79,4 | 79,488 (+88) | 13 | 45 min | 79,505 (+105) | 8 | 28 min | +18 µm |
| Zahn | 80,9 | 80,634 (−266) | 11 | 36 min | 80,643 (−257) | 10 | 37 min | +9 µm |
| Kaktusnadel | 284,2 | 284,928 (+728) | 13 | 45 min | 284,890 (+690) | 14 | 53 min | −38 µm |
| Mg-Draht | 470,8 | 471,228 (+428) | 12 | 42 min | 471,238 (+438) | 12 | 42 min | +10 µm |

Spinnenhaar und Zahn kalt stammen aus Etappe 3 (Laborbuch 03, 5.7), Kaktusnadel und Mg-Draht
kalt sind neu. Beim Warmstart der Kaktusnadel lief die letzte Rekonstruktion neben der
Testsuite, deren Cluster-Tests `OMP_NUM_THREADS` auf alle Kerne setzen; sie dauerte 405 statt im
Mittel 215 s, ohne das wären es etwa 50 min. [2] rechnet auf der GPU, nach eigenen Angaben 6 s
pro Rekonstruktion bei Padding 4 und 16 s bei Padding 7 (Laborbuch 03, 5.9). Mit den
Rekonstruktionen aus [2] Table 1 sind das 3,5 min für das Spinnenhaar (13, Padding 7), 1,1 min
für den Zahn (11, Padding 4), 3,2 min für die Kaktusnadel (12, Padding 7) und 0,9 min für den
Mg-Draht (9, Padding 4).

1. **Kalt landen Kaktusnadel und Mg-Draht 0,73 bzw. 0,43 mm über [2].** Mit Padding 2 waren es
   +0,34 und −0,35 mm (Etappe 3). Der MFE ist dort fast flach: Eine Parabel durch die
   Rekonstruktionen im Umkreis von 1,6 mm um das Minimum hat, relativ zum Minimum, die Krümmung
   0,03 bzw. 0,02 %/mm², beim Spinnenhaar 4,9 und beim Zahn 1,5 %/mm². 0,5 mm neben dem
   Minimum liegt der MFE bei Kaktusnadel und Mg-Draht also weniger als 10⁻⁴ über seinem
   Minimum, und wo die Suche in diesem Tal anhält, hängt an Details wie dem Padding. Genauer als
   auf etwa 1 mm legt der MFE $z_{01}$ für diese beiden Proben nicht fest; dazu passt, dass die
   Werte der Autoren dort bis zu 1,2 mm streuen (Abschnitt 2).
2. **Der Warmstart spart Rekonstruktionen, wenn die Netzschätzung nahe am Minimum liegt.** Beim
   Spinnenhaar liegt sie 0,33 mm über dem kalten Ergebnis; der Warmstart braucht 8 statt 13
   Rekonstruktionen, 28 statt 45 min. Beim Zahn liegt sie 1,0 mm darüber, und er braucht 10
   statt 11. Eine davon ist doppelt: Der Simplex erreicht 80,1433 mm auf zwei Wegen, die sich in
   den letzten Bits unterscheiden, und `search.py` erkennt Wiederholungen nur an bitgleichen
   Werten. Bei Kaktusnadel und Mg-Draht liegt die Netzschätzung 4,0 bzw. 2,4 mm über dem
   kalten Ergebnis, weiter weg als der Schätzwert (je 1,17 mm); der Warmstart braucht 14
   statt 13 bzw. ebenfalls 12 Rekonstruktionen, beim Mg-Draht wieder mit einer doppelten
   (472,1133 mm).
3. **Wo der Warmstart endet.** Alle vier Warmstarts enden nahe am kalten Ergebnis (+18, +9,
   −38 und +10 µm), auch wenn sie bis zu 4 mm daneben starten. Das Ergebnis mit Padding 4
   hängt also kaum vom Start ab; dass Padding 2 bei Kaktusnadel und Mg-Draht 0,39 bzw.
   0,78 mm tiefer endet (Punkt 1), liegt am Padding, nicht an der Suche.
4. **Zeit.** Eine Schätzung des Netzes dauert 2,3–4,0 ms (5.2), eine Suche mit Padding 4 auf der
   CPU 36–45 min mit 3 Threads (rund 210 s pro Rekonstruktion), [2] auf der GPU nach eigenen
   Angaben geschätzt 0,9–3,5 min. Der Warmstart kostet so viel wie die Suche, die er abkürzt;
   das Netz spart dabei 5, 1, −1 und 0 Rekonstruktionen (Spinnenhaar, Zahn, Kaktusnadel,
   Mg-Draht), zusammen 5 von 49.

### 5.4 Testfehler gegen Abweichung

Die Abweichung auf dem Messhologramm in der Verteilung der Fehler auf dem passenden
Test-Split (`results/measured/comparison.csv`, Abb. `stage5-sim2real.png`). Perzentil: Anteil
der Test-Samples mit kleinerem Fehler. Fehler in µm:

| Probe | Test-MAE | Median | \|Netz − [2]\| | Perzentil | \|Netz − Padding 4\| | Perzentil | \|Netz − [2]\| / Test-MAE |
|---|---|---|---|---|---|---|---|
| Spinnenhaar | 241 | 168 | 418 | 88 % | 330 | 80 % | 1,7 |
| Zahn | 209 | 132 | 743 | 95 % | 1009 | 97 % | 3,6 |
| Kaktusnadel | 735 | 470 | 4753 | 99,6 % | 4024 | 99 % | 6,5 |
| Mg-Draht | 1273 | 760 | 2813 | 88 % | 2385 | 82 % | 2,2 |

1. **Einzeln betrachtet fallen nur Kaktusnadel und Zahn auf.** Beim Spinnenhaar und beim
   Mg-Draht läge die Abweichung zu [2] auf simulierten Daten beim 88. Perzentil, wäre also
   ein gewöhnlicher Fehler. Beim Zahn liegt sie am Rand (95. bzw. 97. Perzentil); bei der
   Kaktusnadel hat nur eines der 250 Test-Samples einen größeren Fehler (5,27 mm).
2. **Gemeinsam fällt die Richtung auf.** Auf den Test-Splits ist der Bias klein (−47 bis
   +226 µm), das Vorzeichen eines Fehlers also etwa zufällig; dass alle vier dasselbe
   Vorzeichen haben, hätte dann eine Wahrscheinlichkeit von etwa 1/8. Für sich ist die
   Richtung bei vier Hologrammen schwach belegt; sie passt aber zu den Objekten in 5.5: `fiber`,
   `object` und `combined` schieben die Schätzung bei allen vier Proben nach oben.
3. **Mit den gemessenen Eigenschaften wären die Abweichungen gewöhnlich.** Unter den Fehlern auf
   dem Split `combined` (5.5) läge die Abweichung zu [2] beim 37. (Spinnenhaar), 10. (Zahn), 73.
   (Kaktusnadel) und 57. Perzentil (Mg-Draht; `measured_percentile_in_combined` in
   `stage5-metrics.json`, Abb. `stage5-sim2real.png`). Der passende Test-Split unterschätzt den
   Fehler auf den Messdaten, weil ihm die gemessenen Eigenschaften fehlen; welche das sind,
   zerlegt 5.5.

### 5.5 Gepaarte Störungen

Verschiebung der Schätzung durch eine Annahme der Messung (gestört minus ungestört, 3.7),
gemittelt über die 250 gepaarten Samples, mit 95-%-Bootstrap-Intervall, in µm
(`results/sim2real/<probe>.json`, Abb. `stage5-perturbations.png`):

| Störung | Spinnenhaar | Zahn | Kaktusnadel | Mg-Draht |
|---|---|---|---|---|
| `psf` | +24 (+2 bis +47) | +7 (−29 bis +37) | +57 (−31 bis +142) | +30 (−76 bis +138) |
| `flat-field` | −133 (−211 bis −53) | −264 (−397 bis −137) | +933 (+738 bis +1134) | +571 (+382 bis +760) |
| `noise` | −120 (−198 bis −40) | −278 (−408 bis −152) | +1017 (+819 bis +1224) | +697 (+507 bis +884) |
| `fiber` | +629 (+446 bis +813) | +541 (+360 bis +723) | +985 (+733 bis +1240) | +1594 (+1305 bis +1883) |
| `object` | +768 (+551 bis +981) | +1646 (+1400 bis +1896) | +1108 (+692 bis +1542) | +1152 (+785 bis +1516) |
| `combined` | +366 (+170 bis +562) | +2034 (+1817 bis +2250) | +1665 (+1241 bis +2097) | +1019 (+677 bis +1359) |

MAE in µm und Überdeckung der 68-%-Intervalle auf den gestörten Splits:

| Störung | Spinnenhaar | Zahn | Kaktusnadel | Mg-Draht |
|---|---|---|---|---|
| keine | 241 / 77 % | 209 / 88 % | 735 / 74 % | 1273 / 66 % |
| `psf` | 251 / 76 % | 226 / 86 % | 829 / 67 % | 1528 / 62 % |
| `flat-field` | 436 / 61 % | 697 / 53 % | 1654 / 47 % | 2034 / 59 % |
| `noise` | 421 / 64 % | 689 / 58 % | 1718 / 45 % | 2005 / 61 % |
| `fiber` | 1224 / 26 % | 1102 / 31 % | 2080 / 38 % | 2734 / 42 % |
| `object` | 1381 / 33 % | 2211 / 14 % | 3120 / 35 % | 2984 / 32 % |
| `combined` | 1123 / 35 % | 2337 / 9 % | 3375 / 31 % | 2811 / 43 % |

1. **Die PSF verschiebt kaum.** Mit der gemessenen PSF (1,6 px) bleibt die Schätzung im Mittel
   fast, wo sie war (+7 bis +57 µm; nur beim Spinnenhaar schließt das Intervall die Null knapp
   aus); der MAE steigt um 4–20 %. Die Unschärfe dämpft das Spektrum zu hohen Frequenzen hin,
   die Perioden der CTF verschiebt sie nicht. Detektorunschärfe erklärt die Abweichung also
   nicht.
2. **Der Flatfield-Rest verschiebt, mit einem Vorzeichen je nach Aufbau.** Bei Spinnenhaar und
   Zahn schiebt er die Schätzung nach unten (−133 bzw. −264 µm), bei Kaktusnadel und Mg-Draht
   nach oben (+933 bzw. +571 µm); der MAE steigt um den Faktor 1,6–3,3. Die beiden Paare
   unterscheiden sich in $Fr$ um den Faktor 3,5–6 und damit in der Zahl der CTF-Perioden im
   Spektrum (100 bzw. 63 gegen 17–18, 3.3); woran das Vorzeichen hängt, habe ich nicht weiter
   untersucht. Die feste Photonenzahl der Messung (`noise`) ändert daran wenig (−14 bis
   +126 µm gegenüber `flat-field`).
3. **Die Objekte verschieben am stärksten, immer nach oben.** Fasern mit den Parametern von
   Etappe 2 (`fiber`) schieben die Schätzung um +0,54 bis +1,59 mm, das Objekt der Messung
   (`object`) um +0,77 bis +1,65 mm. Der MAE steigt unter `object` auf 1,4–3,1 mm, bei
   Kaktusnadel und Mg-Draht über den des Schätzwerts allein (2,5 mm). Was das Netz an den
   kompakten Phantomen von Etappe 2 gelernt hat, gilt für Fasern nicht; welche Objekte im
   Training vorkommen, wiegt schwerer als jede Annahme über den Detektor.
4. **Die Effekte addieren sich nicht.** `combined` (PSF, Rauschen und Objekt) schiebt um +0,37
   bis +2,03 mm. Die Summe der drei einzelnen Verschiebungen wäre +0,67 (Spinnenhaar), +1,38
   (Zahn), +2,18 (Kaktusnadel) und +1,88 mm (Mg-Draht); unter `combined` ist die Verschiebung
   beim Zahn größer als diese Summe, bei den anderen drei Proben kleiner.
5. **Das Netz zeigt einen Teil der Unsicherheit an.** Unter `combined` wird $\hat\sigma$ im
   Median 1,3- bis 2,3-mal so groß wie auf dem ungestörten Split, der MAE aber 2,2- bis 11-mal;
   die 68-%-Intervalle decken nur noch 9–43 %. Auf den Messhologrammen ist $\hat\sigma$ 1,7- bis
   1,9-mal so groß wie der Median auf dem Test-Split, außer beim Spinnenhaar (5.2, Punkt 3).
6. **Richtung und Größe passen zur Messung, im Einzelnen nicht.** Unter `combined` schiebt die
   Schätzung bei allen vier Proben nach oben, wie auf den Messhologrammen (5.2). Beim
   Spinnenhaar ist die Verschiebung etwa so groß wie die Abweichung von [2] (+0,37 gegen
   +0,42 mm), beim Zahn größer (+2,03 gegen +0,74 mm), bei Kaktusnadel und Mg-Draht kleiner
   (+1,67 gegen +4,75 mm bzw. +1,02 gegen +2,81 mm). Die Annahmen über die Messung erklären die
   Lücke also der Richtung und der Größenordnung nach, vor allem über die Objekte. Dass die
   simulierten Objekte dabei die CTF bei drei der vier Proben stärker verwischen als die echten,
   zeigen die Spektren (5.6).

### 5.6 Spektrale Merkmale

Perzentil des Messhologramms in den simulierten Splits (Anteil der Samples mit kleinerem Wert;
`results/sim2real/spectra-<probe>.json`, Abb. `stage5-spectra.png`) für die Merkmale aus 3.8;
`psf`, `noise` und `fiber` stehen in der Datei und in der Abbildung:

| Probe | Split | Boden | oberes Band | tiefste Frequenzen | Steigung | CTF |
|---|---|---|---|---|---|---|
| Spinnenhaar | passend | 70 | 84 | 48 | 53 | 79 |
| | `flat-field` | 0 | 98 | 82 | 24 | 90 |
| | `object` | 70 | 98 | 16 | 20 | 98 |
| | `combined` | 0 | 100 | 60 | 0,4 | 98 |
| Zahn | passend | 83 | 61 | 95 | 62 | 3 |
| | `flat-field` | 0 | 86 | 100 | 20 | 3 |
| | `object` | 84 | 85 | 8 | 56 | 28 |
| | `combined` | 0,4 | 98 | 61 | 14 | 39 |
| Kaktusnadel | passend | 73 | 76 | 86 | 52 | 12 |
| | `flat-field` | 38 | 90 | 100 | 22 | 30 |
| | `object` | 81 | 93 | 24 | 41 | 99,6 |
| | `combined` | 99 | 98 | 68 | 26 | 99,6 |
| | realistisch | 45 | 98 | 78 | 21 | 99 |
| Mg-Draht | passend | 97 | 97 | 14 | 29 | 70 |
| | `flat-field` | 83 | 100 | 72 | 6 | 94 |
| | `object` | 99,6 | 98 | 0,4 | 32 | 100 |
| | `combined` | 100 | 99,6 | 9 | 16 | 100 |
| | realistisch | 78 | 100 | 11 | 18 | 100 |

1. **Die simulierten Objekte der Messung löschen die CTF fast aus.** Die Korrelation des
   Spektrums mit $\sin^2(\pi f^2/Fr)$ liegt unter `object` und `combined` bei allen vier Proben
   im Median bei −0,04 bis +0,05, auf den passenden Splits bei 0,15 bis 0,62. Die
   Messhologramme haben 0,25 (Spinnenhaar), 0,32 (Kaktusnadel) und 0,72 (Mg-Draht), mehr als
   98–100 % der Samples unter `object` und `combined`. Nur der Zahn hat mit −0,04 ebenfalls kaum
   CTF-Modulation (3. Perzentil des passenden Splits). Die realistischen Sätze haben das Objekt
   von `object`, und ihre Korrelation liegt ebenso bei −0,03 bzw. −0,02: Dem Netz fehlt dort
   weitgehend die Modulation, die $z_{01}$ im Spektrum trägt (5.7). Eine mögliche Erklärung,
   nicht geprüft: Die simulierten Objekte sind homogene Zylinder mit 1–20 rad Phasenhub und
   einem Zehntel bis fast dem ganzen Sichtfeld als Durchmesser (`MEASURED_MODELS`). Ihre
   Struktur steckt in den Rändern, wo die lineare CTF bei diesem Phasenhub nicht gilt; die
   echten Proben haben dazu innere Struktur (Abb. `stage5-holograms.png`), deren schwache
   Phasenvariation die CTF über viele Frequenzen moduliert.
2. **Die Messung hat mehr feine Struktur.** Das obere CNN-Band liegt im Messhologramm weiter
   über dem Boden als bei 98–100 % der Samples unter `combined` (passend: 61.–97. Perzentil).
   Auch das passt zu Objekten mit innerer Struktur.
3. **Der Flatfield-Rest ist bei Spinnenhaar und Zahn an den höchsten Frequenzen zu stark.** Ohne
   ihn liegt der Boden des Messhologramms beim 70. bzw. 83. Perzentil, mit ihm unter allen
   simulierten Samples (0. Perzentil unter `flat-field`, `noise` und `combined`; beim Zahn
   unter `combined` bis auf eines, 0,4 %). Das Modell aus den Ecken (3.6) legt dort also mehr
   Leistung hin, als das ganze Messhologramm hat; dass Simulator und Fit dasselbe Modell
   meinen, prüft `test_corner_spectra_measure_blur_white_noise_and_residual`. Bei Kaktusnadel
   und Mg-Draht passt der Boden (38. bzw. 83. Perzentil unter `flat-field`). Spinnenhaar und
   Zahn sind auch die beiden Proben, bei denen der Flatfield-Rest die Schätzung nach unten
   schiebt (5.5, Punkt 2); ob das zusammenhängt, habe ich nicht geprüft.
4. **Steigung und tiefste Frequenzen** liegen auf den passenden Splits in der Verteilung
   (29.–62. bzw. 14.–95. Perzentil). Unter `combined` ist das Messhologramm steiler als die
   meisten Samples (0.–26. Perzentil); das kommt vom Flatfield-Rest, der die hohen Frequenzen
   anhebt, denn schon unter `flat-field` allein sind es 6.–24., unter `psf` 40.–60.
5. **Woran der Simulator der Messung am wenigsten ähnelt**, ist also die CTF-Modulation der
   Objekte. Die Merkmale brauchen kein Training: Ein neues Objektmodell lässt sich an ihnen
   prüfen, bevor ein Netz darauf trainiert wird (Abschnitt 7, Punkt 2).

### 5.7 Realistische Trainingssätze

Das Modell der realistischen Sätze (3.9) neben dem ausgewählten Modell der passenden Sätze
(`results/learned/matched-<probe>-realistic/`, `results/sim2real/realistic/`); MAE in µm, in
Klammern der Bias:

| Probe | Modell | beste Epoche | realistischer Test-Split | passender Test-Split | `combined` | Messung − [2] | Messung − Padding 4 |
|---|---|---|---|---|---|---|---|
| Kaktusnadel | passend (Standard) | 24 | – | 735 (+226) | 3375 (+1890) | +4753 | +4024 |
| Kaktusnadel | realistisch | 10 | 1929 (−91) | 3566 (−1213) | 1682 (−169) | −1669 | −2397 |
| Mg-Draht | passend (Standard) | 7 | – | 1273 (+148) | 2811 (+1167) | +2813 | +2385 |
| Mg-Draht | realistisch | 20 | 2605 (+267) | 2589 (+111) | 2607 (−8) | +1701 | +1273 |

Der Schätzwert allein hat einen MAE von 2500 µm.

1. **Auf den realistischen Sätzen lernt das Netz wenig oder nichts.** Bei der Kaktusnadel
   kommt es auf dem realistischen Test-Split auf 1929 µm, knapp ein Viertel unter dem
   Schätzwert, beim Mg-Draht auf 2605 µm, also nicht besser als der Schätzwert; die besten
   Epochen sind 10 und 20. Mit den gemessenen Eigenschaften als Bereichen enthält das
   Spektrum für das Netz bei 6000 Samples kaum noch Information über $z_{01}$.
2. **Auf sauberen Daten ist das realistische Modell verschoben.** Auf dem passenden
   Test-Split schätzt das Modell der Kaktusnadel im Mittel 1,2 mm zu tief (MAE 3566 µm), das
   Spiegelbild der Verschiebung des passenden Modells unter `combined` (+1,7 mm, 5.5). Die
   gemessenen Eigenschaften lassen ein Hologramm für das Netz also wie ein größeres
   $z_{01}$ aussehen; ein Modell, das sie kennt, rechnet das heraus. Unter `combined`, also mit
   den Eigenschaften der Messung, trifft es mit 1682 µm MAE (Bias −169 µm) besser als das
   passende Modell (3375 µm) und deckt mit seinen 68-%-Intervallen 76 % der Samples. Beim
   Mg-Draht bleibt der MAE auf allen Splits bei 2587–2607 µm, und keine Störung verschiebt die
   Schätzung um mehr als 0,16 mm: Das Modell antwortet fast unabhängig vom Hologramm.
3. **Auf den Messhologrammen ist es nicht besser als der Schätzwert.** Bei der Kaktusnadel
   schätzt es 282,531 mm: 1,67 mm unter [2], wo der Schätzwert 1,90 mm darüber liegt, aber
   2,40 mm unter unserer Suche mit Padding 4, von der der Schätzwert nur 1,17 mm entfernt ist.
   Beim Mg-Draht bleibt es mit 472,501 mm beim Schätzwert (+0,10 mm), wie es zu einem Modell
   passt, das fast unabhängig vom Hologramm antwortet. Das passende Modell lag mit +4,75 und
   +2,81 mm weiter von [2] weg. Im Spektrum liegt das Messhologramm für das realistische
   Modell außerhalb seiner Trainingsdaten: Seine CTF-Modulation ist stärker als bei 99 bzw.
   100 % des realistischen Test-Splits (5.6).
4. **Entscheidung: vorerst keine weiteren realistischen Sätze.** Für Spinnenhaar und Zahn habe
   ich keine erzeugt und für Maxwell keine großen vorgesehen. Mehr Daten aus denselben
   Bereichen brächten dem Netz ein Problem bei, das die Messung so nicht stellt: Die simulierten
   Objekte löschen die CTF aus, die echten nicht (5.6). Zuerst braucht der Simulator Objekte,
   deren Spektrum zu den Messhologrammen passt (Abschnitt 7, Punkt 2).

### 5.8 Rauschen und Unschärfe der Messhologramme

Fits der vier Ecken (`results/sim2real/noise.json`, Abb. `stage5-noise.png`; PSF in unbinnten
Pixeln, Rest $\rho$ und Grenzfrequenz $f_c$ auf dem 4×-Gitter):

| Probe | PSF $\sigma$ | Median | Rest $\rho$ | Median | $f_c$ (Median) | Mittelwerte der Ecken |
|---|---|---|---|---|---|---|
| Spinnenhaar | 1,70–1,73 px | 1,71 px | 2,3–2,7 % | 2,5 % | 0,18 | 0,98–0,99 |
| Zahn | 1,45–1,81 px | 1,64 px | 3,2–6,5 % | 3,9 % | 0,11 | 0,96–1,00 |
| Kaktusnadel | 1,37–1,78 px | 1,51 px | 1,9–3,5 % | 2,6 % | 0,19 | 0,96–1,00 |
| Mg-Draht | 1,77–2,65 px | – | 3,5–8,1 % | – | 0,02 | 1,06–1,52 |

- Beim Spinnenhaar stimmen die vier Ecken überein (1,70–1,73 px, 2,3–2,7 %); das NPS
  folgt dem Modell aus verschmiertem und weißem Rauschen über den ganzen Frequenzbereich.
- Bei Zahn und Kaktusnadel unterscheiden sich die rechten Ecken von den linken: Mittelwert
  0,96 statt 1,00, ein um 40–60 % höherer weißer Boden und eine breitere PSF (1,64–1,81 gegen
  1,37–1,48 px). Ob die PSF über den Detektor variiert, der Strahl dort schwächer ist oder
  Objektstruktur in die rechten Ecken reicht, lässt sich aus den Ecken allein nicht trennen.
  Der Bereich 1,2–2,0 px der realistischen Sätze (3.9) deckt die Werte ab.
- Beim Mg-Draht liegt der Draht in den Ecken (Mittelwerte bis 1,52); Grenzfrequenz 0,02 und
  PSF bis 2,65 px beschreiben das Objekt, nicht den Detektor (E6).
- Der weiße Teil passt zur Photonenzahl aus Laborbuch 02, 5.4 (3.6).

### 5.9 Laufzeiten

Auf der VM (4 Kerne), aus den Logs in `runs/stage5/` (nicht versioniert) und den
Ergebnisdateien:

| Schritt | Threads | Wandzeit |
|---|---|---|
| Passende Datensätze, je 6500 Samples (3.2) | 3 | 13 min (Mg-Draht) bis 72 min (Spinnenhaar), zusammen 2,2 h |
| Realistische Sätze, je 6500 Samples (3.9) | 3 bzw. 1 | 28 min (Kaktusnadel, 3 Threads), 38 min (Mg-Draht, 1 Thread) |
| Training mit Vorhersagen auf Val und Test, 8 Modelle (3.3) | 1 | je 24 min (Standard), 25 min (realistisch), 46 bzw. 48 min (`-hf`); zusammen 4,0 h |
| Gestörte Splits, je 250 Samples (3.7) | 1 | pro Störung 1,0–1,5 min (Kaktusnadel, Mg-Draht), 3–6 min (Zahn), 6–10 min (Spinnenhaar); zusammen 1,4 h |
| Schätzung auf den Messhologrammen mit allen Varianten, Binning-Phasen und Zeitmessung (3.4) | 1 | unter 15 s für alle vier Proben |
| Suche mit Padding 4, kalt (3.5) | 3 | 45 min (Kaktusnadel), 42 min (Mg-Draht) |
| Warmstart mit Padding 4 (3.5) | 3 | 28 min (Spinnenhaar), 37 min (Zahn), 53 min (Kaktusnadel), 42 min (Mg-Draht) |
| Spektren aller Splits (3.8) | 1 | 81 s |
| Testsuite (278 Tests) | 1, die Cluster-Tests 4 | 35 s (neben dem Warmstart der Kaktusnadel 5,5 min, 5.3) |
| Abbildungen | 1 | 6 s |

Zusammen sind das etwa 27 Kernstunden (Threads mal Wandzeit): je knapp die Hälfte für die
Datensätze mit dem Training (48 %) und für die Suchen mit Padding 4 (46 %), gut 5 % für die
gestörten Splits. Ist das Netz einmal trainiert, kostet eine Schätzung Millisekunden, eine
Suche eine Dreiviertelstunde auf der CPU.

## 6 Offene Punkte

1. **Referenz auf Messdaten.** Für die vier Hologramme gibt es kein wahres $z_{01}$. Die Werte
   der Autoren streuen um bis zu 1,2 mm ([1] gegen [2] Table 1), und unsere Suche mit Padding 4
   liegt beim Zahn 266 µm neben [2] (Abschnitt 2). Gemessen wird deshalb eine Abweichung, kein
   Fehler, und die Schwelle von 100 µm lässt sich nicht prüfen (E5). Eine Fokusserie derselben
   Probe mit bekannten Motorschritten in $z_{01}$ gäbe eine relative Referenz: Die Differenzen
   der Schätzungen müssen den Schritten folgen, unabhängig vom absoluten Wert. Ebenso die
   Projektionen einer Tomographie, die alle dasselbe $z_{01}$ haben: Die Streuung der
   Schätzungen über die Projektionen misst die Wiederholbarkeit auf echten Daten. → J. Dora
   (über den Nutzer): Gibt es an P05 solche Serien oder Projektionen, und wie genau ist
   $z_{01}$ an der Beamline mechanisch bekannt?
2. **Feste oder relative Erfolgsschwelle.** Das Netz liest die Fresnelzahl in allen fünf
   Aufbauten auf etwa 0,3 % genau (5.1, Punkt 1). Die feste Schwelle von 100 µm ist beim
   Spinnenhaar 0,13 % von $z_{01}$, beim Mg-Draht 0,02 %; ob sie erreicht wird, hängt damit vor
   allem am Aufbau. → Daniel: Gilt 100 µm für alle Aufbauten, oder soll die Schwelle relativ zu
   $z_{01}$ sein (oder zur Schärfentiefe des Aufbaus)?
3. **Simulator an den Testhologrammen eingestellt.** Die Konstanten der Störungen und der
   realistischen Sätze (PSF, Rauschen, Flatfield-Rest, Objekt) stammen aus denselben vier
   Hologrammen, auf denen das Netz getestet wird; deren $z_{01}$-Referenzen nutzt das Training
   nicht (E8). → Daniel: Ist das für die Arbeit zulässig, oder braucht es getrennte Hologramme
   zum Einstellen und zum Testen? Mit den Daten aus Punkt 1 ließe sich das trennen.
4. **Vier Hologramme.** Jede Probe ist ein einzelner Datenpunkt in ihrem eigenen Aufbau; eine
   Streuung gibt es nur über die Binning-Phasen derselben Messung (5.2), nicht über
   Hologramme. → Nutzer: weitere Messdaten von P05 (Punkt 1)?
5. **GPU:** Die großen passenden Datensätze mit Training und Schätzung auf den Messhologrammen
   (`docs/maxwell-gpu.md`, 7.1) rechnet der Nutzer auf Maxwell. Realistische Sätze sind dort
   noch nicht vorgesehen (5.7, Punkt 4).

## 7 Nächste Schritte

1. **Große passende Datensätze auf Maxwell** (`docs/maxwell-gpu.md`, 7.1): 50 000
   Trainings-Samples pro Probe, Training und Schätzung auf den Messhologrammen. Das zeigt, ob
   mehr Daten nur den simulierten Fehler senken (5.1, Punkt 3) oder auch die Abweichung auf den
   Messdaten (5.4). Nach den Störungen (5.5) erwarte ich vor allem Ersteres: Die Abweichung
   folgt aus Annahmen, die mehr Daten desselben Simulators nicht ändern.
2. **Objekte, die die CTF der Messung erhalten, dann realistische Sätze.** Die simulierten
   Objekte der Messung löschen die CTF-Modulation aus, die echten nicht (5.6), und auf den
   realistischen Sätzen lernt das Netz deshalb wenig oder nichts (5.7). Bevor es realistische
   Sätze für alle vier Proben und deren große Versionen auf Maxwell gibt, braucht der Simulator
   Objekte, bei denen die spektralen Merkmale der Messhologramme (3.8) in der Verteilung der
   simulierten Samples liegen; das lässt sich ohne Training prüfen. Dazu der Flatfield-Rest
   bei Spinnenhaar und Zahn, der an den höchsten Frequenzen zu stark ist (5.6, Punkt 3).
3. **Objekte mit innerer Struktur** (Etappe 6). In den Messhologrammen haben die Objekte
   innere Struktur, am deutlichsten beim Mg-Draht, schwächer beim Zahn (Abb.
   `stage5-holograms.png`); die simulierten Fasern sind homogene Zylinder aus einem Material.
   Im Simulator hieße das: Fasern mit einer zufälligen Textur der Phase im Inneren (etwa ein
   Gaußfeld wie der Flatfield-Rest, mit dem Faserprofil multipliziert) oder Objekte aus den
   Phasenrekonstruktionen von Etappe 3, mit dem Vorbehalt aus 6.3.
4. **NPE mit sbi** (Plan, Etappe 7): eine Posterior-Verteilung von $\Delta z_{01}$ mit dem
   Spektrum-CNN als Einbettung, auf denselben passenden Datensätzen, geprüft mit
   simulationsbasierter Kalibrierung auf den Test-Splits. Auf den Messhologrammen dieselben
   Abstände wie hier, dazu die gepaarten Störungen als Prüfung der Robustheit.
5. **Geometrie-konditioniertes Modell** für die Beamline mit vielen Aufbauten (E1): ein Netz
   auf der Achse $u = f^2/Fr^\text{guess}$ (Laborbuch 04, Abschnitt 6), sobald die Lücke zum
   Simulator kleiner ist. Weil der relative Fehler in allen Aufbauten gleich ist (5.1), wäre
   $\Delta \ln Fr$ dort die natürliche Zielgröße statt $\Delta z_{01}$.

## 8 Bezug zur Masterarbeit

- **Methoden:** passende Datensätze im Aufbau der Messung (3.2); Detektor-PSF, Flatfield-Rest
  und Fasern im Simulator (3.1); das Rauschleistungsspektrum der Messhologramme (3.6); die
  gepaarten Störungen als Zerlegung der Sim-to-Real-Lücke (3.7, E7); spektrale Merkmale (3.8)
  und die Binning-Phasen als Prüfung der Stabilität (3.4).
- **Ergebnisse:** gleicher relativer Fehler auf passenden simulierten Daten (5.1); die
  Schätzungen auf den Messhologrammen gegen [2] Table 1 und die Baseline mit Padding 4,
  Warmstart und Laufzeiten (5.2, 5.3, 5.9); Testfehler gegen Abweichung (5.4); der Beitrag
  jeder Simulatorannahme (5.5, 5.6); realistische Trainingssätze (5.7); Rauschen und PSF des
  P05-Detektors (5.8).
- **Diskussion:** die Sim-to-Real-Lücke und welche Annahmen sie erklären; dass es auf
  Messdaten nur Abweichungen zwischen Schätzungen gibt (E5, 6.1); feste gegen relative
  Schwelle (6.2); das Einstellen des Simulators an den Testhologrammen (E8, 6.3); Millisekunden
  gegen Minuten auch auf Messdaten.

Reproduktion: alle Befehle in `docs/usage.md`, Abschnitt „Measured holograms (Stage 5)“.
