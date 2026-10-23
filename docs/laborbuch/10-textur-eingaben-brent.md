# Laborbuch 10 – Etappe 10: Über das radiale Spektrum hinaus

| | |
|---|---|
| Datum | 23.–25.10.2026 |
| Branch | `cursor/stage10-beyond-radial-spectrum-1091` |
| Code | die Pegel des Spektrums in `spectral_features` (`src/holofocus/learned/sim2real.py`) und in `fit_objective` (`LEVEL_NAMES`, `LEVEL_SCALE`, `TEXTURE_FITS_LEVELS` in `src/holofocus/learned/objectmodel.py`), `--levels` für `scripts/run_stage6_study.py fit`; die Darstellung `polar` (`src/holofocus/learned/config.py`, `representations.py`) und `PolarSpectrumCNN` mit `AngularConv` und `AnglePool` (`models.py`), die Einbettung des NPE über `build_cnn` (`npe.py`); `configs/learned/polar-cnn.yaml`, `polar-cnn-hf.yaml`, `image-cnn-b1.yaml`, `configs/dataset/loo2-<probe>[-large].yaml`; `scripts/run_jobs.py`; `scripts/run_stage10_study.py` (`configs`, `plan`, `ensembles`, `mixture`, `measured`, `search`, `merge`, `texture`, `spectra`, `compare`, `evaluate`), `scripts/make_stage10_figures.py`; `search_pair` in `scripts/run_stage9_study.py`; `tests/test_run_jobs.py`, Erweiterungen von `tests/test_learned.py`, `test_npe.py`, `test_objectmodel.py`, `test_sim2real.py` und `test_cluster_kit.py`; Abschnitt 8.5 in `docs/maxwell-gpu.md` |
| Ergebnisse | `results/objectmodel/levels/`, `results/objectmodel/loo2-ranges.json`, `results/datasets/loo2-<probe>-{val,test}.csv`, `results/stage10/` (Trainings und Ensembles auf cpu-4x und loo2, Suchen, `texture-<probe>.json`, `spectra-<probe>.json`, `compare.json` und CSVs), `results/measured/<familie>-loo2-ensemble/`, `results/measured/brent-box95-<familie>-loo2-ensemble-b1-p4/` |
| Abbildungen | `docs/laborbuch/figures/stage10-*.png`, Zahlen in `stage10-metrics.json` |
| Quellen | [2] Dora et al., Opt. Express 33(4), 6641 (2025); R. P. Brent, *Algorithms for Minimization without Derivatives*, Prentice-Hall (1973), Kap. 5; E. B. Wilson, J. Am. Stat. Assoc. 22, 209 (1927); D. A. Lifka, *The ANL/IBM SP scheduling system*, in: Job Scheduling Strategies for Parallel Processing, LNCS 949, 295 (1995) |

## 1 Ziel

Drei Teile, alle auf der CPU der VM; was eine GPU braucht, läuft hier verkleinert und steht
voll in `docs/maxwell-gpu.md` (E11):

1. **Das Texturmodell bei hohen Frequenzen** (Laborbuch 06, 5.2, Punkt 6, und 6, Punkt 3; in
   Etappe 8 noch offen). Die Textur, die Etappe 6 an die Messhologramme angepasst hat, legt bei
   Kaktusnadel und Mg-Draht weit mehr Leistung in die hohen Frequenzen als die Messung. Das
   Spektrum-CNN sieht diesen Teil zum Teil nicht, ein Bild-CNN und ein Spektrum bis
   $f^2 = 0{,}25$ sähen ihn. Die Textur soll auch die Pegel des Spektrums treffen, bevor
   Eingaben über das radiale Spektrum hinaus auf diesen Sätzen lernen.
2. **Eingaben über das radiale Spektrum hinaus.** Ein CNN auf dem zweidimensionalen
   Leistungsspektrum (polar, nach Richtung aufgelöst) und ein CNN auf dem Hologramm selbst
   gegen das CNN und den NPE auf dem radialen Spektrum, im selben fairen Aufbau wie zuvor: drei
   Seeds und ihr Ensemble pro Familie, Leave-one-out-Sätze mit Textur, ausgewertet auf den 48
   synthetischen Samples der Baseline und auf den vier Messhologrammen. Berichtet werden der
   Anteil innerhalb 100 µm, der MAE und die Abdeckung des 95-%-Intervalls. In Etappe 9 verfehlte
   das Intervall des CNN die Wahrheit auf 6 der 48 Samples (Laborbuch 09, 5.6, Punkt 4).
3. **Brent im gelernten Intervall als Verfeinerung.** Brents Verfahren in den Grenzen des
   95-%-Intervalls jedes Ensembles, auf denselben 48 Samples und vier Hologrammen; Nelder–Mead
   kalt (Etappe 3) und Brent kalt (Etappe 9) bleiben die Referenzen.

## 2 Ausgangslage

Etappe 9 ist in `main` gemergt
([PR #17](https://github.com/AliTUHH/master_thesis_autofocus/pull/17), Merge-Commit
`b66cb46`); dieser Branch geht von dort aus.

**Die Textur von Etappe 6** (`TEXTURE_FITS`, Laborbuch 06, 3.3 und 5.2). Das Maß der Anpassung
(`fit_objective`) vergleicht die Form des radialen Log-Leistungsspektrums: die Korrelation mit
der CTF über das Band des CNN und in vier Bändern, das Signal-Rausch-Verhältnis des oberen Bandes
über dem Boden und die Steigung. Wie viel Leistung eine Textur hinzufügt, sieht es nicht. Bei
Kaktusnadel und Mg-Draht lag die beste Steigung am Rand des Gitters (1,75), und der Boden des
Spektrums lag mit der Textur bei −6,4 statt −7,6 (Kaktusnadel) und bei −3,9 statt −6,8
(Mg-Draht), im natürlichen Logarithmus der Leistung. Zu Beginn dieser Etappe nachgemessen, auf
den Sätzen der Anpassung von Etappe 6 (16 Test-Samples pro Textur): Ab $f^2 = 0{,}02$ hat die
Simulation mit der Textur von Etappe 6 das 5,8-Fache (Kaktusnadel) und das 20-Fache (Mg-Draht)
der Leistung des Messhologramms, beim Spinnenhaar das 1,4-Fache; ohne Textur das 0,73-, 0,31-
und 0,64-Fache, beim Zahn (ohne Textur angepasst) das 0,98-Fache. Was die Textur hinzufügt, ist
das 18-Fache (Kaktusnadel) und das 29-Fache (Mg-Draht) dessen, was die Messung über der
Simulation ohne Textur hat, beim Spinnenhaar das 2-Fache. Die Leave-one-out-Sätze von Etappe 6
(`loo-<probe>`) erben diese Texturen über ihre Bereiche.

**Die gelernten Schätzer auf dem radialen Spektrum.** Auf cpu-4x (10 000 Trainings-Samples) das
Gauß-CNN von Etappe 4 (`results/learned/spectrum-cnn-wide`, ein Seed) und das NPE-Ensemble von
Etappe 7 (fünf Seeds; sein 95-%-Intervall deckt auf den 250 Test-Samples 96,8 % bei 894 µm
Breite im Median, Laborbuch 07). Auf den Leave-one-out-Sätzen von Etappe 6 je drei Seeds des
CNN und des NPE und ihre Ensembles (Laborbuch 08). Auf den vier Messhologrammen trifft das
CNN-Ensemble [2] auf keinem innerhalb 100 µm (MAE 912 µm), das NPE-Ensemble auf einem (MAE
688 µm); beide 95-%-Intervalle enthalten [2] auf allen vier. Ein Bild-CNN gibt es aus Etappe 4
(`image-cnn.yaml`, 2× gebinnt), auf cpu-4x mit einem Seed (`results/learned/study/image-n10000`,
Laborbuch 04).

**Brent** (Laborbuch 09). Auf den 48 synthetischen Samples braucht Brent kalt 8,17
Rekonstruktionen pro Autofokus, Nelder–Mead 11,15, bei MAE 164 und 166 µm. Im 95-%-Intervall
des NPE-Ensembles spart Brent 29 % gegen Brent kalt, in dem des einzelnen CNN 31 %; das
Intervall des CNN schnitt die Wahrheit auf 6 der 48 Samples ab, bei Sample 16 um 1,1 mm. Auf den
Messhologrammen brauchen Nelder–Mead kalt 49 Rekonstruktionen für die vier Proben, Brent kalt
30, Brent in den Intervallen der Ensembles von Etappe 8 28 (CNN) und 27 (NPE).

## 3 Was gemacht wurde

### 3.1 Die Pegel des Spektrums im Maß der Anpassung (`86476c0`, `4b98cbb`)

`spectral_features` gibt zu den Merkmalen von Etappe 6 die mittlere Log-Leistung zweier Bänder
aus: `level_mid` ($f^2$ = 0,02–0,04) und `level_upper` (0,0625–0,125, das obere Band des
CNN); `floor` (0,2–0,25) gab es schon. `fit_objective(median, measured, levels=True)` nimmt die
drei Differenzen zur Messung, jede durch `LEVEL_SCALE` = 0,5 geteilt, zu den Termen von
Etappe 6 hinzu (E1, E2); ohne `levels` ist das Maß das von Etappe 6, bitgenau. `fit --levels`
in `scripts/run_stage6_study.py` rechnet das Gitter von Etappe 6 mit diesem Maß: dieselben
Texturen grob, dann fein um die beste, je 16 Test-Samples des Satzes, der das Messhologramm
nachbildet (`matched-<probe>`), mit dem Median ihrer Merkmale gegen die des Hologramms (E3). Das
Ergebnis steht in `results/objectmodel/levels/fit-<probe>.{json,csv}` und als
`TEXTURE_FITS_LEVELS` im Code (Textur innen, Film über dem Sichtfeld, Steigung):

| Probe | Etappe 6 | Etappe 10 | Maß mit Pegeln | Pegel – Messung (Mitte, oben, Boden) |
|---|---|---|---|---|
| Spinnenhaar | 0 / 0,1 rad / 3,0 | gleich | 1,03 | +0,25, +0,41, +0,43 |
| Zahn | ohne | gleich | 0,70 | +0,31, +0,28, +0,39 |
| Kaktusnadel | 0,3 / 0,03 rad / 1,75 | 0 / 0,1 rad / 2,75 | 1,76 | +0,86, +0,37, +0,08 |
| Mg-Draht | 0,3 / 0,533 rad / 1,75 | 0,3 / 0,0533 rad / 2,0 | 1,88 | +1,02, +0,93, +0,63 |

Spinnenhaar und Zahn behalten ihre Anpassung. Die Kaktusnadel hat statt einer Textur im Objekt
einen Film mit steilerem Abfall, der Mg-Draht ein Zehntel des Films und die Steigung 2,0. Ohne
Textur liegen Spinnenhaar und Zahn schon an den höchsten Frequenzen über der Messung (Boden
+0,33 und +0,39); das ist der Flatfield-Rest, der dort zu stark ist (Laborbuch 05, 5.6,
Punkt 3), keine Textur. Wie sich die Texturen beider Etappen auf denselben Samples gegen die
Messung verhalten, rechnet `run_stage10_study.py texture` (3.7, 5.1).

### 3.2 Die Leave-one-out-Sätze `loo2` (`4b98cbb`)

`run_stage10_study.py configs` schreibt `configs/dataset/loo2-<probe>.yaml`: die Configs
`loo-<probe>.yaml` von Etappe 6 mit den Bereichen aus `TEXTURE_FITS_LEVELS` der anderen drei
Proben, verbreitert um `LOO_FACTOR` = 2 wie dort, sonst gleich, mit eigenem Seed
(20261101–20261104) und 6000/250/250 Samples (E4). Die Bereiche der Textur:

| Probe | RMS innen und Film (`loo`) | `loo2` | Steigung (`loo`) | `loo2` |
|---|---|---|---|---|
| Spinnenhaar | 0,005–1,07 rad | 0,005–0,6 rad | 1,25–2,25 | 1,5–3,25 |
| Zahn | 0,005–1,07 rad | 0,005–0,6 rad | 1,25–3,5 | 1,5–3,5 |
| Kaktusnadel | 0,005–1,07 rad | 0,005–0,6 rad | 1,25–3,5 | 1,5–3,5 |
| Mg-Draht | 0,005–0,6 rad | 0,005–0,2 rad | 1,25–3,5 | 2,25–3,5 |

Die eigene Textur liegt bei drei Proben in ihrem Bereich; beim Mg-Draht nicht (0,3 rad innen,
Steigung 2,0), weil keine der anderen drei Proben so viel Textur im Objekt hat
(`results/objectmodel/loo2-ranges.json`). Das bleibt so: Leave-one-out heißt, dass der Satz
nicht vom Hologramm lernt, auf dem er geprüft wird (Laborbuch 06, E5). `loo2-<probe>-large.yaml`
legt 50 000 Trainings-Samples mit eigenem Seed (20261111–20261114) und derselben Physik dazu,
für die GPU (3.8). Val und Test sind mit `check_cluster_run.py fingerprint` festgehalten
(`results/datasets/loo2-<probe>-{val,test}.csv`), damit Maxwell sie prüfen kann.

### 3.3 Das polare Spektrum und sein CNN (`86476c0`)

Die Darstellung `polar` ist das Leistungsspektrum des Hann-gefensterten Hologramms wie bei
`spectrum`, aber nach der Richtung der Frequenz aufgelöst: `angles` = 8 Sektoren von
$\theta = \operatorname{atan2}(f_y, f_x)$ modulo 180° (ein reelles Hologramm hat
$P(-f) = P(f)$), zentriert bei $k \cdot 22{,}5°$, mal dieselben 1024 Bins von $f^2$ bis 0,125;
Form $(1, 8, 1024)$, Logarithmus, über alle Zellen standardisiert. Jede Frequenz geht mit
linearen Gewichten an die zwei nächsten Sektoren und die zwei nächsten Bins (E5); die
Nullfrequenz bleibt draußen. Eine Faser legt ihr CTF-Muster in die Sektoren quer zu ihrer Achse,
das radiale Mittel mischt es mit der Textur aller Richtungen. Eine Drehung des Hologramms um 90°
verschiebt die Sektoren zyklisch um 4, eine Spiegelung kehrt ihre Reihenfolge um.

`PolarSpectrumCNN` (E6): Die erste Stufe ist eine Faltung über 3 Sektoren × 7 Bins, zyklisch und
symmetrisch im Winkel (`AngularConv`), mit Schrittweite 2 entlang $f^2$; danach Mittel und
Maximum über die Sektoren (`AnglePool`), dann die Stufen, der Kopf, der Plan und der Verlust von
`spectrum-cnn-wide.yaml` (465 000 Parameter, das radiale CNN 450 000). Das Maximum hält die
Sektoren quer zu einer Faser, das Mittel ist das radiale Mittel. Die Ausgabe ändert sich unter
Drehungen um 90° und Spiegelungen nicht, deshalb trainiert es ohne Augmentierung
(`configs/learned/polar-cnn.yaml`). Die Einbettung des NPE nimmt das polare Spektrum über
`build_cnn` auch; trainiert ist ein solcher NPE nicht (E8).

### 3.4 Vier Familien, drei Seeds, ein Ensemble

| Familie | Config | Eingabe | Ensemble |
|---|---|---|---|
| `radial-cnn` | `spectrum-cnn-wide.yaml` | radiales Spektrum, 1024 Bins | Mischung der Gaußverteilungen |
| `npe` | `npe-nsf.yaml` | radiales Spektrum, 1024 Bins | `npe-predict` mit den drei Läufen |
| `polar-cnn` | `polar-cnn.yaml` | polares Spektrum, 8 × 1024 | Mischung |
| `image-cnn` | `image-cnn.yaml` | Hologramm 2× gebinnt, 256² | Mischung |

Jede Familie lernt mit den Seeds 0, 1 und 2 (`holofocus-learn train --seed`) auf cpu-4x
(10 000 Trainings-Samples, Val 250) und auf jedem `loo2`-Satz (6000, Val 250). Auf cpu-4x gibt
es einige Seeds schon, mit denselben Dateien und derselben Config trainiert: Seed 0 des radialen
CNN (Etappe 4), die Seeds 0–2 des NPE (Etappe 7) und Seed 0 des Bild-CNN (Etappe 4, mit
`--limit 10000`, das die neuen Seeds deshalb auch setzen); neu sind 7 Trainings auf cpu-4x und
48 auf `loo2` (E8). Das Ensemble der CNNs ist die Mischung der Gaußverteilungen der Seeds zu
gleichen Gewichten, mit dem auf Val nachkalibrierten $\hat\sigma$ jedes Seeds, wie in Etappe 8;
`ensembles` und `mixture` schreiben sie in den Spalten der NPE-Vorhersagen (Schätzung, zentrale
68- und 95-%-Intervalle, PIT), damit beide Arten von Ensemble dieselben Kennzahlen und dieselbe
Begrenzung einer Suche durchlaufen. Das Ensemble des NPE mischt die drei Posteriors
(`holofocus-learn npe-predict` mit drei Läufen, Laborbuch 07).

Auf den Messhologrammen schätzt `run_stage10_study.py measured` mit jedem Seed, der auf dem
`loo2`-Satz der Probe gelernt hat, 4× gebinnt mit den 16 Phasen des Binning-Gitters, wie die
Schritte `measured` der Etappen 6 und 7, und schreibt das Ensemble nach
`results/measured/<familie>-loo2-ensemble/<probe>.json`.

### 3.5 Brent im gelernten Intervall

Auf den 48 Samples: `run_stage10_study.py search` sucht ein Sample mit
`configs/baseline/paper2-cpu-brent.yaml` (Etappe 9) im 95-%-Intervall des Ensembles einer Familie
auf cpu-4x, auf $\Omega_z$ beschnitten (`--box`, Laborbuch 09, 3.3), nach
`runs/stage10/search/<familie>/<index>`; `merge` fügt die 48 zu
`results/stage10/baseline/cpu-4x-test-brent-box95-<familie>` zusammen. Auf den Messhologrammen:
`run_stage3_experiments.py measured --binning 1 --padding 4 --method brent --revisit-tolerance
1e-9 --box results/measured/<familie>-loo2-ensemble` wie in Etappe 9, pro Probe ein Job. Die
Referenzen werden nicht neu gerechnet (E10): Nelder–Mead kalt (`results/baseline/cpu-4x-test`,
`results/measured/b1-p4`) und Brent kalt (`cpu-4x-test-brent`, `brent-b1-p4`).

### 3.6 Die Jobs und ihr Runner (`768e16a`, `3141583`)

Die Etappe besteht aus 388 Jobs mit Abhängigkeiten: Erzeugen der `loo2`-Sätze, ihre
Fingerabdrücke, Trainings, Vorhersagen, Ensembles, Schätzungen auf den Messhologrammen, 192
Suchen auf den synthetischen Samples, 16 auf den Messhologrammen, `texture`, `spectra` und am
Ende `compare`; zusammen 101,5 Kernstunden nach den vorab gemessenen Zeiten
(`run_stage10_study.py plan`, `runs/stage10/plan.json`). `scripts/run_jobs.py` führt einen
solchen Plan auf einer Maschine aus (E12): Ein Job startet, wenn alle, die er braucht, fertig
sind und seine Threads und sein Speicher in das passen, was die laufenden vom Budget lassen
(4 Threads, 8 GB, 1 GB Reserve; E13). Die höchste Priorität zuerst; passt der erste wartende Job
nicht, reserviert er den Zeitpunkt, an dem die laufenden genug für ihn freigeben, und ein Job
dahinter darf vor ihm starten, wenn er bis dahin endet oder ihm genug lässt (EASY Backfilling,
Lifka 1995). Trainings und Suchen auf den Messhologrammen nehmen bis zu 3 Threads, wenn kein
anderer Job wartet; bisher lief jeder Job mit einem Thread, und jede `run.yaml` hält die Zahl
fest. Jeder Job hat sein Log, seinen Exit-Code und seinen höchsten Speicher; ein neu gestarteter
Runner übernimmt laufende Jobs, `retry` gibt einen fehlgeschlagenen frei. `forecast` spielt den
Rest des Plans mit denselben Regeln durch und gibt das Ende jeder Gruppe in UTC und Berliner Zeit
aus; daraus stammt die Zeit in der Beschreibung des PR.

### 3.7 Auswertung (`ce168d8`, `121683d`, `05e27ac`)

- `texture`: die Sätze der Anpassung jeder Probe ohne Textur, mit der Textur von Etappe 6 und mit
  der von Etappe 10 gegen das Messhologramm: Pegel, Bänder der CTF, die Leistung, die die Textur
  hinzufügt, im Verhältnis zur Messung (`results/stage10/texture-<probe>.json`).
- `spectra`: das Messhologramm unter den Test-Splits von `loo` und `loo2`, Bin für Bin als
  Perzentil (`results/stage10/spectra-<probe>.json`).
- `compare`: die gelernten Schätzungen der vier Familien (Seeds und Ensemble) auf den 48
  Samples, dem ganzen Test-Split von cpu-4x, den Test-Splits von `loo2` und den Messhologrammen
  gegen [2] und Nelder–Mead kalt; die Suchen auf beiden mit Fehler, Anteil innerhalb 100 µm,
  Rekonstruktionen und den Samples, deren Wahrheit außerhalb des Intervalls lag; Paare von
  Suchen Sample für Sample (`search_pair` aus Etappe 9) und Paare von Familien (E14).
- `evaluate`: dieselben Kennzahlen für Läufe, die man über ihren Pfad nennt, etwa die von
  Maxwell; die Intervalle einer Suche findet es in ihrer `run.yaml` und prüft sie über SHA-256.
  `mixture` mischt CNN-Seeds, die über ihren Pfad gegeben sind.
- `scripts/make_stage10_figures.py`: die Eingaben, Textur, Spektren, gelernte Kennzahlen,
  Intervalle auf den Messhologrammen und die Suchen (`docs/laborbuch/figures/stage10-*.png`).

### 3.8 Was auf die GPU geht (`6a04c72`)

`docs/maxwell-gpu.md`, Abschnitt 8.5: die `loo2`-Sätze mit 50 000 weiteren Trainings-Samples,
die vier Familien mit drei Seeds auf diesen 56 000, auf cpu-4x mit 60 000 Samples dieselben
Familien und zwei Varianten, die die CPU der VM nicht trägt – das polare Spektrum bis
$f^2 = 0{,}25$ mit 2048 Bins (`polar-cnn-hf.yaml`) und das Bild-CNN ohne Binning auf 512²
(`image-cnn-b1.yaml`) –, Brent kalt und in den Intervallen auf allen 250 Test-Samples statt auf
48, und die Auswertung dieser Läufe auf der VM mit `evaluate` (E11).

### 3.9 Tests

- `tests/test_learned.py`: die Configs der Etappe (`polar-cnn.yaml` ist `spectrum-cnn-wide.yaml`
  mit dem polaren Spektrum, `polar-cnn-hf.yaml` und `image-cnn-b1.yaml` unterscheiden sich von
  ihren Vorlagen nur in der Darstellung; eine ungerade Zahl von Sektoren oder 2 werden
  abgelehnt); das polare Spektrum (standardisiert; unter den acht Drehungen und Spiegelungen
  des Hologramms Verschiebung der Sektoren um `angles / 2` und Umkehr; Streifen in vier
  Richtungen im richtigen Sektor und Bin; keine leere Zelle auf 472² und 512², auf zu kleinen
  Rastern ein Fehler); das polare CNN invariant unter jeder zyklischen Verschiebung und Umkehr
  der Sektoren, auch durch die Darstellung hindurch, aber nicht unter der Änderung eines
  Sektors; ein kleines polares CNN von Ende zu Ende trainiert, geladen und ausgewertet.
- `tests/test_npe.py`: die Einbettung auf dem polaren Spektrum, invariant unter Umkehr der
  Sektoren. `tests/test_objectmodel.py`: das Maß mit Pegeln (zehn Terme, eine Skala Abweichung
  zählt $\sqrt{1/10}$, ohne `levels` wie in Etappe 6) und der Median der Pegel.
  `tests/test_sim2real.py`: `upper_snr = level_upper − floor`.
- `tests/test_run_jobs.py`: die Auswahl im Budget mit EASY Backfilling, ein Job, der genau in
  den Rest passt, freie Threads für einen elastischen Job, wenn keiner wartet; die Vorhersage
  mit denselben Regeln; ungültige Pläne; der Runner von Ende zu Ende: Übernahme eines laufenden
  Jobs, schon vorhandene Ausgaben, Exit 0 ohne Ausgabe als Fehler, `clean` vor dem Start, die
  Threads in `OMP_NUM_THREADS`, blockierte Abhängige und `retry`.
- `tests/test_cluster_kit.py`: die Kette aus `docs/maxwell-gpu.md` 8.5 mit `--local` auf dem
  CI-Datensatz (Seeds, `mixture`, Brent in den Intervallen in zwei Stücken, `merge`,
  `evaluate`).
- CI (`35c6fbe`): `configs` gibt die committeten `loo2`-Configs bytegleich; das CNN des
  Rauchtests, mit sich selbst gemischt, hat denselben MAE und dieselbe Abdeckung wie das CNN;
  Brent in den Intervallen der Mischung hat denselben MAE wie Brent in denen des CNN in
  Etappe 9, und `evaluate` findet über SHA-256 die Intervalle, in denen die Suche lief; der Plan
  und seine Vorhersage.

## 4 Entscheidungen und Begründung

**E1 – Die Pegel ins Maß, nicht ein neues Texturmodell.** Das Texturmodell von Etappe 6 hat drei
Größen (RMS innen, Film, Steigung des Potenzgesetzes), und die Anpassung fand mit ihnen Formen,
die die CTF-Bänder treffen; sie hatte nur keinen Grund, die Leistung klein zu halten, weil das
Maß nur Formen vergleicht. Die Pegel dreier Bänder ins Maß zu nehmen ändert nichts am Modell und
an den Sätzen außer den Werten; ein Modell mit eigener Grenzfrequenz oder Band hätte eine weitere
Größe gebraucht, die unter Leave-one-out nur gelten darf, wenn eine andere Probe sie auch zeigt
(Laborbuch 06, 6, Punkt 2). Den Boden ließ Etappe 6 aus dem Maß, weil der Flatfield-Rest ihn bei
Spinnenhaar und Zahn hebt (`FIT_SCALES`). Als Pegel gehört er dazu, weil die Textur von Etappe 6
gerade bei den höchsten Frequenzen zu viel Leistung hatte (2); dass er bei Spinnenhaar und Zahn
schon ohne Textur 0,33 und 0,39 über der Messung liegt, gilt für jede Textur dieser Proben, und
das Maß bestraft dort eine Textur für jede Leistung, die sie bei den höchsten Frequenzen
dazulegt.

**E2 – `LEVEL_SCALE` = 0,5.** Eine halbe Einheit der Log-Leistung zählt so viel wie 0,1 der
CTF-Korrelation (`FIT_SCALES`). Der Boden lag mit der Textur von Etappe 6 bei Kaktusnadel und
Mg-Draht 1,2 und 2,9 Einheiten über der Messung (2), das sind 2,4 und 5,8 Skalen und damit ein
Abstand, der im Maß deutlich zählt; Abweichungen der Pegel um wenige Zehntel sollen die Form
dagegen nicht überstimmen.

**E3 – Dasselbe Gitter und dieselben Samples wie in Etappe 6.** So zeigt der Vergleich mit
Etappe 6, was das Maß ändert und nicht, was ein anderes Gitter ändert. Die Texturen von Etappe 6
sind bei Kaktusnadel und Mg-Draht nicht unter den fein gerechneten Punkten; wie sie unter dem
neuen Maß liegen, rechnet `texture` auf denselben Samples (5.1).

**E4 – Neue Sätze mit eigenem Seed.** Die `loo`-Sätze von Etappe 6 bleiben, wie sie sind; die
Läufe der Etappen 6 bis 9 bleiben damit nachrechenbar. `loo2` hat dieselbe Größe wie `loo`
(6000/250/250), damit die Familien mit denselben Zahlen lernen wie das CNN und der NPE dort.

**E5 – Acht Sektoren mit linearen Gewichten.** Mit 8 Sektoren zu 22,5° bleibt eine Faser in ein
bis zwei Sektoren, und 8 × 1024 Zellen kosten das Achtfache der Eingabe des radialen CNN. Ordnet
man jede Frequenz nur dem nächsten Sektor und Bin zu, bleiben nahe $f^2 = 0{,}125$ Zellen leer:
Ein Bin von 1024 ist dort ein Ring von 0,09 Pixeln Breite, dessen wenige Frequenzen des Rasters
auf 512² und 472² nicht jeden Sektor erreichen. Mit linearen Gewichten in Winkel und $f^2$
bekommt jede Zelle Beiträge (auf 512² mindestens 8 Frequenzen bis $f^2 = 0{,}25$, auf 472²
mindestens 4), und die Verschiebung unter Drehungen um 90° bleibt exakt, für jede gerade Zahl von
Sektoren.

**E6 – Invarianz durch Bauart statt Augmentierung.** Das Bild-CNN lernt die Diedergruppe aus
zufälligen Drehungen und Spiegelungen (Etappe 4). Beim polaren Spektrum sind beides
Permutationen der Sektoren; eine Faltung, die zyklisch und symmetrisch im Winkel ist, und
Mittel und Maximum über die Sektoren machen die Ausgabe exakt invariant (Test, 3.9), ohne dass
das Netz Trainingszeit dafür braucht. Danach folgen die Stufen des radialen CNN, damit der
Vergleich der beiden Spektren nicht am Netz hängt. Die Schrittweite 2 der ersten Stufe ersetzt
das Max-Pooling dahinter; auf allen 8 Sektoren hätte es doppelt so viel Zeit gekostet wie das
ganze radiale CNN.

**E7 – Das Bild-CNN 2× gebinnt.** Ohne Binning (512²) hat es viermal so viele Pixel: 512 KB
pro Sample als float16, auf den 10 000 Samples von cpu-4x 5,2 GB, und gebinnt brauchte ein
Training schon das 1,7- bis 1,9-Fache seiner Eingaben (2,2–2,5 GB für 1,3 GB, E13); das ginge
über das Budget von 8 GB. Die Zeit wüchse mit den Pixeln auf grob 2,3 s pro Trainings-Sample,
für drei Seeds auf cpu-4x etwa 19 h mit einem Thread (geschätzt, nicht gemessen). 2× gebinnt
(256²) ist die Config von Etappe 4, deren Seed 0 auf cpu-4x es gibt. Die Variante ohne Binning
steht in `docs/maxwell-gpu.md` (E11).

**E8 – Drei Seeds pro Familie, vorhandene Seeds wiederverwendet.** Der Vergleich gilt Familie
gegen Familie, Ensemble gegen Ensemble und Seed gegen Seed (Laborbuch 08, 8); ein Ensemble aus
drei Seeds ist der kleinste Satz, der die Streuung zwischen Seeds zeigt. Vorhandene Seeds auf
cpu-4x habe ich übernommen, wo Dateien, Config und Seed gleich sind (geprüft über ihre
`run.yaml`); den NPE-Seeds 3 und 4 von Etappe 7 nehme ich nicht dazu, damit jede Familie drei
hat. Ein NPE auf dem polaren Spektrum oder dem Bild wäre eine fünfte und sechste Familie; die
Frage dieser Etappe ist die Eingabe, und die vergleicht das CNN auf allen drei.

**E9 – Was „fair“ heißt.** Alle Familien lernen auf denselben Dateien mit denselben Seeds und
werden auf denselben Splits ausgewertet; die CNNs kalibrieren ihr $\hat\sigma$ auf demselben
Val-Split; die Ensembles mischen dieselbe Zahl von Seeds; die Suchen nehmen für jede Familie
dasselbe Niveau (95 %), dieselbe Config und dieselben 48 Samples. Auf den Messhologrammen schätzt
jede Familie mit Modellen, die das Hologramm nicht gesehen haben (`loo2`). Verschieden bleiben
nur Eingabe und Netz – und die Zahl der Epochen des Bild-CNN (40 statt 100, wie in Etappe 4).

**E10 – Die kalten Suchen nicht neu.** Nelder–Mead kalt (Etappen 3 und 5) und Brent kalt
(Etappe 9) sind auf denselben Samples und Hologrammen gerechnet; was sich seitdem geändert hat,
ist nur das Intervall. Neu zu rechnen hätte rund 38 Kernstunden gekostet (auf den 48 Samples
927 Rekonstruktionen zu 92 s mit einem Thread, auf den Messhologrammen 79 zu etwa
630 Kernsekunden) und nichts geändert: Brent kalt rechnet schon mit `one_thread`, das mit jeder
Threadzahl dieselben Bits gibt (Laborbuch 09, E1), und Nelder–Mead kalt endet mit dieser Numerik
auf allen vier Messhologrammen und auf den acht Samples der Wiederholung am selben $z_{01}$ wie
vorher (Laborbuch 09, 5.5, Punkt 6, und 5.6, Punkt 5). Die kalten Suchen mit Nelder–Mead behalten
damit ihre Numerik 3 (Laborbuch 09, E2); MFE-Werte vergleiche ich deshalb nur mit Brent kalt.

**E11 – Was auf die GPU geht.** Hier läuft, was auf 4 Kernen in etwa einem Tag Platz hat: die
`loo2`-Sätze mit 6000 Trainings-Samples, das Bild-CNN 2× gebinnt, das polare Spektrum bis
$f^2 = 0{,}125$, die Suchen auf den 48 Samples von Etappe 3. Auf Maxwell kommen dazu, was diese
Zahlen zu einer Aussage über größere Datenmengen macht (56 000 und 60 000 Samples) und die beiden
Eingaben, die mehr von den hohen Frequenzen sehen, um die es in Teil 1 geht (`polar-cnn-hf`,
`image-cnn-b1`); dazu Brent auf allen 250 Test-Samples, damit 6 von 48 nicht die einzige Zahl
für die Abdeckung bleibt.

**E12 – Ein Runner statt Lanes.** In Etappe 9 liefen die Suchen in zwei Lanes, die ich von Hand
belegt habe. Hier hängen 388 Jobs voneinander ab (eine Suche braucht das Ensemble, das die
Vorhersagen dreier Trainings braucht, die einen Satz brauchen), mit Laufzeiten von Sekunden bis
sieben Stunden. Ohne Backfilling ließ die Vorhersage des ersten Plans die Kerne lange halb leer
(28,0 h für 94,7 Kernstunden, die auf 4 Kernen 23,7 h bräuchten); mit EASY Backfilling waren es
24,2 h. Die Suchen auf den synthetischen Samples haben die niedrigste Priorität und füllen die
Lücken.

**E13 – Das Budget.** Die VM hat 16,4 GB; ohne die Anpassungen standen etwa 10,6 GB zur
Verfügung. Mit 8 GB für alle Jobs und 1 GB Reserve bleibt Platz für eine kurze Prüfung daneben.
Die Eingaben eines Trainings liegen im Arbeitsspeicher und beim Zusammenfügen der Dateien kurz
doppelt: Das Bild-CNN auf cpu-4x brauchte bis zu 2,5 GB, das polare CNN 1,1 GB.

**E14 – Statistik.** Anteile mit Wilson-Intervall (Wilson 1927), MAE und Differenzen mit
Bootstrap-Intervall (10 000 Ziehungen, wie in Etappe 9), gepaarte Anteile mit exaktem
McNemar-Test auf den Samples, die nur eine von zwei Methoden innerhalb 100 µm schätzt. Auf vier
Messhologrammen gibt es keine Statistik, nur die Zahlen.

## 5 Ergebnisse

## 6 Offene Punkte

## 7 Nächste Schritte

## 8 Bezug zur Masterarbeit

Reproduktion: `docs/usage.md`, Abschnitt „Stage 10: beyond the radial power spectrum“.
