# Laborbuch 09 – Etappe 9: Die Stufen der MFE und Brents Verfahren neben Nelder–Mead

| | |
|---|---|
| Datum | 22.–23.10.2026 |
| Branch | `cursor/stage9-mfe-steps-brent-1091` |
| Code | `one_thread` in `src/holofocus/device.py`, damit in `src/holofocus/asrm.py`, `src/holofocus/propagation.py` und `src/holofocus/autofocus/criteria.py` die Produkte komplexer Felder, die Norm von $\mathrm{Im}\,y$ und der Verlust auf einem Thread laufen; `method="brent"` in `find_focus` und `search_points` (`src/holofocus/autofocus/search.py`), `search.method` in `BaselineConfig`, `Box` und `--box`/`--box-level` für `holofocus-baseline run` (`src/holofocus/baseline/`), `--method` in `scripts/run_stage3_experiments.py measured`; `configs/baseline/paper2-cpu-brent.yaml`, `paper2-cpu-box.yaml`, `ci-brent.yaml`; `scripts/run_stage9_study.py` (`sources`, `queue`, `merge`, `scan`, `steps`, `compare`), `scripts/make_stage9_figures.py`; Erweiterungen von `tests/test_asrm.py`, `tests/test_autofocus.py` und `tests/test_baseline.py` |
| Ergebnisse | `results/stage9/` (Scans, `steps.json`, `mfe-points.csv`, `steps-objects.json`, `compare.json`, `measured.csv`, `synthetic.csv`, `lanes.csv`, Intervalle des CNN), `results/measured/brent-b1-p4/`, `results/measured/brent-box95-{cnn,npe}-loo-ensemble-b1-p4/`, `results/measured/b1-p4-repeat/`, `results/baseline/cpu-4x-test-{brent,brent-box95-npe-ensemble,box95-npe-ensemble,brent-box95-cnn,box95-cnn,repeat}/` |
| Abbildungen | `docs/laborbuch/figures/stage9-*.png`, Zahlen in `stage9-metrics.json` |
| Quellen | [2] Dora et al., Opt. Express 33(4), 6641 (2025); R. P. Brent, *Algorithms for Minimization without Derivatives*, Prentice-Hall (1973), Kap. 5 |

Kurz: ‹TODO›

## 1 Ziel

Zwei Fragen aus Etappe 8, auf den vier Messhologrammen und, für die zweite, auch auf den 48
synthetischen Samples der Baseline von Etappe 3:

1. **Die Stufe der MFE beim Zahn** (Laborbuch 08, 6, Punkt 9). Bei etwa 80,69 mm springt die
   MFE des Zahns um etwa 680 ppm, und ob ein Punkt davor oder dahinter landete, hing an der Zahl
   der Threads (Laborbuch 08, 5.6, Punkt 9). Wechselt die Rekonstruktion dort zwischen zwei
   Lösungen? Haben die anderen Proben solche Stufen? Die MFE soll von der Zahl der Threads
   unabhängig werden, wenn das geht; sonst ist die Threadzahl für jeden Vergleich von
   MFE-Werten festzulegen und zu dokumentieren.
2. **Brents Verfahren in Grenzen als klassische Suche neben Nelder–Mead** (Laborbuch 08, 7,
   Punkt 1). Nelder–Mead bleibt das Verfahren von [2]. Brent läuft kalt in
   $\Omega_z = $ Schätzwert ± 5 mm und begrenzt auf das 95-%-Intervall eines gelernten
   Ensembles: auf den vier Messhologrammen gegen die volle Suche mit Nelder–Mead und die
   Referenzen von [2], mit dem CNN- und dem NPE-Ensemble von Etappe 8; auf den 48 synthetischen
   Samples von Etappe 3, deren Wahrheit bekannt ist, mit den Modellen, die auf diesen Daten
   trainiert sind. Berichtet werden Fehler, Anteil innerhalb 100 µm, Rekonstruktionen und Zeit.
   Die Hauptfrage: Spart ein gelerntes Intervall mit Brent mehr als mit Nelder–Mead?

Alles andere steht als offener Punkt in Abschnitt 6. Keiner der beiden Teile braucht die GPU
(E12).

## 2 Ausgangslage

Etappe 8 ist in `main` gemergt
([PR #16](https://github.com/AliTUHH/master_thesis_autofocus/pull/16), Merge-Commit
`4349d6a`); dieser Branch geht von dort aus.

**Die Stufe.** Mit drei Threads wiederholt, gab eine Rekonstruktion bitgenau dieselbe MFE, mit
einem Thread eine andere: bei Kaktusnadel und Mg-Draht 4–9 ppm, beim Zahn bei 80,6926 mm
675 ppm (Laborbuch 08, 5.6, Punkt 9). Über der kleinsten MFE mit drei Threads, bei 80,663 mm,
lagen 80,678 mm bei 29 ppm, 80,6926 mm bei 677 ppm und 80,696 mm bei 706 ppm; mit einem Thread
lag 80,6926 mm noch in der Senke. Die Suchen auf den Messhologrammen liefen alle mit drei
Threads, bis auf eine:

| Suche (`results/measured/`) | Etappe | Threads |
|---|---|---|
| `b1-p4`, kalt in $\Omega_z$ | 3 (Spinnenhaar, Zahn), 5 (Kaktusnadel, Mg-Draht) | 3 |
| `warm-b1-p4`, Warmstart vom CNN von Etappe 5 | 5 | 3 |
| `warm-{cnn,npe}-loo-ensemble-b1-p4` | 8 | 3 |
| `box95-npe-loo-ensemble-b1-p4` | 8 | 3 |
| `box95-cnn-loo-ensemble-b1-p4` | 8 | 1 |

**Die Suchen auf den Messhologrammen** (Laborbuch 08, 2 und 3.7). Kalt, mit Nelder–Mead in
$\Omega_z$: Spinnenhaar 79,4875 mm (13 Rekonstruktionen), Zahn 80,6344 mm (11), Kaktusnadel
284,9281 mm (13), Mg-Draht 471,2281 mm (12). Die Referenzen von [2] (Nelder–Mead / visuell):
79,4 / 79,4, 80,9 / 81,0, 284,2 / 284,2 und 470,8 / 470,7 mm. Die 95-%-Intervalle der beiden
Ensembles, in mm (Breite):

| Probe | $\Omega_z$ | CNN-Ensemble | NPE-Ensemble |
|---|---|---|---|
| Spinnenhaar | 74,8–84,8 | 77,826–80,772 (2,95) | 78,781–80,184 (1,40) |
| Zahn | 77,9–87,9 | 78,834–82,439 (3,61) | 78,595–82,634 (4,04) |
| Kaktusnadel | 281,1–291,1 | 282,954–289,988 (7,03) | 282,251–290,122 (7,87) |
| Mg-Draht | 467,4–477,4 | 467,400–473,247 (5,85) | 467,636–475,433 (7,80) |

Nelder–Mead darin sparte 14 % (NPE-Ensemble) und 16 % (CNN-Ensemble) der Rekonstruktionen
(Laborbuch 08, 5.6).

**Die synthetische Baseline von Etappe 3** (`results/baseline/cpu-4x-test`, Laborbuch 03): die
ersten 48 Samples des Test-Splits von cpu-4x (512² px, 4× gebinnt), Nelder–Mead nach [2] mit
`configs/baseline/paper2-cpu.yaml`, 3 Threads. MAE 166 µm, Median 124 µm, 31 % innerhalb
100 µm, 11,1 Rekonstruktionen (9–13), 29 s pro Rekonstruktion.

**Die gelernten Modelle auf cpu-4x.** Das NPE-Ensemble von Etappe 7
(`results/npe/npe-nsf-ensemble`, fünf Seeds, Mischung der Dichten) gibt zentrale Intervalle;
auf den 250 Test-Samples deckt sein 95-%-Intervall 96,8 % bei 894 µm Breite im Median
(Laborbuch 07). Vom Gauß-CNN von Etappe 4 (`results/learned/spectrum-cnn-wide`, auf dem
Val-Split nachkalibriertes $\hat\sigma$) gibt es auf cpu-4x nur ein Netz, kein Ensemble (E6).

**Brent auf Modellkurven** (Laborbuch 08, 7, Punkt 1). Von denselben Intervallen aus braucht
Brents Verfahren auf einer Parabel ab 1 mm Breite im Mittel 5,5–6,0 Auswertungen, Nelder–Mead
5,3–11,2; auf den MFE-Kurven der Messhologramme ließ das etwa 6 statt 12–13 Rekonstruktionen
bei Kaktusnadel und Mg-Draht erwarten und 9 statt 11 beim Zahn. Gewarnt war vor der Stufe des
Zahns: An ihr kann eine Parabel durch drei Punkte weit danebenliegen.

## 3 Was gemacht wurde

### 3.1 Die MFE ohne Abhängigkeit von der Threadzahl (`e809b07`)

Welche Operationen von der Threadzahl abhängen, habe ich einzeln geprüft: jede Operation der
ASRM-Schleife auf Feldern von 1024² und 2048² Elementen mit 1, 2, 3 und 4 Threads, Bit für Bit
gegen einen Thread (torch 2.14.1 für die CPU, MKL, AVX-512; nicht versioniert). Zwei Arten
geben andere Bits:

- **Volle Reduktionen** (`mean`, `norm`): Jeder Thread summiert seinen Teil, die Teilsummen
  werden danach addiert; mit einer anderen Threadzahl ist die Reihenfolge der Additionen eine
  andere.
- **Produkte komplexer Felder** (`a * b` mit komplexem `a` und `b`): Jeder Thread rechnet seinen
  Abschnitt mit Vektorbefehlen, die letzten Elemente eines Abschnitts aber skalar mit fused
  multiply-add, also anders gerundet. Mit drei Threads enden die Abschnitte von $2^n$ Elementen
  nicht auf einer Vektorgrenze, und einige Elemente pro Feld wechseln den Pfad.

FFTs, `exp`, `abs`, Wurzel, Division und die reellen elementweisen Operationen geben mit jeder
Threadzahl dieselben Bits. Ein Unterschied im letzten Bit weniger Elemente breitet sich über die
FFTs auf das ganze Feld aus und wächst über 1000 Iterationen bis zu den ppm der MFE.

`one_thread()` (`src/holofocus/device.py`) setzt die Threadzahl von torch für die eingeschlossenen
Operationen auf eins und danach zurück. Darin laufen jetzt die Produkte mit der
Übertragungsfunktion und ihrer Konjugierten und das Produkt mit $-i\,\overline{\psi}$ in
`data_gradient` (an Ort und Stelle, `_times`), die Norm von $\mathrm{Im}\,y$ für den Term mit
$\beta$ und der Verlust jeder Iteration in `asrm_reconstruct`, das Produkt der beiden
Faktoren des Fresnel-Kerns und das Produkt mit ihm in `FresnelPropagator` und der Mittelwert in
`model_fit_error`. Die FFTs bleiben auf allen Threads. Die Operationen in `one_thread()` sind
durch den Speicher begrenzt, nicht durch die Rechnung: Mit einem Thread sind sie kaum langsamer
als mit drei.

Geprüft auf drei Wegen:

1. Synthetisches Sample 0 des Test-Splits von cpu-4x, volle ASRM bei 100 mm (nicht versioniert):
   Der alte Code gibt mit einem Thread die MFE 2,0915255299769342·10⁻⁴, mit drei Threads
   2,0915214554406703·10⁻⁴ (−1,9 ppm). Der neue Code gibt mit 1, 2, 3 und 4 Threads dieselbe MFE
   wie der alte mit einem Thread, dazu dieselbe Verlustkurve und dasselbe Objekt (MD5 über die
   Bytes). Die Rekonstruktion dauert mit drei Threads 30,2 statt 28,5 s (+6 %), mit einem Thread
   76,8 wie vorher.
2. Der Zahn bei 80,69255535982456 mm, dem Punkt von Laborbuch 08, 5.6, Punkt 9, mit dem neuen
   Code und drei Threads: MFE 3,160280757583678·10⁻⁴, bitgleich mit der Rekonstruktion mit einem
   Thread aus Etappe 8 (`box95-cnn-loo-ensemble-b1-p4`). Versioniert noch einmal in
   `results/stage9/threads-3/` gegen den Scan mit einem Thread (`repeats` in `steps.json`, 5.1).
3. `tests/test_asrm.py`: eine kleine volle Rekonstruktion mit 1, 2 und 3 Threads gibt dieselbe
   Verlustkurve, dasselbe Objekt, denselben Kern, dasselbe propagierte Feld und dieselbe MFE nach
   Eq. (15).

Damit ist die MFE seit dieser Etappe die eines Laufs mit einem Thread, mit jeder Threadzahl.
Rekonstruktionen von vorher mit einem Thread gelten bitgenau weiter, die mit drei Threads nicht
(E1, E2).

### 3.2 Brents Verfahren (`e809b07`)

`find_focus` und `search_points` nehmen `method="brent"`: Statt Nelder–Mead sucht
`scipy.optimize.minimize_scalar(method="bounded")` in demselben Intervall ($\Omega_z$ oder
`bounds`) und in denselben Einheiten (Millimeter), mit `xatol` = 100 µm. Das Verfahren wechselt
zwischen Schritten des Goldenen Schnitts und Parabeln durch die drei besten bisherigen Punkte
und beginnt im Intervall, beim 0,382-fachen seiner Breite, nicht an seinen Enden. Es hört auf,
wenn der beste Punkt auf etwa ±⅔ `xatol` eingeschlossen ist, und wertet keinen Punkt näher als
⅓ `xatol` am bisher besten aus. Einen Anfangssimplex gibt es nicht; ein Punkt, der einem früheren bis auf
`revisit_tolerance` gleicht, gibt dessen MFE zurück statt des Strafwerts von Nelder–Mead, weil
Brent die Werte in eine Parabel steckt.

`BaselineConfig` hat `search.method` (Vorgabe `nelder-mead`; in `to_dict` nur, wenn es Brent
ist, damit die Läufe von vorher ihren Schlüssel behalten). `configs/baseline/paper2-cpu-brent.yaml`
ist `paper2-cpu.yaml` mit Brent, `paper2-cpu-box.yaml` Nelder–Mead wie dort; beide ohne die
Rekonstruktion bei der Wahrheit, die nicht von der Suche abhängt und in den Läufen von Etappe 3
steht. `run_stage3_experiments.py measured` nimmt `--method brent`.

### 3.3 Suchintervalle auf den synthetischen Samples (`e809b07`)

`holofocus-baseline run --box CSV --box-level 95` sucht jedes Sample in seinem Intervall
`z01_lower_95`–`z01_upper_95` aus der Tabelle, geschnitten mit $\Omega_z$ (`Box` in
`src/holofocus/baseline/runner.py`); Nelder–Mead beginnt an den Enden dieses Intervalls, Brent
darin. Quelle, SHA-256 und Niveau der Tabelle gehören zum Schlüssel eines Laufs. Die Intervalle:

- **NPE-Ensemble:** `results/npe/npe-nsf-ensemble/test/predictions.csv` hat die zentralen
  Intervalle der Mischung.
- **CNN:** `run_stage9_study.py sources` schreibt $\hat z_{01} \pm 1{,}96\,\hat\sigma$ (und
  $\pm 0{,}99\,\hat\sigma$ für 68 %) des Gauß-CNN nach `results/stage9/intervals-cnn-test.csv`.

### 3.4 Scans und Stufen (`run_stage9_study.py scan`, `steps`)

`scan` rekonstruiert ein Messhologramm an gegebenen $z_{01}$, mit den Einstellungen der Suchen
(ungebinnt, Padding 4, Plan von [2] Table 3), und hält pro Punkt fest: MFE, Verlust jeder
10. Iteration, Verluste am Ende der Stufen des Plans und Kennzahlen des Objekts
(`results/stage9/scan-<probe>.json`). Objekt und ganze Verlustkurve gehen nach
`results/stage9/scan-<probe>/<z01 in mm>.npz` (nicht versioniert, 1,8 MB pro Punkt). Mit `--z01`
in Metern trifft ein Scan bitgenau den Punkt einer früheren Suche. Für die Numerik von vorher
läuft dasselbe Skript in einem Checkout von `4349d6a` mit drei Threads
(`results/stage9/threads-3-4349d6a/`).

Zu jedem Punkt hält `scan` außerdem den **Verlust am zurückgegebenen Objekt** fest
(`object_loss`): den ersten Verlust einer weiteren Iteration der letzten Stufe, gestartet vom
Objekt ohne Impuls. Die MFE ist dagegen der Verlust am Vorgriff der letzten Iteration,
$y = \tilde O - \gamma v$, mit dem Impuls $v$ und $\gamma = 1$ in allen drei Stufen von [2]
Table 3; das zurückgegebene Objekt ist der Schritt danach. Den Rand außerhalb des Sichtfelds
baut jede Iteration aus dem Sichtfeld neu auf (gespiegelt und ausgeblendet), den Impuls nicht.
Dass `object_loss` den Verlust der Schleife trifft, prüft eine Rekonstruktion mit einer
Iteration vom gespeicherten Objekt aus: bitgleich (nicht versioniert).

`scan --stages 1` rekonstruiert nur die **erste Stufe** des Plans (Sichtfeld 16-fach
verkleinert, beim Zahn 118² px in 512² px, 700 Iterationen), mit einem Thread in etwa 15 s
statt 9 min. Ihr Verlust am Ende ist bitgleich mit `stage_losses[0]` der vollen Rekonstruktion
am selben $z_{01}$; das Objekt geht mit dem ganzen Feld samt Rand nach
`results/stage9/first-stage/scan-<probe>/` (nicht versioniert). Gescannt ist die erste Stufe
beim Zahn alle 10 µm in 80,50–80,90 mm und alle 1 µm in 80,700–80,730 mm, mit der Numerik von
vorher alle 2 µm in 80,660–80,700 mm und alle 50 µm in 80,50–80,90 mm, und bei allen vier
Proben in ganz $\Omega_z$ alle 200 µm.

`steps` sammelt alle Rekonstruktionen der Messhologramme mit Padding 4 (die Suchen der Etappen
3, 5, 8 und 9 und die Scans) als MFE über $z_{01}$, getrennt nach Numerik (E2), und
sucht darin Stufen (E10): einen Sprung zwischen Nachbarn höchstens 50 µm auseinander, um mehr als
50 ppm und mindestens dreimal so steil wie beide benachbarten Sekanten; oder einen Abfall um mehr
als 50 ppm vom Minimum weg (Umkehr). Dazu, wie dicht die Punkte um das Minimum liegen (Anteil von
±250 und ±500 µm, in dem Nachbarn höchstens 50 µm auseinanderliegen), dieselbe Prüfung mit beiden
Numeriken zusammen, die Rekonstruktionen am selben $z_{01}$, wo die Verlustkurven zweier
Scans auseinanderlaufen, und die Objekte der Scans gegeneinander (`results/stage9/steps.json`,
`mfe-points.csv`, `steps-objects.json`).

### 3.5 Die Suchen

Auf den Messhologrammen (`results/measured/`, alle mit 3 Threads in Lane A, ungebinnt, Padding 4):

| Verzeichnis | Verfahren | Intervall |
|---|---|---|
| `brent-b1-p4` | Brent | $\Omega_z$ |
| `brent-box95-cnn-loo-ensemble-b1-p4` | Brent | 95-%-Intervall des CNN-Ensembles ∩ $\Omega_z$ |
| `brent-box95-npe-loo-ensemble-b1-p4` | Brent | 95-%-Intervall des NPE-Ensembles ∩ $\Omega_z$ |
| `b1-p4-repeat` | Nelder–Mead | $\Omega_z$: `b1-p4` noch einmal, mit der Numerik dieser Etappe (E8) |

Die Suchen mit Brent mit `--revisit-tolerance 1e-9` wie die von Etappe 8, `b1-p4-repeat` ohne,
wie `b1-p4`.

Auf den synthetischen Samples (`results/baseline/cpu-4x-test-<Name>`):

| Name | Konfiguration | Intervall | Samples |
|---|---|---|---|
| `brent` | `paper2-cpu-brent` | $\Omega_z$ | 0–47 |
| `brent-box95-npe-ensemble` | `paper2-cpu-brent` | 95 % des NPE-Ensembles | 0–47 |
| `box95-npe-ensemble` | `paper2-cpu-box` | 95 % des NPE-Ensembles | 0–47 |
| `brent-box95-cnn` | `paper2-cpu-brent` | 95 % des CNN | 0–47 |
| `box95-cnn` | `paper2-cpu-box` | 95 % des CNN | 0–47 |
| `repeat` | `paper2-cpu` | $\Omega_z$ | 11, 46, 0–5 (E8) |

Das sind 248 Suchen. `run_stage9_study.py queue` lässt zwei Lanes (A mit 3, B mit einem
Thread) die nächste Suche nehmen, die noch keine genommen hat (eine Datei pro Suche, angelegt
mit `open(..., "x")`); jede schreibt nach `runs/stage9/cpu-4x-test-<Name>/lane-<Lane>`, und
`merge` fügt die Lanes zusammen und schreibt nach `results/stage9/lanes.csv`, welche Lane welche
Suche mit wie vielen Threads rechnete.

### 3.6 Der Vergleich (`run_stage9_study.py compare`)

`compare` stellt pro Messhologramm alle Suchen nebeneinander (Ergebnis, Abstand zu den Referenzen
von [2] und zur kalten Suche, MFE, Rekonstruktionen, Zeit) und fasst die synthetischen Suchen
zusammen: MAE mit Bootstrap-Intervall, Median, 90. Perzentil, Maximum, Anteil innerhalb 50, 100,
250 und 1000 µm, Rekonstruktionen, Zeit mit 3 Threads; dazu, was das Intervall pro Verfahren
spart, gepaart über die Samples (E9; `results/stage9/compare.json`, `measured.csv`,
`synthetic.csv`).

### 3.7 Tests

‹TODO: Zahl der Tests, ruff, CI-Schritt›

## 4 Entscheidungen und Begründung

**E1 – Die MFE unabhängig von der Threadzahl machen, statt die Threadzahl festzulegen.** Es geht:
Nur zwei Arten von Operationen hängen an der Threadzahl (3.1), beide sind durch den Speicher
begrenzt, und auf einem Thread verlängern sie eine Rekonstruktion mit drei Threads um 6 %.
Eine feste Threadzahl hätte jeden Vergleich an eine Einstellung der Maschine gebunden; die Lane
mit einem Thread neben der mit drei (Regel der VM) hätte nie dieselbe MFE gerechnet.

**E2 – Die Numerik eines Threads.** Statt die Reduktionen und Produkte auf einem Thread zu
rechnen, ginge auch eine eigene, von der Threadzahl unabhängige Summe (paarweise oder in
doppelter Genauigkeit) und ein komplexes Produkt aus reellen Operationen. Beides gäbe eine dritte
Numerik, verschieden von beiden alten, und das Produkt aus reellen Operationen kostete auf 2048²
Elementen mit drei Threads 18–22 ms, das von torch auf einem Thread 5,9 ms (mit drei Threads
6,1 ms, durch den Speicher begrenzt). Mit einem
Thread gelten alle Rekonstruktionen von vorher mit einem Thread bitgenau weiter: in Etappe 8 die
Suche im Intervall des CNN-Ensembles. Die mit drei Threads (die übrigen Suchen auf den
Messhologrammen und die Baseline von Etappe 3) behalten ihre Numerik und heißen hier
„Numerik 3“, alles mit einem Thread und alles seit dieser Etappe „Numerik 1“. MFE-Werte vergleiche
ich nur innerhalb einer Numerik; Lagen ($z_{01}$ eines Ergebnisses) über beide.

**E3 – Brent aus SciPy.** `minimize_scalar(method="bounded")` ist das Verfahren von Brent (1973,
Kap. 5) in der Fassung von `fminbound`: Goldener Schnitt mit Parabeln, ohne Ableitung, in
gegebenen Grenzen. Eine eigene Fassung brächte nichts, und die von SciPy kann jeder nachrechnen.
`xatol` = 100 µm wie bei Nelder–Mead, in Millimetern wie dort (Laborbuch 08, 3.8). Die beiden
Toleranzen meinen nicht genau dasselbe: Nelder–Mead hört auf, wenn seine zwei Punkte höchstens
100 µm auseinanderliegen; Brent, wenn das Minimum auf etwa ±67 µm um den besten Punkt
eingeschlossen ist.

**E4 – Das Intervall als Grenzen, sonst nichts.** Brent in Grenzen hat keinen Startpunkt; das
gelernte Modell geht nur über das Intervall ein, wie bei der Suche im Intervall von Etappe 8.
Ein Warmstart von der Schätzung, wie in Etappe 8 für Nelder–Mead, ist für Brent nicht
vorgesehen; er bräuchte eine Klammer um das Minimum, die das Verfahren dann selbst nicht prüft.

**E5 – Nelder–Mead im selben Intervall.** Ob ein Intervall mit Brent mehr spart als mit
Nelder–Mead, zeigt nur Nelder–Mead im selben Intervall. Auf den Messhologrammen gibt es das aus
Etappe 8 (`box95-{cnn,npe}-loo-ensemble-b1-p4`), auf den synthetischen Samples rechne ich es
neu (`box95-npe-ensemble`, `box95-cnn`), mit den Enden des Intervalls als Anfangssimplex.

**E6 – Die Modelle auf den synthetischen Samples.** „Die Modelle, die auf diesen Daten trainiert
sind“, sind auf cpu-4x das NPE-Ensemble von Etappe 7 (fünf Seeds) und das Gauß-CNN von Etappe 4.
Ein CNN-Ensemble gibt es auf cpu-4x nicht; es neu zu trainieren lag außerhalb dieser Etappe
(6, Punkt ‹TODO›). Das einzelne CNN steht deshalb für das CNN, mit seinem nachkalibrierten
$\hat\sigma$.

**E7 – Was „sparen“ heißt.** Pro Sample und Verfahren: Rekonstruktionen kalt minus
Rekonstruktionen im Intervall; das Mittel über die Samples mit Bootstrap-Intervall (10 000
Ziehungen) und relativ als 1 − (Summe im Intervall / Summe kalt). Die Hauptfrage beantwortet die
gepaarte Differenz: Ersparnis mit Brent minus Ersparnis mit Nelder–Mead, Sample für Sample.

**E8 – Nelder–Mead noch einmal.** Die kalten Suchen mit Nelder–Mead (`b1-p4`, Baseline von
Etappe 3) liefen mit Numerik 3, alle neuen mit Numerik 1. Ein Unterschied zwischen Brent und
Nelder–Mead könnte deshalb auch an der Numerik liegen: Nelder–Mead vergleicht nur Werte, und
zwei Punkte, deren MFE sich um weniger als die 2–9 ppm der Numerik unterscheidet, können die Plätze
tauschen. Die Wiederholung zeigt, wie viel die Numerik allein an einer Suche ändert: auf allen
vier Messhologrammen und auf 8 synthetischen Samples, darunter die beiden, bei denen in der Suche
von Etappe 3 zwei MFE-Werte weniger als die Numerik auseinanderliegen (11 und 46).

**E9 – Zeit: Rekonstruktionen zählen, Sekunden mit 3 Threads.** Wie in Etappe 8 (E13): Fast die
ganze Zeit einer Suche sind ihre Rekonstruktionen, und ihre Zahl hängt nicht an der Maschine.
Die Zeit gebe ich als Zahl der Rekonstruktionen mal der Rate mit 3 Threads an: auf den
Messhologrammen der Median der Sekunden pro Rekonstruktion der Suchen dieser Etappe mit 3 Threads
derselben Probe, auf den synthetischen Samples der Median der Suchen der Lane mit 3 Threads.

**E10 – Wann ein Sprung eine Stufe ist.** Die Numerik verschiebt die MFE um höchstens 9 ppm,
wo keine Stufe ist; die MFE steigt nahe am Minimum um 1–10 ppm pro µm. Eine Stufe wie die des
Zahns (etwa 700 ppm auf wenigen µm) fällt mit 50 ppm zwischen Nachbarn im Abstand von höchstens
50 µm, dreimal so steil wie die Sekanten daneben, sicher auf; die Krümmung einer glatten Kurve
nicht. Weiter auseinanderliegende Punkte prüft die Regel nicht, deshalb steht die Dichte der
Punkte daneben.

**E11 – Wo gescannt wird.** Beim Zahn dicht um die Stufe (80,56–80,75 mm), an den Punkten von
Etappe 8 bitgenau mit beiden Numeriken und dort, wo die Suchen mit Numerik 3 eine zweite
Unregelmäßigkeit zeigen (80,87–80,91 mm). Beim Spinnenhaar an der Schulter der oberen Flanke und
am lokalen Minimum bei 80,1 mm, das die kalte Suche mit Brent fand. Sonst nur dort, wo die Punkte
aller Suchen innerhalb ±250 µm um das Minimum Lücken über 50 µm lassen. Ein Punkt kostet mit
einem Thread etwa 9 min.

**E12 – Keine GPU.** ‹TODO: Kernstunden›; nichts davon braucht Maxwell, `docs/maxwell-gpu.md`
bleibt unverändert.

## 5 Ergebnisse

‹TODO›

## 6 Offene Punkte

‹TODO›

## 7 Nächste Schritte

‹TODO›

## 8 Bezug zur Masterarbeit

‹TODO›

Reproduktion: `docs/usage.md`, Abschnitt „Stage 9: steps of the MFE, Brent next to Nelder–Mead“.
