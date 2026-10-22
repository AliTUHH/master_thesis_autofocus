# Laborbuch 06 – Etappe 6: Objektmodell mit Textur

| | |
|---|---|
| Datum | 22.10.2026 |
| Branch | `cursor/stage6-objektmodell-1091` |
| Code | `src/holofocus/phantoms.py` (`texture_field`), Textur und Phase pro Pixel in `src/holofocus/dataset/`; `src/holofocus/learned/objectmodel.py`; `configs/dataset/loo-*.yaml`; `scripts/run_stage6_study.py`, `scripts/make_stage6_figures.py`; `tests/test_objectmodel.py` und Erweiterungen der Tests von Phantomen, Datensatz-Config und Generator |
| Ergebnisse | `results/objectmodel/` (Regionen, Texturfits, Leave-one-out-Bereiche, Spektren, Vergleich, Stage-5-Modelle auf den neuen Test-Splits, CNNs mit den Seeds 1 und 2), `results/learned/loo-*` (Modelle, Vorhersagen pro Sample), `results/datasets/loo-*` (Fingerabdrücke der Val- und Test-Splits), `results/measured/cnn-loo/` (Messdaten) |
| Abbildungen | `docs/laborbuch/figures/stage6-*.png`, Zahlen in `stage6-metrics.json` |
| Quellen | [1] Dora et al., Opt. Express 32(7), 10801 (2024), mit dem Zenodo-Archiv 10.5281/zenodo.8349365; [2] Dora et al., Opt. Express 33(4), 6641 (2025) |

## 1 Ziel

Die Sim-to-Real-Analyse von Etappe 5 hat als größte Lücke die CTF-Modulation gefunden: Die
simulierten Fasern löschen sie aus, drei der vier Messhologramme haben sie (Laborbuch 05, 5.6).
Die Aufgabe dieser Etappe:

- **Ziel:** simulierte Hologramme, deren Spektren die CTF-Modulation so zeigen wie drei der
  vier Messhologramme. Zuerst an den Spektren prüfen, ohne Training.
- **Regel:** nie an dem Hologramm einstellen, auf dem getestet wird. Leave-one-out über die
  vier Proben: an drei einstellen, an der vierten testen. Ob das Einstellen an den
  Testhologrammen zulässig ist, wird Daniel gefragt (Laborbuch 05, 6.3); bis zur Antwort gilt
  Leave-one-out.
- **Dann:** die Spektrum-CNNs pro Geometrie auf dem neuen Objektmodell neu trainieren und auf
  den Messhologrammen gegen [2] Table 1 und unsere Suche mit Padding 4 testen. Prüfen, ob die
  68-%-Intervalle die Abweichung jetzt abdecken.
- **Größen:** CPU-taugliche Sätze hier, die großen Versionen als Maxwell-Jobs
  (`docs/maxwell-gpu.md`, 7.2).

NPE ist Etappe 7; sie entstand parallel auf einem eigenen Zweig, und diese Etappe fasst sie
nicht an. Das Objektmodell bleibt allein über die Dataset-Configs wählbar, damit Etappe 7
danach unverändert darauf laufen kann.

## 2 Ausgangslage

Etappe 5 ist in `main` gemergt
([PR #12](https://github.com/AliTUHH/master_thesis_autofocus/pull/12), Merge-Commit `57dbee6`);
CI grün. Etappe 7 (NPE mit sbi) ist vor dieser Etappe gemergt worden
([PR #13](https://github.com/AliTUHH/master_thesis_autofocus/pull/13), Merge-Commit `54c72a5`);
dieser Zweig ist darauf rebased (E10). Laborbuch 05 hat festgehalten:

- **CTF-Modulation (5.6):** Die Korrelation des Spektrums mit $\sin^2(\pi f^2/Fr)$ über das
  Band des Spektrum-CNN liegt bei den Messhologrammen bei 0,25 (Spinnenhaar), −0,04 (Zahn),
  0,32 (Kaktusnadel) und 0,72 (Mg-Draht). Die simulierten Fasern mit dem Detektor und dem Objekt
  der Messung kommen im Median auf −0,04 bis +0,05; die Messung liegt über 98–100 % ihrer
  Samples. Der Zahn hat auch gemessen kaum Modulation.
- **Realistische Sätze (5.7):** Auf Sätzen mit den gemessenen Eigenschaften als Bereichen lernt
  das Netz wenig oder nichts; auf den Messhologrammen ist es nicht besser als der Schätzwert.
- **Nächste Schritte (7.2, 7.3):** Objekte, deren Spektren zu den Messhologrammen passen,
  geprüft an den spektralen Merkmalen ohne Training; als Kandidat eine schwache Phasentextur
  im Inneren der Fasern.
- **Schätzungen der Etappe 5:** Kein 68-%-Intervall der passenden Modelle enthielt [2]
  (Abstand in Standardabweichungen 1,98, 1,24, 3,08 und 1,37) oder unsere Suche mit Padding 4
  (1,56, 1,68, 2,60, 1,16).

Daniel hat auf die Frage zum Einstellen an den Testhologrammen noch nicht geantwortet.

## 3 Was gemacht wurde

### 3.1 Textur im Simulator (`b8cc065`)

`holofocus.phantoms.texture_field` erzeugt ein Gaußsches Zufallsfeld mit dem
Leistungsspektrum $(1 + (f/f_c)^2)^{-s/2}$, $f$ in Zyklen pro gebinntem Pixel, auf dem
Simulationsgitter, aber bandbegrenzt auf das gebinnte Gitter (die Überabtastung der
Simulation ändert es nicht), mit Mittelwert 0 und erwarteter Varianz 1. Der Generator addiert
es zur Phase des Objekts:

$$\varphi = -\varphi_0\,t - \left(\sigma_\text{obj}\sqrt{t} + \sigma_\text{film}\right) g,$$

mit der normierten Dicke $t$ (1 an der dicksten Stelle im Sichtfeld), dem Feld $g$, der RMS
$\sigma_\text{obj}$ an der dicksten Stelle des Objekts und der RMS $\sigma_\text{film}$ eines
Films über dem ganzen Sichtfeld. $\sqrt{t}$, weil sich eine zufällige Dichte des Materials
entlang des Strahls aufsummiert und ihre Varianz mit der Weglänge wächst. Die Absorption folgt
der Phase über $\delta/\beta$ wie beim übrigen Objekt. RMS-Werte (log-uniform), Steigung $s$
(uniform) und das Feld kommen aus einem dritten Zufallsstrom pro Sample: Ein Sample behält mit
und ohne Textur Aufbau, Objekt und Rauschen, und Configs ohne Textur geben bitgleich dieselben
Samples wie vorher (Test). $f_c$ = 0,01 liegt unter dem Band des Spektrum-CNN ($f \ge 0{,}045$),
dort zählen also nur RMS und Steigung.

Dazu kann die Phase der Fasern der projizierten Dicke folgen statt einem festen Phasenhub:
`phase_per_pixel_range` gibt die Phase pro gebinntem Pixel Weg vor, so dass dünne Fasern
weniger schieben als dicke. Die Werte der Proben stammen aus dem Phasenhub und dem Durchmesser
in `MEASURED_MODELS` (`objectmodel.phase_per_pixel`, Spinnenhaar 0,021, Zahn 0,031, Kaktusnadel
0,048, Mg-Draht 0,036 rad pro Pixel).

### 3.2 Woher die CTF-Modulation kommt (`731b6b9`)

Bevor das Modell Textur bekommt, sollte klar sein, wo sie in den Messhologrammen sitzt. Die
Phasenrekonstruktion von Etappe 3 (Padding 4, an der Distanz der Suche) trennt Objekt und
Hintergrund: Kern ist, wo die Phase unter 15 % ihres Minimums liegt, zur Kontrolle unter 5 %.
`objectmodel.region_weights` macht daraus drei glatte Gewichte, Objekt (Kern um 8 Pixel
vergrößert), Rand (±16 Pixel um die Kante) und Hintergrund (Rest ohne Kern plus 16 Pixel), und
`region_ctf` berechnet die CTF-Korrelation jeder Region an $Fr$ von [2] mit dem Gewicht im
Fenster der Fourier-Transformation. Als Kontrolle dieselbe Korrelation an 1,3 $Fr$: Dort
sollte keine Modulation sein (`results/objectmodel/regions.json`, Abb. `stage6-regions.png`).

### 3.3 Texturfit pro Hologramm (`5578b29`, `e18fb5e`)

Für jedes Messhologramm ein Satz von 16 Test-Samples mit seinem Detektor und seinem Objekt
(`objectmodel.fit_changes`): PSF und Grenzfrequenz des Flatfield-Rests aus den Fits der Ecken
(`DETECTOR_FITS`; beim Mg-Draht, dessen Ecken den Draht enthalten, die Werte von Etappe 5),
Photonen und Rest-RMS aus `MEASURED_MODELS`, Fasern mit den Durchmessern und der Phase pro
Pixel der Probe. Darauf jede Textur eines groben Gitters (RMS im Objekt 0–1 rad und Film
0–0,3 rad, je etwa Faktor 3 auseinander, Steigung 2–3,5; 97 Einstellungen), dann ein feineres
Gitter um die beste (Faktor $10^{1/4}$, Steigung ±0,25). Maß ist der RMS der skalierten
Differenzen zwischen den Median-Merkmalen des Satzes und denen des Messhologramms
(`fit_objective`): die CTF-Korrelation über das CNN-Band und in den vier Bändern (Skala 0,1),
das obere Band über dem Boden (0,2) und die Steigung (2). Boden und tiefste Frequenzen bleiben
draußen, weil sie mehr vom Flatfield-Rest und vom Objekt abhängen als von der Textur
(`results/objectmodel/fit-<probe>.{json,csv}`, Abb. `stage6-fit.png`). Das Ergebnis steht in
`objectmodel.TEXTURE_FITS`.

### 3.4 Leave-one-out-Sätze (`242c3e4`)

`objectmodel.loo_ranges` macht aus allem, was die Messhologramme über Detektor und Objekt
zeigen, Bereiche über die drei anderen Proben, um den Faktor 2 erweitert
(`LOO_FACTOR`): Photonen, Flatfield-Rest, Phase pro Pixel und Faserdurchmesser vom kleinsten
Wert der anderen durch 2 bis zum größten mal 2 (Durchmesser höchstens das Sichtfeld), die PSF
über die Fits der anderen ±0,3 Pixel, die Grenzfrequenz des Rests deren Median. Die Textur
sitzt bei einer Probe im Objekt, bei einer anderen im Film; beide Bereiche reichen deshalb von
0,005 rad (`TEXTURE_FLOOR`/2, eine Probe darf ohne Textur sein) bis zur größten Amplitude der
anderen mal 2, und die Steigung über die Fits der texturierten anderen ±0,5. Vom Hologramm der
Probe selbst bleiben nur der Aufbau (Energie, Distanzen, Detektor, Schätzwert ± 5 mm wie in
Etappe 5) und $\delta/\beta$, das aus der Zusammensetzung folgt. Dass ein Satz nichts über
das ausgelassene Hologramm weiß, prüft ein Test: Ändert man dessen Werte, bleibt die Config
gleich. `scripts/run_stage6_study.py configs` schreibt `configs/dataset/loo-<probe>.yaml`
(6000/250/250 Samples wie in Etappe 5, eigene Seeds) und `loo-<probe>-large.yaml` (50 000
Trainings-Samples für Maxwell) und hält in `results/objectmodel/loo-ranges.json` fest, ob die
Werte der Probe selbst in ihren Bereichen liegen.

### 3.5 Spektrale Prüfung ohne Training (`b1cbc20`)

`scripts/run_stage6_study.py spectra` vergleicht die Merkmale jedes Messhologramms mit vier
simulierten Sätzen: dem Test-Split der Etappe 5 (kompakte Objekte ohne Textur), den 16 Samples
des Fits ohne Textur (eigener Detektor, eigene Fasern), denselben mit der gefitteten Textur und
dem Leave-one-out-Test-Split (250 Samples). Pro Satz Median und 5–95 % jedes Merkmals, das
Perzentil des Messhologramms und das Spektrum eines typischen Samples (CTF-Korrelation am
nächsten am Median; `results/objectmodel/spectra-<probe>.json`, Abb. `stage6-spectra.png`).
Die Fit-Sätze sind an der Probe selbst eingestellt und zeigen, was die Textur kann; nur der
Leave-one-out-Satz beantwortet die Frage der Etappe.

### 3.6 Training und Schätzung (`6b22951`, `a89563c`, `a03c22b`, `55d7332`, `9fdea02`)

Pro Probe ein Spektrum-CNN mit derselben Config wie in Etappe 5
(`configs/learned/spectrum-cnn-wide.yaml`, 100 Epochen, ein Thread), auf dem Leave-one-out-Satz
trainiert, mit Vorhersagen auf dessen Val- und Test-Split und auf dem Test-Split der Etappe 5;
dazu das Modell der Etappe 5 auf dem neuen Test-Split. Nur die Daten ändern sich, Unterschiede
kommen also vom Objektmodell. `measured` schätzt wie in Etappe 5 auf dem Messhologramm, mit
16 Binning-Phasen und 20 gemessenen Schätzungen; `compare` stellt Leave-one-out-CNN, CNN der
Etappe 5, [2] und Padding 4 nebeneinander, mit den Testfehlern genau der Läufe, die geschätzt
haben (`results/objectmodel/comparison.{csv,json}`), und mit dem Schätzwert als Bezug (E11):
seinem Abstand zu [2] und Padding 4 und dem Perzentil, das ein Netz bekäme, das nur den
Schätzwert ausgibt, dazu dem Perzentil von $\hat\sigma$ unter den $\hat\sigma$ des Test-Splits.
Dazu dieselben CNNs mit den
Trainings-Seeds 1 und 2 (das Hauptmodell hat den Seed 0 der Config), mit Vorhersagen auf dem
Test-Split und Schätzungen auf den Messhologrammen (`results/objectmodel/seeds/seed-<k>/`, ohne
`model.pt`); `compare` fasst ihre Streuung zusammen.

### 3.7 Tests

- **Textur und Phase pro Pixel** (`tests/test_phantoms.py`, `test_dataset_generator.py`,
  `test_dataset_config.py`): RMS 1 und Potenzgesetz des Feldes für drei Steigungen,
  Bandbegrenzung auf das gebinnte Gitter, Dicke der Fasern in Pixeln; ein Sample behält mit
  Textur Aufbau, Objekt, Photonen und Detektor; der Film deckt das ganze Sichtfeld; die Phase
  folgt der Dicke; ohne dritten Zufallsstrom keine Textur; ungültige Bereiche werden abgelehnt.
- **Regionen** (`tests/test_objectmodel.py`): Über das ganze Bild gibt `region_ctf` die
  Korrelation der spektralen Merkmale; die Gewichte trennen Objekt, Rand und Hintergrund; in
  einem Bild, das nur in einer Hälfte CTF-moduliert ist, findet `region_ctf` die Modulation
  dort und nicht in der anderen.
- **Fit:** Detektorwerte sind die Mediane der Eckenfits, die Phase pro Pixel gibt beim
  mittleren Durchmesser den gemessenen Phasenhub, die Fit-Configs haben Detektor, Objekt und
  Textur der Probe, das grobe Gitter hat die Einstellung ohne Textur einmal und zuerst, das
  feine umschließt die beste, das Maß skaliert jedes Merkmal; `TEXTURE_FITS` ist das Beste der
  Fit-Dateien.
- **Leave-one-out:** Die Bereiche einer Probe hängen nicht von ihren eigenen Werten ab (alle
  Werte des ausgelassenen Hologramms geändert, Config gleich), sie enthalten die Werte der
  anderen drei, Objekt- und Film-Textur teilen einen Bereich, und die committeten Configs sind
  die passenden mit `loo_changes` und eigenem Seed.

## 4 Entscheidungen und Begründung

**E1 – Textur als parametrisches Zufallsfeld, nicht Objekte aus den Rekonstruktionen.**
Laborbuch 05, 7.3 nennt beides. Die Phasenrekonstruktionen von Etappe 3 stammen aus genau den
vier Testhologrammen, tragen deren Artefakte und gibt es nur viermal. Ein Feld mit wenigen
Parametern lässt sich dagegen an drei Proben einstellen und an der vierten testen, und jedes
Sample bekommt ein neues.

**E2 – RMS proportional zu $\sqrt{t}$ im Objekt, konstant im Film, ein Feld für beides.** Eine
zufällige Dichte, über den Weg durch das Objekt summiert, hat eine Varianz proportional zur
Weglänge. Den Film gibt es, weil die Modulation beim Spinnenhaar im Hintergrund sitzt und beim
Mg-Draht im Objekt (5.1). Objekt und Film nutzen dasselbe Feld; im radial gemittelten Spektrum
spielt das kaum eine Rolle, im Bild setzt der Film das Muster im Objekt fort. Zwei unabhängige
Felder wären physikalisch näher und kosten eine zweite Ziehung pro Sample.

**E3 – Eigener Zufallsstrom.** Configs ohne Textur bleiben bitgleich, und alle Texturen eines
Fits sehen dieselben 16 Aufbauten, Objekte und Rauschwerte: Unterschiede im Maß kommen von der
Textur, nicht vom Ziehen neuer Samples. 16 Samples pro Einstellung, weil das grobe Gitter
97 Einstellungen hat und das Spinnenhaar dabei gut eine Stunde braucht.

**E4 – Das Fitmaß.** Die CTF-Korrelation über das CNN-Band und in den vier Bändern ist das
Ziel der Etappe; oberes Band und Steigung verhindern, dass eine Textur die Korrelation durch
Leistung an den falschen Stellen erkauft. Boden und tiefste Frequenzen bleiben draußen: Den
Boden setzt bei Spinnenhaar und Zahn der Flatfield-Rest der Eckenfits zu hoch an
(Laborbuch 05, 5.6, Punkt 3), die tiefsten Frequenzen folgen dem Objekt. Die Skalen (0,1 für
die Korrelationen, 0,2 für das obere Band, 2 für die Steigung) liegen im Bereich der Streuung
zwischen den Samples der Fit-Sätze (Standardabweichung aus 5–95 %: Korrelation 0,03–0,17,
oberes Band 0,02–0,24, Steigung 0,3–5). Der Preis steht in 5.2, Punkt 6.

**E5 – Leave-one-out mit Bereichen und Faktor 2.** Liegen vier Werte unabhängig aus derselben
Verteilung vor, ist der vierte mit Wahrscheinlichkeit 1/2 der größte oder der kleinste; ein
Bereich nur über die anderen drei verfehlt ihn also oft. Der Faktor 2 nach beiden Seiten ist
eine runde Wahl. Die Textur sitzt je nach Probe im Objekt oder im Film, deshalb teilen beide
einen Bereich, der bei fast 0 beginnt. Aufbau und $\delta/\beta$ gehören zur Probe, nicht zu
ihrem Hologramm: Energie, Distanzen und Detektor stehen in den Metadaten, $\delta/\beta$ folgt
aus der Zusammensetzung (Laborbuch 05, 3.1). Ausgedacht habe ich das Schema, nachdem ich die
Regionen und Fits aller vier Hologramme gesehen hatte; es gilt aber für alle vier gleich, und
an keinem Satz habe ich nach dem Blick auf seine Testverteilung etwas geändert.

**E6 – Das zweite Band des Spinnenhaars bleibt außerhalb.** Den Leave-one-out-Test-Split des
Spinnenhaars habe ich als ersten erzeugt und geprüft; sein zweites Band lag außerhalb
(5.4, Punkt 2). Den Satz daraufhin zu ändern hieße, am Testhologramm einzustellen. Er bleibt,
wie die anderen drei Proben ihn vorgeben.

**E7 – Dasselbe Netz und Training wie in Etappe 5.** Config, Epochen, Seeds und Größen der
Splits sind gleich; Unterschiede zwischen den Modellen kommen also von den Daten.

**E8 – Typische Samples nach der CTF-Korrelation.** Für die Spektren und Bilder in den
Abbildungen nehme ich pro Satz das Sample, dessen Korrelation am nächsten am Median liegt,
nicht ein ausgesuchtes.

**E9 – Drei Trainings-Seeds pro Probe.** Zwischen benachbarten Epochen springt der Bias auf dem
Val-Split um mehrere hundert µm (Kaktusnadel, Epochen 4–15: −880 bis +290 µm; in Etappe 5
ebenso, −980 bis +550 µm), und auf den Leave-one-out-Sätzen liegt der kleinste Val-MAE schon bei
Epoche 6 bzw. 7. Welche Schätzung das ausgewählte Modell auf dem Messhologramm abgibt, kann
deshalb am Zufall des Trainings hängen, und $\hat\sigma$ enthält diesen Anteil nicht. Zwei weitere Seeds pro Probe zeigen, wie groß er
ist; sie liefen auf Kernen, die sonst frei gewesen wären. Ausgewählt wird weiter nur das Modell
mit dem Seed der Config, damit die Schätzung nicht nach dem Blick auf die Messung gewählt wird.
Die Modelle der Seeds versioniere ich nicht (je 1,8 MB); sie lassen sich mit `--seed`
nachtrainieren, und jede Schätzung hält den SHA-256 ihres Modells fest.

**E10 – Eingefrorener Code und Rebase.** Die langen Läufe liefen aus Worktrees mit festem
Stand, damit Commits während eines Laufs nichts an ihm ändern. Danach ist der Zweig auf `main`
mit Etappe 7 rebased worden, und die Commits, die die Ergebnisdateien nennen, heißen dort
anders. Der Rebase bringt nur die Dateien der Etappe 7 dazu: Jeder alte Commit unterscheidet
sich von seinem neuen genau um die Änderungen von `57dbee6` nach `54c72a5`, und Etappe 7 hat
weder Generator noch Training des CNN geändert. Die alten Commits liegen auf dem Branch
`cursor/stage6-prerebase-1091`.

| aufgezeichnet | in `main` | Commit | genannt von |
|---|---|---|---|
| `2779559` | `5578b29` | Texturfit (3.3) | Regionen, deren Code erst mit dem nächsten Commit kam |
| `d0f8a89` | `731b6b9` | Regionen (3.2) | Texturfits, Daten des Spinnenhaars; Leave-one-out-Bereiche, deren Code erst mit `9fd4155` kam |
| `9fd4155` | `242c3e4` | Leave-one-out-Configs (3.4) | übrige Daten, alle Trainings und Vorhersagen auf simulierten Splits |
| `e54d58c` | `b1cbc20` | spektrale Prüfung (3.5) | Spektren |

Wo der Code erst nach dem Ergebnis versioniert wurde, habe ich es mit dem versionierten Code
wiederholt: `configs` gibt dieselben Configs Byte für Byte und dieselben Leave-one-out-Bereiche
(bis auf den Commit), die Regionen stehen in 5.8. Die Configs, aus denen die Daten entstanden
sind (`data/loo-<probe>/config.yaml`), sind dieselben wie die versionierten.

**E11 – Der Schätzwert als Bezug.** Ein 68-%-Intervall deckt leicht, wenn es breit ist, und auf
den Messhologrammen liegt schon der Schätzwert näher an den Referenzen als auf einem typischen
Test-Sample (5.6, Punkt 3). Ob die Intervalle die Abweichung decken, sagt deshalb allein wenig.
Neben jeder Schätzung steht der Schätzwert mit seinem Abstand zu [2] und Padding 4 und dem
Perzentil, das er unter denselben Testfehlern bekäme. Eingeführt habe ich das, nachdem ich die
ersten Schätzungen gesehen hatte (die Kaktusnadel blieb am Schätzwert); es ändert keine
Schätzung und kein Modell, nur, wie sie gelesen werden.

## 5 Ergebnisse

### 5.1 Woher die CTF-Modulation kommt

CTF-Korrelation an $Fr$ von [2] über das CNN-Band, in Klammern die vier Bänder (Kern unter
15 % des Phasenminimums; Anteil der Fläche nach Gewicht):

| Probe | ganzes Hologramm | Objekt | Rand | Hintergrund |
|---|---|---|---|---|
| Spinnenhaar | 0,25 (0,53/0,28/−0,03/−0,03) | 0,08 (0,31/0,01/…), 12 % | 0,08, 11 % | 0,17 (0,30/0,35/−0,05/0,01), 85 % |
| Zahn | −0,04 (−0,14/−0,01/0,06/0,02) | 0,01, 34 % | 0,01, 11 % | −0,01, 63 % |
| Kaktusnadel | 0,32 (0,35/0,47/0,27/0,07) | 0,09 (0,24/0,02/…), 38 % | 0,02, 17 % | 0,05, 58 % |
| Mg-Draht | 0,72 (0,89/0,81/0,60/0,25) | 0,76 (0,92/0,87/0,62/0,25), 71 % | 0,10, 13 % | 0,01, 25 % |

1. **Beim Mg-Draht sitzt die Modulation im Draht.** Das Objekt allein hat 0,76, so viel wie das
   ganze Hologramm, der Hintergrund 0,01. Die innere Struktur des Drahts (Laborbuch 05, 7.3)
   ist also die Quelle.
2. **Beim Spinnenhaar sitzt mehr im Hintergrund als im Haar** (0,17 gegen 0,08), im zweiten
   Band nur dort (0,35). Das passt zu einem Film oder einer feinen Struktur um das Haar, nicht
   zu Textur im Haar.
3. **Bei der Kaktusnadel erklärt keine Region allein das ganze Hologramm.** Das Objekt hat
   0,09 (mit dem Kern unter 5 %: 0,14), der Hintergrund 0,05, das ganze Hologramm 0,32. Der
   Rand korreliert an 1,3 $Fr$ mit +0,12, die Kontrolle schlägt dort also an; alle anderen
   Kontrollen liegen bei 0 (±0,07).
4. **Der Zahn hat nirgends Modulation.**
5. **Vorbehalt:** Das Gewicht jeder Region multipliziert das Bild, sein Spektrum wird also mit
   dem des Gewichts gefaltet. Das verschmiert die Modulation, am stärksten bei den tiefsten
   Frequenzen; im ersten Band liegen die Regionen deshalb unter dem ganzen Hologramm, auch wenn
   sie die Modulation tragen. Die Zahlen zeigen, wo sie sitzt, nicht wie stark.

### 5.2 Texturfits

Beste Textur pro Messhologramm, ihr Maß und das ohne Textur, und die Merkmale des Fits
gegen die Messung (`results/objectmodel/fit-<probe>.json`):

| Probe | Objekt | Film | Steigung | Maß | ohne Textur | CTF Messung / Fit / ohne | Bänder Messung | Bänder Fit |
|---|---|---|---|---|---|---|---|---|
| Spinnenhaar | 0 | 0,1 rad | 3,0 | 1,13 | 2,56 | 0,25 / 0,16 / 0,04 | 0,53/0,28/−0,03/−0,03 | 0,42/0,07/0,01/−0,02 |
| Zahn | 0 | 0 | – | 0,72 | 0,72 | −0,04 / 0,01 / 0,01 | −0,14/−0,01/0,06/0,02 | 0,03/0,01/0,00/0,00 |
| Kaktusnadel | 0,3 rad | 0,03 rad | 1,75 | 1,73 | 3,87 | 0,32 / 0,30 / −0,14 | 0,35/0,47/0,27/0,07 | 0,35/0,26/0,35/0,27 |
| Mg-Draht | 0,3 rad | 0,53 rad | 1,75 | 0,89 | 5,62 | 0,72 / 0,71 / 0,06 | 0,89/0,81/0,60/0,25 | 0,83/0,77/0,63/0,41 |

1. **Die Textur bringt die CTF-Modulation zurück**, wo die Messung sie hat: Bei Kaktusnadel und
   Mg-Draht trifft der Fit die Korrelation über das CNN-Band auf 0,02, beim Spinnenhaar kommt
   er von 0,04 auf 0,16 (gemessen 0,25). Das Maß fällt um den Faktor 2,3 (Spinnenhaar), 2,2
   (Kaktusnadel) und 6,3 (Mg-Draht).
2. **Die Fits passen zu den Regionen (5.1):** ein Film um das Spinnenhaar, keine Textur beim
   Zahn, Textur vor allem in der Kaktusnadel. Beim Mg-Draht legt der Fit auch einen Film von
   0,53 rad über das Sichtfeld, obwohl der Hintergrund keine Modulation hat. Der Draht füllt
   den größten Teil des Sichtfelds, und der Fit sieht nur das ganze Hologramm: Textur im Draht
   und Film sind dort fast austauschbar (Objekt 0,53 und Film 0,3 rad gibt 0,91 statt 0,89).
3. **Das zweite Band des Spinnenhaars erreicht keine Textur.** Gemessen 0,28, mit dem besten
   Fit im Median 0,07, mit keiner der 105 Einstellungen im Median über 0,08. Ein isotropes Feld
   mit Potenzgesetz legt die Modulation vor allem ins erste Band.
4. **Die Kaktusnadel ist ein Kompromiss.** Die Korrelation stimmt, ihre Verteilung auf die
   Bänder nicht (viertes Band 0,27 statt 0,07), und das obere Band liegt 1,76 statt 1,12 über
   dem Boden.
5. **Bei Kaktusnadel und Mg-Draht liegt die beste Steigung am Rand des feinen Gitters** (1,75,
   das grobe Gitter beginnt bei 2), beim Mg-Draht auch der Film (0,53 rad). Flacher als 1,75
   ist nicht geprüft.
6. **An den höchsten Frequenzen ist die Textur zu stark.** Boden und tiefste Frequenzen sind
   nicht Teil des Maßes (3.3). Mit der gefitteten Textur liegt der Boden der Kaktusnadel bei
   −6,4 statt −7,6, der des Mg-Drahts bei −3,9 statt −6,8 (natürlicher Logarithmus der
   Leistung, beim Mg-Draht also das 18-Fache). Die flache Steigung, die die Bänder braucht, legt
   zu viel Leistung über das CNN-Band hinaus ($f^2 > 0{,}125$, das CNN sieht diesen Teil
   nicht).

### 5.3 Leave-one-out-Bereiche

Bereiche der Leave-one-out-Sätze (`results/objectmodel/loo-ranges.json`; PSF in unbinnten
Pixeln, Phase pro gebinntem Pixel, Durchmesser relativ zum Sichtfeld):

| Probe | PSF $\sigma$ | Photonen | Flatfield-Rest | $f_c$ Rest | Phase pro Pixel | Durchmesser | Textur (Objekt und Film) | Steigung |
|---|---|---|---|---|---|---|---|---|
| Spinnenhaar | 1,20–1,94 | 1,55·10⁴–7,8·10⁴ | 1,3–7,8 % | 0,151 | 0,016–0,096 | 0,04–1 | 0,005–1,07 rad | 1,25–2,25 |
| Zahn | 1,20–2,01 | 1,55·10⁴–1,36·10⁵ | 1,25–5,2 % | 0,185 | 0,011–0,096 | 0,04–1 | 0,005–1,07 rad | 1,25–3,5 |
| Kaktusnadel | 1,34–2,01 | 1,55·10⁴–1,36·10⁵ | 1,25–7,8 % | 0,145 | 0,011–0,072 | 0,05–1 | 0,005–1,07 rad | 1,25–3,5 |
| Mg-Draht | 1,20–2,01 | 1,75·10⁴–1,36·10⁵ | 1,25–7,8 % | 0,179 | 0,011–0,096 | 0,04–1 | 0,005–0,6 rad | 1,25–3,5 |

Bis auf eine Ausnahme liegen die Werte jeder Probe in ihren eigenen Bereichen, obwohl diese
ohne sie entstanden sind: Die Steigung 3,0 des Spinnenhaars liegt außerhalb von 1,25–2,25,
denn von den anderen haben nur Kaktusnadel und Mg-Draht Textur, beide mit 1,75. Beim Mg-Draht
gibt es keinen eigenen Detektorwert zum Vergleichen.

### 5.4 Spektrale Prüfung ohne Training

Perzentil der CTF-Korrelation des Messhologramms im jeweiligen Satz (Anteil der Samples mit
kleinerem Wert), über das CNN-Band und in Klammern in den vier Bändern
(`results/objectmodel/spectra-<probe>.json`, Abb. `stage6-spectra.png`):

| Probe | Etappe 5 | Fit ohne Textur (16) | Fit mit Textur (16) | Leave-one-out (250) |
|---|---|---|---|---|
| Spinnenhaar | 79 (64/92/22/16) | 94 (94/100/19/31) | 88 (75/100/38/44) | **92** (81/100/32/22) |
| Zahn | 3 (2/15/71/66) | 0 (0/19/100/69) | 0 (wie ohne) | **4** (3/13/74/65) |
| Kaktusnadel | 12 (1/24/28/27) | 100 (100/100/100/100) | 69 (50/81/44/6) | **34** (30/42/40/30) |
| Mg-Draht | 70 (62/69/65/58) | 100 (100/100/100/100) | 62 (81/75/44/31) | **68** (72/66/61/59) |

Die übrigen Merkmale im Leave-one-out-Satz (Perzentile von Boden, oberem Band, tiefsten
Frequenzen und Steigung): Spinnenhaar 1/21/80/61, Zahn 20/22/90/77, Kaktusnadel 8/38/99/67,
Mg-Draht 36/68/57/40.

1. **Bei Kaktusnadel und Mg-Draht ist das Ziel erreicht.** Die Messung liegt im
   Leave-one-out-Satz über das CNN-Band beim 34. bzw. 68. Perzentil und in allen vier Bändern
   zwischen dem 30. und dem 72. Ohne Textur lag sie über allen Samples, auch in jedem Band.
   Die Bereiche stammen von den anderen Proben, keiner davon ist an diesen Hologrammen
   eingestellt.
2. **Beim Spinnenhaar teilweise.** Über das CNN-Band beim 92. Perzentil, im ersten Band beim
   81., im zweiten über allen 250 Samples (gemessen 0,28, 95. Perzentil des Satzes 0,15). Dass
   das zweite Band außerhalb liegt, zeigt schon der Fit an der Probe selbst (5.2, Punkt 3); am
   Leave-one-out-Satz habe ich deshalb nichts geändert. Der Boden liegt beim 1. Perzentil:
   Der Flatfield-Rest der anderen reicht bis 7,8 %, das Spinnenhaar hat 2,5 %, und die Textur
   hebt den Boden zusätzlich (5.2, Punkt 6).
3. **Der Zahn liegt knapp darunter** (4. Perzentil, im ersten Band 3.). Seine Korrelation ist
   gemessen leicht negativ (−0,04, im ersten Band −0,14); das erzeugt keiner der Sätze, auch der
   ohne Textur nicht. Der Leave-one-out-Satz enthält Samples mit wenig Textur, aber kaum eines
   mit so wenig Modulation.
4. **Die passenden Sätze der Etappe 5** lagen bei drei Proben ebenfalls in der Verteilung (79.,
   12. und 70. Perzentil); ihre kompakten Objekte schwächen die CTF nicht so stark wie
   Fasern mit dem Phasenhub der Messung. Erst die Fasern löschen sie aus (Fit ohne Textur:
   94.–100. Perzentil), und die Textur bringt sie zurück.
5. **Bei der Kaktusnadel hat die Messung mehr Leistung bei den tiefsten Frequenzen** als 99 %
   des Satzes. Das ist die Seite des Objekts, nicht der Textur; ob es an der Form der Nadel
   liegt (sie verjüngt sich, Laborbuch 05, 3.1), habe ich nicht geprüft.

### 5.5 Training auf den Leave-one-out-Sätzen

Pro Probe das Spektrum-CNN des Leave-one-out-Satzes (LOO) neben dem Modell der Etappe 5, jedes
auf beiden Test-Splits (je 250 Samples; `results/learned/loo-<probe>/`,
`results/objectmodel/matched-model/<probe>/`, Abb. `stage6-training.png` und
`stage6-sim2real.png`). MAE in µm, in Klammern der Bias:

| Probe | beste Epoche | Val-MAE | LOO-CNN auf LOO-Test | Deckung 68/95 % | LOO-CNN auf Test der Etappe 5 | CNN der Etappe 5 auf LOO-Test | CNN der Etappe 5 auf eigenem Test | Schätzwert auf LOO-Test |
|---|---|---|---|---|---|---|---|---|
| Spinnenhaar | 19 | 993 | 982 (−242) | 72/91 | 695 (−343) | 1392 (+668) | 241 (+56) | 2349 |
| Zahn | 8 | 777 | 880 (+20) | 71/94 | 713 (−159) | 1209 (+449) | 209 (−47) | 2599 |
| Kaktusnadel | 6 | 1195 | 1296 (−439) | 67/94 | 1974 (−793) | 2279 (+1862) | 735 (+226) | 2584 |
| Mg-Draht | 7 | 1449 | 1567 (−328) | 62/95 | 2515 (−1706) | 2864 (+2296) | 1273 (+148) | 2535 |

1. **Mit Textur ist die Aufgabe schwerer, aber lernbar.** Auf dem eigenen Test-Split liegt das
   Netz weit unter dem Schätzwert (Spinnenhaar 982 statt 2349 µm, Zahn 880 statt 2599,
   Kaktusnadel 1296 statt 2584, Mg-Draht 1567 statt 2535), aber über dem Modell der Etappe 5
   auf dessen Daten, bei Spinnenhaar und Zahn um das Vierfache, bei Kaktusnadel und Mg-Draht um
   das 1,8- bzw. 1,2-Fache.
   Auf den realistischen Sätzen der Etappe 5, deren Fasern die CTF auslöschten, kam das Netz
   nur auf 1929 bzw. 2605 µm, obwohl deren Bereiche enger um die Messung lagen (Laborbuch 05,
   5.7). Die Intervalle sind auf dem eigenen Test-Split kalibriert (die 68-%-Intervalle decken
   62–72 %, die 95-%-Intervalle 91–95 %).
2. **Das Modell der Etappe 5 schätzt auf texturierten Fasern zu hoch**, im Mittel um 0,67 mm
   (Spinnenhaar), 0,45 mm (Zahn), 1,86 mm (Kaktusnadel) und 2,30 mm (Mg-Draht), und seine
   68-%-Intervalle decken dort nur 20–39 %. Auf den Messhologrammen lag es 0,42, 0,74, 4,75 und
   2,81 mm über [2] (Laborbuch 05, 5.2): Die Textur verschiebt in dieselbe Richtung.
3. **Umgekehrt schätzt das neue Modell auf den kompakten Objekten der Etappe 5 zu tief**
   (−0,34, −0,16, −0,79 und −1,71 mm). Bei Spinnenhaar und Zahn ist es dort sogar genauer als
   auf dem eigenen Test-Split (695 und 713 µm), beim Mg-Draht nicht besser als der Schätzwert
   (2515 gegen 2591 µm). Was das Spektrum über $z_{01}$ verrät, hängt für beide Modelle am
   Objekt.
4. **Die beste Epoche kommt meist früher** als in Etappe 5 (Spinnenhaar 19 statt 44, Zahn 8
   statt 53, Kaktusnadel 6 statt 24, Mg-Draht 7 wie dort). Danach fällt der Trainingsverlust
   weiter, der Val-MAE kaum noch (Abb. `stage6-training.png`); wo auf diesem Plateau sein
   Minimum liegt, ist Zufall (Punkt 5).
5. **Andere Trainings-Seeds** (E9, Seeds 0/1/2, `results/objectmodel/seeds/`): Der MAE auf dem
   eigenen Test-Split ist 982/994/1013 µm (Spinnenhaar), 880/814/908 (Zahn), 1296/1211/1327
   (Kaktusnadel) und 1567/1614/1593 (Mg-Draht), streut also um 30–116 µm (3–12 %). Die beste
   Epoche springt dagegen (19/15/11, 8/55/7, 6/9/66 und 7/12/10), bei einem Val-MAE, der sich
   zwischen den Seeds um höchstens 9 % unterscheidet. Die 68-%-Intervalle decken 62–78 %, die
   95-%-Intervalle 91–96 %.

### 5.6 Schätzungen auf den Messhologrammen

Leave-one-out-CNN und CNN der Etappe 5 gegen [2] Table 1 und unsere Suche mit Padding 4, mit
dem Schätzwert als Bezug (`results/measured/cnn-loo/<probe>.json`,
`results/objectmodel/comparison.{csv,json}`, Abb. `stage6-measured.png`). Abstände in µm,
in Klammern $\hat\sigma$; „ja/nein“ heißt: [2] bzw. Padding 4 liegt im 68-%-Intervall:

| Probe | [2] | Schätzwert − [2] | Padding 4 − [2] | LOO-CNN − [2] | LOO-CNN − Padding 4 | 68 % | Etappe 5 − [2] | Etappe 5 − Padding 4 | 68 % |
|---|---|---|---|---|---|---|---|---|---|
| Spinnenhaar | 79,40 mm | +400 | +88 | −231 (550) | −319 | ja/ja | +418 (211) | +330 | nein/nein |
| Zahn | 80,90 mm | +2000 | −266 | −167 (727) | +99 | ja/ja | +743 (601) | +1009 | nein/nein |
| Kaktusnadel | 284,20 mm | +1900 | +728 | +2049 (1873) | +1321 | nein/ja | +4753 (1545) | +4024 | nein/nein |
| Mg-Draht | 470,80 mm | +1600 | +428 | −771 (1698) | −1200 | ja/ja | +2813 (2052) | +2385 | nein/nein |

1. **Die 68-%-Intervalle decken die Abweichung jetzt bei Spinnenhaar, Zahn und Mg-Draht**, für
   [2] und für Padding 4; bei der Kaktusnadel nur Padding 4 ([2] liegt 1,09 $\hat\sigma$
   entfernt, im 95-%-Intervall). In Etappe 5 lag keine der beiden Referenzen im 68-%-Intervall
   eines passenden Modells.
2. **Gedeckt, weil die Intervalle breit sind, außer beim Zahn.** $\hat\sigma$ ist 550, 727,
   1873 und 1698 µm (Etappe 5: 211, 601, 1545 und 2052). Bei Spinnenhaar, Kaktusnadel und
   Mg-Draht liegen die Schätzungen weiter von [2] als die Suche mit Padding 4 (231, 2049 und
   771 µm gegen 88, 728 und 428); beim Zahn liegt das Netz näher an [2] als die Suche (167 gegen
   266 µm) und 99 µm neben ihr. Näher an [2] als die Modelle der Etappe 5 (418, 743, 4753 und
   2813 µm) liegen die Schätzungen bei allen vier Proben.
3. **Der Schätzwert als Bezug.** Er liegt 0,40, 2,00, 1,90 und 1,60 mm über [2], bei allen vier
   Proben näher als auf einem typischen Test-Sample (dort gleichverteilt bis 5 mm, MAE
   2,3–2,6 mm). Ein Intervall von ±2 mm um den Schätzwert hätte bei Spinnenhaar, Kaktusnadel und
   Mg-Draht [2] und Padding 4 gedeckt, beim Zahn [2] genau auf dem Rand (2,00 mm) und Padding 4
   (2,27 mm) nicht.
   Was das Netz hinzufügt, ist verschieden: Beim Zahn rückt es am weitesten, 2,17 mm nach unten
   zwischen die beiden Referenzen ($\hat\sigma$ beim 39. Perzentil). Beim Spinnenhaar rückt es
   0,63 mm nach unten, an [2] vorbei auf 0,23 mm darunter, mit einem $\hat\sigma$ kleiner als
   bei 88 % seiner Test-Samples; beim Mg-Draht 2,37 mm nach unten, an Padding 4 und [2] vorbei
   auf 0,77 mm unter [2] ($\hat\sigma$ beim 49. Perzentil). Bei der Kaktusnadel bleibt es beim
   Schätzwert (+0,15 mm), mit einem $\hat\sigma$ beim 70. Perzentil: Das Spektrum sagt ihm dort
   wenig, und das Intervall deckt Padding 4, weil es breit ist.
4. **Die Binning-Phasen** verschieben die Schätzung um 10, 17, 172 und 61 µm
   (Standardabweichung der 16 Phasen), wenig gegen $\hat\sigma$. Eine Schätzung dauert
   2,4–4,1 ms mit einem Thread, einschließlich Spektrum, beim Zahn wie in Etappe 5 am längsten
   (FFT auf 472² Pixeln, Laborbuch 05, 5.2, Punkt 7).
5. **Andere Trainings-Seeds** verschieben die Schätzung um 155, 67, 230 und 187 µm
   (Standardabweichung der drei Seeds; Spannweite 269, 120, 461 und 357 µm), 9–28 % von
   $\hat\sigma$; in Quadratur dazugenommen, würden die Intervalle um 0,4–4 % breiter. Beim
   Spinnenhaar ist das mehr als der Abstand von [2] und Padding 4 (88 µm). Punkt 1 gilt für
   jeden Seed: Bei Spinnenhaar, Zahn und Mg-Draht liegen [2] und Padding 4 in allen drei
   68-%-Intervallen, bei der Kaktusnadel nur Padding 4.

### 5.7 Testfehler gegen Abweichung

Perzentil der Abweichung auf dem Messhologramm unter den Fehlern desselben Modells auf seinem
Test-Split (Anteil der Test-Samples mit kleinerem Fehler; Etappe 5 auf dem passenden Test-Split,
Laborbuch 05, 5.4; Etappe 6 auf dem Leave-one-out-Test-Split), dazu derselbe Wert für den
Abstand des Schätzwerts, also für ein Netz, das nur den Schätzwert ausgäbe, und das Perzentil
von $\hat\sigma$ unter den $\hat\sigma$ des Test-Splits (`results/objectmodel/comparison.json`,
Abb. `stage6-sim2real.png`). In Prozent:

| Probe | Etappe 5: [2] | Etappe 5: Padding 4 | LOO-CNN: [2] | LOO-CNN: Padding 4 | Schätzwert: [2] | Schätzwert: Padding 4 | $\hat\sigma$ |
|---|---|---|---|---|---|---|---|
| Spinnenhaar | 88 | 80 | 24 | 34 | 40 | 32 | 12 |
| Zahn | 95 | 97 | 19 | 10 | 88 | 91 | 39 |
| Kaktusnadel | 99,6 | 99 | 78 | 66 | 75 | 63 | 70 |
| Mg-Draht | 88 | 82 | 38 | 50 | 63 | 50 | 49 |

1. **Für das neue Modell sind die Messhologramme so schwer wie ein typisches Test-Sample.** In
   Etappe 5 lagen die Abweichungen über 80 % der Testfehler, bei Zahn und Kaktusnadel über
   95 %; jetzt zwischen 10 und 78 %, am niedrigsten beim Zahn (19 und 10 %). Dazu tragen die
   kleineren Abweichungen bei (5.6, Punkt 2) und die größeren Testfehler (5.5, Punkt 1).
2. **Klar besser als der Schätzwert nur beim Zahn.** Dort liegt das Netz gegen beide Referenzen
   weit unter ihm (19 gegen 88 %, 10 gegen 91 %). Gegen [2] auch beim Spinnenhaar und beim
   Mg-Draht (24 gegen 40 %, 38 gegen 63 %); gegen Padding 4 liegen Spinnenhaar, Kaktusnadel und
   Mg-Draht aber gleichauf mit dem Schätzwert (34 gegen 32 %, 66 gegen 63 %, 50 gegen 50 %):
   Beim Spinnenhaar und beim Mg-Draht rückt das Netz so weit unter Padding 4, wie der
   Schätzwert darüber lag. Ob es dort mehr weiß als den Schätzwert, hängt also an der Referenz;
   [2] und Padding 4 liegen 88–728 µm auseinander, und keine von beiden ist die Wahrheit.
3. **Vorbehalt:** Die Perzentile nehmen die Referenz als Wahrheit, und es sind vier
   Hologramme. Sie zeigen, ob ein Messhologramm aus dem Rahmen der Test-Samples fällt, nicht
   wie genau das Netz auf Messdaten ist.

### 5.8 Laufzeiten

Auf der VM (4 Kerne), aus den Ergebnisdateien und den HDF5-Attributen, also mit der Uhr der
Programme selbst. Die VM stand während langer Wartezeiten zeitweise still; die Zeiten, die die
Lane-Skripte in `runs/stage6/` (nicht versioniert) mit der Wanduhr messen, enthalten diese
Pausen und sind deshalb zum Teil länger.

| Schritt | Threads | Zeit |
|---|---|---|
| Regionen (3.2) | 1 | 2 s; mit dem versionierten Code (`f3db7f7`) wiederholt, dieselben Werte (E10) |
| Texturfits (3.3) | 2 | 65 min (Spinnenhaar, 105 Einstellungen), 31 min (Zahn, 97), 8 min (Kaktusnadel, 123), 10 min (Mg-Draht, 123); zusammen 1,9 h |
| Leave-one-out-Sätze, je 6500 Samples (3.4) | 1; Spinnenhaar 2 (Val, Test) und 3 (Train) | 3,75 h (Spinnenhaar; 10,9 Kernstunden), 3,1 h (Zahn), 40 min (Kaktusnadel), 37 min (Mg-Draht) |
| Spektrale Prüfung (3.5) | 1 | 50 s |
| Training, 4 Modelle (3.6) | 1 | 24 min (Spinnenhaar), 22 min (Zahn), 24 min (Kaktusnadel), 23 min (Mg-Draht) |
| Weitere Seeds, 8 Modelle (E9) | 1 | 22–24 min je Modell, zusammen 3,0 h |
| Schätzung auf den Messhologrammen (3.6) | 1 | 3 s für die vier Proben (Laden, 16 Binning-Phasen, 20 gemessene Schätzungen), ebenso je Seed; `compare` 1 s |
| Testsuite (352 Tests) | 1, die Cluster-Tests 4 | 58 s |
| Abbildungen | 1 | 5 s; der CI-Schritt (`compare` und die Abbildungen im Schnellmodus) 4 s |

Zusammen sind das etwa 24 Kernstunden (Threads mal Wandzeit): knapp zwei Drittel für die
Leave-one-out-Sätze (15,3, davon 10,9 für das Spinnenhaar), 16 % für die Texturfits (3,8) und
19 % für die zwölf Trainings (4,6). Erste Versuche mit der Textur vor dem Fit habe ich nicht
gemessen; sie sind nicht mitgezählt. Eine Schätzung kostet wie in Etappe 5 Millisekunden.

## 6 Offene Punkte

1. **Einstellen an den Testhologrammen → Daniel** (Laborbuch 05, 6.3). Bis zur Antwort gilt
   Leave-one-out. Streng getrennt ist das nicht: Die Form der Textur (E1, E2) und das Schema
   der Bereiche (E5) habe ich entworfen, nachdem ich Regionen und Fits aller vier Hologramme
   gesehen hatte; nur die Zahlen eines Satzes stammen allein von den anderen drei. Eine strenge
   Trennung bräuchte Hologramme, die nur zum Einstellen dienen (Laborbuch 05, 6.1 und 6.4).
2. **Das zweite Band des Spinnenhaars.** Kein isotropes Feld mit Potenzgesetz erreicht es
   (5.2, Punkt 3; 5.4, Punkt 2). Welche Struktur dort moduliert, etwa ein Film mit einer
   typischen Länge oder die Oberfläche des Haars, ist offen. Ein Feld mit einem Band statt
   eines Potenzgesetzes ließe sich unter Leave-one-out nur einführen, wenn eine andere Probe es
   auch zeigt.
3. **Zu viel Leistung über dem CNN-Band.** Die gefittete Textur hebt den Boden bei Kaktusnadel
   und Mg-Draht deutlich über die Messung (5.2, Punkt 6). Das Spektrum-CNN sieht diesen Teil
   nicht; für die Variante `-hf` (Spektrum bis $f^2 = 0{,}25$) und für das Bild-CNN wäre er
   falsch.
4. **Der Zahn und die tiefsten Frequenzen der Kaktusnadel.** Die leicht negative Korrelation
   des Zahns (−0,04, im ersten Band −0,14) erzeugt keiner der Sätze (5.4, Punkt 3), und die
   Kaktusnadel hat bei den tiefsten Frequenzen mehr Leistung als 99 % ihres Satzes (5.4,
   Punkt 5). Beides habe ich nicht untersucht; der Schätzung des Zahns hat es nicht erkennbar
   geschadet (5.6, Punkt 3).
5. **Ein Feld für Objekt und Film** (E2). Zwei unabhängige Felder wären physikalisch näher;
   im radial gemittelten Spektrum erwarte ich kaum einen Unterschied, geprüft ist es nicht.
6. **GPU:** die großen Leave-one-out-Sätze mit Training und Schätzung rechnet der Nutzer auf
   Maxwell (`docs/maxwell-gpu.md`, 7.2); die Zeit pro Sample auf der A100 ist dort als
   PRÜFEN markiert, weil die Textur einen Teil auf der CPU rechnet.

## 7 Nächste Schritte

1. **Daniels Antwort** (6, Punkt 1): Ist das Einstellen an den Testhologrammen zulässig,
   lassen sich die Sätze mit der Textur und dem Detektor der Probe selbst erzeugen
   (`TEXTURE_FITS`, `fit_changes`) und mit diesen vergleichen; sonst braucht es Hologramme, die
   nur zum Einstellen dienen.
2. **NPE (Etappe 7) unverändert auf den Leave-one-out-Sätzen** (Laborbuch 07, 7, Punkt 1;
   `docs/usage.md`, „Object model (Stage 6)“): ob die Posterior auf den Messhologrammen die
   Referenzen deckt und ob sie sich vom Schätzwert löst, wo das CNN es nicht tut. Gerechnet in
   Laborbuch 07, Abschnitt 9.
3. **Maxwell** (`docs/maxwell-gpu.md`, 7.2): die Sätze mit 50 000 Trainings-Samples, mit
   Training und Schätzung. Daran zeigt sich, ob mehr Daten das Netz bei der Kaktusnadel vom
   Schätzwert lösen und die Intervalle schmaler machen.
4. **Das Objektmodell nachbessern, ohne am Testhologramm einzustellen:** eine steilere Textur
   über dem CNN-Band (6, Punkt 3), zwei unabhängige Felder für Objekt und Film (6, Punkt 5) und
   für das zweite Band des Spinnenhaars ein Feld mit einer typischen Länge, wenn eine andere
   Probe es auch zeigt (6, Punkt 2).
5. **Mehr Messhologramme** (Laborbuch 05, 6.1 und 6.4): Mit vier Proben beruht jede Aussage über
   die Deckung auf vier Werten, und Leave-one-out ist die einzige Trennung zwischen Einstellen
   und Testen.

## 8 Bezug zur Masterarbeit

- **Methoden:** Textur und Phase pro Pixel im Objektmodell (3.1); die Lage der CTF-Modulation
  mit Regionen aus der Phasenrekonstruktion (3.2); der Fit der Textur an spektralen Merkmalen
  (3.3); Leave-one-out-Sätze als Schutz gegen das Einstellen am Testhologramm (3.4, E5); die
  spektrale Prüfung ohne Training (3.5).
- **Ergebnisse:** wo die Modulation in den Messhologrammen sitzt (5.1); was die Textur erreicht
  und wo nicht (5.2, 5.4); die neu trainierten Modelle auf simulierten und gemessenen Daten
  (5.5–5.7); Laufzeiten (5.8).
- **Diskussion:** wie nah ein Simulator der Messung kommt, ohne das Testhologramm zu kennen (E5,
  6.1); die Grenzen eines isotropen Feldes (5.2, Punkt 3; 6.2); dass gedeckte Intervalle allein
  kein Erfolg sind, wenn schon der Schätzwert nah an den Referenzen liegt, und der Zahn als der
  Fall, in dem das Netz sich vom Schätzwert löst (5.6, Punkt 3; 5.7, Punkt 2).

Reproduktion: alle Befehle in `docs/usage.md`, Abschnitt „Object model (Stage 6)“.
