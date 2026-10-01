# Verifikation der HoloForge-Fresnel-Labels: Woher kommt der −0,4-%-Offset des Ring-Fits?

Stand: HoloWizard/HoloForge 3.0.6, Branch `cursor/forge-label-verification-6175`.
Skript: `tools/verify_forge_fresnel.py` (Rohdaten: `reports/forge_label_verification.json`),
Regressionstests: `tests/test_forge_propagation_consistency.py`.

**Ergebnis in einem Satz:** Die HoloForge-Labels sind korrekt (Propagation und gespeichertes Fr stimmen auf
~1e-7 überein); der Offset von −0,38 % ist ein Schätzfehler des CTF-Ring-Fits durch den Term zweiter Ordnung
`−φ²/2` der exakten Objekttransmission `exp(iφ)`, der bei den „schwachen“ Mg-Phantomen (0,17–0,66 rad) wie
eine frequenzabhängige Absorption wirkt und die Ringminima verschiebt – er wächst linear mit der Objektstärke
und mit Fr und verschwindet für das linearisierte (CTF-)Modell derselben Objekte (+0,002 %).

## 1. Fragestellung

Auf `data_small_weak` (Mg 1–2 µm, rauschfreies `images/gt_hologram`) misst der Ring-Fit
(`src/baseline/ctf_ringfit.py`, Nullstellen bei |ξ|² = n·Fr) systematisch −0,38 % weniger Fr als das Label
(Terzile −0,13 / −0,38 / −0,65 %, Steigung −52 %/Fr). Ein unabhängiger Simulator in `tests/test_ctf_ringfit.py`
zeigte keinen vergleichbaren Offset. Für das Thesis-Ziel (0,1 mm in z01 ≙ 0,04 % in Fr) wäre ein Labelfehler
dieser Größe fatal (10× Ziel). Zu klären war: simuliert HoloForge mit einem anderen Fr als es labelt, verändert
eine Verarbeitungsstufe die effektive Pixelgröße, oder liegt es am Schätzer?

## 2. Hypothesen

| # | Hypothese | Test | Ergebnis |
|---|---|---|---|
| H1 | Kernel-Fr ≠ Label-Fr (Rundung, Einheiten, hc, px vs. px·s, float32) | Arithmetik + Wellenfeld-Vergleich | **verworfen** (Abweichungen ≤ 3e-5, Propagation ≙ Label auf 1e-7) |
| H2 | Nachverarbeitung skaliert das Bild um (1+ε), ε ≈ 1/512 … 1/2048 (Resize/Binning/Interpolation) | Code-Analyse + Stufentest | **verworfen**: HoloForge resampelt nie; `downsample_factor` vergröbert nur das Simulationsgitter; Crop ist ganzzahlig und zentriert |
| H3 | `kernel` wird mehrfach pro Sample gezogen (Label ≠ Propagation) | Zähler `num_draws` nach Generierung | **verworfen** (101 Aufrufe bei 100 Samples: genau einer pro Hologramm + Initialisierung) |
| H4 | Amplitude \|ψ\| statt Intensität \|ψ\|² verschiebt die Ringe | Ring-Fit auf beiden | **verworfen** (identisch bis 0,01 %) |
| H5 | Schätzer-Bias durch Nichtlinearität des Objekts (Term 2. Ordnung von exp(iφ)) | exaktes vs. linearisiertes Modell derselben Phantome, Skalierung φ×0,5/×2, Transmission 2. Ordnung | **bestätigt** |

Eine Skalierung um (1+ε) würde einen *konstanten* relativen Fr-Offset erzeugen (2ε), unabhängig von Fr und von
der Objektstärke. Beobachtet wird aber ein Offset ∝ Fr (−0,13 → −0,65 %) und ∝ Objektstärke (×0,5: −0,19 %,
×1: −0,39 %, ×2: −0,87 %). Vorsicht vor einer Zufallskoinzidenz: Der mittlere Offset meiner Kontroll-Phantome in
der data_small-Geometrie (−0,097 %) entspräche ε = −4,8e-4 ≈ 1/(s·N) = 1/2048 – rein numerisch passend, aber
physikalisch ausgeschlossen, weil der Offset für das linearisierte Modell derselben Objekte verschwindet.

## 3. Code-Analyse HoloForge 3.0.6 (`/tmp/holo311/lib/python3.11/site-packages/holowizard/forge/`)

* `experiment/setup/setup.py:26–31`: `detector_size = int(detector_size / s)`, `detector_px_size = px · s`,
  `probe_size = detector_size · padding_factor`. **Das ist die gesamte „Downsampling“-Behandlung** – es gibt keine
  Bild-Resample-Stufe (grep nach `interpolate|resize|avg_pool|zoom` findet nur Dataset-Transforms, die bei der
  Generierung nicht benutzt werden). `objects/shape_sampler.py:51–56` teilt lediglich die Formgrößen durch s.
* `experiment/setup/nfh_setup.py:65–81` (`create_kernel`): `fftfreq(probe_size)` in Zyklen/Pixel (float32),
  Kernel `exp(−iπ/Fr·(ξ²+η²))`; `:84–86` (`kernel`): `Fr = calc_Fr(energy, z01, z02, self.detector_px_size)` – also
  mit der **gebinnten** Pixelgröße px·s, derselben, die `as_dict()`/der Labeller speichern.
* `utils/utilities.py:32–47` (`calc_Fr`): λ = 1,2398/E[keV] nm, M = z02/z01,
  Fr = px² / (λ·(z02−z01)·M) mit mm→nm über `unit_conversions.py:29–38` (`round(z, 3)·1e6`; die Docstrings
  „cm/m/nm“ sind veraltet, die Rechnung ist für mm/mm/mm/keV konsistent). Identisch zu
  `src/utils/physics.fresnel_number` (Abweichung 0,0).
* `experiment/simulation/nfh_simulation.py:52–55`: `pad_to(probe_size)` (symmetrisch, `int((G−N)/2)` pro Seite),
  `obj = exp(1j·(φ + iA)) = exp(iφ)·exp(−A)`, `psi_det = ifft2(fft2(probe·obj)·kernel)`;
  `:68–69`: `holo = torch.abs(psi_det)` (**Amplitude**, der Docstring sagt „squared magnitude“), danach
  `crop_center` (`utils/utilities.py:50–63`: Start `G//2 − N//2`, ganzzahlig, zentriert).
* `generators/data_generator.py:75–93`: pro Sample genau ein `create_hologram` (ein `setup.kernel`-Zugriff
  → ein `get_distances()`), danach `labeller.cache(..., setup=setup)` mit `as_dict()` desselben Zustands.
* `utils/labeller/hdf5_labeller.py:168–170`: Setup-Werte (z01, z02, Fr) werden als **float32** gepuffert und
  gespeichert → relative Label-Unschärfe ≤ 6e-8 (gemessen 8,8e-8).
* `generators/phantom_generator.py:111, 126–129`: Dicke ganzzahlig in µm, λ = 1,2398/E, φ = −k·δ·t,
  `absorption = 2·β·k·t` – als **Amplituden**-Dämpfung `exp(−A)` verwendet, d. h. Intensität ∝ exp(−4kβt) statt
  exp(−2kβt) (Faktor 2 zu stark; für Fr irrelevant). Glättung (`:69–81`) wirkt vor der Propagation auf
  exp(φ)/exp(−A) und ändert kein Fr.
* Probe (`configs/data_small*.yaml`: constant 1, linear/square 0) ist exakt 1; Flatfield ist `None`
  (`flatfield_dataset: null`); Rauschen wird additiv auf die Amplitude gegeben, nur in `images/hologram`.

## 4. Experimente (`tools/verify_forge_fresnel.py --geometry all --n-objects 4 --forge-dataset 100`)

### 4.1 Label-Arithmetik (data_small-Geometrie, z01 = 132 mm, Fr = 7,9696e-3)

| Prüfung | rel. Abweichung |
|---|---|
| `setup.Fr` (Label) vs. `calc_Fr(E, z01, z02, px·s)` | 0 |
| `setup.Fr` vs. `src.utils.physics.fresnel_number` | 0 |
| Fr(feines px)·s² vs. Fr(gebinntes px) | 0 |
| Kernel-Pixelgröße vs. Label-Pixelgröße | 0 |
| Forge-Kernel vs. exp(−iπ/Fr·(ξ²+η²)) in float64 | max \|Δ\| = 1,4e-5 (float32-Kernel) |
| hc = 1,2398 vs. CODATA 1,23984198 | −3,4e-5 (0,003 %) |
| 1-µm-Rundung von z01 (worst case 50,0004 mm) | −8,0e-6 |
| float32-Speicherung von Fr | 5e-8 |

### 4.2 Wellenfeld-Ebene: Forge vs. Referenz-Propagator auf identischem Exit-Wave

Kontrolliertes Phantom (Kugel-/Rechteckprofile, σ = 1 px geglättet) → `NFHSimulation.forward` → `psi_exit`,
`psi_det`, `gt_hologram`. Referenz: numpy float64, `fftfreq(G)`, Kernel `exp(−iπ/Fr_label·(ξ²+η²))`.

| Geometrie (N, Gitter) | rel. L2(ψ_Forge, ψ_Ref) | max \|Δ\|ψ\|\| | rel. L2, falls Ref. Fr·(1−0,4 %) nutzt | 1-D-Fit Fr_sim/Fr_label − 1 | Crop zentriert/ganzzahlig |
|---|---|---|---|---|---|
| small 2048/8 → 256, pad 2 (512) | 1,1e-7 | 5,8e-7 | 1,5e-5 (φ 0,02) / 2,5e-4 (φ 0,33) | +1,6e-8 / −2,5e-8 | ja (Start 128) |
| nodown 512/1 → 512, pad 2 (1024) | 1,1e-7 | 5,8e-7 | 1,1e-5 / 1,8e-4 | −4,2e-7 / −2,2e-8 | ja (256) |
| ds4 2048/4 → 512, pad 2 (1024) | 1,2e-7 | 7,3e-7 | 2,8e-5 / 4,6e-4 | −1,4e-7 / −5,9e-9 | ja (256) |
| pad4 2048/8 → 256, pad 4 (1024) | 0,9e-7 | 5,7e-7 | 0,8e-5 / 1,3e-4 | −2,5e-7 / −8,9e-9 | ja (384) |

Ein Fr-Fehler von 0,4 % wäre um Faktor 100–2000 sichtbarer als die gemessene Abweichung (float32-Rauschen).
Über alle 60 Samples der Geometriestudie: max. rel. L2 1,4e-7, max. |Fr_fit/Fr_label − 1| = 1,2e-7.

### 4.3 Stufen zwischen Propagation und HDF5 (small, z01 = 132 mm; Ring-Fit Fr/Label − 1 in %)

| Stufe | φ_max 0,02 rad (linear) | φ_max 0,33 rad (Mg 2 µm) |
|---|---|---|
| Referenz float64 \|ψ\| auf dem vollen 512-Gitter | −0,061 | −0,189 |
| Forge \|ψ_det\| auf dem vollen 512-Gitter (float32-FFT) | −0,061 | −0,189 |
| `crop_center` → `images/gt_hologram` (256 px) | +0,025 | −0,016 |
| Intensitätskonvention \|ψ\|² nach Crop | +0,024 | −0,029 |
| float32-Roundtrip (HDF5) | +0,025 | −0,016 |
| + Gauß-Rauschen (10 % der Streifen-rms), clip < 0, Mittel/sd von 4 | −0,11 ± 0,11 | −0,24 ± 0,11 |

Keine Stufe verschiebt die Ringperiode systematisch; Unterschiede zwischen Voll- und Crop-Gitter sind
Schätzerstreuung (anderes N, anderer Bildinhalt), Rauschen streut nur (nullmittelig, kann Nullstellen nicht verschieben).

### 4.4 Geometrie-Abhängigkeit: exaktes vs. linearisiertes Vorwärtsmodell derselben Objekte

Pro Geometrie 4 Objekte × 3–4 Fr-Werte; „− linear“ = gepaarte Differenz zum CTF-Modell `1 − Im D(φ) − Re D(A)`
desselben Objekts (isoliert den nichtlinearen Objektterm von der objektabhängigen Schätzerstreuung).

| Geometrie | Modell | Mittel [%] | sd [%] | − linear: Mittel ± sd [%] |
|---|---|---|---|---|
| small (N 256) | Forge gt_hologram = exakt, φ 0,33 | −0,097 | 0,443 | −0,148 ± 0,447 |
| | exakt, φ 0,02 | +0,049 | 0,494 | −0,002 ± 0,005 |
| | exakt, φ 0,66 | −0,507 | 1,353 | −0,558 ± 1,379 |
| | linearisiert | +0,051 | 0,493 | – |
| nodown (N 512) | Forge gt_hologram = exakt, φ 0,33 | −0,032 | 0,077 | −0,008 ± 0,050 |
| | exakt, φ 0,02 | −0,025 | 0,052 | −0,002 ± 0,004 |
| | exakt, φ 0,66 | +0,000 | 0,109 | +0,024 ± 0,098 |
| | linearisiert | −0,023 | 0,051 | – |
| ds4 (N 512, Fr/4) | Forge gt_hologram = exakt, φ 0,33 | −0,034 | 0,047 | −0,026 ± 0,033 |
| | exakt, φ 0,02 | −0,010 | 0,021 | −0,002 ± 0,002 |
| | exakt, φ 0,66 | −0,063 | 0,076 | −0,054 ± 0,063 |
| | linearisiert | −0,008 | 0,020 | – |
| pad4 (N 256, Gitter 1024) | Forge gt_hologram = exakt, φ 0,33 | −0,097 | 0,443 | −0,147 ± 0,447 |
| | exakt, φ 0,02 | +0,049 | 0,493 | −0,002 ± 0,005 |
| | exakt, φ 0,66 | −0,507 | 1,353 | −0,557 ± 1,379 |
| | linearisiert | +0,051 | 0,493 | – |

Befunde: Forge-Hologramm und exaktes float64-Modell liefern in allen 60 Fällen identische Ring-Fit-Werte; im
linearen Regime (φ 0,02) ist die Differenz zum CTF-Modell null (−0,002 %); bei φ 0,33/0,66 wächst sie mit φ.
Padding 2 → 4 ändert nichts (identische Zahlen bis auf die 3. Stelle: das Objekt ist in beiden Fällen vollständig
im Gitter, der Crop identisch). Binning (8 → 4 → 1) ändert den Offset nur über N und die Objektklasse (bei N = 512
ist die Schätzerstreuung 10× kleiner), nicht über eine Skalierung ε ∝ 1/N.

### 4.5 Realer Datensatz `data_small_weak` (100 Hologramme, Seed 3000, Phantome mitgespeichert)

Gespeichertes `gt_hologram` vs. float64-Referenz aus **gespeichertem Phantom + gespeichertem Label**: rel. L2 des
Kontrasts max 4,1e-5, Mittel 1,0e-5 (float32-Speicherung). Spitzen-Phasenschub im Mittel 0,36 rad (max 0,66).
`setup.kernel`-Zugriffe während der Generierung: 101 = 100 + 1 (Label i gehört zur Propagation i).

| Hologramm-Modell | Bias [%] | sd [%] | Fr-Terzil 1 / 2 / 3 [%] | Steigung [%/Fr] | Minima-Methode [%] | − linear [%] |
|---|---|---|---|---|---|---|
| gespeichertes gt_hologram | **−0,385** | 0,367 | −0,134 / −0,377 / −0,652 | −52,3 | −0,422 | −0,387 ± 0,267 |
| exakte float64-Referenz (Phantom, Label-Fr) | −0,385 | 0,367 | −0,134 / −0,377 / −0,652 | −52,3 | −0,421 | −0,387 ± 0,267 |
| **linearisiertes CTF-Modell** | **+0,002** | 0,187 | +0,092 / −0,009 / −0,081 | −17,8 | +0,023 | 0 |
| exakt, Objekt × 0,5 | −0,193 | 0,260 | −0,021 / −0,196 / −0,368 | −35,5 | −0,200 | −0,195 ± 0,134 |
| exakt, Objekt × 2 | −0,873 | 0,665 | −0,501 / −0,795 / −1,335 | −82,7 | −1,002 | −0,875 ± 0,617 |
| Transmission 2. Ordnung `1 + iφ − A − φ²/2` | −0,393 | 0,362 | −0,143 / −0,382 / −0,661 | −52,0 | −0,440 | −0,395 ± 0,263 |

Der ursprüngliche Befund (−0,385 %, Terzile −0,13/−0,38/−0,65 %) ist exakt reproduziert, und zwar **auch von der
unabhängigen float64-Referenz mit dem Label-Fr** – die Propagation ist also nicht die Ursache. Die Ring-Fit-Variante
auf dem linearisierten Modell ist unverzerrt; das Verhältnis ×0,5 : ×1 : ×2 = 0,19 : 0,39 : 0,87 % zeigt einen
Bias ≈ linear in der Objektstärke; die auf zweite Ordnung gekürzte Transmission reproduziert den vollen Effekt.

## 5. Ursache (Mechanismus)

Für ein reines Phasenobjekt ist exp(iφ) ≈ 1 + iφ − φ²/2. Der Ring-Fit modelliert nur den linearen Term
(Spektrum φ̃·sin χ, χ = π|ξ|²/Fr, Nullstellen bei n·Fr). Der Term −φ²/2 ist reell und negativ, wirkt also wie eine
**Absorption** mit der räumlichen Verteilung φ² (andere Kantenprofile als φ, z. B. Parabel statt Wurzel bei Kugeln).
Das Hologrammspektrum wird φ̃ sin χ − μ̃ cos χ ∝ sin(χ − θ(ξ)) mit θ(ξ) = atan(μ̃/φ̃), μ̃ = FT[φ²/2]. Wäre θ
konstant, verschöbe sich nur die Phase (die Template- und die Minima-Methode haben eine freie Phase/einen freien
Achsenabschnitt). Weil μ̃/φ̃ aber frequenzabhängig ist, ist dθ/du ≠ 0, und die scheinbare Periode in u = |ξ|² wird
Fr·(1 + Fr·θ′/π): ein **relativer Fehler ∝ Fr** (gemessen −52 %/Fr) mit Vorzeichen und Betrag ∝ Objektstärke
(Ringverschiebung aus dem Kreuzterm 1.×2. Ordnung ∝ φ³/φ² = φ). Faustformel für die data_small_weak-Objekte:
ΔFr/Fr [%] ≈ −φ_peak[rad] · Fr/0,01 (0,36 rad, Fr 0,0105 → −0,38 %). Ein zusätzlicher Beitrag kommt vom Term
(Re D(φ))²/2 der Amplitude |ψ| = √I; Amplitude und Intensität unterscheiden sich im Ergebnis aber nur um 0,01 %.

Gegenprobe am Schätzer selbst (100 Hologramme `weak_clean`): Fenster none/Tukey, 4N oder 4096 Bins, mit/ohne
Vorglättung, mit/ohne SNR-Gewichte ändern den Offset nicht (−0,37 … −0,39 %), das comb-Template gibt −0,26 %,
`u_max` 0,125 statt 0,25 (nur innere Ringe) −0,20 % – der Bias sitzt also in den äußeren Ringen des Objektspektrums,
nicht in Binning oder Fensterung.

Der „unabhängige Simulator“ in `tests/test_ctf_ringfit.py` zeigte den Offset nicht, weil seine Objektklasse (Scheiben
und Rechtecke konstanter Dicke, φ_max 0,2) einen anderen, kleineren und teils positiven Bias hat (meine Messung über
8 Seeds: +0,25 % bei Fr 0,004 bis −0,10 % bei Fr 0,016, ebenfalls Fr-abhängig) und die Tests nur auf 2 % prüfen.
Das war kein Beleg für die Unverzerrtheit des Schätzers auf dem 0,1-%-Niveau.

## 6. Fix

**Kein Code-Fix** – weder in HoloForge noch an den Labels oder am Ring-Fit:

* Die Labels sind richtig; eine Korrektur `Fr_true = Fr_label·(1+ε)²` wäre falsch und würde einen Fehler *einführen*.
* Der Ring-Fit-Bias ist eine prinzipielle Grenze der Schwachobjekt-Näherung: `sin²(πu/Fr − θ₀ − θ₁u)` ist mit einem
  anderen Fr exakt entartet, aus einem einzelnen Hologramm ohne Objektwissen nicht trennbar. Die ±0,1–0,2 %
  Schätzerstreuung (lineares Modell: sd 0,19 % bei N = 256, 0,02–0,05 % bei N = 512) kommt hinzu.
* Praktische Reduktion ohne Modelländerung: Beschränkung auf niedrige Frequenzen (`u_max` 0,25 → 0,125 halbiert
  den Bias auf −0,20 %, kostet Auflösung) oder größeres N; beides keine Korrektur, nur Milderung.

Stattdessen: Regressionstests (`tests/test_forge_propagation_consistency.py`, 9 Tests, ~10 s):
Forge-Wellenfeld = Referenz (rel. L2 < 1e-5, Fr-Fit |Δ| < 1e-5, Crop zentriert); gespeichertes Hologramm aus
gespeichertem Phantom + Label reproduzierbar (rel. L2 < 1e-4, 12 Samples, `num_draws` = 13); Ring-Fit im linearen
Regime unverzerrt (|Mittel| < 0,1 %, end-to-end durch Forge mit φ 0,02 und auf dem linearisierten Modell der
Mg-Phantome); gepaarte Differenz exakt − linear < −0,2 % und monoton in der Objektstärke; **strict xfail**:
Ring-Fit vs. Label < 0,1 % auf den nominellen Mg-1–2-µm-Objekten (aktuell −0,38 %).

## 7. Auswirkung auf bisherige Ergebnisse

* `data_small`/`data_small_weak`-Labels sind um **0 %** verschoben – nichts muss neu generiert oder korrigiert werden.
* CNN (13 mm z01-MAE auf data_small): unverändert gültig; das Netz lernt gegen korrekte Labels. Die 30-µm-Mg-Objekte
  (φ ≈ 5 rad) liegen weit außerhalb des Schwachobjekt-Regimes – die 50 % Fehlschläge des Ring-Fits dort sind dieselbe
  Nichtlinearität in voller Stärke, kein Labelproblem.
* Ring-Fit auf `weak_clean` (0,85 mm z01-MAE, Bias −0,80 mm): Der Bias ist ein Methodenfehler des Ring-Fits, keine
  Untergrenze „der Daten“. Ohne Bias bliebe eine Streuung von ≈ 0,4 mm; der Satz in `reports/experiments_spectral.md`
  („bis dahin ist 0,4 % die Untergrenze jeder physikbasierten Methode auf diesen Daten“) ist zu korrigieren: 0,4 % ist
  die Grenze *CTF-basierter* Methoden bei φ ≈ 0,3 rad und Fr ≈ 0,01; sie skaliert ∝ φ·Fr. Für das 0,1-mm-Ziel
  (0,04 % in Fr) dürfte ein CTF-Ring-Fit bei Fr ≈ 0,01 nur Objekte mit φ ≲ 0,04 rad sehen – oder braucht eine
  objektabhängige Korrektur, die ein gelerntes Modell implizit liefern kann (Argument für den lernbasierten Ansatz).
* Sim-to-real: HoloForge-Hologramme sind physikalisch konsistent propagiert. Zwei Konventionsabweichungen bleiben
  (Amplitude statt Intensität, Absorption um Faktor 2 zu stark) – für Fr irrelevant, für den Kontrastabgleich mit
  realen P05-Daten aber zu beachten (reale Detektoren messen Intensität; `hologram**2` verwenden, siehe
  `HOLOGRAM_CONVENTION` in `src/data/generate_data.py`).

## 8. Nebenbefunde zu HoloForge 3.0.6 (kein Einfluss auf Fr)

1. `NFHSimulation._make_hologram` gibt `torch.abs(psi_det)` zurück (Amplitude), Docstring „squared magnitude“.
2. `PhantomGenerator._create_physical_properties`: `absorption = 2·β·k·t` wird als Amplitudendämpfung exp(−A)
   verwendet; physikalisch ist die Amplitudendämpfung k·β·t (Intensität exp(−2kβt) = exp(−μt)). Absorption also 2×.
3. Docstrings in `calc_Fr`/`unit_conversions.py` nennen cm/m/nm, die Rechnung ist für mm/mm/mm/keV korrekt.
4. Setup-Labels werden als float32 gespeichert (rel. 6e-8) – harmlos, aber `dtype=np.float64` wäre sauberer.

## 9. Text für Johannes Dora / Upstream (keine Bug-Meldung zu den Labels nötig)

> Subject: HoloForge 3.0.6 – Fresnel labels verified; two small convention questions
>
> We checked whether HoloForge propagates with exactly the Fresnel number it stores: for the stored phantom and
> label of each sample, an independent float64 propagator `ifft2(fft2(exp(i*phi - A)) * exp(-i*pi/Fr*(xi^2+eta^2)))`
> reproduces `gt_hologram` to a relative L2 of 1e-5 (float32 storage), and a 1-D fit of Fr to Forge's wave field
> returns the label to 1e-7. So the labels are correct – the 0.4 % offset we had seen is a weak-object bias of our
> CTF ring fit (second-order term of exp(i*phi) for 0.2–0.7 rad objects), not a HoloForge issue.
> Two conventions we would like to confirm: (1) `NFHSimulation._make_hologram` returns `torch.abs(psi_det)`, i.e.
> the amplitude, while the docstring says "squared magnitude" – is the amplitude intended as the training target?
> (2) `PhantomGenerator` sets `absorption = 2*beta*k*t` and the simulation applies it as `exp(-A)` to the amplitude,
> so the intensity falls off as exp(-4*k*beta*t) instead of exp(-2*k*beta*t) = exp(-mu*t). Is the factor 2 intended
> (e.g. meant for an intensity convention), or should it be `beta*k*t` for the amplitude?
> Minimal check (Python API): build an `NFHSetup`, `NFHSimulation(setup).forward(Phantom(phi + 1j*A, N), Probe(ones, G))`,
> compare `simulation.psi_det` with the expression above using `setup.Fr` – agreement at float32 level.

## 10. Reproduktion

```bash
export OMP_NUM_THREADS=2
/tmp/holo311/bin/python tools/verify_forge_fresnel.py --geometry all --n-objects 4 --forge-dataset 100 \
    --json reports/forge_label_verification.json          # ~2 min, erzeugt data/processed/verify_forge_weak/
/tmp/holo311/bin/python tools/verify_forge_fresnel.py --hdf5 data/processed/small_weak_ph/test.hdf5 --geometry small
/tmp/holo311/bin/python -m pytest -q tests/test_forge_propagation_consistency.py   # 8 passed, 1 xfailed
```
