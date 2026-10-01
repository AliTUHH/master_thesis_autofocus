# Projektplan – Masterarbeit „Learning-based autofocus for Holography“

Stand: Oktober 2026. Dieser Plan enthält bewusst keine Kalenderangaben (keine Tage/Wochen), sondern Reihenfolge, Abhängigkeiten und grobe Anteile am Gesamtaufwand. Begriffe und Formeln: siehe `01_thesis_erklaerung.md`; Quellen [n]: siehe `02_literatur.md`. Repository-Stand: Branch `cursor/holowizard-foundation-6175` mit `src/data/forge_setup.py` (`NFHRandomDistSetup`), `src/data/generate_data.py`, `src/data/dataset.py` (`HologramHDF5Dataset`, `make_dataloaders`), `src/models/cnn.py` (`AutofocusCNN`, ResNet-18-Variante), `src/train.py`, `src/evaluate.py`, `src/utils/{physics,metrics,config,plotting,targets,torch_utils}.py`, `configs/{base,data_small,data_p05}.yaml`, `tests/` (pytest-Suite), `requirements.txt`/`environment.yml`, `thesis/` (LaTeX-Gerüst). HoloWizard 3.0.6 (Python 3.11) liegt unter `/tmp/holo311`.

---

## 1. Ziele und Forschungsfragen

**Hauptziel:** Ein lernbasiertes Verfahren, das aus einem einzelnen Nahfeldhologramm die Fresnel-Zahl $\mathrm{Fr}$ bzw. den Fokus-Objekt-Abstand $z_{01}$ in Millisekunden schätzt – mit Unsicherheitsangabe – und das den modellbasierten Autofokus [2] ersetzt (bei ausreichender Genauigkeit) oder beschleunigt (Hybrid: ML-Schätzung als Startintervall).

**Forschungsfragen** (Kurzform; Kriterien in `01_thesis_erklaerung.md`, Abschnitt 9):
- F1 Genauigkeit eines CNN auf synthetischen P05-Hologrammen (Ziel MAE$(z_{01}) \le 0.5$ mm, Minimalziel $\le 1$ mm, Referenz [2]: $0.1$ mm).
- F2 Beste Eingaberepräsentation (roh / log / Amplitude / Leistungsspektrum / radiales Profil / Kombination).
- F3 Beste Zielparametrisierung ($\mathrm{Fr}$, $\log\mathrm{Fr}$, $z_{01}$) und Verlustfunktion.
- F4 Kalibrierte Posterior $q(z_{01}\mid I)$ per SBI/NPE [6] oder einfachere Unsicherheitsmodelle.
- F5 Beschleunigung von [2] durch den Hybridansatz ohne Genauigkeitsverlust.
- F6 Sim-to-real-Lücke auf P05-Messdaten, mit und ohne Fine-Tuning.
- F7 Downstream-Rekonstruktionsqualität mit geschätztem $z_{01}$.
- F8 Laufzeit (GPU/CPU) gegenüber [2].

**Nicht-Ziele:** siehe `01_thesis_erklaerung.md`, Abschnitt 10.

---

## 2. Arbeitspakete

Aufwandsanteile (grob): WP0 5 % · WP1 10 % · WP2 15 % · WP3 20 % · WP4 10 % · WP5 12 % · WP6 10 % · WP7 3 % · WP8 15 %.

### WP0 – Fundament: Repository lauffähig (5 %)

**Ziel:** Reproduzierbare Umgebung; Datengenerierung, Training, Test laufen auf CPU mit kleinen Daten durch.
**Schritte:**
1. Umgebung festschreiben: Python 3.11 + HoloWizard 3.0.6 (Version pinnen), PyTorch, h5py, xraylib, scipy, pytest, TensorBoard; `environment.yml` entsprechend ergänzen (CUDA-Variante für GPU-Rechner).
2. Smoke-Test: `python -m src.data.generate_data --config configs/data_small.yaml` → `src/train.py` mit `configs/base.yaml` → Evaluation; pytest grün.
3. Konfigurationsschema dokumentieren (`configs/*.yaml`: `setup`, `phantom`, `probe`, `hologram_noise`, `store`, Seeds pro Split).
4. Physik-Hilfsfunktionen (`src/utils/physics.py`: `fresnel_number`, `z01_from_fresnel_number`, `target_to_fr`, ...) gegen `holowizard.forge.utils.calc_Fr` und `ConeBeam.get_fr/get_z01` testen (Toleranz $10^{-9}$ relativ).
**Deliverables:** lauffähige Pipeline, `README` mit Befehlen, Tests.
**Abnahmekriterien:** `pytest` grün; ein Trainingslauf auf `data_small` beendet sich mit geloggten Metriken; `fresnel_number(250, 20000, 11, 0.0065) == 2.37252e-4` (relativ $<10^{-9}$ zu `calc_Fr`).
**Abhängigkeiten:** keine.
**Risiken → Gegenmaßnahmen:** HoloWizard-API-Änderungen → Version pinnen, Adapterschicht in `src/data/forge_setup.py` kapseln. Abtastbedingung $N \ge 1/\mathrm{Fr}$ (siehe `01_thesis_erklaerung.md`, 3.5): `configs/data_small.yaml` erfüllt sie bereits (2048 px um Faktor 8 gebinnt → 256 px, Pixel 52 µm, Padding 2, $\mathrm{Fr}\in[3.0\cdot10^{-3}, 1.8\cdot10^{-2}] \ge 1/512$), `generate_data` prüft sie automatisch (`propagator_sampling_check`, Ergebnis in `meta.json`) → bei neuen Konfigurationen die Warnung ernst nehmen; `data_small` liegt allerdings in einem anderen $\mathrm{Fr}$-Regime als P05 → nur als Funktions-/Methodentest kennzeichnen, P05-Aussagen mit `data_p05.yaml` (bzw. einer heruntergetasteten P05-Variante) belegen.

### WP1 – Onboarding und Baseline (10 %)

**Ziel:** Den Stand der Technik [2] im Code verstehen, selbst ausführen und Laufzeit/Genauigkeit messen; Referenzprotokoll festlegen.
**Schritte:**
1. Lesen: [2], `core/find_focus/find_focus_z01.py`, `core/api/functions/find_focus/find_focus.py`, `core/scripts/examples/find_focus/magnesium_wire.py`, `core/models/cone_beam.py`, `forge/experiment/simulation/nfh_simulation.py` (Leseleitfaden in `02_literatur.md`).
2. HoloWizard-Autofokus auf Forge-Hologrammen ausführen: Forge-Amplitude quadrieren (Core erwartet Intensität und bildet `sqrt`), `Measurement(z01=z_true+Δ, z01_confidence=5.0)`, `BeamSetup(energy, px_size, z02)`, Multigrid-`Options` wie im Beispielskript (Stufen 700/300/500, Downsampling 16/4/4), `Padding(MIRROR_ALL, padding_factor=4, a0=1.0)`, `DataDimensions(window_type="blackman")`.
3. Messen: Laufzeit pro Rekonstruktion und gesamt, Zahl der Nelder–Mead-Auswertungen, $|z^\ast_{01} - z_{01}^\mathrm{true}|$ für ≥ 10 Hologramme unterschiedlicher $z_{01}$ und Startfehler $\Delta \in \{-5, -2, 0, +2, +5\}$ mm.
4. Fokusserie (`focus_series_singledim`-Vorlage) für zwei Hologramme: MFE-Kurve und klassische Kriterien (`focus_loss_metrics`) plotten – Abbildung für Kapitel 2/3.
5. Tab. 2 aus [2] mit `calc_Fr` nachrechnen (bereits getan: Drähte exakt, andere Objekte $<1$ % Abweichung) und die Diskrepanz bei den Betreuern erfragen.
**Deliverables:** Notebook/Skript `tools/baseline_holowizard.py`, Tabelle „Baseline: Laufzeit & Genauigkeit“, zwei Fokuskurven-Abbildungen.
**Abnahmekriterien:** Baseline-Autofokus reproduziert $z_{01}^\mathrm{true}$ auf synthetischen Daten innerhalb $0.1$–$0.2$ mm bei Startfehler $\le 5$ mm; Laufzeiten dokumentiert (GPU) bzw. auf heruntergetasteten Daten (CPU) mit klarer Kennzeichnung.
**Abhängigkeiten:** WP0; GPU-Zugang für 2048²-Hologramme (CPU: `downsample_factor` 4–8 und kleinere Iterationszahlen, nur zur Funktionsprüfung).
**Risiken → Gegenmaßnahmen:** Kein GPU-Zugang zu Beginn → Baseline zunächst auf 512 px (downsampled) messen, Werte später auf GPU nachziehen. Baseline konvergiert auf Forge-Daten nicht → `a0`, Padding, Rauschpegel prüfen; Rücksprache mit Johannes Dora.

### WP2 – Datengenerierung mit HoloForge (15 %)

**Ziel:** Versionierte, physikalisch gültige, dokumentierte Datensätze für Training/Validierung/Test inklusive Out-of-Distribution-Testsets.
**Schritte:**
1. **$z_{01}$-Verteilung:** Prior $\mathcal U[50, 500]$ mm *und* log-uniform (gleichmäßig in $\log\mathrm{Fr}$) erzeugen; Entscheidung nach WP3-Ergebnis (Fehler-vs-$z_{01}$-Kurve). Randbereiche 20–50 mm und 500–700 mm als separate OOD-Testsets.
2. **Geometrie:** $z_{02}$ fest 20 000 mm (Hauptdatensatz) plus Variante $z_{02}\in[19\,600, 20\,000]$ mm und Energie $\{11, 17\}$ keV (Testsets für Generalisierung; Labels immer als $\mathrm{Fr}$ *und* $z_{01}$ speichern).
3. **Phantome:** Formen Rectangle/Polygon/Ellipse/Ball/Cylinder/CylinderRoundTip, 1–10 Formen, Materialien Mg (Haupt), Al, Fe, Cu, Ag, Au, Dicken 1–30 µm (Phasenschübe von $<\pi$ bis $>10\pi$), Radien/Größen über zwei Größenordnungen, Rotation; Glättung wie Forge-Standard. Separates Testset „große glatte Objekte“ und „sehr kleine Objekte“.
4. **Probe/Flatfield/Rauschen:** polynomiale Probe (linear 0–0.1, quadratisch 0–0.05), optionales Probe-Rauschen (Poisson, Forge-Standard), Hologrammrauschen gaußisch mit Intensität 0.02–0.2 (randomisiert pro Sample, Pegel als Metadatum), später gemessene Flatfields via `forge_overrides.flatfield_dataset` (TIFF) einspeisen.
5. **Größe und Format:** Hauptdatensatz 10 000 / 1 000 / 1 000 (train/val/test) in P05-Geometrie. Speicher: 2048² float32 = 16 MB pro Hologramm → 160 GB; daher (a) float16 oder (b) Speicherung heruntergetastet auf 512² (`downsample_factor 4`, $\mathrm{Fr}$ ×16, physikalisch konsistent) oder (c) zufällige 512²-Crops aus 2048²-Hologrammen mit gleichem Label (Translation ändert $\mathrm{Fr}$ nicht). Empfehlung: (b) für den Großteil, (c) für ein Teilset zur Vollauflösungs-Evaluierung.
6. **Gültigkeit prüfen:** für jede Konfiguration `detector_size/downsample_factor·padding_factor ≥ 1/Fr_min` sicherstellen (bei 2048·4 = 8192 ist $\mathrm{Fr}_\min \ge 1.2\cdot10^{-4}$, also $z_{01}\gtrsim 130$ mm bei 20 m – `data_p05.yaml` startet deshalb bei 150 mm; für kleinere $z_{01}$ Padding 8 oder Downsampling verwenden). `generate_data` führt diese Prüfung automatisch aus (Warnung) und schreibt sie als `propagator_sampling` in `meta.json`.
7. **Versionierung/Metadaten:** `meta.json` pro Datensatz mit Config-Hash, HoloWizard-Version, Seeds, Git-Commit, Erzeugungsdatum, Statistik der Labels; Datensatzname `name_vN`; niemals Dateien in-place überschreiben.
8. **Sanity-Checks:** Label-Konsistenz (`Fr == calc_Fr(energy, z01, z02, px·ds)`), Leistungsspektrum einiger Hologramme zeigt Ringe bei $|\xi|^2 = n\,\mathrm{Fr}$, Histogramme von Mittelwert/Std/Min/Max der Hologramme, Anteil gesättigter/negativer Werte = 0.
**Deliverables:** Konfigurationen `configs/data_*.yaml`, Datensätze v1 (+ OOD-Sets), `tools/inspect_dataset.py` (Statistiken, Ringplot), Datenblatt (Markdown) pro Datensatz.
**Abnahmekriterien:** Alle Sanity-Checks bestanden; Label-Rundreise $z_{01} \to \mathrm{Fr} \to z_{01}$ exakt; Datenblatt vorhanden; Erzeugung mit identischen Seeds reproduziert identische Hologramme (Hash-Vergleich einer Stichprobe).
**Abhängigkeiten:** WP0; GPU für 2048²-Erzeugung (Kernel 8192² complex64 ≈ 0.5 GB pro Sample).
**Risiken → Gegenmaßnahmen:** Speicher/Erzeugungszeit → heruntergetastete Hauptdaten, Crops, float16, Erzeugung im Batch auf GPU. Unrealistische Phantome (Sim-to-real) → Domänenrandomisierung [35], gemessene Flatfields, Materialvielfalt. Faktor 2 in der Forge-Absorption (`2*beta*k*t`) → mit Johannes Dora klären; bis dahin dokumentieren, da für $\mathrm{Fr}$-Schätzung unkritisch.

### WP3 – Modelle (20 %)

**Ziel:** Vom CNN-Baseline-Regressor zur besten Kombination aus Eingaberepräsentation, Architektur, Ziel und Verlust; Hybrid mit HoloWizard.
**Schritte (in dieser Reihenfolge, jeweils mit 3 Seeds):**
1. **Baseline:** vorhandenes `AutofocusCNN` (adaptive Pooling) und ResNet18-Variante (1 Eingangskanal, zufällig initialisiert, *ohne* Resize) auf rohen, auf Mittelwert 1 normierten Hologrammen (`normalization: divide_mean`); Ziel z-standardisiertes $\log\mathrm{Fr}$; Verlust L1 (in `src.train` bisher `mse` | `huber` – L1 ergänzen); AdamW, Cosine-Schedule, Early Stopping (derzeit auf Val-Loss; Val-MAE in mm steht in `history.json`/TensorBoard).
2. **Eingaberepräsentationen** (gleiches Netz, gleiches Budget): roh, $\log I$, $\sqrt I$, log-Leistungsspektrum $\log(|\mathcal F[I/\bar I - 1]|^2+\epsilon)$ (fftshift, ggf. Radialgewichtung), radiales Profil über $|\xi|^2$ (1D-CNN/MLP), Zweikanal (Bild + Spektrum), Multi-Crop-Aggregation (Median über $k$ Crops).
3. **Architekturen:** kleines CNN, ResNet18/34, ggf. ConvNeXt-T-artig; für Spektren zusätzlich ein explizites „Ringfit“-Modell (nicht lernend: Periodensuche im radialen Profil über $|\xi|^2$, analog CTFFIND4 [13]) als Baseline.
4. **Ziel/Verlust:** $\log\mathrm{Fr}$ vs. $\mathrm{Fr}$ vs. $z_{01}$; L1 vs. Huber vs. MSE; Klassifikation in 1-mm-Bins mit Softmax-Erwartungswert als Vergleich.
5. **Hybrid:** `Measurement(z01=ẑ01, z01_confidence=c·σ̂)` (bei Punktschätzern: $c\sigma$ aus Validierungs-Residuen, z. B. 99 %-Quantil) → HoloWizard `find_focus` → Zahl der Rekonstruktionen und Endfehler messen; Vergleich mit Start $\pm5$ mm.
6. **Regularisierung/Augmentierung:** Flips, 90°-Rotationen, Crops, Rauschinjektion, Intensitätsskalierung ($A_0$-Simulation: Multiplikation mit $a\in[0.9,1.2]$); *kein* Resize/Zoom.
**Deliverables:** Modellzoo in `src/models/`, Vorverarbeitung in `src/data/transforms.py`, Ergebnistabelle (Repräsentation × Architektur × Ziel), Hybrid-Skript.
**Abnahmekriterien:** Beste Konfiguration erreicht auf dem In-Distribution-Testset MAE$(z_{01}) \le 0.5$ mm (mindestens $\le 1$ mm) und Anteil $|\Delta z_{01}|\le1$ mm $\ge 95$ %; Hybrid reduziert die Zahl der NM-Rekonstruktionen gegenüber $\pm5$ mm um $\ge 40$ % bei gleichem Endergebnis ($\le 0.1$ mm Differenz).
**Abhängigkeiten:** WP2 (Daten v1), WP1 (Baseline für Hybrid).
**Risiken → Gegenmaßnahmen:** Netz lernt Objektgröße statt Streifenskala → Spektrum/Radialprofil als Eingabe, Robustheitstest über Objektgröße. Speicher bei 2048² → Crops/Downsampling, Mixed Precision. Überanpassung an Forge-Formen → Materialien/Formen randomisieren, OOD-Sets früh auswerten.

### WP4 – Simulationsbasierte Inferenz mit `sbi` (10 %)

**Ziel:** Kalibrierte Posterior $q_\phi(z_{01}\mid I)$ und ihr Nutzen für den Hybrid.
**Schritte:**
1. Prior: $\mathcal U[50,500]$ mm (bzw. log-uniform in $\mathrm{Fr}$); Prior Predictive Check (Hologramme/Spektren plausibel?).
2. NPE mit `sbi` (`NPE(prior, density_estimator="nsf", embedding_net=<bestes CNN aus WP3 ohne Kopf>)`), Training auf demselben Datensatz; Embedding end-to-end oder eingefroren (beides testen). Simulationsbudget-Kurve: 1k, 3k, 10k.
3. Diagnostik: SBC-Ränge (Uniformität, KS-Test), Expected Coverage für $\alpha\in\{0.5,0.8,0.9,0.95\}$, TARP, L-C2ST (falls Rechenzeit erlaubt), Posterior Predictive Check (simuliere Hologramm mit Posterior-Mittel und vergleiche radiale Spektren), NPE-Ensemble (3–5 Netze).
4. Vergleich mit einfacheren Unsicherheitsmodellen: heteroskedastische Regression (Gauß-NLL, [32]), Deep Ensemble (5 Netze, [33]), MC-Dropout [34].
5. Hybrid mit Posterior: $\Omega_z$ = 99 %-HPD-Intervall → `z01_confidence`.
**Deliverables:** `src/sbi/` (Training, Diagnostik), Kalibrierungsplots, Tabelle Unsicherheitsmethoden (NLL, Coverage, Intervallbreite, Korrelation Fehler/Std).
**Abnahmekriterien:** Expected Coverage innerhalb $\pm5$ Prozentpunkten der Nominalabdeckung auf dem Testset; SBC-KS-Test $p>0.05$; Posterior-Std korreliert mit $|\Delta z_{01}|$ (Spearman $\ge 0.5$); 95 %-Intervall im Mittel $\le 2$ mm breit. Falls nicht erreichbar: dokumentierter Fallback auf heteroskedastische Regression/Ensemble mit gleicher Diagnostik.
**Abhängigkeiten:** WP3 (Embedding-Netz, Daten).
**Risiken → Gegenmaßnahmen:** Überkonfidenz (bekanntes SBI-Problem [31]) → Ensembles, Kalibrierungs-Nachkorrektur (Temperature/Conformal-Intervalle). Fehlspezifikation auf Messdaten → WP6-Diagnostik, breitere Randomisierung.

### WP5 – Evaluation (12 %)

**Ziel:** Vollständige, reproduzierbare Bewertung gegen Baselines inklusive Ablationen, Robustheit, Laufzeit und Downstream-Qualität.
**Schritte:**
1. Evaluationsskript (`src/evaluate.py`, vorhanden): lädt Checkpoint + beliebige HDF5-Datei, berechnet die Metriken aus Abschnitt 4.2 (bisher MAE/RMSE/MedAE/p95/Bias von $z_{01}$, relativer Fr-Fehler, Inferenzzeit), speichert `eval_results.json`, `predictions.npz`, Scatter- und Fehler-vs-$z_{01}$-Plots nach `runs/<run>/eval_<split>/`; zu ergänzen: Trefferquoten $\mathrm{Acc}_\tau$, Residuenhistogramm, Kalibrierungsplot.
2. Baselines: (a) HoloWizard-Autofokus [2] (Genauigkeit/Laufzeit aus WP1), (b) Ringfit im Spektrum (WP3.3), (c) triviale Prädiktoren (Mittelwert des Priors, kNN auf Radialprofil).
3. Ablationen: Repräsentation, Ziel, Verlust, Architektur, Datenmenge (1k/3k/10k), Augmentierung, Auflösung (512 vs. 2048 Crops).
4. Robustheit: Rauschpegel-Sweep, Objektgröße (klein/mittel/groß/leeres Bild), Material/Dicke (Phasenschub-Bins), OOD-$z_{01}$ (20–50, 500–700 mm), $z_{02}$-Variation, 17 keV, Probe-Gradienten, $A_0$-Offsets.
5. Downstream: ASRM-Rekonstruktion mit $\hat z_{01}$ vs. $z_{01}^\mathrm{true}$ für 20 Testhologramme: MFE-Verhältnis, PSNR/SSIM der Phase, visuelle Beispiele.
6. Laufzeit: Median/95 %-Quantil der Inferenz pro Hologramm (Batch 1, inkl. Vorverarbeitung) auf GPU und CPU; Vergleich mit [2] (6–16 s pro Rekonstruktion × 9–13 Rekonstruktionen).
**Deliverables:** Ergebnistabellen (CSV + Markdown), Abbildungen, `results.json`/`eval_results.json` pro Lauf.
**Abnahmekriterien:** Jede Zahl in der Thesis ist über Run-ID, Config-Hash und Seed rückverfolgbar; Hauptmetriken als Mittel ± Std über 3 Seeds.
**Abhängigkeiten:** WP3, WP4; GPU für Downstream-Rekonstruktionen.
**Risiken → Gegenmaßnahmen:** Zu viele Kombinationen → Ablationen auf 512²-Daten, nur die Top-2 bei Vollauflösung.

### WP6 – Messdaten und Sim-to-real (10 %)

**Ziel:** Bewertung auf echten P05-Hologrammen mit Referenz-$z_{01}$; Quantifizierung und Verringerung der Sim-to-real-Lücke.
**Schritte:**
1. Datenzugang über Johannes Dora (DESY): mindestens die fünf Datensätze aus [2] (Spinnenhaar, Zahn, Kaktusnadel, Mg-Draht, Mg-Draht in Flusszelle) mit Rohhologrammen, Flatfields, Darkfields, Metadaten ($E$, $\Delta x$, $z_{02}$, Motorposition $z_{01}^\mathrm{est}$); ideal zusätzlich Fokusserien oder Tomographie-Projektionen (viele Hologramme pro Objekt) und weitere Beamtimes.
2. Vorverarbeitung mit HoloWizard Core: Flatfield-/Darkfield-Korrektur (ggf. `find_focus_flatfieldcorrection`-Komponentenmodell), $\sqrt I$ oder $I$ konsistent zur Trainingsrepräsentation, Downsampling/Crop exakt wie im Training (Label-Umrechnung $\mathrm{Fr}\propto\Delta x^2$).
3. Referenz-$z_{01}$: Werte aus [2] Tab. 1/2 bzw. eigener Lauf von `find_focus` (WP1) – Unsicherheit der Referenz ($\approx0.1$ mm) mitführen.
4. Evaluation ohne Anpassung (Zero-Shot): MAE, Residuen pro Objekt, Posterior-Breite; Fehlspezifikations-Diagnostik: Vergleich der Embedding-Statistiken und der radialen Spektren Mess- vs. Simulationsdaten, Posterior Predictive Check.
5. Anpassung: (a) Simulation näher an Messung (gemessene Flatfields in Forge, Rauschpegel aus Messdaten, Probe-Struktur), (b) Fine-Tuning mit wenigen Messhologrammen (Leave-one-object-out, damit kein Objekt in Training und Test), (c) Test-Time-Augmentation über Crops.
6. Hybrid auf Messdaten: ML-Intervall → `find_focus`; Laufzeit und Endergebnis gegen $\pm5$ mm-Start.
**Deliverables:** Vorverarbeitungsskript, Tabelle „Messdaten: Zero-Shot vs. Fine-Tuning vs. Hybrid“, Diagnostikplots.
**Abnahmekriterien:** Zero-Shot-MAE $\le 1$ mm (Ziel) über alle verfügbaren Messobjekte; nach Fine-Tuning $\le 0.5$ mm; Hybrid liefert auf allen Objekten das Referenz-$z_{01}$ innerhalb $0.1$ mm mit weniger Rekonstruktionen.
**Abhängigkeiten:** WP3/WP4 (Modelle), WP1 (Referenzprotokoll), Datenzugang.
**Risiken → Gegenmaßnahmen:** Daten kommen spät oder spärlich → früh anfragen (nächste Schritte), Tomographie-Projektionen als Vielzahl von Hologrammen nutzen; falls nur fünf Objekte: Ergebnisse als Fallstudie darstellen, Statistik über Crops/Projektionen. Große Sim-to-real-Lücke → spektrale Repräsentation, gemessene Flatfields, Fine-Tuning; ehrlich dokumentieren.

### WP7 – Integration (3 %)

**Ziel:** Schneller Inferenzpfad, der sich in HoloWizard-Workflows einfügt, ohne HoloWizard zu verändern.
**Schritte:** Funktion `predict_z01(hologram, beam_setup) -> (z01, sigma)` mit Vorverarbeitung und Label-Rückrechnung (`ConeBeam.get_z01`), CLI (`python -m src.predict --holo file.tiff --energy 11 --px 0.0065 --z02 20000`), Export als TorchScript/ONNX (optional), Beispiel, wie `Measurement(z01, z01_confidence)` aus der Vorhersage gebaut und an `find_focus` übergeben wird; Skizze einer Livereco-Task analog `holowizard.pipe.tasks.find_focus.FindFocusTask`.
**Deliverables:** `src/predict.py`, Beispielnotebook, Latenzmessung.
**Abnahmekriterien:** Latenz $\le 50$ ms (GPU) / $\le 2$ s (CPU) pro 2048²-Hologramm inkl. Vorverarbeitung; Beispiel läuft Ende-zu-Ende.
**Abhängigkeiten:** WP3/WP4, WP5.
**Risiken:** gering; bei Zeitdruck auf CLI + Hybrid-Beispiel beschränken.

### WP8 – Schreiben (15 %)

**Ziel:** Thesis nach Gliederung (Abschnitt 6), fortlaufend ab WP1.
**Schritte:** Kapitel 2 (Grundlagen) parallel zu WP1 aus `01_thesis_erklaerung.md` ausarbeiten; Kapitel 3 aus `02_literatur.md`; Methoden während WP2–WP4; Ergebnisse aus WP5/WP6 mit automatisch erzeugten Tabellen/Abbildungen (Skripte, keine Handarbeit); Diskussion/Ausblick zuletzt; zwei Korrekturschleifen mit Betreuern.
**Deliverables:** Thesis-PDF (LaTeX-Gerüst in `thesis/`), Kolloquiumsfolien.
**Abnahmekriterien:** Alle Forschungsfragen F1–F8 beantwortet (auch negativ), alle Zahlen rückverfolgbar, Literatur nur geprüfte Quellen.
**Abhängigkeiten:** alle WPs.
**Risiken → Gegenmaßnahmen:** Schreiben wird nach hinten geschoben → feste Regel: pro WP ein Abschnitt im Entwurf, bevor das nächste WP beginnt.

---

## 3. Ressourcen und Zugang

| Ressource | Bedarf | Status/Aktion |
|---|---|---|
| GPU | Erzeugung 2048²-Daten (Kernel 8192² complex64), Training ResNet auf 512²–2048², ASRM-Rekonstruktionen für Baseline/Downstream (6–16 s je Rekonstruktion laut [2]); ≥ 16 GB VRAM empfohlen | TUHH-Institutsserver und/oder DESY-Rechner (Maxwell) über Betreuer anfragen; Fallback Kaggle (30 GPU-h/Woche, 12-h-Sitzungen, P100/T4), Colab nur Notfall; hier verfügbare Umgebung: nur CPU |
| Speicher | 100–200 GB für Vollauflösungsdaten, 10–20 GB für 512²-Hauptdaten | Netzlaufwerk/Scratch klären; Datensätze außerhalb des Git-Repos (`data/` in `.gitignore`) |
| Messdaten | P05-Hologramme + Flatfields + Metadaten der Objekte aus [2]; Fokusserien; weitere Beamtimes | Anfrage an Johannes Dora (DESY), Hinweis auf Beamtime-IDs in [2] |
| Software | Python 3.11, HoloWizard 3.0.6 (gepinnt), PyTorch (CUDA), `sbi`, h5py, xraylib, scipy, pytest, TensorBoard | `environment.yml` pflegen; `/tmp/holo311` als Referenzumgebung |
| Kontakte | Daniel Hernández Durán (TUHH, Erstansprechpartner ML/Thesis), Johannes Dora (DESY, HoloWizard/Daten/Baseline), Tobias Knopp (TUHH, Betreuer) | regelmäßige Kurzberichte; Fragenliste in `01_thesis_erklaerung.md`/Abschlussbericht |

---

## 4. Experiment-Tracking, Reproduzierbarkeit, Metriken

### 4.1 Tracking

- Ein Lauf = ein Ordner `runs/<run-name>/` (`--run-name` von `src.train`). `src.train` schreibt dort bereits: `config.yaml` (aufgelöste Konfiguration inkl. Seed), `best.pt`, `history.json`, `results.json` (Test-Metriken, Zeiten, Inferenzzeit, Datensatz-Zusammenfassung), Plots und TensorBoard-Logs `tb/` (Train/Val-Loss, Lernrate, Val-MAE im Zielraum und in mm, rel. Fr-Fehler); `src.evaluate` ergänzt `eval_<split>/eval_results.json`. Noch zu ergänzen: `git_commit.txt` (Hash + `git diff` falls uncommitted), `meta.json`-Hash des Datensatzes, `pip freeze` – bis dahin stehen Commit-SHA und `meta.json`-Hash im Logbuch (`docs/notizen/experimente.md`).
- Namenskonvention: `JJJJ-MM-TT_<kurzname>_s<seed>` (siehe `04_werkzeuge_und_workflow.md`, Abschnitt 4.2); der Kurzname kodiert Datensatz/Repräsentation/Modell/Ziel, z. B. `small-raw-resnet18-logfr`.
- Aggregation: `tools/collect_results.py` (geplant) sammelt alle `results.json`/`eval_results.json` in eine CSV/Markdown-Tabelle (Mittel ± Std über Seeds).
- Hauptergebnisse immer mit 3 Seeds; Ablationen mindestens 1 Seed, Top-Kandidaten 3.
- Datensatzversionen unveränderlich; neue Version = neuer Name.

### 4.2 Metriken (Definitionen)

Für $N$ Testhologramme mit wahren Werten $z_i$ (bzw. $\mathrm{Fr}_i$) und Schätzungen $\hat z_i$:
- $\mathrm{MAE} = \frac1N\sum_i|\hat z_i - z_i|$ (mm), $\mathrm{RMSE} = \sqrt{\frac1N\sum_i(\hat z_i - z_i)^2}$, $\mathrm{MedAE} = \operatorname{median}_i|\hat z_i - z_i|$, Bias $=\frac1N\sum_i(\hat z_i - z_i)$ (auch je $z_{01}$-Bin).
- Relativer Fresnel-Fehler $e_{\mathrm{Fr},i} = |\widehat{\mathrm{Fr}}_i - \mathrm{Fr}_i|/\mathrm{Fr}_i$; Median und 95 %-Quantil. Näherung: $e_\mathrm{Fr}\approx\big(\tfrac{1}{z_{01}}+\tfrac{1}{z_{02}-z_{01}}\big)|\Delta z_{01}|$.
- Trefferquote $\mathrm{Acc}_\tau = \frac1N\sum_i\mathbf 1[|\hat z_i - z_i|\le\tau]$ für $\tau\in\{0.1, 0.5, 1, 5\}$ mm.
- Für probabilistische Modelle $q_i(z)$: $\mathrm{NLL} = -\frac1N\sum_i\log q_i(z_i)$; Coverage $C(\alpha) = \frac1N\sum_i\mathbf 1[z_i\in\mathrm{HPD}_\alpha(q_i)]$; Kalibrierungsfehler $\mathrm{CE} = \frac1K\sum_{k}|C(\alpha_k) - \alpha_k|$ über $\alpha_k\in\{0.1,\dots,0.9,0.95\}$; mittlere Breite des 95 %-Intervalls; SBC-Rang $r_i = \#\{z^{(s)}\sim q_i : z^{(s)} < z_i\}$ mit KS-Test gegen Uniformverteilung; Spearman-Korrelation zwischen Posterior-Std und $|\hat z_i - z_i|$; für Gauß-Vorhersagen zusätzlich CRPS.
- Hybrid: Zahl der ASRM-Rekonstruktionen $n_\mathrm{rec}$ bis Abbruch (100 µm), Gesamtlaufzeit, $|z^\ast_{01,\mathrm{hybrid}} - z^\ast_{01,\pm5\,\mathrm{mm}}|$.
- Downstream: $R_\mathrm{MFE} = \mathrm{MFE}(\hat z_{01})/\mathrm{MFE}(z_{01}^\mathrm{ref})$ (Ziel $\le1.05$); PSNR und SSIM der rekonstruierten Phase gegen die Referenzrekonstruktion.
- Laufzeit: Median und 95 %-Quantil der Latenz pro Hologramm (Batch 1) inkl. Vorverarbeitung, getrennt GPU/CPU; Speedup $= t_{[2]}/t_\mathrm{ML}$.

---

## 5. Meilensteine und Entscheidungstore

| Meilenstein | Inhalt | Tor (Entscheidung) |
|---|---|---|
| M0 | Pipeline läuft Ende-zu-Ende auf CPU (WP0) | pytest grün, Smoke-Training beendet → weiter zu WP1/WP2 |
| M1 | Baseline [2] auf Forge-Daten ausgeführt und vermessen (WP1) | GPU-Zugang bestätigt? Nein → Downsampling-Protokoll festlegen und Beschaffung eskalieren |
| M2 | Datensatz v1 + OOD-Sets mit Datenblatt (WP2) | Sanity-Checks bestanden? Ringe im Spektrum sichtbar? Nein → Konfiguration korrigieren (Padding/Downsampling) |
| M3 | CNN-Baseline auf v1 (WP3.1) | MAE $\le 1$ mm? Ja → Repräsentationsstudie; Nein → zuerst Spektrum/Radialprofil, Datenmenge, Ziel prüfen |
| M4 | Repräsentations-/Zielstudie und Hybrid abgeschlossen (WP3) | MAE $\le 0.5$ mm und Hybrid-Speedup? → SBI (WP4) starten; sonst heteroskedastische Regression + Ensemble als Unsicherheitsweg und Fokus auf Robustheit |
| M5 | Unsicherheit kalibriert (WP4) | Coverage innerhalb $\pm5$ pp? Nein → Fallback dokumentieren, Konformal-Intervalle |
| M6 | Messdaten ausgewertet (WP6) | Zero-Shot-MAE $\le 1$ mm? Nein → Fine-Tuning/Flatfield-Simulation; Ergebnisse als Fallstudie |
| M7 | Vollständiger Thesis-Entwurf (WP8) | Betreuer-Feedback eingearbeitet → Abgabe |

---

## 6. Gliederung der Thesis

1. **Einleitung** – Motivation (Online-Rekonstruktion, In-situ-Experimente an P05); Problem (Fresnel-Zahl/$z_{01}$ unbekannt auf $\pm5$ mm); Stand der Technik und seine Laufzeit [2]; Ziel und Forschungsfragen F1–F8; Beitrag; Aufbau.
2. **Grundlagen** – Röntgenbrechungsindex, Projektionsnäherung, Austrittswelle; Fresnel-Propagation und Fresnel-Zahl (Pixel vs. Struktur), Nah-/Fernfeld, Abtastbedingung; Kegelstrahl und Fresnel-Skalierungstheorem mit Zahlenbeispiel; Phasenproblem, CTF/TIE/Paganin, iterative Verfahren, ASRM; Autofokus als geschachteltes Problem, klassische Kriterien, MFE; Lernbasierte Regression, Unsicherheit (aleatorisch/epistemisch), SBI/NPE und Kalibrierungsdiagnostik.
3. **Stand der Forschung** – Autofokus in DH/DHM (klassisch, lernbasiert: Ren 2018, Pitkäaho 2019, Jaferzadeh 2019, FocusNET 2023; Review Rivenson 2019); Defokusschätzung aus Spektren (CTFFIND4); Deep Learning in NFH (SelfPhish); SBI in der Bildgebung; Abgrenzung: Kegelstrahl-NFH, starke Objekte, kontinuierliches $z_{01}$, Unsicherheit.
4. **Methoden** – Simulator und Datensätze (HoloForge, Randomisierung, Splits, Gültigkeit); Eingaberepräsentationen; Modelle und Ziele; Verluste und Training; SBI-Aufbau und Diagnostik; Hybridverfahren; Baselines (HoloWizard-Autofokus, Ringfit, triviale Prädiktoren); Evaluationsprotokoll und Metriken; Vorverarbeitung der Messdaten.
5. **Experimente und Ergebnisse** – Baseline-Vermessung; Ergebnisse auf synthetischen Daten (Hauptmetriken, Fehler-vs-$z_{01}$); Ablationen (Repräsentation, Ziel, Architektur, Datenmenge); Robustheit (Rauschen, Objektgröße, Material, OOD); Unsicherheit und Kalibrierung; Hybrid (Rekonstruktionen, Laufzeit); Messdaten (Zero-Shot, Fine-Tuning, Hybrid); Downstream-Rekonstruktionen; Laufzeiten.
6. **Diskussion** – Beantwortung F1–F8; Vergleich mit [2] in Genauigkeit und Laufzeit; Sim-to-real-Lücke und Fehlspezifikation; Grenzen (Datenbasis, Geometrieabhängigkeit, Referenzunsicherheit); Bedrohungen der Validität.
7. **Zusammenfassung und Ausblick** – Kernergebnisse; Empfehlungen für den Einsatz an P05 (Livereco); Ausblick: gemeinsame Schätzung von $z_{01}$ und $A_0$, Übertragung auf andere Beamlines/Energien, aktive Simulation (sequenzielle SBI), Kopplung mit lernbasiertem Phase Retrieval.

---

## 7. Globale Risiken

| Risiko | Wirkung | Gegenmaßnahme |
|---|---|---|
| Kein/zu später GPU-Zugang | Vollauflösungsdaten, Baseline-Rekonstruktionen und Downstream-Tests nicht möglich | früh anfragen; Fallback Kaggle; alle Studien zunächst auf 512²-Daten (downsampled, physikalisch konsistent) |
| Messdaten fehlen | F6 nicht beantwortbar | Anfrage sofort stellen; Tomographie-Projektionen als Hologramm-Pool; Fallstudien-Format |
| Sim-to-real-Lücke groß | Modell auf Messdaten unbrauchbar | spektrale Eingaben, gemessene Flatfields in Forge, Fine-Tuning, Hybrid als „sicherer“ Nutzen |
| SBI-Posterior überkonfident | F4 negativ | Ensembles, Konformal-Intervalle, heteroskedastischer Fallback |
| 256-px-Smoke-Daten (`data_small`, Fr $3\cdot10^{-3}$–$1.8\cdot10^{-2}$) liegen in einem anderen Fr-Regime als P05 ($10^{-4}$–$5\cdot10^{-4}$) | falsche Schlüsse aus CPU-Experimenten | Abtastbedingung ist erfüllt und wird von `generate_data` geprüft; trotzdem nur Funktions-/Methodentests; P05-Aussagen mit `data_p05.yaml` oder heruntergetasteter P05-Variante (Schritt 2 in Abschnitt 8) belegen |
| Konventionsabweichungen (Amplitude vs. Intensität, Faktor 2 Absorption) | Verteilungsverschiebung Training/Test | in Vorverarbeitung festnageln (`sqrt` konsistent), mit Betreuern klären, Tests schreiben |

---

## 8. Nächste 5 konkrete Schritte

1. **Betreuer anschreiben** (Daniel Hernández Durán, Johannes Dora): GPU-Zugang (TUHH/DESY), P05-Messdaten der Objekte aus [2] inkl. Flatfields und Metadaten, Klärung der offenen Fragen (Forge-Amplitude vs. Intensität, Faktor 2 in der Absorption, Abweichungen <1 % bei Tab. 2 in [2], welche Unsicherheitsangabe an der Beamline nützlich ist).
2. **Heruntergetastete P05-Konfiguration für CPU anlegen** (`configs/data_cpu512.yaml`: `detector_size: 2048` mit `downsample_factor: 4` → 512 px und 26-µm-Pixel, `padding_factor: 4` → Gitter 2048, $z_{01}\in[50,500]$ mm; damit $\mathrm{Fr}\in[7.5\cdot10^{-4}, 7.7\cdot10^{-3}]$ und $1/\mathrm{Fr}_\min \approx 1331 \le 2048$ – `generate_data` prüft das automatisch), Datensatz v1 (2k/500/500) erzeugen, `tools/inspect_dataset.py` mit Radialspektrum-Plot schreiben und die Ringe bei $|\xi|^2 = n\,\mathrm{Fr}$ sichtbar machen.
3. **Baseline [2] auf 10 Forge-Hologrammen laufen lassen** (`find_focus` mit quadrierter Forge-Amplitude, Startfehler ±2/±5 mm), Rekonstruktionen zählen, Laufzeit messen, Fokuskurve plotten (`focus_series_singledim`-Vorlage).
4. **CNN-Baseline mit 3 Seeds trainieren** (ResNet18, $\log\mathrm{Fr}$ z-standardisiert, L1) und zusätzlich die Spektrum-Variante; Fehler-vs-$z_{01}$-Plot; Entscheidung am Tor M3.
5. **Kapitel 2 der Thesis beginnen** – Abschnitte 2–6 aus `01_thesis_erklaerung.md` in `thesis/chapters/` übertragen, Abbildungen Fokuskurve (Schritt 3) und Ringspektrum (Schritt 2) einbauen, Literatur aus `02_literatur.md` in `thesis/references.bib` übernehmen (nur geprüfte Einträge).
