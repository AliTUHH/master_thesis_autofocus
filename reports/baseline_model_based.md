# Modellbasierter Autofokus (HoloWizard `find_focus`) als Baseline und Downstream-Test

Branch `cursor/model-based-baseline-6175`. Kondensierte Fassung des Prototyp-Berichts
(`reports/baseline_model_based_prototype_report.md`, 1:1-Kopie von `/tmp/baseline_proto/REPORT.md`); dort stehen
alle Details (API-Analyse Zeile für Zeile, Einzelwerte, Anhang mit Workarounds). Alle Zahlen stammen aus den
Prototyp-Läufen (HoloWizard 3.0.6, torch CPU, 2 Threads, 256-px-Hologramme, RSS < 1,4 GB; die Maschine war
während aller Läufe durch weitere Agenten ausgelastet – Laufzeiten sind obere Schranken). Die Ergebnisdateien
liegen unverändert in `reports/baseline_model_based/{p05bin8,small,figures}/` und `reports/downstream/{p05bin8,small}/`;
der Code wurde in `src/` integriert (Abschnitt 2) und mit Smoke-Läufen der neuen CLIs validiert (Abschnitt 10).

**Regime-Hinweis (wichtig für den Vergleich mit den Thesis-Ergebnissen):** Der Prototyp nennt das Regime mit
`detector_size 2048`, `downsample_factor 8` (256 px, px 0,052 mm, Fr 3e-3…2,4e-2) „p05bin8“ und das ungebinnte
256-px-Regime (px 0,0065 mm, Fr 5e-5…4e-4) „small“. **Der Datensatz dieses Repos `data/processed/small`
(`configs/data_small.yaml`) entspricht dem Prototyp-Regime „p05bin8“**, nicht dem Prototyp-Regime „small“ (das in
diesem Repo wegen der Abtastbedingung Fr ≥ 1/(N·pf) nicht mehr verwendet wird). Die Ordnernamen unter `reports/`
folgen dem Prototyp.

| Verzeichnis / Datei | Inhalt |
|---|---|
| `reports/baseline_model_based/p05bin8/` | Baseline auf 16 Hologrammen (4 je z01 ∈ {50, 100, 250, 400} mm): `results.csv` (Prototyp-Schema, von `src.baseline.results.read_samples_csv` lesbar), `summary.json`, `histories.json` (Zielfunktionswerte je Auswertung), `scatter.png`, `objective_curves.png` |
| `reports/baseline_model_based/small/` | dasselbe für 12 Hologramme im Fr-1e-4-Regime |
| `reports/baseline_model_based/figures/` | `scan_focus_landscape_p05bin8.{json,png}` (Zielfunktions-Scan, Abschnitt 7), `bench_iteration_time.json` (Laufzeit-Benchmark, Abschnitt 8) |
| `reports/downstream/p05bin8/`, `reports/downstream/small/` | Downstream-Test: `downstream_results.csv`, `downstream_summary.json`, `downstream_table.md`, `downstream_error_vs_fr_error.png`, `downstream_image_grid.png` |

## 0. Kurzfassung

* **Autofokus-API**: `holowizard.core.api.functions.find_focus.find_focus.find_focus(reco_params, viewer=None, plotter=None)`
  → scipy Nelder-Mead in 1-D über z01 innerhalb `Measurement.z01_bounds = z01 ± z01_confidence`; Kriterium = Daten-Residuum
  (MSE der Amplituden im FOV) der **letzten Iteration einer kompletten mehrstufigen ASRM/PGD-Rekonstruktion** pro Auswertung
  (Dora et al., Opt. Express 33(4), 6641 (2025)). Kein hart kodiertes CUDA. Es gibt **keinen** klassischen (metrikbasierten)
  Autofokus in der API; die sechs Schärfemetriken in `core/find_focus/focus_loss_metrics.py` werden nirgends benutzt.
* **Fr-Definition in `core` und `forge` identisch** (Fr = px²·z01/(λ·z02·(z02−z01)), mm/mm/keV, λ = 1,2398/E nm; rel. Abweichung
  ≤ 4,7e-8 durch µm-Rundung in forge) – dieselbe Formel wie `src.utils.physics.fresnel_number`. Zwei Konventionen sind für die
  Kopplung zwingend: (1) HoloForge speichert die **Amplitude** |ψ| (+ Rauschen), die Core-API erwartet **Intensitäten**
  (zieht intern `torch.sqrt`) → Hologramm quadrieren; (2) Core dreht die Messung um 90° (`torch.rot90`) → Rekonstruktionen mit
  `torch.rot90(x, k=-1)` zurückdrehen, bevor sie mit `images/phantoms` verglichen werden.
* **Downstream-Test (p05bin8, 8 Hologramme × 9 Fr-Fehler, 699 s)**: GT-NRMSE der Phase bei ±20 % Fr-Fehler praktisch unverändert
  (Median 0,75 → 0,70–0,80; dominiert von der Niederfrequenz-Schlechtgestelltheit der Einzel-Distanz-Rekonstruktion). Empfindlich
  sind der **Gradienten-NRMSE** (+5 % bei |e| = 5 %, +10 % bei 20 %) und das **Vorwärtsmodell-Residuum beim wahren Fr**
  (×1,3 bei ±1 %, ×2,1–2,2 bei ±5 %, ×2,8–3,1 bei ±20 %).
* **Anforderung**: Ein relativer Fr-Fehler e wirkt wie eine Rest-Defokussierung mit Unschärfe **b = sqrt(|e|/Fr)** Detektorpixel
  (`src.utils.fresnel.defocus_blur_px`); Ziel b ≤ 1–2 px → bei P05-Vollauflösung |e| ≤ 0,02–0,1 % (Δz01 ≤ 0,06–0,25 mm bei
  z01 = 250 mm), im 8×-gebinnten 256-px-Regime |e| ≤ 1,5–6 %. `z01_tol = 0,1 mm` (P05) entspricht b ≈ 1,3 px (Abschnitt 5).
* **Baseline (Suchbereich Fr ± 50 %, 256 px, CPU)**: p05bin8 (16 Hologramme) – `find_focus` MAE 3,68 % (Median 2,70 %, p95 10,3 %),
  MAE Δz01 10,8 mm, 21 Auswertungen, 213 s pro Hologramm (9,9 s pro Auswertung); bei Fr ≤ 6e-3 ≤ 2,1 % Fehler, bei Fr ≥ 1,5e-2
  Bias −3…−12 % (asymmetrische Zielfunktion, Abschnitt 7). Klassischer Fallback (TV der rückpropagierten Amplitude): Median 0,67 %,
  aber 4/16 grobe Fehlschätzungen (+15…+44 %), 0,19 s. Fr-1e-4-Regime (12 Hologramme): `find_focus` MAE 2,90 % (Median 1,05 %),
  179 s – obwohl die Rekonstruktion dort scheitert; der Fallback versagt dort vollständig (Median 27 %).
* **Hochrechnung 2048 px**: P05-Standardkonfiguration ≈ 7,7 min pro Auswertung, ≈ 2,7 h pro Hologramm auf 2 CPU-Threads → auf CPU in
  voller Auflösung nicht praktikabel (P05-Betrieb: GPU). Die 256-px-Variante (≈ 3,5 min pro Hologramm) ist als Referenz auf einigen
  Dutzend Hologrammen machbar.

## 1. HoloWizard-API und Konventionen

| Ebene | Pfad (relativ zu `holowizard/`) | Inhalt |
|---|---|---|
| Öffentliche API | `core/api/functions/find_focus/find_focus.py` | `find_focus(reco_params, viewer=None, plotter=None) -> (z01, z01_history, loss_history)`; zieht zuerst `torch.sqrt` auf `measurements[i].data` (erwartet Intensitäten) |
| Mit Flatfield-PCA | `core/api/functions/find_focus/find_focus_flatfieldcorrection.py` | Variante, die `pipe/tasks/find_focus.py` und `livereco` nutzen |
| Optimierer | `core/find_focus/find_focus_z01.py` | `scipy.optimize.minimize(..., method="Nelder-Mead", bounds=[z01_bounds], options={"xatol": options[-1].z01_tol, "fatol": 1e6, "initial_simplex": [[z01_lo],[z01_hi]]})`; jede Auswertung = `reconstruct_multistage.reconstruct` → `float(loss_se_all[-1])`; Cache bereits besuchter z01 (zweite Anfrage → `sys.float_info.max`); Historien sind **Modul-Globals** (nicht threadsicher) |
| Varianten (nicht verdrahtet) | `find_focus_z01_a0.py`, `find_focus_z01_a0_orthogonal_search.py`, `focus_loss_metrics.py` | 2-D-Suche (z01, a0); alternierende Suche; sechs ungenutzte Schärfemetriken |
| Default-Parameter | `core/api/parameters/default_parameters/find_focus.py` | 700/300/500 Iterationen, update_rate 0,9/1,1/1,1, l2 0,1j/0,1j/0,01j, fwhm (2+0j)/(2+8j)/(2+8j), Nesterov 1,0 mit (8+8j)/(16+16j)/(16+16j), MIRROR_ALL, pf 4, dsf 16/4/4 |
| P05-Pipeline | `pipe/scripts/config/find_focus/defaults.yaml`, `scan.yaml` | gleiche Stufen mit l2 **10j/10j/1j**, `z01_tol 0.1` mm, `z01_confidence 5` mm, 2048×2048, blackman; Beispiel E 17 keV, z01 190 mm, z02 19826,47 mm |
| Rekonstruktion | `core/api/functions/single_projection/reconstruction.py` | `reconstruct(reco_params, viewer=[]) -> (x (complex, FOV-Crop), se_losses_all)`; ebenfalls `sqrt` auf die Daten (mutiert `measurements[i].data`) |
| Verlust | `core/reconstruction/gradients/analytical.py::get_gradient` | Σ_FOV (A_pred − A_meas)² / N_FOV / n_meas mit A_pred = abs(D_Fr(exp(i·O)·P)), A_meas = sqrt(I) nach /a0, rot90, Downsampling, Spiegel-Padding, Blackman-Fenster |

Parameterobjekte (`core/parameters/*.py`): `BeamSetup(energy, px_size, z02)` [keV, mm, mm]; `Measurement(z01, data, z01_confidence=5.0)`
[mm]; `Regularization(iterations, update_rate, gaussian_filter_fwhm=0j, values_min/max (Phase ≤ 0, Absorption ≥ 0), l2_weight=0j, l1_weight=0j)`
(Realteil = Phase, Imaginärteil = Absorption); `Padding(padding_mode, padding_factor=2, down_sampling_factor=1, cutting_band=0, a0=1.0)`;
`Options(regularization_object, nesterov_object, z01_tol=0.1, padding, verbose_interval=100)`;
`DataDimensions(total_size, fov_size, window_type, fading_width=[(80, 80), (80, 80)])`; `RecoParams(beam_setup, measurements, reco_options, data_dimensions, output_path)`.
Vorverarbeitung je Stufe: `px_size·dsf` (Fr·dsf²), Messung /a0, `rot90`, bilineares Downsampling, Spiegelung auf 4×FOV, Crop auf
`total_size·pf/dsf`, Blackman-Fenster (`fading_width`/dsf). Rechengitter einer Stufe bei 256 px, pf 2: 512² (dsf 1), 256² (dsf 2), 128² (dsf 4).

Konventionen `core` vs. `forge` (numerisch verifiziert, Prototyp-Bericht Abschnitt 1.5):

| Aspekt | Befund | Konsequenz im Repo |
|---|---|---|
| Fr-Formel, Einheiten, Downsampling (Fr·dsf²), Padding (`fftfreq` in Zyklen/Pixel), Propagator `exp(−iπ/Fr·(ξ²+η²))` | identisch | `src.utils.physics` ist die einzige Fr↔z01-Implementierung; `src.utils.fresnel` implementiert den Kernel |
| Messgröße | forge: Amplitude abs(ψ) + Gauß-Rauschen σ 0,05 (`images/hologram`); core: Intensität (API zieht `sqrt`) | `model_based_autofocus`/`reconstruct` quadrieren; `ForgeSample.hologram_intensity` |
| Objekt | O = φ + iμ, Welle exp(i·O)·P, φ ≤ 0, μ ≥ 0 (beide) | `images/phantoms.real` ist die GT-Phase |
| Orientierung | core: `torch.rot90` vor der Rekonstruktion, Ergebnis nicht zurückgedreht | `torch.rot90(result, k=-1)` in `src.eval.downstream.reconstruct` (Test `tests/test_model_based_autofocus.py`) |
| Metadaten | `metadata/setup/*` float32; `detector_px_size` = **effektiver** (gebinnter) Pixel, `detector_size` gebinnt | `src.data.forge_samples` liest sie direkt; gespeichertes Fr hat float32-Genauigkeit (rel. 1,4e-8) |
| Logging | `Logger.configure` registriert custom Log-Level, Geometrie-Datei je Rekonstruktion | einmalige Konfiguration in `configure_holowizard` (Standard: `$HOLOWIZARD_LOG_DIR` oder Temp-Verzeichnis) |
| Import | `holowizard.core` ruft `nvidia-smi` auf, setzt `tempfile.tempdir` | Import lazy (`_import_core`), Module ohne HoloWizard importierbar |

## 2. Module in diesem Repo

| Modul / Funktion | Zweck |
|---|---|
| `src/utils/fresnel.py` – `fresnel_kernel`, `propagate`, `backpropagate`, `hologram_amplitude`, `defocus_blur_px`, `blur_px_from_fresnel_numbers` | Torch-Vorwärtsmodell in HoloForge/Core-Konvention (Zero-Padding, zentraler Crop), Unschärfe b = sqrt(\|e\|/Fr) |
| `src/utils/metrics.py` – `metrics_in_physical_units` | zusätzlich `blur_px_mean/median/p95` (wird von `src.train`/`src.evaluate` automatisch mitgeschrieben) |
| `src/data/forge_samples.py` – `ForgeSample`, `read_forge_samples` | Roh-Reader für Hologramm (Amplitude), Phantom und Geometrie-Labels |
| `src/baseline/results.py` – `RESULT_COLUMNS`, `make_result_row`, `summarize`, `read_samples_csv`, `plot_scatter`, `summary_markdown_table` | gemeinsames Ergebnisschema aller Methoden (`index, source, fr_true, z01_true_mm, fr_est, z01_est_mm, rel_err_fr_pct, dz01_mm, blur_px, runtime_s, n_evals, method`); liest auch Ring-Fit- und Prototyp-CSVs |
| `src/baseline/model_based_autofocus.py` – `Geometry`, `AutofocusResult`, `PRESETS`, `model_based_autofocus`, `classical_autofocus`, CLI | Wrapper um `find_focus` (Suche in z01, Intervallmitte als Startwert, `z01_confidence` = halbe Breite), klassischer Schärfemetrik-Fallback, Baseline-CLI |
| `src/eval/downstream.py` – `reconstruct`, `reconstruction_metrics`, `data_residual`, `downstream_eval`, `downstream_curve`, CLI | Rekonstruktion bei gegebenem Fr (zurückgedreht), Phasenmetriken (NRMSE, offsetfrei, Gradient, Pearson, SSIM), Vorwärtsmodell-Residuum, Fehlerkurve und Kandidatenvergleich (`--candidates-csv`) |
| `src/baseline/ctf_ringfit.py`, `src/evaluate.py` | schreiben jetzt zusätzlich das gemeinsame Schema (`samples.csv`, Methoden `ringfit_<template>` bzw. `ml_<arch>`) |

Signaturen:

```python
Geometry(energy_kev, px_mm, z02_mm).fresnel(z01_mm) / .z01_from_fresnel(fr)        # delegiert an src.utils.physics
model_based_autofocus(hologram_amplitude, geom, fr_bounds, preset="p05filter_dsf2_1", z01_tol_mm=0.1,
                      iterations_scale=1.0, threads=2) -> AutofocusResult(fr_est, z01_est_mm, n_evals, runtime_s,
                                                                           z01_history, loss_history, method, fr_history, fr_bounds, preset)
classical_autofocus(hologram_amplitude, fr_bounds, metric="tv"|"var"|"lap", n_grid=21, geom=None) -> AutofocusResult
reconstruct(hologram_amplitude, geom, fr, preset="quality_256", iterations_scale=1.0) -> RecoResult(phase, absorption, losses, runtime_s, z01_mm, fr, preset)
reconstruction_metrics(phase_hat, phase_gt, border=16) -> {nrmse, nrmse_offset_free, nrmse_grad, pearson, ssim}
data_residual(obj_hat, hologram_amplitude, fr_true, padding_factor=2.0) -> float      # RMSE, = Rauschniveau 0,05 für das wahre Objekt
downstream_eval(sample, fr_candidates={"true": Fr, "ml_cnn": Fr_ml, ...}) -> list[dict]  # alle Metriken + blur_px + *_rel_true
downstream_curve(sample, errors_pct=(-20, -10, -5, -1, 0, 1, 5, 10, 20)) -> list[dict]
```

## 3. Stufen-Presets (`src.baseline.model_based_autofocus.PRESETS`)

| Preset | Zweck | pf | Stufen (Iterationen, dsf, l2 Absorption, fwhm Objekt / Nesterov, Nesterov-Rate) | Gitter bei 256 px | Bemerkung |
|---|---|---:|---|---|---|
| `p05filter_dsf2_1` (**Default Autofokus**) | focus | 2 | 300 (dsf 2, 1j, 2+8j / 8+8j, 1,0); 300 (dsf 1, 1j, 1+8j / 4+4j, 1,0) | 256², 512² | einzige Variante mit glatter Mulde bei 256 px; Bias −5…−10 % bei Fr ≥ 1,5e-2; 7,8–9,9 s/Auswertung |
| `p05_default_2048` | focus | 4 | 700 (dsf 16, 10j, 2+0j / 8+8j); 300 (dsf 4, 10j, 2+8j / 16+16j); 500 (dsf 4, 1j, 2+8j / 16+16j) | – (für 2048 px) | `pipe/.../find_focus/defaults.yaml`; ≈ 7,7 min/Auswertung auf CPU |
| `p05scaled_dsf4_2` | focus | 2 | 300 (dsf 4, 10j); 200 (dsf 2, 1j) | 128², 256² | P05 1:1 skaliert: **kein Minimum** bei 256 px (Negativergebnis, läuft an die Suchgrenze) |
| `quality_256` (**Default Downstream**) | quality | 2 | 300 (dsf 4, 10j, 2+0j / 4+4j, 0,9); 200 (dsf 2, 1j, 2+8j / 8+8j, 0,9); 300 (dsf 1, 0,1j, 0+2j / 2+2j, 0,9) | 128², 256², 512² | ≈ 8 s; Nesterov 0,9 statt 1,0 (sonst Divergenzen); als Autofokus-Kriterium ungeeignet (monoton) |
| `smoke` | focus | 2 | 20 (dsf 2); 20 (dsf 1) | 64², 128² bei 64 px | nur Tests/Smoke-Läufe |

`iterations_scale` skaliert alle Iterationszahlen (150/150 statt 300/300 ≈ halbe Laufzeit). Mit Nesterov 1,0 divergierte die
schwach regularisierte Rekonstruktion bei einigen Phantomen (Phase → −600 rad, Loss ≈ 1, NaN); die stark regularisierten
Autofokus-Presets behalten 1,0 (keine Divergenz beobachtet).

## 4. Downstream-Test: Rekonstruktionsqualität vs. Fr-Fehler

Pro Sample und e ∈ {−20, −10, −5, −1, 0, 1, 5, 10, 20} % wird mit Fr' = Fr_true·(1+e/100) rekonstruiert (Preset `quality_256`),
Fr' über z01' = z01(Fr') eingespeist. Metriken gegen `images/phantoms.real` (Rand 16 px abgeschnitten): NRMSE, offsetfreier NRMSE,
NRMSE der Bildgradienten, Pearson, SSIM; zusätzlich das Core-Loss und das Residuum des Vorwärtsmodells beim **wahren** Fr
(Untergrenze = Rauschniveau 0,05). Mediane über die Hologramme, „rel.“ = gepaart relativ zu e = 0.

**p05bin8** (8 Hologramme, je 2 aus z01 = 50/100/250/400 mm; 72 Rekonstruktionen, 699 s; `reports/downstream/p05bin8/`):

| e [%] | b [px] | div. | NRMSE | NRMSE offsetfrei | NRMSE Gradient | Pearson | SSIM | Core-Loss | Residuum Fr_true | NRMSE rel. | Grad. rel. | Residuum rel. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −20 | 5,1 | 0/8 | 0,802 | 0,838 | 1,414 | 0,616 | 0,093 | 3,615e-4 | 0,1804 | 1,00 | 1,04 | 3,09 |
| −10 | 3,6 | 0/8 | 0,782 | 0,845 | 1,366 | 0,617 | 0,098 | 2,940e-4 | 0,1566 | 1,02 | 1,01 | 2,63 |
| −5 | 2,6 | 0/8 | 0,767 | 0,822 | 1,286 | 0,614 | 0,106 | 2,594e-4 | 0,1415 | 1,01 | 0,97 | 2,21 |
| −1 | 1,1 | 0/8 | 0,757 | 0,823 | 1,285 | 0,625 | 0,116 | 2,217e-4 | 0,0788 | 1,01 | 0,99 | 1,35 |
| 0 | 0,0 | 0/8 | 0,750 | 0,814 | 1,321 | 0,641 | 0,117 | 2,128e-4 | 0,0538 | 1,00 | 1,00 | 1,00 |
| +1 | 1,1 | 0/8 | 0,749 | 0,813 | 1,344 | 0,630 | 0,114 | 2,089e-4 | 0,0759 | 0,99 | 1,02 | 1,29 |
| +5 | 2,6 | 0/8 | 0,719 | 0,769 | 1,395 | 0,699 | 0,097 | 1,887e-4 | 0,1315 | 0,98 | 1,05 | 2,13 |
| +10 | 3,6 | 0/8 | 0,731 | 0,791 | 1,432 | 0,656 | 0,082 | 1,740e-4 | 0,1488 | 0,97 | 1,08 | 2,49 |
| +20 | 5,1 | 0/8 | 0,695 | 0,751 | 1,463 | 0,706 | 0,066 | 1,387e-4 | 0,1650 | 0,99 | 1,10 | 2,76 |

**small = Fr-1e-4-Regime** (4 Hologramme aus z01 = 100/250 mm; 36 Rekonstruktionen, 678 s; `reports/downstream/small/`):

| e [%] | b [px] | div. | NRMSE | NRMSE offsetfrei | NRMSE Gradient | Pearson | SSIM | Core-Loss | Residuum Fr_true | NRMSE rel. | Grad. rel. | Residuum rel. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −20 | 37,6 | 0/4 | 3,307 | 1,290 | 3,909 | 0,205 | 0,007 | 3,808e-2 | 0,5146 | 0,86 | 1,09 | 1,13 |
| −10 | 26,6 | 0/4 | 3,189 | 1,373 | 4,414 | 0,162 | 0,005 | 4,100e-2 | 0,4617 | 0,83 | 1,10 | 1,08 |
| −5 | 18,8 | 0/4 | 3,480 | 1,408 | 4,634 | 0,135 | 0,005 | 3,952e-2 | 0,4793 | 0,92 | 1,08 | 1,07 |
| −1 | 8,4 | 0/4 | 3,321 | 1,312 | 3,342 | 0,180 | 0,010 | 2,984e-2 | 0,4362 | 0,97 | 0,92 | 1,03 |
| 0 | 0,0 | 0/4 | 3,430 | 1,348 | 3,824 | 0,142 | 0,007 | 3,620e-2 | 0,4284 | 1,00 | 1,00 | 1,00 |
| +1 | 8,4 | 0/4 | 2,994 | 1,335 | 3,806 | 0,126 | 0,005 | 3,672e-2 | 0,4663 | 0,91 | 1,04 | 1,05 |
| +5 | 18,8 | 0/4 | 3,165 | 1,409 | 4,623 | 0,132 | 0,005 | 4,874e-2 | 0,4590 | 0,88 | 1,21 | 1,09 |
| +10 | 26,6 | 0/4 | 3,007 | 1,496 | 5,904 | 0,054 | 0,003 | 5,695e-2 | 0,4581 | 0,82 | 1,45 | 1,14 |
| +20 | 37,6 | 0/4 | 2,957 | 1,537 | 5,906 | −0,010 | 0,004 | 5,382e-2 | 0,4892 | 0,85 | 1,27 | 1,14 |

Interpretation:

1. Rückpropagation mit Fr' und Vorwärtsmodell mit Fr ergeben zusammen einen Rest-Propagator mit Fr_eff ≈ Fr/|e|; die Unschärfe
   (Radius der ersten Fresnel-Zone) ist **b = sqrt(|e|/Fr)** Detektorpixel – die tolerierbare relative Fr-Genauigkeit ist proportional zu Fr.
2. Im p05bin8-Regime ist der GT-NRMSE innerhalb ±20 % praktisch konstant (gepaart 0,97–1,02): die Phasenrekonstruktion aus einer
   Distanz ist bei 256 px von Regularisierung und fehlender Niederfrequenz-Information dominiert. Der Gradienten-NRMSE verschlechtert
   sich ab b ≈ 2,6 px messbar (+5 % bei +5 %, +8 % bei +10 %, +10 % bei +20 %); das Residuum beim wahren Fr reagiert am empfindlichsten
   (×1,3 bei ±1 %, ×2,1–2,2 bei ±5 %, ×2,8–3,1 bei ±20 %): ein Fr-Fehler erzeugt Kanten-Ringing, das das Hologramm nicht erklärt.
3. Das Core-Loss der Quality-Stufen ist **nicht** beim wahren Fr minimal, sondern fällt monoton zu größerem Fr (2,13e-4 → 1,39e-4 bei +20 %):
   mit schwacher Absorptions-Regularisierung „fittet“ der Absorptionskanal die Daten für zu großes Fr weg → Autofokus-Stufen brauchen die
   starke P05-Regularisierung (Abschnitt 7).
4. Im Fr-1e-4-Regime scheitert bereits die Rekonstruktion mit korrektem Fr (NRMSE 3,4, Pearson 0,14, Core-Loss 3,6e-2 statt 2e-4): bei
   256 px liegt das erste CTF-Maximum bei Perioden sqrt(2/Fr) ≈ 90–150 px ≈ FOV; der Downstream-Test ist dort nicht aussagekräftig.
   Ergebnis bei korrektem Fr (p05bin8): NRMSE 0,40–0,89, Pearson 0,40–0,92, Residuum 0,043–0,135 (Rauschen 0,05); ≈ 8 s pro Rekonstruktion.

## 5. Anforderung an den Autofokus: b = sqrt(|e|/Fr) ≤ 1–2 px

| Regime | Fr (z01 = 250 mm, falls nicht anders angegeben) | rel. Fr-Fehler für b ≤ 1 px | rel. Fr-Fehler für b ≤ 2 px | Δz01 bei z01 = 250 mm |
|---|---:|---:|---:|---|
| **P05-Vollauflösung** (px 6,5 µm, 11 keV, z02 20 m) | 2,37e-4 (50 mm: 4,7e-5; 400 mm: 3,8e-4) | 0,024 % (50 mm: 0,0047 %; 400 mm: 0,038 %) | 0,1 % | 0,06 mm (1 px), 0,24 mm (2 px); 50 mm: 2 µm; 400 mm: 0,15 mm (1 px) |
| **8×-gebinnt / 256-px-Autofokusstufe** (Fr × 64; = Repo-Datensatz `small`) | 1,52e-2 | 1,5 % | 6 % | 3,7 mm / 15 mm |
| **Prototyp-„small“ = ungebinnte 256 px** (Fr 5e-5…4e-4) | 2,37e-4 | 0,005–0,04 % | 0,02–0,15 % | wie Vollauflösung; 1 % Fehler ⇔ b = 6,5 px (Fr 2,4e-4) … 14,6 px (Fr 4,7e-5) |

P05-Toleranz `z01_tol = 0,1 mm` ⇔ |e| ≈ 0,04 % ⇔ b ≈ 1,3 px bei 250 mm; Default-Suchbereich `z01_confidence = 5 mm` ⇔ |e| ≈ 2 % ⇔ b ≈ 9 px.
Der Downstream-Test bestätigt im 256-px-Regime: |e| = 1 % (b 1,1 px) keine messbare Verschlechterung, 5 % (2,6 px) beginnende, 20 % (5,1 px)
deutliche Artefakte. Für die ML-Pipeline heißt das: neben dem relativen Fr-Fehler (MAE, Median, p95 – schwer-tailig) und Δz01 immer
**b in Pixeln** angeben (`blur_px_*` in `metrics_in_physical_units`), als Downstream-Metriken Gradienten-NRMSE und Vorwärtsmodell-Residuum
relativ zur Rekonstruktion mit wahrem Fr (`*_rel_true`), sowie Laufzeit/Auswertungen pro Hologramm.

## 6. Baseline-Ergebnisse (Suchbereich Fr_true ± 50 %, Preset `p05filter_dsf2_1`, 300/300 Iterationen, `z01_tol` 0,1 mm)

Startwert = Intervallmitte in z01 (nicht der wahre Wert), `z01_confidence` = halbe Intervallbreite. Statistiken aus
`src.baseline.results.summarize` über die kopierten `results.csv` (identisch mit den `summary.json` des Prototyps):

| Methode (Regime, n) | MAE rel. Fr [%] | Median [%] | p95 [%] | Bias [%] | MAE Δz01 [mm] | Median Δz01 [mm] | b Median / p95 [px] | innerhalb 2 % / 5 % | Laufzeit [s/Hologramm] | Auswertungen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `holowizard_find_focus` (p05bin8, 16) | **3,68** | 2,70 | 10,26 | −2,86 | **10,80** | 4,86 | 1,57 / 2,56 | 44 % / 75 % | 190 (Mittel 212,7; 9,9 s/Auswertung) | 21,2 |
| `fallback_tv` = `classical_tv` (p05bin8, 16) | 8,64 | 0,67 | 37,07 | +7,47 | 20,27 | 1,25 | 0,96 / 8,41 | 62 % / 75 % | 0,186 | 33,1 |
| `holowizard_find_focus` (Fr-1e-4-Regime, 12) | **2,90** | 1,05 | 9,59 | −1,68 | **9,09** | 2,06 | 11,07 / 16,99 | 67 % / 75 % | 177 (Mittel 178,6; 8,1 s/Auswertung) | 22,2 |
| `fallback_tv` (Fr-1e-4-Regime, 12) | 28,30 | 26,60 | 47,81 | −1,15 | 61,83 | 47,95 | 34,4 / 80,3 | 0 % / 8 % | 0,205 | 34,8 |

Einzelwerte p05bin8 (signierter rel. Fr-Fehler in %, 4 Hologramme je z01):

| z01 [mm] | Fr_true | `find_focus` | Δz01 [mm] | Auswertungen / t [s] | `fallback_tv` |
|---:|---:|---|---|---|---|
| 50 | 3,006e-3 | +0,72, +1,30, +2,09, +0,72 (MAE 1,21) | +0,36, +0,65, +1,04, +0,36 | 18/151, 21/177, 18/163, 20/183 | +15,51, +43,85, −0,18, −0,76 |
| 100 | 6,028e-3 | −0,52, −0,52, +1,54, +0,27 (MAE 0,71) | −0,51, −0,51, +1,53, +0,26 | 20/157, 18/199, 19/168, 20/187 | −0,07, −0,26, +0,58, −1,55 |
| 250 | 1,518e-2 | −9,80, −4,66, −5,46, −3,31 (MAE 5,81) | −24,2, −11,5, −13,5, −8,2 | 24/222, 22/261, 22/200, 24/193 | −0,53, −0,35, −0,32, −0,49 |
| 400 | 2,448e-2 | −4,84, −7,93, −11,66, −3,62 (MAE 7,01) | −19,0, −31,2, −45,8, −14,2 | 22/172, 24/354, 25/416, 23/199 | +34,81, −2,15, +34,16, −2,72 |

Einzelwerte Fr-1e-4-Regime (3 Hologramme je z01): z01 50 mm (Fr 4,70e-5): +0,13, −0,84, −0,65; 100 mm (9,42e-5): −0,91, −1,88, −0,12;
250 mm (2,37e-4): +0,91, −2,56, −5,66; 400 mm (3,83e-4): −1,18, −13,65, +6,27. Fallback dort: −47…+49 %.

Beobachtungen:

* Bei Fr ≤ 6e-3 schätzt `find_focus` auf ≤ 2,1 % genau (Δz01 ≤ 1,5 mm), bei Fr ≥ 1,5e-2 systematisch zu klein (−3…−12 %). Der Bias ist eine
  Eigenschaft der Zielfunktion (asymmetrische Mulde, Minimum bei −2…−5 % im Scan), nicht des Optimierers – in allen 16 Fällen inneres Minimum.
  Gemessen an b ≤ 1 px (|e| ≤ 1,5–6 % in diesem Regime) reicht das bei kleinem Fr, bei großem nicht.
* 18–25 Auswertungen à 7,8–9,9 s → 2,5–7 min pro Hologramm; die Zahl der Auswertungen hängt kaum von der Intervallbreite ab
  (Nelder-Mead halbiert etwa pro Schritt: log2(250 mm/0,1 mm) ≈ 11 plus Shrink-Schritte).
* Fallback: in 12/16 Fällen |e| ≤ 1,6 %, in 4/16 Nebenminimum der flachen TV-Kurve (+15…+44 %) → als alleinige Baseline ungeeignet, als
  Initialisierung/Plausibilitätscheck (0,19 s) brauchbar. Grund für die Grenzen: die rückpropagierte **Amplitude** eines schwachen Phasenobjekts
  ist im Fokus nicht flach, sondern ein Zwillingsbild bei Fr/2 – ein Minimum existiert nur, wenn das Objektspektrum weit über die erste
  CTF-Nullstelle |u|² = Fr hinausreicht (viele Säume); im Fr-1e-4-Regime (wenige Säume im FOV) versagt es vollständig (Median 27 %).
* Im Fr-1e-4-Regime hat das Residuum der stark regularisierten Autofokus-Rekonstruktion trotz unbrauchbarer Rekonstruktion ein **scharfes
  Minimum nahe Fr_true** (Kernaussage von Dora et al. 2025); die Genauigkeit ist durch `z01_tol` (0,04–0,2 %) und die Mulde begrenzt – aber um
  ein bis zwei Größenordnungen schlechter als b ≤ 1 px dort verlangt (1 % ⇔ 6,5–14,6 px).

## 7. Zielfunktions-Scan: warum `p05filter_dsf2_1` (`reports/baseline_model_based/figures/scan_focus_landscape_p05bin8.{json,png}`)

Loss(e)/Loss(0) für zwei Hologramme aus `p05bin8_z01_250.hdf5` (Fr 1,52e-2):

| Variante | −20 | −10 | −5 | −2 | −1 | 0 | +1 | +2 | +5 | +10 | +20 | s/Eval | Form |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S0 `p05scaled_dsf4_2` #0 | 0,997 | 0,994 | 0,995 | 0,998 | 0,999 | 1 | 1,001 | 1,003 | 1,008 | 1,019 | 1,049 | 1,9 | flach, Minimum −10 % |
| S0 #1 | 0,980 | 0,984 | 0,990 | 0,996 | 0,998 | 1 | 1,002 | 1,005 | 1,013 | 1,027 | 1,060 | 2,4 | monoton fallend zu kleinem Fr |
| S1 dsf 2/1, schwache Filter #0 | 1,090 | 1,155 | 1,039 | 1,027 | 1,001 | 1 | 1,065 | 1,180 | 0,985 | 1,248 | 1,381 | 29,5 | verrauscht, argmin +5 % |
| S1 #1 | 1,256 | 1,069 | 1,101 | 0,934 | 0,952 | 1 | 0,977 | 1,057 | 0,956 | 1,136 | 1,202 | 12,5 | verrauscht, argmin −2 % |
| **S2 `p05filter_dsf2_1`** #0 | 1,149 | 0,983 | 0,985 | 0,996 | 0,998 | 1 | 1,003 | 1,008 | 1,027 | 1,069 | 1,146 | 7,5 | glatte Mulde, Minimum −5…−10 % |
| S2 #1 | 1,067 | 1,002 | 0,970 | 0,982 | 0,991 | 1 | 1,013 | 1,026 | 1,069 | 1,097 | 1,130 | 7,5 | glatte Mulde, Minimum ≈ −5 % |
| S3 dsf 4/2, schwache Filter #0 | 1,405 | 1,035 | 1,019 | 0,988 | 1,020 | 1 | 0,911 | 0,834 | 0,684 | 0,643 | 0,620 | 2,1 | monoton fallend zu großem Fr |
| S3 #1 | 2,151 | 2,073 | 1,729 | 1,372 | 1,122 | 1 | 0,904 | 0,805 | 0,659 | 0,598 | 0,549 | 2,1 | monoton fallend zu großem Fr |

(i) Die 1:1 skalierte P05-Konfiguration (64/128-px-Gitter) hat auf 256-px-Daten kein brauchbares Minimum (zu wenige Säume nach Downsampling).
(ii) Schwache Regularisierung macht die Zielfunktion verrauscht (S1) oder monoton (S3). (iii) Nur „volle 256 px in der letzten Stufe + P05-
Regularisierung“ (S2) ergibt eine glatte Mulde – bei Fr ≥ 1,5e-2 um −5…−10 % verschoben (Bias in Abschnitt 6), bei Fr ≤ 6e-3 innerhalb ±2 %.

## 8. Laufzeit-Benchmark und Hochrechnung auf 2048 px (`reports/baseline_model_based/figures/bench_iteration_time.json`)

| FOV [px] | Rechengitter | ms / ASRM-Iteration | ms / FFT-Paar | RSS [MB] |
|---:|---:|---:|---:|---:|
| 128 | 256² | 5,98 | 0,17 | 726 |
| 256 | 512² | 22,7 | 0,71 | 771 |
| 512 | 1024² | 93,5 | 2,77 | 932 |
| 1024 | 2048² | 560 | 12,0 | 1391 |
| 2048 | 4096² | extrapoliert 2200–3400 | 122 | ≈ 3,3 GB (nicht gemessen) |

Die ASRM-Iteration ist ≈ 30× teurer als ihre zwei FFT-Paare (elementweise Operationen, Gauß-Filter, Constraints dominieren; Skalierung ≈ N²).
Plausibilität: `p05filter_dsf2_1` 300·5,98 + 300·22,7 ms = 8,6 s/Auswertung (gemessen 7,8–9,9 s); `quality_256` ≈ 8,5 s (gemessen 8,2 s).

| Konfiguration | Gitter | Iterationen | s / Auswertung | pro Hologramm (21 Auswertungen) |
|---|---|---|---:|---:|
| Baseline 256 px (`p05filter_dsf2_1`), gemessen | 256² / 512² | 300 / 300 | 7,8–9,9 | 2,5–7 min (Mittel 3,5 min) |
| **P05-Autofokus 2048 px** (`p05_default_2048`: pf 4, dsf 16/4/4) | 512² / 2048² / 2048² | 700 / 300 / 500 | 700·0,0227 + 800·0,560 = **464 s (7,7 min)** | **≈ 2,7 h** CPU |
| Baseline-Variante auf 2048 px (pf 2, dsf 2/1) | 2048² / 4096² | 300 / 300 | 830–1190 s | 4,8–6,9 h CPU, RSS ≈ 3,3 GB |
| Fallback TV 256 px → 2048 px | 512² → 4096² | 33 Auswertungen | 5,7 ms → ≈ 0,4–0,5 s | 0,19 s → ≈ 15 s |

Fazit: Auf CPU ist der modellbasierte Autofokus in P05-Vollauflösung nicht praktikabel (P05-Betrieb: GPU, Gerätewahl automatisch). Für die
Thesis ist die 256-px-Variante als Referenz auf einigen Dutzend Hologrammen machbar; hunderte Hologramme brauchen GPU oder
`--iterations-scale 0.5` (≈ 1,8 min). Der eigentliche Vorteil eines ML-Autofokus liegt in Laufzeit und Modellauswertungen (21 Rekonstruktionen
vs. ein Forward-Pass); empfohlenes Hybrid-Experiment: ML-Schätzung als z01-Startwert mit ±(2–3 σ_ML) als `z01_confidence` für `find_focus`.

## 9. Offene Fragen an die Betreuer

1. **Standard-Autofokus-Parameter an P05**: Sind `pipe/scripts/config/find_focus/defaults.yaml` (700/300/500 Iterationen, pf 4, dsf 16/4/4,
   l2 10j/10j/1j, Nesterov 1,0, `z01_tol` 0,1 mm) und `z01_confidence = 5 mm` die im Betrieb verwendeten Werte? Wird die (z01, a0)-Variante
   (`find_focus_z01_a0.py`) oder die Flatfield-PCA-Korrektur im Autofokus eingesetzt? Welcher a0 ist typisch (Beispiele: 0,98, 1,1)?
2. **Typische z01-Suchbereiche**: ±5 mm (≈ ±2 % Fr bei 250 mm) ist der Default – wie unsicher ist z01 in der Praxis (Motorposition,
   Holder-Höhe, nach Energiewechsel)? Welche z01-Werte kommen an P05 vor (die Thesis nutzt 50–400 mm)? Soll der ML-Autofokus einen breiten
   Bereich (Fr ± 50 %) abdecken oder nur die Verfeinerung in ±5 mm?
3. **Genauigkeitsanforderung**: Ist das Kriterium „b = sqrt(|e|/Fr) ≤ 1–2 px in der Rekonstruktionsauflösung“ (⇔ Δz01 ≈ 0,06–0,25 mm bei
   250 mm Vollauflösung; `z01_tol` 0,1 mm ⇔ 1,3 px) akzeptiert, oder gibt es eine Anforderung aus der Tomographie (Konsistenz über
   Projektionen) bzw. aus der Rekonstruktionsqualität (z. B. ΔNRMSE)? Soll die Referenz-Baseline mit kleinerem `z01_tol` (z. B. 0,01 mm,
   mehr Auswertungen) gerechnet werden?
4. **Auflösung des Autofokus**: Die P05-Zielfunktion wird in den letzten Stufen auf dsf 4 (512² aus 2048 px) ausgewertet. Ist die 256-px-Variante
   (dsf 8 äquivalent) als Referenz akzeptabel, und ist der Bias (−5…−10 % bei Fr ≥ 1,5e-2 im 256-px-Setting) aus dem Betrieb bekannt? Auf
   welcher Hardware (GPU-Typ) läuft der Autofokus an P05 und wie lange dauert er pro Hologramm?
5. **Daten-Konvention**: HoloForge speichert Amplituden |ψ| (+ additives Gauß-Rauschen) statt Intensitäten; reale P05-Daten sind
   flatfield-korrigierte Intensitäten mit Photonenrauschen. Soll die Pipeline auf Amplituden trainiert werden (Konvertierung vor der Core-API,
   abweichendes Rauschmodell) oder sollen die Forge-Daten als Intensitäten interpretiert/erzeugt werden?
6. **Trainingsregime**: Im ungebinnten 256-px-Regime (Fr 5e-5…4e-4) ist die Einzel-Distanz-ASRM-Rekonstruktion schlecht gestellt (NRMSE > 3
   bei korrektem Fr). Das Repo trainiert inzwischen im gebinnten Regime (`data_small`: Fr 3e-3…1,8e-2) – ist das beabsichtigt, oder soll
   zusätzlich das Vollauflösungsregime als reine Fr-Regression abgedeckt werden? Alternativ: Downstream-Test mit mehreren Distanzen oder Support?
7. **Stabilität**: Mit Nesterov-Momentum 1,0 (P05-Default) divergierte die Rekonstruktion bei einigen Forge-Phantomen (Phase → −600 rad,
   Loss ≈ 1, NaN); mit 0,9 nicht. Ist das auf realen Daten bekannt (Phantome mit −5…−7 rad Phasenschub und harten Kanten sind ggf. „strenger“)?
8. **Lizenz/Zitation**: Soll der Baseline-Wrapper HoloWizard (MIT) als Abhängigkeit einbinden (Version 3.0.6 pinnen) und die beiden
   Optics-Express-Paper (DOI 10.1364/OE.544573, 10.1364/OE.514641) zitieren – oder nur die Ergebnisse (CSV) ins Repo?

## 10. Reproduktion mit den CLIs dieses Repos

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/tmp/holo311/bin/python          # Python 3.11 mit holowizard==3.0.6 (siehe requirements.txt)

# Daten: Repo-Datensatz "small" (= Regime p05bin8; store.phantom: true für den Downstream-Test)
$PY -m src.data.generate_data --config configs/data_small.yaml

# Baseline (find_focus + klassischer Fallback), Suchbereich Fr_true ± 50 %, ca. 3,5 min pro Hologramm:
$PY -m src.baseline.model_based_autofocus --data data/processed/small/test.hdf5 --n 16 --search-width 50 \
    --preset p05filter_dsf2_1 --method both --out reports/baseline_model_based/small_test
# schneller (150/150 Iterationen): --iterations-scale 0.5 ; Negativergebnis reproduzieren: --preset p05scaled_dsf4_2

# Downstream-Test: Fehlerkurve (9 Fr-Fehler, ca. 8 s pro Rekonstruktion) ...
$PY -m src.eval.downstream --data data/processed/small/test.hdf5 --n 2 --errors -20 -10 -5 -1 0 1 5 10 20 \
    --preset quality_256 --out reports/downstream/small_test
# ... und mit den Fr-Schätzungen beliebiger Methoden (samples.csv der Baseline-CLI, von src.evaluate oder des Ring-Fits):
$PY -m src.evaluate --config configs/base.yaml --checkpoint runs/<run>/best.pt --split test     # schreibt runs/<run>/eval/samples.csv
$PY -m src.eval.downstream --data data/processed/small/test.hdf5 --n 8 \
    --candidates-csv reports/baseline_model_based/small_test/samples.csv runs/<run>/eval/samples.csv \
    --out reports/downstream/small_test_methods

# Smoke-Läufe (Sekunden; Ergebnisse nicht im Repo):
$PY -m src.baseline.model_based_autofocus --data <forge.hdf5> --n 1 --iterations-scale 0.2 --method both --out /tmp/smoke_mb
$PY -m src.eval.downstream --data <forge.hdf5> --n 1 --errors -5 0 5 --iterations-scale 0.2 --out /tmp/smoke_ds
$PY -m pytest -q                                     # Tests inkl. find_focus-Smoke (Marker "slow")
```

Die Prototyp-Läufe selbst (eigene Datensätze `p05bin8_z01_*`/`small_z01_*`, Skripte `run_autofocus_baseline.py`, `run_downstream_test.py`,
`scan_focus_landscape.py`, `bench_iteration_time.py`) sind in Abschnitt 7 des Prototyp-Berichts dokumentiert; die HoloWizard-Konsolenausgabe
wird in beiden Fällen auf Header-Level reduziert (`configure_holowizard`, Session-Logs im Log-Verzeichnis).

## Anhang: Abweichungen von der Integrationsempfehlung des Prototyps (Abschnitt 5 des Prototyp-Berichts)

* Vorwärtsmodell in `src/utils/fresnel.py` statt `src/physics/fresnel.py`; Metriken in `src/eval/downstream.py` statt `src/eval/metrics.py`
  (kein zusätzliches Paket für drei Funktionen; `src/utils` ist der bestehende Ort für Physik-Helfer).
* `model_based_autofocus` nimmt die gespeicherte **Amplitude** (quadriert intern) statt der Intensität – konsistent mit `classical_autofocus`
  und `reconstruct`, damit alle Funktionen dasselbe `images/hologram`-Array erhalten.
* Ergebnis-CSV im Schema der Aufgabenstellung (`index, source, fr_true, z01_true_mm, fr_est, z01_est_mm, rel_err_fr_pct, dz01_mm, blur_px,
  runtime_s, n_evals, method`, `src/baseline/results.py`) statt des Prototyp-Schemas (`z01_true, z01_est, …`); `read_samples_csv` liest beide.
* `AutofocusResult` hat zusätzlich `fr_history`, `fr_bounds`, `preset`; `downstream_eval` gibt eine Liste von Dicts (kein pandas) zurück.
* Zusätzliches Modul `src/data/forge_samples.py` (Roh-Reader), weil `src.data.dataset.HologramHDF5Dataset` Tensoren/Repräsentationen
  für das Training liefert, die Baselines aber Rohamplitude, Phantom und Geometrie je Sample brauchen.
* Methodenlabel `classical_<metric>` statt `fallback_<metric>`; Fading-Breite des Blackman-Fensters wird für Gitter < 256 px skaliert
  (`default_fading_width_px`), im CLI wird `--border` auf 1/8 der Bildgröße begrenzt (kleine Testgitter).
