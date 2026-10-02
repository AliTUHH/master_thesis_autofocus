# Baseline-Prototyp: Modellbasierter Autofokus (HoloWizard) und Downstream-Test

Arbeitsverzeichnis: `/tmp/baseline_proto/` (alles außerhalb des Repos; `/workspace` wurde nicht verändert).
Interpreter: `/tmp/holo311/bin/python` (Python 3.11, HoloWizard 3.0.6, torch 2.10 CPU, scipy, h5py, scikit-image).
Alle Läufe mit `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `torch.set_num_threads(2)`, 256-px-Hologramme, ≤ 16 Hologramme
pro Lauf, Prozessspeicher (RSS) < 1,4 GB (typisch 0,8 GB). Die Maschine war während aller Läufe durch drei weitere
Agenten ausgelastet (`nproc` = 2 sichtbare Kerne, Load 2–4) – alle Laufzeiten sind daher obere Schranken.
Datum: 2026-10-01.

Dateien:

| Datei | Inhalt |
|---|---|
| `holo_baseline_lib.py` | Gemeinsame Bibliothek: Forge-HDF5-Reader, Fr↔z01, Stufen-Parameter, minimaler Reko-Aufruf, Autofokus-Wrapper, Metriken, Vorwärtsmodell |
| `gen_data.py` | Erzeugt die kleinen Forge-Datensätze (zwei Regime, s. Abschnitt 3.1) aus Kopien von `/tmp/forge_test/smallcfg.json` |
| `run_downstream_test.py` | Downstream-Test: Rekonstruktionsfehler vs. Fr-Fehler (CSV, JSON, Markdown-Tabelle, Abbildung, Bildraster) |
| `regen_image_grid.py` | Erzeugt nur das Bildraster (2-zeilig) für ein Sample neu |
| `run_autofocus_baseline.py` | Baseline-Skript (Paket-Autofokus + klar gekennzeichneter klassischer Fallback), CLI mit argparse |
| `scan_focus_landscape.py` | Scan der Autofokus-Zielfunktion über den Fr-Fehler für vier Reko-Stufen-Varianten |
| `bench_iteration_time.py` | Laufzeit pro ASRM-Iteration und pro FFT vs. Gittergröße (Hochrechnung auf 2048 px) |
| `queue_small_runs.sh` | Warteschlange für die "small"-Läufe (sequentiell, um die CPU nicht zu überlasten) |
| `data/` | Erzeugte Datensätze (`p05bin8_z01_{50,100,250,400}`, `small_z01_{50,100,250,400}`), je mit Forge-Config |
| `downstream_p05bin8/`, `downstream_small/` | Ergebnisse Downstream-Test (`downstream_results.csv`, `downstream_summary.json`, `downstream_table.md`, `downstream_error_vs_fr_error.png`, `downstream_image_grid.png`, `downstream_grid_images.npz`) |
| `baseline_p05bin8/`, `baseline_small/` | Ergebnisse Baseline (`results.csv`, `summary.json`, `histories.json`, `scatter.png`, `objective_curves.png`) |
| `baseline_smoke/`, `downstream_smoke/`, `downstream_p05bin8_v1_nesterov1.0/` | Erste Testläufe (Smoke-Tests; v1 = Lauf mit Nesterov 1.0 und Divergenzen, s. Anhang A) |
| `figures/` | Zusatzabbildungen (Parametersuche `smoke*.png`, `explore_reco_params.png`, Zielfunktions-Scan, Benchmark-JSON) |
| `logs/` | Konsolen-Logs aller Läufe (`*.log`) und HoloWizard-Session-Logs |

## 0. Kurzfassung

* **Autofokus-API gefunden und auf CPU lauffähig**: `holowizard.core.api.functions.find_focus.find_focus.find_focus(reco_params, viewer=None, plotter=None)`
  → Nelder-Mead (scipy) über z01 innerhalb `Measurement.z01_bounds = (z01 ± z01_confidence)`; Kriterium = Daten-Residuum
  (MSE der Amplituden im FOV) der **letzten Iteration einer kompletten mehrstufigen ASRM/PGD-Rekonstruktion** pro Funktionsauswertung.
  Kein hart kodiertes CUDA (Gerät über `nvidia-smi`-Erkennung in `holowizard/core/__init__.py`). Es gibt **keinen** klassischen
  (metrikbasierten) Autofokus in der API; `core/find_focus/focus_loss_metrics.py` enthält zwar sechs Schärfemetriken, die aber nirgends benutzt werden.
* **Fr-Definition in `core` und `forge` ist identisch** (Fr = px²·z01/(λ·z02·(z02−z01)), Einheiten mm/mm/keV, λ = 1,2398/E nm, numerisch verifiziert,
  rel. Abweichung ≤ 4,7e-8 durch Rundung in `forge` auf µm). **Aber** zwei Konventions-Unterschiede sind für die Kopplung zwingend:
  (1) Forge speichert die **Amplitude** |ψ| (+ Rauschen) als "hologram", die Core-API erwartet **Intensitäten** (sie zieht intern `torch.sqrt`) → Forge-Hologramme quadrieren;
  (2) Core dreht die Messung intern um 90° (`torch.rot90`) → Rekonstruktion mit `torch.rot90(x, k=-1)` zurückdrehen, bevor man mit `images/phantoms` vergleicht.
* **Downstream-Test (256 px, "p05bin8"-Regime, 8 Hologramme, 9 Fr-Fehler, 699 s)**: Der GT-NRMSE der Phase ist bei ±20 % Fr-Fehler praktisch
  unverändert (Median 0,75 → 0,70–0,80), weil er bei Einzel-Distanz-Rekonstruktion von der Niederfrequenz-Schlechtgestelltheit dominiert wird.
  Empfindlich sind der **Gradienten-NRMSE** (+5 % bei |e| = 5 %, +10 % bei |e| = 20 %) und die **Modellkonsistenz** (Residuum des Vorwärtsmodells
  beim wahren Fr: ×1,3 bei ±1 %, ×2,1–2,2 bei ±5 %, ×2,8–3,1 bei ±20 %). Physikalische Anforderung: Defokus-Unschärfe
  b = sqrt(|e|/Fr) [Detektorpixel] ≤ 1–2 px → bei P05-Vollauflösung (Fr ≈ 2,4e-4) **|e| ≤ 0,02–0,1 %** (Δz01 ≤ 0,06–0,25 mm bei z01 = 250 mm);
  die P05-Toleranz `z01_tol = 0,1 mm` entspricht b ≈ 1,3 px.
* **Baseline (16 Hologramme, Suchbereich Fr ± 50 %, 256 px, CPU)**: Paket-Autofokus MAE 3,68 % (Median 2,70 %, p95 10,3 %), MAE z01 10,8 mm,
  21 Funktionsauswertungen, 213 s pro Hologramm (9,9 s pro Auswertung). Bei kleinem Fr (3e-3, 6e-3) ≤ 2,1 % Fehler, bei großem Fr
  (1,5e-2, 2,4e-2) systematischer Bias −3…−12 % (Zielfunktion asymmetrisch, s. 4.4). Klassischer Fallback (TV der rückpropagierten Amplitude):
  Median 0,67 %, aber 4/16 grobe Fehlschätzungen (+15…+44 %), 0,19 s pro Hologramm. Im Thesis-Regime "small" (12 Hologramme, Fr 5e-5…4e-4): Paket MAE 2,90 %
  (Median 1,05 %, ≤ 1,9 % bei Fr ≤ 1e-4), 179 s pro Hologramm – obwohl die Rekonstruktion dort scheitert; der klassische Fallback versagt dort vollständig (Median 27 %).
* **Hochrechnung 2048 px**: P05-Standardkonfiguration (700 It. auf 512² + 800 It. auf 2048²-Gitter) ≈ 7,7 min pro Funktionsauswertung,
  ≈ 2,7 h pro Hologramm auf 2 CPU-Threads → der modellbasierte Autofokus ist auf CPU in voller Auflösung nicht praktikabel, auf GPU (P05-Betrieb) schon.

## 1. Autofokus-API in HoloWizard 3.0.6

Paketpfad: `/tmp/holo311/lib/python3.11/site-packages/holowizard/` (Repository: https://github.com/DESY-FS-PETRA/holowizard,
Paper: Dora et al., "Model-based autofocus for near-field phase retrieval", Opt. Express 33(4), 6641–6657 (2025), DOI 10.1364/OE.544573;
ASRM: Dora et al., Opt. Express 32(7), 10801 (2024), DOI 10.1364/OE.514641). Metadaten aus `holowizard-3.0.6.dist-info/METADATA`.

### 1.1 Module, Klassen, Funktionen

| Ebene | Pfad (relativ zu `holowizard/`) | Inhalt |
|---|---|---|
| Öffentliche API | `core/api/functions/find_focus/find_focus.py` | `find_focus(reco_params: RecoParams, viewer: List[Viewer] = None, plotter: List[Plotter] = None) -> (z01: float, z01_values_history: list[float], loss_values_history: list[np.ndarray])`. Zieht **zuerst `torch.sqrt` auf alle `measurements[i].data`** (erwartet also Intensitäten) und ruft `find_focus_z01.find_focus(measurements[0], beam_setup, reco_options, data_dimensions, viewer, plotter)` |
| Öffentliche API mit Flatfield | `core/api/functions/find_focus/find_focus_flatfieldcorrection.py` | `find_focus(flatfield_correction_params: FlatfieldCorrectionParams, reco_params, viewer=None, plotter=None)`: lädt gepickelte PCA-Flatfield-Komponenten (`components_path`), korrigiert `measurements[0].data` mit `correct_flatfield` und ruft die obige Funktion. Diese Variante benutzen `pipe/tasks/find_focus.py` (`FindFocusTask.__call__`) und `livereco/server/find_focus.py` |
| Optimierer | `core/find_focus/find_focus_z01.py` | `find_focus(measurement, beam_setup, options: List[Options], data_dimensions, viewer, plotter)`: `scipy.optimize.minimize(get_loss_reconstruction, x0=measurement.z01, method="Nelder-Mead", bounds=[measurement.z01_bounds], options={"xatol": options[-1].z01_tol, "fatol": 1e6, "initial_simplex": [[z01_lo],[z01_hi]]})`. `get_loss_reconstruction(z01, …)` setzt `measurement.z01 = z01[0]`, ruft `reconstruct_multistage.reconstruct(reco_params, viewer)` → `(result, loss_se_all, fov_size)` und gibt `float(loss_se_all[-1])` zurück. Bereits ausgewertete z01 werden gecached (`check_history`); wird ein z01 ein zweites Mal angefragt (passiert, wenn Nelder-Mead aus den Bounds reflektiert und scipy auf die Grenze clippt), wird `sys.float_info.max` als Strafe zurückgegeben. Historien sind **Modul-Globals** (`z01_values_history`, `loss_values_history`) – nicht threadsicher |
| Varianten (nicht in der API verdrahtet) | `core/find_focus/find_focus_z01_a0.py` | 2-D-Nelder-Mead über (z01, a0) (a0 = Flatfield-Offset/Hintergrund), `adaptive=True`, Bounds a0 ∈ (0,3; 1,8)·10^⌊log10 z01⌋ (skaliert auf die Größenordnung von z01), Start-Simplex aus drei Punkten zwischen Guess und Bounds |
| | `core/find_focus/find_focus_z01_a0_orthogonal_search.py` | alternierende 1-D-Suchen (`get_loss_reconstruction_a0`, `get_loss_reconstruction_z01`) |
| | `core/find_focus/focus_loss_metrics.py` | klassische Schärfemetriken `get_var, get_spec, get_gra, get_lap, get_gog, get_tog` – **werden von keinem Optimierer importiert** (grep über das Paket: keine Referenz) |
| Default-Parameter | `core/api/parameters/default_parameters/find_focus.py` | `get_default_options(a0=0.98, phase_max=0.0, padding_options=None) -> [options_warmup, options_upscale_4, options_upscale_4_lowreg]`: 700 / 300 / 500 Iterationen, update_rate 0,9 / 1,1 / 1,1, l2_weight 0,1j / 0,1j / 0,01j, fwhm (2+0j) / (2+8j) / (2+8j), Nesterov 1,0 mit fwhm (8+8j) / (16+16j) / (16+16j), Padding MIRROR_ALL, padding_factor 4,0, down_sampling_factor 16 / 4 / 4 |
| P05-Pipeline-Defaults | `pipe/scripts/config/find_focus/defaults.yaml` | gleiche drei Stufen, aber l2_weight **10j / 10j / 1,0j**, `z01_tol: 0.1` (mm) je Stufe, `padding_factor: 4`, `down_sampling_factor: 16 / 4 / 4`, `values_min/max: auto` (→ Phase ≤ 0, Absorption ≥ log(a0), gesetzt in `pipe/utils/reco_params.py`) |
| | `pipe/scripts/config/scan.yaml` | `z01_confidence: 5` (mm), `px_size: 0.0065` (mm), `total_size/fov_size: 2048×2048`, `window_type: blackman`, Beispiel `energy: 17.0`, `z01: 190`, `z02: 19826.47` |
| Beispiele | `core/scripts/examples/find_focus/{magnesium_wire,spider_hair}.py` (per `holowizard_core_create_examples <dir>` auch aus dem Paket erzeugbar) | `z01_guess = 470.51`, `z01_confidence = 10.0`, `BeamSetup(energy=11.0, px_size=0.0065, z02=19_661.0)`, `Padding(MIRROR_ALL, padding_factor=4, down_sampling_factor=16, a0=1.1)`, drei Stufen wie oben mit l2 10j/10j/1j, `DataDimensions((2048,2048),(2048,2048),"blackman")`, Aufruf `find_focus(reco_params, viewer=[LossViewer()], plotter=[NelderMeadPlotter()])` |
| Rekonstruktion | `core/api/functions/single_projection/reconstruction.py` | `reconstruct(reco_params, viewer=[]) -> (x_predicted: complex Tensor (gepaddetes Gitter, auf FOV gecroppt), se_losses_all: Tensor)`; ebenfalls `torch.sqrt` auf die Daten |
| | `core/reconstruction/single_projection/reconstruct_multistage.py` | `reconstruct(reco_params, viewer) -> (x, loss_se_all, fov_size)`: Schleife über `reco_options` (Stufen); pro Stufe `process_padding_options` (Downsampling, Padding, Fenster), Propagator `FresnelPropagatorTorch` mit `ConeBeam.get_fr`, dann `reconstruct.py` (PGD mit Nesterov, Gauss-Filter, L2/L1, Wertebereichs-Constraints) |
| Verlustfunktion | `core/reconstruction/gradients/analytical.py::get_gradient` | `loss = Σ_FOV |A_pred − A_meas|² / N_FOV / num_measurements` mit `A_pred = |D_Fr(exp(i·O)·P)|` (Amplitude), `A_meas = sqrt(I)` (nach dem `sqrt` der API, /a0, rot90, Downsampling, Spiegel-Padding, Blackman-Fenster). Nebenbemerkung: die Update-Richtung benutzt `sqrt(A_meas/A_pred)` statt `A_meas/A_pred`, entspricht also (bis auf Gewichtung) dem Gradienten von Σ(√A_pred − √A_meas)², während **berichtet und als Fokuskriterium verwendet** der Amplituden-MSE wird |

### 1.2 Parameter-Objekte (Signaturen aus `core/parameters/*.py`)

```python
BeamSetup(energy, px_size, z02, flat_field=None, probe=None)      # keV, mm, mm   (unit_energy()=("keV",1000), unit_px_size()=unit_z02()=("mm",1e6))
Measurement(z01, data_path="", data=None, z01_confidence=5.0)      # z01 in mm; z01_bounds -> (z01 - z01_confidence, z01 + z01_confidence)
Regularization(iterations=0, update_rate=0, gaussian_filter_fwhm=0j,
               values_min=-sys.float_info.max + 0j, values_max=0 + sys.float_info.max*1j,   # Phase <= 0, Absorption >= 0
               l2_weight=0j, l1_weight=0j)                         # Realteil = Phase, Imaginärteil = Absorption
Padding(padding_mode=PaddingMode.CONSTANT, padding_factor=2, down_sampling_factor=1, cutting_band=0, a0=1.0, prototype_field=None)
Options(update_blocks=0, regularization_object=Regularization(), regularization_probe=Regularization(),
        nesterov_object=Regularization(), z01_tol=0.1, padding=Padding(), verbose_interval=100, prototype_field=None)
DataDimensions(total_size, fov_size, window_type, fading_width=[(80, 80), (80, 80)], window=None)
RecoParams(beam_setup, measurements: List[Measurement], reco_options: List[Options], data_dimensions, output_path: str,
           session_params=None, initial_guess=None)
```

Wichtige Verarbeitungsdetails (aus `core/preprocessing/process_image.py`, `process_data_dimensions.py`, `process_padding_options.py`):
`process_beam_setup` multipliziert `px_size` mit `down_sampling_factor` (→ Fr skaliert mit dsf²); die Messung wird durch `a0` geteilt,
um 90° gedreht (`torch.rot90`), bilinear mit Antialiasing heruntergesampelt, auf das 4-fache FOV gespiegelt (MIRROR_ALL: `ReflectionPad2d(fov−1)`),
dann auf `total_size·padding_factor/dsf` zentral gecroppt und mit einem Blackman-Fenster (`fading_width` 80 px / dsf) an die Konstante a0
angeglichen. Das Rechengitter einer Stufe ist also `total_size·padding_factor/dsf` (bei 256 px, pf 2: 512² bzw. 256²/128² für dsf 2/4).

### 1.3 Kriterium, Optimierer und Kopplung an ASRM/PGD

* **Kriterium**: `loss_se_all[-1]` – das Daten-Residuum der letzten Iteration der letzten Stufe, berechnet auf dem (heruntergesampelten, gepaddeten) Gitter dieser Stufe im FOV-Bereich.
  Es ist **kein** Bildschärfe-Kriterium: der Autofokus minimiert, wie gut eine vollständige regularisierte Rekonstruktion die Messung erklärt.
* **Optimierer**: scipy Nelder-Mead in 1-D (z01), Start-Simplex = die beiden Intervallgrenzen, Abbruch bei Simplexbreite ≤ `z01_tol` (0,1 mm) – `fatol = 1e6` schaltet das Funktionswert-Kriterium praktisch ab.
  Jede Funktionsauswertung = eine komplette mehrstufige Rekonstruktion (P05: 700 + 300 + 500 Iterationen). In unseren Läufen 16–25 Auswertungen.
* **Kopplung**: `find_focus_z01.get_loss_reconstruction` baut aus (`measurement`, `beam_setup`, `options`, `data_dimensions`) ein `RecoParams`, setzt `measurement.z01`, und ruft `reconstruct_multistage.reconstruct` – also exakt denselben Code wie die normale Rekonstruktion. Fr wird intern in jeder Stufe aus (z01, z02, px_size·dsf, energy) über `ConeBeam.get_fr` berechnet; die Suche läuft **in z01 (mm)**, nicht in Fr.
  Das Ergebnis (z01) wird in `pipe` in die Scan-Konfiguration übernommen und für die eigentliche Rekonstruktion (`pipe/scripts/config/reconstruction/defaults_thick.yaml`) verwendet.
* **Mehrere Distanzen**: `RecoParams.measurements` ist eine Liste (Multi-Distanz-Rekonstruktion wird vom Propagator unterstützt), aber der Autofokus optimiert nur `measurements[0].z01`.

### 1.4 GPU-Annahmen

`holowizard/core/__init__.py`: `torch_running_device_name = "cpu"`; nur wenn `subprocess.check_output("nvidia-smi")` erfolgreich ist, wird `cuda:0`
gesetzt und `cupy` importiert. Alle Tensoren werden auf `holowizard.core.torch_running_device` gelegt; der Propagator (`core/models/fresnel_propagator_torch.py`)
nutzt `torch.fft`. Kein hart kodiertes `.cuda()`. Auf dieser Maschine ohne GPU läuft alles auf CPU – getestet (Rekonstruktion, Autofokus, Logging).
Nebenwirkung beim Import: `tempfile.tempdir = "/tmp/"`.

### 1.5 Fr-Definition und Einheiten: `core` vs. `forge`

| Aspekt | `holowizard.core` (`core/models/cone_beam.py::ConeBeam.get_fr`) | `holowizard.forge` (`forge/utils/utilities.py::calc_Fr`) | Unterschied |
|---|---|---|---|
| Formel | z12 = z02 − z01; M = z02/z01; dx_eff = px/M; z_eff = z12/M; λ = 1,2398/E nm; **Fr = dx_eff²/(λ·z_eff) = px²·z01/(λ·z02·(z02−z01))** | λ = 1,2398/E; M = z02/z01; **Fr = px²/(λ·(z02−z01)·M)** | algebraisch identisch |
| Einheiten | z01, z02, px in mm (×1e6 → nm), E in keV | Config: z01, z02, detector_px_size in mm, energy in keV (`convert_z01 = round(z01, 3)·1e6`, `convert_energy` = Identität; die Docstrings "cm/m/eV" sind veraltet) | identisch; forge rundet z01/z02 auf 3 Dezimalstellen (µm) |
| Numerisch (eigener Test, identische Eingaben) | `ConeBeam.get_fr`: Fr(z01=250, z02=20000, E=11, px=0,052) = 1,518415587651e-2; Fr(50, 20000, 11, 0,0065) = 4,697479300768e-5 | `calc_Fr`: 1,518415587651e-2; 4,697479300768e-5 | rel. Abw. 0 bzw. 1,4e-16; bei z02 = 19826,473541 mm (P05-`scan.yaml`): 2,414503496e-4 vs. 2,414503384e-4, rel. 4,66e-8 durch forges Rundung von z02 auf µm |
| Umkehrung | `ConeBeam.get_z01(setup, fr) = round(Fr·λ·z02²/(px² + Fr·λ·z02)) nm` (rundet auf nm) | – (nicht vorhanden) | Round-Trip exakt bis auf nm-Rundung; eigene Implementierung `holo_baseline_lib.z01_from_fresnel` |
| Downsampling | `px_size·down_sampling_factor` → Fr·dsf² | `NFHConstantDistSetup`: `detector_px_size·downsample_factor`, `detector_size/downsample_factor` | gleiche Konvention (Fr pro effektivem Pixel) |
| Padding | FFT auf gepaddetem Gitter mit `torch.fft.fftfreq` (Zyklen/Pixel) → Fr pro Pixel unverändert | `padding_factor` 2, Zero-Padding von O, gleiche fftfreq-Konvention | identisch |
| Propagator | `exp(−iπ/Fr·(ξ²+η²))`, Rückpropagation mit `−Fr` | `exp(−iπ/Fr·(ξ²+η²))` | identisch |
| Messgröße | **Intensität** I; API zieht `sqrt` → Amplituden-Fit | **Amplitude** `|ψ_det|` (+ Gauß-Rauschen σ = 0,05, numerisch verifiziert: Residuum-Std 0,0500) wird als `images/hologram` gespeichert, `store_gt_hologram=false` | **Forge-Hologramm quadrieren**, bevor es an die Core-API geht (`ForgeSample.hologram_intensity`); Rauschmodell ist dann nicht mehr additiv-gaußsch |
| Objektkonvention | Objekt O komplex, Welle `exp(i·O)·P`, Realteil = Phasenschub ≤ 0, Imaginärteil = Absorption ≥ 0 (Constraints `values_min/max`) | `images/phantoms` complex64, Realteil = Phasenschub (≤ 0, hier bis ≈ −5…−7 rad), Imaginärteil = Absorption (≥ 0) | identisch → `images/phantoms.real` ist die GT-Phase |
| Orientierung | `process_image`: `torch.rot90(image)` vor der Rekonstruktion; Ergebnis wird **nicht** zurückgedreht | – | Rekonstruktion mit `torch.rot90(result, k=-1)` zurückdrehen (numerisch verifiziert über GT-Korrelation) |
| Mehrere Distanzen | Fr-Liste (eine pro Messung) | eine Distanz | – |
| Metadaten | – | `metadata/setup/{Fr, z01, z02, energy, detector_px_size, detector_size, downsample_factor, padding_factor, probe_size}` pro Sample, **alle float32** | `detector_px_size` ist bereits der **effektive** (gebinnte) Pixel (0,052 statt 0,0065 mm), `detector_size` die gebinnte Größe; das gespeicherte Fr hat float32-Genauigkeit (p05bin8_z01_250: 1,5184155665e-2 statt 1,5184155877e-2, rel. 1,4e-8) – für Fehlermetriken irrelevant |

## 2. Minimaler Rekonstruktionsaufruf (holowizard.core, CPU)

```python
import torch, numpy as np, h5py
torch.set_num_threads(2)
from holowizard.core.logging.logger import Logger
from holowizard.core.api.parameters import BeamSetup, Measurement, Padding, Options, Regularization, DataDimensions, RecoParams
from holowizard.core.api.functions.single_projection.reconstruction import reconstruct
from holowizard.core.api.functions.find_focus.find_focus import find_focus

Logger.current_log_level = Logger.level_num_header            # Pflicht: custom Log-Level + Geometrie-Datei brauchen eine konfigurierte Session
Logger.configure(working_dir="/tmp/baseline_proto/logs", session_name="demo")

with h5py.File("data/p05bin8_z01_250/p05bin8_z01_250.hdf5") as f:   # Forge-HDF5
    amp = f["images/hologram"][0].astype(np.float32)                 # ACHTUNG: Amplitude |psi| (+ Rauschen)
    gt_phase = f["images/phantoms"][0].real                           # GT: Phasenschub (rad, <= 0)
    s = f["metadata/setup"]; z01, z02, E, px = (float(s[k][0]) for k in ("z01", "z02", "energy", "detector_px_size"))

def stage(iters, lr, l2_abs, fwhm_obj, fwhm_nesterov, dsf, nesterov=0.9):
    return Options(regularization_object=Regularization(iterations=iters, update_rate=lr, l2_weight=l2_abs * 1j,
                                                        gaussian_filter_fwhm=fwhm_obj),
                   nesterov_object=Regularization(update_rate=nesterov, gaussian_filter_fwhm=fwhm_nesterov),
                   verbose_interval=100000, z01_tol=0.1,
                   padding=Padding(padding_mode=Padding.PaddingMode.MIRROR_ALL, padding_factor=2,
                                   down_sampling_factor=dsf, cutting_band=0, a0=1.0))

stages = [stage(300, 0.9, 10.0, 2 + 0j, 4 + 4j, 4),        # Warmup auf 64 px  (holo_baseline_lib.reco_stages_quality)
          stage(200, 1.1, 1.0, 2 + 8j, 8 + 8j, 2),         # 128 px
          stage(300, 1.1, 0.1, 0 + 2j, 2 + 2j, 1)]         # 256 px, schwache Filter (P05-Filterbreiten sind für 2048 px ausgelegt)

reco_params = RecoParams(beam_setup=BeamSetup(energy=E, px_size=px, z02=z02),
                         measurements=[Measurement(data=amp**2, z01=z01, z01_confidence=5.0)],   # Intensität übergeben!
                         reco_options=stages,
                         data_dimensions=DataDimensions(total_size=(256, 256), fov_size=(256, 256), window_type="blackman"),
                         output_path="")
result, losses = reconstruct(reco_params, viewer=[])      # complex Tensor (FOV), losses pro Iteration
result = torch.rot90(result, k=-1)                        # Core dreht die Messung um 90 deg -> zurückdrehen
phase, absorption = result.real.numpy(), result.imag.numpy()

# Autofokus (Suchbereich z01 +- z01_confidence, Kriterium = losses[-1] der Stufenfolge in reco_options):
# z01_found, z01_hist, loss_hist = find_focus(reco_params, viewer=None, plotter=None)
```

Entspricht `holo_baseline_lib.reconstruct_sample`/`build_reco_params` (Datei `holo_baseline_lib.py`, Z. 130–234).
Laufzeit dieser Rekonstruktion (300/200/300 Iterationen, dsf 4/2/1, Gitter 128²/256²/512²): **≈ 8 s** auf 2 CPU-Threads
(Median über die 72 Rekonstruktionen des Downstream-Tests 8,1 s, Mittel 9,7 s, Min 7,6 s, Max 33 s; die erste Rekonstruktion eines Prozesses dauert 15–20 s wegen Initialisierung und CPU-Konkurrenz; Benchmark Abschnitt 4.5).

**Bedeutung der GT** (`images/phantoms`): die komplexe Objektfunktion O = φ + iμ auf dem 256²-Gitter (Forge: `Phantom`, Material Mg, Dicke 30, 1–3 Formen),
nicht ein rauschfreies Hologramm (`store_gt_hologram=false`). Verglichen wird `Re(O)` (Phasenschub) mit der rekonstruierten Phase, Rand 16 px abgeschnitten, Metriken:
NRMSE = ‖φ̂−φ‖/‖φ‖, offsetfreier NRMSE (Mittelwert abgezogen, da die Phase nur bis auf eine Konstante bestimmt ist), NRMSE der Bildgradienten (`np.gradient`),
Pearson-Korrelation, SSIM (`skimage.metrics.structural_similarity`, `data_range` = GT-Spanne). Zusätzlich das **Daten-Residuum beim wahren Fr**:
RMSE zwischen `|D_Fr_true(exp(i·Ô))|` (Forge-identisches Vorwärtsmodell, Zero-Padding 2×) und dem gemessenen Amplituden-Hologramm (Rauschniveau 0,05 = Untergrenze).

Ergebnis bei korrektem Fr (p05bin8, 8 Hologramme): NRMSE 0,40–0,89 (Median 0,750), offsetfrei 0,42–0,92, Pearson 0,40–0,92 (Median 0,641), SSIM 0,02–0,26,
Residuum 0,043–0,135 (Rauschen 0,05). Die Rekonstruktion trifft Kanten und Form, aber der niederfrequente Phasenverlauf (Plateaus bis −5 rad) wird bei Einzel-Distanz-Rekonstruktion eines 256-px-FOV nur teilweise erfasst (Bildraster `downstream_p05bin8/downstream_image_grid.png`).
Bei kleinem Fr ist das ausgeprägter (Fr 1,5e-2/2,4e-2: NRMSE 0,78–0,89; Fr 3e-3/6e-3: 0,40–0,72).

## 3. Downstream-Test: Rekonstruktionsqualität vs. Fr-Fehler

### 3.1 Datensätze (zwei Regime, erzeugt mit `gen_data.py` aus Kopien von `/tmp/forge_test/smallcfg.json`)

| Regime | Forge-Setup | Fr bei z01 = 50 / 100 / 250 / 400 mm | Samples | Zweck |
|---|---|---|---|---|
| `p05bin8` | `detector_size 2048`, `downsample_factor 8` → 256 px, px_eff 0,052 mm, `radius_range [64,512]`, `size_range [64,1024]` (ShapeSampler teilt durch dsf), z02 20000 mm, 11 keV, pf 2, Rauschen σ 0,05 | 3,006e-3 / 6,028e-3 / 1,518e-2 / 2,448e-2 | 4 × 4 | emuliert 2048-px-P05-Daten nach 8×-Binning – die Auflösung, auf der die P05-Autofokus-Zielfunktion (dsf 16/4) tatsächlich ausgewertet wird; Einzel-Distanz-Rekonstruktion ist hier gut gestellt |
| `small` | Thesis-`smallcfg`: 256 px, dsf 1, px 0,0065 mm, `radius_range [8,64]`, `size_range [8,128]` | 4,697e-5 / 9,419e-5 / 2,373e-4 / 3,825e-4 | 4 × 3 | Thesis-Trainingsregime; erste CTF-Maximum-Periode sqrt(2/Fr) = 72…206 px ≈ FOV → Einzel-Distanz-Rekonstruktion schlecht gestellt |

Pro Sample und Fr-Fehler e ∈ {−20, −10, −5, −1, 0, +1, +5, +10, +20} % wird mit Fr' = Fr_true·(1+e/100) rekonstruiert; Fr' wird über z01' = z01(Fr')
(Umkehrformel, `holo_baseline_lib.z01_from_fresnel`) in die Core-API eingespeist, weil die API Fr aus z01 berechnet (Round-Trip-Fehler ≈ 3e-16 relativ, Maschinengenauigkeit).
Stufen: `reco_stages_quality` (Abschnitt 2), Nesterov 0,9 (siehe Anhang A).

### 3.2 Ergebnis `p05bin8` (8 Hologramme: je 2 aus z01 = 50/100/250/400 mm; 72 Rekonstruktionen, 699 s; Mediane über die 8 Hologramme)

| Fr-Fehler e [%] | b = sqrt(abs(e)/Fr) [px] (Mittel) | divergiert | NRMSE Phase | NRMSE offsetfrei | NRMSE Gradient | Pearson | SSIM | Core-Loss (Kriterium) | Residuum bei Fr_true | NRMSE rel. e=0 (gepaart) | NRMSE-Grad rel. e=0 (gepaart) | Residuum rel. e=0 (gepaart) |
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

Abbildungen: `downstream_p05bin8/downstream_error_vs_fr_error.png` (NRMSE, Gradienten-NRMSE, Core-Loss, Gradienten-NRMSE vs. b),
`downstream_p05bin8/downstream_image_grid.png` (Hologramm, GT, 9 Rekonstruktionen von `p05bin8_z01_250.hdf5 #0`). Rohdaten: `downstream_results.csv` (eine Zeile pro Sample × e).

### 3.3 Ergebnis `small` (Thesis-Regime; 4 Hologramme aus z01 = 100/250 mm; 36 Rekonstruktionen, 678 s)

| Fr-Fehler e [%] | b [px] (Mittel) | divergiert | NRMSE Phase | NRMSE offsetfrei | NRMSE Gradient | Pearson | SSIM | Core-Loss | Residuum bei Fr_true | NRMSE rel. | NRMSE-Grad rel. | Residuum rel. |
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

Hier scheitert bereits die Rekonstruktion mit **korrektem** Fr (NRMSE 3,4, Pearson 0,14, Core-Loss 3,6e-2 statt 2e-4, Phase bis −14 rad statt −5):
bei Fr ≈ 1e-4…2,4e-4 und 256 px liegt das erste Maximum der Phasen-CTF bei Perioden von sqrt(2/Fr) ≈ 90–150 px, d. h. das Hologramm enthält praktisch
keine Niederfrequenz-Information über das Objekt und die Einzel-Distanz-ASRM-Rekonstruktion ohne Support ist in diesem Regime schlecht gestellt. Der Downstream-Test
liefert in diesem Regime deshalb keine belastbare Aussage über den Einfluss des Fr-Fehlers (die Metriken schwanken unsystematisch, nur Residuum und Gradienten-NRMSE steigen tendenziell für |e| ≥ 5 %).

### 3.4 Interpretation und Anforderung an den Autofokus

1. **Ein Fr-Fehler wirkt wie eine Rest-Defokussierung**: Das Produkt aus Rückpropagation mit Fr' und Vorwärtsmodell mit Fr ist ein Fresnel-Propagator mit effektiver
   Fresnel-Zahl Fr_eff = Fr·Fr'/|Fr' − Fr| ≈ Fr/|e|. Die zugehörige Unschärfe (Radius der ersten Fresnel-Zone) in Detektorpixeln ist **b = sqrt(|e|/Fr)**.
   Für 1 px Unschärfe gilt |e| = Fr; die tolerierbare relative Fr-Genauigkeit ist also **proportional zu Fr** und damit abhängig von Pixelgröße (Binning) und Geometrie.
2. **Im p05bin8-Regime** (b = 1,1 px bei |e| = 1 %, 2,6 px bei 5 %, 5,1 px bei 20 %) ist der GT-NRMSE der Phase innerhalb ±20 % praktisch konstant (gepaart 0,97–1,02) – die
   Phasenrekonstruktion aus einer Distanz ist bei 256 px von Regularisierung und fehlender Niederfrequenz-Information dominiert, nicht vom Fr-Fehler. Die hochfrequente
   Qualität (Gradienten-NRMSE) verschlechtert sich ab b ≈ 2,6 px messbar (+5 % bei +5 %, +8 % bei +10 %, +10 % bei +20 %; für negative e kompensiert die stärkere Glättung teilweise).
   Die Modellkonsistenz (Residuum beim wahren Fr) reagiert am empfindlichsten: ×1,3 bei ±1 %, ×2,1–2,2 bei ±5 %, ×2,8–3,1 bei ±20 % – ein Fr-Fehler produziert Artefakte (Ringing an Kanten, sichtbar im Bildraster), die das Hologramm **nicht** erklären.
3. **Das Core-Loss (Autofokus-Kriterium) ist mit den Quality-Stufen NICHT beim wahren Fr minimal**, sondern fällt monoton zu größerem Fr (2,13e-4 bei 0 → 1,39e-4 bei +20 %): mit schwacher
   Absorptions-Regularisierung kann der Absorptionskanal die Daten für zu große Fr "wegfitten". Daraus folgt für die Baseline: die Autofokus-Stufen brauchen die starke
   Absorptions-Regularisierung der P05-Defaults (l2 1j–10j, fwhm 8j), sonst hat die Zielfunktion kein Minimum (Abschnitt 4.4).
4. **Anforderung** ("Autofokus muss besser als X % sein, damit …"):
   * damit die Defokus-Unschärfe ≤ 1 px bleibt: |e| ≤ Fr → **P05-Vollauflösung** (px 6,5 µm, 11 keV, z02 20 m): |e| ≤ 0,024 % bei z01 = 250 mm (Fr 2,37e-4; Δz01 ≈ 0,06 mm), 0,038 % bei 400 mm (Δz01 ≈ 0,15 mm), 0,0047 % bei 50 mm (Δz01 ≈ 2 µm);
     für ≤ 2 px das Vierfache (0,1 %, Δz01 ≈ 0,24 mm bei 250 mm). Die P05-Toleranz `z01_tol = 0,1 mm` entspricht bei 250 mm |e| ≈ 0,04 % ⇔ b ≈ 1,3 px; der Default-Suchbereich `z01_confidence = 5 mm` entspricht |e| ≈ 2 % ⇔ b ≈ 9 px.
   * **8×-gebinnt / 256-px-Autofokusstufe** (Fr ×64): |e| ≤ 1,5 % (b ≤ 1 px) bzw. ≤ 6 % (b ≤ 2 px) bei z01 = 250 mm. Der Downstream-Test bestätigt: bei |e| = 1 % keine messbare Verschlechterung, bei 5 % beginnende, bei 20 % deutliche Artefakte (Gradienten-NRMSE +10 %, Residuum ×3).
   * **Thesis-Trainingsregime** (256 px, px 6,5 µm, Fr 5e-5…4e-4): 1 px Unschärfe entspricht nur |e| = 0,005–0,04 % – ein ML-Autofokus mit ~1 % Genauigkeit liefert hier b ≈ 5–15 px und muss durch eine modellbasierte Verfeinerung (Abschnitt 5) ergänzt werden, falls die Rekonstruktion in voller Auflösung das Ziel ist.
   * Metrik-Empfehlung für die ML-Pipeline: neben dem relativen Fr-Fehler immer **b = sqrt(|e|/Fr)** in Pixeln angeben (geometrie- und binning-unabhängig interpretierbar) sowie den Gradienten-NRMSE und das Vorwärtsmodell-Residuum relativ zu e = 0 als Downstream-Metriken.

## 4. Baseline: HoloWizard-Autofokus auf Forge-Daten (256 px, CPU)

### 4.1 Setup (`run_autofocus_baseline.py`)

* Eingabe: Forge-HDF5 (Glob), `--n-samples` pro Datei. Suchbereich: Fr_true·(1 ∓ `--search-width`/100), Default **±50 %** (deutlich breiter als der P05-Default ±5 mm ≈ ±2 %) → in z01 umgerechnet;
  `Measurement(z01 = Intervallmitte in z01, z01_confidence = halbe Intervallbreite)` – der Startwert ist also **nicht** der wahre Wert (z. B. 249,23 statt 250 mm; Nelder-Mead startet ohnehin mit dem Simplex aus den beiden Grenzen).
* Autofokus-Stufen (`holo_baseline_lib.reco_stages_focus`, Variante `p05filter_dsf2_1`, Default): 2 Stufen à 300 Iterationen, (dsf 2, lr 0,9, l2 1j, fwhm 2+8j, Nesterov 1,0/8+8j) und (dsf 1, lr 1,1, l2 1j, fwhm 1+8j, Nesterov 1,0/4+4j), pf 2, MIRROR_ALL, a0 = 1, `z01_tol` 0,1 mm – die P05-Filter/Regularisierung, aber auf 256 px skaliert (Gitter 256² und 512²). Begründung in 4.4.
* Ausgabe pro Sample: wahres Fr, geschätztes Fr, relativer Fr-Fehler [%], z01-Fehler [mm] (Umkehrformel mit z02/E/px aus den Metadaten), Laufzeit [s], Zahl der Funktionsauswertungen (`len(loss_values_history)`), plus die Zielfunktions-Historie (`histories.json`). `summary.json`: MAE/Median/p95/Max des |rel. Fr-Fehlers|, MAE/Median |Δz01|, mittlere/mediane/minimale Laufzeit, mittlere Auswertungen, Laufzeit pro Auswertung. `scatter.png`: Fr_est vs. Fr_true, rel. Fehler vs. Fr_true, Laufzeit vs. Fr_true. `objective_curves.png`: Zielfunktionen der ersten vier Samples.
* **Fallback (klar gekennzeichnet, kein HoloWizard-Modell)**, Klasse `ClassicalFocus`: Spiegel-Padding 2×, einmalige FFT, Rückpropagation `D_{-Fr}` der Amplitude, Fokuskriterium **Total Variation** der rückpropagierten Amplitude (minimal im Fokus; entspricht `focus_loss_metrics.get_gra`; alternativ `var`, `lap`), 21-Punkte-Log-Gitter über [Fr_lo, Fr_hi] + Brent-Verfeinerung (`scipy.optimize.minimize_scalar(method="bounded")` in log Fr, `xatol` 1e-4). Gleiche Ausgabeformate (`method = fallback_tv`).

### 4.2 Ergebnis `p05bin8` (16 Hologramme = 4 je z01 ∈ {50, 100, 250, 400} mm; `baseline_p05bin8/`)

Zusammenfassung (`summary.json`):

| Methode | n | MAE rel. Fr | Median | p95 | Max | MAE Δz01 | Median Δz01 | Laufzeit Mittel / Median / Min | Auswertungen (Mittel) | s pro Auswertung (Mittel / Min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `holowizard_find_focus` | 16 | **3,68 %** | 2,70 % | 10,26 % | 11,66 % | **10,80 mm** | 4,86 mm | **212,7 s** / 190,0 s / 151,5 s | 21,25 | 9,89 / 7,80 |
| `fallback_tv` | 16 | 8,64 % | 0,67 % | 37,07 % | 43,85 % | 20,27 mm | 1,25 mm | 0,188 s / 0,186 s / 0,160 s | 33,06 | 0,0057 / 0,0055 |

Einzelwerte (signierter rel. Fr-Fehler in %, Reihenfolge Sample 0–3 je Datei; Paket: Auswertungen / Laufzeit s):

| z01 [mm] | Fr_true | Paket: rel. Fehler [%] | Paket: Δz01 [mm] | Paket: Evals / t [s] | Fallback TV: rel. Fehler [%] |
|---:|---:|---|---|---|---|
| 50 | 3,006e-3 | +0,72, +1,30, +2,09, +0,72 (MAE 1,21) | +0,36, +0,65, +1,04, +0,36 | 18/151, 21/177, 18/163, 20/183 | +15,51, +43,85, −0,18, −0,76 |
| 100 | 6,028e-3 | −0,52, −0,52, +1,54, +0,27 (MAE 0,71) | −0,51, −0,51, +1,53, +0,26 | 20/157, 18/199, 19/168, 20/187 | −0,07, −0,26, +0,58, −1,55 |
| 250 | 1,518e-2 | −9,80, −4,66, −5,46, −3,31 (MAE 5,81) | −24,2, −11,5, −13,5, −8,2 | 24/222, 22/261, 22/200, 24/193 | −0,53, −0,35, −0,32, −0,49 |
| 400 | 2,448e-2 | −4,84, −7,93, −11,66, −3,62 (MAE 7,01) | −19,0, −31,2, −45,8, −14,2 | 22/172, 24/354, 25/416, 23/199 | +34,81, −2,15, +34,16, −2,72 |

Beobachtungen:
* Bei Fr ≤ 6e-3 schätzt der Paket-Autofokus auf **≤ 2,1 %** genau (Δz01 ≤ 1,5 mm), bei Fr ≥ 1,5e-2 systematisch **zu klein** (−3…−12 %, Δz01 −8…−46 mm). Der Bias ist eine Eigenschaft der Zielfunktion
   (asymmetrische Mulde, Minimum bei −2…−5 % im Scan, Abschnitt 4.4), nicht des Optimierers – Nelder-Mead findet in allen 16 Fällen ein inneres Minimum (keine Konvergenz an eine Grenze).
   Gemessen an der Anforderung aus 3.4 (b ≤ 1 px ⇔ |e| ≤ 1,5–6 % in diesem Regime) ist das Ergebnis bei kleinem Fr ausreichend, bei großem Fr nicht.
* **Laufzeit**: 18–25 Auswertungen à 7,8–9,9 s → 2,5–7 min pro Hologramm bei 256 px auf 2 CPU-Threads (unter Last; s. Benchmark 4.5). Die Zahl der Auswertungen hängt kaum von der Breite des Intervalls ab (Nelder-Mead halbiert ~ pro Schritt: log2(250 mm / 0,1 mm) ≈ 11 Reflexionen/Kontraktionen plus Shrink-Schritte).
* **Fallback**: in 12/16 Fällen sehr genau (|e| ≤ 1,6 %; Fr 6e-3 und 1,5e-2: ≤ 0,6 %), in 4/16 Fällen grobe Fehlschätzung (Nebenminimum der TV-Kurve bei +15…+44 %, je zwei bei Fr 3e-3 und 2,4e-2; Kurven in `objective_curves.png`, rechtes Panel: die TV-Mulde ist flach, Nebenminima liegen nur wenige Promille höher). Als alleinige Baseline ungeeignet, als Initialisierung/Plausibilitätscheck (0,19 s) brauchbar.
* Speicher: RSS ≈ 0,80 GB (Paket-Autofokus inkl. Fallback im selben Prozess).

### 4.3 Ergebnis `small` (Thesis-Regime, 12 Hologramme = 3 je z01 ∈ {50, 100, 250, 400} mm; `baseline_small/`)

Gleiche Einstellungen wie 4.2 (Suchbereich Fr ± 50 %, Variante `p05filter_dsf2_1`, 300/300 Iterationen). Lauf: 12 × (178,6 + 0,2) s ≈ 36 min.

| Methode | n | MAE rel. Fr | Median | p95 | Max | MAE Δz01 | Median Δz01 | Laufzeit Mittel / Median / Min | Auswertungen (Mittel) | s pro Auswertung (Mittel / Min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `holowizard_find_focus` | 12 | **2,90 %** | 1,05 % | 9,59 % | 13,65 % | **9,09 mm** | 2,06 mm | **178,6 s** / 176,6 s / 146,1 s | 22,17 | 8,07 / 7,66 |
| `fallback_tv` | 12 | 28,30 % | 26,60 % | 47,81 % | 48,82 % | 61,8 mm | 47,9 mm | 0,202 s / 0,205 s / 0,184 s | 34,83 | 0,0058 / 0,0057 |

| z01 [mm] | Fr_true | Paket: rel. Fehler [%] | Paket: Δz01 [mm] | Paket: Evals / t [s] | Fallback TV: rel. Fehler [%] |
|---:|---:|---|---|---|---|
| 50 | 4,697e-5 | +0,13, −0,84, −0,65 (MAE 0,54) | +0,07, −0,42, −0,32 | 20/153, 22/169, 19/146 | +39,85, −15,62, −4,79 |
| 100 | 9,419e-5 | −0,91, −1,88, −0,12 (MAE 0,97) | −0,90, −1,87, −0,12 | 19/165, 24/191, 23/181 | −11,12, −25,64, −46,98 |
| 250 | 2,373e-4 | +0,91, −2,56, −5,66 (MAE 3,04) | +2,24, −6,32, −13,97 | 22/221, 24/195, 24/186 | +48,82, −25,15, −19,81 |
| 400 | 3,825e-4 | −1,18, −13,65, +6,27 (MAE 7,03) | −4,64, −53,65, +24,55 | 25/194, 22/170, 22/172 | −27,55, +45,37, +28,87 |

Beobachtungen:
* Obwohl die **Rekonstruktion** in diesem Regime scheitert (Abschnitt 3.3), hat das Daten-Residuum der (stark regularisierten) Autofokus-Rekonstruktion ein **scharfes Minimum nahe Fr_true**
  (`baseline_small/objective_curves.png`, linkes Panel: Mulde deutlich schmaler als im p05bin8-Regime, weil die Fresnel-Säume bei kleinem Fr stark von Fr abhängen). Das entspricht der Kernaussage von Dora et al. 2025,
  dass das modellbasierte Kriterium auch mit unvollkommenen Rekonstruktionen funktioniert. Bei Fr ≤ 1e-4 liegt der Fehler bei ≤ 1,9 % (Δz01 ≤ 1,9 mm), bei Fr ≥ 2,4e-4 streut er stärker (−13,7…+6,3 %, zwei Ausreißer).
* Gegen die Anforderung aus 3.4 (1 px Unschärfe ⇔ |e| ≤ 0,005–0,04 % in diesem Regime) ist aber auch der Paket-Autofokus bei 256 px um ein bis zwei Größenordnungen zu ungenau: 1 % Fehler entspricht hier b = 6,5 px (Fr 2,4e-4) bzw. 14,6 px (Fr 4,7e-5).
  Die Genauigkeit ist durch `z01_tol = 0,1 mm` (⇔ 0,04–0,2 %) und die Breite/Rauhigkeit der Mulde begrenzt, nicht durch den Suchbereich.
* Der **klassische Fallback versagt vollständig** (Median 27 %, keine erkennbare Mulde der TV-Kurve im rechten Panel): bei Fr ~ 1e-4 enthält die rückpropagierte Amplitude bei falschem Fr keine schärfere Struktur als bei richtigem,
  weil das 256-px-FOV nur wenige Fresnel-Säume umfasst. Schärfemetriken sind in diesem Regime als Baseline unbrauchbar.
* Laufzeit und Auswertungen wie im p05bin8-Regime (19–25 Auswertungen, 7,7–8,1 s pro Auswertung; die Maschine war zu dieser Zeit etwas weniger ausgelastet).

### 4.4 Zielfunktions-Scan: warum die Autofokus-Stufen angepasst wurden (`scan_focus_landscape.py`, `figures/scan_focus_landscape_p05bin8.{json,png}`)

Loss(e)/Loss(0) für zwei Hologramme aus `p05bin8_z01_250.hdf5` (Fr 1,52e-2), e = Fr-Fehler in %, s = Sekunden pro Auswertung:

| Stufen-Variante | −20 | −10 | −5 | −2 | −1 | 0 | +1 | +2 | +5 | +10 | +20 | s/Eval | Form |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S0 `p05scaled_dsf4_2` (P05 find_focus 1:1 skaliert: dsf 4/2, l2 10j/1j, 300/200 It.) #0 | 0,997 | 0,994 | 0,995 | 0,998 | 0,999 | 1 | 1,001 | 1,003 | 1,008 | 1,019 | 1,049 | 1,9 | monoton (Minimum bei −10 %, flach) |
| S0 #1 | 0,980 | 0,984 | 0,990 | 0,996 | 0,998 | 1 | 1,002 | 1,005 | 1,013 | 1,027 | 1,060 | 2,4 | monoton fallend zu kleinem Fr |
| S1 dsf 2/1, schwache Filter (wie Quality-Stufen) #0 | 1,090 | 1,155 | 1,039 | 1,027 | 1,001 | 1 | 1,065 | 1,180 | 0,985 | 1,248 | 1,381 | 29,5 | verrauscht, argmin +5 % |
| S1 #1 | 1,256 | 1,069 | 1,101 | 0,934 | 0,952 | 1 | 0,977 | 1,057 | 0,956 | 1,136 | 1,202 | 12,5 | verrauscht, argmin −2 % |
| **S2 `p05filter_dsf2_1` (dsf 2/1, P05-Filter, l2 1j; Baseline-Default)** #0 | 1,149 | 0,983 | 0,985 | 0,996 | 0,998 | 1 | 1,003 | 1,008 | 1,027 | 1,069 | 1,146 | 7,5 | glatte Mulde, Minimum ≈ −5…−10 % |
| S2 #1 | 1,067 | 1,002 | 0,970 | 0,982 | 0,991 | 1 | 1,013 | 1,026 | 1,069 | 1,097 | 1,130 | 7,5 | glatte Mulde, Minimum ≈ −5 % |
| S3 dsf 4/2, schwache Filter #0 | 1,405 | 1,035 | 1,019 | 0,988 | 1,020 | 1 | 0,911 | 0,834 | 0,684 | 0,643 | 0,620 | 2,1 | monoton fallend zu großem Fr |
| S3 #1 | 2,151 | 2,073 | 1,729 | 1,372 | 1,122 | 1 | 0,904 | 0,805 | 0,659 | 0,598 | 0,549 | 2,1 | monoton fallend zu großem Fr |

Konsequenzen: (i) Die 1:1 auf 256 px skalierte P05-Konfiguration (S0: 64-px- und 128-px-Gitter) hat auf den Forge-Daten **kein brauchbares Minimum** – der erste Baseline-Smoke-Test damit lief an die
untere Suchgrenze (−49,9 %, 16 Auswertungen, 29,6 s; `baseline_smoke/`). Grund: bei 64/128 px bleiben nach Downsampling zu wenige Fresnel-Säume übrig. (ii) Schwache Regularisierung (S1, S3) macht die Zielfunktion
entweder verrauscht (Nelder-Mead springt in lokale Minima) oder monoton (Absorptionskanal fittet die Daten für zu großes Fr). (iii) Nur die Kombination "volle 256 px in der letzten Stufe + P05-Regularisierung"
(S2) ergibt eine glatte Mulde; diese ist jedoch bei Fr ≥ 1,5e-2 um −5…−10 % verschoben, was den Bias aus 4.2 erklärt. Bei Fr ≤ 6e-3 liegt das Minimum innerhalb ±2 % (Baseline-Ergebnis).
Die Zielfunktionen der Baseline-Läufe sind in `baseline_p05bin8/objective_curves.png` (linkes Panel: Residuum vs. z01/z01_true) dargestellt.

### 4.5 Laufzeit-Benchmark und Hochrechnung auf 2048 px (`bench_iteration_time.py`, `figures/bench_iteration_time.json`, 2 CPU-Threads, unter Last)

ASRM-Iteration (eine Stufe, dsf 1, pf 2; Differenzmessung t(45 It.) − t(5 It.), Minimum aus 3 Wiederholungen) und reine FFT-Paare (fft2 + ifft2, complex64):

| FOV [px] | Rechengitter (gepaddet) | ms pro ASRM-Iteration | ms pro FFT-Paar | RSS [MB] |
|---:|---:|---:|---:|---:|
| 128 | 256² | 5,98 | 0,17 | 726 |
| 256 | 512² | 22,7 | 0,71 | 771 |
| 512 | 1024² | 93,5 | 2,77 | 932 |
| 1024 | 2048² | 560 | 12,0 | 1391 |
| 2048 | 4096² | nicht gemessen (RSS ≈ 3,3 GB erwartet, Limit 3 GB); extrapoliert 2200–3400 (×4 Pixel; Faktor 1,0–1,5 für Cache-Effekte wie zwischen 1024² und 2048²) | 122 | – |

Die ASRM-Iteration ist ~30× teurer als die zwei FFT-Paare, die sie enthält – elementweise Operationen, Gauß-Filter und Constraints dominieren; die Skalierung ist ≈ N² (86 ns/Pixel bei 512², 134 ns/Pixel bei 2048²), der log-N-Anteil der FFT ist nachrangig.
Zum Vergleich der formale FFT-Faktor N² log N von 512² (256-px-FOV, pf 2) auf 4096² (2048-px-FOV, pf 2): 64·(24/18) ≈ 85; gemessen für das reine FFT-Paar 122/0,71 ≈ 172 (Cache-Effekte oberhalb von 2048²), für die ASRM-Iteration von 512² auf 2048²: 560/22,7 ≈ 25 (formal 16·(22/18) ≈ 20).
Plausibilitätscheck: die Baseline-Variante `p05filter_dsf2_1` (300 It. auf 256² + 300 It. auf 512²) ergibt 300·5,98 + 300·22,7 ms = **8,6 s** pro Auswertung – gemessen 7,8–9,9 s. Die Quality-Rekonstruktion (300 It. 128² + 200 It. 256² + 300 It. 512²) ≈ 8,5 s – gemessen ≈ 8,2 s.

Hochrechnung (21 Auswertungen pro Hologramm wie gemessen):

| Konfiguration | Gitter der Stufen | Iterationen | s pro Auswertung | pro Hologramm |
|---|---|---|---:|---:|
| Baseline 256 px (`p05filter_dsf2_1`, pf 2), gemessen | 256² / 512² | 300 / 300 | 7,8–9,9 | 2,5–7 min (Mittel 3,5 min) |
| **P05-Standard-Autofokus bei 2048 px** (`find_focus/defaults.yaml`: pf 4, dsf 16 / 4 / 4) | 512² / 2048² / 2048² | 700 / 300 / 500 | 700·0,0227 + 800·0,560 = **464 s (7,7 min)** | **≈ 2,7 h** CPU |
| Baseline-Variante auf 2048 px skaliert (pf 2, dsf 2 / 1) | 2048² / 4096² | 300 / 300 | 300·0,56 + 300·(2,2…3,4) = 830–1190 s (14–20 min) | 4,8–6,9 h CPU; RSS ≈ 3,3 GB |
| Fallback TV, 256 px gemessen → 2048 px | 512² → 4096² | 33 Auswertungen | 5,7 ms → ≈ 0,4–0,5 s (N²-Skalierung + FFT 122 ms) | 0,19 s → **≈ 15 s** CPU |

Fazit: Auf CPU ist der modellbasierte Autofokus in P05-Vollauflösung mit Stunden pro Hologramm nicht praktikabel; der P05-Betrieb setzt eine GPU voraus (Gerätewahl automatisch, Abschnitt 1.4; GPU-Messung hier nicht möglich).
Für die Thesis-Evaluation ist die 256-px-Variante (≈ 3,5 min pro Hologramm, 2 Threads) als Referenz auf einigen Dutzend Hologrammen machbar; ein Vergleich auf hunderten Hologrammen braucht GPU oder weniger Iterationen (z. B. 150/150 → ≈ 1,8 min).

## 5. Integrationsempfehlung für das Repo

Vorgeschlagene Module (Prototyp-Code in `/tmp/baseline_proto/holo_baseline_lib.py` und `run_autofocus_baseline.py` ist direkt übertragbar):

**`src/baseline/model_based_autofocus.py`** – Wrapper um die HoloWizard-API, optional importierbar (`holowizard` als Extra-Dependency; Import lazy, weil `holowizard.core` beim Import `nvidia-smi` aufruft und `tempfile.tempdir` setzt):

```python
@dataclass
class Geometry:            # Einheiten wie HoloWizard: mm, mm, keV
    energy_kev: float; px_mm: float; z02_mm: float
    def fresnel(self, z01_mm) -> float      # Fr = px^2 z01 / (lam z02 (z02 - z01)), lam = 1.2398/E nm  (identisch core/forge)
    def z01_from_fresnel(self, fr) -> float # Umkehrung

@dataclass
class AutofocusResult:
    fr_est: float; z01_est_mm: float; n_evals: int; runtime_s: float
    z01_history: list[float]; loss_history: list[float]; method: str

def model_based_autofocus(hologram_intensity: np.ndarray, geom: Geometry, fr_bounds: tuple[float, float],
                          stages: str | list = "p05filter_dsf2_1", z01_tol_mm: float = 0.1, threads: int = 2) -> AutofocusResult
    # - konfiguriert Logger einmalig (Logger.configure, level_num_header), torch.set_num_threads
    # - baut RecoParams: BeamSetup, Measurement(z01 = Mitte(z01(fr_bounds)), z01_confidence = halbe Breite, data = Intensität),
    #   Options-Stufen aus Preset ("p05_default_2048": pipe/find_focus/defaults.yaml; "p05filter_dsf2_1": 256-px-Variante), DataDimensions(n, n, "blackman")
    # - ruft holowizard.core.api.functions.find_focus.find_focus(reco_params, viewer=None, plotter=None)
    # - rechnet z01 -> Fr zurück, zählt len(loss_history) als Auswertungen

def classical_autofocus(hologram_amplitude, fr_bounds, metric="tv", n_grid=21) -> AutofocusResult   # Fallback, klar als "classical" gekennzeichnet
```

Wichtige Konventionen im Wrapper: Forge-Hologramme sind Amplituden → `intensity = amp**2`; `rot90` ist für Fr irrelevant, für Rekonstruktionen aber zurückzudrehen;
`Options.z01_tol` steuert den Abbruch; die Modul-Globals in `find_focus_z01` verbieten parallele Aufrufe im selben Prozess (Parallelisierung über Prozesse).

**`src/eval/downstream.py`** – Downstream-Metrik "Rekonstruktionsqualität bei geschätztem Fr":

```python
def reconstruct(hologram_intensity, geom: Geometry, fr: float, stages="quality_256", ) -> RecoResult(phase, absorption, losses, runtime_s)
def downstream_eval(sample, fr_candidates: dict[str, float], gt_phase, border=16) -> pd.DataFrame
    # pro Kandidat (z. B. {"true": Fr, "ml": Fr_ml, "baseline": Fr_hw}): NRMSE, NRMSE_offsetfrei, NRMSE_Gradient, Pearson, SSIM,
    # Core-Loss, Residuum des Vorwärtsmodells beim WAHREN Fr, b = sqrt(|Fr_cand/Fr_true - 1| / Fr_true), alle auch relativ zum Kandidaten "true"
def downstream_curve(sample, errors_pct=(-20,-10,-5,-1,0,1,5,10,20), ...) -> pd.DataFrame     # wie run_downstream_test.py
```

mit `src/eval/metrics.py` (NRMSE-Varianten, Pearson, SSIM, `data_residual` über das Forge-identische Vorwärtsmodell `src/physics/fresnel.py: propagate(obj, fr, pad=2)`), und Plot-Helfern (Fehler-vs-Fr-Fehler, Bildraster).

**Schnittstellen / Datenfluss**: Beide Module lesen dasselbe Sample-Format wie der Forge-Loader der Thesis (`hologram` [N,256,256] f32, `phantoms` complex64, `metadata/setup/*`).
Die Ergebnis-CSV sollte das Schema von `baseline_p05bin8/results.csv` übernehmen (`sample, source, index, fr_true, z01_true, z02, energy, px_size, fr_lo, fr_hi, method, fr_est, z01_est, rel_err_fr_pct, dz01_mm, runtime_s, n_evals`),
sodass ML-Vorhersagen (`method = "ml_<modell>"`) und Baselines mit demselben Auswertecode (`summarize`) verglichen werden.

**Was die ML-Pipeline als Metrik übernehmen sollte**: (1) relativer Fr-Fehler (MAE, Median, p95 – die Verteilung ist schwer-tailig, siehe Fallback) und Δz01 in mm; (2) **b = sqrt(|e|/Fr) in Pixeln** als physikalisch interpretierbares Genauigkeitsmaß (Ziel ≤ 1–2 px in der Auflösung, in der rekonstruiert wird); (3) Downstream: Gradienten-NRMSE und Vorwärtsmodell-Residuum relativ zur Rekonstruktion mit wahrem Fr (der reine GT-NRMSE ist bei Einzel-Distanz-Rekonstruktion unempfindlich); (4) Laufzeit pro Hologramm und Zahl der Modellauswertungen – hier liegt der eigentliche Vorteil eines ML-Autofokus (Baseline: 21 Rekonstruktionen ≈ 3,5 min bei 256 px, ≈ 2,7 h bei 2048 px auf CPU).
Empfohlene Hybrid-Nutzung als eigenes Experiment: ML-Schätzung als `z01`-Startwert und ±(2–3 σ_ML) als `z01_confidence` für `find_focus` → Reduktion der Auswertungen und Entfernung des ML-Restfehlers; umgekehrt kann die Baseline-Schätzung auf realen P05-Daten (kein GT) als Pseudo-Label dienen.

## 6. Offene Fragen an die Betreuer

1. **Standard-Autofokus-Parameter an P05**: Sind `pipe/scripts/config/find_focus/defaults.yaml` (700/300/500 Iterationen, pf 4, dsf 16/4/4, l2 10j/10j/1j, Nesterov 1,0, `z01_tol` 0,1 mm) und `z01_confidence = 5 mm` die im Betrieb verwendeten Werte? Wird die (z01, a0)-Variante (`find_focus_z01_a0.py`) oder die Flatfield-PCA-Korrektur im Autofokus eingesetzt? Welcher a0 ist typisch (Beispiele: 0,98, 1,1)?
2. **Typische z01-Suchbereiche**: ±5 mm (≈ ±2 % Fr bei 250 mm) ist der Default – wie unsicher ist z01 in der Praxis (Motorposition, Holder-Höhe, nach Energiewechsel)? Welche z01-Werte kommen an P05 vor (die Thesis nutzt 50–400 mm)? Soll der ML-Autofokus einen breiten Bereich (Fr ± 50 %) abdecken oder nur die Verfeinerung in ±5 mm?
3. **Genauigkeitsanforderung**: Ist das Kriterium "Defokus-Unschärfe b = sqrt(|e|/Fr) ≤ 1–2 px in der Rekonstruktionsauflösung" (⇔ Δz01 ≈ 0,06–0,25 mm bei 250 mm Vollauflösung; `z01_tol` 0,1 mm ⇔ 1,3 px) akzeptiert, oder gibt es eine Anforderung aus der Tomographie (Konsistenz über Projektionen) bzw. aus der Rekonstruktionsqualität (z. B. ΔNRMSE)? Soll die Referenz-Baseline für die Thesis mit kleinerem `z01_tol` (z. B. 0,01 mm, mehr Auswertungen) gerechnet werden, da im 256-px-Regime die Mulde scharf ist, die Genauigkeit aber durch die Toleranz begrenzt wird?
4. **Auflösung des Autofokus**: Die P05-Zielfunktion wird in den letzten Stufen auf dsf 4 (512² Gitter aus 2048 px) ausgewertet. Ist die 256-px-Variante (dsf 8 äquivalent) für die Thesis als Referenz akzeptabel, und ist der beobachtete Bias (−5…−10 % bei Fr ≥ 1,5e-2 im 256-px-Setting) aus dem Betrieb bekannt? Auf welcher Hardware (GPU-Typ) läuft der Autofokus an P05 und wie lange dauert er pro Hologramm?
5. **Daten-Konvention**: Forge speichert Amplituden |ψ| (+ additives Gauß-Rauschen) statt Intensitäten; reale P05-Daten sind flatfield-korrigierte Intensitäten mit Photonenrauschen. Soll die Thesis-Pipeline auf Amplituden trainiert werden (dann Konvertierung vor der Core-API nötig und Rauschmodell abweichend) oder sollen die Forge-Daten als Intensitäten interpretiert/erzeugt werden?
6. **Trainingsregime**: Im 256-px-Thesis-Regime (Fr 5e-5…4e-4, dsf 1) ist die Einzel-Distanz-ASRM-Rekonstruktion schlecht gestellt (NRMSE > 3 schon bei korrektem Fr). Ist dieses Regime beabsichtigt (Autofokus als reine Fr-Regression) oder sollten die Trainingsdaten das gebinnte P05-Regime (Fr·64, `detector_size 2048`, `downsample_factor 8`) abbilden, damit der Downstream-Test aussagekräftig ist? Alternativ: Downstream-Test mit mehreren Distanzen oder Support-Constraint?
7. **Stabilität**: Mit dem P05-Nesterov-Momentum 1,0 divergierte die Rekonstruktion bei einigen Forge-Phantomen (Phase → −600 rad, Loss ≈ 1, NaN); mit 0,9 nicht. Ist das auf realen Daten bekannt (Phantome mit −5…−7 rad Phasenschub und harten Kanten sind ggf. "strenger" als reale Proben)?
8. **Lizenz/Zitation**: Soll der Baseline-Wrapper HoloWizard (MIT) als Abhängigkeit einbinden (Version 3.0.6 pinnen) und die beiden Optics-Express-Paper zitieren – oder nur die Ergebnisse (CSV) ins Repo?

## 7. Reproduktion (alle Befehle; Arbeitsverzeichnis `/tmp/baseline_proto`)

```bash
cd /tmp/baseline_proto
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/tmp/holo311/bin/python

# Daten (Forge; je ein Prozess pro z01; Seeds 1234+i; ~1 min)
$PY gen_data.py --regime p05bin8 --z01 50 100 250 400 --n 4 --out /tmp/baseline_proto/data
$PY gen_data.py --regime small   --z01 50 100 250 400 --n 3 --out /tmp/baseline_proto/data

# Downstream-Test p05bin8 (8 Hologramme x 9 Fr-Fehler, 699 s) und small (4 x 9, 678 s)
$PY run_downstream_test.py --inputs data/p05bin8_z01_50/p05bin8_z01_50.hdf5 data/p05bin8_z01_100/p05bin8_z01_100.hdf5 \
    data/p05bin8_z01_250/p05bin8_z01_250.hdf5 data/p05bin8_z01_400/p05bin8_z01_400.hdf5 \
    --n-samples 2 --errors -20 -10 -5 -1 0 1 5 10 20 --out /tmp/baseline_proto/downstream_p05bin8 --grid-sample 4
$PY run_downstream_test.py --inputs data/small_z01_250/small_z01_250.hdf5 data/small_z01_100/small_z01_100.hdf5 \
    --n-samples 2 --errors -20 -10 -5 -1 0 1 5 10 20 --out /tmp/baseline_proto/downstream_small --grid-sample 0
# nur Bildraster (2-zeilig) neu erzeugen (~100 s):
$PY regen_image_grid.py data/p05bin8_z01_250/p05bin8_z01_250.hdf5 0 downstream_p05bin8
$PY regen_image_grid.py data/small_z01_250/small_z01_250.hdf5 0 downstream_small

# Baseline p05bin8 (16 Hologramme, Paket + Fallback, 57 min) und small (12 Hologramme, 36 min)
$PY run_autofocus_baseline.py --inputs "data/p05bin8_z01_*/p05bin8_z01_*.hdf5" --n-samples 4 \
    --out /tmp/baseline_proto/baseline_p05bin8 --search-width 50 --method both
$PY run_autofocus_baseline.py --inputs "data/small_z01_*/small_z01_*.hdf5" --n-samples 3 \
    --out /tmp/baseline_proto/baseline_small --search-width 50 --method both
# Smoke-Test mit der 1:1 skalierten P05-Konfiguration (laeuft an die Suchgrenze):
$PY run_autofocus_baseline.py --inputs data/p05bin8_z01_250/p05bin8_z01_250.hdf5 --n-samples 1 \
    --out /tmp/baseline_proto/baseline_smoke --search-width 50 --method both --focus-variant p05scaled_dsf4_2 --iters 300 200

# Zielfunktions-Scan (4 Varianten x 2 Hologramme x 11 Fr-Fehler, ~12 min) und Benchmark (~4 min)
$PY scan_focus_landscape.py data/p05bin8_z01_250/p05bin8_z01_250.hdf5 2 figures/scan_focus_landscape_p05bin8.json
$PY bench_iteration_time.py 128 256 512 1024
```

Alle Skripte haben `--help`. Die HoloWizard-Konsolenausgabe wird über `holo_baseline_lib.quiet_holowizard_logging` auf Header-Level reduziert; Session-Logs liegen in `logs/<session>/`.

## Anhang A: Probleme und Workarounds beim Betrieb von holowizard.core auf CPU

| Problem | Ort | Lösung im Prototyp |
|---|---|---|
| `AttributeError: module 'logging' has no attribute 'comment'/'loss'/'image_final'` bzw. Schreiben der Geometrie-Datei schlägt fehl | `core/logging/logger.py` registriert custom Log-Level erst in `Logger.configure`; `reconstruction/logging.py` schreibt nach `Logger.working_dir/session_name` | `Logger.current_log_level = Logger.level_num_header; Logger.configure(working_dir, session_name)` vor dem ersten Aufruf (`quiet_holowizard_logging`) |
| Rekonstruktion passt nicht zur GT (Korrelation ≈ 0) | `process_image`: `torch.rot90` | `torch.rot90(result, k=-1)` |
| Zu geringe Rekonstruktionsamplitude / falsches Rauschmodell | Forge-Hologramm ist Amplitude, API zieht `sqrt` | `hologram_amp**2` übergeben |
| Phase überschießt auf das Doppelte (−10 statt −5 rad) mit P05-Filterbreiten bei 256 px | Filter für 2048 px ausgelegt | letzte Quality-Stufe mit fwhm 0+2j, Nesterov-Filter 2+2j, l2 0,1j (`figures/smoke*.png`, `explore_reco_params.png`) |
| Divergenz (NaN, Phase −600 rad, Loss ≈ 1) bei Nesterov 1,0 (P05-Default) auf einigen Phantomen (erster Downstream-Lauf `downstream_p05bin8_v1_nesterov1.0/`: mehrere divergierte Rekonstruktionen) | `reconstruct.py`, Momentum | Nesterov 0,9 in den Quality-Stufen (0/72 divergiert); die Autofokus-Stufen behalten 1,0 (keine Divergenz beobachtet, starke Regularisierung) |
| Autofokus läuft an die Suchgrenze | Zielfunktion ohne Minimum bei 64/128-px-Gittern (S0) | Variante `p05filter_dsf2_1` (Abschnitt 4.4) |
| Forge-CLI `--seed` wirkungslos (`if seed := args.seed is not None` → `seed=True`) | `forge/scripts/generate_data.py` | `gen_data.py` ruft `torch_settings.set_reproducibility(seed)` direkt und nutzt die Generator-Klassen |
| Laufzeit-Benchmark bei kleinen Gittern durch Initialisierung/Konkurrenz verfälscht (negative Werte) | – | Differenzmessung t(45) − t(5), Minimum aus 3 Wiederholungen (`bench_iteration_time.py`) |
| `grep` puffert in tmux-Pipelines | – | `grep --line-buffered` / `tee` |
