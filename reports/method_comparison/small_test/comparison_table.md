# Methodenvergleich data_small/test

100 identical holograms (`test.hdf5` of `data_small`, indices 0–99); statistics from `src.baseline.results.summarize` over the `samples.csv` of every method.

Inputs: `reports/baseline_model_based/small_test/samples.csv`, `reports/method_comparison/small_test/ml_cnn_hologram_samples.csv`, `reports/method_comparison/small_test/ml_cnn_hologram_spectrum_samples.csv`, `reports/method_comparison/small_test/ringfit_cos_samples.csv`, `reports/method_comparison/small_test/npe_radial_ens5_samples.csv`

## Summary (English column names, `summary_markdown_table`)

| Method | n | MAE rel. Fr [%] | Median [%] | p95 [%] | Bias [%] | MAE z01 [mm] | Median z01 [mm] | blur b median / p95 [px] | within 2 % / 5 % | Runtime [s/hologram] | Evaluations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `holowizard_find_focus` (Fr_true ± 50 %) | 100 | 4.15 | 3.10 | 9.79 | -3.07 | 7.19 | 4.28 | 1.80 / 3.12 | 36 % / 74 % | 163 | 21.1 |
| `holowizard_find_focus` ohne Grenzfälle (|e| ≥ 40 %) | 99 | 3.72 | 3.04 | 9.55 | -2.63 | 6.63 | 4.18 | 1.79 / 2.84 | 36 % / 75 % | 163 | 21.2 |
| `classical_tv` (Fr_true ± 50 %) | 100 | 9.33 | 0.76 | 44.12 | +8.17 | 14.28 | 1.55 | 0.83 / 8.55 | 64 % / 70 % | 0.194 | 32.4 |
| `ml_cnn` (Hologramm-CNN) | 100 | 7.82 | 6.06 | 18.60 | -2.11 | 13.04 | 8.69 | 2.45 / 5.78 | 15 % / 42 % | 0.0112 | 1.0 |
| `ml_cnn_hologram_spectrum` (2-Kanal-CNN) | 100 | 7.10 | 5.95 | 17.22 | -0.43 | 11.42 | 7.23 | 2.42 / 5.16 | 25 % / 46 % | 0.0113 | 1.0 |
| `npe_radial_ens5_median` (NPE, Posterior-Median) | 100 | 9.36 | 6.67 | 28.14 | +3.08 | 11.05 | 8.27 | 2.61 / 8.23 | 20 % / 43 % | 0.029 | 1000.0 |
| `ringfit_cos` (CTF-Ring-Fit) | 100 | 26.80 | 5.58 | 103.86 | -3.73 | 38.22 | 6.48 | 2.45 / 12.84 | 43 % / 49 % | 0.0204 | 191.8 |

## Zusammenfassung (deutsche Spaltennamen, identische Zahlen)

| Methode | n | MAE rel. Fr [%] | Median [%] | p95 [%] | Bias [%] | MAE z01 [mm] | Median z01 [mm] | b Median / p95 [px] | innerhalb 2 % / 5 % | Laufzeit [s/Hologramm] | Auswertungen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `holowizard_find_focus` (Fr_true ± 50 %) | 100 | 4.15 | 3.10 | 9.79 | -3.07 | 7.19 | 4.28 | 1.80 / 3.12 | 36 % / 74 % | 163 | 21.1 |
| `holowizard_find_focus` ohne Grenzfälle (|e| ≥ 40 %) | 99 | 3.72 | 3.04 | 9.55 | -2.63 | 6.63 | 4.18 | 1.79 / 2.84 | 36 % / 75 % | 163 | 21.2 |
| `classical_tv` (Fr_true ± 50 %) | 100 | 9.33 | 0.76 | 44.12 | +8.17 | 14.28 | 1.55 | 0.83 / 8.55 | 64 % / 70 % | 0.194 | 32.4 |
| `ml_cnn` (Hologramm-CNN) | 100 | 7.82 | 6.06 | 18.60 | -2.11 | 13.04 | 8.69 | 2.45 / 5.78 | 15 % / 42 % | 0.0112 | 1.0 |
| `ml_cnn_hologram_spectrum` (2-Kanal-CNN) | 100 | 7.10 | 5.95 | 17.22 | -0.43 | 11.42 | 7.23 | 2.42 / 5.16 | 25 % / 46 % | 0.0113 | 1.0 |
| `npe_radial_ens5_median` (NPE, Posterior-Median) | 100 | 9.36 | 6.67 | 28.14 | +3.08 | 11.05 | 8.27 | 2.61 / 8.23 | 20 % / 43 % | 0.029 | 1000.0 |
| `ringfit_cos` (CTF-Ring-Fit) | 100 | 26.80 | 5.58 | 103.86 | -3.73 | 38.22 | 6.48 | 2.45 / 12.84 | 43 % / 49 % | 0.0204 | 191.8 |

## Boundary cases of `holowizard_find_focus` (|rel. error| ≥ 40 %)

1 of 100 holograms (1 %). Objective range over the search interval (loss max / loss min − 1 over all Nelder-Mead evaluations, `histories.json`): boundary cases #0: 0.121; regular cases median 0.390 (min 0.066, p10 0.216, p90 0.555, max 0.944, n = 99).

| # | Fr_true | z01_true [mm] | find_focus e [%] | Δz01 [mm] | Fr_est above lower / below upper bound [%] | evals | objective range | classical_tv e [%] | ml_cnn e [%] | ml_cnn_hologram_spectrum e [%] | npe_radial_ens5_median e [%] | ringfit_cos e [%] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.971e-03 | 132.0 | -47.5 | -62.5 | 5.0 / 65.0 | 19 | 0.121 | +21.3 | +0.9 | +9.4 | -5.3 | +11.7 |
