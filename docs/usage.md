# Usage

## Installation

Python ≥ 3.10. PyTorch is installed separately so that you can choose the CPU or the CUDA
build.

```bash
python -m venv .venv && source .venv/bin/activate          # or: conda create -n holofocus python=3.12
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU only (VM, CI)
# pip install torch --index-url https://download.pytorch.org/whl/cu124  # GPU machine
pip install -e ".[dev]"
```

The package does not hard-code a device. Arrays are created on the device of their inputs;
`holofocus.get_device("auto")` returns `cuda` when available. Default precision is
`complex64`/`float32`; the tests also run in `complex128`.

## Tests and lint

```bash
pytest              # all tests; CUDA tests are skipped without a GPU
ruff check . && ruff format --check .
```

GitHub Actions (`.github/workflows/tests.yml`) runs lint, the test suite and the Stage 1 and
2 figure scripts in quick mode on CPU for Python 3.10 and 3.12. It also generates the demo
dataset of `configs/dataset/ci.yaml`, inspects it and keeps it for 30 days as the workflow
artifact `demo-dataset-ci` (Python 3.12 job). On the demo test split it runs the baseline
CLI with `configs/baseline/ci.yaml` (3 samples) and then the Stage 3 figure script in quick
mode on that run and on the committed `results/`.

## Conventions

* Lengths in metres, energies in keV, phases in radians.
* Distances from the focus of the optics: `z01` focus–sample, `z12` sample–detector,
  `z02 = z01 + z12` focus–detector ([1] Fig. 1).
* `ConeBeamGeometry(energy_kev, z01, z02, pixel_size=6.5e-6)` gives `magnification`,
  `effective_pixel_size`, `effective_distance`, `fresnel_number` ([1] Eq. (9)) and
  `fresnel_number_at(z)` ([2] Eq. (5)–(7)). `binned(b)` multiplies the pixel size by `b`
  (`Fr` grows by `b²`); `defocused(dz)` is the geometry with `z01 + dz` at fixed `z02`.
* The Fresnel propagator uses `exp(-iπ(kx² + ky²)/Fr)` with `k` in cycles per pixel. A
  negative `Fr` propagates backwards. Arrays should satisfy `N ≥ 1/|Fr|` ([1] Sec. 3.2.2);
  `FresnelPropagator` warns otherwise.
* Objects are the projected refractive index `O~ = φ + iμ` (`φ ≤ 0`, `μ ≥ 0`), the exit wave
  is `exp(iO~)`, the hologram `|D_Fr(exp(iO~))|²` ([1] Sec. 3.1).

## Stage 1 figures

```bash
python scripts/make_stage1_figures.py                     # writes docs/laborbuch/figures/stage1-*
python scripts/make_stage1_figures.py --quick --out /tmp/figures   # small smoke test
```

The full run takes about 1.5 min on 4 CPU threads and writes `stage1-metrics.json` with the
numbers quoted in the lab notebook.

## Synthetic dataset (Stage 2)

A YAML file describes the geometry, the sampling of `z01`, the phantoms, the noise model, the
storage options and the size of every split. Three configs are provided:

| Config | Detector grid | `z01` | Samples (train/val/test) | CPU time | Size |
|---|---|---|---|---|---|
| `configs/dataset/cpu-4x.yaml` | P05 binned 4x: 512² px of 26 µm, simulated 2x finer | 100 mm ± 5 mm | 2000/250/250 | ~16 min | ~1.8 GB |
| `configs/dataset/cpu-4x-z01-range.yaml` | as above | guess 80–300 mm, ± 5 mm | 2000/250/250 | ~23 min | ~1.8 GB |
| `configs/dataset/ci.yaml` | binned 16x: 128² px of 104 µm | 100 mm ± 5 mm | 32/8/8 | < 1 s | ~4 MB |

```bash
holofocus-dataset generate configs/dataset/cpu-4x.yaml --out data/cpu-4x
holofocus-dataset generate configs/dataset/ci.yaml --out /tmp/demo --splits val --limit 8
holofocus-dataset generate configs/dataset/cpu-4x.yaml --out data/cpu-4x --device cuda
holofocus-dataset inspect data/cpu-4x/*.h5        # shapes, labels, seed, commit, run time
python -m holofocus.dataset generate ...           # same CLI without the console script
```

`generate` writes one HDF5 file per split plus a copy of the config (`config.yaml`). A split
that already exists with the same config and size is skipped; a different one is an error
unless `--overwrite` is given. Files are written to `*.h5.part` and renamed when complete.
Unknown keys in the YAML file are rejected; missing keys take the defaults of
`holofocus.dataset.DatasetConfig`.

### What a sample is

For sample `i` of split `s`, a random stream `SeedSequence(seed, spawn_key=(crc32(s), i))`
draws, in this order: `z01_guess` (if a range is given), `dz01` uniform in `dz01_range`
(`±5 mm`, the box `Omega_z` of [2] Eq. (17)), the phantom class, the maximal phase shift,
`A0`, the photon count `N` (log-uniform), the phantom and finally the noise. The hologram is
simulated with the true `z01 = z01_guess + dz01` at fixed `z02`; phantoms are defined on the
detector pixel grid, so their apparent size carries no information about `z01`. Noise is
`(Poisson(N I) + Normal(0, read_noise²)) / N` on the flat-field corrected intensity (an
assumption; [2] simulates noise-free holograms). The lab notebook
([02-datensatz.md](laborbuch/02-datensatz.md)) explains every choice.

Samples are reproducible one by one and independent of the order and number generated.
Labels, phantoms and noise-free holograms agree between CPU and CUDA up to float32 rounding;
the noise realization is bit-identical only on the same device and library versions.

### HDF5 layout (one file per split)

```
hologram        (N, H, W) float32   noisy, flat-field corrected (1 = free space)
hologram_clean  (N, H, W) float32   noise-free (storage.store_clean)
phase           (N, H, W) float32   ground-truth phase in rad (storage.store_phase)
labels/<key>    (N,) float64        dz01, z01, z01_guess, fresnel_number, fresnel_number_guess
meta/<key>      (N,)                phantom_type, phase_shift, photons, a0, effective_pixel_size
attrs                               config_yaml, seed, seed_rule, label_convention, git_commit, ...
```

`dz01 = z01 − z01_guess` (true minus guess, metres), so a prediction is added to the guess;
[2] Eq. (9) uses the opposite sign.

### Reading a split in PyTorch

```python
import torch
from holofocus.dataset import HologramDataset, read_config, read_labels

ds = HologramDataset("data/cpu-4x/train.h5", target="dz01_mm")  # items: (1, H, W), scalar
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
labels = read_labels("data/cpu-4x/train.h5")  # all labels as numpy arrays
config = read_config("data/cpu-4x/train.h5")  # the generating DatasetConfig
```

Targets: `dz01` (m), `dz01_mm`, `z01`, `fresnel_number` (true `Fr`) and `fresnel_ratio`
(`Fr / Fr_guess − 1`, about `dz01 / z01`). `with_phase=True` also returns the ground-truth
phase. Each DataLoader worker opens its own HDF5 handle.

### Stage 2 figures

```bash
python scripts/make_stage2_figures.py --holograms DIR       # writes docs/laborbuch/figures/stage2-*
python scripts/make_stage2_figures.py --quick --out /tmp/figures   # small smoke test
```

The full run takes about 3 min on 4 CPU threads and needs about 8 GB of RAM (the binning study
simulates the full 2048² detector in a 13230² array). `DIR` is `Examples/data/holograms` of the
Zenodo record of [1]; without `--holograms` the measured-noise figure is skipped.
`stage2-metrics.json` holds the numbers quoted in the lab notebook.

## Classical autofocus baseline (Stage 3)

The baseline is the model-based autofocus of [2]. The inner reconstruction is ASRM ([1]
Alg. 5, ported from holowizard 3.0.6 into `holofocus.asrm`) with the schedule of [2] Table 3.
The objective is the model fit error (MFE, [2] Eq. (14)–(15)), and an outer projected
Nelder–Mead search runs on `z01_guess ± 5 mm` ([2] Eq. (22)).

```python
import torch
from holofocus.asrm import ASRMSettings, asrm_reconstruct
from holofocus.autofocus import find_focus
from holofocus.dataset import read_config

config = read_config("data/cpu-4x/test.h5")
settings = ASRMSettings.for_binning(config.geometry.binning)  # [2] Table 3 on the binned grid
result = find_focus(hologram, config.geometry.at(z01_guess), settings)  # ± 5 mm, stop 100 µm
result.z01, result.n_reconstructions, result.seconds, result.evaluations
reco = asrm_reconstruct(hologram, config.geometry.at(result.z01).fresnel_number, settings)
reco.obj.real  # phase in rad on the field of view; reco.final_loss is the MFE
```

`hologram` is a `(H, W)` tensor of flat-field corrected intensity (free space `a0²`); its
dtype sets the precision. `ASRMSettings.for_binning(b)` divides the down-sampling of [2]
Table 3 (16×/4×/4× of the 2048² detector) by the binning, so on 4×-binned data the stages run
at 4×/1×/1× and keep the physical pixel sizes of [2]. Where the reference code differs from
the printed algorithm, the reference behaviour is the default (`update`, `gradient`,
`pin_phase`; see the docstring of `holofocus.asrm`).

`holofocus.autofocus` also has the image criteria VAR, SPEC, GRA, LAP, ToG and GoG of [2]
Sec. 3.7 (`image_criteria`, with the 4-px Gaussian prefilter of the reference script) and
the focus curves of [2] Fig. 7 (`simulated_focus_curve`, no reconstruction) and Fig. 8
(`reconstruction_focus_curve`, one reconstruction per point). The Zenodo holograms of [1] and
the reference values of [2] Table 1 are in `holofocus.measured` (`MEASURED_SAMPLES`,
`load_measured(directory, sample, binning)`).

### Baseline runs on a dataset split

| Config | Data | ASRM | Search |
|---|---|---|---|
| `configs/baseline/paper2-cpu.yaml` | `cpu-4x` (512², binning 4) | [2] Table 3 at 4×/1×/1×, padding 2 (1024² arrays) | ± 5 mm, stop at 100 µm |
| `configs/baseline/ci.yaml` | `ci` (128², binning 16) | Table 3 with 20× fewer iterations | ± 5 mm, stop at 1 mm |

```bash
holofocus-baseline run configs/baseline/paper2-cpu.yaml --data data/cpu-4x/test.h5 \
    --out results/baseline/cpu-4x-test --limit 48 --threads 3
holofocus-baseline run configs/baseline/paper2-cpu.yaml --data data/cpu-4x/test.h5 \
    --out runs/subset --indices 3 5 8 --device cuda
holofocus-baseline summary results/baseline/cpu-4x-test
```

A run directory holds:

* `run.yaml`: method config, ASRM settings for the binning of the data, dataset identity
  (path, SHA-256, seed, generator commit), git commit, torch version, device, threads, host;
* `results.csv`: one row per sample, keyed by the `index` in the split: `phantom_type`,
  `photons`, `z01_true`, `z01_guess`, `dz01_true`, `z01_est` (m), `error_um`, `abs_error_um`,
  `n_reconstructions`, `n_evaluations` (including penalized repeats), `nm_iterations`,
  `converged`, `seconds` (whole autofocus), `seconds_per_reconstruction`, `mfe_est`,
  `mfe_true` (an extra reconstruction at the true `z01`, `evaluate_truth: true`),
  `phase_rmse_est` and `phase_rmse_true` (rad, against the ground-truth phase);
* `traces.jsonl`: every probing point of every search (`z01`, MFE, seconds, reconstructed);
* `summary.json`: error statistics, fraction within 50/100/250/1000 µm, reconstructions and
  seconds per autofocus, rewritten after every sample.

Runs are resumable: an interrupted run continues with the missing samples, and a directory
that holds a run with another config or dataset is refused. To compare a learned estimator
with the baseline (Stage 4), evaluate it on the `index` values of `results.csv`.

On CPU, one reconstruction at 1024² takes about 25–30 s with 3 threads; an autofocus needs
11–13 reconstructions (plus one at the true `z01` for `mfe_true`), so about 5–6 min per sample.
Wall times are only comparable when the CPU is not shared: pin long runs to cores
(`taskset -c 0-2 holofocus-baseline run ... --threads 3`) and run other work on the remaining
core.

### Stage 3 experiments and figures

```bash
# autofocus of the four Zenodo holograms on the 4x grid, compared with [2] Table 1: binned 4x
# (block mean) or unbinned with the 4x down-sampling of [2] Table 3 inside ASRM (--binning 1)
python scripts/run_stage3_experiments.py --threads 4 measured --holograms DIR --out results/measured/p2
python scripts/run_stage3_experiments.py --threads 4 measured --holograms DIR --binning 1 \
    --out results/measured/b1-p2
python scripts/run_stage3_experiments.py --threads 4 measured --holograms DIR --binning 1 \
    --padding 4 --samples spider_hair tooth --out results/measured/b1-p4
# focus curves with one reconstruction per point ([2] Fig. 8), 41 points over ± 10 mm
python scripts/run_stage3_experiments.py curves --data data/cpu-4x/test.h5 --indices 5 0 \
    --out results/focus-curves
python scripts/run_stage3_experiments.py curves --holograms DIR --samples spider_hair \
    --binning 1 --points 21 --out results/focus-curves
# MFE in 0.2 mm steps around the true z01 for noisy, noise-free and grid-simulated holograms
python scripts/run_stage3_experiments.py bias --data data/cpu-4x/test.h5 --indices 2 \
    --out results/bias
python scripts/run_stage3_experiments.py --iteration-scale 4 bias --data data/cpu-4x/test.h5 \
    --indices 2 --variants grid-nofilter --out results/bias/iterations-4x
# padding: five-point MFE curves of the spider hair (p = 2, p = 2 with the L2 weight x 2/3,
# 1022 px arrays, p = 3, p = 7) and the synthetic scan with p = 4, with the L2 compensation
# and on 1022 px arrays
python scripts/run_stage3_experiments.py curves --holograms DIR --samples spider_hair \
    --binning 1 --padding 2 --points 5 --half-range 1e-3 --out results/padding/p2
python scripts/run_stage3_experiments.py --l2-scale 0.6667 curves --holograms DIR \
    --samples spider_hair --binning 1 --padding 2 --points 5 --half-range 1e-3 \
    --out results/padding/p2-l2
python scripts/run_stage3_experiments.py curves --holograms DIR --samples spider_hair \
    --binning 1 --padding 1.99609375 --points 5 --half-range 1e-3 --out results/padding/p2-1022
python scripts/run_stage3_experiments.py curves --holograms DIR --samples spider_hair \
    --binning 1 --padding 3 --points 5 --half-range 1e-3 --out results/padding/p3
python scripts/run_stage3_experiments.py curves --holograms DIR --samples spider_hair \
    --binning 1 --padding 7 --points 5 --half-range 1e-3 --out results/padding/p7
python scripts/run_stage3_experiments.py bias --data data/cpu-4x/test.h5 --indices 2 \
    --variants noisy --padding 4 --out results/padding/bias-p4
python scripts/run_stage3_experiments.py --l2-scale 0.6667 bias --data data/cpu-4x/test.h5 \
    --indices 2 --variants noisy --out results/padding/bias-p2-l2
python scripts/run_stage3_experiments.py bias --data data/cpu-4x/test.h5 --indices 2 \
    --variants noisy --padding 1.99609375 --out results/padding/bias-p2-1022
# figures and stage3-metrics.json from results/
python scripts/make_stage3_figures.py --data data/cpu-4x/test.h5
python scripts/make_stage3_figures.py --quick --out /tmp/figures   # smoke test (CI)
```

`DIR` is `Examples/data/holograms` of the Zenodo record of [1]. The experiment commands skip
outputs that already exist; `--iteration-scale 0.02` gives a quick smoke test.
`--paper-padding` takes the array sizes of [2] Table 1 (padding 7 or 4). `bias`
regenerates the sample from its seed (the stored phase must match) and reconstructs four data
models (`--variants`, default all): the stored noisy hologram (`noisy`), the same hologram
without noise (`clean`), a noise-free hologram simulated directly on the reconstruction grid
without sub-pixel integration (`grid`), and the latter without the object filters of [2]
Table 3 (`grid-nofilter`). `--l2-scale` multiplies the L2 weight on the absorption of every
stage. The L2 step is normalized by the norm of the absorption over the whole padded array,
which holds 2×2 mirrored copies of the field of view at padding 2 and the 3×3 block at
padding 4 or more; at padding 2 the per-pixel step is therefore 3/2 times larger, and
`--l2-scale 0.6667` compensates that. `--padding` also takes non-integer factors:
`1.99609375` gives arrays of 1022 px for the 512 px field of view of stages 2 and 3, the
period 2n − 2 of the mirror padding (which does not repeat the edge pixel), so the periodic
FFT continues the mirrored copies without a seam (lab notebook 03, section 5.8). The phase at
the estimate (`<sample>-phase.npy`) is not tracked in git; without it the phase panels of
`stage3-measured.png` stay empty.

The `results/` directory of the repository holds the outputs that the lab notebook quotes:

```
results/baseline/cpu-4x-test/   first 48 samples of the cpu-4x test split (paper2-cpu.yaml)
results/measured/p2/            Zenodo holograms, binned 4x (block mean), padding 2
results/measured/b1-p2/         unbinned input, 4x down-sampling inside ASRM as in [2], padding 2
results/measured/b1-p4/         the same with padding 4 (spider hair, tooth)
results/focus-curves/           synthetic-<index>.json, measured-<sample>.json
results/bias/                   bias-<index>.json (MFE scans for four data models)
results/bias/iterations-4x/     the filter-free variant with four times the iterations
results/padding/                spider-hair curves (p2, p2-l2, p2-1022, p3, p7) and MFE scans
                                of sample 2 (bias-p4, bias-p2-l2, bias-p2-1022)
results/crosscheck/             reports of scripts/crosscheck_asrm_holowizard.py
```

## Learned autofocus (Stage 4)

`holofocus.learned` estimates `dz01 = z01 - z01_guess` (±5 mm) from one hologram as a Gaussian
mean and standard deviation. Two input representations are implemented:

| `representation.kind` | Input | Model |
|---|---|---|
| `spectrum` | radially averaged power spectrum of the Hann-windowed hologram on a uniform `f^2` axis (1024 bins up to 0.125 cycles²/px²), log, standardized; its minima lie at `f^2 = k Fr` | `SpectrumCNN` (1-D convolutions) |
| `image` | the hologram binned 2×2, standardized; random D4 transforms (rotations by 90°, flips) during training | `ImageCNN` (2-D convolutions) |

Both models end in a Gaussian head (mean and variance) trained with the β-NLL. Training keeps
the epoch with the lowest validation MAE and calibrates the standard deviation on the
validation split (`sigma_scale`).

```python
from holofocus.learned import LearnedAutofocus

autofocus = LearnedAutofocus.load("results/learned/spectrum-cnn-wide")  # run dir or model.pt
dz01, sigma, sigma_raw = autofocus.estimate(holograms)  # (B, H, W) or (H, W) tensor; metres
z01_est = z01_guess + dz01
```

### Training, prediction and comparison

| Config | Representation | Model |
|---|---|---|
| `configs/learned/spectrum-cnn.yaml` | spectrum, 1024 bins | 4 stages (16/32/64/64 channels), 178k parameters, 100 epochs |
| `configs/learned/spectrum-cnn-wide.yaml` | spectrum, 1024 bins | 4 stages (32/64/128/128 channels), 450k parameters, 100 epochs |
| `configs/learned/image-cnn.yaml` | image, binning 2 (256²) | 4 stages (16/32/64/128 channels), 308k parameters, 40 epochs |
| `configs/learned/ci.yaml` | spectrum, 128 bins | 2 stages, 3 epochs (CI) |

```bash
# 8000 more training samples with the physics of cpu-4x (about 60 min with 3 threads)
holofocus-dataset generate configs/dataset/cpu-4x-extra.yaml --out data/cpu-4x-extra --splits train
# train both spectrum CNNs on 10000 samples (both files in this order), select and calibrate
# on val; the main model is the one with the lower validation MAE (spectrum-cnn-wide)
for name in spectrum-cnn spectrum-cnn-wide; do
    holofocus-learn train configs/learned/$name.yaml \
        --train data/cpu-4x/train.h5 data/cpu-4x-extra/train.h5 --val data/cpu-4x/val.h5 \
        --out results/learned/$name --threads 1
    holofocus-learn predict results/learned/$name --data data/cpu-4x/val.h5 \
        --out results/learned/$name/val
    holofocus-learn predict results/learned/$name --data data/cpu-4x/test.h5 \
        --out results/learned/$name/test
done
# baseline on the first 24 validation samples: its mean error is the offset of the
# corrected baseline
taskset -c 0-2 holofocus-baseline run configs/baseline/paper2-cpu.yaml \
    --data data/cpu-4x/val.h5 --out results/baseline/cpu-4x-val --limit 24 --threads 3
# warm start: Nelder-Mead from (z01_est - 0.5 mm, z01_est + 0.5 mm) instead of the ends of
# Omega_z; the bounds and the stopping rule stay those of [2]. Same cores and threads as
# results/baseline/cpu-4x-test, so the times compare.
taskset -c 0-2 holofocus-baseline run configs/baseline/paper2-cpu-warm.yaml \
    --data data/cpu-4x/test.h5 --out results/baseline/cpu-4x-test-warm --limit 48 --threads 3 \
    --warm-start results/learned/spectrum-cnn-wide/test/predictions.csv --half-width 5e-4
# diagnosis: the four samples where warm and cold start ended farthest apart, from
# (z01_est, z01_est + 0.5 mm), so that the estimate itself is the first probe
taskset -c 0-2 holofocus-baseline run configs/baseline/paper2-cpu-warm.yaml \
    --data data/cpu-4x/test.h5 --out results/baseline/cpu-4x-test-warm-vertex \
    --indices 30 35 40 41 --threads 3 --anchor vertex --half-width 5e-4 \
    --warm-start results/learned/spectrum-cnn-wide/test/predictions.csv
# learned vs. raw and offset-corrected baseline (and the warm start) on the same samples
holofocus-learn compare --learned results/learned/spectrum-cnn-wide/test/predictions.csv \
    --baseline results/baseline/cpu-4x-test --baseline-val results/baseline/cpu-4x-val \
    --warm results/baseline/cpu-4x-test-warm --out results/comparison/cpu-4x-test
# study of plan step 11 (learning curve, seeds, loss and width variants, image CNN on 2000
# and 10000 samples); every run takes the first n samples of the concatenated files
python scripts/run_stage4_study.py --train data/cpu-4x/train.h5 data/cpu-4x-extra/train.h5 \
    --val data/cpu-4x/val.h5 --threads 1
# padding check: MFE scans around the true z01 of test samples 0 and 1 at padding 2 and 4
# (sample 2 is in results/bias and results/padding/bias-p4 from Stage 3)
for p in 2 4; do
    python scripts/run_stage3_experiments.py --threads 1 bias --data data/cpu-4x/test.h5 \
        --indices 0 1 --variants noisy --padding $p --out results/padding/bias-p$p
done
# figures and stage4-metrics.json from results/
python scripts/make_stage4_figures.py
```

`--device cuda` (or `device: cuda` in the config) trains and predicts on a GPU; `auto` takes
CUDA when it is available. Several `--train` files are concatenated in the given order, and
`--limit N` keeps the first `N` samples of the concatenation. `--warm-start` reads `index`,
`z01_est` and, with `--sigmas k`, `sigma` from a `predictions.csv`; the half width of the
initial simplex is `max(--half-width, k * sigma)`, shifted inside `Omega_z` if needed.
`--anchor vertex` starts from `(z01_est, z01_est + w)` instead (`z01_est - w` at the upper end
of `Omega_z`), so the MFE at the estimate is always evaluated; the default `center` keeps
`z01_est ± w`. The source file, its SHA-256, the widths and a non-default anchor belong to the
run key, so a run only resumes with the same warm start.

A training run directory holds `config.yaml`, `history.csv` (loss, validation MAE, RMSE, bias,
NLL and 68 % coverage per epoch), `model.pt` (weights of the selected epoch, config, field of
view, `sigma_scale`), `summary.json` (validation metrics) and `run.yaml` (datasets with SHA-256
and generator commit, sizes, timing, software, host). `predict` writes `predictions.csv` (one
row per sample: true and estimated `dz01`, `z01_est`, `error_um`, calibrated and raw `sigma`,
`z_score`, true and estimated Fresnel number, seconds), `summary.json` (metrics overall, per
phantom type and per flat-field count range) and `run.yaml`. `compare` writes
`comparison.csv` (per sample: learned, raw, corrected and warm-start errors, reconstructions
and seconds) and `comparison.json` (metrics per method, the offset with its standard error and
bootstrap interval, paired differences of the absolute errors).

The time per estimate is that of one hologram (batch 1, 1 thread) including the spectrum,
after a warm-up call. The `results/learned/` directory holds:

```
results/learned/spectrum-cnn-wide/  main model (10000 training samples) with model.pt, val/, test/
results/learned/spectrum-cnn/    the standard width on the same samples, with model.pt, val/, test/
results/learned/study/<run>/     runs of scripts/run_stage4_study.py with val/ (no weights)
results/baseline/cpu-4x-val/     baseline on the first validation samples (offset)
results/baseline/cpu-4x-test-warm/  warm-started baseline on the 48 baseline test samples
results/baseline/cpu-4x-test-warm-vertex/  the same with --anchor vertex on samples 30, 35, 40, 41
results/comparison/cpu-4x-test/  learned vs. baseline on those samples
results/padding/bias-p2/         MFE scans of test samples 0 and 1 at padding 2 (padding check;
                                 bias-p4/ holds them at padding 4)
```

## Comparison with the paper code (optional)

The code of [1] is published on Zenodo (v1.0.0, `livereco_server`) and maintained as
`holowizard` on PyPI. holowizard pins Python 3.11 and a large dependency set, so it lives in
its own environment:

```bash
uv python install 3.11 && uv venv --python 3.11 /tmp/hw/venv311
VIRTUAL_ENV=/tmp/hw/venv311 uv pip install holowizard==3.0.6   # 162 packages, ~9 GB
VIRTUAL_ENV=/tmp/hw/venv311 uv pip install --no-deps -e .
/tmp/hw/venv311/bin/python scripts/crosscheck_holowizard.py      # Fr and propagator agree
/tmp/hw/venv311/bin/python scripts/holowizard_asrm_cpu.py --data spider_hair.tiff
```

`spider_hair.tiff` is in `Examples/data/holograms/` of the Zenodo archive
(`livereco_server-v1_0_0.zip`). The ASRM run (16x and 4x stages) takes about 7 min on 4 CPU
threads.

The Stage 3 port is checked against holowizard in the same environment:

```bash
/tmp/hw/venv311/bin/python scripts/crosscheck_asrm_holowizard.py reconstruct \
    --data data/cpu-4x/test.h5 --index 0 --out results/crosscheck
/tmp/hw/venv311/bin/python scripts/crosscheck_asrm_holowizard.py focus \
    --data data/cpu-4x/test.h5 --index 0 --iteration-scale 0.1 --threads 1 --out results/crosscheck
```

`reconstruct` runs the full [2] Table 3 schedule with both codes on the same hologram and
compares the loss of every iteration and the object; `focus` runs both autofocus searches
and compares the probing points.
