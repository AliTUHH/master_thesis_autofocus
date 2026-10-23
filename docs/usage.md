# Usage

## Installation

Python ≥ 3.10. PyTorch is installed separately so that you can choose the CPU or the CUDA
build.

```bash
python -m venv .venv && source .venv/bin/activate          # or: conda create -n holofocus python=3.12
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU only (VM, CI)
# pip install torch --index-url https://download.pytorch.org/whl/cu126  # GPU machine
# (DESY Maxwell: cluster/maxwell/setup_env.sh, see docs/maxwell-gpu.md)
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
storage options and the size of every split. The provided configs:

| Config | Detector grid | `z01` | Samples (train/val/test) | CPU time | Size |
|---|---|---|---|---|---|
| `configs/dataset/cpu-4x.yaml` | P05 binned 4x: 512² px of 26 µm, simulated 2x finer | 100 mm ± 5 mm | 2000/250/250 | ~16 min | ~1.8 GB |
| `configs/dataset/cpu-4x-z01-range.yaml` | as above | guess 80–300 mm, ± 5 mm | 2000/250/250 | ~23 min | ~1.8 GB |
| `configs/dataset/cpu-4x-extra.yaml` | as above, another seed, no phase maps (Stage 4) | 100 mm ± 5 mm | 8000/–/– | ~60 min | ~5.4 GB |
| `configs/dataset/cpu-4x-large.yaml` | as above, a third seed, no phase maps (for the GPU) | 100 mm ± 5 mm | 50 000/–/– | ~5–8 h (estimated) | ~34 GB |
| `configs/dataset/ci.yaml` | binned 16x: 128² px of 104 µm | 100 mm ± 5 mm | 32/8/8 | < 1 s | ~4 MB |
| `configs/dataset/matched-*.yaml` | the setups of the four measured holograms (Stage 5) | see [Measured holograms](#measured-holograms-stage-5) | | | |

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
| `configs/baseline/paper2-cpu-warm.yaml` | `cpu-4x` | as above | as above, initial simplex from `--warm-start` (Stage 4), no reconstruction at the true `z01` |
| `configs/baseline/paper2-cpu-p4.yaml`, `paper2-cpu-warm-p4.yaml` | `cpu-4x` | as above with padding 4 (2048² arrays, about 190 s per reconstruction on CPU; for the GPU) | as the two above |
| `configs/baseline/ci.yaml` | `ci` (128², binning 16) | Table 3 with 20× fewer iterations | ± 5 mm, stop at 1 mm |

```bash
holofocus-baseline run configs/baseline/paper2-cpu.yaml --data data/cpu-4x/test.h5 \
    --out results/baseline/cpu-4x-test --limit 48 --threads 3
holofocus-baseline run configs/baseline/paper2-cpu.yaml --data data/cpu-4x/test.h5 \
    --out runs/subset --indices 3 5 8 --device cuda
holofocus-baseline summary results/baseline/cpu-4x-test
# a split in 10 contiguous chunks (e.g. the tasks of a job array), then one run directory
for k in $(seq 0 9); do
    holofocus-baseline run configs/baseline/paper2-cpu.yaml --data data/cpu-4x/test.h5 \
        --out runs/test/chunk-$k-of-10 --chunk $k --chunks 10
done
holofocus-baseline merge runs/test/chunk-*-of-10 --out runs/test
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

`--chunk k --chunks n` keeps the k-th (from 0) of `n` contiguous parts of the selected samples
(after `--indices` and `--limit`; the sizes differ by at most one). `merge` joins runs with the
same config, dataset and warm start: `results.csv` and `traces.jsonl` sorted by `index`, a new
`summary.json`, and the `run.yaml` of the first run with `merged_from` (path, number of
samples, software, device and host of every run). A sample in two runs is refused; `--out` may
contain the runs, and an earlier merge there is replaced. The Maxwell kit runs the chunks as a
SLURM job array ([maxwell-gpu.md](maxwell-gpu.md), German).

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
mean and standard deviation. Three input representations are implemented:

| `representation.kind` | Input | Model |
|---|---|---|
| `spectrum` | radially averaged power spectrum of the Hann-windowed hologram on a uniform `f^2` axis (1024 bins up to 0.125 cycles²/px²), log, standardized; its minima lie at `f^2 = k Fr` | `SpectrumCNN` (1-D convolutions) |
| `image` | the hologram binned 2×2, standardized; random D4 transforms (rotations by 90°, flips) during training | `ImageCNN` (2-D convolutions) |
| `polar` (Stage 10) | the same power spectrum in `angles` sectors (8) of the direction of the frequency, modulo 180°, times the bins of `f^2` | `PolarSpectrumCNN` (invariant to rotations by 90° and flips) |

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
| `configs/learned/polar-cnn.yaml` | polar, 8 sectors × 1024 bins | the stages of `spectrum-cnn-wide.yaml` after an angular first stage, 465k parameters, 100 epochs (Stage 10) |
| `configs/learned/polar-cnn-hf.yaml`, `image-cnn-b1.yaml` | polar, 2048 bins up to 0.25; image, binning 1 (512²) | as `polar-cnn.yaml` and `image-cnn.yaml`; for the GPU (Stage 10) |
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
`--limit N` keeps the first `N` samples of the concatenation. `--seed k` replaces
`training.seed` of the config (the copy in the run directory records it), e.g. for an
ensemble of seeds. `--warm-start` reads `index`,
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

## Measured holograms (Stage 5)

The four holograms of the Zenodo record of [1] (10.5281/zenodo.8349365,
`Examples/data/holograms`, called `$H` below) each get a spectrum CNN trained on data with
their energy, geometry and detector path (`MEASURED_SAMPLES`): the guess is the centre of the
initial simplex of [2] Table 1 (as in the Stage 3 runs), `dz01` uniform in ±5 mm, the P05
detector binned 4x by the block mean (the tooth without its 80 px cutting band). Phantoms,
noise and numerics are those of `cpu-4x`.

| Config | Sample | Guess | Field of view / simulation | Samples (train/val/test) | CPU time, 3 threads | Size |
|---|---|---|---|---|---|---|
| `matched-spider_hair.yaml` | spider hair, 11 keV | 79.8 mm | 512² / 4480² | 6000/250/250 | 72 min | 4.4 GB |
| `matched-tooth.yaml` | tooth, 17 keV | 82.9 mm | 472² / 3087² | 6000/250/250 | 34 min | 3.8 GB |
| `matched-cactus_needle.yaml` | cactus needle, 17 keV | 286.1 mm | 512² / 1620² | 6000/250/250 | 16 min | 4.3 GB |
| `matched-magnesium_wire.yaml` | Mg wire, 11 keV | 472.4 mm | 512² / 1568² | 6000/250/250 | 13 min | 4.3 GB |
| `matched-<sample>-large.yaml` | as above, own seeds (for Maxwell, [maxwell-gpu.md](maxwell-gpu.md) 7.1) | | | 50 000/–/– | | 29–34 GB |
| `matched-<sample>-realistic.yaml` | cactus needle and Mg wire with the measured detector blur, noise, flat-field residual and object as ranges | | | 6000/250/250 | 28 min (cactus needle), Mg wire 38 min with 1 thread | 4.7–4.8 GB |

`holofocus.learned.sim2real` holds what the measured holograms show about these assumptions
(`MEASURED_MODELS`, `PSF_SIGMA`, `FLAT_FIELD_CUTOFF`), the perturbations of a test split that
change one of them (`PERTURBATIONS`; same seed, so every perturbed sample pairs with the
original) and `realistic_changes`, the changes of the realistic sets.

```bash
for s in spider_hair tooth cactus_needle magnesium_wire; do
    holofocus-dataset generate configs/dataset/matched-$s.yaml --out data/matched-$s
    holofocus-learn train configs/learned/spectrum-cnn-wide.yaml --train data/matched-$s/train.h5 \
        --val data/matched-$s/val.h5 --out results/learned/matched-$s --threads 1
    for split in test val; do
        holofocus-learn predict results/learned/matched-$s --data data/matched-$s/$split.h5 \
            --out results/learned/matched-$s/$split --threads 1
    done
done
# cactus needle and Mg wire: the spectrum up to f^2 = 0.25 as a variant (-hf), and the
# realistic set; the realistic model is also predicted on the matched test split
for s in cactus_needle magnesium_wire; do
    holofocus-learn train configs/learned/spectrum-cnn-wide-hf.yaml \
        --train data/matched-$s/train.h5 --val data/matched-$s/val.h5 \
        --out results/learned/matched-$s-hf --threads 1
    holofocus-dataset generate configs/dataset/matched-$s-realistic.yaml \
        --out data/matched-$s-realistic
    holofocus-learn train configs/learned/spectrum-cnn-wide.yaml \
        --train data/matched-$s-realistic/train.h5 --val data/matched-$s-realistic/val.h5 \
        --out results/learned/matched-$s-realistic --threads 1
    holofocus-learn predict results/learned/matched-$s-realistic \
        --data data/matched-$s/test.h5 --out results/learned/matched-$s-realistic/matched-test
done   # and predict test/ and val/ of every run as above
# cold padding-4 search of Stage 3 for the two samples that had none
python scripts/run_stage3_experiments.py --threads 3 measured --holograms $H --binning 1 \
    --padding 4 --samples cactus_needle magnesium_wire --out results/measured/b1-p4
# the CNNs on the measured holograms: every variant, the one with the lower validation MAE
# is the estimate; references [2], [1] and the cold baselines; 16 binning phases; timing
python scripts/run_stage5_study.py --threads 1 measured --holograms $H
# padding-4 search from the CNN estimate +- 0.5 mm (records the SHA-256 of its source, so
# run it after the last `measured`)
python scripts/run_stage3_experiments.py --threads 3 measured --holograms $H --binning 1 \
    --padding 4 --warm-start results/measured/cnn --out results/measured/warm-b1-p4
# sim-to-real gap: perturbed test splits (data/sim2real/, not tracked), paired shifts
python scripts/run_stage5_study.py --threads 1 perturb
for s in cactus_needle magnesium_wire; do   # the same perturbed splits, realistic model
    python scripts/run_stage5_study.py --threads 1 perturb --samples $s \
        --model results/learned/matched-$s-realistic \
        --reference results/learned/matched-$s-realistic/matched-test/predictions.csv \
        --out results/sim2real/realistic
done
python scripts/run_stage5_study.py --threads 1 spectra --holograms $H
python scripts/run_stage5_study.py --threads 1 noise --holograms $H
python scripts/run_stage5_study.py compare
# figures and stage5-metrics.json
python scripts/make_stage5_figures.py --holograms $H
```

`perturb` reuses perturbed splits and predictions that still match their config and model.
The results:

```
results/learned/matched-<sample>/        CNN per measured setup with model.pt, val/, test/
results/learned/matched-<sample>-hf/     spectrum up to f^2 = 0.25 (cactus needle, Mg wire)
results/learned/matched-<sample>-realistic/  CNN of the realistic set, also matched-test/
results/datasets/matched-*-{val,test}.csv  fingerprints of the val and test splits
results/measured/cnn/<sample>.json       CNN estimates with intervals, references, timing
results/measured/b1-p4/, warm-b1-p4/     padding-4 searches, cold and from the CNN
results/measured/comparison.{csv,json}   all estimates per sample and the matched test errors
results/sim2real/<sample>.{json,csv}     perturbations: paired shifts, errors, coverage
results/sim2real/<sample>/<perturbation>/  predictions on each perturbed test split
results/sim2real/realistic/              the same for the models of the realistic sets
results/sim2real/spectra-<sample>.json   spectral features, measured against simulated
results/sim2real/noise.json              noise spectra of the background corners and fits
```

## Object model (Stage 6)

The fibers of Stage 5 give spectra without the CTF modulation of the measured holograms.
Stage 6 adds a weak-phase texture inside the object (`phantoms.texture_rms_range`) and as a
film over the field of view (`background_texture_rms_range`): a Gaussian random field with
the power `(1 + (f / texture_cutoff)^2)^(-texture_slope / 2)` from its own random stream, so
configs without texture give the same holograms as before. `holofocus.learned.objectmodel`
locates the modulation in each measured hologram (`region_weights`, `region_ctf`), fits the
texture of each hologram on the spectral features of 16 test samples (`fit_objective`,
`TEXTURE_FITS`) and builds the leave-one-out training sets: every property estimated from a
measured hologram (detector, bulk object, texture) becomes a range over the other three
samples, widened by a factor of 2 (`loo_ranges`, `loo_changes`), so a set never saw the
hologram it is tested on.

The object model lives in the dataset config alone: the `phantoms` keys `texture_rms_range`,
`background_texture_rms_range`, `texture_slope_range`, `texture_cutoff` and
`phase_per_pixel_range` (fibers only) switch it on in any config. The leave-one-out sets are
ordinary splits, so every command that takes a split runs on them unchanged, the NPE of
Stage 7 included (`npe-train --train data/loo-<sample>/train.h5 --val data/loo-<sample>/val.h5`;
`run_stage7_study.py` takes the roots as `--data`, `--npe` and `--cnn-learned` and expects the
names `matched-<sample>` below them; its `measured` reads the CNN to compare with from
`cnn/<sample>.json` below `--measured`, the format of `results/measured/cnn-loo/<sample>.json`).
`scripts/run_stage7_loo_study.py` does this for the leave-one-out sets, with the leave-one-out
CNNs and their seeds (see "Stage 7 on the object model" below).

| Config | Sample | Samples (train/val/test) | CPU time | Size |
|---|---|---|---|---|
| `loo-spider_hair.yaml` | spider hair, ranges from tooth, cactus needle, Mg wire | 6000/250/250 | 3.8 h, 3 threads (val, test 2) | 5.1 GB |
| `loo-tooth.yaml` | tooth | 6000/250/250 | 3.1 h, 1 thread | 4.3 GB |
| `loo-cactus_needle.yaml` | cactus needle | 6000/250/250 | 40 min, 1 thread | 5.0 GB |
| `loo-magnesium_wire.yaml` | Mg wire | 6000/250/250 | 37 min, 1 thread | 4.9 GB |
| `loo-<sample>-large.yaml` | as above, own seeds (for Maxwell, [maxwell-gpu.md](maxwell-gpu.md) 7.2) | 50 000/–/– | | 33–39 GB |

```bash
# where the CTF modulation comes from (needs the phase files of results/measured/b1-p4)
python scripts/run_stage6_study.py regions --holograms $H
# texture fits, 16 test samples per texture (resumable)
python scripts/run_stage6_study.py --threads 2 fit --holograms $H
# leave-one-out configs from the fits (configs/dataset/loo-*.yaml, loo-ranges.json)
python scripts/run_stage6_study.py configs
for s in spider_hair tooth cactus_needle magnesium_wire; do
    holofocus-dataset generate configs/dataset/loo-$s.yaml --out data/loo-$s
    for split in val test; do
        python scripts/check_cluster_run.py fingerprint data/loo-$s/$split.h5 \
            --out results/datasets/loo-$s-$split.csv
    done
done
# the spectral check without training
python scripts/run_stage6_study.py spectra --holograms $H
for s in spider_hair tooth cactus_needle magnesium_wire; do
    holofocus-learn train configs/learned/spectrum-cnn-wide.yaml --train data/loo-$s/train.h5 \
        --val data/loo-$s/val.h5 --out results/learned/loo-$s --threads 1
    for split in test val; do
        holofocus-learn predict results/learned/loo-$s --data data/loo-$s/$split.h5 \
            --out results/learned/loo-$s/$split --threads 1
    done
    holofocus-learn predict results/learned/loo-$s --data data/matched-$s/test.h5 \
        --out results/learned/loo-$s/matched-test --threads 1
    holofocus-learn predict results/learned/matched-$s --data data/loo-$s/test.h5 \
        --out results/objectmodel/matched-model/$s --threads 1
done
# the leave-one-out CNNs on the measured holograms
python scripts/run_stage6_study.py --threads 1 measured --holograms $H
# the same CNNs with the training seeds 1 and 2, for the spread of the estimates
for k in 1 2; do
    for s in spider_hair tooth cactus_needle magnesium_wire; do
        out=results/objectmodel/seeds/seed-$k/loo-$s
        holofocus-learn train configs/learned/spectrum-cnn-wide.yaml --train data/loo-$s/train.h5 \
            --val data/loo-$s/val.h5 --out $out --seed $k --threads 1
        holofocus-learn predict $out --data data/loo-$s/test.h5 --out $out/test --threads 1
    done
    python scripts/run_stage6_study.py --threads 1 measured --holograms $H \
        --learned results/objectmodel/seeds/seed-$k --out results/objectmodel/seeds/seed-$k/measured
done
# side by side with Stage 5, [2] and padding 4
python scripts/run_stage6_study.py compare
python scripts/make_stage6_figures.py --holograms $H
```

The results:

```
results/objectmodel/regions.json         CTF correlation of object, edge and background
results/objectmodel/fit-<sample>.{json,csv}  texture fit: every texture tried and its features
results/objectmodel/loo-ranges.json      leave-one-out ranges and the sample's own values
results/objectmodel/spectra-<sample>.json  spectral features, measured against simulated
results/learned/loo-<sample>/            CNN of the leave-one-out set with model.pt, val/,
                                         test/ and matched-test/ (Stage 5 test split)
results/objectmodel/matched-model/<sample>/  the Stage 5 CNN on the leave-one-out test split
results/objectmodel/seeds/seed-<k>/      the leave-one-out CNNs trained with the seed k (without
                                         model.pt), test predictions and measured/ estimates
results/datasets/loo-*-{val,test}.csv    fingerprints of the val and test splits
results/measured/cnn-loo/<sample>.json   estimates of the leave-one-out CNNs
results/objectmodel/comparison.{csv,json}  leave-one-out CNN, Stage 5 CNN, [2], padding 4
```

## Neural posterior estimation (Stage 7)

`holofocus.learned.npe` estimates the whole posterior of `dz01` given a hologram with sbi
(neural posterior estimation, [6]) instead of a Gaussian mean and standard deviation. The
spectrum representation of Stage 4 goes through the convolution stack of `SpectrumCNN` and an
MLP to a 32-dimensional embedding; a conditional density estimator of sbi on it gives
`q(dz01 | x)` on the prior box (uniform, the `dz01` range of the training data). Training keeps
the epoch with the lowest validation loss `-log q(dz01_true | x)`. The posterior is evaluated
on a grid of 2001 points: the median is the point estimate, with central and HPD intervals and,
for calibration, the PIT (posterior CDF at the truth, the SBC rank) and the HPD level of the
truth (expected coverage). Several runs form an ensemble whose densities are averaged.

| Config | Density estimator | Notes |
|---|---|---|
| `configs/learned/npe-nsf.yaml` | neural spline flow (5 transforms, 10 bins) | 482k parameters, 100 epochs, batch 32 |
| `configs/learned/npe-mdn.yaml` | mixture density network (10 Gaussians) | variant |
| `configs/learned/npe-ci.yaml` | small flow, 128 bins | 3 epochs (CI) |

The NPE needs the extra `npe` (sbi 0.27; `holofocus.learned.npe` uses internal functions of
it). `holofocus.learned.posterior` (grid posterior and calibration statistics) does not.

```bash
pip install -e ".[dev,npe]"
```

```python
from holofocus.learned.npe import NPEAutofocus

autofocus = NPEAutofocus.load("results/npe/npe-nsf")  # or a list of runs: the ensemble
post = autofocus.posterior(holograms)  # GridPosterior of t = dz01 / 5 mm
est = autofocus.estimate(holograms)  # dz01 (median), dz01_mean, sigma, lower_68, upper_95, ...
```

```bash
# calibration split: 2000 prior draws with the physics of cpu-4x (16 min with 3 threads, 1.4 GB)
holofocus-dataset generate configs/dataset/cpu-4x-calibration.yaml --out data/cpu-4x-calibration
# the NPE on the 10000 training samples of Stage 4, four more seeds and the MDN variant
train="--train data/cpu-4x/train.h5 data/cpu-4x-extra/train.h5 --val data/cpu-4x/val.h5"
holofocus-learn npe-train configs/learned/npe-nsf.yaml $train --out results/npe/npe-nsf --threads 1
for k in 1 2 3 4; do
    holofocus-learn npe-train configs/learned/npe-nsf.yaml $train --seed $k \
        --out results/npe/npe-nsf-seed$k --threads 3
done
holofocus-learn npe-train configs/learned/npe-mdn.yaml $train --out results/npe/npe-mdn --threads 3
# one NPE per measured setup, on the matched data of Stage 5
for s in spider_hair tooth cactus_needle magnesium_wire; do
    holofocus-learn npe-train configs/learned/npe-nsf.yaml --train data/matched-$s/train.h5 \
        --val data/matched-$s/val.h5 --out results/npe/matched-$s --threads 3
done
# single commands: posterior summaries of a split (several runs: the ensemble) and sbi's SBC,
# expected coverage and TARP from posterior samples
holofocus-learn npe-predict results/npe/npe-nsf --data data/cpu-4x/test.h5 --out /tmp/npe-test
holofocus-learn npe-check results/npe/npe-nsf --data data/cpu-4x-calibration/calibration.h5 \
    --out /tmp/npe-check
# the study of Laborbuch 07 and its figures
python scripts/run_stage7_study.py --threads 3 predict    # all runs, the ensemble, CNN on calibration
python scripts/run_stage7_study.py --threads 3 sbc --runs npe-nsf npe-mdn
python scripts/run_stage7_study.py compare               # test, 48 baseline samples, calibration
python scripts/run_stage7_study.py --threads 1 timing
python scripts/run_stage7_study.py examples
python scripts/run_stage7_study.py --threads 3 perturb    # perturbed test splits of Stage 5
python scripts/run_stage7_study.py --threads 1 measured --holograms $H
python scripts/make_stage7_figures.py
```

The study takes other runs and data unchanged (the object model of Stage 6, Maxwell runs,
[maxwell-gpu.md](maxwell-gpu.md) 8.1): `--npe` (the outputs go below it), `--cnn` and the split
options of each command. The results:

```
results/npe/<run>/                       model.pt, config.yaml, history.csv, summary.json, run.yaml
results/npe/<run>/{val,test,calibration}/  posterior summaries per sample (predictions.csv)
results/npe/<run>/calibration/sbi-*      sbi's ranks, SBC, expected coverage and TARP
results/npe/npe-nsf-ensemble/            the five seeds as one posterior (test, calibration)
results/npe/comparison/                  NPE against the CNN: test, 48 baseline samples, calibration;
                                         posteriors.json: the posteriors of a few test samples
results/npe/timing.json                  time per estimate, CNN against NPE, same threads
results/npe/sim2real/<sample>.{json,csv}  matched NPE and CNN on the perturbed test splits
results/npe/measured/<sample>.json       the matched NPEs on the four measured holograms
results/learned/spectrum-cnn-wide/calibration/  the Stage 4 CNN on the calibration split
results/datasets/cpu-4x-calibration-calibration.csv  fingerprint of the calibration split
```

### Stage 7 on the object model

The same NPE (`configs/learned/npe-nsf.yaml`, unchanged) on the leave-one-out sets of Stage 6,
with the training seeds of the leave-one-out CNNs (0, 1 and 2) and the ensemble of the three
seeds (Laborbuch 07, Abschnitt 9). `scripts/run_stage7_loo_study.py` compares them on the
leave-one-out test splits with the leave-one-out CNNs and with the guess, whose intervals are
those of the uniform prior, and on the measured holograms with [2], padding 4 and the
leave-one-out CNNs (`results/measured/cnn-loo`). The data are the sets of Stage 6
(`data/loo-<sample>`, see above).

```bash
for s in spider_hair tooth cactus_needle magnesium_wire; do
    loo="--train data/loo-$s/train.h5 --val data/loo-$s/val.h5"
    holofocus-learn npe-train configs/learned/npe-nsf.yaml $loo --out results/npe-loo/matched-$s \
        --threads 3
    for k in 1 2; do
        holofocus-learn npe-train configs/learned/npe-nsf.yaml $loo --seed $k \
            --out results/npe-loo/matched-$s-seed$k --threads 3
    done
done
python scripts/run_stage7_loo_study.py --threads 3 predict   # every run on val and test, ensemble on test
python scripts/run_stage7_loo_study.py compare               # NPE, ensemble, CNN and guess on test
python scripts/run_stage7_loo_study.py --threads 1 measured --holograms $H
python scripts/run_stage7_loo_study.py --threads 1 timing
python scripts/make_stage7_loo_figures.py
```

Other runs, such as the large sets of [maxwell-gpu.md](maxwell-gpu.md) 8.2, take `--npe`,
`--cnn`, `--cnn-seeds` and `--seeds`; the outputs go below `--npe`. The results:

```
results/npe-loo/matched-<sample>[-seed<k>]/  NPE run (model.pt, history.csv, ...) with val/, test/
results/npe-loo/matched-<sample>-ensemble/test/  the three seeds as one posterior
results/npe-loo/comparison/              NPE seeds, ensemble, CNN seeds and guess on the test splits
results/npe-loo/measured/<sample>.json   seeds and ensemble on the four measured holograms
results/npe-loo/timing.json              time per estimate, CNN against NPE and ensemble
```

## Stage 8: misspecification, ensembles, warm start

Three checks on the leave-one-out sets of Stage 6 with the models of Stage 7 (Laborbuch 08),
all on the CPU. `scripts/run_stage8_study.py`:

- `sources`: the ensemble of the three leave-one-out CNNs (the equal-weight mixture of their
  Gaussians, `holofocus.learned.posterior.GaussianMixture`) and the NPE ensemble of Stage 7 on
  the measured holograms, with central intervals; reads only committed records.
- `misspec`: the distance of each measured hologram (and its 16 binning phases) to its
  leave-one-out training set, in the network input and in the embedding of each NPE seed
  (`holofocus.learned.misspecification`: mean distance to the k nearest training samples,
  Mahalanobis distance in principal components), as a percentile among the test split; on the
  test split, how the distances go with the errors, widths and seed spread of the two
  ensembles; the 25 test samples nearest to the hologram; and, with `--control results/npe`,
  the same distances to the sets of the Stage 5 simulator (`data/matched-<sample>`) as a
  positive control, with the failures of the NPE trained on them on its test split, on the
  hologram and among the hologram's nearest test samples. It needs `data/loo-<sample>` (and
  `data/matched-<sample>` for the control); `--cache DIR` keeps the network inputs of the
  splits.
- `ensembles`: CNN ensemble against NPE ensemble on the leave-one-out test splits, with the
  seeds and the guess, per probe and pooled, and on the measured holograms.
- `warmstart`: the padding-4 searches on the measured holograms side by side, each also on the
  parabola about its result (`holofocus.autofocus.search_points`: the points `find_focus`
  probes on a given MFE curve, without reconstructing), and the reconstructions on a parabola
  against the width of the start interval and the distance of a warm start; for comparison,
  the evaluations of Brent's bounded method (SciPy) on model curves from the same intervals,
  and how the MFE of each probe rises from its minimum, from the points of all its searches
  with the thread count of the full search (with one thread instead of three, the same
  reconstruction gives an MFE that differs by a few ppm, at a step of the tooth by 675 ppm).

The searches are `run_stage3_experiments.py measured` with `--warm-start` (start from the
estimate +- 0.5 mm) or `--box` (search only inside the central interval `--box-level`, default
95 %, cut to `Omega_z`), both with `--revisit-tolerance 1e-9` (points within 1 nm of an earlier
one count as repeated). With 3 threads a search takes 30–45 min per probe (8–12
reconstructions of about 3.5 min), all four probes about 2 to 2.5 h.

```bash
python scripts/run_stage8_study.py sources
for e in cnn npe; do
    for mode in warm-start box; do
        out=results/measured/$([ $mode = box ] && echo box95 || echo warm)-$e-loo-ensemble-b1-p4
        python scripts/run_stage3_experiments.py --threads 3 measured --holograms $H \
            --binning 1 --padding 4 --revisit-tolerance 1e-9 \
            --$mode results/measured/$e-loo-ensemble --out $out
    done
done
python scripts/run_stage8_study.py --threads 3 misspec --holograms $H --control results/npe \
    --cache /tmp/stage8-inputs
python scripts/run_stage8_study.py ensembles
python scripts/run_stage8_study.py warmstart
python scripts/make_stage8_figures.py
```

`misspec` and `ensembles` take the runs as options (`--npe`, `--cnn`, `--cnn-seeds`, `--seeds`,
`--samples`, `--data`), so the large Maxwell runs go through the same commands. The results:

```
results/measured/{cnn,npe}-loo-ensemble/<sample>.json   the two ensembles on the holograms
results/misspecification/<sample>.json   distances, percentiles, failures, neighbours, control
results/misspecification/test-<sample>.csv  per test sample: distances, errors, widths, spread
results/misspecification/summary.csv     one row per probe, space and distance
results/ensembles/                       CNN ensemble against NPE ensemble (test, measured)
results/measured/{warm,box95}-{cnn,npe}-loo-ensemble-b1-p4/  the searches
results/measured/warmstart.{json,csv}    all searches against the full one
```

## Stage 9: steps of the MFE, Brent next to Nelder–Mead

Two parts (Laborbuch 09), both on the CPU.

**The MFE and the thread count.** Since Stage 9 a reconstruction gives the bits of a run with
one thread for every `torch.set_num_threads` (`holofocus.device.one_thread`: the products of
complex arrays, the norm of `Im(y)` and the loss run on one thread; about 6 % slower with 3
threads). Reconstructions made before with one thread keep their MFE bit for bit; those made
with 3 threads (most searches on the measured holograms, the Stage 3 baseline) differ by a few
to a few tens of ppm; where the two numerics reach different solutions of the reconstruction,
by about 700 ppm (the tooth at its step) and by 2–3 % (two points of synthetic sample 5).
Compare MFE values of older runs only within one thread count; `steps` labels every
reconstruction with its numerics.

**Brent's bounded method.** `find_focus(..., method="brent")` searches with
`scipy.optimize.minimize_scalar(method="bounded")` instead of Nelder–Mead, in the same interval,
with the same `xatol`. In a baseline config it is `search: {method: brent}`
(`configs/baseline/paper2-cpu-brent.yaml`), for the measured holograms
`run_stage3_experiments.py measured --method brent`. `holofocus-baseline run --box CSV
[--box-level 95]` searches every sample inside its own interval, cut to `Omega_z`: the CSV has
the columns `index`, `z01_lower_<level>` and `z01_upper_<level>` in metres, like the
`predictions.csv` of an NPE.

```bash
holofocus-baseline run configs/baseline/paper2-cpu-brent.yaml --data data/cpu-4x/test.h5 \
    --out runs/baseline/cpu-4x-test-brent-box --limit 48 \
    --box results/npe/npe-nsf-ensemble/test/predictions.csv
```

`scripts/run_stage9_study.py`:

- `sources`: the central 68 and 95 % intervals of the Stage 4 CNN on the cpu-4x test split
  (`results/stage9/intervals-cnn-test.csv`); the NPE ensemble has them in its predictions.
- `queue` and `merge`: the 248 searches on the 48 synthetic baseline samples (Brent cold, Brent
  and Nelder–Mead in the 95 % intervals of the NPE ensemble and of the CNN, and the Stage 3
  search again on 8 samples), taken one at a time by any number of lanes; `merge` writes them
  to `results/baseline/cpu-4x-test-<search>` and which lane ran what to
  `results/stage9/lanes.csv`.
- `scan`: reconstructions of a measured hologram at given `z01` (`--z01-mm`, or `--z01` in
  metres to hit a recorded point exactly) with the settings of the padding-4 searches: MFE,
  loss history, losses at the ends of the stages, object statistics and the loss at the
  returned object (`object_loss`: one more iteration of the last stage from the object without
  momentum; the MFE is the loss at the lookahead of the last iteration) per point. Points
  already in `results/stage9/scan-<sample>.json` are skipped. The objects go to
  `results/stage9/scan-<sample>/*.npz` (not tracked). `--stages 1` runs only the first stage
  of the plan (about 15 s instead of 9 min with one thread; its loss is bit for bit
  `stage_losses[0]` of the full reconstruction) and keeps the padded arrays; write those scans
  to `results/stage9/first-stage`.
- `synthetic`: reconstructions of a synthetic sample (`--index`; the cpu-4x test split unless
  `--data`) at given `z01` in metres, as the baseline does them (`--config`, by default
  `configs/baseline/paper2-cpu.yaml`): MFE, loss history, losses at the ends of the stages and
  the smallest loss of the last stage per point, in `results/stage9/synthetic-<index>.json`;
  `--stages 1` as for `scan`, written to `results/stage9/first-stage`.
- `steps`: every padding-4 reconstruction of the measured holograms (all searches and scans)
  as MFE against `z01`, per probe and numerics, screened for steps (jumps between close
  neighbours, reversals away from the minimum), with the density of the points about the
  minimum, the same `z01` with the same and with the other numerics, where the loss histories
  of two scans part and the scan objects against each other; the loss at the returned object
  and the smallest loss of the last stage screened the same way; the first-stage scans, their
  branch (upper where the loss lies more than 5 % over the smallest within 1 mm) and where
  they switch, with the objects on both sides; for every reconstruction whose first stage is
  known its branch, and for every pair of neighbouring full reconstructions at most 2 µm apart
  on different branches the MFE and the losses of the stages upper over lower, as computed and
  at one `z01` (less the slope of the MFE on both branches; `branches`); on the synthetic
  samples the Stage 3 search against its repeat with this stage's numerics
  (`synthetic_repeat`) and, where `synthetic` ran with both numerics, the same `z01` side by
  side (`synthetic_points`) and the branches of the first stage (`synthetic_first_stage`).
- `compare`: Brent against Nelder–Mead, cold and in the learned intervals, on the measured
  holograms (against the references of [2] and the full search) and on the synthetic samples
  (against the truth): error, share within 100 µm, reconstructions, time at the 3-thread rate,
  and the reconstructions the interval saves with each method, paired over the samples; two
  searches paired as well (`pairs`: the difference of the absolute errors with a bootstrap
  interval, an exact McNemar test on the samples only one of them gets within 100 µm, and the
  difference of their reconstructions).

The runs of Laborbuch 09, on a 4-core machine (3 threads next to at most one 1-thread run):

```bash
H=/path/to/Examples/data/holograms      # Zenodo record 10.5281/zenodo.8349365
m="python scripts/run_stage3_experiments.py --threads 3 measured --holograms $H --binning 1 --padding 4"
$m --method brent --revisit-tolerance 1e-9 --out results/measured/brent-b1-p4
for e in cnn npe; do
    $m --method brent --revisit-tolerance 1e-9 --box results/measured/$e-loo-ensemble \
        --out results/measured/brent-box95-$e-loo-ensemble-b1-p4
done
$m --out results/measured/b1-p4-repeat
python scripts/run_stage9_study.py sources
python scripts/run_stage9_study.py --threads 3 queue --lane a    # next to: --threads 1 --lane b
python scripts/run_stage9_study.py merge
s="python scripts/run_stage9_study.py --threads 1 scan --holograms $H"
f="--stages 1 --out results/stage9/first-stage"
$s --sample tooth --z01-mm 80.451 80.56 80.6 80.62 80.65 80.665 80.68 80.694 80.696 80.698 \
    80.702 80.71 80.713 80.716 80.717 80.719 80.722 80.725 80.825 80.87 80.88 80.89 80.9 \
    80.901 80.902 80.91                       # about the step, and both sides of each switch
$s --sample spider_hair --z01-mm 79.272 79.48 79.6 79.62 79.64 79.66 79.68 79.7 79.8 79.95 80.11
$s --sample cactus_needle --z01-mm 284.755 285.116  # where the searches leave gaps (E11)
$s --sample tooth $f --z01-mm $(seq -f %.3f 80.500 0.010 81.100) \
    $(seq -f %.3f 80.700 0.001 80.730) $(seq -f %.3f 80.900 0.001 80.920)
$s --sample spider_hair $f --z01-mm $(seq -f %.1f 74.8 0.2 84.8)     # all of Omega_z
$s --sample tooth $f --z01-mm $(seq -f %.1f 77.9 0.2 87.9)
$s --sample cactus_needle $f --z01-mm $(seq -f %.1f 281.1 0.2 291.1)
$s --sample magnesium_wire $f --z01-mm $(seq -f %.1f 467.4 0.2 477.4)
points() {  # the z01 at which the searches with numerics $1 reconstructed the tooth
    python -c "import csv; print(*sorted({r['z01'] for r in csv.DictReader(open(
        'results/stage9/mfe-points.csv')) if r['sample'] == 'tooth' and r['numerics'] == '$1'
        and not r['source'].startswith('scan')}))"
}
$s --sample tooth $f --z01 $(points 1)        # the first stage at every point of the searches
$s --sample tooth --z01-mm 81.5 81.7 81.9 82.1 82.3 82.5 83.7 83.9 84.1 84.3 84.5 \
    --out results/stage9/omega                # the full MFE across the branches of that grid
$s --sample tooth $f --z01-mm 79.200 79.150 79.125 79.138 79.144 79.141 79.143 81.800 81.850 \
    81.825 81.838 81.844 81.847 81.849 81.848 84.000 83.950 83.925 83.938 83.931 83.928 83.929 \
    83.930                                    # three switches of that grid, halved to 1 um
$s --sample tooth --z01-mm 79.123 79.143 79.144 79.164 81.848 81.849 83.929 83.930 \
    --out results/stage9/omega
python scripts/run_stage9_study.py --threads 3 scan --holograms $H --sample tooth \
    --z01-mm 79.15 --out results/stage9/threads-3     # a point of the searches, with its history
Z5="0.104375 0.10359375 0.103125 0.10375"   # synthetic sample 5, where the numerics differ by 2-3 %
F5="$(python -c "print(*[round(0.095 + k * 2e-4, 7) for k in range(51)])")
    $(python -c "print(*[round(0.1033 + k * 2.5e-5, 8) for k in range(49)])") 0.1 0.1025 0.10375
    0.104375 0.103125 0.1040625 0.1034375 0.10390625 0.10359375 0.103828125"
python scripts/run_stage9_study.py --threads 3 synthetic --index 5 --z01 $Z5
python scripts/run_stage9_study.py --threads 3 synthetic --index 5 --z01 $F5 --stages 1 \
    --out results/stage9/first-stage          # Omega_z, 103.3-104.5 mm, the points of its searches
Z27="0.096171875 0.09617184868636408 0.0961188401002897"  # sample 27: two searches 0.03 um apart
python scripts/run_stage9_study.py --threads 3 synthetic --index 27 --z01 $Z27
python scripts/run_stage9_study.py --threads 3 synthetic --index 27 --z01 $Z27 --stages 1 \
    --out results/stage9/first-stage
python scripts/run_stage9_study.py steps
python scripts/run_stage9_study.py compare
python scripts/make_stage9_figures.py
```

Every record lists its points, so all of them can be computed again from it, into another
directory (points already in the record there are skipped; add `--stages 1` for the records in
`first-stage/`, and use `synthetic --index <index>` for `synthetic-<index>.json`):

```bash
r=results/stage9/scan-tooth.json
python scripts/run_stage9_study.py --threads 1 scan --holograms $H --sample tooth --out /tmp/again \
    --z01 $(python -c "import json; print(*[repr(p['z01']) for p in json.load(open('$r'))['points']])")
```

The scans with the library of before Stage 9 run in a checkout of `4349d6a`:

```bash
git worktree add /tmp/before 4349d6a && cp scripts/run_stage9_study.py /tmp/before/scripts/
b() { (cd /tmp/before && PYTHONPATH=src python scripts/run_stage9_study.py --threads 3 "$@"); }
o3=results/stage9/threads-3-4349d6a
f3=results/stage9/first-stage/threads-3-4349d6a
b scan --holograms $H --sample tooth --z01 0.08069255535982456 0.080694 0.07915000000000001 --out $o3
b scan --holograms $H --sample tooth --stages 1 --out $f3 \
    --z01-mm $(seq -f %.3f 80.660 0.002 80.700) $(seq -f %.3f 80.500 0.050 80.900)
b scan --holograms $H --sample tooth --stages 1 --out $f3 --z01 $(points 3)
D=$PWD/data/cpu-4x/test.h5                  # the synthetic records likewise, with --data
b synthetic --index 5 --data $D --z01 $Z5 --out $o3
b synthetic --index 5 --data $D --z01 $F5 --stages 1 --out $f3
b synthetic --index 27 --data $D --z01 $Z27 --out $o3
b synthetic --index 27 --data $D --z01 $Z27 --stages 1 --out $f3
cp -r /tmp/before/$o3 results/stage9/ && cp -r /tmp/before/$f3 results/stage9/first-stage/
```

The results:

```
results/stage9/scan-<sample>.json          scans with this stage's library, one thread
results/stage9/omega/                      full scans of the tooth over Omega_z, one thread
results/stage9/threads-3/                  the same points with 3 threads (the same bits)
results/stage9/threads-3-4349d6a/          the same points with the library before, 3 threads
results/stage9/first-stage/                the first stage only, both numerics (threads-3-4349d6a/)
results/stage9/synthetic-<index>.json      a synthetic sample at given z01 (also under first-stage/
                                           and both threads-3-4349d6a/)
results/stage9/steps.json, mfe-points.csv, steps-objects.json   the screening for steps
results/stage9/compare.json, measured.csv, synthetic.csv        Brent against Nelder–Mead
results/stage9/lanes.csv, intervals-cnn-test.{csv,json}
results/measured/brent-b1-p4/, brent-box95-{cnn,npe}-loo-ensemble-b1-p4/, b1-p4-repeat/
results/baseline/cpu-4x-test-{brent,brent-box95-npe-ensemble,box95-npe-ensemble,
                              brent-box95-cnn,box95-cnn,repeat}/
```

## Stage 10: beyond the radial power spectrum

Three parts (Laborbuch 10), all on the CPU; the full runs for a GPU are in
[maxwell-gpu.md](maxwell-gpu.md) 8.5.

**Texture fits with the levels of the spectrum.** `fit_objective(median, measured, levels=True)`
adds the absolute levels of the radial log power spectrum to the shape features of Stage 6: the
mean log power of the mid band, of the upper band of the CNN and of the highest frequencies
(`LEVEL_NAMES`), each difference over `LEVEL_SCALE`. `run_stage6_study.py fit --levels` runs the
texture grids of Stage 6 with this objective; `TEXTURE_FITS_LEVELS` holds the result.

**Inputs beyond the radial spectrum.** Four families, each trained with the seeds 0, 1 and 2
(`holofocus-learn train --seed k`) and combined into an ensemble:

| Family | Config | Ensemble |
|---|---|---|
| `radial-cnn` | `spectrum-cnn-wide.yaml` | equal-weight mixture of the Gaussians (calibrated `sigma`) |
| `npe` | `npe-nsf.yaml` | `holofocus-learn npe-predict` with the three runs |
| `polar-cnn` | `polar-cnn.yaml` | mixture |
| `image-cnn` | `image-cnn.yaml` | mixture |

`PolarSpectrumCNN` starts with a convolution over 3 sectors × `kernel_size` bins that is cyclic
and symmetric in the angle (`AngularConv`), takes the mean and the maximum over the sectors
(`AnglePool`: the maximum keeps the sectors across a fiber, the mean is the radial average) and
continues with the stages of `SpectrumCNN`. A mixture is written in the columns of the NPE
predictions (intervals, `pit`), so a CNN ensemble and an NPE ensemble go through the same
metrics and bound a search the same way.

**Brent in the learned interval.** The 95 % interval of each ensemble bounds Brent's method
(Stage 9: `holofocus-baseline run --box CSV --box-level 95`, `run_stage3_experiments.py measured
--method brent --box DIR`), on the 48 synthetic samples of Stage 3 and on the four measured
holograms. Cold Nelder–Mead (Stage 3) and cold Brent (Stage 9) are the references.

`scripts/run_jobs.py` runs the jobs of a plan on one machine within a budget of threads and
memory: the jobs whose inputs are done, by priority, with EASY backfilling (a job may jump the
queue only if it does not delay the first waiting one). Every job has a log, its exit code and
its peak memory in the state folder. A runner started again takes over the jobs that still run.
`forecast` plays the rest of the plan through and prints when each group ends, in UTC and in
Berlin time. `retry` releases a failed job.

`scripts/run_stage10_study.py`:

- `configs`: the leave-one-out configs of the texture fits with levels,
  `configs/dataset/loo2-<sample>.yaml` (6000/250/250 samples), and `loo2-<sample>-large.yaml`
  (50 000 more training samples, for the GPU); the ranges and each probe's own values in
  `results/objectmodel/loo2-ranges.json`.
- `plan`: every job of the stage for `run_jobs.py` (`runs/stage10/plan.json`). It covers the
  loo2 sets and their fingerprints, the trainings and predictions of the four families on
  cpu-4x and on the loo2 sets, the ensembles, the estimates on the measured holograms, the
  searches, `texture`, `spectra` and, once all others are done, `compare`. Seeds trained in
  earlier stages are reused (radial CNN seed 0, NPE seeds 0–2, image CNN seed 0).
- `ensembles`: the seeds of a CNN family on a test split as one mixture; `mixture` does the same
  for predictions given by path, e.g. seeds trained on Maxwell.
- `measured`: the seeds of a family trained on the loo2 set of a measured hologram, applied to
  that hologram (4× binned, with the 16 phases of the binning grid), and their ensemble. The
  ensemble record is the interval of the bounded search. `--runs` and `--out` take runs from
  elsewhere.
- `search` and `merge`: Brent in the 95 % interval of a family's ensemble on one synthetic sample
  (`runs/stage10/search/<family>/<index>`), and all of them joined into one baseline run.
- `texture`: the fit set of each measured hologram without texture and with the textures of
  Stages 6 and 10, against its spectrum (levels, CTF bands, the power the texture adds).
  `spectra`: the measured hologram among the leave-one-out test splits of Stages 6 and 10, bin
  by bin.
- `compare`: the four families side by side. Learned estimates: share within 100 µm (Wilson
  interval), MAE (bootstrap interval) and coverage of the 95 % interval, on the 48 synthetic
  samples, the full cpu-4x test split, the loo2 test splits and the measured holograms.
  Searches: the same metrics, reconstructions, the samples whose truth lies outside the
  interval, and the pairs of `SEARCH_PAIRS` sample by sample (each search in an interval
  against both cold searches, the polar and the image CNN against the radial CNN and the NPE,
  exact McNemar test).
- `evaluate`: the metrics of `compare` for predictions (`NAME=CSV[,CSV...]`, several CSVs are
  mixed), searches (`NAME=DIR[,CSV]`, the intervals from the `run.yaml` of the search, checked
  by SHA-256) and estimates on the measured holograms given by path, e.g. the runs of Maxwell;
  every two of them paired.

The runs of Laborbuch 10, on a 4-core machine:

```bash
H=/path/to/Examples/data/holograms      # Zenodo record 10.5281/zenodo.8349365
python scripts/run_stage6_study.py fit --holograms $H --levels --out results/objectmodel/levels
python scripts/run_stage10_study.py configs
python scripts/run_stage10_study.py plan --holograms $H
python scripts/run_jobs.py run runs/stage10/plan.json --state runs/stage10   # in tmux
python scripts/run_jobs.py forecast runs/stage10/plan.json --state runs/stage10
python scripts/make_stage10_figures.py
```

The plan has 388 jobs, 101.5 core-hours with a budget of 4 threads and 8 GB. The results:

```
results/objectmodel/levels/fit-<sample>.{json,csv}   the texture fits with levels
results/objectmodel/loo2-ranges.json                 the loo2 ranges and own values
results/datasets/loo2-<sample>-{val,test}.csv        fingerprints of the loo2 splits
results/stage10/cpu-4x/<family>/seed-<k>/            trainings on cpu-4x (models not tracked)
results/stage10/cpu-4x/<family>-ensemble/test/       the ensembles on the cpu-4x test split
results/stage10/loo2/<family>/seed-<k>/loo-<sample>/ the CNN trainings on loo2, and their
results/stage10/loo2/<family>/ensemble/loo-<sample>/ ensembles on the loo2 test splits
results/stage10/loo2/npe/matched-<sample>[-seed<k>][-ensemble]/   the NPE on loo2
results/stage10/baseline/cpu-4x-test-brent-box95-<family>/  Brent in the intervals (48 samples)
results/measured/<family>-loo2-ensemble/<sample>.json       the ensembles on the holograms
results/measured/brent-box95-<family>-loo2-ensemble-b1-p4/  Brent in their intervals
results/stage10/texture-<sample>.json, spectra-<sample>.json
results/stage10/compare.json, learned.csv, measured.csv, searches-{synthetic,measured}.csv
```

## GPU runs on DESY Maxwell

`cluster/maxwell/` is a SLURM kit for the A100 nodes of DESY Maxwell; [maxwell-gpu.md](maxwell-gpu.md)
(German) explains it step by step, including the settings to look up on Maxwell (`CHECK_ME` in
`config.example.sh`; the scripts refuse to run while one is left). In short:

```bash
cp cluster/maxwell/config.example.sh cluster/maxwell/config.sh   # replace every CHECK_ME
cluster/maxwell/setup_env.sh                 # venv or conda with CUDA PyTorch on scratch
cluster/maxwell/submit.sh gpu-check          # a GPU node against the CPU and the committed results
cluster/maxwell/submit.sh --local gpu-check  # the same without SLURM, with --device cpu
```

`scripts/check_cluster_run.py` compares runs made elsewhere with the committed ones: `data` a
split against its fingerprint in `results/datasets/` (labels, and SHA-256, mean and standard
deviation of every hologram; `fingerprint` writes one), `predictions` and `baseline` sample by
sample against committed runs, and `device` PyTorch on a device against the CPU.

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
