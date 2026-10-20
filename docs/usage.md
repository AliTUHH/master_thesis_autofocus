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

GitHub Actions (`.github/workflows/tests.yml`) runs lint, the test suite and both figure
scripts in quick mode on CPU for Python 3.10 and 3.12. It also generates the demo dataset of
`configs/dataset/ci.yaml`, inspects it and keeps it for 30 days as the workflow artifact
`demo-dataset-ci` (Python 3.12 job).

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
