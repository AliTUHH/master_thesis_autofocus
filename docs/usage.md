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

GitHub Actions (`.github/workflows/tests.yml`) runs lint, the test suite and the figure
script in quick mode on CPU for Python 3.10 and 3.12.

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
