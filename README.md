# holofocus — learning-based autofocus for X-ray near-field holography

Code for the master's thesis *Learning-based autofocus for Holography* (Mohamad Ali Rajeh,
TUHH, Institute for Biomedical Imaging). The thesis is planned for submission in March 2027.

In near-field holography (NFH) the hologram depends on the setup through the Fresnel number
`Fr`. In the cone-beam setup of the P05 beamline, `Fr` follows from the photon energy, the
detector pixel size, the focus–detector distance `z02` (known) and the focus–sample distance
`z01`, which is only known to about ±5 mm from a manual guess. A wrong `z01` blurs every phase
reconstruction. The model-based autofocus of [2] finds `z01` by nested optimization
(reconstruction with ASRM [1], model fit error, Nelder–Mead); this thesis aims at a learned
estimator that is much faster.

**Target quantity.** `Δz01 = z01 − z01_guess` inside the box `±5 mm` of [2]; `Fr` is derived
from `z01` and is available as an alternative label.

## Status

| Stage | Content | Lab notebook |
|---|---|---|
| 1 | Physics core: cone-beam geometry, Fresnel propagator, forward model, preprocessing of [1] Sec. 4.1, simple back-propagation/PGD, defocus figures | [docs/laborbuch/01-physik-kern.md](docs/laborbuch/01-physik-kern.md) |
| 2 | Synthetic dataset generator (phantoms, noise, `z01` sampling, HDF5) | planned |
| 3 | Baseline of [2]: ASRM + MFE + Nelder–Mead | planned |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
pytest                                   # ~5 s on a laptop CPU
python scripts/make_stage1_figures.py    # Stage 1 figures, ~2 min on 4 CPU threads
```

```python
import torch
from holofocus import ConeBeamGeometry, object_from_phase, simulate_hologram

geometry = ConeBeamGeometry(energy_kev=11.0, z01=0.100, z02=20.0).binned(4)  # Fr = 1.5e-3
phase = -torch.rand(512, 512)  # any phase map phi <= 0 in rad
obj = object_from_phase(phase, delta_beta_ratio=200.0)
hologram = simulate_hologram(obj, geometry.fresnel_number, padded_shape=(2048, 2048))
```

All tensors live on the device you create them on; the same code runs on CPU and CUDA
(`holofocus.get_device()` picks CUDA if available). See [docs/usage.md](docs/usage.md) for
installation details, tests, figures and the comparison with the paper code.

## Repository layout

```
src/holofocus/      package (geometry, propagation, forward model, preprocessing, ...)
tests/              pytest suite (CPU; CUDA tests are skipped without a GPU)
scripts/            figure scripts and cross-checks with the paper code (holowizard)
docs/usage.md       how to run things (English)
docs/laborbuch/     lab notebook, one entry per stage (German), figures in figures/
```

## History

The first commit of this repository ([3579d34](https://github.com/AliTUHH/master_thesis_autofocus/commit/3579d34))
contained an unverified first draft created with DeepSeek (phase-only spheres, parallel beam,
regression of a distance around 1 m). It was removed in Stage 1 because it does not model the
cone-beam geometry of [1]/[2]; it remains available in the git history.

## References

- [1] J. Dora et al., "Artifact-suppressing reconstruction of strongly interacting objects in
  X-ray near-field holography without a spatial support constraint," Opt. Express 32(7),
  10801 (2024), [doi:10.1364/OE.514641](https://doi.org/10.1364/OE.514641).
  Code: [doi:10.5281/zenodo.8349365](https://doi.org/10.5281/zenodo.8349365) (successor:
  [holowizard](https://pypi.org/project/holowizard/)).
- [2] J. Dora et al., "Model-based autofocus for near-field phase retrieval," Opt. Express
  33(4), 6641 (2025), [doi:10.1364/OE.544573](https://doi.org/10.1364/OE.544573).
