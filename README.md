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
| 2 | Synthetic dataset generator: random single-material phantoms in the spirit of [2] Sec. 3.5, Poisson/Gaussian detector noise, `Δz01` uniform in ±5 mm, labels `Δz01` and `Fr`, YAML configs, per-sample seeds, HDF5 + PyTorch `Dataset`, CLI | [docs/laborbuch/02-datensatz.md](docs/laborbuch/02-datensatz.md) |
| 3 | Classical autofocus baseline of [2]: ASRM ([1] Alg. 5) ported from the paper code with the [2] Table 3 schedule, model fit error, projected Nelder–Mead, image criteria and focus curves; runs on the synthetic test split and on the Zenodo holograms of [1], cross-checked against holowizard; per-sample results in `results/` | [docs/laborbuch/03-baseline.md](docs/laborbuch/03-baseline.md) |
| 4 | First learned autofocus: a small CNN estimates `Δz01` with a Gaussian uncertainty from the radial power spectrum (or the binned hologram), trained on 10 000 synthetic samples on CPU; compared per sample with the raw and offset-corrected baseline of [2], and used as a warm start for its Nelder–Mead search | [docs/laborbuch/04-lernmodell.md](docs/laborbuch/04-lernmodell.md) |
| 5 | The learned autofocus on the four measured Zenodo holograms: one spectrum CNN per measured setup (energy, geometry, detector path, guess ± 5 mm), compared with [2] Table 1 and our padding-4 baseline, and as a warm start; the sim-to-real gap measured with test splits that change one simulator assumption at a time (detector blur, flat-field residual, noise, fibers, the measured object) and with training sets that have the measured properties | [docs/laborbuch/05-messdaten.md](docs/laborbuch/05-messdaten.md) |
| 6 | A more realistic object model: a weak-phase Gaussian random texture inside the object and as a film over the field of view, fitted per measured hologram on its spectral features; leave-one-out training sets (every property of a measured hologram estimated as a range over the other three), checked on the spectra before training; the spectrum CNNs of Stage 5 retrained on them and re-tested on the measured holograms against [2] Table 1 and our padding-4 search | [docs/laborbuch/06-objektmodell.md](docs/laborbuch/06-objektmodell.md) |
| 7 | The posterior of `Δz01` instead of a point estimate, by neural posterior estimation with sbi [6]: a spline flow on an embedding of the radial power spectrum, trained on the 10 000 synthetic samples of Stage 4 on CPU (five seeds as an ensemble, a mixture-density variant); calibration with SBC, expected coverage and TARP on 2000 prior draws, compared with the Stage 4 CNN on the test split and on the 48 baseline samples, on the perturbed test splits of Stage 5 and, for completeness, on the four measured holograms; re-run unchanged on the leave-one-out sets of Stage 6 with three seeds and their ensemble, against the leave-one-out CNNs on the simulated test splits and on the measured holograms (section 9) | [docs/laborbuch/07-npe.md](docs/laborbuch/07-npe.md) |
| 8 | Checks on the leave-one-out models of Stages 6 and 7: how far each measured hologram lies from its training set in the network input and in the NPE embedding, as a percentile among held-out simulated samples, against where the learned estimates fail, with the Stage 5 simulator as a positive control; the three-seed CNN ensemble (a mixture of Gaussians) against the NPE ensemble on the test splits and the measured holograms; the padding-4 search on the measured holograms started from either ensemble's estimate or restricted to its 95 % interval, against the full search | [docs/laborbuch/08-fehlspezifikation-warmstart.md](docs/laborbuch/08-fehlspezifikation-warmstart.md) |

**GPU runs.** The heavy runs (the baseline on the whole synthetic test split, larger training
runs, NPE ensembles and larger NPE runs) go to the A100 nodes of DESY Maxwell with the SLURM
kit in `cluster/maxwell/`; [docs/maxwell-gpu.md](docs/maxwell-gpu.md) is the step-by-step
guide (German).
`scripts/check_cluster_run.py` compares those runs sample by sample with the committed results.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
pytest                                   # ~40 s on a laptop CPU
python scripts/make_stage1_figures.py    # Stage 1 figures, ~2 min on 4 CPU threads
holofocus-dataset generate configs/dataset/ci.yaml --out /tmp/demo   # demo dataset, seconds
holofocus-dataset generate configs/dataset/cpu-4x.yaml --out data/cpu-4x   # ~16 min on CPU
holofocus-baseline run configs/baseline/ci.yaml --data /tmp/demo/test.h5 --out /tmp/baseline
holofocus-baseline run configs/baseline/paper2-cpu.yaml --data data/cpu-4x/test.h5 \
    --out results/baseline/cpu-4x-test --limit 48   # [2] baseline, ~5 min per sample on CPU
holofocus-learn train configs/learned/ci.yaml --train /tmp/demo/train.h5 --val /tmp/demo/val.h5 \
    --out /tmp/learned                          # tiny CNN on the demo dataset, seconds
holofocus-learn predict results/learned/spectrum-cnn-wide --data data/cpu-4x/test.h5 \
    --out /tmp/predictions                      # trained model, a few ms per hologram on CPU
pip install -e ".[dev,npe]"                     # sbi, for the posterior estimation of Stage 7
holofocus-learn npe-train configs/learned/npe-ci.yaml --train /tmp/demo/train.h5 \
    --val /tmp/demo/val.h5 --out /tmp/npe       # tiny NPE on the demo dataset, seconds
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
installation details, tests, figures, the dataset generator, the baseline, the learned
autofocus and the comparison with the paper code.

The main dataset uses the P05 detector binned 4x (512² pixels of 26 µm, sample pixel 130 nm
at `z01 = 100 mm`, `z02 = 20 m`, 11 keV), so `Fr = 1.5e-3` instead of `9.4e-5` with the same
field of view; it is simulated on a 2x finer grid and binned. Lab notebook entry 02 explains
the trade-off.

## Repository layout

```
src/holofocus/            package (geometry, propagation, forward model, preprocessing,
                          phantoms, noise, ASRM, measured Zenodo holograms)
src/holofocus/dataset/    dataset generator: YAML config, sampling, HDF5 storage, CLI
src/holofocus/autofocus/  MFE and image criteria, Nelder-Mead, nested autofocus, focus curves
src/holofocus/baseline/   baseline runs on dataset splits: YAML config, per-sample results, CLI
src/holofocus/learned/    learned autofocus: representations, CNNs with a Gaussian head,
                          training, inference, metrics, comparison with the baseline, CLI;
                          sim2real.py: measured noise, blur and object models, perturbations;
                          objectmodel.py: spectral regions, texture fits, leave-one-out ranges;
                          npe.py: neural posterior estimation with sbi; posterior.py: posterior
                          on a grid and its calibration statistics, mixtures of Gaussians;
                          misspecification.py: distances to a training set
configs/dataset/          dataset configs (cpu-4x, cpu-4x-extra, cpu-4x-large, cpu-4x-z01-range,
                          cpu-4x-calibration, ci; matched-<sample> for the measured setups with
                          -large, and -realistic for the cactus needle and the Mg wire;
                          loo-<sample> with -large, the leave-one-out sets with texture)
configs/baseline/         baseline configs (paper2-cpu, paper2-cpu-warm, both at padding 4 as
                          -p4, ci)
configs/learned/          learned-model configs (spectrum-cnn, spectrum-cnn-wide, -wide-hf,
                          image-cnn, ci; npe-nsf, npe-mdn, npe-ci for the NPE)
results/                  per-sample results quoted in the lab notebook (baseline, learned
                          models, comparison, measured holograms, sim-to-real study, object
                          model, NPE, misspecification, ensembles) and fingerprints of the val,
                          test and calibration splits
cluster/maxwell/          SLURM run kit for the A100 nodes of DESY Maxwell: settings template,
                          environment setup, submit script, jobs
tests/                    pytest suite (CPU; CUDA tests are skipped without a GPU)
scripts/                  figure scripts, Stage 3 experiments, Stage 4 to 8 studies, cross-checks
                          with holowizard, checks of cluster runs against the committed results
docs/usage.md             how to run things (English)
docs/maxwell-gpu.md       GPU runs on DESY Maxwell, step by step (German)
docs/laborbuch/           lab notebook, one entry per stage (German), figures in figures/
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
- [6] M. Deistler et al., "Simulation-based inference: A practical guide," arXiv:2508.12939
  (2025), [arxiv.org/abs/2508.12939](https://arxiv.org/abs/2508.12939).
