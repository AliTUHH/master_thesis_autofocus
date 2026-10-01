"""Sketch of an *online* HoloForge simulator ``theta -> x`` for ``sbi`` (not used by the experiments).

The NPE runs in :mod:`src.sbi.npe` train on the pre-simulated HDF5 files (amortised, one round).  For
sequential/multi-round NPE around a measured hologram (``SNPE``: ``proposal = posterior.set_default_x(x_o)``)
or for generating training data on the fly, ``sbi`` expects a callable ``simulator(theta) -> x`` with
``theta`` of shape ``[N, 1]`` and ``x`` of shape ``[N, *x_shape]``.  This module shows how the existing
HoloForge pipeline of :mod:`src.data.generate_data` is wrapped into that interface:

1. the data-set YAML (``configs/data_small.yaml``) is translated with :func:`build_forge_config` into the
   HoloForge JSON config, but the ``setup`` block uses :class:`NFHFixedDistSetup`, a subclass of
   :class:`src.data.forge_setup.NFHRandomDistSetup` whose next ``z01`` can be *set* instead of drawn;
2. per parameter value HoloForge's generators create a random phantom, probe and (noisy) hologram with
   the propagation kernel of exactly that ``z01`` (``HologramGenerator.create_hologram`` reads
   ``setup.kernel`` -> ``get_distances`` -> the queued value);
3. the hologram is pre-processed like the training data (:meth:`HologramHDF5Dataset.preprocess`:
   crop, pooling, normalisation, representation), so ``x`` matches the embedding network's input.

Cost: one 256 px hologram of ``data_small`` takes ~0.4 s on CPU (2048 px with padding 4: GPU only),
hence the simulator is *sketched and smoke-testable* here but the experiments use the stored files.

Usage sketch::

    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    simulator = HoloForgeSimulator(load_yaml("configs/data_small.yaml"), preprocess=dataset.preprocess, seed=0)
    prior = BoxUniform(torch.tensor([50.0]), torch.tensor([300.0]))
    theta = prior.sample((1000,))
    x = simulator(theta)                      # [1000, 256] radial profiles (or [1000, 1, 256, 256])
    inference = NPE(prior).append_simulations(theta, x)
    # multi-round around an observation x_o: proposal = inference.build_posterior().set_default_x(x_o)
"""

from __future__ import annotations

import json
import random
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import holowizard.forge.experiment.setup as forge_setup_module
import holowizard.forge.generators as forge_generators
import holowizard.forge.utils.torch_settings as torch_settings
import torch
from holowizard.forge.configs.parse_config import ConfigParser
from holowizard.forge.generators import HologramGenerator, PhantomGenerator, ProbeGenerator
from holowizard.forge.utils import calc_Fr

from src.data.forge_setup import NFHRandomDistSetup, register_forge_setup, reset_global_forge_setup
from src.data.generate_data import build_forge_config
from src.utils.physics import DISTANCE_DECIMALS_MM

__all__ = ["NFHFixedDistSetup", "HoloForgeSimulator"]


class NFHFixedDistSetup(NFHRandomDistSetup):
    """Random-distance setup whose next ``z01`` can be queued explicitly.

    :meth:`get_distances` returns the queued value (rounded to 1 µm like HoloWizard) and clears the
    queue; without a queued value it falls back to the random draw of the parent class.  ``z02`` stays at
    the configured (fixed) value; a varying ``z02`` would make ``theta`` two-dimensional.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._next_z01: float | None = None
        super().__init__(*args, **kwargs)

    def queue_z01(self, z01_mm: float) -> None:
        lo, hi = self.z01_bounds
        if not lo <= z01_mm <= hi:
            raise ValueError(f"z01={z01_mm} mm outside the setup bounds [{lo}, {hi}] mm")
        self._next_z01 = float(z01_mm)

    def get_distances(self) -> tuple[float, float]:
        if self._next_z01 is None:
            return super().get_distances()
        z01, self._next_z01 = self._next_z01, None
        z02 = self.z02_bounds[0]
        self._z01 = round(z01, DISTANCE_DECIMALS_MM)
        self._z02 = round(z02, DISTANCE_DECIMALS_MM)
        self._Fr = float(calc_Fr(self.energy, self._z01, self._z02, self.detector_px_size))
        self._num_draws += 1
        return self._z01, self._z02


class HoloForgeSimulator:
    """``sbi``-compatible simulator: ``theta [N, 1]`` (z01 in mm) -> ``x [N, *x_shape]``.

    Args:
        data_cfg: Data-set YAML mapping (``setup``, ``phantom``, ``probe``, ``hologram_noise`` ...), see
            :func:`src.data.generate_data.build_forge_config`.
        preprocess: Maps a raw hologram ``[1, H, W]`` (float32 amplitude) to the network input; pass
            :meth:`src.data.dataset.HologramHDF5Dataset.preprocess` of the training set so that online
            simulations are represented exactly like the stored ones.  ``None`` returns the raw hologram.
        seed: Seed for HoloForge (phantom shapes, probe, noise); ``z01`` itself comes from ``theta``.
    """

    def __init__(
        self,
        data_cfg: dict[str, Any],
        preprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed: int = 0,
        name: str = "sbi_online",
    ) -> None:
        forge_cfg = build_forge_config(data_cfg, name, seed)
        forge_cfg["setup"]["type"] = NFHFixedDistSetup.__name__
        self.forge_config = forge_cfg
        self.preprocess = preprocess
        self._seed = int(seed)

        register_forge_setup()
        setattr(forge_setup_module, NFHFixedDistSetup.__name__, NFHFixedDistSetup)
        reset_global_forge_setup()
        random.seed(self._seed)
        torch_settings.set_reproducibility(self._seed)
        with tempfile.TemporaryDirectory(prefix="forge_sbi_") as tmp:
            cfg_path = Path(tmp) / f"{name}.json"
            cfg_path.write_text(json.dumps(forge_cfg, indent=2), encoding="utf-8")
            config = ConfigParser(cfg_path)
        self._hologram_generator = HologramGenerator(config)
        self._phantom_generator = PhantomGenerator(config)
        self._probe_generator = ProbeGenerator(config)
        self._flatfield_generator = config.init_obj("flatfield_generator", forge_generators, config=config)
        self.setup: NFHFixedDistSetup = self._hologram_generator.simulation.setup
        if not isinstance(self.setup, NFHFixedDistSetup):
            raise RuntimeError("HoloForge did not install NFHFixedDistSetup; is another setup registered?")

    @property
    def z01_bounds(self) -> tuple[float, float]:
        return self.setup.z01_bounds

    @torch.no_grad()
    def simulate_one(self, z01_mm: float) -> torch.Tensor:
        """One hologram ``[1, H, W]`` (raw amplitude incl. noise) for the given ``z01``."""
        self.setup.queue_z01(float(z01_mm))
        phantom = self._phantom_generator.create_phantom()
        probe = self._probe_generator.create_probe()
        flatfield = self._flatfield_generator.create_flatfield()
        hologram = self._hologram_generator.create_hologram(phantom=phantom, probe=probe, flatfield=flatfield)
        if abs(self.setup.z01 - round(float(z01_mm), DISTANCE_DECIMALS_MM)) > 1e-9:
            raise RuntimeError(f"setup used z01={self.setup.z01} instead of the queued {z01_mm}")
        return hologram.detach().float().cpu().unsqueeze(0)

    @torch.no_grad()
    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """Simulate a batch: ``theta`` ``[N, 1]`` or ``[N]`` -> ``x`` ``[N, *x_shape]`` (float32)."""
        values = torch.as_tensor(theta, dtype=torch.float64).reshape(-1)
        outputs = []
        for z01 in values.tolist():
            hologram = self.simulate_one(z01)
            outputs.append(self.preprocess(hologram) if self.preprocess is not None else hologram)
        return torch.stack(outputs, dim=0).float()
