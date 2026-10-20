"""holofocus: physics core and data tools for learning-based autofocus in X-ray NFH.

Sources: [1] Dora et al., Opt. Express 32(7), 10801 (2024) (forward model, preprocessing,
ASRM); [2] Dora et al., Opt. Express 33(4), 6641 (2025) (model-based autofocus).
"""

from .device import get_device
from .forward import exit_wave, object_from_phase, simulate_hologram
from .geometry import ConeBeamGeometry, wavelength_from_energy
from .propagation import FresnelPropagator, propagate

__version__ = "0.1.0"

__all__ = [
    "ConeBeamGeometry",
    "FresnelPropagator",
    "exit_wave",
    "get_device",
    "object_from_phase",
    "propagate",
    "simulate_hologram",
    "wavelength_from_energy",
]
