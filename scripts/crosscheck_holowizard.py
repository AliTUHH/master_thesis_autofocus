"""Compare holofocus with the paper code (holowizard >= 3.0) on Fresnel numbers and propagation.

Needs an environment with both packages, e.g. ``pip install holowizard==3.0.6`` (Python 3.11)
plus ``pip install --no-deps -e .`` of this repository. Not part of the test suite.
"""

from __future__ import annotations

import torch
from holowizard.core.models.cone_beam import ConeBeam
from holowizard.core.models.fresnel_propagator_torch import FresnelPropagator as HwPropagator
from holowizard.core.parameters.beam_setup import BeamSetup
from holowizard.core.parameters.measurement import Measurement

from holofocus.geometry import ConeBeamGeometry
from holofocus.propagation import FresnelPropagator

ROWS = [  # [1] Table 3
    ("spider hair", 11.0, 79.95, 19661.0),
    ("cactus needle", 17.0, 285.41, 19652.0),
    ("tooth", 17.0, 81.708, 19652.0),
    ("magnesium wire", 11.0, 470.51, 19661.0),
]


def main() -> None:
    print("sample            Fr holofocus    Fr holowizard   rel. diff")
    for name, energy, z01_mm, z02_mm in ROWS:
        ours = ConeBeamGeometry(energy, z01_mm * 1e-3, z02_mm * 1e-3, 6.5e-6).fresnel_number
        setup = BeamSetup(energy=energy, px_size=0.0065, z02=z02_mm)
        theirs = ConeBeam.get_fr(setup, Measurement(z01=z01_mm), verbose=False)
        print(f"{name:16s}  {ours:.6e}    {theirs:.6e}    {ours / theirs - 1:+.2e}")

    shape, fr = (256, 256), 2.4e-2
    g = torch.Generator().manual_seed(0)
    field = torch.exp(1j * (-torch.rand(shape, generator=g))).to(torch.complex64)
    ours = FresnelPropagator(shape, fr)(field)
    theirs = HwPropagator([fr], shape, torch.device("cpu")).propagate_forward(field, 0)
    diff = (ours - theirs).abs().max().item()
    back = HwPropagator([fr], shape, torch.device("cpu")).propagate_back(ours, 0)
    round_trip = (back - field).abs().max().item()
    print(f"propagator max |diff| (complex64): {diff:.2e}")
    print(f"holofocus forward + holowizard back, max |diff| to input: {round_trip:.2e}")


if __name__ == "__main__":
    main()
