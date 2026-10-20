import math

import pytest
import torch

from holofocus.arrays import crop_center
from holofocus.forward import (
    exit_wave,
    object_from_phase,
    refractive_object,
    simulate_hologram,
    single_material_object,
)
from holofocus.phantoms import demo_object
from holofocus.propagation import propagate


def test_refractive_object_signs():
    wavelength = 1e-10
    k = 2 * math.pi / wavelength
    delta_t = torch.tensor([1e-11], dtype=torch.float64)
    beta_t = torch.tensor([1e-13], dtype=torch.float64)
    obj = refractive_object(delta_t, beta_t, wavelength)
    assert obj.real.item() == pytest.approx(-k * 1e-11)
    assert obj.imag.item() == pytest.approx(k * 1e-13)
    thickness = torch.tensor([2e-6], dtype=torch.float64)
    same = single_material_object(thickness, 5e-6, 2.5e-8, wavelength)
    assert same.real.item() == pytest.approx(-k * 5e-6 * 2e-6)
    assert same.imag.item() == pytest.approx(k * 2.5e-8 * 2e-6)


def test_object_from_phase_fixed_delta_beta():
    phase = torch.tensor([-2.0, -1.0, 0.0], dtype=torch.float64)
    obj = object_from_phase(phase, delta_beta_ratio=200.0)
    torch.testing.assert_close(obj.imag, torch.tensor([0.01, 0.005, 0.0], dtype=torch.float64))
    assert torch.all(object_from_phase(phase, None).imag == 0)


def test_exit_wave_modulus_and_phase():
    obj = torch.complex(torch.tensor([-1.0]), torch.tensor([0.2]))
    psi = exit_wave(obj, a0=0.9)
    assert psi.abs().item() == pytest.approx(0.9 * math.exp(-0.2), rel=1e-6)
    assert torch.angle(psi).item() == pytest.approx(-1.0, rel=1e-6)


def test_empty_object_gives_flat_hologram():
    obj = torch.zeros((64, 64), dtype=torch.complex128)
    torch.testing.assert_close(
        simulate_hologram(obj, 2e-2), torch.ones(64, 64, dtype=torch.float64)
    )


def test_mean_intensity_equals_transmitted_energy():
    t = torch.from_numpy(demo_object((128, 128)))
    obj = object_from_phase(-1.5 * t, delta_beta_ratio=50.0)
    hologram = simulate_hologram(obj, 2e-2)
    expected = torch.exp(-2 * obj.imag).mean().item()
    assert hologram.mean().item() == pytest.approx(expected, rel=1e-10)


def test_weak_phase_object_follows_ctf():
    """|D(exp(i phi))|^2 ~ 1 + 2 F^-1[sin(pi k^2/Fr) F[phi]] for a weak phase object."""
    fr = 1.5e-2
    t = torch.from_numpy(demo_object((128, 128)))
    phase = -1e-4 * t
    hologram = simulate_hologram(object_from_phase(phase, None), fr)
    ky = torch.fft.fftfreq(128, dtype=torch.float64)[:, None]
    kx = torch.fft.fftfreq(128, dtype=torch.float64)[None, :]
    ctf = torch.sin(math.pi * (kx**2 + ky**2) / fr)
    linear = 1 + 2 * torch.fft.ifft2(ctf * torch.fft.fft2(phase)).real
    torch.testing.assert_close(hologram, linear, rtol=0, atol=1e-7)


def test_free_space_padding_matches_larger_simulation():
    fr = 2e-2
    t = torch.from_numpy(demo_object((96, 96)))
    obj = object_from_phase(-1.0 * t, delta_beta_ratio=200.0)
    padded = simulate_hologram(obj, fr, padded_shape=(256, 256))
    big = torch.zeros((512, 512), dtype=torch.complex128)
    big[208:304, 208:304] = obj
    reference = crop_center(propagate(exit_wave(big), fr).abs().pow(2), (96, 96))
    torch.testing.assert_close(padded, reference, rtol=0, atol=2e-3)
