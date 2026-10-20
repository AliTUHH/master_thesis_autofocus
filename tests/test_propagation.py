import math

import pytest
import torch

from holofocus.propagation import (
    FresnelPropagator,
    check_sampling,
    compose_fresnel_numbers,
    propagate,
)


def random_field(shape, dtype=torch.complex128, seed=0):
    g = torch.Generator().manual_seed(seed)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    re = torch.randn(shape, generator=g, dtype=real_dtype)
    im = torch.randn(shape, generator=g, dtype=real_dtype)
    return torch.complex(re, im)


def test_kernel_matches_paper_formula():
    shape, fr = (64, 48), 0.03
    ky = torch.fft.fftfreq(shape[0], dtype=torch.float64)[:, None]
    kx = torch.fft.fftfreq(shape[1], dtype=torch.float64)[None, :]
    expected = torch.exp(-1j * math.pi * (kx**2 + ky**2) / fr)
    kernel = FresnelPropagator(shape, fr, dtype=torch.complex128).kernel
    torch.testing.assert_close(kernel, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("dtype, atol", [(torch.complex128, 1e-12), (torch.complex64, 2e-5)])
def test_round_trip_is_identity(dtype, atol):
    field = random_field((96, 80), dtype)
    prop = FresnelPropagator(field.shape, 2e-2, dtype=dtype)
    back = prop.inverse()(prop(field))
    torch.testing.assert_close(back, field, rtol=0, atol=atol)


def test_propagator_is_unitary_energy_conserving():
    field = random_field((128, 128))
    out = propagate(field, 5e-3, warn_undersampled=False)
    assert out.abs().pow(2).sum().item() == pytest.approx(
        field.abs().pow(2).sum().item(), rel=1e-12
    )


def test_pure_phase_object_keeps_mean_intensity():
    g = torch.Generator().manual_seed(1)
    phase = -2.0 * torch.rand((128, 128), generator=g, dtype=torch.float64)
    hologram = propagate(torch.exp(1j * phase), 1e-2).abs().pow(2)
    assert hologram.mean().item() == pytest.approx(1.0, rel=1e-12)


def test_composition_of_propagations():
    field = random_field((64, 64))
    fr_a, fr_b = 0.05, -0.12
    two_steps = propagate(propagate(field, fr_a), fr_b)
    one_step = propagate(field, compose_fresnel_numbers(fr_a, fr_b))
    torch.testing.assert_close(two_steps, one_step, rtol=0, atol=1e-11)
    assert math.isinf(compose_fresnel_numbers(0.1, -0.1))


def test_gaussian_beam_matches_analytic_fresnel_solution():
    """Gaussian beam: exp(-r^2/w^2) -> exp(-r^2 / (w^2 q)) / q with q = 1 + i / (pi w^2 Fr)."""
    n, w, fr = 256, 8.0, 0.01
    coords = torch.arange(n, dtype=torch.float64) - n // 2
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    r2 = x**2 + y**2
    field = torch.exp(-r2 / w**2).to(torch.complex128)
    zeta = 1.0 / (math.pi * w**2 * fr)
    q = 1 + 1j * zeta
    expected = torch.exp(-r2 / (w**2 * q)) / q
    shifted = torch.fft.ifftshift(field)
    result = torch.fft.fftshift(propagate(shifted, fr))
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-12)


def test_identity_and_validation():
    field = random_field((16, 16))
    torch.testing.assert_close(propagate(field, math.inf), field)
    with pytest.raises(ValueError):
        FresnelPropagator((16, 16), 0.0)
    with pytest.raises(ValueError):
        FresnelPropagator((16, 16), 0.1)(random_field((8, 8)))


def test_sampling_condition_warning():
    assert check_sampling((1000, 1000), 1e-3)
    assert not check_sampling((999, 1000), 1e-3)
    with pytest.warns(UserWarning, match="undersamples"):
        FresnelPropagator((64, 64), 1e-3)


def test_batch_dimensions():
    fields = random_field((3, 32, 32))
    batched = propagate(fields, 0.05)
    for i in range(3):
        torch.testing.assert_close(batched[i], propagate(fields[i], 0.05))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_and_cuda_agree():
    field = random_field((128, 128), torch.complex64)
    cpu = propagate(field, 1e-2)
    gpu = propagate(field.cuda(), 1e-2).cpu()
    torch.testing.assert_close(gpu, cpu, rtol=0, atol=1e-4)
