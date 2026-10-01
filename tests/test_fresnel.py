"""``src.utils.fresnel``: kernel convention, unitarity, round trip and the defocus-blur measure."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from src.utils import fresnel


def _weak_phase_object(size: int, seed: int = 0, phi_max: float = 0.01, width: float = 6.0) -> np.ndarray:
    """Smooth random phase pattern (|phi| <= phi_max, Gaussian taper of sigma ``size / width``, i.e. zero near
    the border) for the weak-object limit."""
    rng = np.random.default_rng(seed)
    phase = gaussian_filter(rng.standard_normal((size, size)), 3.0)
    yy, xx = np.mgrid[:size, :size]
    taper = np.exp(-(((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2 * (size / width) ** 2)))
    phase = phase * taper
    return phase / np.abs(phase).max() * phi_max


def test_kernel_matches_formula_and_sign() -> None:
    fr = 7.5e-3
    kernel = fresnel.fresnel_kernel((32, 48), fr)
    xi = np.fft.fftfreq(32)[:, None]
    eta = np.fft.fftfreq(48)[None, :]
    expected = np.exp(-1j * np.pi / fr * (xi**2 + eta**2))
    np.testing.assert_allclose(kernel.numpy(), expected.astype(np.complex64), atol=1e-6)
    assert kernel.dtype == torch.complex64
    # the back-propagation kernel is the complex conjugate
    np.testing.assert_allclose(fresnel.fresnel_kernel((32, 48), -fr).numpy(), np.conj(expected), atol=1e-6)
    with pytest.raises(ValueError):
        fresnel.fresnel_kernel((8, 8), 0.0)


def test_weak_object_ctf_has_period_fr_in_u() -> None:
    """Weak phase object: FFT(a - 1) = sin(pi u / Fr) FFT(phi) with u = xi^2 + eta^2 (sign fixes the kernel
    convention, the zeros at u = n Fr fix the period)."""
    size, fr = 128, 1.5e-2
    phi = _weak_phase_object(size, seed=1, phi_max=0.002)
    amplitude = fresnel.hologram_amplitude(phi.astype(np.complex64), fr, pad_factor=1.0).numpy().astype(np.float64)
    contrast_spectrum = np.fft.fft2(amplitude - 1.0)
    freqs = np.fft.fftfreq(size)
    u = freqs[:, None] ** 2 + freqs[None, :] ** 2
    predicted = np.sin(np.pi * u / fr) * np.fft.fft2(phi)
    scale = np.abs(predicted).max()
    # residual = second-order terms O(phi^2) (largest at DC, where the linear prediction vanishes)
    assert np.abs(contrast_spectrum - predicted).max() < 3e-3 * scale
    # the opposite sign convention would be badly wrong
    assert np.abs(contrast_spectrum + predicted).max() > 0.5 * scale
    # ring zeros: power at u = Fr (first zero) is far below the power at u = 1.5 Fr (first maximum)
    power = np.abs(contrast_spectrum) ** 2 / np.maximum(np.abs(np.fft.fft2(phi)) ** 2, 1e-30)
    ring_zero = power[(u > 0.95 * fr) & (u < 1.05 * fr)]
    ring_max = power[(u > 1.45 * fr) & (u < 1.55 * fr)]
    assert np.median(ring_zero) < 0.05 * np.median(ring_max)


def test_propagation_is_unitary_without_padding() -> None:
    rng = np.random.default_rng(2)
    field = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
    out = fresnel.propagate(field, 3.0e-3, pad_factor=1.0)
    assert out.shape == (64, 64) and out.dtype == torch.complex64
    assert torch.sum(torch.abs(out) ** 2).item() == pytest.approx(float(np.sum(np.abs(field) ** 2)), rel=1e-4)


def test_roundtrip_forward_backward() -> None:
    rng = np.random.default_rng(3)
    field = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
    fr = 2.0e-3
    back = fresnel.backpropagate(fresnel.propagate(field, fr, pad_factor=1.0), fr, pad_factor=1.0)
    np.testing.assert_allclose(back.numpy(), field.astype(np.complex64), atol=1e-4)
    # with padding (plane wave outside) the round trip of a compact object is still accurate as long as the
    # fringes stay inside the field of view and the chirp is sampled (Fr >= 1 / padded grid = 1/128)
    fr = 1.0e-2
    phi = _weak_phase_object(64, seed=4, phi_max=0.5, width=10.0)
    psi = np.exp(1j * phi)
    back2 = fresnel.propagate(fresnel.propagate(psi, fr, pad_factor=2.0, pad_value=1.0), -fr, pad_factor=2.0, pad_value=1.0)
    assert np.abs(back2.numpy() - psi).max() < 2e-3
    # batched input is propagated per image
    batch = fresnel.propagate(np.stack([psi, np.conj(psi)]), fr, pad_factor=2.0, pad_value=1.0)
    np.testing.assert_allclose(batch[0].numpy(), fresnel.propagate(psi, fr, pad_factor=2.0, pad_value=1.0).numpy(), atol=1e-6)


def test_hologram_amplitude_conventions() -> None:
    size = 64
    phi = _weak_phase_object(size, seed=5, phi_max=0.3)
    obj = (-np.abs(phi) + 0.05j * np.abs(phi) / phi.max()).astype(np.complex64)  # phase <= 0, absorption >= 0
    amplitude = fresnel.hologram_amplitude(obj, 1.0e-2, pad_factor=2.0)
    assert amplitude.shape == (size, size) and amplitude.dtype == torch.float32
    # an empty object gives a flat unit amplitude; a probe given on the padded grid (HoloForge `crop_probe`) multiplies in
    empty = np.zeros((size, size), dtype=np.complex64)
    np.testing.assert_allclose(fresnel.hologram_amplitude(empty, 1.0e-2).numpy(), 1.0, atol=1e-5)
    probe = 0.5 * np.ones((2 * size, 2 * size), dtype=np.complex64)
    np.testing.assert_allclose(fresnel.hologram_amplitude(empty, 1.0e-2, probe=probe).numpy(), 0.5, atol=1e-5)
    # a probe of the object size is padded with ones (its edge diffracts), the centre stays at the probe value
    centre = fresnel.hologram_amplitude(empty, 1.0e-2, probe=probe[:size, :size]).numpy()[size // 2, size // 2]
    assert centre == pytest.approx(0.5, abs=0.02)
    # absorption reduces the mean amplitude, Fresnel fringes conserve the mean of the intensity contrast only approximately
    assert float(amplitude.mean()) < 1.0
    # padding changes the result (fringes of the wrapped-around copies are removed)
    unpadded = fresnel.hologram_amplitude(obj, 1.0e-2, pad_factor=1.0)
    assert not np.allclose(unpadded.numpy(), amplitude.numpy(), atol=1e-4)


def test_defocus_blur_px() -> None:
    # 1 px blur for |e| = Fr; 1 % error at Fr = 1e-2 -> 1 px; the P05 tolerance z01_tol = 0.1 mm at z01 = 250 mm
    # (|e| ~ 0.04 %, Fr = 2.37e-4) gives ~1.3 px
    assert fresnel.defocus_blur_px(1.0e-2, 1.0e-2) == pytest.approx(1.0)
    assert fresnel.defocus_blur_px(-0.01, 1.0e-2) == pytest.approx(1.0)
    assert fresnel.defocus_blur_px(0.20, 1.518e-2) == pytest.approx(3.63, rel=0.01)
    assert fresnel.defocus_blur_px(4.0e-4, 2.3725e-4) == pytest.approx(1.30, rel=0.01)
    arr = fresnel.defocus_blur_px(np.array([0.0, 0.04, -0.01]), np.array([1e-2, 1e-2, 1e-4]))
    np.testing.assert_allclose(arr, [0.0, 2.0, 10.0])
    assert fresnel.blur_px_from_fresnel_numbers(1.01e-2, 1.0e-2) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fresnel.defocus_blur_px(0.1, 0.0)
