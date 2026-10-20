import pytest
import torch

from holofocus.forward import exit_wave, object_from_phase, simulate_hologram
from holofocus.geometry import ConeBeamGeometry
from holofocus.phantoms import demo_object
from holofocus.propagation import FresnelPropagator
from holofocus.reconstruction import backpropagate, data_fidelity_gradient, pgd_reconstruct

# [2] simulation geometry with a 16x binned detector: Fr ~ 2.4e-2 on a 128 px detector.
GEOMETRY = ConeBeamGeometry(energy_kev=11.0, z01=0.1, z02=20.0).binned(16)


def correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum().item() / (a.norm() * b.norm()).item()


def test_analytic_gradient_matches_autograd():
    g = torch.Generator().manual_seed(3)
    shape, fr = (32, 32), 0.05
    target = (1 + 0.2 * torch.rand(shape, generator=g, dtype=torch.float64)).sqrt()
    re = -0.5 * torch.rand(shape, generator=g, dtype=torch.float64)
    im = 0.05 * torch.rand(shape, generator=g, dtype=torch.float64)
    prop = FresnelPropagator(shape, fr, dtype=torch.complex128)
    grad, loss = data_fidelity_gradient(torch.complex(re, im), target, prop)

    re_ad, im_ad = re.clone().requires_grad_(), im.clone().requires_grad_()
    amp = prop(exit_wave(torch.complex(re_ad, im_ad))).abs()
    loss_ad = 0.5 * (amp - target).pow(2).sum()
    loss_ad.backward()
    assert loss.item() == pytest.approx(loss_ad.item(), rel=1e-12)
    torch.testing.assert_close(grad.real, re_ad.grad, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(grad.imag, im_ad.grad, rtol=1e-9, atol=1e-12)


def _hologram_and_truth(max_phase=-0.5):
    t = torch.from_numpy(demo_object((128, 128)))
    obj = object_from_phase(max_phase * t, delta_beta_ratio=200.0)
    hologram = simulate_hologram(obj, GEOMETRY.fresnel_number, padded_shape=(512, 512))
    return hologram, obj


def test_backpropagation_correlates_best_at_true_focus():
    hologram, obj = _hologram_and_truth()
    scores = {}
    for dz in (-5e-3, -2.5e-3, 0.0, 2.5e-3, 5e-3):
        fr = GEOMETRY.defocused(dz).fresnel_number
        field = backpropagate(hologram, fr, padded_shape=(256, 256))
        scores[dz] = correlation(torch.angle(field), obj.real)
    assert max(scores, key=scores.get) == 0.0


def highpass_error(estimate, truth, cutoff=0.03):
    """Relative error above ``cutoff`` cycles/pixel, where de-focus blur acts."""
    ky = torch.fft.fftfreq(truth.shape[-2], dtype=torch.float64)[:, None]
    kx = torch.fft.fftfreq(truth.shape[-1], dtype=torch.float64)[None, :]
    mask = (kx**2 + ky**2).sqrt() > cutoff
    diff = torch.fft.fft2(estimate - truth) * mask
    return (diff.abs().norm() / (torch.fft.fft2(truth) * mask).abs().norm()).item()


def test_pgd_converges_and_is_sharpest_at_true_focus():
    """Plain PGD fits the data but recovers low frequencies only partly ([1] Sec. 2, 'weak
    reconstruction'); the de-focus shows up as a larger high-frequency error."""
    hologram, obj = _hologram_and_truth()
    errors = {}
    for dz in (-5e-3, 0.0, 5e-3):
        fr = GEOMETRY.defocused(dz).fresnel_number
        result = pgd_reconstruct(
            hologram, fr, iterations=300, step=1.0, momentum=0.9, padded_shape=(256, 256)
        )
        assert result.loss[-1] < 1e-3 * result.loss[0]
        assert torch.all(result.obj.real <= 0)
        errors[dz] = highpass_error(result.obj.real, obj.real)
    assert errors[0.0] < 0.55
    assert errors[0.0] < 0.9 * min(errors[-5e-3], errors[5e-3])


def test_holograms_identical_via_defocus_plane():
    """[2] Eq. (11), (15): propagating psi_exit to the de-focus plane and from there with the
    de-focused geometry reproduces the hologram exactly (MFE is zero without projection)."""
    _, obj = _hologram_and_truth(max_phase=-1.0)
    psi = exit_wave(obj)
    dz = 3e-3
    direct = FresnelPropagator(psi.shape, GEOMETRY.fresnel_number, dtype=psi.dtype)(psi)
    defocus_fr = GEOMETRY.fresnel_number_at(dz)
    rest_fr = GEOMETRY.defocused(dz).fresnel_number
    psi_hat = FresnelPropagator(psi.shape, defocus_fr, dtype=psi.dtype, warn_undersampled=False)(
        psi
    )
    via_plane = FresnelPropagator(psi.shape, rest_fr, dtype=psi.dtype)(psi_hat)
    torch.testing.assert_close(via_plane.abs(), direct.abs(), rtol=0, atol=1e-10)
