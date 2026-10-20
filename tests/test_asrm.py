import math

import pytest
import torch

from holofocus.asrm import (
    PAPER2_TABLE3,
    ASRMSettings,
    ASRMStage,
    StageGrid,
    asrm_reconstruct,
    data_gradient,
    fade_profile,
    gaussian_transfer,
    mirror_embed,
    pad_object,
    prepare_stage_data,
    project_refractive,
    resize_complex,
    schedule_for_binning,
    stage_grid,
    stage_window,
)
from holofocus.forward import object_from_phase, simulate_hologram
from holofocus.geometry import ConeBeamGeometry
from holofocus.phantoms import demo_object
from holofocus.propagation import FresnelPropagator

# [2] simulation geometry with a 16x binned detector: Fr ~ 2.4e-2 on a 128 px detector.
GEOMETRY = ConeBeamGeometry(energy_kev=11.0, z01=0.1, z02=20.0).binned(16)
# Short two-stage schedule with momentum filter and L2 term (no object filter): enough for a
# noise-free 128^2 hologram to converge within a second per reconstruction.
SMALL = ASRMSettings(
    stages=(
        ASRMStage(150, 0.9, 2, 1.0, 4.0, 0.0, 0.0, 1.0),
        ASRMStage(150, 1.1, 1, 1.0, 4.0, 0.0, 0.0, 0.1),
    ),
    padding_factor=2.0,
    fade_width=5,
)


@pytest.fixture(scope="module")
def hologram_and_object():
    t = torch.from_numpy(demo_object((128, 128)))
    obj = object_from_phase(-0.5 * t, delta_beta_ratio=200.0)
    hologram = simulate_hologram(obj, GEOMETRY.fresnel_number, padded_shape=(512, 512))
    return hologram, obj


def test_paper2_schedule_and_binning_adaptation():
    assert [s.iterations for s in PAPER2_TABLE3] == [700, 300, 500]
    assert [s.down_sampling for s in PAPER2_TABLE3] == [16, 4, 4]
    for binning, expected in ((1, [16, 4, 4]), (2, [8, 2, 2]), (4, [4, 1, 1]), (16, [1, 1, 1])):
        adapted = schedule_for_binning(PAPER2_TABLE3, binning)
        assert [s.down_sampling for s in adapted] == expected
        for a, s in zip(adapted, PAPER2_TABLE3, strict=True):
            assert (a.iterations, a.step, a.momentum_fwhm, a.l2_weight) == (
                s.iterations,
                s.step,
                s.momentum_fwhm,
                s.l2_weight,
            )
    with pytest.raises(ValueError):
        schedule_for_binning(PAPER2_TABLE3, 0)


def test_settings_for_binning_and_validation():
    settings = ASRMSettings.for_binning(4)
    assert [s.down_sampling for s in settings.stages] == [4, 1, 1]
    assert settings.fade_width == 20
    assert settings.iterations == 1500
    assert ASRMSettings.for_binning(4, fade_width=7).fade_width == 7
    for kwargs in (
        {"update": "nesterov"},
        {"gradient": "intensity"},
        {"padding_factor": 0.5},
        {"fade_width": -1},
        {"a0": 0.0},
        {"stages": ()},
    ):
        with pytest.raises(ValueError):
            ASRMSettings(**kwargs)
    for args in ((-1, 1.0), (10, 1.0, 0)):
        with pytest.raises(ValueError):
            ASRMStage(*args)
    with pytest.raises(ValueError):
        ASRMStage(10, 1.0, momentum_fwhm=-1.0)


def test_stage_grid_sizes_and_field_of_view():
    assert stage_grid((512, 512), 4, 2.0) == StageGrid((256, 256), (128, 128))
    assert stage_grid((512, 512), 1, 2.0) == StageGrid((1024, 1024), (512, 512))
    assert stage_grid((512, 512), 4, 7.0).shape == (896, 896)
    assert stage_grid((128, 96), 1, 1.0).shape == (128, 96)
    grid = StageGrid((10, 9), (4, 3))
    assert grid.fov == (slice(3, 7), slice(3, 6))
    with pytest.raises(ValueError):
        stage_grid((100, 100), 16, 2.0)


def test_fade_profile():
    p = fade_profile(64, 8, 56, 16, dtype=torch.float64)
    assert p.shape == (64,)
    assert torch.all(p[:8] == 0) and torch.all(p[56:] == 0)
    assert torch.all(p[16:48] == 1)
    assert torch.all((p > -1e-15) & (p <= 1))
    assert torch.all(p[9:16].diff() > 0) and torch.all(p[48:56].diff() < 0)
    box = fade_profile(32, 4, 28, 0)
    torch.testing.assert_close(box, ((torch.arange(32) >= 4) & (torch.arange(32) < 28)).float())
    wide = fade_profile(16, -8, 24, 4)
    assert torch.all(wide == 1)
    with pytest.raises(ValueError):
        fade_profile(32, 10, 14, 8)


def test_stage_window_is_flat_for_padding_two_and_faded_beyond():
    m = 32
    flat = stage_window(stage_grid((m, m), 1, 2.0), 8)
    assert flat.shape == (2 * m, 2 * m)
    assert torch.all(flat == 1)
    grid = stage_grid((m, m), 1, 4.0)
    faded = stage_window(grid, 8)
    assert faded.shape == (4 * m, 4 * m)
    assert faded[0, 0] == 0 and faded.max() == 1
    assert torch.all(faded[grid.fov] == 1)
    torch.testing.assert_close(faded, faded.T)


def test_mirror_embed_places_and_mirrors_the_field_of_view():
    m = 8
    fov = torch.arange(m, dtype=torch.float64)[None, :] + 100 * torch.arange(m)[:, None]
    grid = stage_grid((m, m), 1, 2.0)
    out = mirror_embed(fov, grid)
    rows, cols = grid.fov
    torch.testing.assert_close(out[rows, cols], fov)
    torch.testing.assert_close(out[rows, cols.start - 1], fov[:, 1])
    torch.testing.assert_close(out[rows, cols.stop], fov[:, m - 2])
    # p = 2: the even extension is continuous across the periodic boundary.
    torch.testing.assert_close(out[rows, 0], fov[:, m // 2])
    torch.testing.assert_close(out[rows, -1], fov[:, m // 2 - 1])
    wide = mirror_embed(fov, stage_grid((m, m), 1, 4.0), fill=-1.0)
    assert wide.shape == (4 * m, 4 * m) and wide[0, 0] == -1.0


def test_prepare_stage_data():
    g = torch.Generator().manual_seed(0)
    amplitude = 1 + 0.1 * torch.rand((64, 64), generator=g, dtype=torch.float64)
    grid = stage_grid((64, 64), 1, 2.0)
    data, window = prepare_stage_data(amplitude, grid, 1, 8)
    torch.testing.assert_close(data[grid.fov], amplitude)
    assert torch.all(window == 1)
    grid4 = stage_grid((64, 64), 4, 4.0)
    data4, window4 = prepare_stage_data(amplitude, grid4, 4, 8)
    assert data4.shape == (64, 64)
    assert torch.all(data4[window4 == 0] == 1)
    assert data4[grid4.fov].mean().item() == pytest.approx(amplitude.mean().item(), rel=1e-3)
    flat, _ = prepare_stage_data(torch.full((64, 64), 0.7), grid4, 4, 8)
    torch.testing.assert_close(flat[grid4.fov], torch.full((16, 16), 0.7))


def test_gaussian_transfer_matches_scipy():
    ndimage = pytest.importorskip("scipy.ndimage")
    import numpy as np

    shape, fwhm = (24, 30), 6.0
    sigma = fwhm / 2.35
    full = gaussian_transfer(shape, fwhm, dtype=torch.float64)
    expected = ndimage.fourier_gaussian(np.ones(shape), sigma)
    np.testing.assert_allclose(full.numpy(), expected, rtol=1e-12)
    half = gaussian_transfer(shape, fwhm, dtype=torch.float64, onesided=True)
    expected_half = ndimage.fourier_gaussian(np.ones((24, 16)), sigma, n=30, axis=-1)
    np.testing.assert_allclose(half.numpy(), expected_half, rtol=1e-12)
    assert gaussian_transfer(shape, 0.0) is None


@pytest.mark.parametrize("rule", ["amplitude", "reference"])
def test_data_gradient_matches_autograd(rule):
    g = torch.Generator().manual_seed(7)
    shape = (32, 32)
    data = (1 + 0.2 * torch.rand(shape, generator=g, dtype=torch.float64)).sqrt()
    re = -0.5 * torch.rand(shape, generator=g, dtype=torch.float64)
    im = 0.05 * torch.rand(shape, generator=g, dtype=torch.float64)
    forward = FresnelPropagator(shape, 0.05, dtype=torch.complex128).kernel
    grad, modulus = data_gradient(torch.complex(re, im), data, forward, rule)

    re_ad, im_ad = re.clone().requires_grad_(), im.clone().requires_grad_()
    psi = torch.exp(1j * torch.complex(re_ad, im_ad))
    a = torch.fft.ifft2(torch.fft.fft2(psi) * forward).abs()
    if rule == "amplitude":
        loss = 0.5 * (a - data).square().sum()
    else:
        loss = (0.5 * a**2 - 2.0 / 3.0 * data.sqrt() * a**1.5).sum()
    loss.backward()
    torch.testing.assert_close(modulus, a.detach())
    torch.testing.assert_close(grad.real, re_ad.grad, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(grad.imag, im_ad.grad, rtol=1e-9, atol=1e-12)


def test_pad_object_pins_the_phase_and_projection():
    g = torch.Generator().manual_seed(1)
    m = 16
    grid = stage_grid((m, m), 1, 2.0)
    fov = torch.complex(
        -0.2 - torch.rand((m, m), generator=g), 0.1 * torch.rand((m, m), generator=g)
    )
    obj = mirror_embed(fov, grid)
    window = stage_window(grid, 4)
    pinned = pad_object(obj, grid, window, pin_phase=True)[grid.fov]
    assert pinned.real.max().item() == pytest.approx(0.0, abs=1e-7)
    torch.testing.assert_close(pinned.real - pinned.real.max(), fov.real - fov.real.max())
    torch.testing.assert_close(pinned.imag, fov.imag)
    torch.testing.assert_close(pad_object(obj, grid, window, pin_phase=False)[grid.fov], fov)

    projected = project_refractive(torch.tensor([1 - 1j, -1 + 1j, -2 - 3j]))
    torch.testing.assert_close(projected, torch.tensor([0 + 0j, -1 + 1j, -2 + 0j]))


def test_resize_complex():
    x = torch.full((8, 8), 2.0 - 0.5j)
    assert resize_complex(x, (8, 8)) is x
    torch.testing.assert_close(resize_complex(x, (16, 12)), torch.full((16, 12), 2.0 - 0.5j))


def test_reconstruction_converges_and_mfe_marks_the_focus(hologram_and_object):
    hologram, obj = hologram_and_object
    results = {
        dz: asrm_reconstruct(hologram, GEOMETRY.defocused(dz).fresnel_number, SMALL)
        for dz in (-2e-3, 0.0, 2e-3)
    }
    focus = results[0.0]
    assert focus.obj.shape == (128, 128) and focus.obj.dtype == torch.complex128
    assert focus.padded.shape == (256, 256)
    assert len(focus.loss) == SMALL.iterations and focus.stage_ends == [150, 300]
    assert focus.final_loss < 1e-2 * focus.loss[0]
    assert torch.all(focus.obj.real <= 0) and torch.all(focus.obj.imag >= 0)
    phase_rmse = (focus.obj.real - obj.real).square().mean().sqrt().item()
    assert phase_rmse < 0.15 * obj.real.abs().max().item()
    assert focus.final_loss < 0.5 * min(results[-2e-3].final_loss, results[2e-3].final_loss)


def test_reconstruction_options_precision_and_input_checks(hologram_and_object):
    hologram, _ = hologram_and_object
    stages = (ASRMStage(30, 0.9, 1, 1.0, 4.0, 2.0, 2.0, 1.0),)
    fr = GEOMETRY.fresnel_number
    single = asrm_reconstruct(hologram.float(), fr, ASRMSettings(stages=stages, fade_width=5))
    assert single.obj.dtype == torch.complex64
    for kwargs in ({"update": "printed"}, {"gradient": "amplitude"}, {"pin_phase": False}):
        variant = asrm_reconstruct(
            hologram, fr, ASRMSettings(stages=stages, fade_width=5, **kwargs)
        )
        assert math.isfinite(variant.final_loss)
        assert variant.final_loss < variant.loss[0]
    warm = asrm_reconstruct(hologram, fr, ASRMSettings(stages=stages, fade_width=5), init=None)
    resumed = asrm_reconstruct(
        hologram, fr, ASRMSettings(stages=stages, fade_width=5), init=warm.padded
    )
    assert resumed.loss[0] < warm.loss[0]
    with pytest.raises(ValueError):
        asrm_reconstruct(hologram[None], fr, SMALL)
