import pytest
import torch

from holofocus.preprocessing import (
    blackman,
    fade_window_1d,
    flat_field_correct,
    preprocess_hologram,
)


def test_blackman_formula_of_paper1_eq17():
    w = blackman(9, dtype=torch.float64)
    assert w[0].item() == pytest.approx(0.0, abs=1e-12)
    assert w[4].item() == pytest.approx(1.0, abs=1e-12)
    torch.testing.assert_close(w, w.flip(0))


def test_fade_window_1d_layout():
    w = fade_window_1d(40, 5, 35, 10, dtype=torch.float64)
    assert torch.all(w[:5] == 0) and torch.all(w[35:] == 0)
    assert torch.all(w[10:30] == 1)
    assert torch.all(torch.diff(w[5:10]) >= 0) and torch.all(torch.diff(w[30:35]) <= 0)
    with pytest.raises(ValueError):
        fade_window_1d(40, 5, 12, 10)


def test_flat_field_correction():
    raw = torch.tensor([2.0, 3.0])
    flat = torch.tensor([2.0, 1.5])
    torch.testing.assert_close(flat_field_correct(raw, flat), torch.tensor([1.0, 2.0]))


def test_preprocessing_keeps_fov_and_pads_with_a0():
    g = torch.Generator().manual_seed(0)
    hologram = 0.1 + torch.rand((32, 40), generator=g, dtype=torch.float32)
    pre = preprocess_hologram(hologram, (128, 128), a0=0.97)
    assert torch.equal(pre.crop(pre.data), hologram)
    assert pre.data[0, 0].item() == pytest.approx(0.97)
    assert pre.data[-1, -1].item() == pytest.approx(0.97)
    rows, cols = pre.fov
    left_mirror = pre.data[rows, cols.start - 5 : cols.start]
    torch.testing.assert_close(left_mirror, hologram[:, 1:6].flip(-1))
    assert torch.all(pre.window[rows, cols] == 1)


def test_preprocessing_without_mirror_is_constant_padding():
    hologram = torch.full((16, 16), 1.2, dtype=torch.float64)
    pre = preprocess_hologram(hologram, (64, 64), a0=1.0, mirror=False)
    assert pre.data.sum().item() == pytest.approx(64 * 64 + 0.2 * 256)


def test_mirror_extension_limited_by_target_shape():
    hologram = torch.rand((64, 64), dtype=torch.float64)
    pre = preprocess_hologram(hologram, (96, 96))
    torch.testing.assert_close(pre.crop(pre.data), hologram)
    with pytest.raises(ValueError):
        preprocess_hologram(hologram, (32, 32))
