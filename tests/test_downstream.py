"""Tests for the downstream reconstruction metrics and the downstream evaluation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.baseline.model_based_autofocus import SMOKE_PRESET, Geometry
from src.baseline.results import make_result_row, write_samples_csv
from src.data.forge_samples import ForgeSample
from src.eval.downstream import (
    TRUE_CANDIDATE,
    candidate_label,
    candidates_from_csv,
    data_residual,
    downstream_markdown_table,
    is_diverged,
    reconstruction_metrics,
    summarize_downstream,
)
from src.utils.fresnel import hologram_amplitude

torch.set_num_threads(2)


def _phantom_phase(size: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    phase = np.zeros((size, size), dtype=np.float64)
    for _ in range(3):
        cx, cy = rng.uniform(size * 0.3, size * 0.7, 2)
        radius = rng.uniform(size * 0.08, size * 0.15)
        phase -= rng.uniform(0.5, 1.0) * np.clip(1.0 - ((x - cx) ** 2 + (y - cy) ** 2) / radius**2, 0.0, None)
    return phase


# --------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------
def test_reconstruction_metrics_identity_offset_scaling() -> None:
    gt = _phantom_phase()
    identical = reconstruction_metrics(gt, gt, border=4)
    assert identical["nrmse"] == pytest.approx(0.0, abs=1e-12)
    assert identical["nrmse_offset_free"] == pytest.approx(0.0, abs=1e-12)
    assert identical["nrmse_grad"] == pytest.approx(0.0, abs=1e-12)
    assert identical["pearson"] == pytest.approx(1.0)
    if identical["ssim"] is not None:
        assert identical["ssim"] == pytest.approx(1.0)

    offset = reconstruction_metrics(gt - 0.7, gt, border=4)
    assert offset["nrmse"] > 0.1
    assert offset["nrmse_offset_free"] == pytest.approx(0.0, abs=1e-12)
    assert offset["nrmse_grad"] == pytest.approx(0.0, abs=1e-12)
    assert offset["pearson"] == pytest.approx(1.0)

    scaled = reconstruction_metrics(2.0 * gt, gt, border=4)
    assert scaled["nrmse"] == pytest.approx(1.0)  # ||2g - g|| / ||g||
    assert scaled["nrmse_grad"] == pytest.approx(1.0)
    assert scaled["pearson"] == pytest.approx(1.0)
    assert set(scaled) == {"nrmse", "nrmse_offset_free", "nrmse_grad", "pearson", "ssim"}


def test_reconstruction_metrics_border_and_shapes() -> None:
    gt = _phantom_phase(32)
    noisy = gt + 0.05 * np.random.default_rng(1).standard_normal(gt.shape)
    small_border = reconstruction_metrics(noisy, gt, border=2)
    no_crop = reconstruction_metrics(noisy, gt, border=16)  # 2 * 16 >= 32 -> no crop
    assert 0 < small_border["nrmse"] < 1 and 0 < no_crop["nrmse"] < 1
    assert reconstruction_metrics(noisy, gt, border=0)["nrmse"] == pytest.approx(no_crop["nrmse"])
    with pytest.raises(ValueError):
        reconstruction_metrics(gt[:-1], gt, border=0)


def test_data_residual_of_true_object_is_at_noise_level() -> None:
    fr = 2e-2
    size = 64
    phase = _phantom_phase(size).astype(np.float32)
    obj = phase + 1j * np.zeros_like(phase)
    clean = hologram_amplitude(torch.as_tensor(obj.astype(np.complex64)), fr, pad_factor=2).numpy()
    sigma = 0.02
    noisy = clean + sigma * np.random.default_rng(2).standard_normal(clean.shape).astype(np.float32)
    residual_true = data_residual(obj, noisy, fr, padding_factor=2.0)
    assert residual_true == pytest.approx(sigma, rel=0.15)
    assert data_residual(obj, clean, fr, padding_factor=2.0) < 1e-5
    assert data_residual(obj, noisy, fr * 1.3, padding_factor=2.0) > 1.1 * residual_true
    assert data_residual(np.zeros_like(obj), noisy, fr, padding_factor=2.0) > 2 * residual_true


def test_data_residual_on_forge_sample(tiny_dataset_dir) -> None:
    """The stored HoloForge hologram is reproduced by src.utils.fresnel up to the 5 % Gaussian noise."""
    from src.data.forge_samples import read_forge_samples

    for sample in read_forge_samples([tiny_dataset_dir / "test.hdf5"], n_per_file=2):
        assert sample.phantom is not None
        residual = data_residual(sample.phantom, sample.hologram_amplitude, sample.fr, padding_factor=sample.padding_factor)
        assert 0.04 < residual < 0.06, residual
        assert data_residual(sample.phantom, sample.hologram_amplitude, sample.fr * 1.3, padding_factor=sample.padding_factor) > residual


def test_is_diverged() -> None:
    assert not is_diverged(2e-4, 0.8)
    assert is_diverged(float("nan"), 0.8)
    assert is_diverged(0.99, 0.8)
    assert is_diverged(2e-4, 1e6)


# --------------------------------------------------------------------------------------------------
# aggregation helpers (no HoloWizard needed)
# --------------------------------------------------------------------------------------------------
def _fake_rows() -> list[dict]:
    rows = []
    for sample in range(2):
        for label, err, nrmse in ((TRUE_CANDIDATE, 0.0, 0.5), ("+10%", 10.0, 0.6), ("ml_cnn", 3.0, 0.55)):
            rows.append(
                {
                    "sample": sample, "source": "a.hdf5", "index": sample, "fr_true": 0.01, "z01_true_mm": 100.0,
                    "candidate": label, "fr_err_pct_nominal": err, "fr_used": 0.01 * (1 + err / 100), "rel_err_fr_pct": err,
                    "blur_px": float(np.sqrt(abs(err / 100) / 0.01)), "nrmse": nrmse + 0.01 * sample, "nrmse_offset_free": nrmse,
                    "nrmse_grad": nrmse, "pearson": 0.9, "ssim": None, "core_loss_final": 1e-4, "data_residual_true_fr": 0.05,
                    "runtime_s": 1.0, "diverged": False, "nrmse_rel_true": nrmse / 0.5, "nrmse_offset_free_rel_true": 1.0,
                    "nrmse_grad_rel_true": 1.0, "core_loss_final_rel_true": 1.0, "data_residual_true_fr_rel_true": 1.0,
                }
            )
    return rows


def test_summarize_downstream_and_table() -> None:
    rows = _fake_rows()
    summary = summarize_downstream(rows)
    assert [e["candidate"] for e in summary] == [TRUE_CANDIDATE, "+10%", "ml_cnn"]
    plus10 = summary[1]
    assert plus10["n"] == 2 and plus10["n_diverged"] == 0
    assert plus10["fr_err_pct_nominal"] == 10.0
    assert plus10["nrmse_median"] == pytest.approx(0.605)
    assert plus10["nrmse_rel_true_median"] == pytest.approx(1.2)
    assert plus10["blur_px_mean"] == pytest.approx(np.sqrt(0.1 / 0.01))
    assert plus10["ssim_median"] is None
    table = downstream_markdown_table(summary)
    assert table.count("\n") == 2 + len(summary)
    assert "| ml_cnn |" in table and "n/a" in table
    assert candidate_label(0) == TRUE_CANDIDATE and candidate_label(-5) == "-5%" and candidate_label(10) == "+10%"


def test_candidates_from_csv(tmp_path) -> None:
    sample_a = ForgeSample(np.ones((8, 8), np.float32), None, 0.01, 100.0, 20000.0, 11.0, 0.052, 2.0, source="a.hdf5", index=0)
    sample_b = ForgeSample(np.ones((8, 8), np.float32), None, 0.02, 200.0, 20000.0, 11.0, 0.052, 2.0, source="a.hdf5", index=1)
    rows = [
        make_result_row(0, "a.hdf5", 0.01, 100.0, 0.0101, 101.0, 0.1, 1, "ml_cnn"),
        make_result_row(1, "a.hdf5", 0.02, 200.0, 0.0190, 190.0, 0.1, 1, "ml_cnn"),
        make_result_row(7, "a.hdf5", 0.03, 300.0, 0.0300, 300.0, 0.1, 1, "ml_cnn"),  # not among the samples
    ]
    path = write_samples_csv(rows, tmp_path / "samples.csv")
    other = write_samples_csv([make_result_row(0, "a.hdf5", 0.01, 100.0, 0.0105, 105.0, 2.0, 21, "holowizard_find_focus")], tmp_path / "other.csv")
    candidates = candidates_from_csv([path, other], [sample_a, sample_b])
    assert candidates[("a.hdf5", 0)] == pytest.approx({"ml_cnn": 0.0101, "holowizard_find_focus": 0.0105})
    assert candidates[("a.hdf5", 1)] == pytest.approx({"ml_cnn": 0.0190})
    with pytest.raises(ValueError, match="no rows match"):
        candidates_from_csv([other], [sample_b])


# --------------------------------------------------------------------------------------------------
# downstream curve smoke test (needs holowizard.core)
# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_downstream_curve_smoke(tiny_dataset_dir) -> None:
    pytest.importorskip("holowizard")
    from src.data.forge_samples import read_forge_samples
    from src.eval.downstream import downstream_curve

    sample = read_forge_samples([tiny_dataset_dir / "test.hdf5"], n_per_file=1)[0]
    images: dict[str, np.ndarray] = {}
    rows = downstream_curve(sample, errors_pct=(-10.0, 0.0, 10.0), preset=SMOKE_PRESET, border=4, threads=2, images=images)
    assert [row["candidate"] for row in rows] == ["-10%", TRUE_CANDIDATE, "+10%"]
    assert set(images) == {"-10%", TRUE_CANDIDATE, "+10%"} and images[TRUE_CANDIDATE].shape == sample.shape
    geom = Geometry.from_sample(sample)
    for row, err in zip(rows, (-10.0, 0.0, 10.0)):
        assert row["fr_err_pct_nominal"] == err
        assert row["fr_used"] == pytest.approx(sample.fr * (1 + err / 100))
        assert row["rel_err_fr_pct"] == pytest.approx(err, abs=1e-6)
        assert row["blur_px"] == pytest.approx(np.sqrt(abs(err / 100) / sample.fr))
        assert row["z01_used_mm"] == pytest.approx(geom.z01_from_fresnel(row["fr_used"]))
        for key in ("nrmse", "nrmse_offset_free", "nrmse_grad", "pearson", "core_loss_final", "data_residual_true_fr", "runtime_s"):
            assert np.isfinite(row[key]), key
        assert row["core_loss_final"] >= 0 and row["data_residual_true_fr"] > 0 and row["runtime_s"] > 0
        assert row["preset"] == SMOKE_PRESET and isinstance(row["diverged"], bool)
    true_row = rows[1]
    for key in ("nrmse_rel_true", "nrmse_grad_rel_true", "data_residual_true_fr_rel_true", "core_loss_final_rel_true"):
        assert true_row[key] == pytest.approx(1.0)
        assert np.isfinite(rows[0][key]) and np.isfinite(rows[2][key])
    assert true_row["nrmse"] < 1.0 and true_row["pearson"] > 0.5
