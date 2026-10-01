"""Forward shapes of both architectures for several input resolutions, plus metrics/plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.cnn import ARCHITECTURES, AutofocusCNN, build_model
from src.utils.metrics import metrics_in_physical_units, regression_metrics
from src.utils.physics import fresnel_number
from src.utils.plotting import plot_error_vs_z01, plot_example_holograms, plot_loss_curves, plot_true_vs_pred
from src.utils.targets import TargetScaler


@pytest.mark.parametrize("arch", ARCHITECTURES)
@pytest.mark.parametrize("size", [128, 256])
def test_forward_shapes(arch: str, size: int) -> None:
    model = build_model({"model": {"arch": arch, "conv_channels": [8, 16, 32, 64], "fc_units": 32, "dropout_rate": 0.1}})
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(3, 1, size, size))
        single = model(torch.randn(1, 1, size, size))
    assert out.shape == (3,) and single.shape == (1,)
    assert torch.isfinite(out).all()


def test_cnn_block_count_follows_channels() -> None:
    model = AutofocusCNN(conv_channels=[4, 8, 16], fc_units=16, dropout_rate=0.0, pool_size=2)
    assert len(model.features) == 3
    assert model(torch.randn(2, 1, 64, 48)).shape == (2,)
    with pytest.raises(ValueError):
        AutofocusCNN(conv_channels=[])
    with pytest.raises(ValueError):
        build_model({"model": {"arch": "vit"}})


def test_cnn_is_trainable() -> None:
    torch.manual_seed(0)
    model = AutofocusCNN(conv_channels=[4, 8], fc_units=8, dropout_rate=0.0, pool_size=1)
    x = torch.randn(8, 1, 32, 32)
    y = x.mean(dim=(1, 2, 3))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < first


def test_regression_metrics_values() -> None:
    true = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 2.0, 2.5, 4.0])
    m = regression_metrics(pred, true)
    assert m["mae"] == pytest.approx(0.25) and m["bias"] == pytest.approx(0.0) and m["n"] == 4
    assert m["rmse"] == pytest.approx(np.sqrt(0.125))
    assert m["median_abs_err"] == pytest.approx(0.25)
    assert 0.0 < m["r2"] < 1.0
    with pytest.raises(ValueError):
        regression_metrics(pred[:2], true)


def test_physical_metrics_independent_of_target_mode() -> None:
    setup = {"z02_mm": 20000.0, "energy_kev": 11.0, "px_mm": 0.052}
    z01_true = np.array([60.0, 120.0, 200.0, 280.0])
    z01_pred = z01_true + np.array([1.0, -2.0, 0.5, -0.5])
    fr_true = fresnel_number(z01_true, 20000.0, 11.0, 0.052)
    fr_pred = fresnel_number(z01_pred, 20000.0, 11.0, 0.052)
    by_z01 = metrics_in_physical_units(z01_pred, z01_true, "z01_mm", setup)
    by_fr = metrics_in_physical_units(fr_pred, fr_true, "fr", setup)
    by_log = metrics_in_physical_units(np.log(fr_pred), np.log(fr_true), "log_fr", setup)
    assert by_z01["z01_mae_mm"] == pytest.approx(1.0)
    assert by_z01["z01_bias_mm"] == pytest.approx(-0.25)
    for key in ("z01_mae_mm", "z01_rmse_mm", "fr_rel_err_mean_pct"):
        assert by_fr[key] == pytest.approx(by_z01[key], rel=1e-5)
        assert by_log[key] == pytest.approx(by_z01[key], rel=1e-5)
    # per-sample z02 arrays are accepted too
    arr_setup = setup | {"z02_mm": np.full(4, 20000.0)}
    assert metrics_in_physical_units(z01_pred, z01_true, "z01_mm", arr_setup)["z01_mae_mm"] == pytest.approx(1.0)
    with pytest.raises(KeyError):
        metrics_in_physical_units(z01_pred, z01_true, "z01_mm", {"energy_kev": 11.0})


def test_physical_metrics_tolerate_invalid_z01_predictions() -> None:
    setup = {"z02_mm": 20000.0, "energy_kev": 11.0, "px_mm": 0.052}
    m = metrics_in_physical_units(np.array([-5.0, 150.0]), np.array([100.0, 150.0]), "z01_mm", setup)
    assert np.isfinite(m["fr_rel_err_mean_pct"]) and m["z01_mae_mm"] == pytest.approx(52.5)


def test_target_scaler_roundtrip() -> None:
    scaler = TargetScaler.from_values(np.array([-9.0, -8.0, -7.0]))
    t = torch.tensor([-9.0, -7.0])
    torch.testing.assert_close(scaler.inverse(scaler.transform(t)), t)
    assert TargetScaler.from_dict(scaler.to_dict()) == scaler
    assert TargetScaler.from_values(np.array([2.0, 2.0])).std == 1.0
    with pytest.raises(ValueError):
        TargetScaler(0.0, 0.0)


def test_plots_are_written(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    true = rng.uniform(50, 300, 40)
    pred = true + rng.normal(0, 5, 40)
    assert plot_loss_curves([1.0, 0.5, 0.3], [1.1, 0.6, 0.4], tmp_path / "loss.png", [1e-3, 5e-4, 1e-4]).is_file()
    assert plot_true_vs_pred(true, pred, tmp_path / "scatter.png", "z01", "mm").is_file()
    assert plot_error_vs_z01(true, pred, tmp_path / "bins.png", num_bins=5).is_file()
    assert plot_example_holograms(rng.random((3, 16, 16)), ["a", "b", "c"], tmp_path / "ex.png").is_file()
    with pytest.raises(ValueError):
        plot_example_holograms(rng.random((3, 16, 16)), ["a"], tmp_path / "bad.png")
