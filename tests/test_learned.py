import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from holofocus.baseline.results import append_row
from holofocus.dataset import DatasetConfig, write_split
from holofocus.forward import object_from_phase, simulate_hologram
from holofocus.learned import (
    ImageCNN,
    LearnedAutofocus,
    LearnedConfig,
    ModelConfig,
    Representation,
    RepresentationConfig,
    SpectrumCNN,
    bootstrap_ci,
    build_model,
    calibration_scale,
    compare_methods,
    d4_transform,
    gaussian_nll,
    interval_metrics,
    load_inputs,
    offset_estimate,
    point_metrics,
    predict_split,
    random_d4,
    read_history,
    read_predictions,
    train_model,
)
from holofocus.learned.cli import main
from holofocus.learned.compare import read_comparison, write_comparison
from holofocus.learned.inference import PREDICTION_FIELDS
from holofocus.learned.metrics import (
    grouped_point_metrics,
    interval_quantile,
    photon_bin_labels,
    photon_bin_names,
)
from holofocus.learned.training import HISTORY_FIELDS, warmup_cosine

ROOT = Path(__file__).resolve().parents[1]
CI_DATASET = ROOT / "configs" / "dataset" / "ci.yaml"
CI_LEARNED = ROOT / "configs" / "learned" / "ci.yaml"
TINY_IMAGE = {
    "name": "tiny-image",
    "representation": {"kind": "image", "binning": 4},
    "model": {"channels": [4, 8], "hidden": 8},
    "training": {"epochs": 1, "batch_size": 4, "warmup_epochs": 0},
    "device": "cpu",
}


def _row(index, error_um, z01_true=0.1):
    """Baseline result row (``holofocus.baseline.results``) with a given error."""
    return {
        "index": index,
        "phantom_type": 1,
        "photons": 1e4,
        "z01_true": z01_true,
        "z01_guess": z01_true - 1e-3,
        "dz01_true": 1e-3,
        "z01_est": z01_true + error_um * 1e-6,
        "error_um": error_um,
        "abs_error_um": abs(error_um),
        "n_reconstructions": 10 + index,
        "n_evaluations": 12 + index,
        "nm_iterations": 7,
        "converged": True,
        "seconds": 100.0 + index,
        "seconds_per_reconstruction": 10.0,
        "mfe_est": 0.5,
        "mfe_true": 1.0,
        "phase_rmse_est": 0.1,
        "phase_rmse_true": 0.1,
    }


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    root = tmp_path_factory.mktemp("learned-data")
    config = DatasetConfig.from_yaml(CI_DATASET)
    sizes = {"train": 16, "val": 6, "test": 4}
    for split, n in sizes.items():
        write_split(config, split, root / f"{split}.h5", n_samples=n, device="cpu", log=None)
    return {split: root / f"{split}.h5" for split in sizes}


def test_repository_configs_and_validation():
    for name in ("spectrum-cnn", "spectrum-cnn-wide", "image-cnn", "ci"):
        config = LearnedConfig.from_yaml(ROOT / "configs" / "learned" / f"{name}.yaml")
        assert LearnedConfig.from_dict(yaml.safe_load(config.to_yaml())) == config
        assert config.target_scale == 5e-3
    spectrum = LearnedConfig.from_yaml(ROOT / "configs" / "learned" / "spectrum-cnn.yaml")
    assert spectrum.representation.bins == 1024 and spectrum.model.channels == (16, 32, 64, 64)
    wide = LearnedConfig.from_yaml(ROOT / "configs" / "learned" / "spectrum-cnn-wide.yaml")
    assert wide.model.channels == (32, 64, 128, 128)
    assert dataclasses.replace(wide.model, channels=spectrum.model.channels) == spectrum.model
    assert (wide.representation, wide.training) == (spectrum.representation, spectrum.training)
    bad = [
        {"nme": "typo"},
        {"representation": {"kind": "wavelet"}},
        {"representation": {"f2_max": 0.7}},
        {"model": {"kernel_size": 4}},
        {"model": {"channels": []}},
        {"training": {"beta": 1.5}},
        {"training": {"batch_size": 1}},
        {"target_scale": 0.0},
    ]
    for data in bad:
        with pytest.raises(ValueError):
            LearnedConfig.from_dict(data)


def test_image_representation_bins_and_standardizes():
    holograms = 1 + 0.1 * torch.rand(3, 16, 16, dtype=torch.float64)
    rep = Representation(RepresentationConfig(kind="image", binning=2), (16, 16))
    x = rep(holograms)
    assert x.shape == (3, 1, 8, 8) and rep.input_shape == (1, 8, 8)
    binned = holograms.reshape(3, 8, 2, 8, 2).mean(dim=(2, 4)).float()
    expected = (binned - binned.mean(dim=(1, 2), keepdim=True)) / binned.std(
        dim=(1, 2), keepdim=True
    )
    torch.testing.assert_close(x[:, 0], expected, rtol=1e-4, atol=1e-4)
    assert rep(holograms[0]).shape == (1, 1, 8, 8)
    with pytest.raises(ValueError):
        Representation(RepresentationConfig(kind="image", binning=3), (16, 16))
    with pytest.raises(ValueError):
        rep(torch.ones(1, 8, 8))


def test_spectrum_has_its_minima_at_multiples_of_the_fresnel_number():
    fr, n, bins, f2_max = 0.02, 256, 250, 0.25
    phase = 1e-3 * torch.randn((n, n), generator=torch.Generator().manual_seed(1))
    hologram = simulate_hologram(object_from_phase(phase.double(), None), fr)
    rep = Representation(RepresentationConfig(kind="spectrum", bins=bins, f2_max=f2_max), (n, n))
    profile = rep(hologram)[0, 0].numpy()
    assert profile.shape == (bins,)
    assert abs(profile.mean()) < 1e-5 and profile.std(ddof=1) == pytest.approx(1.0, rel=1e-4)
    width = f2_max / bins
    for k in (1, 2, 3, 4):
        center = int(round(k * fr / width))
        window = profile[center - 6 : center + 7]
        assert abs(int(np.argmin(window)) - 6) <= 1, k
        assert window.min() < profile[center + 5] - 1.0


def test_spectrum_is_invariant_under_rotations_and_flips():
    hologram = 1 + 0.05 * torch.rand((64, 64), generator=torch.Generator().manual_seed(2))
    rep = Representation(RepresentationConfig(kind="spectrum", bins=64, f2_max=0.25), (64, 64))
    reference = rep(hologram)
    for element in range(8):
        torch.testing.assert_close(rep(d4_transform(hologram, element)), reference)
    with pytest.raises(ValueError):
        Representation(RepresentationConfig(kind="spectrum", bins=4096, f2_max=0.25), (32, 32))


def test_d4_group():
    x = torch.arange(9.0).reshape(1, 1, 3, 3)
    images = [d4_transform(x, e) for e in range(8)]
    assert len({tuple(im.flatten().tolist()) for im in images}) == 8
    torch.testing.assert_close(d4_transform(d4_transform(x, 1), 3), x)
    batch = torch.rand(16, 1, 5, 5)
    out = random_d4(batch, torch.Generator().manual_seed(0))
    for b in range(16):
        assert any(torch.equal(out[b], d4_transform(batch[b], e)) for e in range(8))
    with pytest.raises(ValueError):
        random_d4(torch.rand(2, 1, 4, 5), torch.Generator())


def test_models_output_a_positive_variance():
    config = ModelConfig(channels=(4, 8), hidden=8, pool_to=4, min_sigma=0.01)
    for model, x in (
        (SpectrumCNN(config), torch.randn(3, 1, 64)),
        (ImageCNN(config), torch.randn(3, 1, 32, 32)),
    ):
        mean, var = model.eval()(x)
        assert mean.shape == var.shape == (3,)
        assert torch.all(var >= 0.01**2)
    spectrum = LearnedConfig.from_dict({"model": {"channels": [4]}})
    image = LearnedConfig.from_dict({"representation": {"kind": "image"}})
    assert isinstance(build_model(spectrum), SpectrumCNN)
    assert isinstance(build_model(image), ImageCNN)


def test_gaussian_nll_and_beta_weighting():
    mean = torch.tensor([0.1, -0.3], requires_grad=True)
    var = torch.tensor([0.04, 0.25])
    target = torch.tensor([0.0, 0.2])
    expected = 0.5 * (torch.log(var) + (target - mean) ** 2 / var)
    torch.testing.assert_close(gaussian_nll(mean, var, target), expected.mean())
    (grad,) = torch.autograd.grad(gaussian_nll(mean, var, target, beta=1.0), mean)
    torch.testing.assert_close(grad, (mean - target).detach() / 2)


def test_warmup_cosine_schedule():
    factors = [warmup_cosine(s, 2, 10) for s in range(10)]
    assert factors[:2] == [0.5, 1.0]
    assert factors[2] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(factors[2:], factors[3:], strict=False))
    assert warmup_cosine(10, 2, 10) == pytest.approx(0.01)


def test_point_and_interval_metrics():
    errors = [30.0, -80.0, 300.0, -1200.0]
    m = point_metrics(errors)
    assert m["n"] == 4 and m["bias"] == pytest.approx(-237.5)
    assert m["mae"] == pytest.approx(402.5)
    assert m["rmse"] == pytest.approx(math.sqrt((900 + 6400 + 90000 + 1440000) / 4))
    assert m["median_abs"] == pytest.approx(190.0)
    assert m["within"] == {"50": 0.25, "100": 0.5, "250": 0.5, "1000": 0.75}
    assert point_metrics([]) == {"n": 0}
    assert interval_quantile(0.95) == pytest.approx(1.959964, abs=1e-6)
    rng = np.random.default_rng(0)
    sigma = rng.uniform(50, 500, size=20000)
    sample = rng.normal(0, sigma)
    interval = interval_metrics(sample, sigma)
    assert interval["coverage"]["68"] == pytest.approx(0.68, abs=0.01)
    assert interval["coverage"]["95"] == pytest.approx(0.95, abs=0.005)
    assert interval["z_rms"] == pytest.approx(1.0, abs=0.02)
    assert calibration_scale(sample, sigma / 2) == pytest.approx(2.0, rel=0.02)
    lo, hi = bootstrap_ci(sample[:200] / sigma[:200], resamples=2000)
    assert lo < np.mean(sample[:200] / sigma[:200]) < hi


def test_grouping_by_photons_and_type():
    assert photon_bin_names() == ["1e3-3e3", "3e3-1e4", "1e4-3e4", "3e4-1e5"]
    assert photon_bin_labels([1e3, 2999.0, 5e4, 1e5, 2e5]) == [
        "1e3-3e3",
        "1e3-3e3",
        "3e4-1e5",
        "3e4-1e5",
        "3e4-1e5",
    ]
    grouped = grouped_point_metrics([10.0, -20.0, 40.0], ["b", "a", "b"], order=["b", "a", "c"])
    assert list(grouped) == ["b", "a"]
    assert grouped["b"]["mae"] == pytest.approx(25.0)


def test_offset_estimate_and_comparison(tmp_path):
    offset = offset_estimate([100.0, 140.0, 120.0, 80.0], resamples=2000)
    assert offset["mean"] == pytest.approx(110.0)
    assert offset["se"] == pytest.approx(np.std([100, 140, 120, 80], ddof=1) / 2)
    assert offset["ci"][0] <= 110.0 <= offset["ci"][1]
    with pytest.raises(ValueError):
        offset_estimate([1.0])

    baseline = [_row(1, 150.0), _row(0, 90.0)]
    predictions = [
        {"index": i, "z01_true": 0.1, "error_um": e, "sigma": 2e-4, "seconds": 0.01}
        for i, e in ((0, -40.0), (1, 60.0), (2, 5.0))
    ]
    rows, summary = compare_methods(predictions, baseline, offset)
    assert [r["index"] for r in rows] == [0, 1]
    assert [r["corrected_error_um"] for r in rows] == pytest.approx([-20.0, 40.0])
    assert summary["methods"]["baseline_raw"]["error_um"]["mae"] == pytest.approx(120.0)
    assert summary["methods"]["baseline_corrected"]["error_um"]["mae"] == pytest.approx(30.0)
    assert summary["methods"]["learned"]["interval"]["coverage"]["68"] == 1.0
    assert summary["paired"]["learned_vs_raw"]["mae_difference_um"] == pytest.approx(-70.0)
    assert summary["paired"]["learned_vs_raw"]["fraction_first_better"] == 1.0
    assert summary["speedup_median"] == pytest.approx(100.5 / 0.01)
    assert summary["methods"]["baseline_raw"]["reconstructions"]["mean"] == 10.5

    warm = [{**_row(0, 70.0), "n_reconstructions": 6}, {**_row(1, 20.0), "n_reconstructions": 5}]
    rows, summary = compare_methods(predictions, baseline, offset, warm)
    assert [r["warm_error_um"] for r in rows] == [70.0, 20.0]
    assert [r["warm_corrected_error_um"] for r in rows] == pytest.approx([-40.0, -90.0])
    assert summary["methods"]["warm_start"]["reconstructions"]["mean"] == 5.5
    assert summary["methods"]["warm_start_corrected"]["error_um"]["mae"] == pytest.approx(65.0)
    write_comparison(tmp_path, rows, summary)
    assert read_comparison(tmp_path) == rows
    assert json.loads((tmp_path / "comparison.json").read_text())["n_samples"] == 2
    rows, _ = compare_methods(predictions, baseline, offset)
    write_comparison(tmp_path, rows, summary)
    assert read_comparison(tmp_path / "comparison.csv")[0]["warm_error_um"] is None
    with pytest.raises(ValueError):
        compare_methods(predictions[1:], baseline, offset)
    with pytest.raises(ValueError):
        compare_methods([{**p, "z01_true": 0.2} for p in predictions], baseline, offset)


def test_load_inputs(splits):
    config = LearnedConfig.from_yaml(CI_LEARNED)
    rep = Representation(config.representation, (128, 128))
    data = load_inputs(splits["val"], rep, 5e-3, limit=4)
    assert data.inputs.shape == (4, 1, 128) and data.inputs.dtype == torch.float32
    np.testing.assert_array_equal(data.indices, [0, 1, 2, 3])
    assert torch.all(data.targets.abs() <= 1.0)
    image = Representation(RepresentationConfig(kind="image", binning=2), (128, 128))
    picked = load_inputs(splits["val"], image, 5e-3, indices=[5, 1])
    assert picked.inputs.shape == (2, 1, 64, 64) and picked.inputs.dtype == torch.float16
    assert picked.labels["dz01"].tolist() == pytest.approx(
        (picked.targets.double() * 5e-3).tolist(), abs=1e-9
    )


def test_train_predict_and_reload(splits, tmp_path):
    config = LearnedConfig.from_yaml(CI_LEARNED)
    out = tmp_path / "spectrum"
    messages = []
    summary = train_model(config, splits["train"], splits["val"], out, log=messages.append)
    history = read_history(out / "history.csv")
    assert [h["epoch"] for h in history] == [1, 2, 3]
    assert set(history[0]) == set(HISTORY_FIELDS)
    assert 1 <= summary["best_epoch"] <= 3 and summary["sigma_scale"] > 0
    assert summary["validation"]["interval"]["z_rms"] == pytest.approx(1.0, rel=1e-6)
    assert json.loads((out / "summary.json").read_text()) == summary
    info = yaml.safe_load((out / "run.yaml").read_text())
    assert info["n_train"] == 16 and info["n_val"] == 6 and info["input_shape"] == [1, 128]
    assert len(info["datasets"]["train"][0]["sha256"]) == 64
    assert len(messages) == 5
    with pytest.raises(ValueError):
        train_model(config, splits["train"], splits["val"], out, log=None)

    autofocus = LearnedAutofocus.load(out)
    assert autofocus.sigma_scale == pytest.approx(summary["sigma_scale"])
    result = predict_split(autofocus, splits["test"], tmp_path / "pred", model_path=out, log=None)
    rows = read_predictions(tmp_path / "pred" / "predictions.csv")
    assert result["n_samples"] == len(rows) == 4
    assert list(rows[0]) == list(PREDICTION_FIELDS)
    for row in rows:
        assert row["z01_est"] == pytest.approx(row["z01_guess"] + row["dz01_est"], abs=1e-15)
        assert row["error_um"] == pytest.approx((row["z01_est"] - row["z01_true"]) * 1e6)
        assert row["sigma"] == pytest.approx(autofocus.sigma_scale * row["sigma_raw"])
        assert row["fresnel_error_rel"] == pytest.approx(
            row["fresnel_number_est"] / row["fresnel_number_true"] - 1
        )
        assert row["seconds"] > 0
    assert set(result["by_phantom_type"]) <= {"triangle", "cell", "ball", "mixed"}
    meta = yaml.safe_load((tmp_path / "pred" / "run.yaml").read_text())
    assert len(meta["model_sha256"]) == 64 and meta["dataset"]["split"] == "test"
    dz01, sigma, raw = autofocus.estimate(torch.ones(2, 128, 128))
    assert dz01.shape == sigma.shape == raw.shape == (2,)

    image = LearnedConfig.from_dict(TINY_IMAGE)
    image_summary = train_model(image, splits["train"], splits["val"], tmp_path / "image", log=None)
    assert image_summary["best_epoch"] == 1
    image_autofocus = LearnedAutofocus.load(tmp_path / "image" / "model.pt")
    assert isinstance(image_autofocus.model, ImageCNN)
    assert image_autofocus.representation.input_shape == (1, 32, 32)


def test_cli(splits, tmp_path, capsys):
    run = tmp_path / "run"
    args = ["train", str(CI_LEARNED), "--train", str(splits["train"]), "--val", str(splits["val"])]
    threads = torch.get_num_threads()
    try:
        options = ["--limit", "8", "--threads", "1", "--seed", "3"]
        assert main([*args, "--out", str(run), *options]) == 0
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(threads)
    assert yaml.safe_load((run / "run.yaml").read_text())["n_train"] == 8
    assert yaml.safe_load((run / "config.yaml").read_text())["training"]["seed"] == 3
    two = [str(splits["train"]), str(splits["test"])]
    both = tmp_path / "two-files"
    assert (
        main(
            [
                "train",
                str(CI_LEARNED),
                "--train",
                *two,
                "--val",
                str(splits["val"]),
                "--out",
                str(both),
                "--limit",
                "18",
            ]
        )
        == 0
    )
    info = yaml.safe_load((both / "run.yaml").read_text())
    assert info["n_train"] == 18
    assert [d["split"] for d in info["datasets"]["train"]] == ["train", "test"]
    assert main([*args, "--out", str(run)]) == 1
    assert main([*args, "--out", str(run), "--limit", "8", "--overwrite"]) == 0
    pred = tmp_path / "pred"
    assert main(["predict", str(run), "--data", str(splits["test"]), "--out", str(pred)]) == 0
    assert main(["summary", str(pred)]) == 0
    capsys.readouterr()
    assert main(["summary", str(tmp_path)]) == 1
    assert main(["predict", str(tmp_path), "--data", str(splits["test"]), "--out", str(pred)]) == 1

    predictions = read_predictions(pred / "predictions.csv")
    baseline, validation = tmp_path / "baseline", tmp_path / "baseline-val"
    for directory, rows in ((baseline, predictions[:3]), (validation, predictions)):
        directory.mkdir()
        for p in rows:
            append_row(
                directory / "results.csv", _row(p["index"], 100.0 + p["index"], p["z01_true"])
            )
    out = tmp_path / "comparison"
    compare = ["compare", "--learned", str(pred / "predictions.csv"), "--baseline", str(baseline)]
    assert main([*compare, "--baseline-val", str(validation), "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["n_samples"] == 3 and printed["offset_um"]["mean"] == pytest.approx(101.5)
    summary = json.loads((out / "comparison.json").read_text())
    assert summary["indices"] == [0, 1, 2] and summary["sources"]["warm"] is None
    lines = (out / "comparison.csv").read_text().splitlines()
    assert len(lines) == 4 and lines[0].startswith("index,phantom_type")
    no_warm = [*compare, "--baseline-val", str(baseline), "--out", str(out), "--warm", str(run)]
    assert main(no_warm) == 1
