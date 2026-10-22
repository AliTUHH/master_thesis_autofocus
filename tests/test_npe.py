import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("sbi")

from holofocus.dataset import DatasetConfig, write_split  # noqa: E402
from holofocus.learned import NPEConfig, Representation, load_inputs, read_predictions  # noqa: E402
from holofocus.learned.cli import main  # noqa: E402
from holofocus.learned.npe import (  # noqa: E402
    HISTORY_FIELDS,
    NPEAutofocus,
    build_estimator,
    check_npe_split,
    grid_log_density,
    predict_npe_split,
    prior_box,
    sbi_checks,
    train_npe,
)
from holofocus.learned.posterior import NPE_PREDICTION_FIELDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CI_DATASET = ROOT / "configs" / "dataset" / "ci.yaml"
NPE_CI = ROOT / "configs" / "learned" / "npe-ci.yaml"


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    root = tmp_path_factory.mktemp("npe-data")
    config = DatasetConfig.from_yaml(CI_DATASET)
    sizes = {"train": 16, "val": 6, "test": 4}
    for split, n in sizes.items():
        write_split(config, split, root / f"{split}.h5", n_samples=n, device="cpu", log=None)
    return {split: root / f"{split}.h5" for split in sizes}


def test_repository_configs_and_validation():
    configs = {
        name: NPEConfig.from_yaml(ROOT / "configs" / "learned" / f"{name}.yaml")
        for name in ("npe-nsf", "npe-mdn", "npe-ci")
    }
    for config in configs.values():
        assert NPEConfig.from_dict(yaml.safe_load(config.to_yaml())) == config
        assert config.target_scale == 5e-3
    nsf, mdn = configs["npe-nsf"], configs["npe-mdn"]
    assert (nsf.density.estimator, mdn.density.estimator) == ("nsf", "mdn")
    assert dataclasses.replace(mdn, name=nsf.name, density=nsf.density) == nsf
    assert nsf.embedding.channels == (32, 64, 128, 128) and nsf.representation.bins == 1024
    bad = [
        {"model": {"channels": [8]}},
        {"density": {"estimator": "maf"}},
        {"density": {"num_bins": 0}},
        {"embedding": {"features": 0}},
        {"embedding": {"kernel_size": 4}},
        {"training": {"clip_max_norm": 0.0}},
        {"training": {"batch_size": 1}},
        {"grid_points": 50},
        {"target_scale": -1.0},
    ]
    for data in bad:
        with pytest.raises(ValueError):
            NPEConfig.from_dict(data)


@pytest.mark.parametrize("estimator", ["nsf", "mdn"])
def test_estimator_grid_and_state_round_trip(estimator):
    config = NPEConfig.from_yaml(NPE_CI)
    config = dataclasses.replace(
        config, density=dataclasses.replace(config.density, estimator=estimator)
    )
    torch.manual_seed(0)
    first = build_estimator(config, (-1.0, 1.0), (1, 128)).eval()
    x = torch.randn(3, 1, 128)
    theta = torch.linspace(-1.0, 1.0, 11)
    grid = grid_log_density(first, x, theta, batch_size=2)
    assert grid.shape == (3, 11)
    for j, t in enumerate(theta):
        direct = first.log_prob(t.expand(1, 3, 1), x).detach()
        np.testing.assert_allclose(grid[:, j], direct[0].numpy(), rtol=1e-5, atol=1e-5)
    loss = first.loss(theta[:3, None], x)
    assert loss.shape == (3,) and torch.isfinite(loss).all()
    torch.manual_seed(1)
    second = build_estimator(config, (-1.0, 1.0), (1, 128)).eval()
    second.load_state_dict(first.state_dict())
    np.testing.assert_allclose(grid_log_density(second, x, theta), grid, rtol=1e-6)


def test_prior_box_reads_the_dz01_range(splits, tmp_path):
    assert prior_box([splits["train"], splits["val"]], 5e-3) == pytest.approx((-1.0, 1.0))
    wide = dataclasses.replace(
        DatasetConfig.from_yaml(CI_DATASET),
        sampling=dataclasses.replace(
            DatasetConfig.from_yaml(CI_DATASET).sampling, dz01_range=(-6e-3, 6e-3)
        ),
    )
    write_split(wide, "val", tmp_path / "wide.h5", n_samples=1, device="cpu", log=None)
    assert prior_box([tmp_path / "wide.h5"], 5e-3) == pytest.approx((-1.2, 1.2))
    with pytest.raises(ValueError):
        prior_box([splits["train"], tmp_path / "wide.h5"], 5e-3)


def test_train_predict_ensemble_and_checks(splits, tmp_path):
    config = NPEConfig.from_yaml(NPE_CI)
    messages = []
    summary = train_npe(config, splits["train"], splits["val"], tmp_path / "a", log=messages.append)
    history = (tmp_path / "a" / "history.csv").read_text().splitlines()
    assert history[0].split(",") == list(HISTORY_FIELDS) and len(history) == 4
    assert 1 <= summary["best_epoch"] <= 3 and summary["prior"] == [-1.0, 1.0]
    assert json.loads((tmp_path / "a" / "summary.json").read_text()) == summary
    info = yaml.safe_load((tmp_path / "a" / "run.yaml").read_text())
    assert info["n_train"] == 16 and info["n_val"] == 6 and info["input_shape"] == [1, 128]
    assert info["sbi_version"].startswith("0.27") and len(messages) == 5
    with pytest.raises(ValueError):
        train_npe(config, splits["train"], splits["val"], tmp_path / "a", log=None)

    autofocus = NPEAutofocus.load(tmp_path / "a")
    result = predict_npe_split(
        autofocus, splits["test"], tmp_path / "pred", model_paths=[tmp_path / "a"], log=None
    )
    rows = read_predictions(tmp_path / "pred" / "predictions.csv")
    assert result["n_samples"] == len(rows) == 4 and list(rows[0]) == list(NPE_PREDICTION_FIELDS)
    for row in rows:
        assert row["z01_est"] == pytest.approx(row["z01_guess"] + row["dz01_est"], abs=1e-15)
        assert row["error_um"] == pytest.approx((row["z01_est"] - row["z01_true"]) * 1e6)
        assert row["z01_lower_95"] <= row["z01_lower_68"] <= row["z01_est"]
        assert row["z01_est"] <= row["z01_upper_68"] <= row["z01_upper_95"]
        assert 0 <= row["pit"] <= 1 and 0 <= row["hpd_level"] <= 1
        assert 0 < row["prior_mass"] <= 1 + 1e-6 and row["seconds"] > 0
    meta = yaml.safe_load((tmp_path / "pred" / "run.yaml").read_text())
    assert meta["ensemble_size"] == 1 and len(meta["model_sha256"][0]) == 64
    estimate = autofocus.estimate(torch.ones(2, 128, 128))
    assert estimate["dz01"].shape == estimate["sigma"].shape == (2,)

    seed = dataclasses.replace(config, training=dataclasses.replace(config.training, seed=1))
    train_npe(seed, splits["train"], splits["val"], tmp_path / "b", log=None)
    ensemble = NPEAutofocus.load([tmp_path / "a", tmp_path / "b"])
    rep = Representation(config.representation, (128, 128))
    inputs = load_inputs(splits["test"], rep, 5e-3).inputs
    members = [NPEAutofocus.load(tmp_path / p).posterior_of_inputs(inputs) for p in ("a", "b")]
    np.testing.assert_allclose(
        ensemble.posterior_of_inputs(inputs).density,
        0.5 * (members[0].density + members[1].density),
        rtol=1e-6,
    )
    with pytest.raises(ValueError):
        sbi_checks(ensemble, inputs, torch.zeros(len(inputs)))
    other = dataclasses.replace(config, grid_points=301)
    train_npe(other, splits["train"], splits["val"], tmp_path / "c", log=None)
    with pytest.raises(ValueError):
        NPEAutofocus.load([tmp_path / "a", tmp_path / "c"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        checks = check_npe_split(
            autofocus, splits["val"], tmp_path / "check", 50, 100, model_path=tmp_path / "a"
        )
    assert checks["n"] == 6 and checks["num_posterior_samples"] == 100
    assert 0 <= checks["sbc"]["ks_p_value"] <= 1 and -1 <= checks["tarp"]["atc"] <= 1
    assert 0 <= checks["sbc"]["c2st_ranks_jittered"] <= 1
    ranks = (tmp_path / "check" / "sbi-ranks.csv").read_text().splitlines()
    assert ranks[0] == "index,rank,hpd_level" and len(ranks) == 7
    assert json.loads((tmp_path / "check" / "sbi-checks.json").read_text())["n"] == 6


def test_cli(splits, tmp_path, capsys):
    run = tmp_path / "run"
    args = ["npe-train", str(NPE_CI), "--train", str(splits["train"]), "--val", str(splits["val"])]
    threads = torch.get_num_threads()
    try:
        options = ["--limit", "8", "--seed", "2", "--threads", "1"]
        assert main([*args, "--out", str(run), *options]) == 0
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(threads)
    assert yaml.safe_load((run / "run.yaml").read_text())["n_train"] == 8
    assert yaml.safe_load((run / "config.yaml").read_text())["training"]["seed"] == 2
    assert main([*args, "--out", str(run)]) == 1
    assert main([*args, "--out", str(tmp_path / "two"), "--seed", "3"]) == 0
    pred = tmp_path / "pred"
    two = [str(run), str(tmp_path / "two")]
    assert main(["npe-predict", *two, "--data", str(splits["test"]), "--out", str(pred)]) == 0
    assert yaml.safe_load((pred / "run.yaml").read_text())["ensemble_size"] == 2
    missing = ["npe-predict", str(tmp_path), "--data", str(splits["test"]), "--out", str(pred)]
    assert main(missing) == 1
    check = ["npe-check", str(run), "--data", str(splits["val"]), "--out", str(tmp_path / "chk")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert main([*check, "--samples", "100"]) == 0
    assert (tmp_path / "chk" / "sbi-checks.json").exists()
    capsys.readouterr()
