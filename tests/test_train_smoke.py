"""End-to-end smoke test: two epochs on the tiny dataset produce ``best.pt`` and ``results.json``;
``src.evaluate`` reloads the checkpoint and writes ``eval_results.json``."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from src.evaluate import evaluate_checkpoint, load_checkpoint
from src.train import train
from src.utils.config import PROJECT_ROOT, load_yaml


def _smoke_config(runs_dir: Path, **overrides) -> dict:
    config = copy.deepcopy(load_yaml(PROJECT_ROOT / "configs" / "base.yaml"))
    config["data"].update({"batch_size": 3, "num_workers": 0})
    config["model"].update({"conv_channels": [4, 8], "fc_units": 8, "dropout_rate": 0.0, "pool_size": 2})
    config["train"].update({"epochs": 2, "early_stopping_patience": 0, "device": "cpu", "seed": 1})
    config["paths"]["runs_dir"] = str(runs_dir)
    for section, values in overrides.items():
        config[section].update(values)
    return config


def test_train_writes_checkpoint_and_results(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    config = _smoke_config(tmp_path / "runs")
    results = train(config, data_dir=tiny_dataset_dir, run_name="smoke")
    run_dir = tmp_path / "runs" / "smoke"

    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "results.json").is_file()
    assert (run_dir / "history.json").is_file()
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "tb").is_dir()
    for figure in results["figures"].values():
        assert Path(figure).is_file()

    saved = json.loads((run_dir / "results.json").read_text())
    assert saved["epochs_run"] == 2 and saved["best_epoch"] in (1, 2)
    assert saved["test_metrics"]["target_space"]["n"] == 3
    assert "z01_mae_mm" in saved["test_metrics"]["physical"]
    assert saved["inference_time"]["ms_per_hologram_batch_1"] > 0

    model, checkpoint = load_checkpoint(run_dir / "best.pt", torch.device("cpu"))
    assert checkpoint["target_mode"] == "log_fr"
    assert checkpoint["setup_constants"]["z02_mm"] == 20000.0
    assert checkpoint["input_shape"] == [1, 64, 64]
    with torch.no_grad():
        assert model(torch.randn(1, 1, 64, 64)).shape == (1,)


def test_evaluate_cli_on_checkpoint(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    config = _smoke_config(tmp_path / "runs", train={"loss": "huber", "scheduler": "plateau"}, data={"target_mode": "z01_mm"})
    train(config, data_dir=tiny_dataset_dir, run_name="smoke_eval")
    results = evaluate_checkpoint(
        tmp_path / "runs" / "smoke_eval" / "best.pt",
        tiny_dataset_dir / "val.hdf5",
        out_dir=tmp_path / "eval",
        device_name="cpu",
    )
    assert (tmp_path / "eval" / "eval_results.json").is_file()
    assert results["target_mode"] == "z01_mm"
    assert results["metrics"]["physical"]["n"] == 3
    assert results["inference_time"]["ms_per_hologram_batch_3"] > 0


def test_early_stopping_and_resnet(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    config = _smoke_config(tmp_path / "runs", model={"arch": "resnet18"}, train={"epochs": 3, "early_stopping_patience": 1})
    results = train(config, data_dir=tiny_dataset_dir, run_name="smoke_resnet")
    assert results["epochs_run"] <= 3
    assert results["num_parameters"] > 1_000_000


def test_invalid_loss_raises(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    config = _smoke_config(tmp_path / "runs", train={"loss": "l1"})
    with pytest.raises(ValueError, match="train.loss"):
        train(config, data_dir=tiny_dataset_dir, run_name="bad_loss")
