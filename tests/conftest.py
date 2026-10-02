"""Shared pytest fixtures: project root on ``sys.path`` and a tiny HoloForge dataset per session."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml  # noqa: E402


def tiny_data_config(num_samples: dict[str, int] | None = None, seeds: dict[str, int] | None = None) -> dict[str, Any]:
    """``configs/data_small.yaml`` shrunk to a 64 px grid (2048 px binned by 32) with few shapes/samples."""
    cfg = copy.deepcopy(load_yaml(PROJECT_ROOT / "configs" / "data_small.yaml"))
    cfg["name"] = "tiny"
    cfg["num_samples"] = num_samples or {"train": 6, "val": 3, "test": 3}
    cfg["seeds"] = seeds or {"train": 11, "val": 22, "test": 33}
    cfg["setup"]["detector_size"] = 2048
    cfg["setup"]["downsample_factor"] = 32
    cfg["setup"]["padding_factor"] = 2
    cfg["phantom"]["num_shapes"] = [1, 2]
    cfg["phantom"]["radius_range_px"] = [64, 512]
    cfg["phantom"]["size_range_px"] = [64, 1024]
    cfg["store"]["phantom"] = True  # ground-truth phase for the downstream/reconstruction tests
    return cfg


@pytest.fixture(scope="session")
def tiny_config() -> dict[str, Any]:
    return tiny_data_config()


@pytest.fixture(scope="session")
def tiny_dataset_dir(tmp_path_factory: pytest.TempPathFactory, tiny_config: dict[str, Any]) -> Path:
    """Generate the tiny dataset once per test session and return its directory."""
    from src.data.generate_data import generate_dataset

    out_dir = tmp_path_factory.mktemp("tiny_dataset")
    generate_dataset(copy.deepcopy(tiny_config), out_dir, overwrite=True)
    return out_dir
