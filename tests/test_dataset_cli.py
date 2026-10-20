from pathlib import Path

import h5py
import yaml

from holofocus.dataset.cli import main

CI_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dataset" / "ci.yaml"


def test_generate_skip_and_overwrite(tmp_path, capsys):
    out = tmp_path / "ci"
    args = ["generate", str(CI_CONFIG), "--out", str(out), "--splits", "val", "--limit", "3"]
    assert main(args) == 0
    with h5py.File(out / "val.h5", "r") as fh:
        assert fh["hologram"].shape[0] == 3
    assert (out / "config.yaml").exists()
    assert not (out / "train.h5").exists()

    assert main(args) == 0
    assert "up to date" in capsys.readouterr().out

    changed = yaml.safe_load(CI_CONFIG.read_text())
    changed["seed"] = 8
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump(changed))
    clash = ["generate", str(other), "--out", str(out), "--splits", "val", "--limit", "3"]
    assert main(clash) == 1
    assert yaml.safe_load((out / "config.yaml").read_text())["seed"] == 7
    assert main([*clash, "--overwrite"]) == 0
    with h5py.File(out / "val.h5", "r") as fh:
        assert fh.attrs["seed"] == 8


def test_unknown_split(tmp_path):
    args = ["generate", str(CI_CONFIG), "--out", str(tmp_path), "--splits", "holdout"]
    assert main(args) == 2


def test_inspect(tmp_path, capsys):
    main(["generate", str(CI_CONFIG), "--out", str(tmp_path), "--splits", "test", "--limit", "2"])
    capsys.readouterr()
    assert main(["inspect", str(tmp_path / "test.h5")]) == 0
    text = capsys.readouterr().out
    assert "2 samples" in text and "dz01:" in text and "hologram_clean" in text
