import json
import math
from pathlib import Path

import pytest
import yaml

from holofocus.baseline import BaselineConfig, run_baseline
from holofocus.baseline.cli import main
from holofocus.baseline.results import (
    RESULT_FIELDS,
    append_row,
    append_trace,
    read_rows,
    read_traces,
    summarize,
)
from holofocus.baseline.runner import select_indices
from holofocus.dataset import DatasetConfig, write_split

ROOT = Path(__file__).resolve().parents[1]
CI_DATASET = ROOT / "configs" / "dataset" / "ci.yaml"
TINY = {
    "name": "tiny",
    "reconstruction": {
        "stages": [
            {"iterations": 6, "step": 0.9, "down_sampling": 16, "momentum_fwhm": 8.0},
            {"iterations": 4, "step": 1.1, "down_sampling": 4, "l2_weight": 1.0},
        ],
        "device": "cpu",
    },
    "search": {"confidence": 5.0e-3, "xatol": 2.0e-3},
}


@pytest.fixture(scope="module")
def split_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "test.h5"
    config = DatasetConfig.from_yaml(CI_DATASET)
    write_split(config, "test", path, n_samples=3, device="cpu", log=None)
    return path


def test_repository_configs():
    paper = BaselineConfig.from_yaml(ROOT / "configs" / "baseline" / "paper2-cpu.yaml")
    settings = paper.reconstruction.settings(4)
    assert [s.down_sampling for s in settings.stages] == [4, 1, 1]
    assert [s.iterations for s in settings.stages] == [700, 300, 500]
    assert (settings.fade_width, settings.padding_factor) == (20, 2.0)
    assert (paper.search.confidence, paper.search.xatol) == (5e-3, 1e-4)
    ci = BaselineConfig.from_yaml(ROOT / "configs" / "baseline" / "ci.yaml")
    ci_settings = ci.reconstruction.settings(16)
    assert [s.down_sampling for s in ci_settings.stages] == [1, 1, 1]
    assert [s.iterations for s in ci_settings.stages] == [35, 15, 25]
    assert all(isinstance(s.iterations, int) for s in ci_settings.stages)
    assert ci_settings.fade_width == 5


def test_config_round_trip_and_validation():
    config = BaselineConfig.from_dict(TINY)
    assert BaselineConfig.from_dict(yaml.safe_load(config.to_yaml())) == config
    assert config.reconstruction.full_resolution_stages()[1].l2_weight == 1.0
    bad = [
        {"nme": "typo"},
        {"reconstruction": {"padding": 2.0}},
        {"reconstruction": {"stages": [{"iterations": 5, "step": 1.0, "fwhm": 2.0}]}},
        {"reconstruction": {"stages": [{"iterations": 5.5, "step": 1.0}]}},
        {"reconstruction": {"stages": []}},
        {"reconstruction": {"schedule": "paper1"}},
        {"reconstruction": {"precision": "half"}},
        {"reconstruction": {"update": "nesterov"}},
        {"search": {"confidence": 0.0}},
    ]
    for data in bad:
        with pytest.raises(ValueError):
            BaselineConfig.from_dict(data)


def _row(index, error_um, mfe_true=1.0, converged=True):
    return {
        "index": index,
        "phantom_type": 1,
        "photons": 1e4,
        "z01_true": 0.1,
        "z01_guess": 0.1 - 1e-3,
        "dz01_true": 1e-3,
        "z01_est": 0.1 + error_um * 1e-6,
        "error_um": error_um,
        "abs_error_um": abs(error_um),
        "n_reconstructions": 10 + index,
        "n_evaluations": 12 + index,
        "nm_iterations": 7,
        "converged": converged,
        "seconds": 100.0 + index,
        "seconds_per_reconstruction": 10.0,
        "mfe_est": 0.5,
        "mfe_true": mfe_true,
    }


def test_results_round_trip_and_summary(tmp_path):
    path = tmp_path / "results.csv"
    rows = [_row(0, 30.0), _row(1, -80.0, mfe_true=None), _row(2, 300.0, 0.25, False)]
    for row in rows:
        append_row(path, row)
    assert path.read_text().splitlines()[0] == ",".join(RESULT_FIELDS)
    assert read_rows(path) == rows
    summary = summarize(rows)
    assert summary["n_samples"] == 3
    assert summary["error_um"]["mae"] == pytest.approx(410 / 3)
    assert summary["error_um"]["rmse"] == pytest.approx(math.sqrt((900 + 6400 + 90000) / 3))
    assert summary["error_um"]["max_abs"] == 300.0
    assert summary["fraction_within_um"] == {"50": 1 / 3, "100": 2 / 3, "250": 2 / 3, "1000": 1.0}
    assert summary["guess_error_um"]["mae"] == pytest.approx(1000.0)
    assert summary["reconstructions"] == {"mean": 11.0, "min": 10, "max": 12}
    assert summary["seconds_per_autofocus"]["total"] == pytest.approx(303.0)
    assert summary["converged"] == pytest.approx(2 / 3)
    assert summary["mfe_est_le_mfe_true"] == 0.5
    assert summarize([]) == {"n_samples": 0}
    assert read_rows(tmp_path / "missing.csv") == []

    traces = tmp_path / "traces.jsonl"
    append_trace(traces, {"index": 0, "evaluations": [{"z01": 0.1, "mfe": 1.0}]})
    append_trace(traces, {"index": 1, "evaluations": []})
    assert [t["index"] for t in read_traces(traces)] == [0, 1]


def test_select_indices():
    assert select_indices(4, None, None) == [0, 1, 2, 3]
    assert select_indices(4, None, 2) == [0, 1]
    assert select_indices(4, [3, 1], None) == [3, 1]
    with pytest.raises(ValueError):
        select_indices(4, [4], None)


def test_run_baseline_writes_resumable_results(split_file, tmp_path):
    config = BaselineConfig.from_dict(TINY)
    out = tmp_path / "run"
    messages = []
    first = run_baseline(config, split_file, out, limit=1, log=messages.append)
    assert first["n_samples"] == 1
    second = run_baseline(config, split_file, out, limit=2, log=messages.append)
    assert second["n_samples"] == 2
    assert "1 done, 1 to go" in messages[2]

    rows = read_rows(out / "results.csv")
    assert [r["index"] for r in rows] == [0, 1]
    for row in rows:
        assert abs(row["z01_est"] - row["z01_guess"]) <= 5e-3 + 1e-12
        assert row["error_um"] == pytest.approx((row["z01_est"] - row["z01_true"]) * 1e6)
        assert row["z01_true"] - row["z01_guess"] == pytest.approx(row["dz01_true"])
        assert row["n_reconstructions"] >= 2
        assert row["n_evaluations"] >= row["n_reconstructions"]
        assert row["mfe_true"] is not None and row["mfe_est"] > 0
    traces = read_traces(out / "traces.jsonl")
    assert [t["index"] for t in traces] == [0, 1]
    assert len(traces[0]["evaluations"]) == rows[0]["n_evaluations"]
    assert json.loads((out / "summary.json").read_text()) == second
    info = yaml.safe_load((out / "run.yaml").read_text())
    assert info["binning"] == 16 and len(info["dataset"]["sha256"]) == 64
    assert info["asrm_settings"]["stages"][0]["down_sampling"] == 1

    other = BaselineConfig.from_dict({**TINY, "search": {"xatol": 1e-3}})
    with pytest.raises(ValueError):
        run_baseline(other, split_file, out, log=None)


def test_cli(split_file, tmp_path, capsys):
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(TINY))
    out = tmp_path / "cli-run"
    args = ["run", str(config), "--data", str(split_file), "--out", str(out)]
    assert main([*args, "--indices", "2", "--device", "cpu"]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed[printed.index("{") :])["n_samples"] == 1
    assert [r["index"] for r in read_rows(out / "results.csv")] == [2]
    assert main(["summary", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["n_samples"] == 1
    assert main(["summary", str(tmp_path)]) == 1
    ci = ROOT / "configs" / "baseline" / "ci.yaml"
    assert main(["run", str(ci), "--data", str(split_file), "--out", str(out)]) == 1
