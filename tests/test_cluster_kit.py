"""The Maxwell run kit (cluster/maxwell/) and scripts/check_cluster_run.py, on the CPU."""

from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from holofocus.dataset.config import DatasetConfig
from holofocus.dataset.storage import write_split

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "cluster" / "maxwell"
SCRIPTS = sorted(KIT.glob("*.sh")) + sorted((KIT / "jobs").glob("*.sh"))

spec = importlib.util.spec_from_file_location(
    "check_cluster_run", REPO / "scripts" / "check_cluster_run.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def kit_config(tmp_path: Path, **overrides: str) -> Path:
    values = {
        "HOLOFOCUS_SCRATCH": str(tmp_path / "scratch"),
        "HOLOFOCUS_DUST": str(tmp_path / "dust"),
        "HOLOFOCUS_ENV_KIND": "none",
        "HOLOFOCUS_ENV": "",
        "HOLOFOCUS_PYTHON": "python3",
        "HOLOFOCUS_CONDA": "conda",
        "HOLOFOCUS_MODULES": "",
        "HOLOFOCUS_TORCH_INDEX": "https://download.pytorch.org/whl/cu126",
        "HOLOFOCUS_TORCH_SPEC": "torch==2.14.1",
        "HOLOFOCUS_GPU_PARTITION": "gpu-part",
        "HOLOFOCUS_GPU_CONSTRAINT": "gpu-feature",
        "HOLOFOCUS_GPU_REQUEST": "--gres=gpu:1",
        "HOLOFOCUS_CPU_PARTITION": "cpu-part",
        "HOLOFOCUS_SBATCH_EXTRA": "--account=thesis",
    }
    values.update(overrides)
    path = tmp_path / "config.sh"
    path.write_text("".join(f'{k}="{v}"\n' for k, v in values.items()))
    return path


def kit(*args: str, config: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    environment = {
        **os.environ,
        "HOLOFOCUS_CONFIG": str(config),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        **(env or {}),
    }
    return subprocess.run(
        ["bash", str(KIT / "submit.sh"), *args],
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO,
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_submit_refuses_placeholders(tmp_path):
    result = kit("--dry-run", "gpu-check", config=KIT / "config.example.sh")
    assert result.returncode != 0
    for name in ("HOLOFOCUS_SCRATCH", "HOLOFOCUS_GPU_PARTITION", "HOLOFOCUS_MODULES"):
        assert name in result.stderr


def test_submit_dry_run_builds_the_sbatch_command(tmp_path):
    config = kit_config(tmp_path)
    result = kit(
        "--dry-run",
        "--array",
        "0-3",
        "--time",
        "02:00:00",
        "baseline",
        "baseline/x",
        "configs/baseline/ci.yaml",
        "ci/test.h5",
        "--chunks",
        "4",
        config=config,
    )
    assert result.returncode == 0, result.stderr
    command = result.stdout
    for part in (
        "sbatch --parsable",
        "--partition=gpu-part",
        "--constraint=gpu-feature",
        "--gres=gpu:1",
        "--account=thesis",
        "--array=0-3",
        "--time=02:00:00",
        "%x-%A_%a.out",
        "HOLOFOCUS_DEVICE=cuda",
        "jobs/baseline.sh baseline/x configs/baseline/ci.yaml",
    ):
        assert part in command, part
    cpu = kit("--dry-run", "--cpu", "--after", "17", "merge", "baseline/x", config=config).stdout
    assert "--partition=cpu-part" in cpu and "--dependency=afterok:17" in cpu
    assert "--constraint" not in cpu and "--gres" not in cpu and "HOLOFOCUS_DEVICE=cpu" in cpu
    assert "%x-%j.out" in cpu


def test_submit_rejects_unknown_jobs(tmp_path):
    result = kit("--dry-run", "nope", config=kit_config(tmp_path))
    assert result.returncode != 0 and "unknown job" in result.stderr
    assert "gpu-check" in result.stderr


def test_submit_prints_the_job_id(tmp_path):
    fake = tmp_path / "bin" / "sbatch"
    fake.parent.mkdir()
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$FAKE_SBATCH_ARGS"\necho "4711;maxwell"\n')
    fake.chmod(0o755)
    env = {
        "PATH": f"{fake.parent}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}"
        f"{os.environ['PATH']}",
        "FAKE_SBATCH_ARGS": str(tmp_path / "args"),
    }
    config = kit_config(tmp_path)
    result = kit("--cpu", "merge", "baseline/x", config=config, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "4711\n"
    assert str(tmp_path / "scratch" / "logs" / "holofocus-merge-4711.out") in result.stderr
    args = (tmp_path / "args").read_text().splitlines()
    assert args[0] == "--parsable" and args[-2:] == [str(KIT / "jobs" / "merge.sh"), "baseline/x"]
    assert (tmp_path / "scratch" / "logs").is_dir()
    array = kit("--array", "0-1", "baseline", "b", "c.yaml", "d.h5", config=config, env=env)
    assert array.stdout == "4711\n" and "holofocus-baseline-4711_<task>.out" in array.stderr


@pytest.mark.parametrize("script", sorted((KIT / "jobs").glob("*.sh")), ids=lambda p: p.stem)
def test_job_names_give_the_log_names(script):
    assert f"#SBATCH --job-name=holofocus-{script.stem}\n" in script.read_text()


def test_setup_env_refuses_python_older_than_3_12(tmp_path):
    old = tmp_path / "python3.11"
    old.write_text('#!/bin/bash\n[[ $1 == -V ]] && echo "Python 3.11.9" && exit 0\nexit 1\n')
    old.chmod(0o755)
    env = tmp_path / "env"
    config = kit_config(
        tmp_path, HOLOFOCUS_ENV_KIND="venv", HOLOFOCUS_ENV=str(env), HOLOFOCUS_PYTHON=str(old)
    )
    result = subprocess.run(
        ["bash", str(KIT / "setup_env.sh")],
        capture_output=True,
        text=True,
        env={**os.environ, "HOLOFOCUS_CONFIG": str(config)},
        cwd=REPO,
    )
    assert result.returncode != 0 and "older than Python 3.12 (Python 3.11.9)" in result.stderr
    assert not env.exists()


def test_local_pipeline_datagen_train_baseline_merge(tmp_path):
    config = kit_config(tmp_path)
    scratch, dust = tmp_path / "scratch", tmp_path / "dust"

    def ok(*args, env=None):
        result = kit("--local", *args, config=config, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    ok("datagen", "configs/dataset/ci.yaml", "--limit", "8")
    assert sorted(p.name for p in (scratch / "data" / "ci").glob("*.h5")) == [
        "test.h5",
        "train.h5",
        "val.h5",
    ]
    out = ok(
        "train",
        "learned/ci",
        "configs/learned/ci.yaml",
        "--train",
        "ci/train.h5",
        "--val",
        "ci/val.h5",
        "--test",
        "ci/test.h5",
    )
    assert "device    cpu" in out
    run = scratch / "runs" / "learned" / "ci"
    assert (run / "model.pt").exists() and (run / "test" / "predictions.csv").exists()
    assert (dust / "holofocus" / "runs" / "learned" / "ci" / "model.pt").exists()
    baseline = ["baseline", "baseline/ci", "configs/baseline/ci.yaml", "ci/test.h5", "--limit", "3"]
    for task in ("1", "0"):
        ok(*baseline, "--chunks", "2", env={"SLURM_ARRAY_TASK_ID": task})
    ok("merge", "baseline/ci")
    run = scratch / "runs" / "baseline" / "ci"
    assert sorted(p.name for p in run.glob("chunk-*")) == ["chunk-0-of-2", "chunk-1-of-2"]
    rows = list(csv.DictReader(open(run / "results.csv")))
    assert [int(r["index"]) for r in rows] == [0, 1, 2]
    kept = dust / "holofocus" / "runs" / "baseline" / "ci"
    assert (kept / "results.csv").exists() and (kept / "chunk-1-of-2" / "run.yaml").exists()
    assert not list(kept.rglob("*.h5"))
    for args, env in ((baseline, {"SLURM_ARRAY_TASK_ID": "0"}), ([*baseline, "--chunks=2"], None)):
        result = kit("--local", *args, config=config, env=env)
        assert result.returncode != 0 and "--chunks" in result.stderr


def small_split(tmp_path: Path, n: int = 2) -> Path:
    config = DatasetConfig.from_yaml(REPO / "configs" / "dataset" / "ci.yaml")
    return write_split(config, "test", tmp_path / "test.h5", n_samples=n, log=None)


def test_data_check_passes_warns_and_fails(tmp_path, capsys):
    split = small_split(tmp_path)
    reference = tmp_path / "fingerprint.csv"
    check.write_csv(reference, check.fingerprint_rows(split))
    report = check.Report()
    check.check_data(split, reference, report)
    assert [v for v, _, _ in report.lines] == ["PASS", "PASS"]

    rows = list(csv.DictReader(open(reference)))
    rows[1]["hologram_sha256"] = "0" * 64
    rows[1]["hologram_mean"] = repr(float(rows[1]["hologram_mean"]) * (1 + 1e-7))
    check.write_csv(tmp_path / "other.csv", rows)
    report = check.Report()
    check.check_data(split, tmp_path / "other.csv", report)
    assert [v for v, _, _ in report.lines] == ["PASS", "WARN"]

    rows[0]["dz01"] = repr(float(rows[0]["dz01"]) + 1e-9)
    check.write_csv(tmp_path / "wrong.csv", rows)
    report = check.Report()
    check.check_data(split, tmp_path / "wrong.csv", report)
    assert report.failed


def perturbed_copy(source: Path, target: Path, shift_um: dict[int, float]) -> Path:
    with open(source, newline="") as fh:
        reader = csv.DictReader(fh)
        rows, fields = list(reader), reader.fieldnames
    for row in rows:
        row["z01_est"] = repr(float(row["z01_est"]) + shift_um.get(int(row["index"]), 0) * 1e-6)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return target


def test_prediction_and_baseline_checks(tmp_path):
    predictions = REPO / "results" / "learned" / "spectrum-cnn-wide" / "test" / "predictions.csv"
    baseline = REPO / "results" / "baseline" / "cpu-4x-test"
    cases = {"PASS": {}, "WARN": {3: 5.0}, "FAIL": {i: 5.0 for i in range(250)}}
    for verdict, shift in cases.items():
        report = check.Report()
        check.check_predictions(
            perturbed_copy(predictions, tmp_path / verdict / "p.csv", shift), predictions, report
        )
        assert [v for v, _, _ in report.lines] == [verdict]
        run = tmp_path / verdict / "run"
        perturbed_copy(baseline / "results.csv", run / "results.csv", shift)
        report = check.Report()
        check.check_baseline(run, baseline, report)
        assert [v for v, _, _ in report.lines] == [verdict]
    one = {0: {"z01_true": 0.1, "z01_est": 0.1}}
    moved = {0: {"z01_true": 0.1, "z01_est": 0.1 + 5e-6}}
    report = check.Report()
    check._compare_estimates("baseline", moved, one, report, tolerance_um=1.0)
    assert [v for v, _, _ in report.lines] == ["WARN"] and "too few" in report.lines[0][2]


def test_device_check_on_cpu(tmp_path):
    report = check.Report()
    info = check.check_device(
        "cpu",
        report,
        tmp_path / "device.json",
        dataset_config=REPO / "configs" / "dataset" / "ci.yaml",
        baseline_config=REPO / "configs" / "baseline" / "ci.yaml",
    )
    assert not report.failed
    assert info["fft_relative_difference"] == 0.0
    assert "padding_2" in info["reconstruction"] and (tmp_path / "device.json").exists()


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck():
    subprocess.run(["shellcheck", "-x", *map(str, SCRIPTS)], check=True, cwd=KIT)
