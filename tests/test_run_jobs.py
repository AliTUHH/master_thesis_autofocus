"""scripts/run_jobs.py: the start rules, the forecast and a real run of a small plan."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_jobs.py"

spec = importlib.util.spec_from_file_location("run_jobs", SCRIPT)
run_jobs = importlib.util.module_from_spec(spec)
sys.modules["run_jobs"] = run_jobs
spec.loader.exec_module(run_jobs)


def job(name: str, **options) -> run_jobs.Job:
    return run_jobs.Job(name=name, command="true", done=[name], **options)


def names(chosen) -> list[tuple[str, int]]:
    return [(j.name, threads) for j, threads in chosen]


HOLD = [(100.0, 1, 1.5)]
"""A running job that is expected to end at 100 s, with one thread and 1.5 GB."""


def test_select_backfills_what_does_not_delay_the_reserved_job():
    long, short = job("long", memory_gb=0.2, seconds=500), job("short", memory_gb=0.2, seconds=50)
    big = job("big", memory_gb=1.5, priority=9)
    assert run_jobs.reservation(big, 1, 0.5, HOLD) == (100.0, 1, 0.5)
    assert names(run_jobs.select([big, long], 1, 0.5, 0.0, HOLD)) == [("long", 1)]
    wide = job("wide", threads=2, memory_gb=1.5)
    assert run_jobs.reservation(wide, 1, 0.5, HOLD) == (100.0, 0, 0.5)
    assert names(run_jobs.select([wide, long, short], 1, 0.5, 0.0, HOLD)) == [("short", 1)]


def test_select_fits_a_job_into_exactly_what_is_left():
    free = 8.0 - (3.1 + 3.1 + 1.1)
    assert free < 0.7
    chosen = run_jobs.select([job("gen", memory_gb=0.7)], 1, free, 0.0, [])
    assert names(chosen) == [("gen", 1)]


def test_select_gives_spare_threads_to_the_last_job_if_none_waits():
    first, elastic = job("first"), job("elastic", max_threads=3)
    chosen = run_jobs.select([first, elastic], 4, 9.0, 0.0, [])
    assert names(chosen) == [("first", 1), ("elastic", 3)]
    chosen = run_jobs.select([job("big", memory_gb=5.0), elastic], 4, 2.0, 0.0, [])
    assert names(chosen) == [("elastic", 1)]


def test_forecast_plays_the_plan_through():
    budget = {**run_jobs.BUDGET, "threads": 2, "memory_gb": 10.0}
    jobs = [
        job("a", seconds=100, priority=2),
        job("b", seconds=50, priority=1),
        job("c", seconds=100, max_threads=2, needs=["a", "b"]),
    ]
    ends = run_jobs.forecast(budget, jobs, {}, 0.0)
    assert ends["a"] == 100 and ends["b"] == 50
    assert ends["c"] == pytest.approx(100 + 100 * 0.5**run_jobs.SPEEDUP_EXPONENT)


def test_forecast_backfills_around_a_job_that_waits_for_memory():
    budget = {**run_jobs.BUDGET, "threads": 2, "memory_gb": 2.0}
    jobs = [
        job("hold", memory_gb=1.5, seconds=100, priority=10),
        job("big", memory_gb=1.5, seconds=10, priority=9),
        *(job(f"s{k}", memory_gb=0.2, seconds=20) for k in (1, 2, 3)),
        job("long", memory_gb=0.2, seconds=200),
    ]
    ends = run_jobs.forecast(budget, jobs, {}, 0.0)
    assert ends == {"hold": 100, "s1": 20, "s2": 40, "s3": 60, "big": 110, "long": 260}


def test_forecast_keeps_the_threads_of_a_reserved_job_free():
    budget = {**run_jobs.BUDGET, "threads": 2, "memory_gb": 10.0}
    jobs = [
        job("hold", seconds=100, priority=10),
        job("wide", threads=2, seconds=10, priority=9),
        job("long", seconds=500),
        job("short", seconds=50),
    ]
    ends = run_jobs.forecast(budget, jobs, {}, 0.0)
    assert ends == {"hold": 100, "short": 50, "wide": 110, "long": 610}


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"needs": ["unknown"]}, "unknown needs"),
        ({"threads": 9}, "more threads or memory"),
        ({"clean": ["../outside"]}, "outside the repository"),
        ({"done": []}, "no done paths"),
    ],
)
def test_load_plan_rejects(tmp_path, entry, message):
    plan = tmp_path / "plan.json"
    entry = {"name": "a", "command": "true", "done": ["a"], **entry}
    plan.write_text(json.dumps({"jobs": [entry]}))
    with pytest.raises(ValueError, match=message):
        run_jobs.load_plan(plan)


def test_load_plan_rejects_duplicate_names(tmp_path):
    plan = tmp_path / "plan.json"
    entry = {"name": "a", "command": "true", "done": ["a"]}
    plan.write_text(json.dumps({"jobs": [entry, entry]}))
    with pytest.raises(ValueError, match="duplicate"):
        run_jobs.load_plan(plan)


PLAN = {
    "budget": {"threads": 6, "memory_gb": 4.0},
    "jobs": [
        {
            "name": "a:gated",
            "command": 'cat gate > /dev/null && echo "$OMP_NUM_THREADS {threads}" > gated',
            "done": ["gated"],
            "max_threads": 3,
        },
        {"name": "b:flag", "command": "test -e flag && touch b", "done": ["b"], "priority": 1},
        {"name": "b:after", "command": "touch b-after", "done": ["b-after"], "needs": ["b:flag"]},
        {"name": "c:no-output", "command": "true", "done": ["c"], "priority": 1},
        {
            "name": "d:clean",
            "command": "test ! -e d/old && mkdir d && touch d/new",
            "done": ["d/new"],
            "clean": ["d"],
            "priority": 1,
        },
        {"name": "e:after", "command": "touch e", "done": ["e"], "needs": ["a:gated", "d:clean"]},
        {"name": "f:present", "command": "exit 1", "done": ["f"]},
    ],
}


def jobs_cli(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return result.stdout


def test_run_takes_over_blocks_and_retries(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps(PLAN))
    os.mkfifo(tmp_path / "gate")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "old").touch()
    (tmp_path / "f").touch()

    out = jobs_cli(tmp_path, "run", "plan.json", "--state", "state", "--once")
    assert "a:gated: started with 3 thread(s)" in out
    state = json.loads((tmp_path / "state" / "jobs.json").read_text())
    assert state["a:gated"]["status"] == "running"
    assert state["f:present"] == {"status": "done", "note": "outputs present before the run"}

    subprocess.run(["sh", "-c", "echo go > gate"], cwd=tmp_path, timeout=30, check=True)
    out = jobs_cli(tmp_path, "run", "plan.json", "--state", "state", "--interval", "0.05")
    assert "a:gated: taken over" in out
    assert "every job of the plan is done, failed or blocked" in out
    state = json.loads((tmp_path / "state" / "jobs.json").read_text())
    assert {name: s["status"] for name, s in state.items()} == {
        "a:gated": "done",
        "b:flag": "failed",
        "b:after": "pending",
        "c:no-output": "failed",
        "d:clean": "done",
        "e:after": "done",
        "f:present": "done",
    }
    assert state["b:after"]["note"] == "blocked"
    assert state["b:flag"]["exit_code"] == 1
    assert state["c:no-output"]["exit_code"] == 0
    assert (tmp_path / "gated").read_text().split() == ["3", "3"]
    assert not (tmp_path / "d" / "old").exists()
    assert (tmp_path / "state" / "logs" / "a-gated.log").exists()
    assert "failed  c:no-output" in jobs_cli(tmp_path, "status", "--state", "state")

    (tmp_path / "flag").touch()
    jobs_cli(tmp_path, "retry", "--state", "state", "b:flag")
    jobs_cli(tmp_path, "run", "plan.json", "--state", "state", "--interval", "0.05")
    state = json.loads((tmp_path / "state" / "jobs.json").read_text())
    assert state["b:flag"]["status"] == state["b:after"]["status"] == "done"
    assert jobs_cli(tmp_path, "forecast", "plan.json", "--state", "state").strip() == (
        "nothing left to run"
    )
    rows = (tmp_path / "state" / "jobs.csv").read_text().splitlines()
    assert rows[0].startswith("name,status,threads") and len(rows) == 1 + len(PLAN["jobs"])
