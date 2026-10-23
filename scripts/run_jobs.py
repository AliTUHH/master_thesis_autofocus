"""Run the jobs of a plan on this machine within a budget of threads and memory.

A plan is a JSON file::

    {"budget": {"threads": 4, "memory_gb": 6.0},
     "jobs": [{"name": "gen:loo2-tooth",
               "command": "holofocus-dataset generate configs/dataset/loo2-tooth.yaml ...",
               "threads": 1, "memory_gb": 1.0, "needs": [], "priority": 100,
               "done": ["data/loo2-tooth/test.h5"], "seconds": 11000}, ...]}

``run`` reads the plan again every cycle, so jobs can be added and the budget changed while
it runs. A job is done when all its ``done`` paths exist, also if it never ran. It starts
once every job in ``needs`` is done and its threads and memory fit into what the running
jobs leave of the budget and of the free memory of the machine (``MemAvailable`` minus
``reserve_gb`` and minus what the running jobs are still expected to take): the highest
``priority`` first, then in the order of the plan. The first job that does not fit reserves
the moment when the running jobs will have left enough for it (from their expected
``seconds``); a job after it starts in its place only if it ends by then or leaves enough
for it (EASY backfilling). ``{threads}`` in the command becomes the threads of the job:
``threads``, or up to ``max_threads`` if no other job is waiting. The ``clean`` paths
(leftovers of an interrupted run) are removed before a job starts.

Every job runs in a session of its own with ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` set
to its threads, writes to ``<state>/logs/<name>.log`` and leaves its exit code in
``<state>/exit/<name>``; a runner started again takes over the jobs that still run. A job
fails if its exit code is not 0 or a ``done`` path is missing at its end; it does not run
again, and the jobs that need it are blocked, until ``retry`` releases it. The state is
``<state>/jobs.json`` and ``jobs.csv``, with the time and the peak memory of every job.

``forecast`` plays the rest of the plan through with the same rules and the expected
``seconds`` of each job (for ``threads``; with more threads ``seconds * (threads / t) **
SPEEDUP_EXPONENT``) and prints when each group of jobs (the name up to the first ``:``)
and the whole plan end, in UTC and in Berlin time.

Usage::

    python scripts/run_jobs.py run runs/stage10/plan.json --state runs/stage10
    python scripts/run_jobs.py status --state runs/stage10
    python scripts/run_jobs.py forecast runs/stage10/plan.json --state runs/stage10
    python scripts/run_jobs.py retry --state runs/stage10 NAME [NAME ...]

Run it from the root of the repository: commands and paths are relative to it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import heapq
import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

BUDGET = {"threads": 4, "memory_gb": 6.0, "reserve_gb": 0.5}
SPEEDUP_EXPONENT = 0.85
"""Expected wall time of a job with ``t`` instead of ``threads`` threads:
``seconds * (threads / t) ** SPEEDUP_EXPONENT``."""
DEFAULT_SECONDS = 600.0
"""Expected time of a job without ``seconds``."""
MIN_REMAINING = 60.0
"""A running job past its expected time is expected to run this much longer."""
BERLIN = ZoneInfo("Europe/Berlin")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
GB = 1024**3
FINISHED = ("done", "failed")


@dataclass
class Job:
    name: str
    command: str
    done: list[str]
    threads: int = 1
    max_threads: int | None = None
    memory_gb: float = 1.0
    needs: list[str] = field(default_factory=list)
    clean: list[str] = field(default_factory=list)
    priority: float = 0.0
    seconds: float | None = None

    @property
    def group(self) -> str:
        return self.name.split(":", 1)[0]

    def outputs_exist(self) -> bool:
        return all(Path(p).exists() for p in self.done)

    def threads_for(self, free: int, alone: bool) -> int:
        if alone and self.max_threads:
            return max(self.threads, min(self.max_threads, free))
        return self.threads


def inside_repository(path: str) -> bool:
    p = Path(path)
    return not p.is_absolute() and ".." not in p.parts and p.parts != ()


def load_plan(path: Path) -> tuple[dict, list[Job]]:
    data = json.loads(path.read_text())
    budget = {**BUDGET, **data.get("budget", {})}
    jobs = [Job(**entry) for entry in data["jobs"]]
    names = [job.name for job in jobs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate job names")
    for job in jobs:
        if not job.done:
            raise ValueError(f"{job.name}: no done paths")
        unknown = [n for n in job.needs if n not in names]
        if unknown:
            raise ValueError(f"{job.name}: unknown needs {unknown}")
        if job.threads > budget["threads"] or job.memory_gb > budget["memory_gb"]:
            raise ValueError(f"{job.name}: more threads or memory than the budget")
        bad = [p for p in job.clean if not inside_repository(p)]
        if bad:
            raise ValueError(f"{job.name}: clean paths outside the repository {bad}")
    return budget, jobs


def now_utc(seconds: float | None = None) -> str:
    moment = (
        dt.datetime.now(dt.timezone.utc)
        if seconds is None
        else (dt.datetime.fromtimestamp(seconds, dt.timezone.utc))
    )
    return moment.isoformat(timespec="seconds")


def say(message: str) -> None:
    print(f"{now_utc()} {message}", flush=True)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# --- processes -----------------------------------------------------------------------------


def proc_stat(pid: int) -> list[str] | None:
    """The fields of ``/proc/<pid>/stat`` from the state on (field 3), or ``None``."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return text.rsplit(")", 1)[1].split()


def start_ticks(pid: int) -> int | None:
    fields = proc_stat(pid)
    return None if fields is None else int(fields[19])


def alive(pid: int, ticks: int) -> bool:
    fields = proc_stat(pid)
    return fields is not None and int(fields[19]) == ticks and fields[0] != "Z"


def session_rss_gb() -> dict[int, float]:
    """Resident memory of every session (GB), summed over its processes."""
    out: dict[int, float] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        fields = proc_stat(int(entry.name))
        if fields is None:
            continue
        session = int(fields[3])
        out[session] = out.get(session, 0.0) + int(fields[21]) * PAGE_SIZE / GB
    return out


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / GB
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def exit_code(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def remove(path: str) -> None:
    p = Path(path)
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    elif p.exists() or p.is_symlink():
        p.unlink()


def file_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "-")


# --- state ---------------------------------------------------------------------------------


CSV_FIELDS = (
    "name",
    "status",
    "threads",
    "memory_gb",
    "peak_rss_gb",
    "priority",
    "started_utc",
    "ended_utc",
    "seconds",
    "expected_seconds",
    "exit_code",
)


def load_state(directory: Path) -> dict[str, dict]:
    path = directory / "jobs.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(directory: Path, state: dict[str, dict], jobs: list[Job]) -> None:
    write_atomic(directory / "jobs.json", json.dumps(state, indent=1) + "\n")
    order = {job.name: k for k, job in enumerate(jobs)}
    expected = {job.name: job.seconds for job in jobs}
    rows = []
    for name in sorted(state, key=lambda n: (order.get(n, len(order)), n)):
        s = state[name]
        row = {key: s.get(key, "") for key in CSV_FIELDS}
        row["name"], row["expected_seconds"] = name, expected.get(name, "")
        for key in ("peak_rss_gb", "seconds"):
            if isinstance(row[key], float):
                row[key] = round(row[key], 3 if key == "peak_rss_gb" else 1)
        rows.append(row)
    tmp = directory / "jobs.csv.tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, directory / "jobs.csv")


def status_of(state: dict[str, dict], name: str) -> str:
    return state.get(name, {}).get("status", "pending")


def blocked(job: Job, by_name: dict[str, Job], state: dict, seen: set | None = None) -> bool:
    """Whether a job that ``job`` needs, directly or not, failed."""
    seen = set() if seen is None else seen
    for need in job.needs:
        if need in seen:
            continue
        seen.add(need)
        if status_of(state, need) == "failed" or blocked(by_name[need], by_name, state, seen):
            return True
    return False


def runnable(jobs: list[Job], state: dict) -> list[Job]:
    """Jobs that may start now, highest priority first, then in plan order."""
    ready = [
        job
        for job in jobs
        if status_of(state, job.name) == "pending"
        and all(status_of(state, n) == "done" for n in job.needs)
    ]
    return sorted(ready, key=lambda job: -job.priority)


def expected_seconds(job: Job | None, threads: int) -> float:
    if job is None or job.seconds is None:
        return DEFAULT_SECONDS
    return job.seconds * (job.threads / threads) ** SPEEDUP_EXPONENT


def reservation(
    job: Job, free_threads: int, free_memory: float, running: list[tuple[float, int, float]]
) -> tuple[float, int, float]:
    """When the running jobs (``(end, threads, memory)``) will have left enough for ``job``,
    and the threads and memory left beside it then."""
    threads, memory = free_threads, free_memory
    for end, t, m in sorted(running):
        threads, memory = threads + t, memory + m
        if threads >= job.threads and memory >= job.memory_gb:
            return end, threads - job.threads, memory - job.memory_gb
    return float("inf"), 0, 0.0


def select(
    ready: list[Job],
    free_threads: int,
    free_memory: float,
    now: float,
    running: list[tuple[float, int, float]],
) -> list[tuple[Job, int]]:
    """The jobs of ``ready`` (in order) that start now, with their threads. ``running``:
    ``(expected end, threads, memory)`` of the running jobs. The first job that does not fit
    reserves the moment of :func:`reservation`; a job after it starts only if it ends by
    then or fits into what the reserved job leaves."""
    chosen, running, reserved = [], list(running), None
    for k, job in enumerate(ready):
        threads = job.threads_for(free_threads, alone=reserved is None and k == len(ready) - 1)
        if threads > free_threads or job.memory_gb > free_memory:
            if reserved is None:
                reserved = reservation(job, free_threads, free_memory, running)
            continue
        end = now + expected_seconds(job, threads)
        if reserved is not None and end > reserved[0]:
            at, spare_threads, spare_memory = reserved
            if threads > spare_threads or job.memory_gb > spare_memory:
                continue
            reserved = (at, spare_threads - threads, spare_memory - job.memory_gb)
        chosen.append((job, threads))
        running.append((end, threads, job.memory_gb))
        free_threads -= threads
        free_memory -= job.memory_gb
    return chosen


def refresh(jobs: list[Job], state: dict) -> None:
    """Jobs whose outputs exist are done; done jobs whose outputs went are pending again."""
    for job in jobs:
        s = state.get(job.name)
        current = status_of(state, job.name)
        if current in ("running", "failed"):
            continue
        present = job.outputs_exist()
        if present and current != "done":
            state[job.name] = {**(s or {}), "status": "done"}
            if s is None:
                state[job.name]["note"] = "outputs present before the run"
        elif not present and current == "done":
            state[job.name] = {"status": "pending", "note": "outputs gone, to run again"}


# --- run -----------------------------------------------------------------------------------


class Runner:
    def __init__(self, plan: Path, directory: Path):
        self.plan, self.directory = plan, directory
        self.logs, self.exits = directory / "logs", directory / "exit"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.exits.mkdir(parents=True, exist_ok=True)
        self.state = load_state(directory)
        self.children: dict[str, subprocess.Popen] = {}
        self.stopping = False
        self.budget, self.jobs = load_plan(plan)

    def reload(self) -> None:
        try:
            self.budget, self.jobs = load_plan(self.plan)
        except (OSError, ValueError, TypeError, KeyError) as err:
            say(f"plan not read again ({err}); keeping the last one")

    def retries(self) -> None:
        path = self.directory / "retry.txt"
        if not path.exists():
            return
        for name in path.read_text().split():
            if status_of(self.state, name) == "failed":
                self.state[name] = {"status": "pending", "note": "released by retry"}
                say(f"{name}: released")
        path.unlink()

    def poll(self) -> None:
        by_name = {job.name: job for job in self.jobs}
        rss = session_rss_gb()
        for name, s in self.state.items():
            if s.get("status") != "running":
                continue
            child = self.children.get(name)
            if child is not None:
                finished = child.poll() is not None
            else:
                finished = not alive(s["pid"], s["pid_ticks"])
            if not finished:
                s["rss_gb"] = rss.get(s["pid"], 0.0)
                s["peak_rss_gb"] = max(s.get("peak_rss_gb", 0.0), s["rss_gb"])
                continue
            code = exit_code(self.exits / file_name(name))
            if code is None and child is not None and child.returncode not in (None, 0):
                code = child.returncode
            self.children.pop(name, None)
            ended = time.time()
            job = by_name.get(name)
            ok = code == 0 and (job is None or job.outputs_exist())
            s.update(
                status="done" if ok else "failed",
                exit_code=code,
                ended=ended,
                ended_utc=now_utc(ended),
                seconds=ended - s["started"],
            )
            missing = "" if ok or code != 0 else ", outputs missing"
            say(
                f"{name}: {s['status']} (exit {code}{missing}) after {s['seconds'] / 60:.1f} "
                f"min, peak {s.get('peak_rss_gb', 0):.2f} GB"
            )

    def start(self, job: Job, threads: int) -> None:
        for path in job.clean:
            remove(path)
        exit_file = self.exits / file_name(job.name)
        exit_file.unlink(missing_ok=True)
        command = job.command.replace("{threads}", str(threads))
        script = f"{command}\nstatus=$?\necho $status > '{exit_file}'\nexit $status\n"
        env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
        log_path = self.logs / f"{file_name(job.name)}.log"
        with open(log_path, "ab") as log:
            log.write(f"\n=== {now_utc()} {threads} thread(s): {command}\n".encode())
            log.flush()
            child = subprocess.Popen(
                ["bash", "-c", script],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        self.children[job.name] = child
        started = time.time()
        self.state[job.name] = {
            "status": "running",
            "pid": child.pid,
            "pid_ticks": start_ticks(child.pid),
            "threads": threads,
            "memory_gb": job.memory_gb,
            "priority": job.priority,
            "command": command,
            "log": str(log_path),
            "started": started,
            "started_utc": now_utc(started),
        }
        say(f"{job.name}: started with {threads} thread(s), pid {child.pid}")

    def schedule(self) -> None:
        by_name = {job.name: job for job in self.jobs}
        now = time.time()
        running = {n: s for n, s in self.state.items() if s.get("status") == "running"}
        free_threads = self.budget["threads"] - sum(s["threads"] for s in running.values())
        growth = sum(max(0.0, s["memory_gb"] - s.get("rss_gb", 0.0)) for s in running.values())
        free_memory = min(
            self.budget["memory_gb"] - sum(s["memory_gb"] for s in running.values()),
            mem_available_gb() - self.budget["reserve_gb"] - growth,
        )
        ends = [
            (
                max(
                    s["started"] + expected_seconds(by_name.get(n), s["threads"]),
                    now + MIN_REMAINING,
                ),
                s["threads"],
                s["memory_gb"],
            )
            for n, s in running.items()
        ]
        ready = runnable(self.jobs, self.state)
        for job, threads in select(ready, free_threads, free_memory, now, ends):
            self.start(job, threads)
            save_state(self.directory, self.state, self.jobs)

    def mark_blocked(self) -> None:
        by_name = {job.name: job for job in self.jobs}
        for job in self.jobs:
            if status_of(self.state, job.name) != "pending":
                continue
            noted = self.state.get(job.name, {}).get("note") == "blocked"
            if blocked(job, by_name, self.state):
                if not noted:
                    self.state[job.name] = {"status": "pending", "note": "blocked"}
                    say(f"{job.name}: blocked by a failed job it needs")
            elif noted:
                del self.state[job.name]

    def left(self) -> int:
        """Jobs that still run or may still run."""
        by_name = {job.name: job for job in self.jobs}
        count = sum(1 for s in self.state.values() if s.get("status") == "running")
        for job in self.jobs:
            if status_of(self.state, job.name) == "pending" and not blocked(
                job, by_name, self.state
            ):
                count += 1
        return count

    def cycle(self) -> None:
        self.reload()
        self.retries()
        self.poll()
        refresh(self.jobs, self.state)
        self.mark_blocked()
        if not self.stopping:
            self.schedule()
        save_state(self.directory, self.state, self.jobs)

    def stop(self, signum, frame) -> None:
        self.stopping = True
        say("stopping: no new jobs; the running ones go on and a new runner takes them over")


def cmd_run(args) -> None:
    runner = Runner(args.plan, args.state)
    for name, s in runner.state.items():
        if s.get("status") == "running":
            say(f"{name}: taken over (pid {s['pid']})")
    signal.signal(signal.SIGTERM, runner.stop)
    signal.signal(signal.SIGINT, runner.stop)
    while True:
        runner.cycle()
        if args.once or runner.stopping:
            break
        if runner.left() == 0:
            say("every job of the plan is done, failed or blocked")
            break
        time.sleep(args.interval)


# --- status and forecast -------------------------------------------------------------------


def cmd_status(args) -> None:
    state = load_state(args.state)
    jobs = load_plan(args.plan)[1] if args.plan else []
    names = list(dict.fromkeys([job.name for job in jobs] + list(state)))
    groups: dict[str, dict[str, int]] = {}
    for name in names:
        status = status_of(state, name)
        if status == "pending" and state.get(name, {}).get("note") == "blocked":
            status = "blocked"
        counts = groups.setdefault(name.split(":", 1)[0], {})
        counts[status] = counts.get(status, 0) + 1
    for group, counts in groups.items():
        print(f"{group:<12} " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
    now = time.time()
    for name in names:
        s = state.get(name, {})
        if s.get("status") == "running":
            print(
                f"  running {name}: {s['threads']} thread(s), {(now - s['started']) / 60:.0f} min, "
                f"peak {s.get('peak_rss_gb', 0):.2f} GB"
            )
        elif s.get("status") == "failed":
            print(f"  failed  {name}: exit {s.get('exit_code')}, log {s.get('log')}")


def forecast(budget: dict, jobs: list[Job], state: dict, now: float) -> dict[str, float]:
    """Expected end of every job that may still run (epoch seconds)."""
    state = {name: dict(s) for name, s in state.items()}
    by_name = {job.name: job for job in jobs}
    ends: dict[str, float] = {}
    events: list[tuple[float, str]] = []
    use = {"threads": 0, "memory_gb": 0.0}
    for name, s in state.items():
        if s.get("status") != "running":
            continue
        expected = expected_seconds(by_name.get(name), s["threads"])
        heapq.heappush(events, (max(s["started"] + expected, now + MIN_REMAINING), name))
        use["threads"] += s["threads"]
        use["memory_gb"] += s["memory_gb"]
    t = now
    while True:
        ready = [job for job in runnable(jobs, state) if not blocked(job, by_name, state)]
        free_threads = budget["threads"] - use["threads"]
        free_memory = budget["memory_gb"] - use["memory_gb"]
        running = [(end, state[n]["threads"], state[n]["memory_gb"]) for end, n in events]
        for job, threads in select(ready, free_threads, free_memory, t, running):
            state[job.name] = {"status": "running", "threads": threads, "memory_gb": job.memory_gb}
            use["threads"] += threads
            use["memory_gb"] += job.memory_gb
            heapq.heappush(events, (t + expected_seconds(job, threads), job.name))
        if not events:
            return ends
        t, name = heapq.heappop(events)
        s = state[name]
        use["threads"] -= s["threads"]
        use["memory_gb"] -= s["memory_gb"]
        s["status"] = "done"
        ends[name] = t


def clock(seconds: float) -> str:
    utc = dt.datetime.fromtimestamp(seconds, dt.timezone.utc)
    return f"{utc:%a %d %b %H:%M} UTC = {utc.astimezone(BERLIN):%a %d %b %H:%M %Z} Berlin"


def cmd_forecast(args) -> None:
    budget, jobs = load_plan(args.plan)
    state = load_state(args.state)
    refresh(jobs, state)
    ends = forecast(budget, jobs, state, time.time())
    if not ends:
        print("nothing left to run")
        return
    groups: dict[str, list[float]] = {}
    for name, end in ends.items():
        groups.setdefault(name.split(":", 1)[0], []).append(end)
    for group, values in sorted(groups.items(), key=lambda item: max(item[1])):
        print(f"{group:<12} {len(values):>4} job(s), last ends {clock(max(values))}")
    print(f"{'all':<12} {len(ends):>4} job(s), last ends {clock(max(ends.values()))}")


def cmd_retry(args) -> None:
    args.state.mkdir(parents=True, exist_ok=True)
    with open(args.state / "retry.txt", "a") as fh:
        fh.write("".join(f"{name}\n" for name in args.names))
    print(f"{len(args.names)} job(s) released at the next cycle of the runner")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the plan until nothing is left")
    run.add_argument("plan", type=Path)
    run.add_argument("--state", type=Path, required=True, help="directory of state and logs")
    run.add_argument("--interval", type=float, default=10.0, help="seconds between cycles")
    run.add_argument("--once", action="store_true", help="one cycle, then exit")
    run.set_defaults(func=cmd_run)
    st = sub.add_parser("status", help="jobs per group and status, running and failed jobs")
    st.add_argument("--state", type=Path, required=True)
    st.add_argument("--plan", type=Path, help="also count the jobs that never started")
    st.set_defaults(func=cmd_status)
    fo = sub.add_parser("forecast", help="expected end of every group of jobs")
    fo.add_argument("plan", type=Path)
    fo.add_argument("--state", type=Path, required=True)
    fo.set_defaults(func=cmd_forecast)
    re = sub.add_parser("retry", help="release failed jobs")
    re.add_argument("--state", type=Path, required=True)
    re.add_argument("names", nargs="+")
    re.set_defaults(func=cmd_retry)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
