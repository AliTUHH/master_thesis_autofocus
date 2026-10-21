"""Checks of a run on another machine against the committed results (docs/maxwell-gpu.md).

The committed results were computed on the 4-core VM of the lab notebook. These checks
compare a run elsewhere, e.g. on the A100 nodes of DESY Maxwell, sample by sample (same
split, same ``index``):

* ``fingerprint``: labels and a checksum of every hologram of a split, as CSV;
* ``data``: a split against a committed fingerprint (``results/datasets/``);
* ``predictions``: learned estimates against a committed ``predictions.csv``;
* ``baseline``: a baseline run directory against a committed one;
* ``device``: PyTorch and holofocus on a device against the CPU, and the time per ASRM
  reconstruction.

Every check prints PASS, WARN or FAIL lines; the exit code is 1 if one failed.

::

    python scripts/check_cluster_run.py fingerprint data/cpu-4x/test.h5 \\
        --out results/datasets/cpu-4x-test.csv
    python scripts/check_cluster_run.py data .../cpu-4x/test.h5 \\
        --reference results/datasets/cpu-4x-test.csv
    python scripts/check_cluster_run.py predictions .../predictions.csv \\
        --reference results/learned/spectrum-cnn-wide/test/predictions.csv
    python scripts/check_cluster_run.py baseline .../baseline/cpu-4x-test-p2 \\
        --reference results/baseline/cpu-4x-test
    python scripts/check_cluster_run.py device --device cuda --out device.json
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import torch

from holofocus.asrm import asrm_reconstruct
from holofocus.baseline.config import BaselineConfig
from holofocus.baseline.results import read_rows
from holofocus.dataset.config import DatasetConfig
from holofocus.dataset.generator import HologramGenerator
from holofocus.dataset.storage import read_labels
from holofocus.learned.inference import read_predictions

REPO = Path(__file__).resolve().parents[1]
DATASET_CONFIG = REPO / "configs" / "dataset" / "cpu-4x.yaml"
BASELINE_CONFIG = REPO / "configs" / "baseline" / "paper2-cpu.yaml"
VERDICTS = ("PASS", "WARN", "FAIL")
# A single estimate that moved can be a Nelder-Mead path change by rounding; FAIL needs the
# median of at least this many samples.
MIN_SAMPLES_FOR_FAIL = 3


class Report:
    """Collects PASS/WARN/FAIL lines; ``failed`` decides the exit code."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str, str]] = []

    def add(self, verdict: str, check: str, message: str) -> None:
        assert verdict in VERDICTS
        self.lines.append((verdict, check, message))
        print(f"{verdict} {check}: {message}", flush=True)

    @property
    def failed(self) -> bool:
        return any(v == "FAIL" for v, _, _ in self.lines)


# --- fingerprints and data -------------------------------------------------------------------


def hologram_sha256(hologram: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(hologram, dtype=np.float32).tobytes()).hexdigest()


def fingerprint_rows(path: str | Path) -> list[dict]:
    """Labels, SHA-256, mean and standard deviation of every hologram of a split file."""
    labels = read_labels(path)
    keys = sorted(labels)
    rows = []
    with h5py.File(path, "r") as fh:
        holograms = fh["hologram"]
        for i in range(holograms.shape[0]):
            hologram = np.asarray(holograms[i])
            row = {"index": i}
            row.update({k: labels[k][i].item() for k in keys})
            row["hologram_sha256"] = hologram_sha256(hologram)
            row["hologram_mean"] = float(hologram.mean(dtype=np.float64))
            row["hologram_std"] = float(hologram.std(dtype=np.float64))
            rows.append(row)
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {k: repr(v) if isinstance(v, float) else v for k, v in row.items()} for row in rows
        )


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def check_data(split: str | Path, reference: str | Path, report: Report) -> None:
    """Labels must be identical; holograms are compared by checksum, mean and std."""
    new = {row["index"]: row for row in fingerprint_rows(split)}
    ref = {int(row["index"]): row for row in read_csv(reference)}
    extra = sorted(set(new) - set(ref))
    if extra:
        report.add("FAIL", "data", f"indices {extra[:5]}... of {split} are not in {reference}")
        return
    common = sorted(new)
    keys = [k for k in ref[common[0]] if k not in ("index",) and not k.startswith("hologram_")]
    keys = [k for k in keys if k in new[common[0]]]
    wrong = [(i, k) for i in common for k in keys if float(new[i][k]) != float(ref[i][k])]
    if wrong:
        report.add(
            "FAIL",
            "data labels",
            f"{len(wrong)} label values differ, e.g. sample {wrong[0][0]} {wrong[0][1]}: "
            f"{new[wrong[0][0]][wrong[0][1]]!r} instead of {ref[wrong[0][0]][wrong[0][1]]}; "
            "another config, seed or numpy version?",
        )
    else:
        report.add("PASS", "data labels", f"{len(keys)} labels of {len(common)} samples identical")
    differ = [i for i in common if new[i]["hologram_sha256"] != ref[i]["hologram_sha256"]]
    if not differ:
        report.add("PASS", "data holograms", f"all {len(common)} holograms bit-identical")
        return
    rel = {
        stat: max(
            abs(new[i][f"hologram_{stat}"] / float(ref[i][f"hologram_{stat}"]) - 1) for i in differ
        )
        for stat in ("mean", "std")
    }
    report.add(
        "WARN",
        "data holograms",
        f"{len(differ)} of {len(common)} holograms differ (e.g. sample {differ[0]}), "
        f"mean by at most {rel['mean']:.1e}, std by {rel['std']:.1e} (relative); expected "
        "after a simulation on another device or CPU type (float32 rounding, then a few "
        "noise values)",
    )


# --- per-sample estimates --------------------------------------------------------------------


def _compare_estimates(
    name: str,
    new: dict[int, dict],
    ref: dict[int, dict],
    report: Report,
    tolerance_um: float,
) -> list[int]:
    """Same samples (``z01_true``) and the distance of the ``z01_est`` values."""
    common = sorted(set(new) & set(ref))
    if not common:
        report.add("FAIL", name, "no common sample indices")
        return []
    other = [i for i in common if abs(new[i]["z01_true"] - ref[i]["z01_true"]) > 1e-12]
    if other:
        report.add("FAIL", name, f"z01_true differs for samples {other[:5]}: other data?")
        return []
    diff = {i: abs(new[i]["z01_est"] - ref[i]["z01_est"]) * 1e6 for i in common}
    beyond = [i for i in common if diff[i] > tolerance_um]
    median = statistics.median(diff.values())
    summary = (
        f"{len(common)} samples, |difference of z01_est| median {median:.3g} um, "
        f"max {max(diff.values()):.3g} um, {len(beyond)} above {tolerance_um:g} um"
    )
    if median > tolerance_um and len(common) >= MIN_SAMPLES_FOR_FAIL:
        report.add("FAIL", name, summary + "; the computation differs, not just a few samples")
    elif beyond:
        few = len(common) < MIN_SAMPLES_FOR_FAIL and median > tolerance_um
        hint = "; too few samples to tell rounding from another computation" if few else ""
        report.add("WARN", name, summary + f" (samples {beyond[:10]}){hint}")
    else:
        report.add("PASS", name, summary)
    return beyond


def check_predictions(
    predictions: str | Path, reference: str | Path, report: Report, tolerance_um: float = 1.0
) -> None:
    new = {row["index"]: row for row in read_predictions(predictions)}
    ref = {row["index"]: row for row in read_predictions(reference)}
    beyond = _compare_estimates("predictions", new, ref, report, tolerance_um)
    if beyond:
        print(
            "  samples whose hologram differs (check data) can move by a few um; "
            "a model trained elsewhere differs everywhere"
        )


def check_baseline(
    run: str | Path, reference: str | Path, report: Report, tolerance_um: float = 1.0
) -> None:
    new = {row["index"]: row for row in read_rows(Path(run) / "results.csv")}
    ref = {row["index"]: row for row in read_rows(Path(reference) / "results.csv")}
    beyond = _compare_estimates("baseline", new, ref, report, tolerance_um)
    for i in beyond:
        print(
            f"  sample {i}: z01_est {new[i]['z01_est'] * 1e3:.4f} mm vs "
            f"{ref[i]['z01_est'] * 1e3:.4f} mm, MFE {new[i]['mfe_est']:.6e} vs "
            f"{ref[i]['mfe_est']:.6e}, {new[i]['n_reconstructions']} vs "
            f"{ref[i]['n_reconstructions']} reconstructions"
        )
    if beyond:
        print(
            "  Nelder-Mead takes another path when two MFE values swap order by rounding; "
            "the estimates then differ by up to the simplex size at that step"
        )
    common = sorted(set(new) & set(ref))
    if common:
        t_new = statistics.mean(new[i]["seconds_per_reconstruction"] for i in common)
        t_ref = statistics.mean(ref[i]["seconds_per_reconstruction"] for i in common)
        print(
            f"  time per reconstruction {t_new:.2f} s here, {t_ref:.2f} s in the reference "
            f"({t_ref / t_new:.1f}x)"
        )


# --- device ----------------------------------------------------------------------------------


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


def _timed(fn, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out, time.perf_counter() - start


def check_device(
    device_name: str,
    report: Report,
    out: str | Path | None = None,
    dataset_config: str | Path = DATASET_CONFIG,
    baseline_config: str | Path = BASELINE_CONFIG,
) -> dict:
    """PyTorch on ``device_name`` against the CPU; time per reconstruction at padding 2/4."""
    info: dict = {
        "device": device_name,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "cpu_threads": torch.get_num_threads(),
    }
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            report.add("FAIL", "device", "CUDA is not available (driver, wheel or no GPU?)")
            return info
        props = torch.cuda.get_device_properties(device)
        info.update(
            gpu=props.name,
            gpu_memory_gb=round(props.total_memory / 1e9, 1),
            compute_capability=f"{props.major}.{props.minor}",
        )
        report.add(
            "PASS",
            "device",
            f"{props.name}, {info['gpu_memory_gb']} GB, compute capability "
            f"{info['compute_capability']}, torch {torch.__version__}, CUDA {torch.version.cuda}",
        )
    else:
        report.add("PASS", "device", f"{device}, {torch.get_num_threads()} threads (no GPU)")
    on_cpu = device.type == "cpu"

    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1024, 1024, dtype=torch.complex64, generator=generator)
    rel = _relative(torch.fft.fft2(x.to(device)).cpu(), torch.fft.fft2(x))
    info["fft_relative_difference"] = rel
    report.add("PASS" if rel < 1e-5 else "FAIL", "fft2 1024^2", f"relative difference {rel:.1e}")

    signal = torch.randn(32, 32, 1024, generator=generator)
    weight = torch.randn(64, 32, 7, generator=generator)
    conv = torch.nn.functional.conv1d
    rel = _relative(conv(signal.to(device), weight.to(device)).cpu(), conv(signal, weight))
    info["conv1d_relative_difference"] = rel
    verdict = "PASS" if rel < 1e-5 else "WARN" if rel < 1e-2 else "FAIL"
    hint = "" if verdict == "PASS" else " (TF32? NVIDIA_TF32_OVERRIDE=0 switches it off)"
    report.add(verdict, "conv1d", f"relative difference {rel:.1e}{hint}")

    dataset = DatasetConfig.from_yaml(dataset_config)
    sample, seconds = _timed(
        lambda: HologramGenerator(dataset, device=device).sample("test", 0), device
    )
    reference = HologramGenerator(dataset, device="cpu").sample("test", 0)
    diff = float(np.abs(sample.clean - reference.clean).max())
    noisy = "identical" if np.array_equal(sample.hologram, reference.hologram) else "differs"
    info["forward_max_difference"] = diff
    info["forward_seconds"] = seconds
    report.add(
        "PASS" if diff < 1e-4 else "FAIL",
        "forward model",
        f"noise-free hologram of test sample 0 differs by at most {diff:.1e} from the CPU, "
        f"noisy hologram {noisy}",
    )

    baseline = BaselineConfig.from_yaml(baseline_config)
    binning = dataset.geometry.binning
    fresnel = dataset.geometry.at(sample.labels["z01"]).fresnel_number
    hologram = torch.from_numpy(reference.hologram)
    info["reconstruction"] = {}
    for padding in (2.0, 4.0):
        settings = dataclasses.replace(baseline.reconstruction, padding_factor=padding).settings(
            binning
        )
        if on_cpu and padding > 2:
            print(f"  padding {padding:g}: skipped on the CPU (minutes per reconstruction)")
            continue
        if not on_cpu:
            # The first CUDA call pays for kernel loading and the cuFFT plans.
            _timed(lambda s=settings: asrm_reconstruct(hologram.to(device), fresnel, s), device)
        result, seconds = _timed(
            lambda s=settings: asrm_reconstruct(hologram.to(device), fresnel, s), device
        )
        entry = {"seconds": seconds, "mfe": result.final_loss}
        message = f"{seconds:.2f} s per reconstruction, MFE {result.final_loss:.6e}"
        verdict = "PASS"
        if not on_cpu and padding == 2:
            cpu_result = asrm_reconstruct(hologram, fresnel, settings)
            rel = abs(result.final_loss / cpu_result.final_loss - 1)
            entry["mfe_cpu"] = cpu_result.final_loss
            entry["mfe_relative_difference"] = rel
            message += f", {rel:.1e} relative to the CPU"
            verdict = "PASS" if rel < 1e-3 else "WARN" if rel < 1e-2 else "FAIL"
        info["reconstruction"][f"padding_{padding:g}"] = entry
        report.add(verdict, f"ASRM padding {padding:g}", message)
    # Stage 3: 11.1 reconstructions per autofocus and one at the truth (evaluate_truth).
    for key, entry in info["reconstruction"].items():
        hours = 250 * 12.1 * entry["seconds"] / 3600
        print(f"  {key}: baseline on 250 test samples about {hours:.1f} h on this device")
    if out is not None:
        Path(out).write_text(json.dumps(info, indent=2) + "\n")
    return info


# --- command line ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fp = sub.add_parser("fingerprint", help="labels and hologram checksums of a split as CSV")
    fp.add_argument("split", type=Path)
    fp.add_argument("--out", type=Path, required=True)
    da = sub.add_parser("data", help="a split against a committed fingerprint")
    da.add_argument("split", type=Path)
    da.add_argument("--reference", type=Path, required=True)
    pr = sub.add_parser("predictions", help="predictions.csv against a committed one")
    pr.add_argument("predictions", type=Path)
    pr.add_argument("--reference", type=Path, required=True)
    pr.add_argument("--tolerance-um", type=float, default=1.0)
    ba = sub.add_parser("baseline", help="a baseline run directory against a committed one")
    ba.add_argument("run", type=Path)
    ba.add_argument("--reference", type=Path, required=True)
    ba.add_argument("--tolerance-um", type=float, default=1.0)
    de = sub.add_parser("device", help="PyTorch and holofocus on a device against the CPU")
    de.add_argument("--device", default="cuda")
    de.add_argument("--out", type=Path, help="JSON with the numbers")
    de.add_argument("--dataset-config", type=Path, default=DATASET_CONFIG)
    de.add_argument("--baseline-config", type=Path, default=BASELINE_CONFIG)
    args = parser.parse_args(argv)

    report = Report()
    if args.command == "fingerprint":
        rows = fingerprint_rows(args.split)
        write_csv(args.out, rows)
        print(f"wrote {args.out} ({len(rows)} samples)")
        return 0
    if args.command == "data":
        check_data(args.split, args.reference, report)
    elif args.command == "predictions":
        check_predictions(args.predictions, args.reference, report, args.tolerance_um)
    elif args.command == "baseline":
        check_baseline(args.run, args.reference, report, args.tolerance_um)
    else:
        check_device(args.device, report, args.out, args.dataset_config, args.baseline_config)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
