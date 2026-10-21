"""Stage 4 study on the validation split (plan step 11); results go to ``results/learned/study``.

Every run trains a config of ``configs/learned`` with some settings changed (:data:`RUNS`) on
the first ``n_train`` samples of the training files and predicts the validation split into
``<run>/val``. The weights (``model.pt``) are not tracked.

* ``spectrum-n500`` ... ``spectrum-n5000``: learning curve of the spectrum CNN (the main model
  ``results/learned/spectrum-cnn`` is trained on all 10000 samples);
* ``spectrum-seed1``, ``spectrum-seed2``: other seeds of ``spectrum-n2000``;
* ``spectrum-beta0``, ``spectrum-beta1``: Gaussian NLL without and with full variance
  weighting (``beta``, the configs use 0.5);
* ``spectrum-wide``: twice the channels;
* ``image-n2000``: the image CNN (binned hologram) on the same 2000 samples.

Usage::

    python scripts/run_stage4_study.py --train data/cpu-4x/train.h5 data/cpu-4x-extra/train.h5 \\
        --val data/cpu-4x/val.h5 --threads 1 [--runs spectrum-n500 image-n2000]

Runs whose ``val/summary.json`` exists are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from holofocus.learned import LearnedAutofocus, LearnedConfig, predict_split, train_model

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "learned"
SPECTRUM, IMAGE = "spectrum-cnn.yaml", "image-cnn.yaml"

RUNS: dict[str, tuple[str, int, dict]] = {
    "spectrum-n500": (SPECTRUM, 500, {}),
    "spectrum-n1000": (SPECTRUM, 1000, {}),
    "spectrum-n2000": (SPECTRUM, 2000, {}),
    "spectrum-n5000": (SPECTRUM, 5000, {}),
    "spectrum-seed1": (SPECTRUM, 2000, {"training": {"seed": 1}}),
    "spectrum-seed2": (SPECTRUM, 2000, {"training": {"seed": 2}}),
    "spectrum-beta0": (SPECTRUM, 2000, {"training": {"beta": 0.0}}),
    "spectrum-beta1": (SPECTRUM, 2000, {"training": {"beta": 1.0}}),
    "spectrum-wide": (SPECTRUM, 2000, {"model": {"channels": [32, 64, 128, 128]}}),
    "image-n2000": (IMAGE, 2000, {}),
}


def study_config(name: str) -> LearnedConfig:
    base, n_train, changes = RUNS[name]
    data = LearnedConfig.from_yaml(CONFIGS / base).to_dict()
    for section, values in changes.items():
        data[section].update(values)
    data["name"] = name
    data["training"]["limit"] = n_train
    return LearnedConfig.from_dict(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, nargs="+", required=True, help="training files")
    parser.add_argument("--val", type=Path, required=True, help="validation split file")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "learned" / "study")
    parser.add_argument("--runs", nargs="+", choices=list(RUNS), default=list(RUNS))
    parser.add_argument("--device", help="cpu, cuda or auto (default: from the config)")
    parser.add_argument("--threads", type=int, help="torch CPU threads")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    for name in args.runs:
        run = args.out / name
        if (run / "val" / "summary.json").exists():
            print(f"{name}: done, skipped")
            continue
        if not (run / "model.pt").exists():
            train_model(study_config(name), args.train, args.val, run, args.device)
        autofocus = LearnedAutofocus.load(run, args.device)
        predict_split(autofocus, args.val, run / "val", model_path=run)


if __name__ == "__main__":
    main()
