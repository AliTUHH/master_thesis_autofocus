"""Dataset splits as in-memory network inputs, and the D4 augmentation of image inputs.

The representation is computed once per split (a few seconds for 2000 holograms of 512^2
pixels), so an epoch costs only the network. Image inputs are kept as float16 (2000 samples
at 256^2 take 0.26 GB) and converted to float32 per batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from ..dataset.storage import read_config, read_labels
from .representations import Representation


@dataclass
class SplitInputs:
    """Network inputs and targets ``t = dz01 / target_scale`` of the selected samples."""

    inputs: torch.Tensor
    targets: torch.Tensor
    indices: np.ndarray
    labels: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.indices)


def choose_indices(n_samples: int, indices: Sequence[int] | None, limit: int | None) -> list[int]:
    """``indices`` (validated) or ``range(n_samples)``, cut to the first ``limit``."""
    chosen = list(range(n_samples)) if indices is None else [int(i) for i in indices]
    bad = [i for i in chosen if not 0 <= i < n_samples]
    if bad:
        raise ValueError(f"indices {bad} outside the split (0 .. {n_samples - 1})")
    return chosen if limit is None else chosen[:limit]


def fov_shape(path: str | Path) -> tuple[int, int]:
    return read_config(path).geometry.fov_shape


def load_inputs(
    path: str | Path,
    representation: Representation,
    target_scale: float,
    indices: Sequence[int] | None = None,
    limit: int | None = None,
    chunk: int = 32,
) -> SplitInputs:
    """Read the holograms of a split file and convert them with ``representation``."""
    labels = read_labels(path)
    chosen = choose_indices(len(labels["dz01"]), indices, limit)
    storage = torch.float16 if representation.config.kind == "image" else torch.float32
    inputs = torch.empty((len(chosen), *representation.input_shape), dtype=storage)
    with h5py.File(path, "r") as fh:
        holograms = fh["hologram"]
        for start in range(0, len(chosen), chunk):
            part = chosen[start : start + chunk]
            batch = torch.from_numpy(np.stack([holograms[i] for i in part]))
            inputs[start : start + len(part)] = representation(batch).to("cpu", storage)
    picked = np.asarray(chosen, dtype=np.int64)
    targets = torch.as_tensor(labels["dz01"][picked] / target_scale, dtype=torch.float32)
    return SplitInputs(inputs, targets, picked, {k: v[picked] for k, v in labels.items()})


def load_many(
    paths: Sequence[str | Path],
    representation: Representation,
    target_scale: float,
    limit: int | None = None,
) -> SplitInputs:
    """:func:`load_inputs` of several split files, concatenated; ``limit`` counts in order."""
    parts = []
    for path in paths:
        remaining = None if limit is None else limit - sum(len(p) for p in parts)
        if remaining is not None and remaining <= 0:
            break
        parts.append(load_inputs(path, representation, target_scale, limit=remaining))
    return SplitInputs(
        torch.cat([p.inputs for p in parts]),
        torch.cat([p.targets for p in parts]),
        np.concatenate([p.indices for p in parts]),
        {k: np.concatenate([p.labels[k] for p in parts]) for k in parts[0].labels},
    )


def d4_transform(x: torch.Tensor, element: int) -> torch.Tensor:
    """Element ``0..7`` of the dihedral group: ``element % 4`` quarter turns, then a flip of
    the last axis for ``element >= 4``."""
    y = torch.rot90(x, element % 4, dims=(-2, -1))
    return torch.flip(y, dims=(-1,)) if element >= 4 else y


def random_d4(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Apply an independent random D4 element to every image of the batch ``(B, C, H, W)``."""
    if x.shape[-1] != x.shape[-2]:
        raise ValueError("rotations by 90 degrees need square images")
    elements = torch.randint(0, 8, (x.shape[0],), generator=generator)
    out = torch.empty_like(x)
    for element in range(8):
        mask = (elements == element).to(x.device)
        if bool(mask.any()):
            out[mask] = d4_transform(x[mask], element)
    return out
