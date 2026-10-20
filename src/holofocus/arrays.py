"""Centered crop/pad helpers for arrays of shape ``(..., H, W)``."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def center_slices(outer: tuple[int, int], inner: tuple[int, int]) -> tuple[slice, slice]:
    """Slices selecting a centered ``inner`` block inside an ``outer`` array.

    The block starts at ``(outer - inner) // 2`` in each dimension, so cropping and padding
    with these helpers are exact inverses.
    """
    if inner[0] > outer[0] or inner[1] > outer[1]:
        raise ValueError(f"inner shape {inner} does not fit into {outer}")
    r0 = (outer[0] - inner[0]) // 2
    c0 = (outer[1] - inner[1]) // 2
    return slice(r0, r0 + inner[0]), slice(c0, c0 + inner[1])


def crop_center(array: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    rows, cols = center_slices(tuple(array.shape[-2:]), tuple(shape))
    return array[..., rows, cols]


def pad_center(array: torch.Tensor, shape: tuple[int, int], value: complex | float = 0.0):
    """Embed ``array`` centered in a larger array filled with ``value``."""
    out = torch.full((*array.shape[:-2], *shape), value, dtype=array.dtype, device=array.device)
    rows, cols = center_slices(tuple(shape), tuple(array.shape[-2:]))
    out[..., rows, cols] = array
    return out


def reflect_pad(array: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    """Mirror padding ``(left, right, top, bottom)``; each pad must be smaller than the size.

    Works for real and complex tensors with any number of leading dimensions.
    """
    if array.is_complex():
        return torch.complex(reflect_pad(array.real, pad), reflect_pad(array.imag, pad))
    lead = array.shape[:-2]
    flat = array.reshape(-1, 1, *array.shape[-2:])
    out = F.pad(flat, pad, mode="reflect")
    return out.reshape(*lead, *out.shape[-2:])


def bin_sum(array: torch.Tensor, factor: int) -> torch.Tensor:
    """Sum non-overlapping ``factor x factor`` blocks (detector binning)."""
    if factor == 1:
        return array
    h, w = array.shape[-2:]
    if h % factor or w % factor:
        raise ValueError(f"shape {(h, w)} not divisible by binning factor {factor}")
    lead = array.shape[:-2]
    blocks = array.reshape(*lead, h // factor, factor, w // factor, factor)
    return blocks.sum(dim=(-3, -1))
