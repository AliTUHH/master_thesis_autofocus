"""Device and dtype selection so the same code runs on CPU and CUDA."""

from __future__ import annotations

import torch

COMPLEX_DTYPES = {"single": torch.complex64, "double": torch.complex128}
REAL_DTYPES = {"single": torch.float32, "double": torch.float64}


def get_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve ``"auto"``/``None`` to CUDA when available, otherwise CPU."""
    if device is None or (isinstance(device, str) and device == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def complex_dtype(precision: str = "single") -> torch.dtype:
    try:
        return COMPLEX_DTYPES[precision]
    except KeyError:
        raise ValueError(f"precision must be one of {sorted(COMPLEX_DTYPES)}") from None


def real_dtype(precision: str = "single") -> torch.dtype:
    try:
        return REAL_DTYPES[precision]
    except KeyError:
        raise ValueError(f"precision must be one of {sorted(REAL_DTYPES)}") from None


def real_dtype_of(dtype: torch.dtype) -> torch.dtype:
    """Real counterpart of a complex dtype (float dtypes are returned unchanged)."""
    if dtype == torch.complex128:
        return torch.float64
    if dtype == torch.complex64:
        return torch.float32
    return dtype


def complex_dtype_of(dtype: torch.dtype) -> torch.dtype:
    """Complex counterpart of a real dtype (complex dtypes are returned unchanged)."""
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    return torch.complex64
