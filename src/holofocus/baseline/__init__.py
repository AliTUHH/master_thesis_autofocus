"""Stage 3 baseline: the classical autofocus of [2] on a dataset split, with stored results."""

from .config import BaselineConfig, ReconstructionConfig, SearchConfig
from .results import RESULT_FIELDS, read_rows, read_traces, summarize
from .runner import WarmStart, merge_runs, run_baseline, select_indices

__all__ = [
    "RESULT_FIELDS",
    "BaselineConfig",
    "ReconstructionConfig",
    "SearchConfig",
    "WarmStart",
    "merge_runs",
    "read_rows",
    "read_traces",
    "run_baseline",
    "select_indices",
    "summarize",
]
