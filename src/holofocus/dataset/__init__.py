"""Synthetic dataset for learning ``dz01`` (Stage 2): YAML config, generator, HDF5 storage."""

from .config import DatasetConfig
from .generator import LABEL_KEYS, META_KEYS, HologramGenerator, Sample, sample_seed
from .storage import (
    TARGETS,
    HologramDataset,
    read_config,
    read_labels,
    target_values,
    write_split,
)

__all__ = [
    "LABEL_KEYS",
    "META_KEYS",
    "TARGETS",
    "DatasetConfig",
    "HologramDataset",
    "HologramGenerator",
    "Sample",
    "read_config",
    "read_labels",
    "sample_seed",
    "target_values",
    "write_split",
]
