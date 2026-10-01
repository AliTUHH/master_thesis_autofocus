"""YAML configuration loading and project-relative path handling."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

__all__ = ["PROJECT_ROOT", "load_yaml", "save_yaml", "resolve_path", "deep_update", "require_keys"]

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""Repository root (the directory containing ``src/`` and ``configs/``)."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping; raises ``FileNotFoundError``/``ValueError`` with the offending path."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a mapping at top level, got {type(data).__name__}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Write a mapping as YAML (keys kept in insertion order) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return path


def resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve ``path`` relative to ``root`` unless it is absolute."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``base`` with nested mappings from ``updates`` merged in."""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def require_keys(mapping: dict[str, Any], keys: list[str] | tuple[str, ...], where: str) -> None:
    """Raise ``KeyError`` naming the missing keys of ``mapping`` (``where`` describes the section)."""
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"missing key(s) {missing} in config section '{where}'")
