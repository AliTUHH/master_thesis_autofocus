"""Strict YAML -> frozen dataclass conversion shared by the dataset and baseline configs.

Unknown keys are rejected so that typos do not silently fall back to defaults. PyYAML follows
YAML 1.1, which reads ``1e5`` or ``1.0e3`` as strings; numbers are therefore converted from
the annotated field types instead of trusting the YAML types.
"""

from __future__ import annotations

import dataclasses
from typing import Any


def build(cls, data: dict[str, Any], where: str, sections: dict[str, type] | None = None):
    """Instantiate dataclass ``cls`` from ``data``; ``sections`` maps keys to nested classes."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    sections = sections or {}
    kwargs = {}
    for name, value in data.items():
        sub = sections.get(name)
        if sub is not None:
            kwargs[name] = build(sub, value or {}, f"{where}.{name}")
        else:
            kwargs[name] = coerce(fields[name].type, value, f"{where}.{name}")
    return cls(**kwargs)


def coerce(annotation: str, value: Any, where: str) -> Any:
    """Convert a YAML value to the annotated field type (``from __future__`` string form)."""
    optional = annotation.endswith("| None")
    base = annotation.removesuffix("| None").strip()
    if value is None:
        if optional:
            return None
        raise ValueError(f"{where}: must not be null")
    try:
        if base == "float":
            if isinstance(value, bool):
                raise TypeError("bool is not a number")
            return float(value)
        if base == "int":
            return _to_int(value)
        if base in ("bool", "str"):
            if not isinstance(value, bool if base == "bool" else str):
                raise TypeError(f"expected {base}")
            return value
        if base.startswith("tuple[") and base.endswith(", ...]"):
            if isinstance(value, (str, dict)):
                raise ValueError("expected a list")
            item = base[len("tuple[") : -len(", ...]")]
            return tuple(coerce(item, v, f"{where}[{i}]") for i, v in enumerate(value))
        if base.startswith("tuple["):
            items = [t.strip() for t in base[len("tuple[") : -1].split(",")]
            if isinstance(value, (str, dict)) or len(value) != len(items):
                raise ValueError(f"expected {len(items)} values")
            return tuple(coerce(t, v, where) for t, v in zip(items, value, strict=True))
        if base.startswith("dict[str, "):
            if not isinstance(value, dict):
                raise TypeError("expected a mapping")
            item = base[len("dict[str, ") : -1]
            return {str(k): coerce(item, v, f"{where}.{k}") for k, v in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: cannot read {value!r} as {annotation} ({exc})") from None
    raise TypeError(f"{where}: unsupported annotation {annotation!r}")


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("bool is not an integer")
    if isinstance(value, int):
        return value
    number = float(value)
    if not number.is_integer():
        raise ValueError("not an integer")
    return int(number)


def plain(value):
    """Dataclass ``asdict`` output with tuples turned into lists (for ``yaml.safe_dump``)."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value
