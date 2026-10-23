"""YAML configuration of the classical autofocus baseline (Stage 3).

Example (``configs/baseline/paper2-cpu.yaml``)::

    name: paper2-cpu
    reconstruction:
      schedule: paper2       # [2] Table 3, down-sampling adapted to the dataset binning
      padding_factor: 2.0
    search:
      confidence: 5.0e-3     # Omega_z = z01_guess +- 5 mm
      xatol: 1.0e-4          # stop at a simplex length of 100 um

Lengths in metres. Unknown keys are rejected.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..asrm import PAPER2_TABLE3, ASRMSettings, ASRMStage, schedule_for_binning
from ..autofocus.search import METHODS
from ..yamlconfig import build, plain

SCHEDULES = {"paper2": PAPER2_TABLE3}


@dataclass(frozen=True)
class ReconstructionConfig:
    """Inner ASRM reconstruction ([2] Eq. (20), Table 3).

    ``schedule`` names a schedule given for the full 2048^2 detector; ``stages`` (list of
    :class:`~holofocus.asrm.ASRMStage` fields) replaces it. Both are adapted to the binning
    of the data with :func:`~holofocus.asrm.schedule_for_binning`. ``fade_width`` defaults
    to ``80 // binning`` (reference value on the full detector).
    """

    schedule: str = "paper2"
    stages: tuple[dict[str, float], ...] | None = None
    padding_factor: float = 2.0
    fade_width: int | None = None
    a0: float = 1.0
    update: str = "reference"
    gradient: str = "reference"
    pin_phase: bool = True
    precision: str = "single"
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.stages is None and self.schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {self.schedule!r}, choose from {list(SCHEDULES)}")
        if self.precision not in ("single", "double"):
            raise ValueError(f"precision must be 'single' or 'double', got {self.precision!r}")
        self.settings(1)

    def full_resolution_stages(self) -> tuple[ASRMStage, ...]:
        if self.stages is None:
            return SCHEDULES[self.schedule]
        return tuple(_stage(stage, i) for i, stage in enumerate(self.stages))

    def settings(self, binning: int) -> ASRMSettings:
        """:class:`~holofocus.asrm.ASRMSettings` for data binned by ``binning``."""
        fade = self.fade_width
        return ASRMSettings(
            stages=schedule_for_binning(self.full_resolution_stages(), binning),
            padding_factor=self.padding_factor,
            fade_width=80 // binning if fade is None else fade,
            a0=self.a0,
            update=self.update,
            gradient=self.gradient,
            pin_phase=self.pin_phase,
        )


@dataclass(frozen=True)
class SearchConfig:
    """Nelder-Mead over ``z01`` ([2] Sec. 3.4): ``Omega_z = guess +- confidence``.

    ``method: brent`` searches with Brent's bounded method instead (Stage 9,
    :func:`~holofocus.autofocus.find_focus`).
    """

    confidence: float = 5.0e-3
    xatol: float = 1.0e-4
    fatol: float = 1.0e6
    max_evaluations: int | None = None
    method: str = "nelder-mead"

    def __post_init__(self) -> None:
        if self.confidence <= 0 or self.xatol <= 0:
            raise ValueError("confidence and xatol must be positive")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")


@dataclass(frozen=True)
class BaselineConfig:
    """Method settings of a baseline run; the data come from the command line.

    ``evaluate_truth`` adds one reconstruction at the true ``z01`` per sample, so the MFE
    at the estimate can be compared with the MFE at the truth (optimizer vs. criterion).
    """

    name: str = "baseline"
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    evaluate_truth: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineConfig:
        return build(cls, data, "config", _SECTIONS)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BaselineConfig:
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        """Plain values; the search method only if it is not Nelder-Mead, so that runs made
        before it existed keep their key."""
        data = plain(dataclasses.asdict(self))
        if self.search.method == "nelder-mead":
            del data["search"]["method"]
        return data

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def _stage(data: dict[str, float], index: int) -> ASRMStage:
    types = {f.name: f.type for f in dataclasses.fields(ASRMStage)}
    unknown = set(data) - set(types)
    if unknown:
        raise ValueError(f"config.reconstruction.stages[{index}]: unknown keys {sorted(unknown)}")
    values = {}
    for key, value in data.items():
        if types[key] == "int" and not float(value).is_integer():
            raise ValueError(f"config.reconstruction.stages[{index}].{key} must be an integer")
        values[key] = int(value) if types[key] == "int" else float(value)
    return ASRMStage(**values)


_SECTIONS = {"reconstruction": ReconstructionConfig, "search": SearchConfig}
