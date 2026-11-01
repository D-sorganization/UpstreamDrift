"""Versioned putting-analytics report with JSON round trip (issue #14174)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, get_args, get_origin, get_type_hints

from src.shared.python.contracts import require
from src.shared.python.putting_analytics.library import PuttClass

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Estimate:
    """A Monte Carlo mean and its standard error."""

    mean: float
    se: float


@dataclass(frozen=True)
class ClassResult:
    """Best action and outcome statistics for one putt class."""

    putt_class: PuttClass
    weight: float
    aim_offset_deg: float
    pace_ft_past: float
    make_rate: Estimate
    expected_putts: Estimate
    sg_player: Estimate
    sg_tour: Estimate


@dataclass(frozen=True)
class BandResult:
    """Make rate over the classes with ``lo_ft <= distance < hi_ft``."""

    lo_ft: float
    hi_ft: float
    weight: float
    make_rate: Estimate


@dataclass(frozen=True)
class PaceResult:
    """Weighted chosen pace (ft past; an action, not a Monte Carlo mean)."""

    distance_ft: float
    stimp: float
    weight: float
    pace_ft_past: float


@dataclass(frozen=True)
class Totals:
    """Weighted per-round aggregates (``holes_per_round`` first putts)."""

    make_rate: Estimate
    expected_putts_per_round: Estimate
    sg_player_per_round: Estimate
    sg_tour_per_round: Estimate


@dataclass(frozen=True)
class PuttingReport:
    """Everything needed to reproduce and audit one aggregate run."""

    seed: int
    samples_per_action: int
    objective: str
    holes_per_round: int
    classes: tuple[ClassResult, ...]
    bands: tuple[BandResult, ...]
    pace: tuple[PaceResult, ...]
    totals: Totals
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> PuttingReport:
        """Rebuild a report; the schema version must match."""
        data = json.loads(text)
        require(data.get("schema_version") == SCHEMA_VERSION, "unsupported schema")
        report: PuttingReport = _build(cls, data)
        return report


def _build(cls: Any, data: Any) -> Any:
    """Rebuild a (nested) dataclass from ``asdict`` output using its type hints."""
    hints, kwargs = get_type_hints(cls), {}
    for f in fields(cls):
        kind, value = hints[f.name], data[f.name]
        if get_origin(kind) is tuple:
            value = tuple(_build(get_args(kind)[0], v) for v in value)
        elif is_dataclass(kind):
            value = _build(kind, value)
        kwargs[f.name] = value
    return cls(**kwargs)
