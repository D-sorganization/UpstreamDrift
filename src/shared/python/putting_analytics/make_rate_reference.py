"""Published putting make rates, kept as cited reference data (#14177).

``data/make_rates.json`` holds every published make-rate table we could find,
by distance and by handicap, each with its publisher, URL, date, population and
underlying data. GolfMagic's PGA Tour table (ShotLink) is the calibration
target; Shot Scope's handicap bands are the amateur check; the rest are kept
for comparison only. See the file's ``provenance`` note for how the figures
were transcribed.

A point is an exact distance or a band. Open bands ("30 ft and over") are
read as running to :data:`OPEN_BAND_CAP_FT`, the longest distance in any
tour table here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.shared.python.contracts import require

MAKE_RATES_PATH = Path(__file__).resolve().parent / "data" / "make_rates.json"
OPEN_BAND_CAP_FT = 60.0
ROLES = ("calibration_target", "check", "reference")
METRICS = ("make_pct", "one_putt_pct")


@dataclass(frozen=True)
class MakeRatePoint:
    """One published figure: make % over ``[low_ft, high_ft]`` (equal if exact)."""

    low_ft: float
    high_ft: float
    pct: float
    approx: bool = False

    def __post_init__(self) -> None:
        require(0.0 <= self.low_ft <= self.high_ft, "band must be ordered", self)
        require(0.0 <= self.pct <= 100.0, "pct must be in [0, 100]", self.pct)

    @property
    def is_band(self) -> bool:
        """True when the figure covers a range of distances."""
        return self.high_ft > self.low_ft


@dataclass(frozen=True)
class MakeRateTable:
    """One population's figures from one cited source."""

    source_id: str
    group: str
    role: str
    metric: str
    publisher: str
    url: str
    published: str | None
    underlying_data: str
    points: tuple[MakeRatePoint, ...]

    def __post_init__(self) -> None:
        require(self.role in ROLES, f"role must be one of {ROLES}", self.role)
        require(self.metric in METRICS, f"metric must be one of {METRICS}")
        require(len(self.points) > 0, "a table needs at least one point", self)


def _point(raw: list[Any]) -> MakeRatePoint:
    where, pct, *flags = raw
    require(flags in ([], ["approx"]), "only an 'approx' flag is allowed", raw)
    if isinstance(where, list):
        low, high = where
        top = OPEN_BAND_CAP_FT if high is None else float(high)
        return MakeRatePoint(float(low), top, float(pct), bool(flags))
    return MakeRatePoint(float(where), float(where), float(pct), bool(flags))


def _tables(source: dict[str, Any]) -> list[MakeRateTable]:
    meta = {
        "source_id": source["id"],
        "role": source["role"],
        "metric": source["metric"],
        "publisher": source["publisher"],
        "url": source["url"],
        "published": source["published"],
        "underlying_data": source["underlying_data"],
    }
    return [
        MakeRateTable(group=group, points=tuple(map(_point, rows)), **meta)
        for group, rows in source["groups"].items()
    ]


@lru_cache(maxsize=4)
def load_reference_tables(path: Path = MAKE_RATES_PATH) -> tuple[MakeRateTable, ...]:
    """Every table in the reference file, in file order."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tables = [table for source in data["sources"] for table in _tables(source)]
    keys = [(t.source_id, t.group) for t in tables]
    require(len(set(keys)) == len(keys), "source and group pairs must be unique")
    return tuple(tables)


def reference_table(source_id: str, group: str) -> MakeRateTable:
    """The table for one source and population group."""
    for table in load_reference_tables():
        if (table.source_id, table.group) == (source_id, group):
            return table
    raise KeyError(f"no make-rate table for {source_id!r} / {group!r}")
