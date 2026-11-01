"""Reference series for putting validation, each with explicit provenance.

Only values traceable to a file in this repository are stored. Broadie (2014)
table A.5, ShotLink and amateur tables are not here, so they are listed as
unavailable with a reason instead of being retyped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from src.shared.python.contracts import require

PUTTING_SOURCE_PATH = "src/shared/python/sg_optimizer/shot_model/putting.py"


def is_strictly_increasing(values: Sequence[float]) -> bool:
    """Return True when each value is greater than its predecessor."""
    return all(a < b for a, b in pairwise(values))


class License(str, Enum):
    """Redistribution status of a reference source."""

    PUBLIC = "public"
    LICENSED = "licensed"


class Quantity(str, Enum):
    """What a reference series measures, with its value units."""

    MAKE_FRACTION = "fraction"
    EXPECTED_PUTTS = "putts"


@dataclass(frozen=True)
class Provenance:
    """Where a series comes from. Distances are always in feet."""

    citation: str
    table: str
    value_units: str
    population: str
    license: License
    handicap_band: str | None = None
    distance_units: str = "feet"

    def __post_init__(self) -> None:
        for name in ("citation", "table", "value_units", "population"):
            require(bool(getattr(self, name).strip()), f"provenance {name} is blank")


@dataclass(frozen=True)
class ReferenceSeries:
    """A stored curve; ``validation_grade`` False marks model self-description."""

    series_id: str
    quantity: Quantity
    provenance: Provenance
    distances_ft: tuple[float, ...]
    values: tuple[float, ...]
    validation_grade: bool

    def __post_init__(self) -> None:
        require(len(self.distances_ft) == len(self.values) > 0, "bad series length")
        require(is_strictly_increasing(self.distances_ft), "distances not increasing")
        require(self.distances_ft[0] > 0, "distances_ft must be > 0")


@dataclass(frozen=True)
class UnavailableSeries:
    """A wanted reference series that cannot be supplied from this repository."""

    series_id: str
    quantity: Quantity
    provenance: Provenance
    reason: str

    def __post_init__(self) -> None:
        require(bool(self.reason.strip()), "unavailable series needs a reason")


MAKE, PUTTS = Quantity.MAKE_FRACTION, Quantity.EXPECTED_PUTTS
PUB, LIC = License.PUBLIC, License.LICENSED
_BROADIE = "Broadie, M. (2014). Every Shot Counts. Gotham Books."
_SHOTLINK = "PGA Tour ShotLink putting data (licensed)."
_AMATEUR = "Published amateur putting tables (source to be named by data owner)."
_NO_A5 = f"Table A.5 values are not stored here; {PUTTING_SOURCE_PATH} has only a fit."
_NO_EXPECTED = "No expected-putts-by-distance table exists in this repository."
_NO_LICENSE = "Licensed data; no license or extract is available to this repository."
_NO_AMATEUR = "No published amateur make-rate table is stored in this repository."

# (series_id, quantity, citation, table, population, license, band, reason)
_UNAVAILABLE_ROWS = (
    ("broadie_2014_tour_make_pct", MAKE, _BROADIE, "A.5", "tour", PUB, None, _NO_A5),
    ("broadie_2014_tour_putts", PUTTS, _BROADIE, "-", "tour", PUB, None, _NO_EXPECTED),
    ("shotlink_tour_make_pct", MAKE, _SHOTLINK, "-", "tour", LIC, None, _NO_LICENSE),
    ("shotlink_tour_putts", PUTTS, _SHOTLINK, "-", "tour", LIC, None, _NO_LICENSE),
    *(
        (f"amateur_make_pct_{b}", MAKE, _AMATEUR, "-", "amateur", PUB, b, _NO_AMATEUR)
        for b in ("scratch", "hcp10", "hcp20")
    ),
)
UNAVAILABLE_SERIES: dict[str, UnavailableSeries] = {
    sid: UnavailableSeries(
        sid, qty, Provenance(cite, table, qty.value, pop, lic, band), reason
    )
    for sid, qty, cite, table, pop, lic, band, reason in _UNAVAILABLE_ROWS
}

# Anchors quoted in the comment above ``_ALPHA`` in putting.py. They describe
# the existing logistic; they are not the published table.
AVAILABLE_SERIES: dict[str, ReferenceSeries] = {
    "sg_optimizer_documented_anchors": ReferenceSeries(
        series_id="sg_optimizer_documented_anchors",
        quantity=MAKE,
        provenance=Provenance(
            citation=f"Code comment in {PUTTING_SOURCE_PATH} (not a published table)",
            table="comment above _ALPHA: ~99% at 3 ft, ~50% at ~8 ft, ~7% at 30 ft",
            value_units=MAKE.value,
            population="tour",
            license=PUB,
        ),
        distances_ft=(3.0, 8.0, 30.0),
        values=(0.99, 0.50, 0.07),
        validation_grade=False,
    ),
}


def get_series(series_id: str) -> ReferenceSeries:
    """Return an available series; unknown or unavailable ids raise with a reason."""
    if series_id in UNAVAILABLE_SERIES:
        reason = UNAVAILABLE_SERIES[series_id].reason
        require(False, f"series {series_id!r} is unavailable: {reason}")
    require(series_id in AVAILABLE_SERIES, f"unknown series {series_id!r}", series_id)
    return AVAILABLE_SERIES[series_id]


def require_validation_grade(series: ReferenceSeries) -> None:
    """Refuse to use a non-published series as calibration evidence."""
    require(
        series.validation_grade,
        f"series {series.series_id!r} is not validation grade; "
        "it cannot support a calibration claim",
    )
