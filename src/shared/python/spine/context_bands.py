"""Golf literature context bands, reported beside model results (SPN-11b).

A context band is a published lumbar load value (Hosea 1990 at L3-L4, Lim 2012
at L4-L5) kept for orientation only. :func:`annotate` places each model peak
next to the matching bands as a ratio. A comparison is never an acceptance
criterion: every :class:`ContextComparison` carries ``is_acceptance`` fixed to
``False``, and the field cannot be set to ``True``.

Band values come only from ``docs/research/segmental_spine/data_extraction.md``
via ``src/config/spine_golf_context_bands.json``. Each band keeps the
``verified_from_primary`` flag of its source row. Model magnitudes are compared
in the band's own units: newtons (N) or body weights (BW, ``m * g``).
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.shared.python.core.constants import GRAVITY_FLOAT
from src.shared.python.spine.levels import SpinalLevel, parse_level
from src.shared.python.spine.loads import SpinalLoadSeries

__all__ = [
    "DEFAULT_BANDS_PATH",
    "ContextBand",
    "ContextComparison",
    "annotate",
    "context_bands",
    "load_context_bands",
]

#: Repository copy of the band table (repository root is four levels up).
DEFAULT_BANDS_PATH: Path = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "config"
    / "spine_golf_context_bands.json"
)

_QUANTITIES: tuple[str, ...] = ("compression", "ap_shear", "ml_shear")
_UNITS: tuple[str, ...] = ("N", "BW")
_QUALIFIERS: tuple[str, ...] = ("exact", "greater_than", "approximately")
_ROW_KEYS: frozenset[str] = frozenset(
    {
        "band_id",
        "study",
        "doi",
        "table_or_page",
        "source_table_id",
        "level",
        "quantity",
        "value",
        "units",
        "body_weight_normalised",
        "qualifier",
        "subjects",
        "club",
        "method",
        "verified_from_primary",
    }
)


def _check_quantity(quantity: object) -> None:
    if not isinstance(quantity, str):
        raise TypeError(f"quantity must be a str, got {type(quantity).__name__}")
    if quantity not in _QUANTITIES:
        raise ValueError(f"quantity must be one of {_QUANTITIES}, got {quantity!r}")


@dataclass(frozen=True)
class ContextBand:
    """One published golf lumbar load value.

    Preconditions (``TypeError``/``ValueError``): ``level`` is a
    :class:`SpinalLevel`; ``quantity`` is one of ``compression``, ``ap_shear``,
    ``ml_shear``; ``units`` is ``N`` or ``BW``; ``value`` is finite and
    positive; ``body_weight_normalised`` is true exactly when ``units`` is
    ``BW``; ``qualifier`` is ``exact``, ``greater_than`` or ``approximately``.
    """

    band_id: str
    study: str
    doi: str | None
    table_or_page: str
    source_table_id: str
    level: SpinalLevel
    quantity: str
    value: float
    units: str
    body_weight_normalised: bool
    qualifier: str
    subjects: str
    club: str | None
    method: str
    verified_from_primary: bool

    def __post_init__(self) -> None:
        if not isinstance(self.level, SpinalLevel):
            raise TypeError("level must be a SpinalLevel")
        _check_quantity(self.quantity)
        if self.units not in _UNITS:
            raise ValueError(f"units must be one of {_UNITS}, got {self.units!r}")
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise TypeError(f"value of {self.band_id} must be a real number")
        if not math.isfinite(self.value) or self.value <= 0:
            raise ValueError(f"value of {self.band_id} must be finite and positive")
        if self.body_weight_normalised is not (self.units == "BW"):
            raise ValueError(
                f"{self.band_id}: body_weight_normalised must match units {self.units}"
            )
        if self.qualifier not in _QUALIFIERS:
            raise ValueError(f"qualifier must be one of {_QUALIFIERS}")


@dataclass(frozen=True)
class ContextComparison:
    """A model peak placed beside one context band.

    ``model_peak`` is a non-negative magnitude in ``model_units`` (which equals
    ``band.units``); ``ratio = model_peak / band.value``. ``is_acceptance`` is
    always ``False``: a comparison is reported, never used as a pass/fail gate.
    """

    band: ContextBand
    model_peak: float
    model_units: str
    ratio: float
    is_acceptance: Literal[False] = False

    def __post_init__(self) -> None:
        if not isinstance(self.band, ContextBand):
            raise TypeError("band must be a ContextBand")
        if self.is_acceptance is not False:
            raise ValueError("ContextComparison is never an acceptance criterion")
        if self.model_units != self.band.units:
            raise ValueError("model_units must equal the band units")
        if not math.isfinite(self.model_peak) or self.model_peak < 0:
            raise ValueError("model_peak must be finite and non-negative")


def _row_to_band(row: dict[str, Any]) -> ContextBand:
    missing = _ROW_KEYS - row.keys()
    unknown = row.keys() - _ROW_KEYS
    if missing or unknown:
        raise ValueError(
            f"band row keys mismatch: missing {sorted(missing)}, "
            f"unknown {sorted(unknown)}"
        )
    return ContextBand(
        band_id=row["band_id"],
        study=row["study"],
        doi=row["doi"],
        table_or_page=row["table_or_page"],
        source_table_id=row["source_table_id"],
        level=parse_level(row["level"]),
        quantity=row["quantity"],
        value=row["value"],
        units=row["units"],
        body_weight_normalised=row["body_weight_normalised"],
        qualifier=row["qualifier"],
        subjects=row["subjects"],
        club=row["club"],
        method=row["method"],
        verified_from_primary=row["verified_from_primary"],
    )


def _parse_bands(path: Path) -> tuple[ContextBand, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise TypeError(f"{path} must contain a JSON list of bands")
    bands = []
    for row in document:
        if not isinstance(row, dict):
            raise TypeError("each band must be a JSON object")
        bands.append(_row_to_band(row))
    return tuple(bands)


@functools.lru_cache(maxsize=1)
def _default_bands() -> tuple[ContextBand, ...]:
    return _parse_bands(DEFAULT_BANDS_PATH)


def load_context_bands(path: str | Path | None = None) -> tuple[ContextBand, ...]:
    """Return the context bands from ``path`` (default: the repository file).

    Preconditions: ``path`` is ``None``, ``str`` or ``Path``; the document is a
    JSON list of objects with exactly the keys of a :class:`ContextBand`.

    Raises:
        TypeError: if ``path`` has the wrong type or the document shape is wrong.
        ValueError: on unknown or missing keys, or an invalid band value.
    """
    if path is None:
        return _default_bands()
    if not isinstance(path, str | Path):
        raise TypeError("path must be None, str or Path")
    return _parse_bands(Path(path))


def context_bands(level: SpinalLevel, quantity: str) -> tuple[ContextBand, ...]:
    """Return the default-file bands for ``level`` and ``quantity``.

    Preconditions: ``level`` is a :class:`SpinalLevel`; ``quantity`` is
    ``compression``, ``ap_shear`` or ``ml_shear``.

    Postconditions: every returned band has the given level and quantity; the
    result is empty when no published band matches.

    Raises:
        TypeError: if ``level`` is not a :class:`SpinalLevel`.
        ValueError: if ``quantity`` is unknown.
    """
    if not isinstance(level, SpinalLevel):
        raise TypeError(f"level must be a SpinalLevel, got {type(level).__name__}")
    _check_quantity(quantity)
    return tuple(
        band
        for band in load_context_bands()
        if band.level is level and band.quantity == quantity
    )


def _body_weight_n(body_mass_kg: object) -> float:
    if isinstance(body_mass_kg, bool) or not isinstance(
        body_mass_kg, int | float | np.floating | np.integer
    ):
        raise TypeError("body_mass_kg must be a real number")
    mass = float(body_mass_kg)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("body_mass_kg must be finite and positive")
    weight = mass * GRAVITY_FLOAT
    if weight <= 0:
        raise ValueError("body weight must be positive")
    return weight


def _peak_magnitudes(
    series: SpinalLoadSeries, quantity: str, weight_n: float
) -> dict[str, dict[SpinalLevel, float]]:
    """Per-level peak magnitude of ``quantity`` in both N and BW."""
    magnitudes = {
        level: abs(value) for level, (value, _t) in series.peak(quantity).items()
    }
    if series.force_unit == "N":
        newtons = magnitudes
        body_weights = {lv: m / weight_n for lv, m in magnitudes.items()}
    else:
        body_weights = magnitudes
        newtons = {lv: m * weight_n for lv, m in magnitudes.items()}
    return {"N": newtons, "BW": body_weights}


def annotate(series: SpinalLoadSeries, body_mass_kg: float) -> list[ContextComparison]:
    """Compare each level's peak load with the matching context bands.

    For every quantity and every level of ``series``, the largest-magnitude
    peak is compared with each default-file band of that level and quantity,
    in the band's units (N, or BW using ``body_mass_kg * g``). A series in N
    or BW gives the same ratios.

    Preconditions: ``series`` is a :class:`SpinalLoadSeries`; ``body_mass_kg``
    is a finite positive real number (``TypeError``/``ValueError`` otherwise).

    Postconditions: every comparison has ``is_acceptance is False``; bands for
    levels not in the series are omitted.
    """
    if not isinstance(series, SpinalLoadSeries):
        raise TypeError("series must be a SpinalLoadSeries")
    weight_n = _body_weight_n(body_mass_kg)
    comparisons: list[ContextComparison] = []
    for quantity in _QUANTITIES:
        peaks = _peak_magnitudes(series, quantity, weight_n)
        for level in series.levels:
            for band in context_bands(level, quantity):
                model_peak = peaks[band.units][level]
                comparisons.append(
                    ContextComparison(
                        band=band,
                        model_peak=model_peak,
                        model_units=band.units,
                        ratio=model_peak / band.value,
                    )
                )
    return comparisons
