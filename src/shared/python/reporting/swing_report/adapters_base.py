"""Engine-result input model and adapters for the swing report builder (#12325).

``EngineResult`` is the single, engine-agnostic hand-off the builder consumes.
Adapters only *re-shape* data an engine already produced (for example the
``ReportingMixin`` host's ``times`` / ``club_head_speed`` arrays); they never
compute new quantities, and non-finite samples become ``None`` (missing), never
zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

CLUBHEAD_SPEED = "clubhead_speed"


@dataclass(frozen=True)
class QuantityData:
    """One engine-supplied series for a quantity id (no uncertainty here)."""

    units: str
    frame: str
    time_s: tuple[float, ...]
    values: tuple[float | None, ...]
    channel: str | None = None


@dataclass(frozen=True)
class EngineResult:
    """What one engine run supplies to the report.

    ``error`` marks an engine that failed or is unavailable; the builder then
    records the run but ignores any quantities.
    """

    engine: str
    run_id: str
    engine_version: str | None
    model_file_hash: str | None
    run_class: str
    quantities: Mapping[str, QuantityData] = field(default_factory=dict)
    force_provenance: str | None = None
    error: str | None = None

    @property
    def healthy(self) -> bool:
        return self.error is None


def finite_or_none(value: Any) -> float | None:
    """Return ``float(value)`` if finite, else ``None`` (never zero-fill)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def result_from_reporting_mixin(
    analyzer: Any,
    *,
    engine: str,
    run_id: str,
    engine_version: str | None,
    model_file_hash: str | None,
    run_class: str,
    force_provenance: str | None = None,
) -> EngineResult:
    """Adapt a ``ReportingMixin`` host into an ``EngineResult``.

    Reads the arrays the mixin already reports from (``times`` and
    ``club_head_speed``); a missing speed array yields no quantity.
    """
    times = getattr(analyzer, "times", None)
    speed = getattr(analyzer, "club_head_speed", None)
    quantities: dict[str, QuantityData] = {}
    if times is not None and speed is not None and len(times) == len(speed):
        quantities[CLUBHEAD_SPEED] = QuantityData(
            units="m/s",
            frame="world_Zup",
            time_s=tuple(float(t) for t in times),
            values=tuple(finite_or_none(v) for v in speed),
        )
    elif speed is not None:
        logger.warning("club_head_speed/times length mismatch; speed not adapted")
    return EngineResult(
        engine=engine,
        run_id=run_id,
        engine_version=engine_version,
        model_file_hash=model_file_hash,
        run_class=run_class,
        quantities=quantities,
        force_provenance=force_provenance,
    )
