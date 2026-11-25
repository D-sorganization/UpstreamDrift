"""Validate optional saved visualization traces without rerunning physics (#15497)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from bunkershot3d.solvers import EnvelopeStatus

from .flight import BunkerFlight


def validate_saved_flight(record: Mapping[str, Any]) -> None:
    """Reject unqualified/incoherent flight data; leave legacy records unchanged.

    Preconditions: ``record`` is a decoded evaluation JSON object. A non-null
    flight requires non-refused shot/carry verdicts and an explicit assumption.
    Postcondition: any trace present satisfies the existing ``BunkerFlight``
    invariants and agrees with the adjacent carry; no input data is modified.
    Raises ``ValueError`` on invalid data, even when optional DbC is disabled.
    """
    trace = record.get("flight")
    if trace is None:  # Additive v1 field: older records have no flight key.
        return
    if not isinstance(trace, dict):
        raise ValueError("flight must be an object or null")
    if record.get("refused") is not False:
        raise ValueError("flight requires an explicitly non-refused shot")
    _require_nonrefused_status(record.get("verdict"), "status")
    carry = _require_nonrefused_status(record.get("carry"), "verdict_status")
    assumption = trace.get("assumption")
    if not isinstance(assumption, str) or not assumption.strip():
        raise ValueError("flight requires an explicit assumption")
    flight = BunkerFlight(
        _path(trace.get("flight_m")),
        _path(trace.get("rollout_m")),
        _number(trace.get("carry_m")),
        assumption,
    )
    if not math.isclose(
        flight.carry_m, _number(carry.get("carry_m")), rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError("flight carry must match the evaluation carry")


def _require_nonrefused_status(value: Any, key: str) -> dict[str, Any]:
    """Require a recognized non-refused verdict before accepting coordinates."""
    if not isinstance(value, dict) or not isinstance(value.get(key), str):
        raise ValueError("flight requires non-refused shot and carry verdicts")
    if value[key] not in {
        status.value
        for status in EnvelopeStatus
        if status is not EnvelopeStatus.REFUSED
    }:
        raise ValueError("flight requires non-refused shot and carry verdicts")
    return value


def _number(value: Any) -> float:
    """Accept finite JSON numbers without coercing strings or booleans."""
    if type(value) not in (int, float):
        raise ValueError("flight coordinates and carry must be finite numbers")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("flight number is outside the finite range") from exc
    if not math.isfinite(number):
        raise ValueError("flight coordinates and carry must be finite numbers")
    return number


def _path(value: Any) -> tuple[tuple[float, float, float], ...]:
    """Decode xyz arrays; geometric invariants belong to ``BunkerFlight``."""
    if not isinstance(value, list):
        raise ValueError("flight paths must be arrays of xyz points")
    points = []
    for point in value:
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError("flight paths must contain xyz points")
        points.append((_number(point[0]), _number(point[1]), _number(point[2])))
    return tuple(points)
