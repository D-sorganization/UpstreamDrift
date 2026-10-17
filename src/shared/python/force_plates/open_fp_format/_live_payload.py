"""Strict validation of one open-fp ``live`` batch (OF-09, #13088).

See :mod:`~src.shared.python.force_plates.open_fp_format.live` for the payload
shape.  Everything here is pure: a payload goes in, SI-unit arrays or a
:class:`LiveBatchError` with a loggable reason come out.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.shared.python.force_plates.contracts import NEGATIVE_LOAD_SIGMAS
from src.shared.python.force_plates.open_fp_format.format import (
    HTTP_FORCE_QUANTUM_N,
    MM_PER_M,
    SOURCE_COP_MIN_LOAD_N,
)

__all__ = ["LiveBatch", "LiveBatchError", "negative_load_floor_n", "parse_live_batch"]

#: Every key the board's ``live`` message carries (no more, no fewer).
LIVE_KEYS: frozenset[str] = frozenset(
    {
        "t",
        "fz",
        "cop",
        "lead_pct",
        "forces",
        "flags",
        "state",
        "bw_n",
        "lead_plate",
        "golfer",
    }
)
PLATE_KEYS: frozenset[str] = frozenset({"L", "R"})
COP_KEYS: frozenset[str] = frozenset({"L", "R", "C"})
CELLS_PER_SAMPLE = 8  # four load cells per plate, two plates
#: Loads are rounded to ``HTTP_FORCE_QUANTUM_N``, so a rounded load is within
#: half a quantum of the true one: CoP may be present down to the threshold
#: minus that, and absent up to the threshold plus it.
_HALF_QUANTUM_N = 0.5 * HTTP_FORCE_QUANTUM_N
_COP_LOAD_FLOOR_N = SOURCE_COP_MIN_LOAD_N - _HALF_QUANTUM_N
_NULL_COP_CEILING_N = SOURCE_COP_MIN_LOAD_N + _HALF_QUANTUM_N
_FLOAT_EPS = 1e-9


def negative_load_floor_n(force_floor_n: float) -> float:
    """Lowest accepted rounded load [N]: the trial contract's noise rule.

    An unloaded plate reads slightly negative from noise; like
    :class:`~src.shared.python.force_plates.contracts.VerticalPlateTrial`, a
    value is kept unclamped down to ``NEGATIVE_LOAD_SIGMAS`` declared floor
    sigmas (here also allowing half a rounding quantum).
    """
    return -(NEGATIVE_LOAD_SIGMAS * force_floor_n + _HALF_QUANTUM_N)


_SIDE = {"L": "left", "R": "right"}


class LiveBatchError(ValueError):
    """A ``live`` batch violated the documented shape; the message says how."""


@dataclass(frozen=True, slots=True)
class LiveBatch:
    """One validated batch in SI units (seconds, newtons, metres)."""

    t_s: list[float]
    load_n: dict[str, list[float]]
    cop_m: dict[str, list[list[float] | None]]

    def __len__(self) -> int:
        return len(self.t_s)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LiveBatchError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise LiveBatchError(f"{name} must be finite, got {value!r}")
    return number


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, list | tuple):
        raise LiveBatchError(f"{name} must be a list")
    return value


def _keyed(value: object, expected: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise LiveBatchError(f"{name} must be an object with keys {sorted(expected)}")
    return value


def _times(raw: object) -> list[float]:
    times = [_number(v, "t") for v in _sequence(raw, "t")]
    if any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise LiveBatchError("t must be strictly increasing")
    return times


def _loads(raw: object, count: int, plate: str, floor_n: float) -> list[float]:
    values = _sequence(raw, f"fz.{plate}")
    if len(values) != count:
        raise LiveBatchError(f"fz.{plate} has {len(values)} samples, t has {count}")
    loads = [_number(v, f"fz.{plate}") for v in values]
    if any(v < floor_n - _FLOAT_EPS for v in loads):
        raise LiveBatchError(
            f"fz.{plate} has a load below {floor_n} N (beyond the noise floor)"
        )
    return loads


def _cop_point(row: object, load_n: float, name: str) -> list[float] | None:
    """One CoP row in metres, or ``None``; its presence must follow the load."""
    if row is None:
        if load_n > _NULL_COP_CEILING_N + _FLOAT_EPS:
            raise LiveBatchError(f"{name} is null at {load_n} N (>= 20 N)")
        return None
    pair = _sequence(row, name)
    if len(pair) != 2:
        raise LiveBatchError(f"{name} row must have two coordinates")
    if load_n < _COP_LOAD_FLOOR_N - _FLOAT_EPS:
        raise LiveBatchError(f"{name} is present at {load_n} N (< 20 N)")
    return [_number(v, name) / MM_PER_M for v in pair]


def _cop_rows(raw: object, loads: list[float], plate: str) -> list[list[float] | None]:
    rows = _sequence(raw, f"cop.{plate}")
    if len(rows) != len(loads):
        raise LiveBatchError(f"cop.{plate} has {len(rows)} rows, t has {len(loads)}")
    return [_cop_point(r, w, f"cop.{plate}") for r, w in zip(rows, loads, strict=True)]


def _check_aux(body: dict[str, Any], count: int) -> None:
    """Shape-check the keys that are accepted but not forwarded."""
    forces = _sequence(body["forces"], "forces")
    if len(forces) != count:
        raise LiveBatchError(f"forces has {len(forces)} rows, t has {count}")
    for row in forces:
        cells = _sequence(row, "forces")
        if len(cells) != CELLS_PER_SAMPLE:
            raise LiveBatchError(f"forces rows need {CELLS_PER_SAMPLE} cells")
        for cell in cells:
            _number(cell, "forces")
    lead = _sequence(body["lead_pct"], "lead_pct")
    if len(lead) != count:
        raise LiveBatchError(f"lead_pct has {len(lead)} samples, t has {count}")
    for value in lead:
        if value is not None:
            _number(value, "lead_pct")
    if isinstance(body["flags"], bool) or not isinstance(body["flags"], int):
        raise LiveBatchError("flags must be an integer")
    if body["bw_n"] is not None:
        _number(body["bw_n"], "bw_n")
    for key in ("state", "lead_plate", "golfer"):
        if body[key] is not None and not isinstance(body[key], str):
            raise LiveBatchError(f"{key} must be a string or null")


def parse_live_batch(payload: object, *, negative_floor_n: float) -> LiveBatch:
    """Validate a decoded ``live`` message and convert it to SI units.

    ``negative_floor_n`` is the lowest accepted load
    (:func:`negative_load_floor_n` of the declared force floor).

    Raises:
        LiveBatchError: unknown or missing keys, ragged or non-finite series,
            non-increasing times, loads below the noise floor, or CoP presence that
            disagrees with the 20 N source threshold.
    """
    body = _keyed(payload, LIVE_KEYS, "live message")
    times = _times(body["t"])
    fz = _keyed(body["fz"], PLATE_KEYS, "fz")
    cop = _keyed(body["cop"], COP_KEYS, "cop")
    loads = {
        p: _loads(fz[p], len(times), p, negative_floor_n) for p in sorted(PLATE_KEYS)
    }
    cop_m = {_SIDE[p]: _cop_rows(cop[p], loads[p], p) for p in sorted(PLATE_KEYS)}
    _cop_shape_only(cop["C"], len(times))
    _check_aux(body, len(times))
    return LiveBatch(
        t_s=times,
        load_n={_SIDE[p]: v for p, v in loads.items()},
        cop_m=cop_m,
    )


def _cop_shape_only(raw: object, count: int) -> None:
    """The net CoP is not forwarded; only its shape and finiteness are checked."""
    rows = _sequence(raw, "cop.C")
    if len(rows) != count:
        raise LiveBatchError(f"cop.C has {len(rows)} rows, t has {count}")
    for row in rows:
        if row is not None:
            for coordinate in _sequence(row, "cop.C"):
                _number(coordinate, "cop.C")
