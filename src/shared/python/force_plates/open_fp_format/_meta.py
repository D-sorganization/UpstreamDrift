"""Strict parsing of the swing ``.json`` file (fail closed on any deviation)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, get_args

from src.shared.python.force_plates.contracts import Handedness, finite_positive
from src.shared.python.force_plates.open_fp_format import format as fmt

__all__ = ["SwingMeta", "parse_meta"]


@dataclass(frozen=True)
class SwingMeta:
    """The fields of a swing JSON that the trial contract uses."""

    handedness: Handedness
    body_weight_n: float
    events_rel_s: Mapping[str, float | None]
    impact_source: str | None


def _reject_constant(name: str, constant: str) -> NoReturn:
    raise ValueError(f"{name}: non-finite JSON literal {constant} is not allowed")


def _number(value: object, where: str) -> float:
    """A JSON number as a finite float; ``ValueError`` otherwise (never overflows)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{where} is too large to be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite, got {value!r}")
    return number


def _unique_object(name: str, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A JSON object, refusing a repeated member name at any level."""
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        repeated = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"{name}: duplicate JSON members {repeated}")
    return dict(pairs)


def _load_object(data: bytes, path: Path) -> dict[str, Any]:
    name = str(path)
    try:
        text = data.decode("utf-8")
        obj = json.loads(
            text,
            parse_constant=lambda c: _reject_constant(name, c),
            object_pairs_hook=lambda pairs: _unique_object(name, pairs),
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name}: not valid UTF-8 JSON ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name}: invalid JSON ({exc})") from exc
    except RecursionError as exc:
        raise ValueError(f"{name}: JSON is nested too deeply") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{name}: top-level JSON value must be an object")
    return obj


def _check_keys(obj: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    missing, extra = sorted(expected - set(obj)), sorted(set(obj) - expected)
    if missing or extra:
        raise ValueError(f"{where}: missing keys {missing}, unexpected keys {extra}")


def _check_events(raw: object, path: Path) -> dict[str, float | None]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 'events' must be an object")
    _check_keys(raw, fmt.EVENT_NAMES, f"{path}: events")
    events: dict[str, float | None] = {}
    for key, value in raw.items():
        events[key] = (
            None if value is None else _number(value, f"{path}: events[{key!r}]")
        )
    return events


def _check_scalars(obj: Mapping[str, Any], path: Path) -> None:
    if obj["id"] != path.stem:
        raise ValueError(f"{path}: 'id' {obj['id']!r} does not equal the file stem")
    created = obj["created"]
    try:
        stamp = datetime.fromisoformat(created) if isinstance(created, str) else None
    except ValueError:
        stamp = None
    if stamp is None or stamp.tzinfo is not None:
        raise ValueError(f"{path}: 'created' must be a naive ISO time, got {created!r}")
    _number(obj["t0_host"], f"{path}: t0_host")
    source = obj["impact_source"]
    if source is not None and (
        not isinstance(source, str) or source not in fmt.IMPACT_SOURCES
    ):
        raise ValueError(f"{path}: invalid impact_source {source!r}")


def parse_meta(data: bytes, path: Path) -> SwingMeta:
    """Parse and validate the swing JSON bytes of ``path``.

    Raises ``ValueError`` on invalid JSON, NaN/Infinity literals, a key set other
    than the documented one, wrong types, an ``id`` that differs from the file
    stem, or an ``impact`` event set while ``impact_source`` is null.
    """
    obj = _load_object(data, path)
    _check_keys(obj, fmt.META_KEYS, str(path))
    _check_scalars(obj, path)
    golfer = obj["golfer"]
    if golfer not in get_args(Handedness):
        raise ValueError(f"{path}: golfer must be 'RH' or 'LH', got {golfer!r}")
    body_weight = finite_positive(
        _number(obj["bw_n"], f"{path}: bw_n"), f"{path}: bw_n"
    )
    events = _check_events(obj["events"], path)
    if obj["impact_source"] is None and events["impact"] is not None:
        raise ValueError(f"{path}: impact_source is null but the impact event is set")
    return SwingMeta(golfer, body_weight, events, obj["impact_source"])
