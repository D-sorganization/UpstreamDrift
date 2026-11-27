"""Can one shaft mode at its rated frequency bend the tip this fast? (#15150 C18)

For free vibration of one mode of angular frequency ``w = 2 pi f`` the energy
``|v|^2 + w^2 |d|^2`` is constant, so the bend vector ``d`` (the tip
displacement perpendicular to the straight shaft, any number of directions)
moves no faster than ``w max|d|``. A run whose peak bend rate exceeds that is
driven by more than the shaft's rated mode: a load (hand, grip, contact, a
tracker gain), centrifugal stiffening of the spinning club, higher modes, or a
measurement error. The check does not say which; it flags the run so its peak
bend is not quoted before its loads are checked.

The rate is the speed of the bend vector, not the rate of its norm: a vector
turning through zero (lag unloading into forward kick) makes the norm dip and
regrow quickly while the vector moves smoothly.

Inputs stop at impact; after contact the ball, not the shaft, sets the rate.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np

from src.shared.python.club_fitting.bend_trace import BendTrace

SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "bend_rate_check.json"
_MIN_SAMPLES = 3


@dataclass(frozen=True)
class BendRateSettings:
    """Flag threshold and unit conversions of the check."""

    ratio_limit: float
    mm_per_m: float
    seconds_per_minute: float

    def __post_init__(self) -> None:
        for f in fields(self):
            value = float(getattr(self, f.name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{f.name} must be positive and finite")


def load_bend_rate_settings(path: Path | None = None) -> BendRateSettings:
    """Settings from ``path`` (default ``src/config/bend_rate_check.json``).

    Raises:
        ValueError: missing or unknown keys, or a non-positive value.
    """
    source = path or SETTINGS_PATH
    data = json.loads(source.read_text(encoding="utf-8"))
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    schema = {f.name for f in fields(BendRateSettings)}
    if set(data) != schema:
        diff = sorted(set(data) ^ schema)
        raise ValueError(f"{source.name} keys differ from the schema: {diff}")
    return BendRateSettings(**data)


@dataclass(frozen=True)
class BendRateCheck:
    """Peak bend rate against the single-mode bound ``2 pi f max|d|``."""

    peak_rate_mps: float
    peak_rate_time_s: float
    peak_bend_m: float
    frequency_hz: float
    single_mode_bound_mps: float
    ratio: float
    flagged: bool


def _checked(time_s: np.ndarray, bend_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_s, dtype=float).reshape(-1)
    d = np.asarray(bend_m, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    if t.size < _MIN_SAMPLES:
        raise ValueError(f"the check needs at least three samples, got {t.size}")
    if d.ndim != 2 or d.shape[0] != t.size:
        raise ValueError("bend_m needs as many rows as time samples")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(d))):
        raise ValueError("time_s and bend_m must be finite")
    if not np.all(np.diff(t) > 0.0):
        raise ValueError("time_s must be strictly increasing")
    return t, d


def bend_rate_check(
    time_s: np.ndarray, bend_m: np.ndarray, frequency_hz: float, ratio_limit: float
) -> BendRateCheck:
    """Compare the bend vector's peak speed with one mode at ``frequency_hz``.

    Args:
        time_s: Increasing sample times (s), up to impact.
        bend_m: ``(n, k)`` bend vectors (m) perpendicular to the straight shaft.
        frequency_hz: The shaft's rated (clamped) frequency.
        ratio_limit: Flag when ``peak rate / bound`` exceeds this.

    Returns:
        The check; ``ratio <= 1`` for any single-mode free vibration.

    Raises:
        ValueError: fewer than three samples, mismatched or non-finite input,
            non-increasing times, a non-positive frequency or limit, or no bend.
    """
    t, d = _checked(time_s, bend_m)
    for name, value in (("frequency_hz", frequency_hz), ("ratio_limit", ratio_limit)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    peak_bend = float(np.linalg.norm(d, axis=1).max())
    if peak_bend <= 0.0:
        raise ValueError("the trace has no bend to check")
    speed = np.linalg.norm(np.gradient(d, t, axis=0), axis=1)
    k = int(np.argmax(speed))
    bound = 2.0 * np.pi * float(frequency_hz) * peak_bend
    ratio = float(speed[k]) / bound
    return BendRateCheck(
        peak_rate_mps=float(speed[k]),
        peak_rate_time_s=float(t[k]),
        peak_bend_m=peak_bend,
        frequency_hz=float(frequency_hz),
        single_mode_bound_mps=bound,
        ratio=ratio,
        flagged=ratio > float(ratio_limit),
    )


def check_bend_trace(
    trace: BendTrace, flex_cpm: float, settings: BendRateSettings | None = None
) -> BendRateCheck:
    """:func:`bend_rate_check` of a lead/droop trace (mm) at ``flex_cpm``."""
    s = settings or load_bend_rate_settings()
    bend = np.column_stack([trace.lead_mm, trace.droop_mm]) / s.mm_per_m
    return bend_rate_check(
        trace.time_s, bend, float(flex_cpm) / s.seconds_per_minute, s.ratio_limit
    )
