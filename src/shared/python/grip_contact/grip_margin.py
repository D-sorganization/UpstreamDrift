"""Two-hand grip margin over pad-contact wrenches (#13885 PR 2).

A pad's normalised Coulomb margin is ``(mu f_n - |f_t|) / (mu f_n)``: 1 with
no tangential load, 0 on the cone, negative outside it. The capacity is
:func:`friction_band.unilateral_friction_capacity` of ``pad_contact``'s
per-pad normal force, so the cone is consumed, not re-derived. Pads with
``f_n`` below a declared floor are :data:`UNLOADED` and excluded from the
minimum. With both hands' wrenches about the club grip origin the hand-hand
grasp map is ``G = [I I]``: the net wrench is ``w_lead + w_trail`` and the
internal (squeeze) wrench, its null-space part, is ``(w_lead - w_trail) / 2``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from src.shared.python.grip_contact.contact_topology import HAND_ROLES
from src.shared.python.grip_contact.friction_band import (
    unilateral_friction_capacity,
)
from src.shared.python.grip_contact.pad_contact import PadWrench
from src.shared.python.physics.grip_contact_model import SLIP_VELOCITY_THRESHOLD

UNLOADED: Final[str] = "UNLOADED"
# Below ~1 N a pad's margin denominator is contact noise, not a grip.
DEFAULT_NORMAL_FLOOR_N: Final[float] = 1.0
# The shared friction law's stick/slip transition speed.
DEFAULT_SLIP_SPEED_M_S: Final[float] = SLIP_VELOCITY_THRESHOLD

PadMargin = float | str


def _require_positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {value}")


def pad_margins(
    wrench: PadWrench, friction: float, normal_floor_n: float
) -> tuple[PadMargin, ...]:
    """Per-pad normalised margin, :data:`UNLOADED` below ``normal_floor_n``.

    Raises:
        ValueError: on a non-positive or non-finite ``friction`` or floor, or
            non-finite pad forces.
    """
    _require_positive("friction", friction)
    _require_positive("normal_floor_n", normal_floor_n)
    normal, tangential = wrench.normal_force_n, wrench.tangential_force_n
    if not (np.all(np.isfinite(normal)) and np.all(np.isfinite(tangential))):
        raise ValueError("pad forces must be finite")
    margins: list[PadMargin] = []
    for f_n, f_t in zip(normal, tangential, strict=True):
        capacity = unilateral_friction_capacity(float(f_n), friction)
        loaded = f_n >= normal_floor_n
        margins.append((capacity - float(f_t)) / capacity if loaded else UNLOADED)
    return tuple(margins)


def min_loaded_margin(margins: Sequence[PadMargin]) -> float:
    """Smallest margin over loaded pads.

    Raises:
        ValueError: when every pad is :data:`UNLOADED`.
    """
    loaded = [float(m) for m in margins if m != UNLOADED]
    if not loaded:
        raise ValueError("no loaded pad: every pad is below the normal floor")
    return min(loaded)


def grasp_wrenches(lead: PadWrench, trail: PadWrench) -> tuple[np.ndarray, np.ndarray]:
    """``(net, internal)`` 6-vectors ``[force, moment]`` of the two hands."""
    w_lead = np.concatenate([lead.force_n, lead.moment_nm])
    w_trail = np.concatenate([trail.force_n, trail.moment_nm])
    return w_lead + w_trail, 0.5 * (w_lead - w_trail)


def slip_events(
    tangential_speed_m_s: np.ndarray, dt_s: float, threshold_m_s: float
) -> tuple[int, float]:
    """``(count, total duration)`` of frames where any pad slides faster than
    ``threshold_m_s``; overlapping pads form one event.

    Raises:
        ValueError: on a non-positive ``dt_s`` or threshold, or a non-finite
            or non-2-D speed array ``(frames, pads)``.
    """
    _require_positive("dt_s", dt_s)
    _require_positive("threshold_m_s", threshold_m_s)
    speed = np.asarray(tangential_speed_m_s, float)
    if speed.ndim != 2 or not np.all(np.isfinite(speed)):
        raise ValueError("tangential speed must be a finite (frames, pads) array")
    slipping = np.any(speed > threshold_m_s, axis=1)
    onsets = np.count_nonzero(slipping[1:] & ~slipping[:-1]) + int(slipping[:1].sum())
    return int(onsets), float(np.count_nonzero(slipping) * dt_s)


@dataclass(frozen=True)
class GripMarginReport:
    """Grip margin of one swing; per-pad margins and wrenches at
    ``peak_frame`` (peak club acceleration). ``mu_bias_bound`` is the
    engine's declared optimistic friction bias (PR 1 calibration), if any.
    """

    friction: float
    normal_floor_n: float
    pad_margin: dict[str, tuple[PadMargin, ...]]
    min_margin: float
    peak_frame: int
    slip_event_count: int
    slip_duration_s: float
    net_wrench: np.ndarray
    internal_wrench: np.ndarray
    mu_bias_bound: float | None

    def __post_init__(self) -> None:
        if self.mu_bias_bound is not None:
            _require_positive("mu_bias_bound", self.mu_bias_bound)
        scalars = [self.friction, self.normal_floor_n, self.min_margin]
        scalars.append(self.slip_duration_s)
        vectors = np.concatenate([self.net_wrench, self.internal_wrench])
        if not (all(map(math.isfinite, scalars)) and np.all(np.isfinite(vectors))):
            raise ValueError("grip margin report must be finite")


def grip_margin_report(
    wrenches: Mapping[str, Sequence[PadWrench]],
    club_acceleration_m_s2: np.ndarray,
    tangential_speed_m_s: np.ndarray,
    dt_s: float,
    friction: float,
    normal_floor_n: float = DEFAULT_NORMAL_FLOOR_N,
    slip_speed_m_s: float = DEFAULT_SLIP_SPEED_M_S,
    mu_bias_bound: float | None = None,
) -> GripMarginReport:
    """Report over a swing of per-frame lead/trail pad wrenches.

    Raises:
        ValueError: on wrenches not keyed by lead and trail, series of unequal
            frame counts, or any precondition of the metric functions.
    """
    if set(wrenches) != set(HAND_ROLES):
        raise ValueError("wrenches must be keyed by lead and trail")
    accel = np.asarray(club_acceleration_m_s2, float)
    frames = {len(accel), len(tangential_speed_m_s)}
    frames |= {len(series) for series in wrenches.values()}
    if len(frames) != 1:
        raise ValueError(f"series must share one number of frames, got {frames}")
    peak = int(np.argmax(np.linalg.norm(accel, axis=1)))
    at_peak = {hand: wrenches[hand][peak] for hand in HAND_ROLES}
    margins = {h: pad_margins(at_peak[h], friction, normal_floor_n) for h in at_peak}
    net, internal = grasp_wrenches(at_peak["lead"], at_peak["trail"])
    count, duration = slip_events(tangential_speed_m_s, dt_s, slip_speed_m_s)
    return GripMarginReport(
        friction,
        normal_floor_n,
        margins,
        min_loaded_margin([m for hand in HAND_ROLES for m in margins[hand]]),
        peak,
        count,
        duration,
        net,
        internal,
        mu_bias_bound,
    )
