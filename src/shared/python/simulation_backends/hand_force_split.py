"""World-frame drift/control split of the hand-on-club force (#15685, K5).

At each state of a tracked swing, the closure reaction between the closing
hand and the club splits pointwise into a drift part (``tau = 0``: gravity
and velocity terms, the ZTCF) and a control part (the applied torques alone,
the ZVCF) with :func:`drift_and_control_reactions`. This module expresses the
linear part of both in the world frame as the force the hand applies to the
club, at the grip-frame origin, ready to draw tip to tail.

Canonical closure convention (``loop_nullspace/closure_convention.py``): rows
are ``[linear; angular]`` in the grip frame, and ``+J_c^T lambda`` is the load
on the tree for the relative twist ``V_hand - V_club``; so ``lambda`` acts on
the hand and ``-lambda`` on the club. Pointwise only: reactions along
integrated counterfactual trajectories do not sum.

The split covers what the closed-loop provider models (the tree and the
closure); a provider without ground contact puts the stance support into
neither part. It is one solution in the null space of the two-hand loop, set
by the torques that produced the motion.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
    drift_and_control_reactions,
    drift_and_control_states,
)
from src.shared.python.simulation_backends.protocol import (
    ConstrainedEquationsOfMotionProvider,
)

Array = NDArray[np.float64]
FrameAt = Callable[[Array], tuple[Array, Array]]
#: ``club_load(q, v, a, gravity) -> (force, moment about the world origin)``:
#: the net wrench the hands must supply to the bare club (and, on a flexible
#: shaft, its links and head), in the provider's coordinate order.
ClubLoad = Callable[[Array, Array, Array, bool], tuple[Array, Array]]
PointAt = Callable[[Array], Array]

#: Hands, in display order. The trail hand closes the loop (the weld).
HANDS = ("lead", "trail")
#: The hand-club weld rows lead the multipliers; planted-ground rows
#: (``GroundedProvider``) follow them.
CLOSURE_ROWS = 6
#: ZTCF: drift (zero torque; gravity, momentum, passive). ZVCF: control
#: (the applied torques alone). Owner decision #14497; they sum to the total.
PARTS = ("ztcf", "zvcf")

__all__ = [
    "HANDS",
    "PARTS",
    "HandForceSplit",
    "HandWrench",
    "HandWrenchSplit",
    "hand_force_split",
    "hand_wrench_split",
]


@dataclass(frozen=True)
class HandForceSplit:
    """Per-state hand-on-club force, world frame, newtons."""

    point_m: Array
    drift_n: Array
    control_n: Array

    @property
    def total_n(self) -> Array:
        """Drift plus control (the reaction under the applied torques)."""
        return self.drift_n + self.control_n


def hand_force_split(
    provider: ConstrainedEquationsOfMotionProvider,
    frame_at: FrameAt,
    q: Array,
    v: Array,
    tau: Array,
) -> HandForceSplit:
    """Split the hand-on-club force at every row of ``q``, ``v``, ``tau``.

    ``frame_at(q_row)`` returns the canonical grip frame ``(R world<-grip,
    origin)`` in the provider's coordinate order.

    Preconditions: ``q``, ``v`` and ``tau`` have the same number of rows; the
    closure has six rows (a weld). ``ValueError`` otherwise.
    Postcondition: ``drift_n + control_n`` is the reaction under ``tau``.
    """
    q, v, tau = (np.atleast_2d(np.asarray(x, dtype=float)) for x in (q, v, tau))
    if not q.shape[0] == v.shape[0] == tau.shape[0]:
        raise ValueError("q, v and tau must have the same number of rows")
    rows = q.shape[0]
    point, drift, control = (np.zeros((rows, 3)) for _ in range(3))
    for k in range(rows):
        lam_d, lam_c = drift_and_control_reactions(provider, q[k], v[k], tau[k])
        if lam_d.shape != (6,):
            raise ValueError("the closure must have six rows (a weld)")
        rot, origin = frame_at(q[k])
        rot = np.asarray(rot, dtype=float)
        point[k] = np.asarray(origin, dtype=float)
        drift[k] = -rot @ lam_d[:3]
        control[k] = -rot @ lam_c[:3]
    return HandForceSplit(point_m=point, drift_n=drift, control_n=control)


@dataclass(frozen=True)
class HandWrench:
    """One hand's load on the club over the rows: force (N) at ``point_m`` and
    the free torque (N m) about that point, world frame."""

    point_m: Array
    force_n: Array
    torque_nm: Array


@dataclass(frozen=True)
class HandWrenchSplit:
    """Both hands' loads on the club, by part (#15150 Track Z, Z2)."""

    parts: Mapping[str, Mapping[str, HandWrench]]

    def part(self, hand: str, part: str) -> HandWrench:
        """``hand`` in :data:`HANDS`, ``part`` in :data:`PARTS`; else ``ValueError``."""
        if hand not in HANDS:
            raise ValueError(f"hand must be one of {HANDS}, got {hand!r}")
        if part not in PARTS:
            raise ValueError(f"part must be one of {PARTS}, got {part!r}")
        return self.parts[hand][part]

    def total(self, hand: str) -> HandWrench:
        """ZTCF plus ZVCF of ``hand``: its load under the applied torques."""
        drift, control = (self.part(hand, p) for p in PARTS)
        return HandWrench(
            drift.point_m,
            drift.force_n + control.force_n,
            drift.torque_nm + control.torque_nm,
        )


def _hand_loads(
    lam: Array,
    rot: Array,
    origin: Array,
    net: tuple[Array, Array],
    lead_point: Array,
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    """``((lead force, torque), (trail force, torque))`` of one part, one row."""
    trail_f, trail_t = -rot @ lam[:3], -rot @ lam[3:6]
    net_f, net_m = (np.asarray(x, dtype=float) for x in net)
    lead_f = net_f - trail_f
    lead_t = net_m - trail_t - np.cross(origin, trail_f) - np.cross(lead_point, lead_f)
    return (lead_f, lead_t), (trail_f, trail_t)


def hand_wrench_split(
    provider: ConstrainedEquationsOfMotionProvider,
    frame_at: FrameAt,
    club_load: ClubLoad,
    lead_point: PointAt,
    q: Array,
    v: Array,
    tau: Array,
) -> HandWrenchSplit:
    """Lead- and trail-hand loads on the club, ZTCF and ZVCF, at every row.

    The trail hand acts through the closure weld (``-lambda`` of each part,
    at the grip origin). The lead hand is the rest of what the club needs:
    ``club_load`` of the part minus the trail load, with its torque about
    ``lead_point(q)``. The ZTCF part is evaluated at ``(q, v, a_ZTCF)`` with
    gravity, the ZVCF part at ``(q, 0, a_ZVCF)`` without, so each hand's parts
    add to its total and gravity is counted once.

    Preconditions: as :func:`hand_force_split`. Postcondition: for each part,
    lead plus trail equals ``club_load`` (force, and moment about any point).
    """
    q, v, tau = (np.atleast_2d(np.asarray(x, dtype=float)) for x in (q, v, tau))
    if not q.shape[0] == v.shape[0] == tau.shape[0]:
        raise ValueError("q, v and tau must have the same number of rows")
    rows = q.shape[0]
    out = {h: {p: np.zeros((3, rows, 3)) for p in PARTS} for h in HANDS}
    for k in range(rows):
        states = drift_and_control_states(provider, q[k], v[k], tau[k])
        rot, origin = (np.asarray(x, dtype=float) for x in frame_at(q[k]))
        point = np.asarray(lead_point(q[k]), dtype=float)
        for part, (a, lam), rates, gravity in zip(
            PARTS, states, (v[k], np.zeros_like(v[k])), (True, False), strict=True
        ):
            if lam.ndim != 1 or lam.shape[0] < CLOSURE_ROWS:
                raise ValueError("the closure must have six rows (a weld)")
            net = club_load(q[k], rates, a, gravity)
            loads = _hand_loads(lam[:CLOSURE_ROWS], rot, origin, net, point)
            for hand, (force, torque), at in zip(
                HANDS, loads, (point, origin), strict=True
            ):
                out[hand][part][:, k] = (at, force, torque)
    return HandWrenchSplit(
        MappingProxyType(
            {
                h: MappingProxyType({p: HandWrench(*out[h][p]) for p in PARTS})
                for h in HANDS
            }
        )
    )
