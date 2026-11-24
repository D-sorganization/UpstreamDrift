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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
    drift_and_control_reactions,
)
from src.shared.python.simulation_backends.protocol import (
    ConstrainedEquationsOfMotionProvider,
)

Array = NDArray[np.float64]
FrameAt = Callable[[Array], tuple[Array, Array]]

__all__ = ["HandForceSplit", "hand_force_split"]


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
