"""Oracle snapshots for the reaction-restriction tests (NSX-26).

Three synthetic constructions, none of them physiological:

* :func:`identity_closure_snapshot` -- six actuated coordinates, ``M = I``,
  ``J_c = I``.  The balance is ``tau = b - lambda``, so under a box capacity the
  torque envelopes are known in closed form for every closure kind.
* :func:`golden_chain_snapshot` -- the 14-DOF chain of the exact
  ``restriction_counts`` golden (rational ``J_c``) with ``h = J_c^T lambda_0``.
* :func:`shaft_aligned` -- re-expresses a snapshot so the grip x axis is the
  bare-club axis (the spatial fixture's club lies along its grip z).
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np

from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    GripFrame,
    InputProvenance,
    LoopTopology,
    ModelProvenance,
    SpatialInertia,
)
from src.shared.python.loop_nullspace.fixtures import load_golden

CLUB = SpatialInertia(0.313, np.array([0.75, 0.0, 0.0]), np.diag([1e-4, 0.026, 0.026]))
PROVENANCE = (ModelProvenance("restriction_oracle", "1"), InputProvenance("synthetic", "nsx-26"))


def _snapshot(
    n: int,
    j_c: np.ndarray,
    h: np.ndarray,
    *,
    actuated: tuple[int, ...] | None = None,
) -> ClosedLoopSnapshot:
    act = tuple(range(n)) if actuated is None else actuated
    topo = LoopTopology(
        ClosureKind.WELD6,
        "trail",
        "lead",
        act,
        tuple(i for i in range(n) if i not in act),
        tuple(range(n)),
        tuple(f"j{i}" for i in range(n)),
    )
    return ClosedLoopSnapshot(
        q=np.zeros(n),
        v=np.zeros(n),
        a=np.zeros(n),
        M=np.eye(n),
        h=h,
        S=np.eye(n)[list(act)],
        J_c=j_c,
        Jdot_v=np.zeros(6),
        grip_frame=GripFrame(np.eye(3), np.zeros(3)),
        bare_club=CLUB,
        welded_hand_inertias=(),
        model_provenance=PROVENANCE[0],
        input_provenance=PROVENANCE[1],
        topology=topo,
    )


def identity_closure_snapshot(b: np.ndarray) -> ClosedLoopSnapshot:
    """``tau = b - lambda`` on six actuated coordinates (``J_c = I``)."""
    return _snapshot(6, np.eye(6), np.asarray(b, dtype=float))


def golden_chain_snapshot(case: str) -> tuple[ClosedLoopSnapshot, dict]:
    """Fixed-base 14-DOF chain with ``h = b`` of golden ``case``; also the case."""
    gold = load_golden("restriction_counts")
    j_c = np.array(
        [[float(Fraction(x)) for x in row] for row in gold.get("oracle", "J_c")]
    )
    data = gold.get("oracle", "cases", case)
    b = np.array([float(Fraction(x)) for x in data["b"]])
    return _snapshot(j_c.shape[1], j_c, b), data


def shaft_aligned(snap: ClosedLoopSnapshot) -> ClosedLoopSnapshot:
    """Same dynamics with the grip x axis along the bare-club axis."""
    axis = snap.bare_club.com / np.linalg.norm(snap.bare_club.com)
    helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
    v = np.cross(axis, helper)
    v /= np.linalg.norm(v)
    p = np.column_stack([axis, v, np.cross(axis, v)])
    blk = np.block([[p.T, np.zeros((3, 3))], [np.zeros((3, 3)), p.T]])
    club = snap.bare_club
    return replace(
        snap,
        J_c=blk @ snap.J_c,
        Jdot_v=blk @ snap.Jdot_v,
        grip_frame=GripFrame(snap.grip_frame.rotation @ p, snap.grip_frame.origin),
        bare_club=SpatialInertia(club.mass, p.T @ club.com, p.T @ club.inertia @ p),
        welded_hand_inertias=tuple(
            SpatialInertia(h.mass, p.T @ h.com, p.T @ h.inertia @ p)
            for h in snap.welded_hand_inertias
        ),
    )
