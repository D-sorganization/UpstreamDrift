"""Constraint handling and joint force computation for golfer model.

Implements closed kinematic loop constraints and methods to compute
forces and accelerations at joints.

The closure ``Phi(q) = 0`` has four rows:

* ``Phi[0:2] = LH_position - club_grip_left_position``;
* ``Phi[2:4]`` is the same residual in the club frame (perpendicular,
  along-club), i.e. an orthogonal rotation of ``Phi[0:2]``.

The club is anchored at the right wrist pin, so only two rows are
independent: the Jacobian has rank 2 on the constraint manifold and the
wrist-angle columns are zero (closure decision in ``golfer_kinematics``).

Canonical source (#12671, ADR-0056):
    The closure, its Jacobians, the joint linear accelerations, the net joint
    forces and the friction torques are implemented once, in the vendored
    Tools package ``double_pendulum_golf.golfer_constraints``; the functions
    below delegate to it.  ``INDEPENDENT_CLOSURE_ROWS`` stays here because it
    is solver policy for UpstreamDrift's least-squares KKT
    (``constraint_solver``); ``test_golfer_canonical_parity`` checks it against
    the Tools Jacobian on every vendor bump.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._canonical_golfer import delegate
from .physics_golfer import N_CONSTRAINTS, GolferParams

__all__ = [
    "INDEPENDENT_CLOSURE_ROWS",
    "N_CONSTRAINTS",
    "analytical_constraint_jacobian",
    "constraint_vector",
    "friction_torque_vector",
    "linear_accelerations",
    "net_joint_forces",
    "numerical_constraint_jacobian",
]

# Rows of ``constraint_vector`` that form an independent closure (#12608): the
# left-wrist pin residual in the world frame.  Rows 2-3 are the same residual
# rotated into the club frame, so they add no rank on the manifold, and off it
# they differ from rows 0-1 only by a term proportional to ``Phi``.  Solvers
# that invert the closure must use these rows only.
INDEPENDENT_CLOSURE_ROWS: tuple[int, ...] = (0, 1)

_ClosureFn = Callable[[np.ndarray, GolferParams], np.ndarray]
_ChainFn = Callable[[np.ndarray, np.ndarray, np.ndarray, GolferParams], dict[str, Any]]

#: ``Phi(q)``, shape ``(4,)``; zero when the loop is closed.
constraint_vector: _ClosureFn = delegate("golfer_constraints", "constraint_vector")

#: ``dPhi/dq`` by central differences, shape ``(4, 8)``.
numerical_constraint_jacobian: _ClosureFn = delegate(
    "golfer_constraints", "numerical_constraint_jacobian"
)

#: ``dPhi/dq`` in closed form, shape ``(4, 8)``.
analytical_constraint_jacobian: _ClosureFn = delegate(
    "golfer_constraints", "analytical_constraint_jacobian"
)

#: Linear acceleration ``J qddot + Jdot qdot`` of each joint point, keyed by name.
linear_accelerations: _ChainFn = delegate("golfer_constraints", "linear_accelerations")

#: Net force ``m a - m g`` at each point mass, keyed by joint name.
net_joint_forces: _ChainFn = delegate("golfer_constraints", "net_joint_forces")

#: Viscous joint friction torques ``-b * qdot``, shape ``(8,)``.
friction_torque_vector: _ClosureFn = delegate(
    "golfer_constraints", "friction_torque_vector"
)
