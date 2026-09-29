# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Balance utilities for the Lagrangian planar-chain dynamics model.

Provides ``balance_pose`` (adjust one joint to land COM at inner BOS center)
and ``_standing_balanced`` (find a near-standing balanced pose).

These helpers are separated from ``LagrangianDynamics`` so that the core
physics class stays focused on dynamics equations while balance-search logic
lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..constants import JOINT_LIMITS, STANDING_DEG

if TYPE_CHECKING:
    pass


class _DynamicsWithBody(Protocol):
    """Structural protocol for the dynamics object used in balance helpers.

    Any object providing ``body.inner_center`` and a ``com_position`` method
    satisfies this protocol at type-check time.
    """

    @property
    def body(self) -> Any:
        """Anthropometric body model (must have ``inner_center`` attribute)."""
        ...

    def com_position(
        self,
        q: NDArray,
        exercise_type: str,
        bar_mass: float,
    ) -> NDArray:
        """Return (x, y) whole-body centre of mass."""
        ...


def balance_pose(
    dyn: _DynamicsWithBody,
    q_init: NDArray,
    exercise_type: str,
    bar_mass: float,
    adjust_joint: int = 2,
) -> NDArray:
    """Adjust one joint angle so the COM lands at the inner BOS center.

    Solves for the angle of ``adjust_joint`` (default: torso) that places
    the whole-body COM_x at ``body.inner_center``.  Uses bisection on the
    COM_x function — guaranteed to converge within joint bounds.

    Preconditions:
        adjust_joint in {0, 1, 2}
        q_init is length-3

    Complexity:
        O(K + B) COM evaluations for fixed 3-link kinematics, where ``K`` is
        the bracket scan count (currently 20) and ``B`` is the Brent iteration
        count.  Each COM evaluation is O(1).
    """
    from scipy.optimize import brentq

    body = dyn.body
    target_x = body.inner_center  # type: ignore[union-attr]  # body is typed as object in Protocol; inner_center is guaranteed by LagrangianDynamics
    # Use actual joint limits from JOINT_LIMITS for the bracket bounds
    # instead of hardcoded values.  For non-monotonic residuals (e.g. hip
    # in a deep squat), scan the bracket to find the first sign change so
    # brentq converges to the nearest root.
    joint_names = ("ankle", "knee", "hip")
    lo, hi = JOINT_LIMITS[joint_names[adjust_joint]]

    def residual(angle: float) -> float:
        q = q_init.copy()
        q[adjust_joint] = angle
        return dyn.com_position(q, exercise_type, bar_mass)[0] - target_x

    # Scan for the first sign change within the bracket.  The residual
    # may be non-monotonic (e.g. hip COM_x peaks mid-range then falls),
    # so a full-bracket brentq can miss roots.  We subdivide into steps
    # and use the first sub-interval that contains a sign change, which
    # yields the smallest-magnitude solution closest to the lower limit.
    n_scan = 20
    angles = np.linspace(lo, hi, n_scan + 1)
    f_vals = np.array([residual(a) for a in angles])
    bracket_lo, bracket_hi = lo, hi
    found_bracket = False
    for k in range(n_scan):
        if f_vals[k] * f_vals[k + 1] <= 0:
            bracket_lo, bracket_hi = angles[k], angles[k + 1]
            found_bracket = True
            break

    if not found_bracket:
        # No root in the joint range -- pick the angle with smallest residual
        best_idx = int(np.argmin(np.abs(f_vals)))
        q = q_init.copy()
        q[adjust_joint] = angles[best_idx]
        return q

    angle_opt = brentq(residual, bracket_lo, bracket_hi, xtol=1e-6)
    q = q_init.copy()
    q[adjust_joint] = angle_opt
    return q


def _standing_balanced(dyn: _DynamicsWithBody, bar_mass: float, exercise_type: str) -> NDArray:
    """Find a near-standing pose with COM at inner BOS center.

    Adjusts shin angle (joint 0) to shift COM forward over mid-foot.

    Complexity:
        Same as :func:`balance_pose`: O(K + B) fixed-chain COM evaluations.
    """
    q_stand = np.array([np.radians(a) for a in STANDING_DEG])
    return balance_pose(dyn, q_stand, exercise_type, bar_mass, adjust_joint=0)
