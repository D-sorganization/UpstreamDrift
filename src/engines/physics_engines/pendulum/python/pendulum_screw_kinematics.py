"""Screw-theoretic kinematics adapter for the Double Pendulum engine.

Implements Guideline C3 (ISA/twist extraction) for the planar 2-DOF double
pendulum model that represents the golf swing (arm + club-shaft).

The pendulum lies in the swing plane (XZ notation: x forward, y up, z lateral).
For the planar kinematic chain the screw axes are always about the out-of-plane
axis (z-axis in world coordinates), so angular velocities are [0, 0, ω].

Reference: docs/assessments/project_design_guidelines.qmd Section C3
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.screw_theory import (
    ScrewAxis,
    Twist,
    compute_screw_axis,
    compute_screw_endpoints,
)

if TYPE_CHECKING:
    from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
        DoublePendulumDynamics,
    )

logger = get_logger(__name__)


class PendulumScrewKinematics:
    """Screw-theoretic kinematics for the double-pendulum golf swing model.

    Computes instantaneous screw axes for the two key task points:
    - ``arm_tip``: distal end of the upper arm segment (wrist joint location)
    - ``clubhead``: tip of the lower segment (club-shaft + head)

    Both joints rotate about the out-of-plane (z) axis, so all screw axes
    have ``axis_direction`` aligned with ±ẑ in the non-singular case.

    Example::

        engine = PendulumPhysicsEngine()
        sk = PendulumScrewKinematics(engine.dynamics)
        q, v = engine.get_state()
        results = sk.analyze_key_points(q, v)
        twist_arm, screw_arm = results["arm_tip"]
        logger.info("Screw pitch (arm tip): %.4f m/rad", screw_arm.pitch)
    """

    BODY_ARM_TIP = "arm_tip"
    BODY_CLUBHEAD = "clubhead"

    def __init__(self, dynamics: DoublePendulumDynamics) -> None:
        """Initialise with a ``DoublePendulumDynamics`` instance.

        Args:
            dynamics: A ``DoublePendulumDynamics`` object whose ``_l1``
                      attribute (arm length) and ``parameters.lower_segment``
                      (shaft/club parameters) are accessible.
        """
        assert dynamics is not None, "dynamics must be provided"
        self.dynamics = dynamics
        self._l1: float = float(dynamics._l1)
        self._l2: float = float(dynamics.parameters.lower_segment.length_m)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_arm_tip_position(self, theta1: float) -> np.ndarray:
        """Position of the arm tip (wrist) in world coordinates.

        Args:
            theta1: Shoulder angle [rad] measured from downward vertical.

        Returns:
            Position vector [x, y, z] [m].
        """
        x = self._l1 * math.sin(theta1)
        y = -self._l1 * math.cos(theta1)
        return np.array([x, y, 0.0])

    def compute_clubhead_position(self, theta1: float, theta2: float) -> np.ndarray:
        """Position of the clubhead tip in world coordinates.

        Args:
            theta1: Shoulder angle [rad].
            theta2: Wrist (relative) angle [rad].

        Returns:
            Position vector [x, y, z] [m].
        """
        p1 = self.compute_arm_tip_position(theta1)
        angle12 = theta1 + theta2
        x2 = self._l2 * math.sin(angle12)
        y2 = -self._l2 * math.cos(angle12)
        return p1 + np.array([x2, y2, 0.0])

    def compute_twist(
        self,
        q: np.ndarray,
        v: np.ndarray,
        body_name: str,
    ) -> Twist:
        """Compute spatial twist for a named body.

        Args:
            q: Joint positions [theta1, theta2] [rad].
            v: Joint velocities [omega1, omega2] [rad/s].
            body_name: One of ``"arm_tip"`` or ``"clubhead"``.

        Returns:
            ``Twist`` with angular and linear components and reference point.

        Raises:
            ValueError: If ``body_name`` is not recognised.
        """
        assert q is not None, "q must be provided"
        assert v is not None, "v must be provided"
        theta1, theta2 = float(q[0]), float(q[1])
        omega1, omega2 = float(v[0]), float(v[1])

        if body_name == self.BODY_ARM_TIP:
            ref = self.compute_arm_tip_position(theta1)
            omega_total = omega1
        elif body_name == self.BODY_CLUBHEAD:
            ref = self.compute_clubhead_position(theta1, theta2)
            omega_total = omega1 + omega2
        else:
            raise ValueError(
                f"Unknown body '{body_name}'. "
                f"Choose '{self.BODY_ARM_TIP}' or '{self.BODY_CLUBHEAD}'."
            )

        # Out-of-plane angular velocity (z-axis rotation in swing plane)
        angular = np.array([0.0, 0.0, omega_total])

        # Linear velocity at reference point: v = ω × r
        # [0, 0, ω] × [x, y, 0] = [-ω*y, ω*x, 0]
        linear = np.array(
            [
                -omega_total * ref[1],
                omega_total * ref[0],
                0.0,
            ]
        )

        return Twist(
            angular=angular,
            linear=linear,
            body_name=body_name,
            reference_point=ref,
        )

    def compute_screw_axis(
        self,
        twist: Twist,
        singularity_threshold: float = 1e-6,
    ) -> ScrewAxis:
        """Compute ISA from a twist. Delegates to shared module."""
        return compute_screw_axis(twist, singularity_threshold)

    def analyze_key_points(
        self,
        q: np.ndarray,
        v: np.ndarray,
    ) -> dict[str, tuple[Twist, ScrewAxis]]:
        """Compute twist and screw axis for both key task points.

        Args:
            q: Joint positions [theta1, theta2] [rad].
            v: Joint velocities [omega1, omega2] [rad/s].

        Returns:
            Dict mapping body name → (Twist, ScrewAxis).
        """
        assert q is not None, "q must be provided"
        results: dict[str, tuple[Twist, ScrewAxis]] = {}
        for name in (self.BODY_ARM_TIP, self.BODY_CLUBHEAD):
            twist = self.compute_twist(q, v, name)
            screw = self.compute_screw_axis(twist)
            results[name] = (twist, screw)
        return results

    def visualize_screw_axis(
        self,
        screw: ScrewAxis,
        length: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate line segment for screw axis visualization.

        Delegates to the shared ``compute_screw_endpoints`` utility.

        Args:
            screw: Screw axis to visualize.
            length: Length of axis segment [m].

        Returns:
            Tuple of (start_point, end_point) [3], [3].
        """
        return compute_screw_endpoints(screw, length)
