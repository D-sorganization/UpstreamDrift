"""Zero Moment Point (ZMP) computation.

This module provides ZMP computation for bipedal balance analysis.
ZMP is the point on the ground where the net ground reaction torque
about the horizontal axes is zero.

Design by Contract:
    ZMP is only valid when robot is in contact with ground.
    Results indicate validity status.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.robotics.core.protocols import HumanoidCapable, RoboticsCapable


@dataclass
class ZMPResult:
    """Result of ZMP computation.

    Attributes:
        zmp_position: ZMP position in world frame (3,), z is ground height.
        cop_position: Center of Pressure position (3,), same as ZMP for flat ground.
        is_valid: Whether ZMP is within support polygon.
        support_margin: Distance from ZMP to nearest support boundary [m].
        total_normal_force: Total vertical ground reaction force [N].
        ground_height: Height of ground plane [m].
    """

    zmp_position: NDArray[np.float64]
    cop_position: NDArray[np.float64]
    is_valid: bool
    support_margin: float
    total_normal_force: float
    ground_height: float = 0.0


class ZMPComputer:
    """Computes Zero Moment Point for balance analysis.

    The ZMP is computed from the robot's CoM position, acceleration,
    and angular momentum rate. For a robot to be balanced, the ZMP
    must lie within the support polygon.

    Design by Contract:
        Preconditions:
            - Engine must provide CoM and centroidal dynamics
            - Ground contact must exist for valid ZMP

        Postconditions:
            - ZMPResult.is_valid indicates if ZMP is in support
            - ZMP position is on ground plane

    Example:
        >>> zmp_computer = ZMPComputer(engine)
        >>> result = zmp_computer.compute_zmp()
        >>> if result.is_valid:
        ...     print(f"ZMP at {result.zmp_position[:2]}")
    """

    GRAVITY = 9.81

    def __init__(
        self,
        engine: RoboticsCapable,
        ground_height: float = 0.0,
    ) -> None:
        """Initialize ZMP computer.

        Args:
            engine: Physics engine with CoM computation capabilities.
            ground_height: Height of ground plane [m].
        """
        self._engine = engine
        self._ground_height = ground_height
        self._is_humanoid = isinstance(engine, HumanoidCapable)

    @property
    def ground_height(self) -> float:
        """Get ground height."""
        return self._ground_height

    @ground_height.setter
    def ground_height(self, value: float) -> None:
        """Set ground height."""
        self._ground_height = value

    def compute_zmp(
        self,
        com_position: NDArray[np.float64] | None = None,
        com_velocity: NDArray[np.float64] | None = None,
        com_acceleration: NDArray[np.float64] | None = None,
        angular_momentum_rate: NDArray[np.float64] | None = None,
        support_polygon: NDArray[np.float64] | None = None,
    ) -> ZMPResult:
        """Compute Zero Moment Point.

        ZMP formula for flat ground:
        x_zmp = x_com - (z_com - z_ground) * (ddx_com + dL_y / m) / (ddz_com + g)
        y_zmp = y_com - (z_com - z_ground) * (ddy_com - dL_x / m) / (ddz_com + g)

        Args:
            com_position: CoM position (3,). Uses engine if None.
            com_velocity: CoM velocity (3,). Uses numerical diff if None.
            com_acceleration: CoM acceleration (3,). Estimated if None.
            angular_momentum_rate: Angular momentum rate (3,). Zero if None.
            support_polygon: Support polygon vertices (n, 2). Uses default if None.

        Returns:
            ZMPResult with ZMP position and validity.
        """
        # Get CoM state
        if com_position is None:
            com_position = self._get_com_position()
        if com_acceleration is None:
            com_acceleration = self._estimate_com_acceleration(com_velocity)
        if angular_momentum_rate is None:
            angular_momentum_rate = np.zeros(3)

        # Get robot mass
        mass = self._estimate_mass()

        # Height above ground
        z_rel = com_position[2] - self._ground_height

        # Denominator: vertical acceleration + gravity
        denom = com_acceleration[2] + self.GRAVITY

        # Avoid division by zero (free fall)
        if abs(denom) < 1e-6:
            # Robot is in free fall, ZMP is undefined
            return ZMPResult(
                zmp_position=np.array([0.0, 0.0, self._ground_height]),
                cop_position=np.array([0.0, 0.0, self._ground_height]),
                is_valid=False,
                support_margin=-1.0,
                total_normal_force=0.0,
                ground_height=self._ground_height,
            )

        # ZMP calculation
        # Include angular momentum contribution
        zmp_x = com_position[0] - z_rel * (
            com_acceleration[0] + angular_momentum_rate[1] / mass
        ) / denom
        zmp_y = com_position[1] - z_rel * (
            com_acceleration[1] - angular_momentum_rate[0] / mass
        ) / denom
        zmp_z = self._ground_height

        zmp = np.array([zmp_x, zmp_y, zmp_z])
        cop = zmp.copy()  # CoP == ZMP for flat ground

        # Compute total normal force
        total_force = mass * denom

        # Check if ZMP is in support polygon
        is_valid, margin = self._check_support(zmp[:2], support_polygon)

        return ZMPResult(
            zmp_position=zmp,
            cop_position=cop,
            is_valid=is_valid,
            support_margin=margin,
            total_normal_force=total_force,
            ground_height=self._ground_height,
        )

    def compute_capture_point(
        self,
        com_position: NDArray[np.float64] | None = None,
        com_velocity: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute Capture Point (Instantaneous Capture Point / DCM).

        The capture point is where the robot should step to come to a stop.
        ICP = CoM + CoM_velocity / omega
        where omega = sqrt(g / z_com)

        Args:
            com_position: CoM position (3,). Uses engine if None.
            com_velocity: CoM velocity (3,). Uses engine if None.

        Returns:
            Capture point position (3,).
        """
        if com_position is None:
            com_position = self._get_com_position()
        if com_velocity is None:
            com_velocity = self._get_com_velocity()

        z_rel = com_position[2] - self._ground_height
        if z_rel <= 0:
            z_rel = 0.01  # Avoid negative/zero height

        # Natural frequency of inverted pendulum
        omega = np.sqrt(self.GRAVITY / z_rel)

        # Capture point (only horizontal components meaningful)
        capture_point = np.zeros(3)
        capture_point[0] = com_position[0] + com_velocity[0] / omega
        capture_point[1] = com_position[1] + com_velocity[1] / omega
        capture_point[2] = self._ground_height

        return capture_point

    def compute_dcm(
        self,
        com_position: NDArray[np.float64] | None = None,
        com_velocity: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute Divergent Component of Motion.

        DCM is equivalent to capture point for LIPM.

        Args:
            com_position: CoM position (3,).
            com_velocity: CoM velocity (3,).

        Returns:
            DCM position (3,).
        """
        return self.compute_capture_point(com_position, com_velocity)

    def compute_stability_margin(
        self,
        zmp_position: NDArray[np.float64],
        support_polygon: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute stability margin (distance from ZMP to support boundary).

        Args:
            zmp_position: ZMP position (2,) or (3,).
            support_polygon: Support polygon vertices (n, 2).

        Returns:
            Stability margin [m]. Negative if outside support.
        """
        _, margin = self._check_support(zmp_position[:2], support_polygon)
        return margin

    def _get_com_position(self) -> NDArray[np.float64]:
        """Get CoM position from engine."""
        if self._is_humanoid:
            engine = self._engine
            if isinstance(engine, HumanoidCapable):
                return engine.get_com_position()

        # Fallback: use origin
        return np.array([0.0, 0.0, 1.0])

    def _get_com_velocity(self) -> NDArray[np.float64]:
        """Get CoM velocity from engine."""
        if self._is_humanoid:
            engine = self._engine
            if isinstance(engine, HumanoidCapable):
                return engine.get_com_velocity()

        return np.zeros(3)

    def _estimate_com_acceleration(
        self,
        com_velocity: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Estimate CoM acceleration.

        Simple estimation assuming quasi-static (zero acceleration).
        """
        # For now, assume quasi-static
        return np.zeros(3)

    def _estimate_mass(self) -> float:
        """Estimate total robot mass."""
        if self._is_humanoid:
            engine = self._engine
            if isinstance(engine, HumanoidCapable):
                return engine.get_total_mass()

        # Default mass
        return 70.0

    def _check_support(
        self,
        point: NDArray[np.float64],
        support_polygon: NDArray[np.float64] | None,
    ) -> tuple[bool, float]:
        """Check if point is in support polygon.

        Args:
            point: Point to check (2,).
            support_polygon: Polygon vertices (n, 2).

        Returns:
            Tuple of (is_inside, margin_to_boundary).
        """
        if support_polygon is None:
            # Default small support polygon
            support_polygon = np.array([
                [-0.1, -0.1],
                [0.1, -0.1],
                [0.1, 0.1],
                [-0.1, 0.1],
            ])

        if len(support_polygon) < 3:
            return False, -1.0

        # Check if inside polygon
        is_inside = self._point_in_polygon(point, support_polygon)

        # Compute margin (distance to boundary)
        margin = self._distance_to_polygon_boundary(point, support_polygon)
        if not is_inside:
            margin = -margin

        return is_inside, margin

    def _point_in_polygon(
        self,
        point: NDArray[np.float64],
        polygon: NDArray[np.float64],
    ) -> bool:
        """Check if point is inside polygon using ray casting."""
        n = len(polygon)
        inside = False
        j = n - 1

        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > point[1]) != (yj > point[1])) and (
                point[0] < (xj - xi) * (point[1] - yi) / (yj - yi + 1e-10) + xi
            ):
                inside = not inside
            j = i

        return inside

    def _distance_to_polygon_boundary(
        self,
        point: NDArray[np.float64],
        polygon: NDArray[np.float64],
    ) -> float:
        """Compute minimum distance from point to polygon boundary."""
        n = len(polygon)
        min_dist = float("inf")

        for i in range(n):
            j = (i + 1) % n
            dist = self._point_to_segment_distance(
                point, polygon[i], polygon[j]
            )
            min_dist = min(min_dist, dist)

        return min_dist

    def _point_to_segment_distance(
        self,
        point: NDArray[np.float64],
        seg_a: NDArray[np.float64],
        seg_b: NDArray[np.float64],
    ) -> float:
        """Compute distance from point to line segment."""
        v = seg_b - seg_a
        u = point - seg_a

        t = np.dot(u, v) / (np.dot(v, v) + 1e-10)
        t = max(0, min(1, t))

        closest = seg_a + t * v
        return float(np.linalg.norm(point - closest))
