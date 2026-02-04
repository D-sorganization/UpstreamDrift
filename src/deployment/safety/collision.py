"""Collision avoidance for safe robot operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.deployment.realtime import RobotState
    from src.engines.protocols import PhysicsEngineProtocol


class ObstacleType(Enum):
    """Type of obstacle."""

    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    HUMAN = "human"
    DYNAMIC = "dynamic"


@dataclass
class Obstacle:
    """Obstacle for collision avoidance.

    Attributes:
        name: Obstacle identifier.
        obstacle_type: Type of obstacle geometry.
        position: Center position [x, y, z].
        dimensions: Size dimensions (depends on type).
        velocity: Obstacle velocity (for dynamic obstacles).
        inflation: Safety inflation radius.
    """

    name: str
    obstacle_type: ObstacleType
    position: NDArray[np.floating]
    dimensions: NDArray[np.floating]
    velocity: NDArray[np.floating] | None = None
    inflation: float = 0.05

    def get_distance(self, point: NDArray[np.floating]) -> float:
        """Compute signed distance from point to obstacle.

        Args:
            point: Query point [x, y, z].

        Returns:
            Signed distance (negative inside obstacle).
        """
        if self.obstacle_type == ObstacleType.SPHERE:
            return float(
                np.linalg.norm(point - self.position)
                - self.dimensions[0]
                - self.inflation
            )

        elif self.obstacle_type in (ObstacleType.BOX, ObstacleType.HUMAN):
            # Box distance
            half_dims = self.dimensions / 2
            local_point = point - self.position
            clamped = np.clip(local_point, -half_dims, half_dims)
            return float(np.linalg.norm(local_point - clamped) - self.inflation)

        elif self.obstacle_type == ObstacleType.CYLINDER:
            # Cylinder distance (axis along z)
            radius = self.dimensions[0]
            height = self.dimensions[1]
            local_point = point - self.position

            # Distance to infinite cylinder
            xy_dist = np.sqrt(local_point[0] ** 2 + local_point[1] ** 2) - radius
            z_dist = np.abs(local_point[2]) - height / 2

            if z_dist > 0 and xy_dist > 0:
                return float(np.sqrt(xy_dist**2 + z_dist**2) - self.inflation)
            return float(max(xy_dist, z_dist) - self.inflation)

        return 0.0

    def get_gradient(self, point: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute gradient of distance function.

        Args:
            point: Query point [x, y, z].

        Returns:
            Gradient vector (points away from obstacle).
        """
        eps = 1e-6
        gradient = np.zeros(3)

        for i in range(3):
            point_plus = point.copy()
            point_plus[i] += eps
            point_minus = point.copy()
            point_minus[i] -= eps

            gradient[i] = (
                self.get_distance(point_plus) - self.get_distance(point_minus)
            ) / (2 * eps)

        norm = np.linalg.norm(gradient)
        if norm > eps:
            gradient /= norm

        return gradient


@dataclass
class HumanState:
    """Detected human state.

    Attributes:
        position: Human center position [x, y, z].
        velocity: Human velocity [vx, vy, vz].
        bounding_box: Human bounding box dimensions.
        skeleton_joints: Joint positions if tracked.
        confidence: Detection confidence (0-1).
        is_moving: Whether human is in motion.
    """

    position: NDArray[np.floating]
    velocity: NDArray[np.floating] = field(
        default_factory=lambda: np.zeros(3)
    )
    bounding_box: NDArray[np.floating] = field(
        default_factory=lambda: np.array([0.5, 0.3, 1.7])
    )
    skeleton_joints: dict[str, NDArray[np.floating]] | None = None
    confidence: float = 1.0
    is_moving: bool = False

    def to_obstacle(self) -> Obstacle:
        """Convert to obstacle for collision checking.

        Returns:
            Obstacle representing the human.
        """
        return Obstacle(
            name="human",
            obstacle_type=ObstacleType.HUMAN,
            position=self.position,
            dimensions=self.bounding_box,
            velocity=self.velocity,
            inflation=0.3,  # Extra safety margin for humans
        )


class CollisionAvoidance:
    """Real-time collision avoidance using potential fields.

    Implements artificial potential field approach for
    reactive collision avoidance during robot motion.

    Attributes:
        robot_model: Physics engine for forward kinematics.
        safety_distance: Minimum safety distance in meters.
        obstacles: List of obstacles to avoid.
    """

    def __init__(
        self,
        robot_model: "PhysicsEngineProtocol",
        safety_distance: float = 0.1,
    ) -> None:
        """Initialize collision avoidance.

        Args:
            robot_model: Physics engine for kinematics.
            safety_distance: Minimum clearance in meters.
        """
        self.model = robot_model
        self.safety_distance = safety_distance
        self._obstacles: list[Obstacle] = []
        self._human_state: HumanState | None = None

        # Potential field parameters
        self._repulsion_gain = 1.0
        self._repulsion_distance = 0.5  # Distance at which repulsion starts
        self._max_repulsion = 10.0

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add obstacle to collision checking.

        Args:
            obstacle: Obstacle to add.
        """
        self._obstacles.append(obstacle)

    def remove_obstacle(self, name: str) -> bool:
        """Remove obstacle by name.

        Args:
            name: Obstacle name.

        Returns:
            True if obstacle was found and removed.
        """
        for i, obs in enumerate(self._obstacles):
            if obs.name == name:
                del self._obstacles[i]
                return True
        return False

    def clear_obstacles(self) -> None:
        """Remove all obstacles."""
        self._obstacles.clear()

    def update_human_position(self, human_state: HumanState) -> None:
        """Update detected human position.

        Args:
            human_state: Current human state.
        """
        self._human_state = human_state

    def get_link_positions(
        self,
        state: "RobotState",
    ) -> dict[str, NDArray[np.floating]]:
        """Get positions of robot links.

        Args:
            state: Current robot state.

        Returns:
            Dictionary mapping link names to positions.
        """
        positions = {}

        # Set robot state
        if hasattr(self.model, "set_joint_positions"):
            self.model.set_joint_positions(state.joint_positions)

        # Get link positions
        if hasattr(self.model, "get_link_positions"):
            return self.model.get_link_positions()

        # Fallback: simple kinematic chain
        n_links = len(state.joint_positions)
        for i in range(n_links):
            positions[f"link_{i}"] = np.array([0, 0, 0.1 * (i + 1)])

        return positions

    def compute_repulsive_field(
        self,
        state: "RobotState",
    ) -> NDArray[np.floating]:
        """Compute artificial potential field repulsion.

        Computes repulsive forces from obstacles that can be
        mapped to joint torques for reactive avoidance.

        Args:
            state: Current robot state.

        Returns:
            Repulsive force in joint space (n_joints,).
        """
        n_joints = len(state.joint_positions)
        repulsion = np.zeros(n_joints)

        # Get all obstacles including human
        obstacles = self._obstacles.copy()
        if self._human_state is not None:
            obstacles.append(self._human_state.to_obstacle())

        if not obstacles:
            return repulsion

        # Get link positions
        link_positions = self.get_link_positions(state)

        # Compute repulsion for each link
        for link_name, link_pos in link_positions.items():
            for obstacle in obstacles:
                distance = obstacle.get_distance(link_pos)

                if distance < self._repulsion_distance:
                    # Compute repulsion magnitude
                    if distance <= 0:
                        # Inside obstacle - maximum repulsion
                        magnitude = self._max_repulsion
                    else:
                        # Inverse square repulsion
                        magnitude = self._repulsion_gain * (
                            1 / distance - 1 / self._repulsion_distance
                        ) / (distance**2)
                        magnitude = min(magnitude, self._max_repulsion)

                    # Get repulsion direction
                    gradient = obstacle.get_gradient(link_pos)

                    # Map to joint space (simplified - use Jacobian for full)
                    # For now, distribute to all joints
                    repulsion += gradient[:n_joints] if n_joints <= 3 else (
                        np.concatenate([gradient, np.zeros(n_joints - 3)])
                    ) * magnitude / len(link_positions)

        return repulsion

    def check_path_clearance(
        self,
        trajectory: NDArray[np.floating],
        min_distance: float | None = None,
    ) -> tuple[bool, float]:
        """Check if trajectory maintains clearance from obstacles.

        Args:
            trajectory: Joint trajectory (n_steps, n_joints).
            min_distance: Minimum required clearance.

        Returns:
            Tuple of (is_clear, minimum_distance_found).
        """
        if min_distance is None:
            min_distance = self.safety_distance

        min_found = float("inf")
        n_steps = len(trajectory)

        # Check each waypoint
        for i in range(n_steps):
            q = trajectory[i]

            # Set state and get link positions
            if hasattr(self.model, "set_joint_positions"):
                self.model.set_joint_positions(q)

            link_positions = self.get_link_positions(
                # Create minimal state
                type("State", (), {"joint_positions": q})()  # noqa
            )

            # Check distance to all obstacles
            obstacles = self._obstacles.copy()
            if self._human_state is not None:
                obstacles.append(self._human_state.to_obstacle())

            for link_pos in link_positions.values():
                for obstacle in obstacles:
                    distance = obstacle.get_distance(link_pos)
                    min_found = min(min_found, distance)

                    if distance < min_distance:
                        return False, min_found

        return True, min_found

    def get_safe_velocity_scaling(
        self,
        state: "RobotState",
    ) -> float:
        """Compute safe velocity scaling based on obstacles.

        Returns scaling factor to reduce velocity when
        close to obstacles.

        Args:
            state: Current robot state.

        Returns:
            Velocity scaling factor (0-1).
        """
        link_positions = self.get_link_positions(state)

        # Get all obstacles including human
        obstacles = self._obstacles.copy()
        if self._human_state is not None:
            obstacles.append(self._human_state.to_obstacle())

        if not obstacles:
            return 1.0

        min_distance = float("inf")

        for link_pos in link_positions.values():
            for obstacle in obstacles:
                distance = obstacle.get_distance(link_pos)
                min_distance = min(min_distance, distance)

        # Scale velocity based on distance
        if min_distance <= 0:
            return 0.0
        elif min_distance < self.safety_distance:
            return min_distance / self.safety_distance
        elif min_distance < self._repulsion_distance:
            return 0.5 + 0.5 * (
                (min_distance - self.safety_distance)
                / (self._repulsion_distance - self.safety_distance)
            )
        else:
            return 1.0

    def get_minimum_distance(self, state: "RobotState") -> float:
        """Get minimum distance to any obstacle.

        Args:
            state: Current robot state.

        Returns:
            Minimum distance in meters.
        """
        link_positions = self.get_link_positions(state)

        obstacles = self._obstacles.copy()
        if self._human_state is not None:
            obstacles.append(self._human_state.to_obstacle())

        if not obstacles:
            return float("inf")

        min_distance = float("inf")

        for link_pos in link_positions.values():
            for obstacle in obstacles:
                distance = obstacle.get_distance(link_pos)
                min_distance = min(min_distance, distance)

        return min_distance
