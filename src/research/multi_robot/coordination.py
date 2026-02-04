"""Multi-robot coordination algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.engines.protocols import PhysicsEngineProtocol


@dataclass
class FormationConfig:
    """Configuration for robot formation.

    Attributes:
        name: Formation name.
        positions: Relative positions for each robot (N, 3).
        orientations: Relative orientations for each robot (N, 4).
        reference_frame: Reference frame type ("leader" or "centroid").
    """

    name: str
    positions: NDArray[np.floating]
    orientations: NDArray[np.floating] | None = None
    reference_frame: str = "leader"

    @classmethod
    def line_formation(
        cls,
        n_robots: int,
        spacing: float = 1.0,
    ) -> "FormationConfig":
        """Create a line formation.

        Args:
            n_robots: Number of robots.
            spacing: Distance between robots.

        Returns:
            Line formation config.
        """
        positions = np.zeros((n_robots, 3))
        for i in range(n_robots):
            positions[i, 1] = i * spacing  # Along y-axis

        return cls(name="line", positions=positions)

    @classmethod
    def circle_formation(
        cls,
        n_robots: int,
        radius: float = 2.0,
    ) -> "FormationConfig":
        """Create a circular formation.

        Args:
            n_robots: Number of robots.
            radius: Circle radius.

        Returns:
            Circle formation config.
        """
        positions = np.zeros((n_robots, 3))
        for i in range(n_robots):
            angle = 2 * np.pi * i / n_robots
            positions[i, 0] = radius * np.cos(angle)
            positions[i, 1] = radius * np.sin(angle)

        return cls(name="circle", positions=positions)

    @classmethod
    def wedge_formation(
        cls,
        n_robots: int,
        spacing: float = 1.5,
        angle: float = 0.5,  # radians
    ) -> "FormationConfig":
        """Create a wedge/V formation.

        Args:
            n_robots: Number of robots.
            spacing: Distance between robots.
            angle: Half-angle of the wedge.

        Returns:
            Wedge formation config.
        """
        positions = np.zeros((n_robots, 3))
        positions[0] = [0, 0, 0]  # Leader at front

        for i in range(1, n_robots):
            row = (i + 1) // 2
            side = 1 if i % 2 == 1 else -1

            positions[i, 0] = -row * spacing * np.cos(angle)
            positions[i, 1] = side * row * spacing * np.sin(angle)

        return cls(name="wedge", positions=positions)


class FormationController:
    """Control robot formations.

    Computes control inputs to maintain formation geometry
    as robots move through the environment.

    Attributes:
        robots: List of robot IDs in formation.
        formation: Current formation configuration.
    """

    def __init__(
        self,
        robots: list[str],
        formation: FormationConfig,
    ) -> None:
        """Initialize formation controller.

        Args:
            robots: List of robot IDs.
            formation: Formation configuration.
        """
        self.robots = robots
        self.formation = formation
        self._gains = {
            "position": 2.0,
            "velocity": 1.0,
            "heading": 1.0,
        }

    def set_gains(
        self,
        position: float = 2.0,
        velocity: float = 1.0,
        heading: float = 1.0,
    ) -> None:
        """Set controller gains.

        Args:
            position: Position error gain.
            velocity: Velocity error gain.
            heading: Heading error gain.
        """
        self._gains = {
            "position": position,
            "velocity": velocity,
            "heading": heading,
        }

    def compute_formation_control(
        self,
        leader_pose: NDArray[np.floating],
        robot_positions: dict[str, NDArray[np.floating]],
        robot_velocities: dict[str, NDArray[np.floating]] | None = None,
    ) -> dict[str, NDArray[np.floating]]:
        """Compute control for each robot to maintain formation.

        Args:
            leader_pose: Leader's pose (7D or 3D for position only).
            robot_positions: Current positions of all robots.
            robot_velocities: Current velocities (optional).

        Returns:
            Dictionary mapping robot IDs to velocity commands.
        """
        commands = {}

        # Leader position and orientation
        if len(leader_pose) >= 7:
            leader_pos = leader_pose[:3]
            leader_quat = leader_pose[3:7]
        else:
            leader_pos = leader_pose[:3]
            leader_quat = np.array([1, 0, 0, 0])  # Identity

        # Compute rotation matrix from quaternion
        R = self._quat_to_rotation(leader_quat)

        for i, robot_id in enumerate(self.robots):
            if robot_id not in robot_positions:
                continue

            # Desired position in world frame
            local_offset = self.formation.positions[i]
            desired_pos = leader_pos + R @ local_offset

            # Current position
            current_pos = robot_positions[robot_id][:3]

            # Position error
            pos_error = desired_pos - current_pos

            # Velocity command (P control)
            velocity_cmd = self._gains["position"] * pos_error

            # Add velocity matching if available
            if robot_velocities and robot_id in robot_velocities:
                # Simple damping
                current_vel = robot_velocities[robot_id][:3]
                velocity_cmd -= self._gains["velocity"] * current_vel

            commands[robot_id] = velocity_cmd

        return commands

    def _quat_to_rotation(
        self,
        quat: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Convert quaternion to rotation matrix.

        Args:
            quat: Quaternion [w, x, y, z].

        Returns:
            3x3 rotation matrix.
        """
        w, x, y, z = quat
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ])

    def set_formation(self, formation: FormationConfig) -> None:
        """Change the formation.

        Args:
            formation: New formation configuration.
        """
        self.formation = formation

    def get_formation_error(
        self,
        leader_pose: NDArray[np.floating],
        robot_positions: dict[str, NDArray[np.floating]],
    ) -> float:
        """Compute total formation error.

        Args:
            leader_pose: Leader's pose.
            robot_positions: Current robot positions.

        Returns:
            Sum of position errors.
        """
        total_error = 0.0

        leader_pos = leader_pose[:3]
        leader_quat = (
            leader_pose[3:7] if len(leader_pose) >= 7
            else np.array([1, 0, 0, 0])
        )
        R = self._quat_to_rotation(leader_quat)

        for i, robot_id in enumerate(self.robots):
            if robot_id not in robot_positions:
                continue

            desired_pos = leader_pos + R @ self.formation.positions[i]
            current_pos = robot_positions[robot_id][:3]
            total_error += float(np.linalg.norm(desired_pos - current_pos))

        return total_error


class CooperativeManipulation:
    """Coordinated manipulation by multiple robots.

    Enables multiple robots to jointly manipulate a single object,
    computing force distribution and coordinated motion.

    Attributes:
        robots: List of robot engines.
        object_model: Object being manipulated.
    """

    def __init__(
        self,
        robots: list["PhysicsEngineProtocol"],
        object_model: str | None = None,
    ) -> None:
        """Initialize cooperative manipulation.

        Args:
            robots: List of robot physics engines.
            object_model: Object model identifier.
        """
        self.robots = robots
        self._object_model = object_model
        self._grasp_points: list[NDArray[np.floating]] = []
        self._grasp_normals: list[NDArray[np.floating]] = []

    @property
    def n_robots(self) -> int:
        """Number of robots in cooperation."""
        return len(self.robots)

    def set_grasp_points(
        self,
        grasp_points: list[NDArray[np.floating]],
        grasp_normals: list[NDArray[np.floating]] | None = None,
    ) -> None:
        """Set grasp contact points on the object.

        Args:
            grasp_points: Contact points in object frame.
            grasp_normals: Contact normals (optional).
        """
        self._grasp_points = grasp_points

        if grasp_normals is None:
            # Default: normals pointing inward to object center
            center = np.mean(grasp_points, axis=0)
            self._grasp_normals = [
                (center - p) / np.linalg.norm(center - p)
                for p in grasp_points
            ]
        else:
            self._grasp_normals = grasp_normals

    def compute_grasp_matrix(
        self,
        object_pose: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute grasp matrix mapping contact forces to object wrench.

        G^T @ f_contacts = w_object

        Args:
            object_pose: Object pose in world frame (7D).

        Returns:
            Grasp matrix (6, 3*n_contacts).
        """
        n_contacts = len(self._grasp_points)
        G = np.zeros((6, 3 * n_contacts))

        object_pos = object_pose[:3]
        object_quat = (
            object_pose[3:7] if len(object_pose) >= 7
            else np.array([1, 0, 0, 0])
        )

        # Rotation matrix
        w, x, y, z = object_quat
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ])

        for i in range(n_contacts):
            # Contact point in world frame
            contact_world = object_pos + R @ self._grasp_points[i]
            r = contact_world - object_pos

            # Force contribution
            G[:3, i*3:(i+1)*3] = np.eye(3)

            # Torque contribution (cross product matrix)
            G[3:6, i*3:(i+1)*3] = np.array([
                [0, -r[2], r[1]],
                [r[2], 0, -r[0]],
                [-r[1], r[0], 0],
            ])

        return G

    def compute_load_sharing(
        self,
        desired_object_wrench: NDArray[np.floating],
        object_pose: NDArray[np.floating],
    ) -> list[NDArray[np.floating]]:
        """Compute optimal force distribution among robots.

        Minimizes total force magnitude while achieving desired wrench.

        Args:
            desired_object_wrench: Desired wrench on object [fx,fy,fz,tx,ty,tz].
            object_pose: Object pose.

        Returns:
            List of force vectors for each contact.
        """
        G = self.compute_grasp_matrix(object_pose)
        n_contacts = len(self._grasp_points)

        # Solve: min ||f||^2 s.t. G^T @ f = w
        # Using pseudo-inverse: f = G @ (G^T @ G)^-1 @ w
        try:
            GTG_inv = np.linalg.inv(G @ G.T + np.eye(6) * 1e-6)
            f_optimal = G.T @ GTG_inv @ desired_object_wrench
        except np.linalg.LinAlgError:
            f_optimal = np.zeros(3 * n_contacts)

        # Split into per-contact forces
        forces = [
            f_optimal[i*3:(i+1)*3] for i in range(n_contacts)
        ]

        return forces

    def plan_cooperative_motion(
        self,
        object_goal_pose: NDArray[np.floating],
        object_current_pose: NDArray[np.floating],
        dt: float = 0.01,
        duration: float = 2.0,
    ) -> list[NDArray[np.floating]]:
        """Plan coordinated motion trajectories for all robots.

        Generates smooth trajectory for the object and corresponding
        end-effector trajectories for each robot.

        Args:
            object_goal_pose: Goal pose for object.
            object_current_pose: Current pose of object.
            dt: Timestep.
            duration: Motion duration.

        Returns:
            List of end-effector trajectories, one per robot.
        """
        n_steps = int(duration / dt)
        n_contacts = len(self._grasp_points)
        trajectories = [np.zeros((n_steps, 7)) for _ in range(n_contacts)]

        # Interpolate object pose
        pos_start = object_current_pose[:3]
        pos_end = object_goal_pose[:3]

        quat_start = (
            object_current_pose[3:7] if len(object_current_pose) >= 7
            else np.array([1, 0, 0, 0])
        )
        quat_end = (
            object_goal_pose[3:7] if len(object_goal_pose) >= 7
            else np.array([1, 0, 0, 0])
        )

        for t in range(n_steps):
            # Smooth interpolation parameter
            s = t / (n_steps - 1) if n_steps > 1 else 1.0
            s_smooth = s * s * (3 - 2 * s)  # Smoothstep

            # Interpolate position
            object_pos = (1 - s_smooth) * pos_start + s_smooth * pos_end

            # Interpolate rotation (SLERP approximation)
            object_quat = self._slerp(quat_start, quat_end, s_smooth)

            # Rotation matrix
            R = self._quat_to_rotation(object_quat)

            # Compute end-effector poses
            for i in range(n_contacts):
                ee_pos = object_pos + R @ self._grasp_points[i]
                trajectories[i][t, :3] = ee_pos
                trajectories[i][t, 3:7] = object_quat

        return trajectories

    def _slerp(
        self,
        q0: NDArray[np.floating],
        q1: NDArray[np.floating],
        t: float,
    ) -> NDArray[np.floating]:
        """Spherical linear interpolation for quaternions.

        Args:
            q0: Start quaternion.
            q1: End quaternion.
            t: Interpolation parameter [0, 1].

        Returns:
            Interpolated quaternion.
        """
        dot = np.dot(q0, q1)

        # If negative dot, negate one quaternion
        if dot < 0:
            q1 = -q1
            dot = -dot

        if dot > 0.9995:
            # Linear interpolation for nearly identical quaternions
            result = (1 - t) * q0 + t * q1
            return result / np.linalg.norm(result)

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        s0 = np.sin((1 - t) * theta) / sin_theta
        s1 = np.sin(t * theta) / sin_theta

        return s0 * q0 + s1 * q1

    def _quat_to_rotation(
        self,
        quat: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = quat
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ])

    def check_force_closure(
        self,
        object_pose: NDArray[np.floating],
        friction_coefficient: float = 0.5,
    ) -> tuple[bool, float]:
        """Check if grasp has force closure.

        Args:
            object_pose: Object pose.
            friction_coefficient: Friction coefficient.

        Returns:
            Tuple of (has_closure, quality_metric).
        """
        G = self.compute_grasp_matrix(object_pose)

        # Simple check: rank of grasp matrix
        rank = np.linalg.matrix_rank(G, tol=1e-6)
        has_closure = rank >= 6

        # Quality metric: smallest singular value
        s = np.linalg.svd(G, compute_uv=False)
        quality = float(s[-1]) if len(s) > 0 else 0.0

        return has_closure, quality
