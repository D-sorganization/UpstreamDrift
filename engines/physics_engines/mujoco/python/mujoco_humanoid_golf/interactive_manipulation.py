"""Interactive drag-and-pose manipulation system for MuJoCo models.

This module provides:
- Mouse picking via ray-casting
- IK-based drag manipulation
- Body constraints (fixed in space or relative to other bodies)
- Pose library (save/load/interpolate poses)
- Visual feedback for selected bodies and constraints
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum

import mujoco
import numpy as np


class ConstraintType(Enum):
    """Types of constraints for fixing segments."""

    NONE = "none"
    FIXED_IN_SPACE = "fixed_in_space"
    RELATIVE_TO_BODY = "relative_to_body"


@dataclass
class BodyConstraint:
    """Constraint for fixing a body segment."""

    body_id: int
    constraint_type: ConstraintType
    target_position: np.ndarray | None = None
    target_orientation: np.ndarray | None = None
    reference_body_id: int | None = None  # For relative constraints
    relative_position: np.ndarray | None = None
    relative_orientation: np.ndarray | None = None
    active: bool = True


@dataclass
class StoredPose:
    """Stored pose configuration."""

    name: str
    qpos: np.ndarray
    qvel: np.ndarray | None = None
    timestamp: float = 0.0
    description: str = ""


class MousePickingRay:
    """Ray-casting for mouse picking in 3D scene."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize mouse picking.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data

    def screen_to_ray(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        camera: mujoco.MjvCamera,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert screen coordinates to 3D ray.

        Args:
            x: Screen x coordinate
            y: Screen y coordinate
            width: Viewport width
            height: Viewport height
            camera: MuJoCo camera

        Returns:
            Tuple of (ray_origin [3], ray_direction [3])
        """
        # Normalize screen coordinates to [-1, 1]
        x_ndc = (2.0 * x) / width - 1.0
        y_ndc = 1.0 - (2.0 * y) / height  # Flip y

        # Get camera position and orientation
        cam_pos = camera.lookat.copy()
        cam_distance = camera.distance
        cam_azimuth = np.deg2rad(camera.azimuth)
        cam_elevation = np.deg2rad(camera.elevation)

        # Compute camera frame vectors
        # Forward vector (from lookat to camera)
        forward = np.array(
            [
                np.cos(cam_elevation) * np.sin(cam_azimuth),
                np.cos(cam_elevation) * np.cos(cam_azimuth),
                np.sin(cam_elevation),
            ],
        )

        # Camera position
        ray_origin = cam_pos - forward * cam_distance

        # Right and up vectors
        up_world = np.array([0, 0, 1])
        right = np.cross(up_world, forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(forward, right)

        # Compute ray direction using field of view
        fovy = 45.0  # Default field of view
        aspect = width / height

        # Ray direction in camera space
        ray_dir = forward.copy()
        ray_dir += right * x_ndc * np.tan(np.deg2rad(fovy / 2)) * aspect
        ray_dir += up * y_ndc * np.tan(np.deg2rad(fovy / 2))
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        return ray_origin, ray_dir

    def pick_body(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        camera: mujoco.MjvCamera,
        max_distance: float = 100.0,
    ) -> tuple[int, np.ndarray, float] | None:
        """Pick a body using mouse coordinates.

        Args:
            x: Screen x coordinate
            y: Screen y coordinate
            width: Viewport width
            height: Viewport height
            camera: MuJoCo camera
            max_distance: Maximum ray distance

        Returns:
            Tuple of (body_id, intersection_point, distance) or None
        """
        ray_origin, ray_dir = self.screen_to_ray(x, y, width, height, camera)

        # Test ray against all body geometries
        closest_body = None
        closest_distance = max_distance
        closest_point = None

        for body_id in range(1, self.model.nbody):  # Skip world body (0)
            # Get body position
            body_pos = self.data.xpos[body_id].copy()

            # Simple sphere intersection test
            # (More sophisticated methods could use actual geom shapes)
            to_body = body_pos - ray_origin
            proj_length = np.dot(to_body, ray_dir)

            if proj_length < 0:
                continue

            # Closest point on ray to body
            closest_on_ray = ray_origin + ray_dir * proj_length
            distance_to_body = np.linalg.norm(closest_on_ray - body_pos)

            # Use body's bounding sphere (approximate)
            body_radius = 0.1  # Default radius

            # Get geometries for this body
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == body_id:
                    # Get geom size
                    geom_size = self.model.geom_size[geom_id]
                    body_radius = max(body_radius, geom_size[0])

            # Check if ray intersects bounding sphere
            if distance_to_body < body_radius * 1.5:  # 1.5x radius for easier picking
                if proj_length < closest_distance:
                    closest_distance = proj_length
                    closest_body = body_id
                    closest_point = closest_on_ray

        if closest_body is not None and closest_point is not None:
            return closest_body, closest_point, closest_distance

        return None


class InteractiveManipulator:
    """Interactive manipulation system with IK-based dragging and constraints."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize interactive manipulator.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data

        # Mouse picking
        self.picker = MousePickingRay(model, data)

        # IK parameters
        self.ik_damping = 0.05
        self.ik_max_iterations = 20
        self.ik_tolerance = 1e-3
        self.ik_step_size = 0.3

        # Selection state
        self.selected_body_id: int | None = None
        self.drag_offset: np.ndarray | None = None
        self.original_qpos: np.ndarray | None = None

        # Constraints
        self.constraints: dict[int, BodyConstraint] = {}

        # Pose library
        self.pose_library: dict[str, StoredPose] = {}

        # Settings
        self.drag_enabled = True
        self.maintain_orientation = False
        self.use_nullspace_posture = True

    def enable_drag(self, enabled: bool) -> None:
        """Enable or disable drag manipulation."""
        self.drag_enabled = enabled

    def select_body(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        camera: mujoco.MjvCamera,
    ) -> int | None:
        """Select a body at screen coordinates.

        Args:
            x: Screen x coordinate
            y: Screen y coordinate
            width: Viewport width
            height: Viewport height
            camera: MuJoCo camera

        Returns:
            Selected body ID or None
        """
        if not self.drag_enabled:
            return None

        result = self.picker.pick_body(x, y, width, height, camera)

        if result is not None:
            body_id, intersection_point, _ = result

            # Store selection
            self.selected_body_id = body_id
            self.original_qpos = self.data.qpos.copy()

            # Compute offset from body center to click point
            body_pos = self.data.xpos[body_id].copy()
            self.drag_offset = intersection_point - body_pos

            return body_id

        return None

    def deselect_body(self) -> None:
        """Deselect current body."""
        self.selected_body_id = None
        self.drag_offset = None
        self.original_qpos = None

    def drag_to(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        camera: mujoco.MjvCamera,
        plane_normal: np.ndarray | None = None,
    ) -> bool:
        """Drag selected body to screen coordinates using IK.

        Args:
            x: Screen x coordinate
            y: Screen y coordinate
            width: Viewport width
            height: Viewport height
            camera: MuJoCo camera
            plane_normal: Normal of drag plane (default: camera view plane)

        Returns:
            True if IK succeeded
        """
        if self.selected_body_id is None or not self.drag_enabled:
            return False

        # Get ray for mouse position
        ray_origin, ray_dir = self.picker.screen_to_ray(x, y, width, height, camera)

        # Compute drag plane
        if plane_normal is None:
            # Use camera forward as plane normal
            cam_azimuth = np.deg2rad(camera.azimuth)
            cam_elevation = np.deg2rad(camera.elevation)
            plane_normal = np.array(
                [
                    np.cos(cam_elevation) * np.sin(cam_azimuth),
                    np.cos(cam_elevation) * np.cos(cam_azimuth),
                    np.sin(cam_elevation),
                ],
            )

        # Plane passes through current body position
        body_pos = self.data.xpos[self.selected_body_id].copy()
        plane_point = body_pos

        # Intersect ray with plane
        denom = np.dot(ray_dir, plane_normal)
        if abs(denom) < 1e-6:
            return False

        t = np.dot(plane_point - ray_origin, plane_normal) / denom
        if t < 0:
            return False

        target_point = ray_origin + ray_dir * t

        # Account for drag offset
        if self.drag_offset is not None:
            target_point -= self.drag_offset

        # Check if this is a mocap body
        mocap_id = self.model.body_mocapid[self.selected_body_id]
        if mocap_id != -1:
            # Direct manipulation of mocap body
            self.data.mocap_pos[mocap_id] = target_point

            # If checking orientation (future enhancement), would update mocap_quat here

            mujoco.mj_forward(self.model, self.data)
            return True

        # Solve IK to move body to target
        success = self._solve_ik_for_body(
            self.selected_body_id,
            target_point,
            maintain_orientation=self.maintain_orientation,
        )

        # Apply constraints
        if success:
            self._apply_constraints()

        return success

    def _solve_ik_for_body(
        self,
        body_id: int,
        target_position: np.ndarray,
        maintain_orientation: bool = False,
    ) -> bool:
        """Solve IK to position a body.

        Args:
            body_id: Body to position
            target_position: Desired position [3]
            maintain_orientation: Whether to maintain orientation

        Returns:
            True if IK succeeded
        """
        # Initialize from current state
        q = self.data.qpos.copy()

        # Target task dimension
        task_dim = 6 if maintain_orientation else 3

        # Get target orientation if needed
        target_quat = None
        if maintain_orientation:
            target_quat = self.data.xquat[body_id].copy()

        # Iterative IK solver
        for _iteration in range(self.ik_max_iterations):
            # Forward kinematics
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)

            # Current position and orientation
            current_pos = self.data.xpos[body_id].copy()
            current_quat = self.data.xquat[body_id].copy()

            # Position error
            pos_error = target_position - current_pos

            # Build task error
            if maintain_orientation and target_quat is not None:
                ori_error = self._orientation_error(current_quat, target_quat)
                task_error = np.concatenate([pos_error, ori_error])
            else:
                task_error = pos_error

            # Check convergence
            if np.linalg.norm(task_error) < self.ik_tolerance:
                self.data.qpos[:] = q
                mujoco.mj_forward(self.model, self.data)
                return True

            # Compute Jacobian (fixed for MuJoCo 3.x API)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)

            J = np.vstack([jacp, jacr]) if maintain_orientation else jacp

            # Damped least-squares
            damping_matrix = self.ik_damping**2 * np.eye(task_dim)
            try:
                J_damped = J.T @ np.linalg.solve(J @ J.T + damping_matrix, task_error)
            except np.linalg.LinAlgError:
                return False

            # Nullspace projection for posture optimization
            if self.use_nullspace_posture and self.original_qpos is not None:
                J_pinv = np.linalg.pinv(J, rcond=self.ik_damping)
                nullspace_proj = np.eye(self.model.nv) - J_pinv @ J

                # Convert position difference to velocity space
                # This handles quaternion joints where nq != nv
                qpos_diff = np.zeros(self.model.nv)
                mujoco.mj_differentiatePos(
                    self.model,
                    qpos_diff,
                    1.0,
                    q,
                    self.original_qpos,
                )

                nullspace_motion = nullspace_proj @ qpos_diff
                J_damped += 0.05 * nullspace_motion

            # Update with step size
            # Use mj_integratePos to handle nq != nv (quaternion joints)
            q_new = np.zeros_like(q)
            mujoco.mj_integratePos(
                self.model,
                q_new,
                J_damped * self.ik_step_size,
                1.0,
            )
            q = q_new

            # Clamp to joint limits
            q = self._clamp_joint_limits(q)

        # Did not converge, but apply best result
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        return False

    def _orientation_error(
        self,
        q_current: np.ndarray,
        q_target: np.ndarray,
    ) -> np.ndarray:
        """Compute orientation error in axis-angle form."""
        # Quaternion difference
        q_current_conj = np.array(
            [q_current[0], -q_current[1], -q_current[2], -q_current[3]],
        )

        # q_error = q_target * q_current_conj
        w1, x1, y1, z1 = q_target
        w2, x2, y2, z2 = q_current_conj

        q_error = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
        )

        # Convert to axis-angle (for small rotations)
        return 2.0 * q_error[1:4]

    def _clamp_joint_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint configuration to limits."""
        q_clamped = q.copy()

        for i in range(min(self.model.njnt, len(q))):
            if self.model.jnt_limited[i]:
                q_min = self.model.jnt_range[i, 0]
                q_max = self.model.jnt_range[i, 1]
                qpos_addr = self.model.jnt_qposadr[i]

                if qpos_addr < len(q_clamped):
                    q_clamped[qpos_addr] = np.clip(q_clamped[qpos_addr], q_min, q_max)

        return q_clamped

    # -------- Constraint Management --------

    def add_constraint(
        self,
        body_id: int,
        constraint_type: ConstraintType,
        reference_body_id: int | None = None,
    ) -> None:
        """Add a constraint to fix a body.

        Args:
            body_id: Body to constrain
            constraint_type: Type of constraint
            reference_body_id: Reference body for relative constraints
        """
        if constraint_type == ConstraintType.FIXED_IN_SPACE:
            # Store current position and orientation
            constraint = BodyConstraint(
                body_id=body_id,
                constraint_type=constraint_type,
                target_position=self.data.xpos[body_id].copy(),
                target_orientation=self.data.xquat[body_id].copy(),
            )
        elif constraint_type == ConstraintType.RELATIVE_TO_BODY:
            if reference_body_id is None:
                msg = "Reference body required for relative constraint"
                raise ValueError(msg)

            # Store relative pose
            rel_pos = self.data.xpos[body_id] - self.data.xpos[reference_body_id]
            constraint = BodyConstraint(
                body_id=body_id,
                constraint_type=constraint_type,
                reference_body_id=reference_body_id,
                relative_position=rel_pos.copy(),
                relative_orientation=self.data.xquat[body_id].copy(),
            )
        else:
            constraint = BodyConstraint(
                body_id=body_id,
                constraint_type=ConstraintType.NONE,
            )

        self.constraints[body_id] = constraint

    def remove_constraint(self, body_id: int) -> None:
        """Remove constraint from body."""
        if body_id in self.constraints:
            del self.constraints[body_id]

    def toggle_constraint(self, body_id: int) -> bool:
        """Toggle constraint active state.

        Returns:
            New active state
        """
        if body_id in self.constraints:
            self.constraints[body_id].active = not self.constraints[body_id].active
            return self.constraints[body_id].active
        return False

    def clear_constraints(self) -> None:
        """Remove all constraints."""
        self.constraints.clear()

    def enforce_constraints(self) -> None:
        """Public wrapper to apply all active constraints.

        Used by the simulation loop to keep constrained bodies pinned even
        when the user is not actively dragging segments.
        """
        self._apply_constraints()

    def _apply_constraints(self) -> None:
        """Apply all active constraints using IK."""
        for body_id, constraint in self.constraints.items():
            if not constraint.active:
                continue

            if constraint.constraint_type == ConstraintType.FIXED_IN_SPACE:
                # Keep body at fixed position
                if constraint.target_position is not None:
                    self._solve_ik_for_body(
                        body_id,
                        constraint.target_position,
                        maintain_orientation=True,
                    )

            elif constraint.constraint_type == ConstraintType.RELATIVE_TO_BODY:
                # Maintain relative pose to reference body
                if (
                    constraint.reference_body_id is not None
                    and constraint.relative_position is not None
                ):
                    ref_pos = self.data.xpos[constraint.reference_body_id]
                    target_pos = ref_pos + constraint.relative_position
                    self._solve_ik_for_body(
                        body_id,
                        target_pos,
                        maintain_orientation=False,
                    )

    def get_constrained_bodies(self) -> list[int]:
        """Get list of currently constrained bodies.

        Returns:
            List of body IDs with active constraints
        """
        return [
            body_id
            for body_id, constraint in self.constraints.items()
            if constraint.active
        ]

    # -------- Pose Library --------

    def save_pose(self, name: str, description: str = "") -> StoredPose:
        """Save current pose to library.

        Args:
            name: Pose name
            description: Optional description

        Returns:
            Stored pose
        """
        pose = StoredPose(
            name=name,
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            timestamp=time.time(),
            description=description,
        )

        self.pose_library[name] = pose
        return pose

    def load_pose(self, name: str, apply_velocities: bool = False) -> bool:
        """Load pose from library.

        Args:
            name: Pose name
            apply_velocities: Whether to apply saved velocities

        Returns:
            True if pose was loaded
        """
        if name not in self.pose_library:
            return False

        pose = self.pose_library[name]
        self.data.qpos[:] = pose.qpos.copy()

        if apply_velocities and pose.qvel is not None:
            self.data.qvel[:] = pose.qvel.copy()
        else:
            self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        return True

    def delete_pose(self, name: str) -> bool:
        """Delete pose from library.

        Args:
            name: Pose name

        Returns:
            True if pose was deleted
        """
        if name in self.pose_library:
            del self.pose_library[name]
            return True
        return False

    def interpolate_poses(
        self,
        pose_name_a: str,
        pose_name_b: str,
        alpha: float,
    ) -> bool:
        """Interpolate between two poses.

        Args:
            pose_name_a: First pose name
            pose_name_b: Second pose name
            alpha: Interpolation factor (0.0 = A, 1.0 = B)

        Returns:
            True if interpolation succeeded
        """
        if pose_name_a not in self.pose_library or pose_name_b not in self.pose_library:
            return False

        pose_a = self.pose_library[pose_name_a]
        pose_b = self.pose_library[pose_name_b]

        # Linear interpolation of joint positions
        alpha = np.clip(alpha, 0.0, 1.0)
        interpolated_qpos = (1 - alpha) * pose_a.qpos + alpha * pose_b.qpos

        self.data.qpos[:] = interpolated_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        return True

    def export_pose_library(self, filepath: str) -> None:
        """Export pose library to JSON file.

        Args:
            filepath: Path to save file
        """
        data = {}
        for name, pose in self.pose_library.items():
            data[name] = {
                "qpos": pose.qpos.tolist(),
                "qvel": pose.qvel.tolist() if pose.qvel is not None else None,
                "timestamp": pose.timestamp,
                "description": pose.description,
            }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def import_pose_library(self, filepath: str) -> int:
        """Import pose library from JSON file.

        Args:
            filepath: Path to load file

        Returns:
            Number of poses imported
        """
        try:
            with open(filepath) as f:
                data = json.load(f)

            count = 0
            for name, pose_data in data.items():
                pose = StoredPose(
                    name=name,
                    qpos=np.array(pose_data["qpos"]),
                    qvel=np.array(pose_data["qvel"]) if pose_data.get("qvel") else None,
                    timestamp=pose_data.get("timestamp", 0.0),
                    description=pose_data.get("description", ""),
                )
                self.pose_library[name] = pose
                count += 1

            return count
        except Exception:
            return 0

    def list_poses(self) -> list[str]:
        """Get list of pose names in library.

        Returns:
            List of pose names
        """
        return list(self.pose_library.keys())

    # -------- Utility Methods --------

    def get_body_name(self, body_id: int) -> str:
        """Get name of body.

        Args:
            body_id: Body ID

        Returns:
            Body name or "body_<id>"
        """
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return name if name else f"body_{body_id}"

    def find_body_by_name(self, name: str) -> int | None:
        """Find body ID by name (case-insensitive, partial match).

        Args:
            name: Body name pattern

        Returns:
            Body ID or None
        """
        for body_id in range(self.model.nbody):
            body_name = self.get_body_name(body_id)
            if name.lower() in body_name.lower():
                return body_id
        return None

    def reset_to_original_pose(self) -> None:
        """Reset to original pose before dragging."""
        if self.original_qpos is not None:
            self.data.qpos[:] = self.original_qpos.copy()
            mujoco.mj_forward(self.model, self.data)
