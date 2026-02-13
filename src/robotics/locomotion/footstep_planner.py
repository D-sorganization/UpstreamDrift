"""Footstep planning for bipedal locomotion.

This module provides footstep generation and planning for
walking and running gaits.

Design by Contract:
    Footsteps are validated for reachability.
    Plans are consistent with gait parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.robotics.locomotion.gait_types import GaitParameters
from src.shared.python.core.contracts import ContractChecker


@dataclass
class Footstep:
    """Single footstep target.

    Attributes:
        position: Footstep position in world frame (3,).
        orientation: Footstep orientation as quaternion (4,) [w, x, y, z].
        foot: Which foot ('left' or 'right').
        step_index: Index in footstep sequence.
        is_support: Whether this is a support foot position.
        timing: Planned timing for this step [s].
        duration: Duration to reach this position [s].
    """

    position: NDArray[np.float64]
    orientation: NDArray[np.float64]
    foot: str
    step_index: int = 0
    is_support: bool = True
    timing: float = 0.0
    duration: float = 0.5

    def __post_init__(self) -> None:
        """Validate footstep data."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)

        if self.position.shape != (3,):
            raise ValueError(f"Position must be (3,), got {self.position.shape}")
        if self.orientation.shape != (4,):
            raise ValueError(f"Orientation must be (4,), got {self.orientation.shape}")
        if self.foot not in ("left", "right"):
            raise ValueError(f"Foot must be 'left' or 'right', got {self.foot}")

    @property
    def yaw(self) -> float:
        """Get yaw angle from orientation quaternion."""
        w, x, y, z = self.orientation
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def get_pose_matrix(self) -> NDArray[np.float64]:
        """Get 4x4 pose matrix."""
        T = np.eye(4)
        T[:3, 3] = self.position
        T[:3, :3] = self._quat_to_rot(self.orientation)
        return T

    def _quat_to_rot(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * w * z,
                    2 * x * z + 2 * w * y,
                ],
                [
                    2 * x * y + 2 * w * z,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * w * x,
                ],
                [
                    2 * x * z - 2 * w * y,
                    2 * y * z + 2 * w * x,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )


@dataclass
class FootstepPlan:
    """Complete footstep plan.

    Attributes:
        footsteps: List of planned footsteps.
        parameters: Gait parameters used for planning.
        start_position: Starting position (3,).
        goal_position: Goal position (3,).
        total_duration: Total plan duration [s].
    """

    footsteps: list[Footstep] = field(default_factory=list)
    parameters: GaitParameters = field(default_factory=GaitParameters)
    start_position: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    goal_position: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    total_duration: float = 0.0

    def __len__(self) -> int:
        return len(self.footsteps)

    def __iter__(self) -> Iterator[Footstep]:
        return iter(self.footsteps)

    def __getitem__(self, index: int) -> Footstep:
        return self.footsteps[index]

    @property
    def n_steps(self) -> int:
        """Get number of steps."""
        return len(self.footsteps)

    def get_footsteps_for_foot(self, foot: str) -> list[Footstep]:
        """Get all footsteps for a specific foot.

        Args:
            foot: 'left' or 'right'.

        Returns:
            List of footsteps for that foot.
        """
        return [fs for fs in self.footsteps if fs.foot == foot]

    def get_footstep_at_time(self, t: float) -> Footstep | None:
        """Get footstep active at given time.

        Args:
            t: Time in plan [s].

        Returns:
            Active footstep or None if time is out of range.
        """
        for fs in self.footsteps:
            if fs.timing <= t < fs.timing + fs.duration:
                return fs
        return None


class FootstepPlanner(ContractChecker):
    """Plans footstep sequences for locomotion.

    Generates footstep positions based on velocity commands,
    goal positions, or terrain constraints.

    Design by Contract:
        Invariants:
            - Step dimension limits are positive
            - Nominal width is non-negative

        Preconditions:
            - Parameters must have valid step dimensions

        Postconditions:
            - Generated steps are within kinematic limits
            - Alternating left/right sequence

    Example:
        >>> planner = FootstepPlanner(GaitParameters())
        >>> plan = planner.plan_to_goal(
        ...     start=np.zeros(3),
        ...     goal=np.array([2.0, 0.0, 0.0]),
        ... )
        >>> for step in plan:
        ...     print(f"{step.foot}: {step.position}")
    """

    def __init__(
        self,
        parameters: GaitParameters,
        max_step_length: float = 0.5,
        max_step_width: float = 0.4,
        max_step_rotation: float = 0.5,
    ) -> None:
        """Initialize footstep planner.

        Args:
            parameters: Gait parameters.
            max_step_length: Maximum forward step length [m].
            max_step_width: Maximum lateral step width [m].
            max_step_rotation: Maximum step rotation [rad].
        """
        self._parameters = parameters
        self._max_step_length = max_step_length
        self._max_step_width = max_step_width
        self._max_step_rotation = max_step_rotation
        self._nominal_width = parameters.step_width

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Define class invariants for FootstepPlanner."""
        return [
            (
                lambda: self._max_step_length > 0,
                "max_step_length must be positive",
            ),
            (
                lambda: self._max_step_width > 0,
                "max_step_width must be positive",
            ),
            (
                lambda: self._max_step_rotation > 0,
                "max_step_rotation must be positive",
            ),
            (
                lambda: self._nominal_width >= 0,
                "nominal_width must be non-negative",
            ),
        ]

    @property
    def parameters(self) -> GaitParameters:
        """Get gait parameters."""
        return self._parameters

    def set_parameters(self, parameters: GaitParameters) -> None:
        """Set new gait parameters."""
        self._parameters = parameters
        self._nominal_width = parameters.step_width

    def plan_to_goal(
        self,
        start: NDArray[np.float64],
        goal: NDArray[np.float64],
        start_yaw: float = 0.0,
        goal_yaw: float | None = None,
        start_foot: str = "left",
    ) -> FootstepPlan:
        """Plan footsteps to reach a goal position.

        Args:
            start: Starting position (3,).
            goal: Goal position (3,).
            start_yaw: Starting orientation [rad].
            goal_yaw: Goal orientation [rad]. Faces goal if None.
            start_foot: Which foot to step with first.

        Returns:
            FootstepPlan to reach goal.
        """
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        # Compute direction to goal
        direction = goal - start
        distance = float(np.linalg.norm(direction[:2]))

        if distance < 1e-3:
            # Already at goal, return empty plan
            return FootstepPlan(
                footsteps=[],
                parameters=self._parameters,
                start_position=start,
                goal_position=goal,
                total_duration=0.0,
            )

        if goal_yaw is None:
            goal_yaw = float(np.arctan2(direction[1], direction[0]))

        # Generate footsteps along path
        footsteps = self._generate_straight_path(
            start, goal, start_yaw, goal_yaw, start_foot
        )

        # Compute timing
        total_duration = 0.0
        for fs in footsteps:
            fs.timing = total_duration
            fs.duration = self._parameters.step_duration
            total_duration += self._parameters.step_duration

        return FootstepPlan(
            footsteps=footsteps,
            parameters=self._parameters,
            start_position=start,
            goal_position=goal,
            total_duration=total_duration,
        )

    def plan_from_velocity(
        self,
        current_position: NDArray[np.float64],
        current_yaw: float,
        velocity_command: NDArray[np.float64],
        n_steps: int = 4,
        start_foot: str = "left",
    ) -> FootstepPlan:
        """Plan footsteps from velocity command.

        Args:
            current_position: Current position (3,).
            current_yaw: Current orientation [rad].
            velocity_command: Velocity command [vx, vy, omega].
            n_steps: Number of steps to plan.
            start_foot: Which foot to step with first.

        Returns:
            FootstepPlan following velocity command.
        """
        current_position = np.asarray(current_position, dtype=np.float64)
        velocity_command = np.asarray(velocity_command, dtype=np.float64)

        vx = velocity_command[0]
        vy = velocity_command[1]
        omega = velocity_command[2] if len(velocity_command) > 2 else 0.0

        footsteps: list[Footstep] = []
        pos = current_position.copy()
        yaw = current_yaw
        foot = start_foot
        timing = 0.0

        for i in range(n_steps):
            step_x, step_y, step_yaw = self._compute_clamped_step(vx, vy, omega)

            yaw += step_yaw / 2
            pos = self._advance_position(pos, yaw, step_x, step_y)
            foot_pos = self._compute_foot_position(pos, yaw, foot)
            yaw += step_yaw / 2

            orientation = self._yaw_to_quat(yaw)
            footstep = Footstep(
                position=foot_pos,
                orientation=orientation,
                foot=foot,
                step_index=i,
                timing=timing,
                duration=self._parameters.step_duration,
            )
            footsteps.append(footstep)

            foot = "right" if foot == "left" else "left"
            timing += self._parameters.step_duration

        return FootstepPlan(
            footsteps=footsteps,
            parameters=self._parameters,
            start_position=current_position,
            goal_position=pos,
            total_duration=timing,
        )

    def _compute_clamped_step(self, vx, vy, omega):
        dt = self._parameters.step_duration
        step_x = np.clip(vx * dt, -self._max_step_length, self._max_step_length)
        step_y = np.clip(vy * dt, -self._max_step_width, self._max_step_width)
        step_yaw = np.clip(
            omega * dt, -self._max_step_rotation, self._max_step_rotation
        )
        return step_x, step_y, step_yaw

    def _advance_position(self, pos, yaw, step_x, step_y):
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        pos[0] += cos_yaw * step_x - sin_yaw * step_y
        pos[1] += sin_yaw * step_x + cos_yaw * step_y
        return pos

    def _compute_foot_position(self, pos, yaw, foot):
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        lateral_offset = self._nominal_width / 2
        if foot == "left":
            offset_x = -sin_yaw * lateral_offset
            offset_y = cos_yaw * lateral_offset
        else:
            offset_x = sin_yaw * lateral_offset
            offset_y = -cos_yaw * lateral_offset

        return np.array([pos[0] + offset_x, pos[1] + offset_y, pos[2]])

    def plan_in_place_turn(
        self,
        current_position: NDArray[np.float64],
        current_yaw: float,
        target_yaw: float,
        start_foot: str = "left",
    ) -> FootstepPlan:
        """Plan footsteps for in-place rotation.

        Args:
            current_position: Current position (3,).
            current_yaw: Current orientation [rad].
            target_yaw: Target orientation [rad].
            start_foot: Which foot to step with first.

        Returns:
            FootstepPlan for rotation.
        """
        current_position = np.asarray(current_position, dtype=np.float64)

        # Compute rotation needed
        delta_yaw = self._normalize_angle(target_yaw - current_yaw)

        # Number of steps needed
        n_steps = max(1, int(np.ceil(abs(delta_yaw) / self._max_step_rotation)))
        step_yaw = delta_yaw / n_steps

        footsteps: list[Footstep] = []
        yaw = current_yaw
        foot = start_foot
        timing = 0.0

        for i in range(n_steps):
            yaw += step_yaw

            # Compute foot position with lateral offset
            lateral_offset = self._nominal_width / 2
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            if foot == "left":
                offset_x = -sin_yaw * lateral_offset
                offset_y = cos_yaw * lateral_offset
            else:
                offset_x = sin_yaw * lateral_offset
                offset_y = -cos_yaw * lateral_offset

            foot_pos = np.array(
                [
                    current_position[0] + offset_x,
                    current_position[1] + offset_y,
                    current_position[2],
                ]
            )

            orientation = self._yaw_to_quat(yaw)
            footstep = Footstep(
                position=foot_pos,
                orientation=orientation,
                foot=foot,
                step_index=i,
                timing=timing,
                duration=self._parameters.step_duration,
            )
            footsteps.append(footstep)

            foot = "right" if foot == "left" else "left"
            timing += self._parameters.step_duration

        return FootstepPlan(
            footsteps=footsteps,
            parameters=self._parameters,
            start_position=current_position,
            goal_position=current_position,
            total_duration=timing,
        )

    def _generate_straight_path(
        self,
        start: NDArray[np.float64],
        goal: NDArray[np.float64],
        start_yaw: float,
        goal_yaw: float,
        start_foot: str,
    ) -> list[Footstep]:
        """Generate footsteps along straight path."""
        direction = goal - start
        distance = float(np.linalg.norm(direction[:2]))
        path_yaw = float(np.arctan2(direction[1], direction[0]))

        # Number of steps
        step_length = min(self._parameters.step_length, self._max_step_length)
        n_steps = max(1, int(np.ceil(distance / step_length)))
        distance / n_steps

        footsteps: list[Footstep] = []
        foot = start_foot

        for i in range(n_steps):
            # Interpolate position along path
            t = (i + 1) / n_steps
            pos_on_path = start + t * direction

            # Interpolate orientation
            yaw = start_yaw + t * self._normalize_angle(goal_yaw - start_yaw)

            # Add lateral offset
            cos_yaw = np.cos(path_yaw)
            sin_yaw = np.sin(path_yaw)
            lateral_offset = self._nominal_width / 2

            if foot == "left":
                offset_x = -sin_yaw * lateral_offset
                offset_y = cos_yaw * lateral_offset
            else:
                offset_x = sin_yaw * lateral_offset
                offset_y = -cos_yaw * lateral_offset

            foot_pos = np.array(
                [
                    pos_on_path[0] + offset_x,
                    pos_on_path[1] + offset_y,
                    pos_on_path[2],
                ]
            )

            orientation = self._yaw_to_quat(yaw)
            footstep = Footstep(
                position=foot_pos,
                orientation=orientation,
                foot=foot,
                step_index=i,
            )
            footsteps.append(footstep)

            foot = "right" if foot == "left" else "left"

        return footsteps

    def _yaw_to_quat(self, yaw: float) -> NDArray[np.float64]:
        """Convert yaw angle to quaternion [w, x, y, z]."""
        return np.array(
            [
                np.cos(yaw / 2),
                0.0,
                0.0,
                np.sin(yaw / 2),
            ]
        )

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
