"""Core interfaces and types for pose editing.

Provides the foundation for pose manipulation across physics engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class JointType(Enum):
    """Types of joints supported by the pose editor."""

    REVOLUTE = auto()  # Rotation around a single axis
    PRISMATIC = auto()  # Translation along a single axis
    SPHERICAL = auto()  # 3-DOF rotation (ball joint)
    PLANAR = auto()  # 2-DOF translation in a plane
    FREE = auto()  # 6-DOF (position + orientation)
    FIXED = auto()  # No DOF (weld)
    UNKNOWN = auto()


@dataclass
class JointInfo:
    """Information about a single joint."""

    name: str
    index: int  # Index in the model
    joint_type: JointType
    position_index: int  # Start index in q vector
    velocity_index: int  # Start index in v vector
    num_positions: int  # Number of position DOFs
    num_velocities: int  # Number of velocity DOFs

    # Joint limits
    lower_limit: float | np.ndarray = -np.pi
    upper_limit: float | np.ndarray = np.pi

    # Current value (updated during editing)
    current_position: float | np.ndarray = 0.0
    current_velocity: float | np.ndarray = 0.0

    # UI metadata
    display_name: str = ""
    group: str = "General"  # For organizing in UI (e.g., "Arm", "Leg", "Spine")
    unit: str = "rad"  # "rad", "m", "deg"

    def __post_init__(self) -> None:
        """Initialize derived fields."""
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()

    def is_single_dof(self) -> bool:
        """Check if this is a single-DOF joint."""
        return self.num_positions == 1

    def get_limit_range(self) -> float:
        """Get the range of motion for the joint."""
        if isinstance(self.lower_limit, np.ndarray):
            return float(np.max(self.upper_limit - self.lower_limit))
        return float(self.upper_limit - self.lower_limit)

    def clamp_position(self, value: float | np.ndarray) -> float | np.ndarray:
        """Clamp a position value to joint limits."""
        return np.clip(value, self.lower_limit, self.upper_limit)


@dataclass
class PoseEditorState:
    """Current state of the pose editor."""

    # Mode
    is_editing: bool = False
    gravity_enabled: bool = True
    physics_paused: bool = False

    # Current pose
    joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.array([]))

    # Selection
    selected_joint_index: int = -1
    selected_body_name: str = ""

    # IK mode (for drag manipulation)
    ik_enabled: bool = False
    ik_target_position: np.ndarray | None = None
    ik_target_body: str = ""

    # Constraints
    constrained_bodies: list[str] = field(default_factory=list)

    def copy(self) -> PoseEditorState:
        """Create a deep copy of the state."""
        return PoseEditorState(
            is_editing=self.is_editing,
            gravity_enabled=self.gravity_enabled,
            physics_paused=self.physics_paused,
            joint_positions=self.joint_positions.copy(),
            joint_velocities=self.joint_velocities.copy(),
            selected_joint_index=self.selected_joint_index,
            selected_body_name=self.selected_body_name,
            ik_enabled=self.ik_enabled,
            ik_target_position=(
                self.ik_target_position.copy()
                if self.ik_target_position is not None
                else None
            ),
            ik_target_body=self.ik_target_body,
            constrained_bodies=self.constrained_bodies.copy(),
        )


@runtime_checkable
class PoseEditorInterface(Protocol):
    """Protocol for pose editor implementations.

    Each physics engine (Drake, Pinocchio, MuJoCo) should implement this
    interface to provide pose editing functionality.
    """

    def get_joint_info(self) -> list[JointInfo]:
        """Get information about all joints in the model.

        Returns:
            List of JointInfo objects for each joint
        """
        ...

    def get_joint_position(self, joint_index: int) -> float | np.ndarray:
        """Get the current position of a joint.

        Args:
            joint_index: Index of the joint

        Returns:
            Current position value(s)
        """
        ...

    def set_joint_position(self, joint_index: int, value: float | np.ndarray) -> None:
        """Set the position of a joint.

        Args:
            joint_index: Index of the joint
            value: New position value(s)
        """
        ...

    def get_all_positions(self) -> np.ndarray:
        """Get all joint positions as a single array.

        Returns:
            Array of all joint positions (q vector)
        """
        ...

    def set_all_positions(self, positions: np.ndarray) -> None:
        """Set all joint positions from a single array.

        Args:
            positions: Array of joint positions (q vector)
        """
        ...

    def get_all_velocities(self) -> np.ndarray:
        """Get all joint velocities.

        Returns:
            Array of all joint velocities (v vector)
        """
        ...

    def set_all_velocities(self, velocities: np.ndarray) -> None:
        """Set all joint velocities.

        Args:
            velocities: Array of joint velocities (v vector)
        """
        ...

    def set_gravity_enabled(self, enabled: bool) -> None:
        """Enable or disable gravity in the simulation.

        Args:
            enabled: True to enable gravity, False to disable
        """
        ...

    def is_gravity_enabled(self) -> bool:
        """Check if gravity is currently enabled.

        Returns:
            True if gravity is enabled
        """
        ...

    def update_visualization(self) -> None:
        """Update the visualization to reflect current pose."""
        ...

    def get_body_names(self) -> list[str]:
        """Get list of body names in the model.

        Returns:
            List of body names
        """
        ...

    def get_body_position(self, body_name: str) -> np.ndarray | None:
        """Get the world position of a body.

        Args:
            body_name: Name of the body

        Returns:
            Position (3,) or None if not found
        """
        ...


class BasePoseEditor(ABC):
    """Base class for pose editor implementations.

    Provides common functionality for pose editing that can be used
    by all physics engine implementations.
    """

    def __init__(self) -> None:
        """Initialize the pose editor."""
        self._state = PoseEditorState()
        self._joint_info: list[JointInfo] = []
        self._original_gravity: float = 9.81
        self._callbacks: dict[str, list[Any]] = {
            "pose_changed": [],
            "gravity_changed": [],
            "selection_changed": [],
        }

    @property
    def state(self) -> PoseEditorState:
        """Get the current editor state."""
        return self._state

    @property
    def joint_info(self) -> list[JointInfo]:
        """Get joint information."""
        return self._joint_info

    def register_callback(self, event: str, callback: Any) -> None:
        """Register a callback for pose editor events.

        Args:
            event: Event name ("pose_changed", "gravity_changed", "selection_changed")
            callback: Callable to invoke
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _notify(self, event: str, *args: Any) -> None:
        """Notify registered callbacks of an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args)
            except (RuntimeError, ValueError, OSError) as e:
                logger.warning("Callback error for %s: %s", event, e)

    def enter_edit_mode(self) -> None:
        """Enter pose editing mode."""
        self._state.is_editing = True
        self._state.physics_paused = True

        # Store current gravity state
        self._original_gravity = self._get_gravity_magnitude()

        logger.info("Entered pose edit mode")

    def exit_edit_mode(self) -> None:
        """Exit pose editing mode."""
        self._state.is_editing = False
        self._state.physics_paused = False

        # Restore gravity if needed
        if not self._state.gravity_enabled:
            self.set_gravity_enabled(True)

        logger.info("Exited pose edit mode")

    def toggle_gravity(self) -> bool:
        """Toggle gravity on/off.

        Returns:
            New gravity state (True = enabled)
        """
        new_state = not self._state.gravity_enabled
        self.set_gravity_enabled(new_state)
        return new_state

    @abstractmethod
    def set_gravity_enabled(self, enabled: bool) -> None:
        """Enable or disable gravity."""

    @abstractmethod
    def _get_gravity_magnitude(self) -> float:
        """Get current gravity magnitude."""

    def select_joint(self, joint_index: int) -> None:
        """Select a joint for editing.

        Args:
            joint_index: Index of the joint to select
        """
        self._state.selected_joint_index = joint_index
        self._notify("selection_changed", joint_index)

    def get_selected_joint(self) -> JointInfo | None:
        """Get the currently selected joint.

        Returns:
            JointInfo or None if no selection
        """
        idx = self._state.selected_joint_index
        if 0 <= idx < len(self._joint_info):
            return self._joint_info[idx]
        return None

    @abstractmethod
    def set_joint_position(self, joint_index: int, value: float | np.ndarray) -> None:
        """Set a joint position."""

    @abstractmethod
    def get_joint_position(self, joint_index: int) -> float | np.ndarray:
        """Get a joint position."""

    @abstractmethod
    def get_all_positions(self) -> np.ndarray:
        """Get all joint positions."""

    @abstractmethod
    def set_all_positions(self, positions: np.ndarray) -> None:
        """Set all joint positions."""

    def zero_velocities(self) -> None:
        """Set all joint velocities to zero.

        Useful when posing to prevent drift.
        """
        velocities = np.zeros(len(self._state.joint_velocities))
        self.set_all_velocities(velocities)
        self._state.joint_velocities = velocities

    @abstractmethod
    def set_all_velocities(self, velocities: np.ndarray) -> None:
        """Set all joint velocities."""

    @abstractmethod
    def update_visualization(self) -> None:
        """Update visualization to reflect current pose."""

    def get_joint_by_name(self, name: str) -> JointInfo | None:
        """Get joint info by name.

        Args:
            name: Joint name

        Returns:
            JointInfo or None if not found
        """
        for joint in self._joint_info:
            if joint.name.lower() == name.lower():
                return joint
        return None

    def get_joints_by_group(self, group: str) -> list[JointInfo]:
        """Get all joints in a group.

        Args:
            group: Group name (e.g., "Arm", "Leg")

        Returns:
            List of joints in the group
        """
        return [j for j in self._joint_info if j.group.lower() == group.lower()]

    def get_all_groups(self) -> list[str]:
        """Get list of all joint groups.

        Returns:
            Unique group names
        """
        groups = set(j.group for j in self._joint_info)
        return sorted(groups)

    def apply_joint_deltas(self, deltas: dict[int, float]) -> None:
        """Apply incremental changes to multiple joints.

        Args:
            deltas: Dictionary mapping joint index to position delta
        """
        for joint_idx, delta in deltas.items():
            if 0 <= joint_idx < len(self._joint_info):
                joint = self._joint_info[joint_idx]
                current = self.get_joint_position(joint_idx)
                if isinstance(current, np.ndarray):
                    current = current[0]
                new_value = joint.clamp_position(float(current) + delta)
                self.set_joint_position(joint_idx, new_value)

    def reset_to_neutral(self) -> None:
        """Reset all joints to neutral (zero) position."""
        neutral_positions = np.zeros(len(self._state.joint_positions))
        self.set_all_positions(neutral_positions)
        self.zero_velocities()
        self._notify("pose_changed", neutral_positions)

    def mirror_pose(self, axis: str = "sagittal") -> None:
        """Mirror the current pose across an axis.

        Args:
            axis: Axis to mirror across ("sagittal", "frontal", "transverse")
        """
        # This requires knowledge of joint naming conventions
        # Implementation depends on specific model structure
        logger.info("Mirror pose requested for axis: %s", axis)
        # Default implementation does nothing
        # Subclasses should override with model-specific logic
