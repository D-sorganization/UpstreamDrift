"""Input devices for teleoperation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.util import find_spec
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from src.shared.python.core.contracts.exceptions import StateError

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class InputDevice(Protocol):
    """Protocol for teleoperation input devices.

    Input devices provide pose, velocity, and button state
    for controlling the robot end-effector.
    """

    def get_pose(self) -> NDArray[np.floating]:
        """Get current device pose (7D: xyz + quaternion).

        Returns:
            Pose [x, y, z, qw, qx, qy, qz].
        """
        ...

    def get_twist(self) -> NDArray[np.floating]:
        """Get current device velocity (6D: linear + angular).

        Returns:
            Twist [vx, vy, vz, wx, wy, wz].
        """
        ...

    def get_gripper_state(self) -> float:
        """Get gripper command (0.0 = closed, 1.0 = open).

        Returns:
            Gripper openness value.
        """
        ...

    def set_force_feedback(self, wrench: NDArray[np.floating]) -> None:
        """Set haptic force feedback (6D).

        Args:
            wrench: Force/torque feedback [fx, fy, fz, tx, ty, tz].
        """
        ...

    def get_buttons(self) -> dict[str, bool]:
        """Get button states.

        Returns:
            Dictionary mapping button names to pressed state.
        """
        ...


class BaseInputDevice(ABC):
    """Base class for input devices."""

    def __init__(self) -> None:
        """Initialize base device."""
        self._pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        self._twist = np.zeros(6, dtype=np.float64)
        self._gripper = 1.0
        self._buttons: dict[str, bool] = {}
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._is_connected

    def connect(self) -> bool:
        """Connect to device.

        Returns:
            True if connection successful.
        """
        self._is_connected = False
        return False

    def disconnect(self) -> None:
        """Disconnect from device."""
        self._is_connected = False

    def _require_connected(self, operation: str) -> None:
        """Raise when device state is queried without a hardware connection."""
        if not self._is_connected:
            raise StateError(
                f"{self.__class__.__name__} is not connected",
                current_state="disconnected",
                required_state="connected",
                operation=operation,
            )

    def _has_any_backend(self, module_names: tuple[str, ...]) -> bool:
        """Return whether any optional hardware backend module is importable."""
        return any(find_spec(module_name) is not None for module_name in module_names)

    def get_pose(self) -> NDArray[np.floating]:
        """Get current pose."""
        self._require_connected("get_pose")
        return self._pose.copy()

    def get_twist(self) -> NDArray[np.floating]:
        """Get current twist."""
        self._require_connected("get_twist")
        return self._twist.copy()

    def get_gripper_state(self) -> float:
        """Get gripper state."""
        self._require_connected("get_gripper_state")
        return self._gripper

    def set_force_feedback(self, wrench: NDArray[np.floating]) -> None:  # noqa: B027
        """Set force feedback (no-op for devices without haptics)."""
        self._require_connected("set_force_feedback")

    def get_buttons(self) -> dict[str, bool]:
        """Get button states."""
        self._require_connected("get_buttons")
        return self._buttons.copy()

    @abstractmethod
    def update(self) -> None:
        """Update device state (poll hardware)."""


class SpaceMouseInput(BaseInputDevice):
    """3Dconnexion SpaceMouse input device.

    Provides 6-DOF input through the SpaceMouse 3D controller.
    """

    def __init__(self, device_index: int = 0) -> None:
        """Initialize SpaceMouse input.

        Args:
            device_index: Device index if multiple connected.
        """
        if not (device_index is not None):
            raise ValueError("device_index must be provided")
        super().__init__()
        self._device_index = device_index
        self._sensitivity = 1.0
        self._buttons = {
            "button_1": False,
            "button_2": False,
        }

    def connect(self) -> bool:
        """Connect to SpaceMouse."""
        self._is_connected = False
        if not self._has_any_backend(("pyspacemouse", "hid")):
            return False
        return False

    def update(self) -> None:
        """Update SpaceMouse state."""
        self._require_connected("update")

    def set_sensitivity(self, sensitivity: float) -> None:
        """Set input sensitivity.

        Args:
            sensitivity: Sensitivity multiplier.
        """
        self._sensitivity = sensitivity


class VRControllerInput(BaseInputDevice):
    """VR controller input device.

    Supports Oculus Touch, Vive Wands, and similar controllers.
    """

    def __init__(
        self,
        hand: str = "right",
        tracking_system: str = "steamvr",
    ) -> None:
        """Initialize VR controller.

        Args:
            hand: "left" or "right" hand.
            tracking_system: VR tracking system.
        """
        if not (hand is not None):
            raise ValueError("hand must be provided")
        super().__init__()
        self._hand = hand
        self._tracking_system = tracking_system
        self._buttons = {
            "trigger": False,
            "grip": False,
            "thumbstick_click": False,
            "a": False,
            "b": False,
        }
        self._trigger_value = 0.0
        self._grip_value = 0.0

    def connect(self) -> bool:
        """Connect to VR system."""
        self._is_connected = False
        if not self._has_any_backend(("openvr",)):
            return False
        return False

    def update(self) -> None:
        """Update VR controller state."""
        self._require_connected("update")

    def get_gripper_state(self) -> float:
        """Get gripper from trigger value."""
        self._require_connected("get_gripper_state")
        return 1.0 - self._trigger_value

    def get_trigger_value(self) -> float:
        """Get analog trigger value.

        Returns:
            Trigger value 0-1.
        """
        return self._trigger_value

    def get_grip_value(self) -> float:
        """Get analog grip value.

        Returns:
            Grip value 0-1.
        """
        return self._grip_value


class HapticDeviceInput(BaseInputDevice):
    """Haptic device input (Phantom, Sigma.7, etc.).

    Provides high-fidelity position input and force feedback.
    """

    def __init__(self, device_name: str = "phantom") -> None:
        """Initialize haptic device.

        Args:
            device_name: Device identifier.
        """
        if not (device_name is not None):
            raise ValueError("device_name must be provided")
        super().__init__()
        self._device_name = device_name
        self._workspace_scale = 0.001  # mm to m
        self._force_scale = 1.0
        self._max_force = 3.3  # N typical for Phantom Omni
        self._buttons = {
            "button_1": False,
            "button_2": False,
        }

    def connect(self) -> bool:
        """Connect to haptic device."""
        self._is_connected = False
        if not self._has_any_backend(("forcedimension", "dhd")):
            return False
        return False

    def update(self) -> None:
        """Update haptic device state."""
        self._require_connected("update")

    def set_force_feedback(self, wrench: NDArray[np.floating]) -> None:
        """Set haptic force feedback.

        Args:
            wrench: Desired force/torque.
        """
        if not (wrench is not None):
            raise ValueError("wrench must be provided")
        self._require_connected("set_force_feedback")

        # Clip to device limits
        force = wrench[:3] * self._force_scale
        force = np.clip(force, -self._max_force, self._max_force)

    def set_workspace_scale(self, scale: float) -> None:
        """Set workspace scaling factor.

        Args:
            scale: Scale from device units to meters.
        """
        self._workspace_scale = scale


class KeyboardMouseInput(BaseInputDevice):
    """Keyboard and mouse input fallback.

    Provides basic teleoperation using keyboard for
    discrete commands and mouse for position.
    """

    def __init__(self) -> None:
        """Initialize keyboard/mouse input."""
        super().__init__()
        self._buttons = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "up": False,
            "down": False,
            "open_gripper": False,
            "close_gripper": False,
            "emergency_stop": False,
        }
        self._key_velocity = 0.1  # m/s per key

    def connect(self) -> bool:
        """Connect keyboard input."""
        self._is_connected = True
        return True

    def update(self) -> None:
        """Update from keyboard state."""
        self._require_connected("update")

        # Compute velocity from key states
        vx = self._key_velocity * (
            1 if self._buttons["forward"] else 0
        ) - self._key_velocity * (1 if self._buttons["backward"] else 0)
        vy = self._key_velocity * (
            1 if self._buttons["left"] else 0
        ) - self._key_velocity * (1 if self._buttons["right"] else 0)
        vz = self._key_velocity * (
            1 if self._buttons["up"] else 0
        ) - self._key_velocity * (1 if self._buttons["down"] else 0)

        self._twist[:3] = [vx, vy, vz]

        # Update gripper
        if self._buttons["open_gripper"]:
            self._gripper = 1.0
        elif self._buttons["close_gripper"]:
            self._gripper = 0.0

    def set_key_state(self, key: str, pressed: bool) -> None:
        """Set key state (for external input handling).

        Args:
            key: Key name.
            pressed: Whether pressed.
        """
        if key in self._buttons:
            self._buttons[key] = pressed
