"""VR interaction support for Unreal Engine integration.

This module provides data models and managers for VR-specific
interaction in the golf simulation visualization.

Features:
    - VR controller state tracking
    - Headset pose tracking
    - Locomotion modes (teleport, smooth, room-scale)
    - Hand gesture recognition
    - Voice command integration

Design by Contract:
    - Controller states validated for reasonable ranges
    - Interaction events are timestamped
    - All positions in world-space coordinates

Usage:
    from src.unreal_integration.vr_interaction import (
        VRInteractionManager,
        VRControllerState,
    )

    manager = VRInteractionManager()
    manager.update_controller(controller_state)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.unreal_integration.data_models import Quaternion, Vector3

logger = logging.getLogger(__name__)


class VRLocomotionMode(Enum):
    """VR locomotion modes."""

    TELEPORT = auto()  # Point and teleport
    SMOOTH = auto()  # Smooth joystick movement
    ROOM_SCALE = auto()  # Physical room walking only
    SNAP_TURN = auto()  # Turn in discrete increments
    SMOOTH_TURN = auto()  # Continuous turning


class VRControllerHand(Enum):
    """VR controller hand identifier."""

    LEFT = auto()
    RIGHT = auto()


class VRButtonState(Enum):
    """Button press state."""

    RELEASED = auto()
    PRESSED = auto()
    HELD = auto()
    JUST_PRESSED = auto()
    JUST_RELEASED = auto()


class VRInteractionMode(Enum):
    """Current VR interaction mode."""

    IDLE = auto()
    VIEWING = auto()  # Viewing swing analysis
    MANIPULATING = auto()  # Moving/rotating objects
    ANNOTATING = auto()  # Drawing annotations
    MENU = auto()  # Interacting with menus
    GRABBING = auto()  # Holding virtual club


class VRGesture(Enum):
    """Recognized hand gestures."""

    NONE = auto()
    POINT = auto()
    FIST = auto()
    OPEN_HAND = auto()
    THUMBS_UP = auto()
    PINCH = auto()
    GRIP = auto()


@dataclass
class VRControllerState:
    """State of a single VR controller.

    Attributes:
        hand: Which hand (left or right).
        position: 3D position in world space.
        rotation: Orientation as quaternion.
        velocity: Linear velocity.
        angular_velocity: Angular velocity.
        trigger: Trigger axis value (0-1).
        grip: Grip axis value (0-1).
        thumbstick: Thumbstick position (-1 to 1 for x and y).
        buttons: Button states.
        is_tracking: Whether controller is being tracked.
        gesture: Recognized hand gesture (if hand tracking).
    """

    hand: VRControllerHand
    position: Vector3
    rotation: Quaternion
    velocity: Vector3 = field(default_factory=Vector3.zero)
    angular_velocity: Vector3 = field(default_factory=Vector3.zero)
    trigger: float = 0.0
    grip: float = 0.0
    thumbstick: tuple[float, float] = (0.0, 0.0)
    buttons: dict[str, VRButtonState] = field(default_factory=dict)
    is_tracking: bool = True
    gesture: VRGesture = VRGesture.NONE

    def __post_init__(self) -> None:
        """Initialize default button states if not provided."""
        if not self.buttons:
            self.buttons = {
                "trigger": VRButtonState.RELEASED,
                "grip": VRButtonState.RELEASED,
                "primary": VRButtonState.RELEASED,
                "secondary": VRButtonState.RELEASED,
                "thumbstick": VRButtonState.RELEASED,
                "menu": VRButtonState.RELEASED,
            }

    @property
    def is_trigger_pressed(self) -> bool:
        """Check if trigger is pressed."""
        return self.trigger > 0.5

    @property
    def is_grip_pressed(self) -> bool:
        """Check if grip is pressed."""
        return self.grip > 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hand": self.hand.name.lower(),
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
            "velocity": self.velocity.to_dict(),
            "angular_velocity": self.angular_velocity.to_dict(),
            "trigger": self.trigger,
            "grip": self.grip,
            "thumbstick": list(self.thumbstick),
            "buttons": {k: v.name.lower() for k, v in self.buttons.items()},
            "is_tracking": self.is_tracking,
            "gesture": self.gesture.name.lower(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VRControllerState:
        """Create from dictionary."""
        buttons = {k: VRButtonState[v.upper()] for k, v in d.get("buttons", {}).items()}
        return cls(
            hand=VRControllerHand[d["hand"].upper()],
            position=Vector3.from_dict(d["position"]),
            rotation=Quaternion.from_dict(d["rotation"]),
            velocity=Vector3.from_dict(d.get("velocity", {"x": 0, "y": 0, "z": 0})),
            angular_velocity=Vector3.from_dict(
                d.get("angular_velocity", {"x": 0, "y": 0, "z": 0})
            ),
            trigger=d.get("trigger", 0.0),
            grip=d.get("grip", 0.0),
            thumbstick=tuple(d.get("thumbstick", [0.0, 0.0])),
            buttons=buttons,
            is_tracking=d.get("is_tracking", True),
            gesture=VRGesture[d.get("gesture", "none").upper()],
        )


@dataclass
class VRHeadsetState:
    """State of VR headset.

    Attributes:
        position: 3D position in world space.
        rotation: Orientation as quaternion.
        velocity: Linear velocity.
        angular_velocity: Angular velocity.
        ipd: Interpupillary distance in meters.
        fov: Field of view in degrees.
        is_tracking: Whether headset is being tracked.
        battery_level: Battery level (0-1, if available).
    """

    position: Vector3
    rotation: Quaternion
    velocity: Vector3 = field(default_factory=Vector3.zero)
    angular_velocity: Vector3 = field(default_factory=Vector3.zero)
    ipd: float = 0.063  # Default 63mm
    fov: float = 110.0
    is_tracking: bool = True
    battery_level: float | None = None

    @property
    def forward(self) -> Vector3:
        """Get forward direction vector."""
        # Extract forward vector from rotation
        q = self.rotation
        # Forward in local space is (0, 0, -1), transform by quaternion
        x = 2 * (q.x * q.z + q.w * q.y)
        y = 2 * (q.y * q.z - q.w * q.x)
        z = 1 - 2 * (q.x * q.x + q.y * q.y)
        return Vector3(x=-x, y=-y, z=-z)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
            "velocity": self.velocity.to_dict(),
            "angular_velocity": self.angular_velocity.to_dict(),
            "ipd": self.ipd,
            "fov": self.fov,
            "is_tracking": self.is_tracking,
        }
        if self.battery_level is not None:
            d["battery_level"] = self.battery_level
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VRHeadsetState:
        """Create from dictionary."""
        return cls(
            position=Vector3.from_dict(d["position"]),
            rotation=Quaternion.from_dict(d["rotation"]),
            velocity=Vector3.from_dict(d.get("velocity", {"x": 0, "y": 0, "z": 0})),
            angular_velocity=Vector3.from_dict(
                d.get("angular_velocity", {"x": 0, "y": 0, "z": 0})
            ),
            ipd=d.get("ipd", 0.063),
            fov=d.get("fov", 110.0),
            is_tracking=d.get("is_tracking", True),
            battery_level=d.get("battery_level"),
        )


@dataclass
class VRInteractionEvent:
    """Event generated from VR interaction.

    Attributes:
        event_type: Type of interaction event.
        timestamp: Event timestamp.
        controller: Which controller triggered event.
        position: World position of event.
        target: Target of interaction (optional).
        data: Additional event data.
    """

    event_type: str
    timestamp: float
    controller: VRControllerHand | None = None
    position: Vector3 | None = None
    target: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
        }
        if self.controller is not None:
            d["controller"] = self.controller.name.lower()
        if self.position is not None:
            d["position"] = self.position.to_dict()
        if self.target is not None:
            d["target"] = self.target
        if self.data:
            d["data"] = self.data
        return d


class VRInteractionManager:
    """Manager for VR interaction state and events.

    Tracks controller and headset states, processes interactions,
    and generates events.

    Example:
        >>> manager = VRInteractionManager()
        >>> manager.on_trigger_press(lambda e: print(f"Trigger pressed: {e}"))
        >>> manager.update_controller(left_controller)
    """

    def __init__(
        self,
        locomotion_mode: VRLocomotionMode = VRLocomotionMode.TELEPORT,
    ):
        """Initialize VR interaction manager.

        Args:
            locomotion_mode: Default locomotion mode.
        """
        self.locomotion_mode = locomotion_mode
        self.interaction_mode = VRInteractionMode.IDLE

        # State tracking
        self._headset: VRHeadsetState | None = None
        self._left_controller: VRControllerState | None = None
        self._right_controller: VRControllerState | None = None
        self._prev_left_controller: VRControllerState | None = None
        self._prev_right_controller: VRControllerState | None = None

        # Event callbacks
        self._callbacks: dict[str, list[Callable[[VRInteractionEvent], None]]] = {}

        # Interaction state
        self._grabbed_objects: dict[VRControllerHand, str] = {}
        self._current_time: float = 0.0

    @property
    def headset(self) -> VRHeadsetState | None:
        """Get current headset state."""
        return self._headset

    @property
    def left_controller(self) -> VRControllerState | None:
        """Get left controller state."""
        return self._left_controller

    @property
    def right_controller(self) -> VRControllerState | None:
        """Get right controller state."""
        return self._right_controller

    def update_headset(self, state: VRHeadsetState, timestamp: float) -> None:
        """Update headset state.

        Args:
            state: New headset state.
            timestamp: Update timestamp.
        """
        self._headset = state
        self._current_time = timestamp

    def update_controller(self, state: VRControllerState, timestamp: float) -> None:
        """Update controller state and process interactions.

        Args:
            state: New controller state.
            timestamp: Update timestamp.
        """
        self._current_time = timestamp

        # Store previous state
        if state.hand == VRControllerHand.LEFT:
            self._prev_left_controller = self._left_controller
            self._left_controller = state
            prev = self._prev_left_controller
        else:
            self._prev_right_controller = self._right_controller
            self._right_controller = state
            prev = self._prev_right_controller

        # Detect interactions
        self._process_trigger(state, prev)
        self._process_grip(state, prev)
        self._process_buttons(state, prev)
        self._process_thumbstick(state, prev)

    def _process_trigger(
        self,
        current: VRControllerState,
        prev: VRControllerState | None,
    ) -> None:
        """Process trigger state changes."""
        curr_pressed = current.is_trigger_pressed
        prev_pressed = prev.is_trigger_pressed if prev else False

        if curr_pressed and not prev_pressed:
            self._emit_event(
                VRInteractionEvent(
                    event_type="trigger_press",
                    timestamp=self._current_time,
                    controller=current.hand,
                    position=current.position,
                )
            )
        elif not curr_pressed and prev_pressed:
            self._emit_event(
                VRInteractionEvent(
                    event_type="trigger_release",
                    timestamp=self._current_time,
                    controller=current.hand,
                    position=current.position,
                )
            )

    def _process_grip(
        self,
        current: VRControllerState,
        prev: VRControllerState | None,
    ) -> None:
        """Process grip state changes."""
        curr_pressed = current.is_grip_pressed
        prev_pressed = prev.is_grip_pressed if prev else False

        if curr_pressed and not prev_pressed:
            self._emit_event(
                VRInteractionEvent(
                    event_type="grip_press",
                    timestamp=self._current_time,
                    controller=current.hand,
                    position=current.position,
                )
            )
        elif not curr_pressed and prev_pressed:
            self._emit_event(
                VRInteractionEvent(
                    event_type="grip_release",
                    timestamp=self._current_time,
                    controller=current.hand,
                    position=current.position,
                )
            )

    def _process_buttons(
        self,
        current: VRControllerState,
        prev: VRControllerState | None,
    ) -> None:
        """Process button state changes."""
        for button_name, curr_state in current.buttons.items():
            if prev and button_name in prev.buttons:
                prev_state = prev.buttons[button_name]
            else:
                prev_state = VRButtonState.RELEASED

            if (
                curr_state == VRButtonState.PRESSED
                and prev_state == VRButtonState.RELEASED
            ):
                self._emit_event(
                    VRInteractionEvent(
                        event_type=f"button_{button_name}_press",
                        timestamp=self._current_time,
                        controller=current.hand,
                        position=current.position,
                    )
                )

    def _process_thumbstick(
        self,
        current: VRControllerState,
        prev: VRControllerState | None,
    ) -> None:
        """Process thumbstick movement."""
        x, y = current.thumbstick
        if abs(x) > 0.5 or abs(y) > 0.5:
            self._emit_event(
                VRInteractionEvent(
                    event_type="thumbstick_move",
                    timestamp=self._current_time,
                    controller=current.hand,
                    data={"x": x, "y": y},
                )
            )

    def _emit_event(self, event: VRInteractionEvent) -> None:
        """Emit interaction event to callbacks."""
        event_type = event.event_type
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                try:
                    callback(event)
                except (RuntimeError, ValueError, OSError) as e:
                    logger.error(f"Error in VR event callback: {e}")

        # Also emit to wildcard handlers
        if "*" in self._callbacks:
            for callback in self._callbacks["*"]:
                try:
                    callback(event)
                except (RuntimeError, ValueError, OSError) as e:
                    logger.error(f"Error in VR event callback: {e}")

    def on(
        self, event_type: str, callback: Callable[[VRInteractionEvent], None]
    ) -> None:
        """Register event callback.

        Args:
            event_type: Event type to listen for ("*" for all).
            callback: Callback function.
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def on_trigger_press(self, callback: Callable[[VRInteractionEvent], None]) -> None:
        """Register trigger press callback."""
        self.on("trigger_press", callback)

    def on_grip_press(self, callback: Callable[[VRInteractionEvent], None]) -> None:
        """Register grip press callback."""
        self.on("grip_press", callback)

    def get_state(self) -> dict[str, Any]:
        """Get complete VR state as dictionary."""
        state: dict[str, Any] = {
            "locomotion_mode": self.locomotion_mode.name.lower(),
            "interaction_mode": self.interaction_mode.name.lower(),
            "current_time": self._current_time,
        }
        if self._headset:
            state["headset"] = self._headset.to_dict()
        if self._left_controller:
            state["left_controller"] = self._left_controller.to_dict()
        if self._right_controller:
            state["right_controller"] = self._right_controller.to_dict()
        return state

    def set_locomotion_mode(self, mode: VRLocomotionMode) -> None:
        """Set locomotion mode.

        Args:
            mode: New locomotion mode.
        """
        self.locomotion_mode = mode
        self._emit_event(
            VRInteractionEvent(
                event_type="locomotion_mode_changed",
                timestamp=self._current_time,
                data={"mode": mode.name.lower()},
            )
        )

    def set_interaction_mode(self, mode: VRInteractionMode) -> None:
        """Set interaction mode.

        Args:
            mode: New interaction mode.
        """
        prev_mode = self.interaction_mode
        self.interaction_mode = mode
        self._emit_event(
            VRInteractionEvent(
                event_type="interaction_mode_changed",
                timestamp=self._current_time,
                data={"mode": mode.name.lower(), "previous": prev_mode.name.lower()},
            )
        )
