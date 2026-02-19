"""Gait state machine for locomotion control.

This module provides a finite state machine for managing
gait transitions and phase timing.

Design by Contract:
    State transitions are deterministic and validated.
    Invalid transitions are rejected.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

from src.robotics.locomotion.gait_types import (
    GaitParameters,
    GaitPhase,
    GaitType,
    SupportState,
)
from src.shared.python.core.contracts import ContractChecker


class GaitEvent(Enum):
    """Events that trigger gait transitions."""

    STEP_COMPLETE = auto()  # Current step finished
    FOOT_CONTACT = auto()  # Foot touched ground
    FOOT_LIFTOFF = auto()  # Foot left ground
    BALANCE_LOST = auto()  # Balance criterion violated
    STOP_REQUESTED = auto()  # Stop command received
    START_REQUESTED = auto()  # Start walking command
    SPEED_CHANGE = auto()  # Velocity command changed
    EMERGENCY_STOP = auto()  # Emergency stop triggered


@dataclass
class GaitState:
    """Current state of the gait controller.

    Attributes:
        gait_type: Current gait type.
        phase: Current gait phase.
        support_state: Current support configuration.
        phase_time: Time elapsed in current phase [s].
        cycle_time: Time elapsed in current gait cycle [s].
        step_count: Number of steps taken.
        is_walking: Whether actively walking.
        stance_foot: Which foot is the stance foot ('left', 'right', 'both').
        next_stance_foot: Which foot will be stance after transition.
    """

    gait_type: GaitType = GaitType.STAND
    phase: GaitPhase = GaitPhase.DOUBLE_SUPPORT
    support_state: SupportState = SupportState.DOUBLE_SUPPORT_CENTERED
    phase_time: float = 0.0
    cycle_time: float = 0.0
    step_count: int = 0
    is_walking: bool = False
    stance_foot: str = "both"
    next_stance_foot: str = "left"


class GaitStateMachine(ContractChecker):
    """Finite state machine for gait management.

    Manages transitions between gait phases and ensures valid
    state sequences for bipedal locomotion.

    Design by Contract:
        Invariants:
            - State is always valid (phase and support state are enum members)
            - Phase time is non-negative
            - Step count is non-negative
            - Callbacks dict is never None

        Preconditions:
            - Events must be valid for current state

        Postconditions:
            - Transitions produce valid next state
            - Callbacks are invoked on transitions

    Example:
        >>> gait = GaitStateMachine()
        >>> gait.set_parameters(GaitParameters(step_duration=0.5))
        >>> gait.start_walking()
        >>> gait.update(0.01)  # Advance time
    """

    def __init__(
        self,
        parameters: GaitParameters | None = None,
    ) -> None:
        """Initialize gait state machine.

        Args:
            parameters: Gait parameters. Uses default walk if None.
        """
        self._parameters = parameters or GaitParameters()
        self._state = GaitState(gait_type=self._parameters.gait_type)
        self._callbacks: dict[str, list[Callable[[GaitState, GaitEvent], None]]] = {
            "phase_change": [],
            "step_complete": [],
            "gait_change": [],
        }

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Define class invariants for GaitStateMachine."""
        return [
            (
                lambda: isinstance(self._state.phase, GaitPhase),
                "Gait phase must be a valid GaitPhase enum member",
            ),
            (
                lambda: self._state.phase_time >= 0.0,
                "Phase time must be non-negative",
            ),
            (
                lambda: self._state.step_count >= 0,
                "Step count must be non-negative",
            ),
            (
                lambda: self._callbacks is not None
                and isinstance(self._callbacks, dict),
                "Callbacks must be a non-None dict",
            ),
            (
                lambda: self._state.stance_foot in ("left", "right", "both"),
                "Stance foot must be 'left', 'right', or 'both'",
            ),
        ]

    @property
    def state(self) -> GaitState:
        """Get current gait state (copy)."""
        return GaitState(
            gait_type=self._state.gait_type,
            phase=self._state.phase,
            support_state=self._state.support_state,
            phase_time=self._state.phase_time,
            cycle_time=self._state.cycle_time,
            step_count=self._state.step_count,
            is_walking=self._state.is_walking,
            stance_foot=self._state.stance_foot,
            next_stance_foot=self._state.next_stance_foot,
        )

    @property
    def parameters(self) -> GaitParameters:
        """Get gait parameters."""
        return self._parameters

    @property
    def is_walking(self) -> bool:
        """Check if currently walking."""
        return self._state.is_walking

    @property
    def phase(self) -> GaitPhase:
        """Get current gait phase."""
        return self._state.phase

    @property
    def phase_progress(self) -> float:
        """Get progress through current phase [0, 1]."""
        phase_duration = self._get_phase_duration(self._state.phase)
        if phase_duration <= 0:
            return 1.0
        return min(1.0, self._state.phase_time / phase_duration)

    def set_parameters(self, parameters: GaitParameters) -> None:
        """Set new gait parameters.

        Args:
            parameters: New gait parameters.
        """
        self._parameters = parameters
        self._state.gait_type = parameters.gait_type

    def start_walking(self) -> None:
        """Start walking from standing."""
        if self._state.is_walking:
            return

        self._state.is_walking = True
        self._state.cycle_time = 0.0
        self._state.phase_time = 0.0

        # Start with double support, prepare to lift right foot
        self._state.phase = GaitPhase.DOUBLE_SUPPORT
        self._state.support_state = SupportState.DOUBLE_SUPPORT_LEFT_LEADING
        self._state.stance_foot = "both"
        self._state.next_stance_foot = "left"

        self._invoke_callbacks("gait_change", GaitEvent.START_REQUESTED)

    def stop_walking(self) -> None:
        """Stop walking and return to standing."""
        if not self._state.is_walking:
            return

        # Wait for double support to stop
        if self._state.phase == GaitPhase.DOUBLE_SUPPORT:
            self._state.is_walking = False
            self._state.gait_type = GaitType.STAND
            self._state.support_state = SupportState.DOUBLE_SUPPORT_CENTERED
            self._state.stance_foot = "both"
            self._invoke_callbacks("gait_change", GaitEvent.STOP_REQUESTED)

    def emergency_stop(self) -> None:
        """Immediately stop all motion."""
        self._state.is_walking = False
        self._state.gait_type = GaitType.STAND
        self._state.phase = GaitPhase.DOUBLE_SUPPORT
        self._state.support_state = SupportState.DOUBLE_SUPPORT_CENTERED
        self._state.phase_time = 0.0
        self._invoke_callbacks("gait_change", GaitEvent.EMERGENCY_STOP)

    def update(self, dt: float) -> GaitState:
        """Update state machine by time step.

        Args:
            dt: Time step [s].

        Returns:
            Updated gait state.
        """
        if dt <= 0:
            return self.state

        self._state.phase_time += dt
        self._state.cycle_time += dt

        if self._state.is_walking:
            self._update_walking_state()

        return self.state

    def handle_event(self, event: GaitEvent) -> None:
        """Handle external event.

        Args:
            event: Event to process.
        """
        if event == GaitEvent.START_REQUESTED:
            self.start_walking()
        elif event == GaitEvent.STOP_REQUESTED:
            self.stop_walking()
        elif event == GaitEvent.EMERGENCY_STOP:
            self.emergency_stop()
        elif event == GaitEvent.FOOT_CONTACT:
            self._handle_foot_contact()
        elif event == GaitEvent.FOOT_LIFTOFF:
            self._handle_foot_liftoff()

    def register_callback(
        self,
        event_type: str,
        callback: Callable[[GaitState, GaitEvent], None],
    ) -> None:
        """Register callback for gait events.

        Args:
            event_type: Type of event ('phase_change', 'step_complete', 'gait_change').
            callback: Function to call with (state, event).
        """
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)

    def get_foot_trajectory_phase(self, foot: str) -> float:
        """Get trajectory phase for foot [0, 1].

        0 = start of swing, 1 = end of swing / stance.

        Args:
            foot: 'left' or 'right'.

        Returns:
            Phase value for trajectory interpolation.
        """
        if not self._state.is_walking:
            return 1.0  # Standing

        if foot == "left":
            if self._state.phase == GaitPhase.LEFT_SWING:
                return self.phase_progress
            return 1.0  # In stance
        else:
            if self._state.phase == GaitPhase.RIGHT_SWING:
                return self.phase_progress
            return 1.0  # In stance

    def _update_walking_state(self) -> None:
        """Update state during walking.

        Processes all phase transitions that should have occurred
        during the elapsed time.
        """
        max_iterations = 20  # Prevent infinite loops
        iterations = 0

        while iterations < max_iterations:
            phase_duration = self._get_phase_duration(self._state.phase)

            if self._state.phase_time >= phase_duration:
                # Store excess time to carry over
                excess_time = self._state.phase_time - phase_duration
                self._transition_phase()
                # Carry over excess time to new phase
                self._state.phase_time = excess_time
                iterations += 1
            else:
                break

    def _transition_phase(self) -> None:
        """Transition to next gait phase."""
        current_phase = self._state.phase
        next_phase: GaitPhase
        next_support: SupportState

        if current_phase == GaitPhase.DOUBLE_SUPPORT:
            # Transition to single support (swing phase)
            if self._state.next_stance_foot == "left":
                next_phase = GaitPhase.RIGHT_SWING
                next_support = SupportState.SINGLE_SUPPORT_LEFT
                self._state.stance_foot = "left"
            else:
                next_phase = GaitPhase.LEFT_SWING
                next_support = SupportState.SINGLE_SUPPORT_RIGHT
                self._state.stance_foot = "right"

        elif current_phase in (GaitPhase.LEFT_SWING, GaitPhase.RIGHT_SWING):
            # Transition back to double support
            next_phase = GaitPhase.DOUBLE_SUPPORT
            if current_phase == GaitPhase.LEFT_SWING:
                next_support = SupportState.DOUBLE_SUPPORT_LEFT_LEADING
                self._state.next_stance_foot = "right"
            else:
                next_support = SupportState.DOUBLE_SUPPORT_RIGHT_LEADING
                self._state.next_stance_foot = "left"
            self._state.stance_foot = "both"
            self._state.step_count += 1
            self._invoke_callbacks("step_complete", GaitEvent.STEP_COMPLETE)

        else:
            # Default to double support
            next_phase = GaitPhase.DOUBLE_SUPPORT
            next_support = SupportState.DOUBLE_SUPPORT_CENTERED

        self._state.phase = next_phase
        self._state.support_state = next_support
        self._state.phase_time = 0.0
        self._invoke_callbacks("phase_change", GaitEvent.STEP_COMPLETE)

    def _handle_foot_contact(self) -> None:
        """Handle foot contact event."""
        # Force transition to double support if in swing
        if self._state.phase in (GaitPhase.LEFT_SWING, GaitPhase.RIGHT_SWING):
            self._transition_phase()

    def _handle_foot_liftoff(self) -> None:
        """Handle foot liftoff event."""
        # Force transition to swing if in double support
        if self._state.phase == GaitPhase.DOUBLE_SUPPORT and self._state.is_walking:
            self._transition_phase()

    def _get_phase_duration(self, phase: GaitPhase) -> float:
        """Get duration for a given phase."""
        if phase == GaitPhase.DOUBLE_SUPPORT:
            return self._parameters.double_support_duration
        elif phase in (GaitPhase.LEFT_SWING, GaitPhase.RIGHT_SWING):
            return self._parameters.swing_duration
        else:
            return self._parameters.step_duration

    def _invoke_callbacks(self, event_type: str, event: GaitEvent) -> None:
        """Invoke registered callbacks."""
        for callback in self._callbacks.get(event_type, []):
            with contextlib.suppress(RuntimeError, ValueError, OSError):
                callback(self.state, event)
