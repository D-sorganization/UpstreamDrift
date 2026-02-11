"""State Management Utilities for Physics Engines.

This module provides common state management patterns shared across
all physics engine implementations.

Includes:
    - StateManager: Centralized state tracking
    - EngineStateMixin: Mixin for engine state lifecycle
    - ForceAccumulator: Accumulate forces from multiple sources

Design by Contract:
    - State transitions follow documented state machine
    - All state modifications preserve invariants
    - Time is always non-negative
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TypeVar

import numpy as np

from src.shared.python.core.contracts import StateError
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class EngineLifecycleState(Enum):
    """Engine lifecycle states.

    State Machine:
        UNINITIALIZED -> [load] -> INITIALIZED
        INITIALIZED -> [reset] -> INITIALIZED (t=0)
        INITIALIZED -> [step] -> STEPPING -> INITIALIZED
        Any -> [error] -> ERROR
        ERROR -> [reset/reload] -> INITIALIZED
    """

    UNINITIALIZED = auto()
    LOADING = auto()
    INITIALIZED = auto()
    STEPPING = auto()
    PAUSED = auto()
    ERROR = auto()


@dataclass
class SimulationState:
    """Complete simulation state for checkpointing.

    This dataclass captures all state needed to restore a simulation
    to a previous point in time.

    Attributes:
        q: Generalized positions
        v: Generalized velocities
        a: Generalized accelerations
        tau: Applied forces/torques
        time: Simulation time
        step_count: Number of steps taken
        metadata: Engine-specific additional state
    """

    q: np.ndarray
    v: np.ndarray
    a: np.ndarray = field(default_factory=lambda: np.array([]))
    tau: np.ndarray = field(default_factory=lambda: np.array([]))
    time: float = 0.0
    step_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> SimulationState:
        """Create a deep copy of the state."""
        return SimulationState(
            q=self.q.copy(),
            v=self.v.copy(),
            a=self.a.copy(),
            tau=self.tau.copy(),
            time=self.time,
            step_count=self.step_count,
            metadata=dict(self.metadata),
        )

    def validate(self) -> bool:
        """Validate state consistency.

        Returns:
            True if state is valid
        """
        # All arrays should be finite
        if not np.all(np.isfinite(self.q)):
            return False
        if not np.all(np.isfinite(self.v)):
            return False
        if len(self.a) > 0 and not np.all(np.isfinite(self.a)):
            return False
        if len(self.tau) > 0 and not np.all(np.isfinite(self.tau)):
            return False
        if self.time < 0:
            return False
        return True


class StateManager:
    """Centralized state management for physics engines.

    Provides:
        - State storage and retrieval
        - State validation
        - State transition tracking
        - Undo/redo buffer

    Example:
        >>> manager = StateManager(nq=7, nv=6)
        >>> manager.set_state(q, v)
        >>> manager.step(dt)
        >>> q, v = manager.get_state()
    """

    def __init__(
        self,
        nq: int,
        nv: int,
        max_history: int = 100,
    ) -> None:
        """Initialize state manager.

        Args:
            nq: Number of position coordinates
            nv: Number of velocity coordinates
            max_history: Maximum history for undo buffer
        """
        self.nq = nq
        self.nv = nv
        self.max_history = max_history

        # Current state
        self._state = SimulationState(
            q=np.zeros(nq),
            v=np.zeros(nv),
            a=np.zeros(nv),
            tau=np.zeros(nv),
        )

        # Lifecycle state
        self._lifecycle = EngineLifecycleState.UNINITIALIZED

        # History buffer (circular)
        self._history: list[SimulationState] = []
        self._history_index: int = -1

    @property
    def state(self) -> SimulationState:
        """Get current state."""
        return self._state

    @property
    def lifecycle(self) -> EngineLifecycleState:
        """Get lifecycle state."""
        return self._lifecycle

    def initialize(self, q0: np.ndarray | None = None) -> None:
        """Initialize state manager with initial configuration.

        Args:
            q0: Initial positions (defaults to zeros)
        """
        self._state = SimulationState(
            q=q0.copy() if q0 is not None else np.zeros(self.nq),
            v=np.zeros(self.nv),
            a=np.zeros(self.nv),
            tau=np.zeros(self.nv),
            time=0.0,
            step_count=0,
        )
        self._lifecycle = EngineLifecycleState.INITIALIZED
        self._save_history()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set current state.

        Args:
            q: Positions
            v: Velocities

        Raises:
            ValueError: If dimensions don't match
        """
        if len(q) != self.nq:
            raise ValueError(f"q dimension mismatch: {len(q)} != {self.nq}")
        if len(v) != self.nv:
            raise ValueError(f"v dimension mismatch: {len(v)} != {self.nv}")

        self._state.q = q.copy()
        self._state.v = v.copy()

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current positions and velocities.

        Returns:
            Tuple of (q, v) arrays
        """
        return self._state.q.copy(), self._state.v.copy()

    def advance_time(self, dt: float) -> None:
        """Advance simulation time.

        Args:
            dt: Time step
        """
        self._state.time += dt
        self._state.step_count += 1

    def reset(self) -> None:
        """Reset to initial state."""
        self._state = SimulationState(
            q=np.zeros(self.nq),
            v=np.zeros(self.nv),
            a=np.zeros(self.nv),
            tau=np.zeros(self.nv),
            time=0.0,
            step_count=0,
        )
        self._history.clear()
        self._history_index = -1

    def _save_history(self) -> None:
        """Save current state to history."""
        if len(self._history) >= self.max_history:
            self._history.pop(0)
        self._history.append(self._state.copy())
        self._history_index = len(self._history) - 1

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._history_index > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._history_index < len(self._history) - 1

    def undo(self) -> bool:
        """Undo to previous state.

        Returns:
            True if undo was successful
        """
        if not self.can_undo():
            return False
        self._history_index -= 1
        self._state = self._history[self._history_index].copy()
        return True

    def redo(self) -> bool:
        """Redo to next state.

        Returns:
            True if redo was successful
        """
        if not self.can_redo():
            return False
        self._history_index += 1
        self._state = self._history[self._history_index].copy()
        return True


class EngineStateMixin:
    """Mixin providing engine state lifecycle management.

    Add this mixin to engine classes to get standardized state
    tracking and lifecycle management.

    Example:
        class MyEngine(EngineStateMixin, PhysicsEngine):
            def load_from_path(self, path: str) -> None:
                self._set_lifecycle(EngineLifecycleState.LOADING)
                # ... load model ...
                self._set_lifecycle(EngineLifecycleState.INITIALIZED)
    """

    def __init__(self) -> None:
        """Initialize mixin state."""
        self._lifecycle_state: EngineLifecycleState = EngineLifecycleState.UNINITIALIZED
        self._lifecycle_callbacks: list[Callable[[EngineLifecycleState], None]] = []

    def _get_lifecycle(self) -> EngineLifecycleState:
        """Get current lifecycle state."""
        return self._lifecycle_state

    def _set_lifecycle(self, state: EngineLifecycleState) -> None:
        """Set lifecycle state and notify callbacks.

        Args:
            state: New lifecycle state
        """
        old_state = self._lifecycle_state
        self._lifecycle_state = state

        logger.debug(f"Engine lifecycle: {old_state.name} -> {state.name}")

        for callback in self._lifecycle_callbacks:
            try:
                callback(state)
            except (RuntimeError, ValueError, OSError) as e:
                logger.error(f"Lifecycle callback error: {e}")

    def _require_lifecycle(
        self,
        *required_states: EngineLifecycleState,
        operation: str = "operation",
    ) -> None:
        """Require engine to be in one of the specified states.

        Args:
            required_states: Allowed states for this operation
            operation: Description of the operation (for error message)

        Raises:
            StateError: If engine is not in a required state
        """
        if self._lifecycle_state not in required_states:
            required_names = ", ".join([s.name for s in required_states])
            raise StateError(
                f"Cannot perform '{operation}' in state {self._lifecycle_state.name}. "
                f"Required: {required_names}",
                current_state=self._lifecycle_state.name,
                required_state=required_names,
                operation=operation,
            )

    def add_lifecycle_callback(
        self,
        callback: Callable[[EngineLifecycleState], None],
    ) -> None:
        """Add callback for lifecycle state changes.

        Args:
            callback: Function called with new state on transitions
        """
        self._lifecycle_callbacks.append(callback)

    def remove_lifecycle_callback(
        self,
        callback: Callable[[EngineLifecycleState], None],
    ) -> None:
        """Remove lifecycle callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._lifecycle_callbacks:
            self._lifecycle_callbacks.remove(callback)


@dataclass
class ForceSource:
    """A named source of force/torque.

    Attributes:
        name: Source identifier
        force: Force vector [N]
        torque: Torque vector [N·m]
        category: Force category for grouping
    """

    name: str
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    category: str = "external"


class ForceAccumulator:
    """Accumulate forces from multiple sources for analysis.

    This class collects forces from different sources (gravity, drag,
    contact, muscles) for induced acceleration analysis.

    Example:
        >>> accumulator = ForceAccumulator(nv=6)
        >>> accumulator.add_force("gravity", F_gravity)
        >>> accumulator.add_force("drag", F_drag)
        >>> total = accumulator.get_total_force()
        >>> by_source = accumulator.get_forces_by_source()
    """

    def __init__(self, nv: int) -> None:
        """Initialize force accumulator.

        Args:
            nv: Number of generalized velocity coordinates
        """
        self.nv = nv
        self._sources: dict[str, ForceSource] = {}
        self._generalized_forces: dict[str, np.ndarray] = {}

    def clear(self) -> None:
        """Clear all accumulated forces."""
        self._sources.clear()
        self._generalized_forces.clear()

    def add_force(
        self,
        name: str,
        force: np.ndarray,
        torque: np.ndarray | None = None,
        category: str = "external",
    ) -> None:
        """Add a force source.

        Args:
            name: Source identifier
            force: Force vector [N]
            torque: Torque vector [N·m] (optional)
            category: Force category
        """
        self._sources[name] = ForceSource(
            name=name,
            force=force.copy(),
            torque=torque.copy() if torque is not None else np.zeros(3),
            category=category,
        )

    def add_generalized_force(
        self,
        name: str,
        tau: np.ndarray,
        category: str = "actuator",
    ) -> None:
        """Add a generalized force in joint space.

        Args:
            name: Source identifier
            tau: Generalized force vector (nv,)
            category: Force category
        """
        if len(tau) != self.nv:
            raise ValueError(f"tau dimension mismatch: {len(tau)} != {self.nv}")
        self._generalized_forces[name] = tau.copy()

    def get_total_force(self) -> np.ndarray:
        """Get total Cartesian force.

        Returns:
            Sum of all force vectors [N]
        """
        if not self._sources:
            return np.zeros(3)
        # Use np.sum for proper numpy array summation to avoid type ambiguity with sum()
        forces = [s.force for s in self._sources.values()]
        return np.sum(forces, axis=0)

    def get_total_torque(self) -> np.ndarray:
        """Get total Cartesian torque.

        Returns:
            Sum of all torque vectors [N·m]
        """
        if not self._sources:
            return np.zeros(3)
        # Use np.sum for proper numpy array summation
        torques = [s.torque for s in self._sources.values()]
        return np.sum(torques, axis=0)

    def get_total_generalized_force(self) -> np.ndarray:
        """Get total generalized force.

        Returns:
            Sum of all generalized forces (nv,)
        """
        if not self._generalized_forces:
            return np.zeros(self.nv)
        # Use np.sum for proper numpy array summation
        forces = list(self._generalized_forces.values())
        return np.sum(forces, axis=0)

    def get_forces_by_source(self) -> dict[str, ForceSource]:
        """Get all force sources.

        Returns:
            Dictionary of source name to ForceSource
        """
        return dict(self._sources)

    def get_forces_by_category(self) -> dict[str, list[ForceSource]]:
        """Get forces grouped by category.

        Returns:
            Dictionary of category to list of ForceSource
        """
        by_category: dict[str, list[ForceSource]] = {}
        for source in self._sources.values():
            if source.category not in by_category:
                by_category[source.category] = []
            by_category[source.category].append(source)
        return by_category

    def get_source_names(self) -> list[str]:
        """Get list of all source names.

        Returns:
            List of source identifiers
        """
        return list(self._sources.keys()) + list(self._generalized_forces.keys())
