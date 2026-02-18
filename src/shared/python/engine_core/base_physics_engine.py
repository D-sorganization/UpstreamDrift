"""Base physics engine implementation with common functionality.

This module provides a base class that implements common patterns shared
across all physics engine implementations, eliminating code duplication.

Design by Contract:
    This module enforces DbC principles through:
    - Preconditions: State requirements before method execution
    - Postconditions: Guarantees after method completion
    - Invariants: Properties that must always hold

Usage:
    from src.shared.python.engine_core.base_physics_engine import BasePhysicsEngine

    class MyPhysicsEngine(BasePhysicsEngine):
        def _load_from_path_impl(self, path: str) -> None:
            # Engine-specific loading logic
            pass

        def _load_from_string_impl(self, content: str, extension: str | None) -> None:
            # Engine-specific loading logic
            pass
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.contracts import (
    ContractChecker,
    StateError,
    invariant_checked,
    require_state,
)
from src.shared.python.core.error_decorators import ErrorContext, log_errors
from src.shared.python.engine_core.checkpoint import StateCheckpoint
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.security.security_utils import validate_path

logger = get_logger(__name__)


class EngineState:
    """Common state representation for physics engines."""

    def __init__(self, nq: int = 0, nv: int = 0) -> None:
        """Initialize engine state.

        Args:
            nq: Number of position coordinates
            nv: Number of velocity coordinates
        """
        self.q: np.ndarray = np.zeros(nq)  # Positions
        self.v: np.ndarray = np.zeros(nv)  # Velocities
        self.a: np.ndarray = np.zeros(nv)  # Accelerations
        self.tau: np.ndarray = np.zeros(nv)  # Torques/forces
        self.time: float = 0.0

    def reset(self) -> None:
        """Reset state to zeros."""
        self.q.fill(0.0)
        self.v.fill(0.0)
        self.a.fill(0.0)
        self.tau.fill(0.0)
        self.time = 0.0


class BasePhysicsEngine(ContractChecker, PhysicsEngine):
    """Base class for physics engines with common functionality.

    This class implements common patterns:
    - Error handling for model loading
    - Path validation
    - State management
    - Checkpoint save/restore with protocol-compatible get_state/set_state
    - Model name tracking
    - Logging
    - Design by Contract enforcement

    Design by Contract:
        Preconditions:
            - load_from_path: path must exist and be in allowed directories
            - load_from_string: content must be non-empty
            - step/forward/reset: engine must be initialized (model loaded)

        Postconditions:
            - load_from_path/load_from_string: _is_initialized becomes True
            - get_state: returns valid (q, v) arrays

        Invariants:
            - If _is_initialized, then model is not None
            - If state exists, q and v have matching dimensions

    Subclasses must implement:
    - _load_from_path_impl()
    - _load_from_string_impl()
    - step()
    - reset()

    Subclasses may optionally override:
    - _get_extra_checkpoint_state(): return engine-specific data
    - _restore_extra_checkpoint_state(): restore engine-specific data
    """

    def __init__(self, allowed_dirs: list[Path] | None = None) -> None:
        """Initialize base physics engine.

        Args:
            allowed_dirs: List of allowed directories for model loading
        """
        self.model: Any = None
        self.data: Any = None
        self.model_path: str | None = None
        self.model_name_str: str = ""
        self.state: EngineState | None = None
        self.allowed_dirs = allowed_dirs or []
        self._is_initialized = False

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Define class invariants for BasePhysicsEngine.

        Returns:
            List of (condition, message) tuples defining invariants.
        """
        return [
            (
                lambda: not self._is_initialized or self.model is not None,
                "Initialized engine must have a loaded model",
            ),
            (
                lambda: self.state is None
                or (
                    len(self.state.q) == len(self.state.v)
                    or len(self.state.q) == len(self.state.v) + 1  # Quaternion case
                ),
                "State arrays q and v must have compatible dimensions",
            ),
            (
                lambda: self.state is None or self.state.time >= 0.0,
                "Simulation time must be non-negative",
            ),
        ]

    @property
    def model_name(self) -> str:
        """Return the name of the currently loaded model."""
        if self.model is not None and hasattr(self.model, "name"):
            return str(self.model.name)
        return self.model_name_str

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized with a model.

        Returns:
            True if a model has been successfully loaded.
        """
        return self._is_initialized

    @log_errors("Failed to load model from path", reraise=True)
    @invariant_checked
    def load_from_path(self, path: str) -> None:
        """Load model from file path with validation and error handling.

        Preconditions:
            - path must point to an existing file
            - path must be in allowed directories (if configured)

        Postconditions:
            - self._is_initialized == True
            - self.model is not None
            - self.model_name returns valid string

        Args:
            path: Path to model file

        Raises:
            FileNotFoundError: If path does not exist
            ValueError: If path is not in allowed directories
            Exception: If model loading fails
        """
        # Validate path exists (Precondition)
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Validate path is in allowed directories (Precondition)
        if self.allowed_dirs:
            validated_path = validate_path(path, self.allowed_dirs, strict=True)
            path = str(validated_path)

        # Extract model name from path
        self.model_name_str = path_obj.stem

        # Call engine-specific implementation
        with ErrorContext(f"Loading model from {path}"):
            self._load_from_path_impl(path)
            self.model_path = path
            self._is_initialized = True

        # Verify postconditions
        assert self._is_initialized, (
            "Postcondition: engine must be initialized after load"
        )
        assert self.model is not None, "Postcondition: model must be loaded"

        logger.info(f"Successfully loaded model: {self.model_name}")

    @log_errors("Failed to load model from string", reraise=True)
    @invariant_checked
    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load model from string content with error handling.

        Preconditions:
            - content must be non-empty

        Postconditions:
            - self._is_initialized == True
            - self.model is not None

        Args:
            content: Model content as string
            extension: File extension hint (e.g., "urdf", "xml")

        Raises:
            ValueError: If content is empty (precondition violation)
            Exception: If model loading fails
        """
        # Precondition: content must be non-empty
        if not content or not content.strip():
            raise ValueError("Precondition violated: Model content cannot be empty")

        self.model_name_str = "StringLoadedModel"

        # Call engine-specific implementation
        with ErrorContext("Loading model from string"):
            self._load_from_string_impl(content, extension)
            self.model_path = None
            self._is_initialized = True

        # Verify postconditions
        assert self._is_initialized, (
            "Postcondition: engine must be initialized after load"
        )
        assert self.model is not None, "Postcondition: model must be loaded"

        logger.info("Successfully loaded model from string")

    @abstractmethod
    def _load_from_path_impl(self, path: str) -> None:
        """Engine-specific implementation of load_from_path.

        Args:
            path: Validated path to model file
        """

    @abstractmethod
    def _load_from_string_impl(self, content: str, extension: str | None) -> None:
        """Engine-specific implementation of load_from_string.

        Args:
            content: Model content as string
            extension: File extension hint
        """

    def get_model(self) -> Any:
        """Get the underlying model object.

        Returns:
            The engine-specific model object, or None if not loaded.
        """
        return self.model

    def get_data(self) -> Any:
        """Get the underlying data object.

        Returns:
            The engine-specific data object, or None if not loaded.
        """
        return self.data

    def require_initialized(self, operation: str = "this operation") -> None:
        """Verify engine is initialized, raising StateError if not.

        Args:
            operation: Description of the operation being attempted.

        Raises:
            StateError: If engine is not initialized.
        """
        if not self._is_initialized:
            raise StateError(
                f"Cannot perform '{operation}' - engine not initialized. "
                "Call load_from_path() or load_from_string() first.",
                current_state="uninitialized",
                required_state="initialized",
                operation=operation,
            )

    @require_state(lambda self: self._is_initialized, "initialized")
    def get_state(  # type: ignore[override]
        self,
    ) -> EngineState | tuple[np.ndarray, np.ndarray] | None:
        """Get current engine state.

        Preconditions:
            - Engine must be initialized

        Returns:
            Current EngineState object if using EngineState-based state,
            or tuple of (q, v) if overridden by subclass.
        """
        return self.state

    @require_state(lambda self: self._is_initialized, "initialized")
    @invariant_checked
    def set_state(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ) -> None:
        """Set engine state.

        Preconditions:
            - Engine must be initialized

        Accepts either:
            - set_state(engine_state) for EngineState-based management
            - set_state(q, v) when overridden by subclass

        Args:
            *args: EngineState or (q, v) arrays depending on subclass.
            **kwargs: Optional keyword arguments.
        """
        if len(args) == 1 and isinstance(args[0], EngineState):
            self.state = args[0]
        elif len(args) == 2:
            # Protocol-compatible (q, v) form
            if self.state is not None:
                self.state.q = np.asarray(args[0]).copy()
                self.state.v = np.asarray(args[1]).copy()
        else:
            raise TypeError(
                f"set_state expects EngineState or (q, v), got {len(args)} args"
            )

    def get_time(self) -> float:
        """Get current simulation time.

        Returns:
            Current simulation time in seconds.
        """
        if self.state is not None:
            return self.state.time
        return 0.0

    @require_state(lambda self: self._is_initialized, "initialized")
    def save_checkpoint(self) -> StateCheckpoint:
        """Save current engine state to a checkpoint.

        Uses get_state() and get_time() for protocol-compatible engines.
        Subclasses can add engine-specific data via
        _get_extra_checkpoint_state().

        Returns:
            StateCheckpoint object containing engine state.
        """
        engine_state_dict: dict[str, Any] = {}
        timestamp = 0.0
        q = np.array([])
        v = np.array([])

        if self.state:
            # Legacy EngineState-based path
            timestamp = self.state.time
            q = self.state.q.copy()
            v = self.state.v.copy()
            engine_state_dict = {
                "q": q,
                "v": v,
                "a": self.state.a.copy(),
                "tau": self.state.tau.copy(),
                "t": self.state.time,
            }
        else:
            # Protocol-compatible path: use get_state() and get_time()
            try:
                timestamp = self.get_time()
                state_result = self.get_state()
                if isinstance(state_result, tuple) and len(state_result) == 2:
                    q = np.asarray(state_result[0]).copy()
                    v = np.asarray(state_result[1]).copy()
                engine_state_dict = {
                    "q": q,
                    "v": v,
                    "t": timestamp,
                }
            except (NotImplementedError, AttributeError):
                pass

        # Allow subclasses to add engine-specific checkpoint data
        extra = self._get_extra_checkpoint_state()
        if extra:
            engine_state_dict.update(extra)

        return StateCheckpoint.create(
            engine_type=self.__class__.__name__,
            engine_state=engine_state_dict,
            q=q,
            v=v,
            timestamp=timestamp,
        )

    def _get_extra_checkpoint_state(self) -> dict[str, Any]:
        """Return engine-specific data to include in checkpoints.

        Override in subclasses to store additional state beyond q, v, t.

        Returns:
            Dictionary of additional state data.
        """
        return {}

    @require_state(lambda self: self._is_initialized, "initialized")
    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Restore engine state from a checkpoint.

        Uses set_state(q, v) for protocol-compatible engines.
        Subclasses can restore engine-specific state via
        _restore_extra_checkpoint_state().

        Args:
            checkpoint: Checkpoint to restore from.
        """
        if not checkpoint.engine_state:
            return

        data = checkpoint.engine_state
        q = data.get("q")
        v = data.get("v")

        if q is not None and v is not None:
            q_arr = np.asarray(q)
            v_arr = np.asarray(v)

            if self.state is not None:
                # Legacy EngineState-based path
                nq = len(q_arr)
                nv = len(v_arr)
                new_state = EngineState(nq, nv)
                new_state.q = q_arr.copy()
                new_state.v = v_arr.copy()
                new_state.time = data.get("t", checkpoint.timestamp)

                if "a" in data:
                    new_state.a = np.array(data["a"])
                if "tau" in data:
                    new_state.tau = np.array(data["tau"])

                self.set_state(new_state)
            else:
                # Protocol-compatible path: use set_state(q, v)
                self.set_state(q_arr, v_arr)

        # Allow subclasses to restore engine-specific state
        self._restore_extra_checkpoint_state(checkpoint)

    def _restore_extra_checkpoint_state(self, checkpoint: StateCheckpoint) -> None:
        """Restore engine-specific state from a checkpoint.

        Override in subclasses to restore additional state.

        Args:
            checkpoint: Checkpoint with engine-specific state data.
        """

    def __repr__(self) -> str:
        """String representation of engine."""
        status = "initialized" if self._is_initialized else "uninitialized"
        return f"{self.__class__.__name__}(model={self.model_name}, status={status})"


class ModelLoadingMixin:
    """Mixin providing common model loading utilities."""

    @staticmethod
    def validate_file_extension(path: str, allowed_extensions: list[str]) -> None:
        """Validate file has an allowed extension.

        Args:
            path: Path to file
            allowed_extensions: List of allowed extensions

        Raises:
            ValueError: If extension is not allowed
        """
        path_obj = Path(path)
        if path_obj.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"Invalid file extension: {path_obj.suffix}. "
                f"Allowed: {', '.join(allowed_extensions)}"
            )

    @staticmethod
    def extract_model_name(path: str) -> str:
        """Extract model name from file path.

        Args:
            path: Path to model file

        Returns:
            Model name (filename without extension)
        """
        return Path(path).stem


class SimulationMixin:
    """Mixin providing common simulation utilities."""

    def __init__(self) -> None:
        """Initialize simulation state."""
        self._simulation_time: float = 0.0
        self._step_count: int = 0

    def get_simulation_time(self) -> float:
        """Get current simulation time."""
        return self._simulation_time

    def get_step_count(self) -> int:
        """Get number of simulation steps taken."""
        return self._step_count

    def reset_simulation_time(self) -> None:
        """Reset simulation time and step count."""
        self._simulation_time = 0.0
        self._step_count = 0

    def _update_simulation_time(self, dt: float) -> None:
        """Update simulation time after a step.

        Args:
            dt: Time step size
        """
        self._simulation_time += dt
        self._step_count += 1
