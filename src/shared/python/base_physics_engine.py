"""Base physics engine implementation with common functionality.

This module provides a base class that implements common patterns shared
across all physics engine implementations, eliminating code duplication.

Usage:
    from src.shared.python.base_physics_engine import BasePhysicsEngine

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
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.error_decorators import ErrorContext, log_errors
from src.shared.python.interfaces import PhysicsEngine
from src.shared.python.logging_config import get_logger
from src.shared.python.security_utils import validate_path

logger = get_logger(__name__)


class EngineState:
    """Common state representation for physics engines."""

    def __init__(self, nq: int = 0, nv: int = 0):
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


class BasePhysicsEngine(PhysicsEngine):
    """Base class for physics engines with common functionality.

    This class implements common patterns:
    - Error handling for model loading
    - Path validation
    - State management
    - Model name tracking
    - Logging

    Subclasses must implement:
    - _load_from_path_impl()
    - _load_from_string_impl()
    - step()
    - reset()
    """

    def __init__(self, allowed_dirs: list[Path] | None = None):
        """Initialize base physics engine.

        Args:
            allowed_dirs: List of allowed directories for model loading (security)
        """
        self.model: Any = None
        self.data: Any = None
        self.model_path: str | None = None
        self.model_name_str: str = ""
        self.state: EngineState | None = None
        self.allowed_dirs = allowed_dirs or []
        self._is_initialized = False

    @property
    def model_name(self) -> str:
        """Return the name of the currently loaded model."""
        if self.model is not None and hasattr(self.model, "name"):
            return str(self.model.name)
        return self.model_name_str

    @log_errors("Failed to load model from path", reraise=True)
    def load_from_path(self, path: str) -> None:
        """Load model from file path with validation and error handling.

        Args:
            path: Path to model file

        Raises:
            FileNotFoundError: If path does not exist
            ValueError: If path is not in allowed directories
            Exception: If model loading fails
        """
        # Validate path exists
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Validate path is in allowed directories (if specified)
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

        logger.info(f"Successfully loaded model: {self.model_name}")

    @log_errors("Failed to load model from string", reraise=True)
    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load model from string content with error handling.

        Args:
            content: Model content as string
            extension: File extension hint (e.g., "urdf", "xml")

        Raises:
            ValueError: If content is empty
            Exception: If model loading fails
        """
        if not content or not content.strip():
            raise ValueError("Model content cannot be empty")

        self.model_name_str = "StringLoadedModel"

        # Call engine-specific implementation
        with ErrorContext("Loading model from string"):
            self._load_from_string_impl(content, extension)
            self.model_path = None
            self._is_initialized = True

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
        """Get the underlying model object."""
        return self.model

    def get_data(self) -> Any:
        """Get the underlying data object."""
        return self.data

    def is_initialized(self) -> bool:
        """Check if engine is initialized with a model."""
        return self._is_initialized

    def get_state(self) -> EngineState | None:  # type: ignore[override]
        """Get current engine state."""
        return self.state

    def set_state(self, state: EngineState) -> None:  # type: ignore[override]
        """Set engine state."""
        self.state = state

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
            allowed_extensions: List of allowed extensions (e.g., [".urdf", ".xml"])

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
