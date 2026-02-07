"""Simulation Controller — standardized simulation control interface.

Provides a uniform control interface (start/stop/pause/step/reset/rewind)
and model positioning (translate/rotate bodies) across all engines.

Design by Contract:
    State Machine:
        IDLE -> [start] -> RUNNING
        RUNNING -> [pause] -> PAUSED
        PAUSED -> [resume] -> RUNNING
        RUNNING|PAUSED -> [stop] -> IDLE
        Any -> [reset] -> IDLE (t=0)

    Invariants:
        - Controls never modify model topology
        - Positioning preserves joint constraints
        - Measurement tools are read-only
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Simulation run state."""

    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STEPPING = auto()


@dataclass
class MeasurementResult:
    """Result of a measurement between two points/bodies.

    Attributes:
        type: Measurement type ('distance', 'angle', 'velocity')
        value: Scalar measurement value
        unit: Unit string
        point_a: First point/body name
        point_b: Second point/body name (if applicable)
        vector: Direction/axis vector (if applicable)
    """

    type: str
    value: float
    unit: str
    point_a: str
    point_b: str = ""
    vector: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to API-friendly dictionary."""
        result: dict[str, Any] = {
            "type": self.type,
            "value": self.value,
            "unit": self.unit,
            "point_a": self.point_a,
            "point_b": self.point_b,
        }
        if self.vector is not None:
            result["vector"] = self.vector.tolist()
        return result


@dataclass
class ForceOverlay:
    """Configuration for force/torque vector overlay.

    Attributes:
        body_name: Body to attach the vector to
        force: Force vector [N] in world frame
        torque: Torque vector [N·m] in world frame
        scale: Visual scaling factor for rendering
        color: RGB color tuple (0-255)
        label: Optional text label
    """

    body_name: str
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    color: tuple[int, int, int] = (255, 100, 100)
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to API-friendly dictionary."""
        return {
            "body_name": self.body_name,
            "force": self.force.tolist(),
            "torque": self.torque.tolist(),
            "scale": self.scale,
            "color": list(self.color),
            "label": self.label,
        }


class SimulationController(ABC):
    """Abstract controller for simulation lifecycle and tools.

    Engines that want full simulation control features should
    implement this alongside the PhysicsEngine protocol.
    """

    def __init__(self) -> None:
        """Initialize controller state."""
        self._mode: SimulationMode = SimulationMode.IDLE
        self._overlays: list[ForceOverlay] = []
        self._measurements: list[MeasurementResult] = []

    @property
    def mode(self) -> SimulationMode:
        """Current simulation mode."""
        return self._mode

    @property
    def is_running(self) -> bool:
        """True if simulation is actively running."""
        return self._mode == SimulationMode.RUNNING

    @property
    def is_paused(self) -> bool:
        """True if simulation is paused."""
        return self._mode == SimulationMode.PAUSED

    # --- Lifecycle ---

    def start(self) -> bool:
        """Start or resume simulation.

        Returns:
            True if transition was successful
        """
        if self._mode in {SimulationMode.IDLE, SimulationMode.PAUSED}:
            self._mode = SimulationMode.RUNNING
            logger.info("Simulation started")
            return True
        return False

    def pause(self) -> bool:
        """Pause simulation.

        Returns:
            True if transition was successful
        """
        if self._mode == SimulationMode.RUNNING:
            self._mode = SimulationMode.PAUSED
            logger.info("Simulation paused")
            return True
        return False

    def stop(self) -> bool:
        """Stop simulation and return to idle.

        Returns:
            True if transition was successful
        """
        if self._mode in {SimulationMode.RUNNING, SimulationMode.PAUSED}:
            self._mode = SimulationMode.IDLE
            logger.info("Simulation stopped")
            return True
        return False

    def single_step(self) -> bool:
        """Execute a single timestep.

        Returns:
            True if step was executed
        """
        if self._mode in {SimulationMode.IDLE, SimulationMode.PAUSED}:
            self._mode = SimulationMode.STEPPING
            self._do_step()
            self._mode = SimulationMode.PAUSED
            return True
        return False

    @abstractmethod
    def _do_step(self) -> None:
        """Execute a single physics step (engine-specific)."""

    # --- Model Positioning ---

    @abstractmethod
    def translate_body(self, body_name: str, delta: np.ndarray) -> bool:
        """Translate a body by a displacement vector.

        Args:
            body_name: Name of the body to move
            delta: Displacement vector [x, y, z] in meters

        Returns:
            True if translation was applied
        """

    @abstractmethod
    def rotate_body(self, body_name: str, axis: np.ndarray, angle: float) -> bool:
        """Rotate a body around an axis.

        Args:
            body_name: Name of the body to rotate
            axis: Rotation axis (unit vector)
            angle: Rotation angle in radians

        Returns:
            True if rotation was applied
        """

    # --- Force Overlays ---

    def add_force_overlay(self, overlay: ForceOverlay) -> None:
        """Add a force/torque vector overlay.

        Args:
            overlay: ForceOverlay configuration
        """
        self._overlays.append(overlay)

    def clear_overlays(self) -> None:
        """Remove all force/torque overlays."""
        self._overlays.clear()

    @property
    def overlays(self) -> list[ForceOverlay]:
        """Get current force overlays."""
        return list(self._overlays)

    # --- Measurements ---

    @abstractmethod
    def measure_distance(self, body_a: str, body_b: str) -> MeasurementResult:
        """Measure distance between two bodies.

        Args:
            body_a: First body name
            body_b: Second body name

        Returns:
            MeasurementResult with distance in meters
        """

    @abstractmethod
    def measure_angle(self, body_a: str, body_b: str, body_c: str) -> MeasurementResult:
        """Measure angle between three bodies (vertex at body_b).

        Args:
            body_a: First body
            body_b: Vertex body
            body_c: Third body

        Returns:
            MeasurementResult with angle in radians
        """

    def get_measurements(self) -> list[dict[str, Any]]:
        """Get all current measurements as dictionaries.

        Returns:
            List of serialized measurement results
        """
        return [m.to_dict() for m in self._measurements]

    def clear_measurements(self) -> None:
        """Clear all measurements."""
        self._measurements.clear()
