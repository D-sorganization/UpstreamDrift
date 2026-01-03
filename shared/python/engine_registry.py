"""Engine registration and discovery."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TypeAlias

from .interfaces import PhysicsEngine


class EngineType(Enum):
    """Available physics engine types."""

    MUJOCO = "mujoco"
    DRAKE = "drake"
    PINOCCHIO = "pinocchio"
    OPENSIM = "opensim"
    MYOSIM = "myosim"
    MATLAB_2D = "matlab_2d"
    MATLAB_3D = "matlab_3d"
    PENDULUM = "pendulum"

class EngineStatus(Enum):
    """Engine status types."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


EngineFactory: TypeAlias = Callable[[], PhysicsEngine]


@dataclass
class EngineRegistration:
    """Registration for a physics engine."""

    engine_type: EngineType
    factory: EngineFactory
    registration_path: Path | None = None
    requires_binary: list[str] = field(default_factory=list)
    probe_class: type | None = None


class EngineRegistry:
    """Registry of available physics engines.

    Separates engine discovery from engine loading.
    """

    def __init__(self) -> None:
        self._registrations: dict[EngineType, EngineRegistration] = {}
        self._root_path: Path | None = None

    def register(self, registration: EngineRegistration) -> None:
        """Register an engine."""
        self._registrations[registration.engine_type] = registration

    def get(self, engine_type: EngineType) -> EngineRegistration | None:
        """Get registration for an engine type."""
        return self._registrations.get(engine_type)

    def all_types(self) -> list[EngineType]:
        """Get all registered engine types."""
        return list(self._registrations.keys())

# Global registry instance
_registry = EngineRegistry()

def get_registry() -> EngineRegistry:
    """Get the global engine registry."""
    return _registry
