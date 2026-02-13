"""Engine registration and discovery."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TypeAlias

from src.shared.python.core.contracts import ContractChecker, precondition

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
    PUTTING_GREEN = "putting_green"


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


class EngineRegistry(ContractChecker):
    """Registry of available physics engines.

    Separates engine discovery from engine loading.

    Design by Contract:
        Invariants:
            - _registrations dict is never None
            - All registered types are valid EngineType values
    """

    def __init__(self) -> None:
        self._registrations: dict[EngineType, EngineRegistration] = {}
        self._root_path: Path | None = None

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Define class invariants for EngineRegistry."""
        return [
            (
                lambda: self._registrations is not None
                and isinstance(self._registrations, dict),
                "Registrations must be a non-None dict",
            ),
            (
                lambda: all(isinstance(k, EngineType) for k in self._registrations),
                "All registration keys must be EngineType instances",
            ),
        ]

    @precondition(
        lambda self, registration: registration is not None,
        "Registration must not be None",
    )
    @precondition(
        lambda self, registration: isinstance(registration, EngineRegistration),
        "Registration must be an EngineRegistration instance",
    )
    def register(self, registration: EngineRegistration) -> None:
        """Register an engine."""
        self._registrations[registration.engine_type] = registration

    @precondition(
        lambda self, engine_type: isinstance(engine_type, EngineType),
        "Engine type must be a valid EngineType enum member",
    )
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
