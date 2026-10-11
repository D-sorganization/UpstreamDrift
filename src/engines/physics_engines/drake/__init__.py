"""Drake physics engine public facade with lazy optional dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .python.drake_physics_engine import DrakePhysicsEngine as Engine

__all__ = ["Engine"]


def __getattr__(name: str) -> Any:
    if name == "Engine":
        module = import_module(".python.drake_physics_engine", __package__)
        return module.DrakePhysicsEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
