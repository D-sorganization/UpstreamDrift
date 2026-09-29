from __future__ import annotations

from typing import Any

__all__ = ["Engine"]


def __getattr__(name: str) -> Any:
    if name == "Engine":
        from .python.mujoco_humanoid_golf.physics_engine import MuJoCoPhysicsEngine

        return MuJoCoPhysicsEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
