"""MuJoCo engine facade; native adapters must not load the generic GUI stack."""

from importlib import import_module
from typing import Any

__all__ = ["Engine"]


def __getattr__(name: str) -> Any:
    """Preserve the Engine export while deferring its unrelated import effects."""
    if name != "Engine":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    engine = import_module(
        ".python.mujoco_humanoid_golf.physics_engine", __name__
    ).MuJoCoPhysicsEngine
    globals()[name] = engine
    return engine
