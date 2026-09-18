"""Engine-specific MatchingPlant implementations."""

from __future__ import annotations

__all__ = [
    "DrakeMatchingPlant",
    "MujocoMatchingPlant",
    "PinocchioMatchingPlant",
]


def __getattr__(name: str) -> object:
    if name == "MujocoMatchingPlant":
        from .mujoco import MujocoMatchingPlant

        return MujocoMatchingPlant
    if name == "DrakeMatchingPlant":
        from .drake import DrakeMatchingPlant

        return DrakeMatchingPlant
    if name == "PinocchioMatchingPlant":
        from .pinocchio import PinocchioMatchingPlant

        return PinocchioMatchingPlant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
