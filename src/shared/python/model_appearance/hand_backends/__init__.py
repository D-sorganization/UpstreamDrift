"""Pluggable grip hand visual backends (Shadow, MyoHand, procedural skin, …)."""

from src.shared.python.model_appearance.hand_backends.registry import (
    build_hand_visual as build_registered_hand_visual,
    default_backend_name,
    list_hand_backends,
    register_hand_backend,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HandSide,
    HandVisualResult,
    JointAngles,
)

__all__ = [
    "HandSide",
    "HandVisualResult",
    "JointAngles",
    "build_registered_hand_visual",
    "default_backend_name",
    "list_hand_backends",
    "register_hand_backend",
]
