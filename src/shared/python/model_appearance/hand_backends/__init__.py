"""Swappable grip hand visual backends (ADR-0057 amendment, epic #15150).

Visual meshes only: contact geometry stays on the capsule pads.
"""

from src.shared.python.model_appearance.hand_backends.registry import (
    DISABLED_BY_DEFAULT,
    HandVisualBackend,
    build_hand_visual,
    get_hand_visual_backend,
    list_hand_visual_backends,
    register_hand_visual_backend,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HAND_SIDES,
    HandJointAngles,
    HandSide,
    HandVisualResult,
)

__all__ = [
    "DISABLED_BY_DEFAULT",
    "HAND_SIDES",
    "HandJointAngles",
    "HandSide",
    "HandVisualBackend",
    "HandVisualResult",
    "build_hand_visual",
    "get_hand_visual_backend",
    "list_hand_visual_backends",
    "register_hand_visual_backend",
]
