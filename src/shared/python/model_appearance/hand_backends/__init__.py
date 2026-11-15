"""Swappable procedural hand **visual** backends (epic #15122).

Contact physics continues to use capsule pads; these modules are appearance-only.
"""

from src.shared.python.model_appearance.hand_backends.registry import (
    get_hand_backend,
    hand_backend_names,
    register_hand_backend,
)

__all__ = [
    "get_hand_backend",
    "hand_backend_names",
    "register_hand_backend",
]
