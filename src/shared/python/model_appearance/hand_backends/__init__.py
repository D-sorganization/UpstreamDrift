"""Swappable grip-hand visual backends (epic #15150).

This lane registers the implicit SDF hand. Sibling lanes register their own
backends on their own branches; ``register_backend`` is the merge point.
"""

from __future__ import annotations

from src.shared.python.model_appearance.hand_backends.registry import (
    build_hand_visual,
    registered_backend_ids,
)

__all__ = ["build_hand_visual", "registered_backend_ids"]
