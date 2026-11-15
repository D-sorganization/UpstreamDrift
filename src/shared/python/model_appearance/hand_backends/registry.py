"""Registry for swappable hand visual backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.types import (
    BACKEND_CAPSULE,
    BACKEND_MYOHAND,
    HandVisualMeshes,
    HandVisualSide,
)

HandVisualBuilder = Callable[
    [GripSpec, HandVisualSide, Mapping[str, float] | None],
    HandVisualMeshes,
]

_REGISTRY: dict[str, HandVisualBuilder] = {}


def register_hand_backend(name: str, builder: HandVisualBuilder) -> None:
    if not name or not name.strip():
        raise ValueError("backend name must be non-empty")
    _REGISTRY[name] = builder


def get_hand_backend(name: str) -> HandVisualBuilder:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown hand visual backend {name!r}; choose from {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def hand_backend_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _register_defaults() -> None:
    from src.shared.python.model_appearance.hand_backends import capsule, myohand

    register_hand_backend(BACKEND_CAPSULE, capsule.build_hand_visual)
    register_hand_backend(BACKEND_MYOHAND, myohand.build_hand_visual)


_register_defaults()

__all__ = [
    "get_hand_backend",
    "hand_backend_names",
    "register_hand_backend",
]
