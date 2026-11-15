"""Registry for grip hand visual backends (#15122)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.types import (
    HandSide,
    HandVisualResult,
    JointAngles,
)

Builder = Callable[..., HandVisualResult]

_REGISTRY: dict[str, Builder] = {}
_DEFAULT_BACKEND = "allegro"


def register_hand_backend(name: str, builder: Builder) -> None:
    """Register ``builder`` under ``name``."""
    key = name.strip().lower()
    if not key:
        raise ValueError("hand backend name must be non-empty")
    _REGISTRY[key] = builder


def list_hand_backends() -> tuple[str, ...]:
    """Return sorted registered backend names."""
    return tuple(sorted(_REGISTRY))


def default_backend_name() -> str:
    """Return the default backend name registered in this package."""
    return _DEFAULT_BACKEND


def build_hand_visual(
    backend: str,
    grip_spec: GripSpec,
    side: HandSide,
    joint_angles: JointAngles,
    **kwargs: Any,
) -> HandVisualResult:
    """Dispatch to a registered backend (raises ``KeyError`` when unknown)."""
    key = backend.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown hand visual backend {backend!r}; have {list(_REGISTRY)}"
        )
    return _REGISTRY[key](grip_spec, side, joint_angles, **kwargs)


def _bootstrap() -> None:
    from src.shared.python.model_appearance.hand_backends import allegro

    register_hand_backend("allegro", allegro.build_hand_visual)


_bootstrap()

__all__ = [
    "build_hand_visual",
    "default_backend_name",
    "list_hand_backends",
    "register_hand_backend",
]
