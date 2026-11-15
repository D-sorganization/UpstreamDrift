"""Registry of grip hand visual backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.joint_angles import (
    HandJointAngles,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HandSide,
    HandVisualResult,
)

HandVisualBuilder = Callable[..., HandVisualResult]

_REGISTRY: dict[str, HandVisualBackend] = {}
_DEFAULT_BACKEND = "skinned_human"


@dataclass(frozen=True)
class HandVisualBackend:
    name: str
    builder: HandVisualBuilder


def register_hand_visual_backend(backend: HandVisualBackend) -> None:
    if backend.name in _REGISTRY:
        raise ValueError(f"backend {backend.name!r} already registered")
    _REGISTRY[backend.name] = backend


def get_hand_visual_backend(name: str = _DEFAULT_BACKEND) -> HandVisualBackend:
    _ensure_default_backends()
    if name not in _REGISTRY:
        raise KeyError(f"unknown hand visual backend {name!r}")
    return _REGISTRY[name]


def list_hand_visual_backends() -> tuple[str, ...]:
    _ensure_default_backends()
    return tuple(sorted(_REGISTRY))


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandSide,
    joint_angles: HandJointAngles | None = None,
    *,
    grip_radius_m: float,
    backend: str = _DEFAULT_BACKEND,
) -> HandVisualResult:
    """Build one hand with ``backend``; ``joint_angles=None`` lets it pose itself."""
    return get_hand_visual_backend(backend).builder(
        grip_spec,
        side,
        joint_angles,
        grip_radius_m=grip_radius_m,
    )


def _ensure_default_backends() -> None:
    if _DEFAULT_BACKEND not in _REGISTRY:
        from src.shared.python.model_appearance.hand_backends import skinned_human

        skinned_human.register()
