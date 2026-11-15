"""Registry of swappable hand visual backends."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends import capsule as capsule_module
from src.shared.python.model_appearance.hand_backends import (
    procedural_skin as procedural_skin_module,
)
from src.shared.python.model_appearance.hand_backends.contract import (
    HandJointAngles,
    HandRole,
    HandVisualResult,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_HAND_VISUAL_SETTINGS,
    HandVisualSettings,
)

BuildHandVisual = Callable[..., HandVisualResult]


@dataclass(frozen=True)
class HandBackendEntry:
    backend_id: str
    build: BuildHandVisual
    description: str


_REGISTRY: dict[str, HandBackendEntry] = {
    capsule_module.BACKEND_ID: HandBackendEntry(
        backend_id=capsule_module.BACKEND_ID,
        build=capsule_module.build_hand_visual,
        description="Legacy rigid capsule hand (baseline)",
    ),
    procedural_skin_module.BACKEND_ID: HandBackendEntry(
        backend_id=procedural_skin_module.BACKEND_ID,
        build=procedural_skin_module.build_hand_visual,
        description="Lofted procedural skin with arched palm",
    ),
}


def registered_backend_ids() -> tuple[str, ...]:
    """Stable ordering for harness layouts."""
    return tuple(sorted(_REGISTRY))


def iter_registered_backends() -> Iterator[HandBackendEntry]:
    for key in registered_backend_ids():
        yield _REGISTRY[key]


def register_backend(
    backend_id: str,
    build: BuildHandVisual,
    *,
    description: str = "",
    overwrite: bool = False,
) -> None:
    """Register another backend (other lanes append here)."""
    if backend_id in _REGISTRY and not overwrite:
        raise ValueError(f"backend {backend_id!r} already registered")
    _REGISTRY[backend_id] = HandBackendEntry(backend_id, build, description)


def build_hand_visual(
    backend_id: str,
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: HandJointAngles,
    *,
    grip_radius_m: float,
    settings: HandVisualSettings = DEFAULT_HAND_VISUAL_SETTINGS,
) -> HandVisualResult:
    try:
        entry = _REGISTRY[backend_id]
    except KeyError as exc:
        raise KeyError(f"unknown hand visual backend {backend_id!r}") from exc
    return entry.build(
        grip_spec,
        side,
        joint_angles,
        grip_radius_m=grip_radius_m,
        settings=settings,
    )


def build_all_registered(
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: HandJointAngles,
    *,
    grip_radius_m: float,
    settings: HandVisualSettings = DEFAULT_HAND_VISUAL_SETTINGS,
) -> dict[str, HandVisualResult]:
    return {
        bid: build_hand_visual(
            bid,
            grip_spec,
            side,
            joint_angles,
            grip_radius_m=grip_radius_m,
            settings=settings,
        )
        for bid in registered_backend_ids()
    }


__all__ = [
    "HandBackendEntry",
    "build_all_registered",
    "build_hand_visual",
    "iter_registered_backends",
    "register_backend",
    "registered_backend_ids",
]
