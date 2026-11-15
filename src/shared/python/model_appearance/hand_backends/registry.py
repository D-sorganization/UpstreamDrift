"""Registry for swappable hand visual backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.types import (
    BACKEND_CAPSULE,
    BACKEND_MYOHAND,
    DEFAULT_HAND_VISUAL_BACKEND,
    MYOHAND_BACKEND_INFO,
    HandBackendInfo,
    HandVisualMeshes,
    HandVisualSide,
)

HandVisualBuilder = Callable[
    [GripSpec, HandVisualSide, Mapping[str, float] | None],
    HandVisualMeshes,
]


@dataclass(frozen=True)
class _Entry:
    builder: HandVisualBuilder
    info: HandBackendInfo


_REGISTRY: dict[str, _Entry] = {}


def register_hand_backend(
    name: str,
    builder: HandVisualBuilder,
    *,
    info: HandBackendInfo | None = None,
) -> None:
    if not name or not name.strip():
        raise ValueError("backend name must be non-empty")
    meta = info or HandBackendInfo(name=name)
    _REGISTRY[name] = _Entry(builder=builder, info=meta)


def hand_backend_info(name: str) -> HandBackendInfo:
    return _REGISTRY[name].info


def get_hand_backend(
    name: str,
    *,
    allow_research: bool = False,
) -> HandVisualBuilder:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown hand visual backend {name!r}; choose from {sorted(_REGISTRY)}"
        )
    entry = _REGISTRY[name]
    if entry.info.research_only and not allow_research:
        raise PermissionError(
            f"backend {name!r} is research-only until HRG-HX6 #13763 sign-off; "
            "pass allow_research=True for lab use"
        )
    return entry.builder


def default_hand_backend() -> HandVisualBuilder:
    return get_hand_backend(DEFAULT_HAND_VISUAL_BACKEND)


def hand_backend_names(*, include_research: bool = False) -> tuple[str, ...]:
    names = []
    for key, entry in sorted(_REGISTRY.items()):
        if entry.info.research_only and not include_research:
            continue
        names.append(key)
    return tuple(names)


def _register_defaults() -> None:
    from src.shared.python.model_appearance.hand_backends import capsule, myohand

    register_hand_backend(BACKEND_CAPSULE, capsule.build_hand_visual)
    register_hand_backend(
        BACKEND_MYOHAND,
        myohand.build_hand_visual,
        info=MYOHAND_BACKEND_INFO,
    )


_register_defaults()

__all__ = [
    "default_hand_backend",
    "get_hand_backend",
    "hand_backend_info",
    "hand_backend_names",
    "register_hand_backend",
]