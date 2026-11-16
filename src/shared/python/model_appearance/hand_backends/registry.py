"""The one registry of grip hand visual backends (ADR-0057 A3).

Backends register a :class:`HandVisualBackend`; the registry never chooses a
default backend. Built-in backends register lazily: each module path in
``_BUILTIN_BACKEND_MODULES`` is imported the first time the registry is read
and its ``register()`` is called. Contact geometry is not handled here; it
stays on the capsule pads. The shared poser (``posing.py``) is not part of
this registry.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.types import (
    HandJointAngles,
    HandSide,
    HandVisualResult,
)

HandVisualBuilder = Callable[..., HandVisualResult]

#: Backends that stay registered but opt-in until an owner decision clears
#: them (MyoHand: HX6 #13763 geometry lineage).
DISABLED_BY_DEFAULT: frozenset[str] = frozenset({"myohand"})

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

#: Module paths whose ``register()`` adds a built-in backend. Tracks append one.
_BUILTIN_BACKEND_MODULES: tuple[str, ...] = (
    "src.shared.python.model_appearance.hand_backends.sdf_hand",
)

_REGISTRY: dict[str, HandVisualBackend] = {}
_builtins_loaded = False


@dataclass(frozen=True)
class HandVisualBackend:
    """One registered backend: a name, its builder and its metadata."""

    name: str
    builder: HandVisualBuilder
    description: str = ""
    capabilities: frozenset[str] = field(default_factory=frozenset)
    default_enabled: bool = True

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise TypeError(f"builder for {self.name!r} must be callable")
        if not isinstance(self.capabilities, frozenset):
            raise TypeError("capabilities must be a frozenset of strings")


def register_hand_visual_backend(backend: HandVisualBackend) -> None:
    """Add ``backend``.

    Raises ``ValueError`` for a name that is not lower-case ``snake_case``, a
    duplicate name, or an attempt to enable a backend in
    :data:`DISABLED_BY_DEFAULT`.
    """
    if not isinstance(backend.name, str) or not _NAME_PATTERN.match(backend.name):
        raise ValueError(
            f"backend name must be lower-case snake_case, got {backend.name!r}"
        )
    if backend.name in _REGISTRY:
        raise ValueError(f"backend {backend.name!r} already registered")
    if backend.name in DISABLED_BY_DEFAULT and backend.default_enabled:
        raise ValueError(
            f"backend {backend.name!r} must be registered with default_enabled=False"
        )
    _REGISTRY[backend.name] = backend


def get_hand_visual_backend(name: str) -> HandVisualBackend:
    """The backend called ``name``; ``KeyError`` lists the known names."""
    _ensure_builtin_backends()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown hand visual backend {name!r}; known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_hand_visual_backends(*, include_disabled: bool = False) -> tuple[str, ...]:
    """Sorted backend names; disabled backends only when asked for."""
    _ensure_builtin_backends()
    return tuple(
        sorted(
            name
            for name, backend in _REGISTRY.items()
            if include_disabled or backend.default_enabled
        )
    )


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandSide,
    joint_angles: HandJointAngles | None = None,
    *,
    grip_radius_m: float,
    backend: str,
    skin_rgb: tuple[float, float, float] | None = None,
) -> HandVisualResult:
    """Build one hand with the named backend (disabled backends build by name)."""
    return get_hand_visual_backend(backend).builder(
        grip_spec,
        side,
        joint_angles,
        grip_radius_m=grip_radius_m,
        skin_rgb=skin_rgb,
    )


def _ensure_builtin_backends() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    for module_path in _BUILTIN_BACKEND_MODULES:
        importlib.import_module(module_path).register()


__all__ = [
    "DISABLED_BY_DEFAULT",
    "HandVisualBackend",
    "HandVisualBuilder",
    "build_hand_visual",
    "get_hand_visual_backend",
    "list_hand_visual_backends",
    "register_hand_visual_backend",
]
