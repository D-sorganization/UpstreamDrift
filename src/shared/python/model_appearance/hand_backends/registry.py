"""Hand-visual backend registry for the SDF lane.

Sibling lanes register their own ids on their own branches. This module
registers ``sdf_hand`` only. ``build_hand_visual`` here is the dispatcher
(backend id first); the SDF builder lives in :mod:`sdf_hand`.
"""

from __future__ import annotations

from typing import Callable

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.contract import (
    HandJointAngles,
    HandRole,
)

Builder = Callable[..., object]

_BACKENDS: dict[str, tuple[Builder, str]] = {}


def register_backend(
    backend_id: str,
    build: Builder,
    description: str,
    *,
    overwrite: bool = False,
) -> None:
    """Register ``build(grip_spec, side, joint_angles, **kwargs)``.

    Preconditions: a non-empty id and a callable. Re-registering the same id
    raises ``ValueError`` unless ``overwrite`` is set.
    """
    if not isinstance(backend_id, str) or not backend_id:
        raise ValueError(f"backend_id must be a non-empty string, got {backend_id!r}")
    if not callable(build):
        raise TypeError("build must be callable")
    if backend_id in _BACKENDS and not overwrite:
        raise ValueError(f"hand backend {backend_id!r} is already registered")
    _BACKENDS[backend_id] = (build, description)


def registered_backend_ids() -> tuple[str, ...]:
    """Sorted ids currently registered in this process."""
    return tuple(sorted(_BACKENDS))


def build_hand_visual(
    backend_id: str,
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: HandJointAngles,
    **kwargs: object,
) -> object:
    """Dispatch to a registered backend.

    Unknown ids raise ``KeyError`` whose message contains ``unknown``.
    """
    try:
        build, _description = _BACKENDS[backend_id]
    except KeyError:
        raise KeyError(f"unknown hand backend {backend_id!r}") from None
    return build(grip_spec, side, joint_angles, **kwargs)


def _build_sdf_hand(
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: HandJointAngles,
    **kwargs: object,
) -> object:
    from src.shared.python.model_appearance.hand_backends.sdf_hand import (
        build_hand_visual as build_sdf,
    )

    return build_sdf(grip_spec, side, joint_angles, **kwargs)


register_backend(
    "sdf_hand",
    _build_sdf_hand,
    "Implicit SDF sculpted hand (smooth-min bones, marching cubes)",
)


__all__ = ["build_hand_visual", "register_backend", "registered_backend_ids"]
