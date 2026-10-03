"""Pure-numpy segment axes built from a joint tree (FTO-11, #11296).

A body's proximal end is the origin of its parent joint and its distal end is
the origin of its single child joint. Bodies with zero or several child joints,
or with a missing joint origin, are skipped and reported -- never guessed.

Headless import safe: numpy only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["SegmentAxis", "segment_axes_from_joint_tree"]

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class SegmentAxis:
    """Proximal-to-distal axis of one body, in world coordinates (metres)."""

    body: str
    proximal_m: Vec3
    distal_m: Vec3

    def __post_init__(self) -> None:
        if not isinstance(self.body, str) or not self.body.strip():
            raise ValueError("body must be a non-empty string")
        for name in ("proximal_m", "distal_m"):
            vec = np.asarray(getattr(self, name), dtype=float)
            if vec.shape != (3,) or not np.isfinite(vec).all():
                raise ValueError(f"{name} must be a finite 3-vector")
            object.__setattr__(self, name, tuple(float(x) for x in vec))
        if np.linalg.norm(np.subtract(self.distal_m, self.proximal_m)) <= 0.0:
            raise ValueError("segment axis must have positive length")


def _origin(origins: Mapping[str, np.ndarray], joint: str) -> Vec3 | None:
    if joint not in origins:
        return None
    vec = np.asarray(origins[joint], dtype=float)
    if vec.shape != (3,) or not np.isfinite(vec).all():
        raise ValueError(f"joint origin for '{joint}' must be a finite 3-vector")
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def segment_axes_from_joint_tree(
    joint_origin_world: Mapping[str, np.ndarray],
    child_joint_of: Mapping[str, str | Sequence[str]],
    body_of_joint: Mapping[str, str],
) -> tuple[tuple[SegmentAxis, ...], tuple[str, ...]]:
    """Build segment axes from joint origins.

    Args:
        joint_origin_world: joint name -> world origin of the joint.
        child_joint_of: body name -> its child joint name (``str``) or all of
            its child joint names (sequence). Absent or empty means a leaf.
        body_of_joint: joint name -> the (child) body that joint attaches.

    Returns:
        ``(axes, skipped)``. ``axes`` holds one axis per body that has exactly
        one child joint and known proximal/distal origins; ``skipped`` lists
        every other body (leaf, branching, or missing joint), sorted by name.

    Raises:
        TypeError: If an argument is not a mapping.
        ValueError: If a joint origin is not a finite 3-vector.
    """
    for name, arg in (
        ("joint_origin_world", joint_origin_world),
        ("child_joint_of", child_joint_of),
        ("body_of_joint", body_of_joint),
    ):
        if not isinstance(arg, Mapping):
            raise TypeError(f"{name} must be a mapping")

    for joint in joint_origin_world:
        _origin(joint_origin_world, joint)

    bodies = set(body_of_joint.values()) | set(child_joint_of)
    parent_joint_of = {body: joint for joint, body in body_of_joint.items()}
    axes: list[SegmentAxis] = []
    skipped: list[str] = []
    for body in sorted(bodies):
        children = child_joint_of.get(body, ())
        children = (children,) if isinstance(children, str) else tuple(children)
        parent = parent_joint_of.get(body)
        if len(children) != 1 or parent is None:
            skipped.append(body)
            continue
        proximal = _origin(joint_origin_world, parent)
        distal = _origin(joint_origin_world, children[0])
        if proximal is None or distal is None or proximal == distal:
            skipped.append(body)
            continue
        axes.append(SegmentAxis(body, proximal, distal))
    return tuple(axes), tuple(skipped)
