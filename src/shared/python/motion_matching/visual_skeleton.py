"""Engine-agnostic visual skeleton derived from a body/joint specification.

Physics models in this repository carry inertias, joint frames and marker
frames but no visual shapes, so every engine renders a black frame. This
module derives one visual description from the specification itself: a
capsule per body from its own joint origin to each child joint origin (or to
its centre of mass for a leaf body), a small sphere at every centre of mass
and marker frame, and a ground plane opposite gravity. Engines only translate
these primitives into their own scene formats; nothing here touches dynamics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
_MIN_RADIUS_M, _MAX_RADIUS_M, _RADIUS_PER_LENGTH = 0.015, 0.05, 0.06
_SPHERE_RADIUS_M = {"com": 0.02, "frame": 0.012}


def _finite_vector(value: Any, size: int, name: str) -> Array:
    v = np.asarray(value, dtype=float)
    if v.shape != (size,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return v


def _transform(value: Any, name: str) -> Array:
    m = np.asarray(value, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    return m


@dataclass(frozen=True)
class Capsule:
    """A capsule in ``body``'s frame between two points, in metres."""

    body: str
    start_m: tuple[float, float, float]
    end_m: tuple[float, float, float]
    radius_m: float

    def length_m(self) -> float:
        return float(np.linalg.norm(np.subtract(self.end_m, self.start_m)))


@dataclass(frozen=True)
class Sphere:
    body: str
    center_m: tuple[float, float, float]
    radius_m: float
    kind: Literal["com", "frame"]
    label: str


@dataclass(frozen=True)
class GroundVisual:
    normal: tuple[float, float, float]
    height_m: float
    calibrated: bool


@dataclass(frozen=True)
class VisualSkeleton:
    capsules: tuple[Capsule, ...]
    spheres: tuple[Sphere, ...]
    ground: GroundVisual


@dataclass(frozen=True)
class WorldSegment:
    body: str
    start_m: Array
    end_m: Array
    radius_m: float


def capsule_radius(length_m: float) -> float:
    """Radius policy: proportional to length, clamped to a visible band."""
    if not math.isfinite(length_m) or length_m < 0:
        raise ValueError("Capsule length must be finite and nonnegative")
    return float(min(_MAX_RADIUS_M, max(_MIN_RADIUS_M, _RADIUS_PER_LENGTH * length_m)))


def _ground(spec: Mapping[str, Any]) -> GroundVisual:
    gravity = _finite_vector(spec["gravity_m_s2"], 3, "gravity_m_s2")
    norm = float(np.linalg.norm(gravity))
    if norm <= 0:
        raise ValueError("Gravity must be nonzero to orient the ground plane")
    normal = -gravity / norm
    contact = spec.get("contact")
    height = None
    if isinstance(contact, Mapping):
        height = contact.get("ground", {}).get("height_m")
    return GroundVisual(
        (float(normal[0]), float(normal[1]), float(normal[2])),
        0.0 if height is None else float(height),
        height is not None,
    )


def derive_visual_skeleton(spec: Mapping[str, Any]) -> VisualSkeleton:
    """Derive capsules, spheres and ground from a native or full-body spec.

    Preconditions: every joint parent is ``world`` or a body, every body is the
    child of exactly one joint, gravity is nonzero. Postcondition: exactly one
    capsule per body with positive length.
    """
    named = [body for body in spec["bodies"] if body["name"] != "world"]
    bodies = {body["name"]: body for body in named}
    if len(bodies) != len(named):
        raise ValueError("Duplicate body names")
    own_joint: dict[str, Array] = {}
    child_joints: dict[str, list[Array]] = {name: [] for name in bodies}
    for joint in spec["joints"]:
        parent, child = joint["parent"], joint["child"]
        if child not in bodies or (parent != "world" and parent not in bodies):
            raise ValueError(f"Joint {joint['name']} references an unknown body")
        if child in own_joint:
            raise ValueError(f"Body {child} has more than one parent joint")
        own_joint[child] = _transform(joint["child_to_follower"], "child_to_follower")[
            :3, 3
        ]
        if parent != "world":
            child_joints[parent].append(
                _transform(joint["parent_to_base"], "parent_to_base")[:3, 3]
            )
    missing = set(bodies) - set(own_joint)
    if missing:
        raise ValueError(f"Bodies without a parent joint: {sorted(missing)}")
    capsules: list[Capsule] = []
    spheres: list[Sphere] = []
    for name, body in bodies.items():
        start = own_joint[name]
        com_points = []
        for solid in body.get("solids", []):
            placement = _transform(solid["placement"], "placement")
            com = (
                placement[:3, :3] @ _finite_vector(solid["com_m"], 3, "com_m")
                + placement[:3, 3]
            )
            if float(solid.get("mass_kg", 0.0)) > 0:
                com_points.append(com)
                spheres.append(
                    Sphere(
                        name,
                        tuple(com.tolist()),
                        _SPHERE_RADIUS_M["com"],
                        "com",
                        solid["name"],
                    )
                )
        targets = child_joints[name] or com_points
        if not targets:
            raise ValueError(
                f"Body {name} has neither child joints nor a massive solid"
            )
        far = max(targets, key=lambda p: float(np.linalg.norm(p - start)))
        if float(np.linalg.norm(far - start)) <= 1e-9:
            far = start + np.array(
                [0.0, 0.0, _MIN_RADIUS_M]
            )  # zero-extent body: a stub
        capsules.append(
            Capsule(
                name,
                tuple(start.tolist()),
                tuple(far.tolist()),
                capsule_radius(float(np.linalg.norm(far - start))),
            )
        )
    for frame in spec.get("frames", []):
        placement = _transform(frame["placement"], "placement")
        spheres.append(
            Sphere(
                frame["body"],
                tuple(placement[:3, 3].tolist()),
                _SPHERE_RADIUS_M["frame"],
                "frame",
                frame["name"],
            )
        )
    return VisualSkeleton(tuple(capsules), tuple(spheres), _ground(spec))


def skeleton_world_segments(
    skeleton: VisualSkeleton, body_poses: Mapping[str, Any]
) -> list[WorldSegment]:
    """Map every capsule into world coordinates through 4x4 body poses."""
    segments = []
    for capsule in skeleton.capsules:
        if capsule.body not in body_poses:
            raise ValueError(f"Missing world pose for body {capsule.body}")
        pose = _transform(body_poses[capsule.body], f"pose of {capsule.body}")
        start = pose[:3, :3] @ np.asarray(capsule.start_m) + pose[:3, 3]
        end = pose[:3, :3] @ np.asarray(capsule.end_m) + pose[:3, 3]
        segments.append(WorldSegment(capsule.body, start, end, capsule.radius_m))
    return segments
