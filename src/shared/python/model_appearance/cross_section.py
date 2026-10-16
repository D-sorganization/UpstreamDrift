"""Cross-section orientation of non-round body segments from spec landmarks.

R1 (#12680, epic #12678). The library gives the trunk, pelvis and foot a
non-round cross-section (``library.PART_ASPECT``, depth/width). Which body
direction is "width" and which is "depth" is not a world default: it comes
from the spec's own anatomical landmarks at the zero pose, in each body's
frame.

- **Pelvis** (the body that parents both hip joints): mediolateral (ML) is
  the line through the two hip joint origins.
- **Trunk** (any other trunk body): ML is the line through the two shoulder
  joint origins, the children of the shoulder hubs.
- **Foot** (talus, calcn, toes): ML is ``up x (calcn -> toes)``, the ground
  normal crossed with the foot's long axis; for an OpenSim calcn this is its
  lateral z axis.

The depth (thin, ``aspect``) axis is anteroposterior for the trunk
(``ML x up``) and up for the foot (the shoe is wider than it is tall). Every
lofted segment's width axis is then ``geometry.section_width_axis(segment,
depth)``. This module is the only place that decides that mapping; the MuJoCo
appearance layer and the skinned glTF body both consume it. Visual only:
nothing here touches mass, inertia or joint frames.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.core.contracts import ensure, require
from src.shared.python.model_appearance import geometry
from src.shared.python.model_appearance.library import PART_ASPECT, classify_body
from src.shared.python.model_appearance.skinned_body import spec_body_poses

Array = NDArray[np.float64]

__all__ = ["BodySection", "body_sections", "width_hint_for"]

TRUNK_PARTS = frozenset({"torso", "pelvis"})
FOOT_PARTS = frozenset({"foot"})
# Landmark axes closer than this to "up" cannot define a horizontal ML/AP pair.
_MIN_TILT_FROM_UP_DEG = 45.0


@dataclass(frozen=True)
class BodySection:
    """Cross-section axes of one body, unit vectors in its spec body frame."""

    body: str
    ml_axis: Array  # mediolateral, from the landmarks (sign is irrelevant)
    depth_axis: Array  # the thin semi-axis: AP for the trunk, up for the foot

    def width_axis(self, start: ArrayLike, end: ArrayLike) -> Array:
        """Full-radius axis for the segment ``start -> end`` of this body."""
        return geometry.section_width_axis(
            np.subtract(np.asarray(end, float), np.asarray(start, float)),
            self.depth_axis,
        )


def width_hint_for(
    sections: Mapping[str, BodySection],
    body: str,
    start: ArrayLike,
    end: ArrayLike,
) -> Array | None:
    """``LoftOptions.width_hint`` for one segment; ``None`` for a round body."""
    section = sections.get(body)
    return None if section is None else section.width_axis(start, end)


def _unit(vector: ArrayLike, what: str) -> Array:
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    require(norm > 1e-9, f"{what} is degenerate", v)
    return v / norm


class _Landmarks:
    """Zero-pose landmark lookups over one spec document."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.joints: list[Mapping[str, Any]] = list(spec["joints"])
        self.poses = spec_body_poses(spec)
        gravity = np.asarray(spec["gravity_m_s2"], dtype=float)
        require(
            gravity.shape == (3,) and bool(np.isfinite(gravity).all()),
            "gravity_m_s2 must be a finite 3-vector",
            gravity,
        )
        self.up = _unit(-gravity, "gravity")
        self.parent_of = {j["child"]: j["parent"] for j in self.joints}

    def origin(self, joint: Mapping[str, Any]) -> Array:
        """World position of a joint origin (its base frame) at the zero pose."""
        base = self.poses[joint["parent"]] @ np.asarray(joint["parent_to_base"], float)
        return base[:3, 3]

    def pair_axis(self, joints: list[Mapping[str, Any]], what: str) -> Array:
        require(len(joints) == 2, f"expected a left/right pair of {what}", len(joints))
        return _unit(self.origin(joints[0]) - self.origin(joints[1]), what)

    def hips(self, body: str) -> list[Mapping[str, Any]]:
        return [
            j
            for j in self.joints
            if j["parent"] == body and classify_body(j["child"]) == "thigh"
        ]

    def shoulders(self) -> list[Mapping[str, Any]]:
        return [j for j in self.joints if classify_body(j["child"]) == "upper_arm"]

    def toe_axis(self, body: str) -> Array:
        """World heel-to-ball direction (calcn origin -> toe joint) of a foot."""
        root = body
        while classify_body(self.parent_of.get(root, "world")) in FOOT_PARTS:
            root = self.parent_of[root]
        toe: Mapping[str, Any] | None = None
        current = root
        while True:
            kids = [
                j
                for j in self.joints
                if j["parent"] == current and classify_body(j["child"]) in FOOT_PARTS
            ]
            if not kids:
                break
            require(len(kids) == 1, f"foot chain at {current} branches", len(kids))
            toe, current = kids[0], kids[0]["child"]
        if toe is None:  # precondition: a foot chain with a toe joint
            raise ValueError(f"foot {body} has no toe joint to orient it")
        return _unit(self.origin(toe) - self.poses[toe["parent"]][:3, 3], "foot")

    def to_body(self, body: str, world_vector: Array) -> Array:
        return self.poses[body][:3, :3].T @ world_vector


def _section(marks: _Landmarks, body: str, part: str) -> BodySection:
    """ML and depth axes of one non-round body, in its own frame."""
    if part in FOOT_PARTS:
        long_axis = marks.toe_axis(body)
        ml_world = _unit(np.cross(marks.up, long_axis), f"{body} foot axis vs up")
        depth_world = marks.up
    else:
        require(part in TRUNK_PARTS, f"no cross-section rule for part {part!r}", body)
        hips = marks.hips(body)
        ml_world = (
            marks.pair_axis(hips, "hip joints")
            if hips
            else marks.pair_axis(marks.shoulders(), "shoulder joints")
        )
        tilt = np.degrees(np.arccos(min(1.0, abs(float(ml_world @ marks.up)))))
        require(
            tilt >= _MIN_TILT_FROM_UP_DEG,
            f"{body} landmark ML axis is too close to vertical",
            tilt,
        )
        depth_world = _unit(np.cross(ml_world, marks.up), f"{body} AP axis")
        ml_world = _unit(np.cross(marks.up, depth_world), f"{body} ML axis")
    ml, depth = marks.to_body(body, ml_world), marks.to_body(body, depth_world)
    return BodySection(body, ml, depth)


def body_sections(spec: Mapping[str, Any]) -> dict[str, BodySection]:
    """Cross-section axes of every body whose part has a non-round aspect.

    Preconditions: the spec has the landmarks each such body needs (a hip
    pair or a shoulder pair for the trunk, a toe joint for the foot); there is
    no silent default. Postconditions: one entry per non-round body; unit
    axes; ML perpendicular to depth.
    """
    wanted = [
        (body["name"], classify_body(body["name"]))
        for body in spec["bodies"]
        if body["name"] != "world"
    ]
    wanted = [(name, part) for name, part in wanted if PART_ASPECT.get(part, 1.0) != 1]
    if not wanted:
        return {}
    marks = _Landmarks(spec)
    sections = {name: _section(marks, name, part) for name, part in wanted}
    for section in sections.values():
        ensure(
            abs(float(np.linalg.norm(section.ml_axis)) - 1.0) < 1e-9
            and abs(float(np.linalg.norm(section.depth_axis)) - 1.0) < 1e-9
            and abs(float(section.ml_axis @ section.depth_axis)) < 1e-9,
            "cross-section axes must be orthonormal",
            section.body,
        )
    return sections
