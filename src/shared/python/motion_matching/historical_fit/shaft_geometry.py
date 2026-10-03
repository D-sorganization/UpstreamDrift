"""Exact authored solid-axis resolution without physical endpoint claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real
from typing import Any

import numpy as np

from src.shared.python.motion_matching.club_models import CLUB_BODY_SUFFIX


@dataclass(frozen=True)
class AuthoredShaftAxis:
    """Two local metre points defining an infinite authored shaft line."""

    body: str
    point_a_m: tuple[float, float, float]
    point_b_m: tuple[float, float, float]
    native_model_sha: str
    definition_sha: str
    resolution_method: str = "authored_shaft_grip_solid_axis_v1"
    physical_geometry_qualified: bool = False

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, Real) or isinstance(value, bool)
            for point in (self.point_a_m, self.point_b_m)
            for value in point
        ):
            raise ValueError(
                "Authored axis coordinates require real numbers, not boolean or text"
            )
        points = np.asarray((self.point_a_m, self.point_b_m), dtype=float)
        if (
            points.shape != (2, 3)
            or not np.isfinite(points).all()
            or np.linalg.norm(points[1] - points[0]) <= 1e-12
        ):
            raise ValueError("Authored axis requires two finite distinct local points")
        if (
            not isinstance(self.body, str)
            or not self.body
            or self.body.strip() != self.body
        ):
            raise ValueError("Authored shaft body is required")
        if (
            self.resolution_method != "authored_shaft_grip_solid_axis_v1"
            or self.physical_geometry_qualified is not False
        ):
            raise ValueError("Only an unqualified authored solid axis is supported")
        for name in ("native_model_sha", "definition_sha"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Exact lowercase native definition hash required")
        object.__setattr__(self, "point_a_m", tuple(float(x) for x in points[0]))
        object.__setattr__(self, "point_b_m", tuple(float(x) for x in points[1]))

    def to_record(self) -> dict[str, Any]:
        return {**asdict(self), "semantic": "infinite_authored_shaft_axis"}


def _solid_pose(solid: dict[str, Any]) -> np.ndarray:
    if any(
        isinstance(value, bool)
        for value in np.asarray(solid["placement"], dtype=object).ravel()
    ):
        raise ValueError("Shaft placement entries must be numeric, not boolean")
    pose = np.asarray(solid["placement"], dtype=float)
    com = np.asarray(solid["com_m"], dtype=float)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or com.shape != (3,)
        or not np.isfinite(com).all()
    ):
        raise ValueError("Declared shaft solid requires finite placement and COM")
    if (
        not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-12, rtol=0)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12, rtol=0)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Shaft placement must be a proper rigid transform")
    if not np.array_equal(com, np.zeros(3)):
        raise ValueError("Supported shaft rule requires centered declared solids")
    return pose


def _club_body(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or not isinstance(document.get("bodies"), list):
        raise ValueError("Native definition requires declared bodies")
    for body in document["bodies"]:
        if not isinstance(body, dict) or not isinstance(body.get("name"), str):
            raise ValueError("Native body requires a string name")
    bodies = [
        body for body in document["bodies"] if body["name"].endswith(CLUB_BODY_SUFFIX)
    ]
    if len(bodies) != 1:
        raise ValueError("Exactly one authored club body is required")
    body = bodies[0]
    if not isinstance(body.get("solids"), list) or any(
        not isinstance(solid, dict) or not isinstance(solid.get("name"), str)
        for solid in body["solids"]
    ):
        raise ValueError("Declared club solids require string names")
    return body


def resolve_authored_shaft_axis(
    definition_bytes: bytes, native_model_sha: str
) -> AuthoredShaftAxis:
    """Resolve the existing generic club's declared shaft/grip solid axis.

    The native plant SHA hashes the exact JSON definition bytes (not MJCF XML).
    This supported rule requires centered, coaxial declared solids; unknown
    models fail explicitly rather than guessing club tips or hand offsets.
    """
    if not isinstance(definition_bytes, bytes):
        raise ValueError("Exact native definition bytes required")
    digest = hashlib.sha256(definition_bytes).hexdigest()
    if digest != native_model_sha:
        raise ValueError("Native definition hash mismatch")
    try:
        document = json.loads(definition_bytes)
        body = _club_body(document)
        poses = []
        for suffix in ("Rigid Shaft", "Grip"):
            solids = [
                solid
                for solid in body["solids"]
                if solid["name"] == body["name"] + "/" + suffix
            ]
            if len(solids) != 1:
                raise ValueError("Exactly one declared shaft and grip solid required")
            poses.append(_solid_pose(solids[0]))
        a, b = poses
        direction = b[:3, 3] - a[:3, 3]
        if (
            not np.allclose(a[:3, 1], b[:3, 1], atol=1e-12, rtol=0)
            or np.linalg.norm(np.cross(direction, a[:3, 1])) > 1e-12
        ):
            raise ValueError("Declared shaft and grip solids are not coaxial")
        return AuthoredShaftAxis(
            body["name"], tuple(a[:3, 3]), tuple(b[:3, 3]), digest, digest
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Unsupported authored shaft geometry") from exc
