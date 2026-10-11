"""World pose of the spec club body and its true face (GCV-13 / GCV-19).

The specification export places a ``Clubhead`` frame on the club body
(``...Clubface Vector``). MuJoCo evaluates that frame as a site, so the world
pose of the spec club-body frame is ``site_pose @ inverse(frame placement)``.
The visual club assembly (``model_appearance.club_assembly``) then gives the
face centre, the face normal (with loft and roll) and the shaft direction in
that body frame. Both the decorative ball and the impact card use this single
helper so the face they see is the same.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
#: Shaft direction toward the grip in the spec club-body frame (club_models).
GRIP_DIRECTION_BODY = np.array([0.0, -1.0, 0.0])


@dataclass(frozen=True)
class FaceWorld:
    """World face centre and unit axes; ``toe x normal = grip`` (right-handed)."""

    centre_m: Array
    normal: Array
    toe: Array
    grip: Array


class ClubPoseModel:
    """MuJoCo forward kinematics of the spec club body and its face.

    Raises ``ImportError`` without MuJoCo and ``ValueError`` when the
    specification has no club body or no ``Clubhead`` frame.
    """

    def __init__(self, spec_bytes: bytes) -> None:
        from src.engines.physics_engines.mujoco.python.overlay_source import (
            MujocoOverlaySource,
        )
        from src.shared.python.model_appearance.club_assembly import (
            assembly_from_spec,
            clubface_centre,
            clubface_vector,
        )

        self.spec: dict[str, Any] = json.loads(spec_bytes)
        club = assembly_from_spec(self.spec)
        frame = next((f for f in self.spec["frames"] if f["name"] == "Clubhead"), None)
        if club is None or frame is None:
            raise ValueError("spec has no club body")
        self._placement_inv = np.linalg.inv(np.asarray(frame["placement"], dtype=float))
        self._centre_body = np.asarray(clubface_centre(club), dtype=float)
        self._normal_body = np.asarray(clubface_vector(club), dtype=float)
        self._source = MujocoOverlaySource(spec_bytes)

    @property
    def ground_height_m(self) -> float:
        return float(self.spec["contact"]["ground"].get("height_m") or 0.0)

    def body_world(self, coords: Mapping[str, float]) -> Array:
        """World 4x4 of the spec club-body frame at ``coords``."""
        site = self._source.frame_poses(dict(coords))["Clubhead"]
        return np.asarray(site @ self._placement_inv, dtype=float)

    def face_world(self, coords: Mapping[str, float]) -> FaceWorld:
        """World face centre, normal, toe and grip axes at ``coords``."""
        return face_axes(self.body_world(coords), self._centre_body, self._normal_body)


def face_axes(body: Array, centre_body: Array, normal_body: Array) -> FaceWorld:
    """Face axes in world from the club-body pose.

    The grip axis is the shaft direction (body -y) made orthogonal to the
    normal; the toe axis completes ``toe x normal = grip``.
    """
    rot = body[:3, :3]
    normal = rot @ normal_body
    normal = normal / np.linalg.norm(normal)
    shaft = rot @ GRIP_DIRECTION_BODY
    toe = np.cross(normal, shaft)
    length = float(np.linalg.norm(toe))
    if length < 1e-9:
        raise ValueError("face normal is parallel to the shaft")
    toe = toe / length
    grip = np.asarray(np.cross(toe, normal), dtype=np.float64)
    return FaceWorld(rot @ centre_body + body[:3, 3], normal, toe, grip)
