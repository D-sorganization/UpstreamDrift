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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
#: Heel-to-toe direction in the head frame (club_head_mesh: the shaft leans to -z).
TOE_HEAD = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class FaceWorld:
    """World face centre and unit axes; ``toe x normal = grip`` (toward the crown)."""

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
            ADDRESS_SQUARE_FACE_ROLL_DEG,
            assembly_from_spec,
            clubface_centre,
            clubface_vector,
        )
        from src.shared.python.model_appearance.club_head_mesh import (
            club_face_normal,
            face_roll_rotation,
            head_to_club_rotation,
            load_club_head,
        )
        from src.shared.python.model_appearance.grip_hands import grip_hand_frame
        from src.shared.python.model_appearance.grip_pose import Hand

        self.spec: dict[str, Any] = json.loads(spec_bytes)
        club = assembly_from_spec(self.spec)
        frame = next((f for f in self.spec["frames"] if f["name"] == "Clubhead"), None)
        if club is None or frame is None:
            raise ValueError("spec has no club body")
        self._placement_inv = np.linalg.inv(np.asarray(frame["placement"], dtype=float))
        self.head_alias: str = club.head_alias  # club head name, e.g. "driver"
        self._centre_body = np.asarray(clubface_centre(club), dtype=float)
        self._normal_body = np.asarray(clubface_vector(club), dtype=float)
        square = ADDRESS_SQUARE_FACE_ROLL_DEG.get(club.head_alias, 0.0)
        head = load_club_head(club.head_alias)
        self._square_normal_body = club_face_normal(head, square)
        # heel-to-toe line of the drawn (rolled) head in the club-body frame
        roll = face_roll_rotation(club.face_roll_deg)
        self._toe_body = roll @ head_to_club_rotation(head) @ TOE_HEAD
        self._source = MujocoOverlaySource(spec_bytes)
        lead = grip_hand_frame(club, Hand.LEAD)[:3, 3]
        trail = grip_hand_frame(club, Hand.TRAIL)[:3, 3]
        self._grip_midpoint_body = np.append(0.5 * (lead + trail), 1.0)

    @property
    def ground_height_m(self) -> float:
        return float(self.spec["contact"]["ground"].get("height_m") or 0.0)

    def body_world(self, coords: Mapping[str, float]) -> Array:
        """World 4x4 of the frame of the body carrying the head at ``coords``.

        The club body for a rigid shaft, the tip link for a chain shaft: both
        use the club-body axes, so the face expressed in them stays valid.
        """
        site = self._source.frame_poses(dict(coords))["Clubhead"]
        return np.asarray(site @ self._placement_inv, dtype=float)

    def body_positions(
        self, coords: Mapping[str, float], names: Sequence[str]
    ) -> dict[str, tuple[float, float, float]]:
        """World position of each model body in ``names`` at ``coords``."""
        return self._source.body_positions_m(coords, names)

    def grip_midpoint_world(self, coords: Mapping[str, float]) -> Array:
        """World point midway between the lead and trail hands on the grip.

        Kinematic only (default grip pose on the club body at ``coords``); it
        does not depend on any force overlay. The grip stays on the club
        body, so with a chain shaft it is read from the grip weld (the club
        body), never from :meth:`body_world`, which follows the bent tip link.
        """
        rotation, origin = self._source.club_frame_world(dict(coords))
        return np.asarray(rotation @ self._grip_midpoint_body[:3] + origin)

    def face_world(self, coords: Mapping[str, float]) -> FaceWorld:
        """World face centre, normal, toe and grip axes at ``coords``."""
        body = self.body_world(coords)
        return face_axes(body, self._centre_body, self._normal_body, self._toe_body)

    def address_square_normal_world(self, coords: Mapping[str, float]) -> Array:
        """World face normal the head would have with its address-square roll.

        The target line of a spec is where this face points at address; the
        head roll that is drawn (``face_roll_deg``) is a display choice and
        may leave the address face open or closed (#15150).
        """
        normal = self.body_world(coords)[:3, :3] @ self._square_normal_body
        return np.asarray(normal / np.linalg.norm(normal), dtype=np.float64)


def face_roll_calibrated_spec_bytes(
    spec_bytes: bytes, address_coords: Mapping[str, float], mode: str | None = None
) -> bytes:
    """``spec_bytes`` with the head roll of the face roll calibration ``mode``.

    ``mode`` defaults to ``face_roll_calibration_settings.json``. The default
    ``address_square`` returns ``spec_bytes`` itself; ``design_loft_at_address``
    solves the roll from this spec's own club-body pose at ``address_coords``
    (#15150). Only the drawn head and its face turn about the shaft: the
    club-body pose and every joint angle are unchanged. Raises as
    ``ClubPoseModel`` and ``spec_with_face_roll_calibration``.
    """
    from src.shared.python.model_appearance import face_roll_calibration as frc

    chosen = frc.validate_mode(
        frc.load_face_roll_settings().mode if mode is None else mode
    )
    if chosen == frc.ADDRESS_SQUARE:
        return spec_bytes
    body = ClubPoseModel(spec_bytes).body_world(address_coords)
    spec = frc.spec_with_face_roll_calibration(json.loads(spec_bytes), body, chosen)
    return json.dumps(spec).encode("utf-8")


def face_axes(
    body: Array, centre_body: Array, normal_body: Array, toe_body: Array
) -> FaceWorld:
    """Face axes in world from the club-body pose.

    The toe axis is ``toe_body``, the heel-to-toe line of the head (club-body
    frame), with its part along the normal removed and normalised. The shaft
    is not used: it leans ``90 deg - lie`` from that line, which turned a
    strike offset by 34 deg on the driver (#15150). ``toe x normal`` is the
    grip axis (toward the crown). Raises ``ValueError`` when ``toe_body`` is
    parallel to the face normal.
    """
    rot = body[:3, :3]
    normal = rot @ normal_body
    normal = normal / np.linalg.norm(normal)
    toe = rot @ toe_body
    toe = toe - float(toe @ normal) * normal
    length = float(np.linalg.norm(toe))
    if length < 1e-9:
        raise ValueError("face toe line is parallel to the face normal")
    toe = toe / length
    grip = np.asarray(np.cross(toe, normal), dtype=np.float64)
    return FaceWorld(rot @ centre_body + body[:3, 3], normal, toe, grip)
