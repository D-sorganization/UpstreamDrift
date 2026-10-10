"""Decorative address ball shared by the MeshCat native exports (GCV-13, #11719).

The swing bundle's first state (``swing.q[0]``) is the address frame: unlike
the static reference pose used elsewhere to place meshes, its clubhead rests
on the ground. The ball sits at the clubface centre of that frame, offset one
ball radius along the face normal -- the same shared placement rule as the
MuJoCo decorative ball (:mod:`src.shared.python.model_appearance.ball`), so
every engine's exported video shows the ball in the same place. Nothing here
touches dynamics: the ball is drawn, never simulated.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.ball import BALL_RADIUS_M, resolve_ball_visual
from src.shared.python.model_appearance.club_assembly import (
    assembly_from_spec,
    club_body_name,
    clubface_centre,
    clubface_vector,
)
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton
from src.tools.native_viewer_export.core import SwingInput

logger = logging.getLogger(__name__)

CLUBHEAD_FRAME = "Clubhead"
BALL_SOURCE = "address_geometry"
# Beyond this, the address frame's clubface centre is not plausibly resting at
# the ball (a mis-detected address frame, or a synthetic/test bundle whose
# first state is not a real address): never guess a position in that case.
MAX_ADDRESS_HEIGHT_M = 0.15


@dataclass(frozen=True)
class AddressBall:
    """The decorative address-ball draw, or the reason it could not be resolved.

    Postcondition: exactly one of ``position_m`` and ``reason`` is set.
    """

    position_m: NDArray[np.float64] | None
    radius_m: float
    source: str | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if (self.position_m is None) == (self.reason is None):
            raise ValueError("exactly one of position_m and reason must be set")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive and finite")


def _unavailable(reason: str) -> AddressBall:
    return AddressBall(None, BALL_RADIUS_M, None, reason=reason)


def resolve_address_ball(swing: SwingInput) -> AddressBall:
    """Decorative ball at the address frame (``swing.q[0]``) of ``swing``.

    The clubface centre and normal come from the club-body frame
    (``model_appearance.club_assembly``) composed with the ``Clubhead``
    frame's MuJoCo forward-kinematics pose at the address state; the ball
    position itself is :func:`model_appearance.ball.resolve_ball_visual`
    (one shared placement rule). Never places a guessed ball: when the
    specification has no club body, MuJoCo is unavailable, or the address
    frame's clubface centre is not plausibly grounded (more than
    :data:`MAX_ADDRESS_HEIGHT_M` from the ground), the reason is returned
    instead of a position.

    Postcondition: returns a position only when it is finite and the
    reported source is ``"address_geometry"``.
    """
    spec = json.loads(swing.bundle.spec_bytes)
    club = assembly_from_spec(spec)
    club_body = club_body_name(spec)
    if club is None or club_body is None:
        return _unavailable("spec has no club body")
    try:
        from src.engines.physics_engines.mujoco.python.overlay_source import (
            MujocoOverlaySource,
        )
    except ImportError as exc:
        return _unavailable(f"mujoco is not available: {exc}")
    source = MujocoOverlaySource(swing.bundle.spec_bytes)
    names = swing.bundle.coordinate_order
    coordinates = dict(zip(names, map(float, swing.q[0]), strict=True))
    frames = source.frame_poses(coordinates)
    if CLUBHEAD_FRAME not in frames:
        return _unavailable(f"spec has no {CLUBHEAD_FRAME!r} frame")
    frame = np.asarray(frames[CLUBHEAD_FRAME], dtype=float)
    centre_world = frame[:3, :3] @ clubface_centre(club) + frame[:3, 3]
    normal_world = frame[:3, :3] @ clubface_vector(club)
    ground_height_m = derive_visual_skeleton(spec).ground.height_m
    height_above_ground = abs(float(centre_world[2]) - ground_height_m)
    if height_above_ground > MAX_ADDRESS_HEIGHT_M:
        return _unavailable(
            "address clubhead is not grounded "
            f"({height_above_ground:.3f} m above ground_height_m)"
        )
    resolved = resolve_ball_visual(
        enabled=True,
        position_m=None,
        source=BALL_SOURCE,
        face_centre_m=centre_world,
        face_normal=normal_world,
        ground_height_m=ground_height_m,
    )
    if resolved is None:  # pragma: no cover - enabled=True always resolves
        return _unavailable("ball resolution failed unexpectedly")
    position, source_label = resolved
    return AddressBall(position, BALL_RADIUS_M, source_label)
