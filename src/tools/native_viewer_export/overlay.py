"""Overlay feed and framing for native exports (needs MuJoCo for the source).

One MuJoCo evaluation of the specification export gives every engine's viewer
the same joint-torque, ground-reaction and weight glyphs (see
``force_overlay.bundle_provider``); velocities come from the bundle reference.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from src.shared.python.force_overlay.bundle_provider import BundleOverlayProvider
from src.tools.native_viewer_export.core import (
    BackendUnavailable,
    OverlayFeed,
    SwingInput,
    default_glyph_style,
)

logger = logging.getLogger(__name__)

LOOKAT_HEIGHT_M = 0.9


def _grip_source(swing: SwingInput, engine: str, mujoco_source: Any) -> tuple[Any, str]:
    """Grip extraction for ``engine``: its own adapter where one exists.

    Drake uses its own KKT multiplier (``FullBodyDrakeModel``); every other
    viewer shows the MuJoCo plant evaluated at that engine's replayed pose
    (documented limitation: only MuJoCo and Drake extract a grip natively).
    """
    if engine == "drake":
        try:
            from src.engines.physics_engines.drake.python.full_body_model import (
                FullBodyDrakeModel,
            )

            return FullBodyDrakeModel(json.loads(swing.bundle.spec_bytes)), "drake:kkt"
        except ImportError:
            logger.warning("drake grip extraction unavailable; using the MuJoCo plant")
    return mujoco_source, "mujoco:kkt"


def build_overlay_feed(
    swing: SwingInput, engine: str, *, grip: bool = False
) -> tuple[OverlayFeed, tuple[float, float, float]]:
    """Overlay feed for ``swing`` plus the framing look-at point.

    With ``grip`` the feed also carries the per-hand/net/couple grip wrenches
    and the grip midpoint (camera focus for tracking views).

    Raises ``BackendUnavailable`` when MuJoCo is not installed.
    """
    try:
        from src.engines.physics_engines.mujoco.python.overlay_source import (
            MujocoOverlaySource,
        )
    except ImportError as exc:
        raise BackendUnavailable(
            f"mujoco is not available for overlays: {exc}"
        ) from exc
    source = MujocoOverlaySource(swing.bundle.spec_bytes)
    grip_source, grip_name = (
        _grip_source(swing, engine, source) if grip else (None, None)
    )
    provider = BundleOverlayProvider(
        swing.bundle,
        source,
        source,
        engine=engine,
        q=swing.q,
    )
    if grip_source is not None:
        provider.with_grip(grip_source, grip_name)
    names = swing.bundle.coordinate_order
    com = source.center_of_mass_m(dict(zip(names, map(float, swing.q[0]), strict=True)))
    lookat = (float(com[0]), float(com[1]), LOOKAT_HEIGHT_M)
    style = default_glyph_style(body_mass_kg=float(source.total_mass_kg))
    feed = OverlayFeed(
        provider.frame_at,
        style,
        grip_analyses=provider.grip_analyses if grip else None,
    )
    return feed, lookat


def detect_impact_time_s(swing: SwingInput) -> float:
    """Impact time of ``swing`` from the shared checked rule (one detector).

    The head trajectory is the face centre, the one clubhead measurement
    point (#12682): the ``Clubhead`` frame of the specification export placed
    by MuJoCo forward kinematics at every state of ``swing.q``, carrying the
    rendered face centre of the club assembly
    (``club_face_target.face_centre_in_frame``) through
    ``club_face_target.face_centre_path``. Impact is
    ``model_appearance.club_face.ball_passage`` (OSV-10), the sub-sample
    instant the face centre comes closest to its address position, with its
    height and ball-radius checks; the receipt (``fd_phase``) and the fixture
    speed timing use the same point.

    Raises ``BackendUnavailable`` without MuJoCo and ``ValueError`` when the
    specification has no club on the ``Clubhead`` frame or no frame is a
    valid impact.
    """
    try:
        from src.engines.physics_engines.mujoco.python.overlay_source import (
            MujocoOverlaySource,
        )
    except ImportError as exc:
        raise BackendUnavailable(f"mujoco is not available: {exc}") from exc
    from src.shared.python.model_appearance.club_face import ball_passage
    from src.shared.python.motion_matching.club_face_target import (
        FACE_FRAME,
        face_centre_in_frame,
        face_centre_path,
    )

    centre = face_centre_in_frame(json.loads(swing.bundle.spec_bytes))
    source = MujocoOverlaySource(swing.bundle.spec_bytes)
    names = swing.bundle.coordinate_order
    poses = np.array(
        [
            source.frame_poses(dict(zip(names, map(float, row), strict=True)))[
                FACE_FRAME
            ]
            for row in swing.q
        ],
        dtype=float,
    ).reshape(-1, 4, 4)
    head = face_centre_path(poses[:, :3, :3], poses[:, :3, 3], centre)
    times = np.asarray(swing.source_times_s, dtype=float)
    t_impact, _, _ = ball_passage(times, head)
    return t_impact
