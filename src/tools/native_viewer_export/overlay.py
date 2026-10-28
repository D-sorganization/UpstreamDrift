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


def _impact_or_none(swing: SwingInput) -> float | None:
    """Detected impact time, or None (logged) when the swing has no valid impact."""
    try:
        return detect_impact_time_s(swing)
    except (ValueError, BackendUnavailable) as exc:
        logger.warning("no ball impact for the axial blackout: %s", exc)
        return None


def palm_thumb_patch_for_spec(spec_bytes: bytes) -> tuple[Any, float]:
    """(``ThumbPatch``, spec static friction) of a full-body specification.

    The club comes from the spec (``assembly_from_spec``), the hands from the
    default grip pose and the sides from ``subject_handedness(spec)``.
    Raises ``ValueError`` if the spec has no club or no static friction.
    """
    from src.shared.python.core.handedness import subject_handedness
    from src.shared.python.model_appearance.club_assembly import assembly_from_spec
    from src.shared.python.model_appearance.grip_contact_patch import (
        thumb_palm_patch,
    )
    from src.shared.python.model_appearance.grip_hands import (
        grip_hand_frame,
        grip_radius_at,
    )
    from src.shared.python.model_appearance.grip_pose import (
        DEFAULT_GRIP_POSE,
        Hand,
    )

    spec = json.loads(spec_bytes)
    club = assembly_from_spec(spec)
    if club is None:
        raise ValueError("the specification has no club for the palm-thumb patch")
    friction = spec.get("contact", {}).get("parameters", {}).get("static_friction")
    if friction is None:
        raise ValueError("the specification has no contact static_friction")
    pose = DEFAULT_GRIP_POSE
    patch = thumb_palm_patch(
        grip_hand_frame(club, Hand.LEAD, pose),
        grip_radius_m=grip_radius_at(club, pose.distance_below_butt_m(Hand.LEAD)),
        pose=pose,
        handedness=subject_handedness(spec),
    )
    return patch, float(friction)


def build_overlay_feed(
    swing: SwingInput,
    engine: str,
    *,
    grip: bool = False,
    axial: bool = False,
    impact_time_s: float | None = None,
    show_force: bool = True,
    show_torque: bool = True,
    palm_thumb: bool = False,
    hand_mobility: str | None = None,
    hand_mobility_mode: str = "constrained",
) -> tuple[OverlayFeed, tuple[float, float, float]]:
    """Overlay feed for ``swing`` plus the framing look-at point.

    With ``grip`` the feed also carries the per-hand/net/couple grip wrenches
    and the grip midpoint (camera focus for tracking views). With ``axial`` each
    frame also carries the segment axial loads (tension positive) from inverse
    dynamics at the provider's own kinematics (``q`` of ``swing``, bundle
    reference velocity, accelerations by finite difference of that velocity), so
    the torque arcs and the shading come from one evaluation. The grip weld
    closure wrench is solved from the bundle efforts and injected; around ball
    impact (``impact_time_s``, default the shared detector's; frames within one
    bundle step) the loads are unavailable, as the impact impulse is not
    modelled (#13179). ``show_force`` and
    ``show_torque`` choose the glyph kinds drawn. With ``palm_thumb`` each loaded
    frame also carries the allocated trail-palm-on-lead-thumb normal force
    (HRG-H1, #13189), drawn independently of the other vectors; it is reporting
    only and leaves the compiled model unchanged. With ``hand_mobility``
    (``mobility``, ``force`` or ``both``) each frame also carries the per-hand
    mobility / effective-mass ellipsoids (#14004) in ``hand_mobility_mode``.

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
    if axial:
        provider.with_axial_loads(source)
        impact = _impact_or_none(swing) if impact_time_s is None else impact_time_s
        if impact is not None:
            provider.with_axial_impact_window(impact, float(swing.bundle.dt_s))
    if palm_thumb:
        spec = json.loads(swing.bundle.spec_bytes)
        friction = spec.get("contact", {}).get("parameters", {}).get("static_friction")
        if friction is None:
            raise ValueError("the specification has no contact static_friction")
        try:
            from src.shared.python.motion_matching.execution.grip_rotation import (
                grip_spec_of,
            )

            grip_spec = grip_spec_of(spec)
        except (ValueError, KeyError):
            from src.shared.python.motion_matching.grip_contract import GripSpec

            grip_spec = GripSpec()
        from src.shared.python.grip_contact.contact_topology import contact_topology

        topo = contact_topology(grip_spec)
        provider.with_allocated_contacts(source, topo, static_friction=float(friction))
    names = swing.bundle.coordinate_order
    com = source.center_of_mass_m(dict(zip(names, map(float, swing.q[0]), strict=True)))
    lookat = (float(com[0]), float(com[1]), LOOKAT_HEIGHT_M)
    style = default_glyph_style(body_mass_kg=float(source.total_mass_kg))
    ellipsoids_at = None
    if hand_mobility is not None:
        from src.tools.native_viewer_export.hand_mobility_glyphs import (
            hand_mobility_glyphs,
        )

        ellipsoids_at = lambda i: hand_mobility_glyphs(  # noqa: E731
            swing.hand_mobility_at(i, hand_mobility_mode), hand_mobility
        )
    feed = OverlayFeed(
        provider.frame_at,
        style,
        grip_analyses=provider.grip_analyses if grip else None,
        show_force=show_force,
        show_torque=show_torque,
        show_palm_thumb=palm_thumb,
        ellipsoids_at=ellipsoids_at,
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
