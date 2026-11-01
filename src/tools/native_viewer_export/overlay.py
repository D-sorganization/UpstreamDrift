"""Overlay feed and framing for native exports (needs MuJoCo for the source).

One MuJoCo evaluation of the specification export gives every engine's viewer
the same joint-torque, ground-reaction and weight glyphs (see
``force_overlay.bundle_provider``); velocities come from the bundle reference.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.force_overlay.bundle_provider import BundleOverlayProvider
from src.shared.python.force_overlay.glyphs import EllipsoidGlyph
from src.tools.native_viewer_export.core import (
    BackendUnavailable,
    OverlayFeed,
    SwingInput,
    default_glyph_style,
)

logger = logging.getLogger(__name__)

LOOKAT_HEIGHT_M = 0.9


@dataclass(frozen=True)
class OverlayOptions:
    """Configurable display toggles and ellipsoid options for build_overlay_feed."""

    show_force: bool = True
    show_torque: bool = True
    palm_thumb: bool = False
    hand_mobility: bool | str = False
    hand_mobility_mode: str = "constrained"
    interface_ellipsoids: Sequence[str] = ()
    ellipsoid_scale: str = "auto"


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


def _extract_physical_radii(hm: Any, kind: str) -> tuple[np.ndarray, tuple[float, ...]]:
    """Return (axes, radii) for a given ellipsoid kind."""
    if kind == "mobility":
        axes = hm.mobility_axes
        r = np.sqrt(np.maximum(hm.mobility_eigenvalues, 0.0))
    elif kind == "effective-mass":
        axes = hm.mobility_axes
        r = 1.0 / np.sqrt(np.maximum(hm.mobility_eigenvalues, 1e-9))
    elif kind == "force-capacity" and hm.force_capacity is not None:
        axes = hm.force_capacity.axes
        r = hm.force_capacity.radii
    elif kind == "acceleration-capacity" and hm.acceleration_capacity is not None:
        axes = hm.acceleration_capacity.axes
        r = hm.acceleration_capacity.radii
    else:
        axes = hm.mobility_axes
        r = np.sqrt(np.maximum(hm.mobility_eigenvalues, 0.0))
    return axes, tuple(map(float, r))


def _build_hand_ellipsoid_feed(
    swing: SwingInput,
    active_ellipsoids: Sequence[str],
    hand_mobility_mode: str,
    ellipsoid_scale: str,
    engine: str = "mujoco",
) -> tuple[
    Callable[[int], tuple[EllipsoidGlyph, ...]], Callable[[int], tuple[str, ...]]
]:
    """Build ellipsoid glyph emitter and HUD capability formatter."""
    from src.shared.python.force_overlay.ellipsoid_display import (
        EllipsoidDisplayBuilder,
        EllipsoidDisplayScale,
        format_hud_capability,
    )

    _hm_cache: dict[int, Any] = {}

    def _get_hm(idx: int) -> dict[str, Any]:
        if idx not in _hm_cache:
            _hm_cache[idx] = swing.hand_mobility_at(
                idx, mode=hand_mobility_mode, engine=engine
            )
        return _hm_cache[idx]

    builders: dict[str, EllipsoidDisplayBuilder] = {}
    scales: dict[str, EllipsoidDisplayScale] = {}
    for kind in active_ellipsoids:
        semi_axes_series: list[list[float]] = []
        stride = max(1, swing.n_states // 20)
        for idx in range(0, swing.n_states, stride):
            hm_sample = _get_hm(idx)
            for h in ("lead", "trail"):
                _, r = _extract_physical_radii(hm_sample[h], kind)
                semi_axes_series.append(list(r))
        scale = EllipsoidDisplayScale.compute(
            semi_axes_series,
            target_radius_m=0.09,
            kind=kind,
            mode=ellipsoid_scale,
        )
        scales[kind] = scale
        builders[kind] = EllipsoidDisplayBuilder(scale)

    def hand_ellipsoids_fn(index: int) -> tuple[EllipsoidGlyph, ...]:
        hm_map = _get_hm(index)
        glyphs_list: list[EllipsoidGlyph] = []
        for kind in active_ellipsoids:
            builder = builders[kind]
            for hand in ("lead", "trail"):
                hm = hm_map[hand]
                axes, radii_phys = _extract_physical_radii(hm, kind)
                disp = builder.build_glyph(
                    hm.centre,
                    axes,
                    radii_phys,
                    hand,
                    label=f"{hand} {kind}",
                )
                glyphs_list.append(
                    EllipsoidGlyph(
                        disp.centre,
                        disp.axes,
                        disp.radii,
                        disp.rgba,
                        disp.label,
                    )
                )
        return tuple(glyphs_list)

    def hand_hud_lines_fn(index: int) -> tuple[str, ...]:
        hm_map = _get_hm(index)
        lines: list[str] = []
        for kind in active_ellipsoids:
            scale = scales[kind]
            lead_hm, trail_hm = hm_map["lead"], hm_map["trail"]
            _, lead_r = _extract_physical_radii(lead_hm, kind)
            _, trail_r = _extract_physical_radii(trail_hm, kind)
            lead_hud = format_hud_capability(
                "lead", kind, lead_r, scale.unit, lead_hm.condition
            )
            trail_hud = format_hud_capability(
                "trail", kind, trail_r, scale.unit, trail_hm.condition
            )
            lines.append(f"{lead_hud} | {trail_hud}")
        return tuple(lines)

    return hand_ellipsoids_fn, hand_hud_lines_fn


def _resolve_active_ellipsoids(
    hand_mobility: bool | str,
    interface_ellipsoids: Sequence[str],
) -> list[str]:
    """Resolve active ellipsoid kinds from CLI and config flags."""
    active: list[str] = list(interface_ellipsoids)
    if hand_mobility:
        if hand_mobility in ("velocity", "mobility", True):
            if "mobility" not in active:
                active.append("mobility")
        elif hand_mobility == "force":
            if "effective-mass" not in active:
                active.append("effective-mass")
        elif hand_mobility == "both":
            if "mobility" not in active:
                active.append("mobility")
            if "effective-mass" not in active:
                active.append("effective-mass")
    return active


def _apply_palm_thumb(
    provider: BundleOverlayProvider, source: Any, spec_bytes: bytes
) -> None:
    """Configure allocated palm-thumb contact overlay on the provider."""
    spec = json.loads(spec_bytes)
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


def build_overlay_feed(
    swing: SwingInput,
    engine: str,
    *,
    grip: bool = False,
    axial: bool = False,
    impact_time_s: float | None = None,
    palm_thumb: bool = False,
    options: OverlayOptions | None = None,
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
    modelled (#13179). Options in ``options`` (or ``palm_thumb``) configure
    vector kinds (``show_force``, ``show_torque``), palm-thumb normal force
    (HRG-H1, #13189), and hand mobility/force interface ellipsoids (HCM-3, #14004).

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
    opts = options or OverlayOptions()
    show_force = opts.show_force
    show_torque = opts.show_torque
    effective_palm_thumb = palm_thumb or opts.palm_thumb

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
    if effective_palm_thumb:
        _apply_palm_thumb(provider, source, swing.bundle.spec_bytes)
    names = swing.bundle.coordinate_order
    com = source.center_of_mass_m(dict(zip(names, map(float, swing.q[0]), strict=True)))
    lookat = (float(com[0]), float(com[1]), LOOKAT_HEIGHT_M)
    style = default_glyph_style(body_mass_kg=float(source.total_mass_kg))

    active_ellipsoids = _resolve_active_ellipsoids(
        opts.hand_mobility, opts.interface_ellipsoids
    )
    hand_ellipsoids_fn = None
    hand_hud_lines_fn = None
    if active_ellipsoids:
        hand_ellipsoids_fn, hand_hud_lines_fn = _build_hand_ellipsoid_feed(
            swing,
            active_ellipsoids,
            opts.hand_mobility_mode,
            opts.ellipsoid_scale,
            engine=engine,
        )

    feed = OverlayFeed(
        provider.frame_at,
        style,
        grip_analyses=provider.grip_analyses if grip else None,
        show_force=show_force,
        show_torque=show_torque,
        show_palm_thumb=effective_palm_thumb,
        hand_ellipsoids=hand_ellipsoids_fn,
        hand_hud_lines=hand_hud_lines_fn,
    )
    return feed, lookat


def face_centre_trajectory(
    swing: SwingInput,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Source times ``(n,)`` and face-centre world path ``(n, 3)`` of ``swing``.

    The face centre (#12682) is the ``Clubhead`` frame placed by MuJoCo forward
    kinematics at every state of ``swing.q``, carrying the club's face centre.

    Raises ``BackendUnavailable`` without MuJoCo and ``ValueError`` when the
    specification has no club on the ``Clubhead`` frame.
    """
    try:
        from src.engines.physics_engines.mujoco.python.overlay_source import (
            MujocoOverlaySource,
        )
    except ImportError as exc:
        raise BackendUnavailable(f"mujoco is not available: {exc}") from exc
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
    return np.asarray(swing.source_times_s, dtype=float), head


def detect_impact_time_s(swing: SwingInput) -> float:
    """Impact time of ``swing`` from the shared checked rule (one detector).

    Impact is ``model_appearance.club_face.ball_passage`` (OSV-10) of the
    :func:`face_centre_trajectory`: the sub-sample instant the face centre
    comes closest to its address position, with height and ball-radius checks.

    Raises ``BackendUnavailable`` without MuJoCo and ``ValueError`` when the
    specification has no club on the ``Clubhead`` frame or no frame is a
    valid impact.
    """
    from src.shared.python.model_appearance.club_face import ball_passage

    times, head = face_centre_trajectory(swing)
    t_impact, _, _ = ball_passage(times, head)
    return t_impact
