"""Overlay feed and framing for native exports (needs MuJoCo for the source).

One MuJoCo evaluation of the specification export gives every engine's viewer
the same joint-torque, ground-reaction and weight glyphs (see
``force_overlay.bundle_provider``); velocities come from the bundle reference.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
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
    mobility_points: str = "hands"
    max_mobility_points: int = 8
    mobility_condition_cutoff: float | None = None
    #: ellipsoid colour encodes the point (default) or the kind (#15150)
    ellipsoid_color_by: str = "point"
    #: blank axial loads around ball impact (skin_load_settings.json default)
    axial_impact_blackout: bool = True


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


def _lookup_point_mobility(m: dict[str, Any], pt: Any) -> Any:
    return m.get(pt.name) or (
        m.get("lead" if "lead" in pt.name else "trail") if pt.group == "hands" else None
    )


def _build_hand_hud_lines_fn(
    get_hm: Callable[[int], dict[str, Any]],
    resolved_pts: Sequence[Any],
    active_ellipsoids: Sequence[str],
    scales: dict[str, Any],
    mobility_condition_cutoff: float | None = None,
) -> Callable[[int], tuple[str, ...]]:
    from src.shared.python.force_overlay.ellipsoid_display import format_hud_capability

    is_hands = len(resolved_pts) == 2 and {p.group for p in resolved_pts} == {"hands"}

    def hand_hud_lines_fn(index: int) -> tuple[str, ...]:
        hm_map, lines = get_hm(index), []
        for kind in active_ellipsoids:
            scale, parts = scales[kind], []
            for pt in resolved_pts:
                hm = _lookup_point_mobility(hm_map, pt)
                if hm is not None and (
                    mobility_condition_cutoff is None
                    or hm.condition <= mobility_condition_cutoff
                ):
                    _, r = _extract_physical_radii(hm, kind)
                    lbl = (
                        ("lead" if "lead" in pt.name else "trail")
                        if is_hands
                        else pt.name
                    )
                    parts.append(
                        format_hud_capability(lbl, kind, r, scale.unit, hm.condition)
                    )
            lines.append(" | ".join(parts))
        return tuple(lines)

    return hand_hud_lines_fn


RGBA = tuple[float, float, float, float]


@dataclass(frozen=True)
class MobilitySelection:
    """Which mobility points the ellipsoid feed samples, and how many (#15150)."""

    points: str = "hands"
    max_points: int = 8
    condition_cutoff: float | None = None


def _ellipsoid_color_picker(
    color_by: str,
) -> Callable[[Any, str, RGBA], RGBA]:
    """Colour rule of an export ellipsoid: by ``"point"`` or by ``"kind"`` (#15150).

    Raises ``ValueError`` for any other ``color_by``. The returned function
    maps ``(point, kind, fallback_rgba)`` to the glyph RGBA.
    """
    from src.shared.python.force_overlay.palette import (
        ellipsoid_kind_rgba,
        require_ellipsoid_color_mode,
    )

    if require_ellipsoid_color_mode(color_by) == "kind":
        return lambda _point, kind, _fallback: ellipsoid_kind_rgba(kind)
    return lambda point, _kind, fallback: (
        point.color if point.color is not None else fallback
    )


def _semi_axes_series(
    samples: Iterable[Any], points: Sequence[Any], kind: str
) -> list[list[float]]:
    """Physical semi-axes of ``kind`` at ``points`` over the sampled states."""
    series: list[list[float]] = []
    for sample in samples:
        for pt in points:
            hm = _lookup_point_mobility(sample, pt)
            if hm is not None:
                series.append(list(_extract_physical_radii(hm, kind)[1]))
    return series


def _ellipsoid_feed_factory(
    swing: SwingInput, engine: str, opts: OverlayOptions, active: Sequence[str]
) -> Callable[[SwingInput], tuple[Any, Any]]:
    """Feed builder for any re-sampling of ``swing``, on ``swing``'s scale."""

    def ellipsoids_for(shown: SwingInput) -> tuple[Any, Any]:
        return _build_hand_ellipsoid_feed(
            shown,
            active,
            opts.hand_mobility_mode,
            opts.ellipsoid_scale,
            engine=engine,
            selection=MobilitySelection(
                opts.mobility_points,
                opts.max_mobility_points,
                opts.mobility_condition_cutoff,
            ),
            ellipsoid_color_by=opts.ellipsoid_color_by,
            scale_from=swing,
        )

    return ellipsoids_for


def _build_hand_ellipsoid_feed(
    swing: SwingInput,
    active_ellipsoids: Sequence[str],
    hand_mobility_mode: str,
    ellipsoid_scale: str,
    engine: str = "mujoco",
    selection: MobilitySelection = MobilitySelection(),
    ellipsoid_color_by: str = "point",
    scale_from: SwingInput | None = None,
) -> tuple[
    Callable[[int], tuple[EllipsoidGlyph, ...]], Callable[[int], tuple[str, ...]]
]:
    """Build ellipsoid glyph emitter and HUD capability formatter.

    Index ``k`` of either is evaluated at the pose ``swing.q[k]``. The display
    scale of each kind comes from the states of ``scale_from`` (default
    ``swing``), so the clips re-sampled from one source share it (#15150).
    """
    scale_swing = swing if scale_from is None else scale_from
    pick_color = _ellipsoid_color_picker(ellipsoid_color_by)
    mobility_condition_cutoff = selection.condition_cutoff
    from src.shared.python.force_overlay.ellipsoid_display import (
        EllipsoidDisplayBuilder,
        EllipsoidDisplayScale,
    )
    from src.shared.python.spatial_algebra.segment_intersection import (
        resolve_intersection_points,
    )

    spec_dict = json.loads(swing.bundle.spec_bytes)
    resolved_pts = resolve_intersection_points(
        spec_dict,
        selection.points,
        max_points=selection.max_points,
    )

    _hm_cache: dict[tuple[bool, int], Any] = {}

    def _get_hm(idx: int, of: SwingInput = swing) -> dict[str, Any]:
        key = (of is swing, idx)
        if key not in _hm_cache:
            if hasattr(of, "segment_mobility_at"):
                _hm_cache[key] = of.segment_mobility_at(
                    idx, points=resolved_pts, mode=hand_mobility_mode, engine=engine
                )
            else:
                _hm_cache[key] = of.hand_mobility_at(
                    idx, mode=hand_mobility_mode, engine=engine
                )
        return _hm_cache[key]

    scales: dict[str, EllipsoidDisplayScale] = {}
    stride = max(1, scale_swing.n_states // 20)
    for kind in active_ellipsoids:
        sampled = (
            _get_hm(i, scale_swing) for i in range(0, scale_swing.n_states, stride)
        )
        scales[kind] = EllipsoidDisplayScale.compute(
            _semi_axes_series(sampled, resolved_pts, kind),
            target_radius_m=0.09,
            kind=kind,
            mode=ellipsoid_scale,
        )
    builders = {kind: EllipsoidDisplayBuilder(s) for kind, s in scales.items()}

    def hand_ellipsoids_fn(index: int) -> tuple[EllipsoidGlyph, ...]:
        hm_map = _get_hm(index)
        glyphs_list: list[EllipsoidGlyph] = []
        for kind in active_ellipsoids:
            builder = builders[kind]
            for pt in resolved_pts:
                hm = _lookup_point_mobility(hm_map, pt)
                if hm is None or (
                    mobility_condition_cutoff is not None
                    and hm.condition > mobility_condition_cutoff
                ):
                    continue
                axes, radii_phys = _extract_physical_radii(hm, kind)
                disp = builder.build_glyph(
                    hm.centre,
                    axes,
                    radii_phys,
                    "trail" if "trail" in pt.name else "lead",
                    label=f"{pt.name} {kind}",
                )
                c = pick_color(pt, kind, disp.rgba)
                glyphs_list.append(
                    EllipsoidGlyph(disp.centre, disp.axes, disp.radii, c, disp.label)
                )
        return tuple(glyphs_list)

    hud_fn = _build_hand_hud_lines_fn(
        _get_hm, resolved_pts, active_ellipsoids, scales, mobility_condition_cutoff
    )
    return hand_ellipsoids_fn, hud_fn


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
    modelled (#13179), unless ``options.axial_impact_blackout`` is off: the
    loads there then come from the motion alone (#15150). Options in ``options`` (or ``palm_thumb``) configure
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
        impact = None
        if opts.axial_impact_blackout:
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
    hand_ellipsoids_fn = hand_hud_lines_fn = ellipsoids_for = None
    if active_ellipsoids:
        ellipsoids_for = _ellipsoid_feed_factory(swing, engine, opts, active_ellipsoids)
        hand_ellipsoids_fn, hand_hud_lines_fn = ellipsoids_for(swing)

    feed = OverlayFeed(
        provider.frame_at,
        style,
        grip_analyses=provider.grip_analyses if grip else None,
        show_force=show_force,
        show_torque=show_torque,
        show_palm_thumb=effective_palm_thumb,
        hand_ellipsoids=hand_ellipsoids_fn,
        hand_hud_lines=hand_hud_lines_fn,
        ellipsoids_for=ellipsoids_for,
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
