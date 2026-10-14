"""Pre-impact kernel bundle export and contact-interval playback handoff.

Provides:
- ``export_pre_impact_kernel_bundle``: Exports versioned ``ud.pre_impact_bundle/1``
  compatible swing, grip-wrench, and shaft-elastic states for Tools impact kernels
  (Issue #9703, child of #9700).
- ``kernel_bundle_to_pre_impact_bundle``: Adapts wire bundle back to typed PreImpactBundle.
- ``interpolate_contact_trajectory_frames``: Deterministic time-resolved frame
  interpolation across the contact interval window [t_0, t_1] (Issue #9549, #9700).
- ``export_impact_run_playback_summary``: Integrates the contact-interval solver audit
  (``audit_and_export_impact_run``) into live run, playback, and export summaries,
  failing closed when contact is truncated (``TIME_LIMIT``) or missed (``NO_CONTACT``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.physics._pre_impact_contracts import (
    FieldOrigin,
    Quantity,
    com_inertia,
    fail,
    unit_vector,
)
from src.shared.python.physics.impact_interval_audit import (
    ImpactAuditError,
    ImpactAuditTolerances,
    audit_and_export_impact_run,
    audit_impact_interval,
)
from src.shared.python.physics.pre_impact_bundle import (
    HEAD_FRAME_ID,
    PRE_IMPACT_BUNDLE_SCHEMA,
    PRE_IMPACT_BUNDLE_VERSION,
    WORLD_FRAME_ID,
    BallState,
    HandWrench,
    HeadState,
    ModalBasis,
    Pose as BundlePose,
    PreImpactBundle,
    Provenance,
    ShaftState,
    TimeBase,
)

UD_PRE_IMPACT_BUNDLE_FORMAT = "ud.pre_impact_bundle/1"
UD_PRE_IMPACT_BUNDLE_VERSION = PRE_IMPACT_BUNDLE_VERSION
UD_IMPACT_PLAYBACK_SCHEMA = "ud.impact_playback/1"
UD_IMPACT_PLAYBACK_VERSION = 1


@dataclass(frozen=True)
class ContactPlaybackFrame:
    """Interpolated state of the ball and club at one timestamp during contact."""

    time_s: float
    ball_position_m: tuple[float, float, float]
    club_position_m: tuple[float, float, float]
    club_orientation: tuple[tuple[float, float, float], ...]
    ball_velocity_mps: tuple[float, float, float]
    club_velocity_mps: tuple[float, float, float]
    ball_angular_velocity_rad_s: tuple[float, float, float]
    club_angular_velocity_rad_s: tuple[float, float, float]
    normal_force_n: float
    friction_force_n: float
    compression_m: float
    face_angle_deg: float
    dynamic_loft_deg: float
    twist_angle_rad: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable dictionary for playback timelines and export."""
        return {
            "time_s": self.time_s,
            "ball_position_m": list(self.ball_position_m),
            "club_position_m": list(self.club_position_m),
            "club_orientation": [list(row) for row in self.club_orientation],
            "ball_velocity_mps": list(self.ball_velocity_mps),
            "club_velocity_mps": list(self.club_velocity_mps),
            "ball_angular_velocity_rad_s": list(self.ball_angular_velocity_rad_s),
            "club_angular_velocity_rad_s": list(self.club_angular_velocity_rad_s),
            "normal_force_n": self.normal_force_n,
            "friction_force_n": self.friction_force_n,
            "compression_m": self.compression_m,
            "face_angle_deg": self.face_angle_deg,
            "dynamic_loft_deg": self.dynamic_loft_deg,
            "twist_angle_rad": self.twist_angle_rad,
        }


def _vec3(arr: Any) -> tuple[float, float, float]:
    vals = [float(x) for x in np.asarray(arr, dtype=np.float64).flat[:3]]
    return (vals[0], vals[1], vals[2])


def _mat3(arr: Any) -> tuple[tuple[float, float, float], ...]:
    m = np.asarray(arr, dtype=np.float64).reshape((3, 3))
    return (
        (float(m[0, 0]), float(m[0, 1]), float(m[0, 2])),
        (float(m[1, 0]), float(m[1, 1]), float(m[1, 2])),
        (float(m[2, 0]), float(m[2, 1]), float(m[2, 2])),
    )


def create_default_provenance(
    *,
    equipment_id: str = "driver-synthetic-01",
    ball_id: str = "ball-synthetic-01",
    calibration_id: str = "calib-synthetic-01",
    model_tier: str = "rigid_head_reduced_modal_shaft",
    source_hashes: Sequence[tuple[str, str]] | None = None,
) -> Provenance:
    """Build a valid Provenance record with deterministic synthetic digests."""
    hashes = (
        source_hashes
        if source_hashes is not None
        else (("synthetic_kernel", "0" * 64),)
    )
    return Provenance(
        equipment_id=equipment_id,
        ball_id=ball_id,
        calibration_id=calibration_id,
        model_tier=model_tier,
        source_hashes=tuple(hashes),
    )


def create_default_timebase(
    *,
    event_time_s: float = 0.001,
    interval_s: tuple[float, float] = (0.0005, 0.0015),
    sample_times_s: Sequence[float] | None = None,
) -> TimeBase:
    """Build a valid bounded TimeBase containing event_time_s."""
    times = (
        np.array(sample_times_s, dtype=np.float64)
        if sample_times_s is not None
        else np.array([0.0, 0.0005, 0.001, 0.0015], dtype=np.float64)
    )
    return TimeBase(
        sample_times_s=times,
        event_time_s=event_time_s,
        interval_s=interval_s,
        interpolant="cubic_hermite",
        time_uncertainty_s=Quantity(
            FieldOrigin.SYNTHETIC, np.asarray(1.0e-6, dtype=np.float64)
        ),
        interpolation_error_m=Quantity(
            FieldOrigin.SYNTHETIC, np.asarray(1.0e-5, dtype=np.float64)
        ),
    )


def create_default_modal_shaft(
    *,
    dimension: int = 2,
    basis_id: str = "synthetic-shaft-modal",
    version: str = "1",
    amplitudes: Sequence[float] | None = None,
    velocities: Sequence[float] | None = None,
    axial_stations_m: Sequence[float] | None = None,
) -> ShaftState:
    """Build a physically consistent 2-DOF mass-normalized shaft modal state."""
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    mass = np.eye(dimension)
    stiffness = (
        np.diag([900.0, 25000.0])
        if dimension == 2
        else np.diag([(k + 1) * 1000.0 for k in range(dimension)])
    )
    basis = ModalBasis(
        basis_id=basis_id,
        version=version,
        normalization="mass",
        dimension=dimension,
        generalized_mass=mass,
        generalized_stiffness=stiffness,
    )
    q = (
        np.asarray(amplitudes, dtype=np.float64)
        if amplitudes is not None
        else np.zeros(dimension)
    )
    qd = (
        np.asarray(velocities, dtype=np.float64)
        if velocities is not None
        else np.zeros(dimension)
    )
    stations = (
        np.asarray(axial_stations_m, dtype=np.float64)
        if axial_stations_m is not None
        else np.array([0.0, 0.45, 0.90])
    )
    force = np.array([120.0, 70.0, 20.0])
    return ShaftState(
        basis=basis,
        basis_id=basis_id,
        basis_version=version,
        amplitudes=Quantity(FieldOrigin.SYNTHETIC, q),
        velocities=Quantity(FieldOrigin.SYNTHETIC, qd),
        axial_stations_m=stations,
        axial_force_n=Quantity(FieldOrigin.PRESCRIBED, force),
    )


def create_default_hand_wrenches(
    *,
    lead_force_n: Sequence[float] | None = None,
    lead_moment_n_m: Sequence[float] | None = None,
) -> tuple[HandWrench, ...]:
    """Build standard lead and trail hand grip wrench records in grip frame."""
    force = (
        np.asarray(lead_force_n, dtype=np.float64)
        if lead_force_n is not None
        else np.array([10.0, -30.0, 120.0])
    )
    moment = (
        np.asarray(lead_moment_n_m, dtype=np.float64)
        if lead_moment_n_m is not None
        else np.array([1.5, 0.4, -2.0])
    )
    lead = HandWrench(
        hand="lead",
        frame_id="grip",
        origin_m=np.array([0.0, 0.0, 0.05]),
        force_n=Quantity(FieldOrigin.IDENTIFIED, force),
        moment_n_m=Quantity(FieldOrigin.IDENTIFIED, moment),
        stiffness=Quantity.absent(),
        damping=Quantity.absent(),
        assumptions=("rigid lead hand-grip attachment point",),
    )
    trail = HandWrench(
        hand="trail",
        frame_id="grip",
        origin_m=np.array([0.0, 0.0, 0.14]),
        force_n=Quantity.absent(),
        moment_n_m=Quantity.absent(),
        stiffness=Quantity.absent(),
        damping=Quantity.absent(),
        assumptions=("trail hand wrench absent",),
    )
    return (lead, trail)


def _build_head_section(
    head: HeadState | Mapping[str, Any] | None, swing: Any | None
) -> HeadState:
    """Build HeadState from either an existing HeadState or swing kinematics."""
    if isinstance(head, HeadState):
        return head

    if isinstance(swing, Mapping):
        vel = np.asarray(swing["clubhead_velocity"], dtype=np.float64)
        ang = np.asarray(
            swing.get("clubhead_angular_velocity", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        mass = float(swing.get("clubhead_mass", 0.200))
        norm_raw = np.asarray(
            swing.get("clubhead_orientation", [1.0, 0.0, 0.0]), dtype=np.float64
        )
        normal = norm_raw[0] if norm_raw.ndim == 2 else norm_raw
        normal = normal / np.linalg.norm(normal)
        moi = float(swing.get("clubhead_moi", 4.5e-4))
        off_v = swing.get("impact_offset")
        offset = (
            np.zeros(3)
            if off_v is None
            else np.array(
                [
                    0.0,
                    float(off_v[0]) if len(off_v) > 0 else 0.0,
                    float(off_v[1]) if len(off_v) > 1 else 0.0,
                ]
            )
        )
    elif swing is not None:
        vel = np.asarray(swing.clubhead_velocity, dtype=np.float64)
        ang = np.asarray(
            getattr(swing, "clubhead_angular_velocity", np.zeros(3)),
            dtype=np.float64,
        )
        mass = float(getattr(swing, "clubhead_mass", 0.200))
        orient = np.asarray(
            getattr(swing, "clubhead_orientation", np.array([1.0, 0.0, 0.0])),
            dtype=np.float64,
        )
        normal = orient[0] if orient.ndim == 2 else orient
        normal = normal / np.linalg.norm(normal)
        moi = float(getattr(swing, "clubhead_moi", 4.5e-4))
        off_v = getattr(swing, "impact_offset", None)
        offset = (
            np.zeros(3)
            if off_v is None
            else np.array(
                [
                    0.0,
                    float(off_v[0]) if len(off_v) > 0 else 0.0,
                    float(off_v[1]) if len(off_v) > 1 else 0.0,
                ]
            )
        )
    else:
        vel = np.array([45.0, 0.0, 0.0])
        ang = np.zeros(3)
        mass = 0.200
        normal = np.array([1.0, 0.0, 0.0])
        moi = 4.5e-4
        offset = np.zeros(3)

    if mass <= 0.0 or not math.isfinite(mass):
        fail("out_of_range", f"clubhead mass must be positive, got {mass}")
    if moi <= 0.0 or not math.isfinite(moi):
        fail("out_of_range", f"clubhead MOI must be positive, got {moi}")

    inertia = np.diag([moi, moi, moi])
    com_inertia(inertia, "head.inertia_com_kg_m2")
    unit_vector(normal, "head.contact_normal")

    return HeadState(
        mass_kg=Quantity(FieldOrigin.MEASURED, np.asarray(mass, dtype=np.float64)),
        inertia_com_kg_m2=Quantity(FieldOrigin.IDENTIFIED, inertia),
        linear_velocity_mps=Quantity(FieldOrigin.MEASURED, vel),
        angular_velocity_rad_s=Quantity(FieldOrigin.MEASURED, ang),
        contact_offset_m=Quantity(FieldOrigin.PRESCRIBED, offset),
        contact_normal=Quantity(FieldOrigin.PRESCRIBED, normal),
        face_curvature_1_m=Quantity.absent(),
    )


def _build_ball_section(ball: BallState | Mapping[str, Any] | None) -> BallState:
    """Build BallState from existing object or default values."""
    if isinstance(ball, BallState):
        return ball
    if ball is not None:
        pos = np.asarray(ball.get("position_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        vel = np.asarray(ball.get("velocity_mps", [0.0, 0.0, 0.0]), dtype=np.float64)
        spin = (
            Quantity(
                FieldOrigin.MEASURED,
                np.asarray(ball["spin_rad_s"], dtype=np.float64),
            )
            if ball.get("spin_rad_s") is not None
            else Quantity.absent()
        )
        return BallState(
            position_m=Quantity(FieldOrigin.MEASURED, pos),
            velocity_mps=Quantity(FieldOrigin.MEASURED, vel),
            spin_rad_s=spin,
        )
    return BallState(
        position_m=Quantity(FieldOrigin.MEASURED, np.zeros(3)),
        velocity_mps=Quantity(FieldOrigin.MEASURED, np.zeros(3)),
        spin_rad_s=Quantity.absent(),
    )


def _build_shaft_section(shaft: ShaftState | None) -> ShaftState:
    """Validate or construct default modal shaft state."""
    if isinstance(shaft, ShaftState):
        return shaft
    if shaft is not None:
        raise TypeError(f"shaft must be ShaftState, got {type(shaft).__name__}")
    return create_default_modal_shaft()


def _build_hands_section(
    hands: tuple[HandWrench, ...] | Sequence[HandWrench] | None,
) -> tuple[HandWrench, ...]:
    """Validate or construct default hand wrench tuple."""
    if hands is None:
        return create_default_hand_wrenches()
    for h in hands:
        if not isinstance(h, HandWrench):
            raise TypeError(f"hand must be HandWrench, got {type(h).__name__}")
    return tuple(hands)


def _build_poses_section() -> tuple[BundlePose, BundlePose]:
    """Construct default head and grip poses in world frame."""
    h_pose = BundlePose(
        WORLD_FRAME_ID,
        HEAD_FRAME_ID,
        np.eye(3),
        np.zeros(3),
    )
    g_pose = BundlePose(
        WORLD_FRAME_ID,
        "grip",
        np.eye(3),
        np.array([0.0, 0.90, 0.0]),
    )
    return h_pose, g_pose


def export_pre_impact_kernel_bundle(
    *,
    bundle: PreImpactBundle | Mapping[str, Any] | None = None,
    swing: Any | None = None,
    head: HeadState | Mapping[str, Any] | None = None,
    ball: BallState | Mapping[str, Any] | None = None,
    shaft: ShaftState | None = None,
    hands: tuple[HandWrench, ...] | Sequence[HandWrench] | None = None,
    provenance: Provenance | None = None,
) -> dict[str, Any]:
    """Export versioned ``ud.pre_impact_bundle/1`` compatible kernel bundle (#9703).

    Preconditions:
        Either a valid :class:`PreImpactBundle` (or dict) is provided, or
        valid swing kinematics/components are supplied. All physical invariants
        (positive mass, SPD inertia, proper rotations, monotonic time) are enforced.
    Postconditions:
        Returns a JSON-serialisable dictionary whose format and schema match
        ``ud.pre_impact_bundle/1`` and which passes bundle round-trip checks.
    """
    if bundle is not None:
        if isinstance(bundle, PreImpactBundle):
            pib = bundle
        elif isinstance(bundle, Mapping):
            clean = {k: v for k, v in bundle.items() if k != "format"}
            pib = PreImpactBundle.from_dict(clean)
        else:
            raise TypeError(
                f"bundle must be PreImpactBundle or Mapping, got {type(bundle).__name__}"
            )
    else:
        head_state = _build_head_section(head, swing)
        ball_state = _build_ball_section(ball)
        shaft_state = _build_shaft_section(shaft)
        hand_tuple = _build_hands_section(hands)
        h_pose, g_pose = _build_poses_section()
        tb = create_default_timebase()
        prov = (
            provenance
            if isinstance(provenance, Provenance)
            else create_default_provenance()
        )
        pib = PreImpactBundle(
            schema=PRE_IMPACT_BUNDLE_SCHEMA,
            version=PRE_IMPACT_BUNDLE_VERSION,
            provenance=prov,
            timebase=tb,
            head_pose=h_pose,
            grip_pose=g_pose,
            head=head_state,
            ball=ball_state,
            shaft=shaft_state,
            hands=hand_tuple,
            constraints=("free detached head during contact interval",),
        )

    raw_dict = pib.to_dict()
    raw_dict["format"] = UD_PRE_IMPACT_BUNDLE_FORMAT
    return raw_dict


def kernel_bundle_to_pre_impact_bundle(
    payload: Mapping[str, Any],
) -> PreImpactBundle:
    """Adapt a kernel bundle wire dictionary back to a typed PreImpactBundle."""
    clean = {k: v for k, v in payload.items() if k != "format"}
    return PreImpactBundle.from_dict(clean)


def _interp1d(result: Any, time_arr: np.ndarray, name: str, t_val: float) -> float:
    arr = getattr(result, name, None)
    if arr is None and hasattr(result, "channel"):
        arr = result.channel(name)
    return float(np.interp(t_val, time_arr, arr)) if arr is not None else 0.0


def _interp3d(
    result: Any, time_arr: np.ndarray, name: str, t_val: float
) -> tuple[float, float, float]:
    arr = getattr(result, name, None)
    if arr is None and hasattr(result, "channel"):
        arr = result.channel(name)
    if arr is None:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(arr, dtype=np.float64)
    res = [float(np.interp(t_val, time_arr, arr[:, col])) for col in range(3)]
    return (res[0], res[1], res[2])


def _interpolate_single_frame(
    result: Any, time_arr: np.ndarray, t_val: float
) -> ContactPlaybackFrame:
    """Interpolate all channels for a single timestamp during contact."""
    sample = None
    if hasattr(result, "at_time") and callable(result.at_time):
        try:
            sample = result.at_time(t_val)
        except Exception:
            sample = None

    if sample is not None:
        ball_pos = _vec3(sample.ball_position_m)
        club_pos = _vec3(sample.club_position_m)
        club_orient = _mat3(sample.club_orientation)
        ball_vel = _vec3(sample.ball_velocity_mps)
        club_vel = _vec3(sample.club_velocity_mps)
        ball_ang = _vec3(sample.ball_angular_velocity_rad_s)
        club_ang = _vec3(sample.club_angular_velocity_rad_s)
        normal_force = float(sample.normal_force_n)
        friction_force = float(sample.friction_force_n)
        compression = float(sample.compression_m)
        face_angle = float(sample.face_angle_deg)
        dynamic_loft = float(sample.dynamic_loft_deg)
        twist_angle = float(sample.twist_angle_rad)
    else:
        ball_pos = _interp3d(result, time_arr, "ball_position_m", t_val)
        club_pos = _interp3d(result, time_arr, "club_position_m", t_val)
        ball_vel = _interp3d(result, time_arr, "ball_velocity_mps", t_val)
        club_vel = _interp3d(result, time_arr, "club_velocity_mps", t_val)
        ball_ang = _interp3d(result, time_arr, "ball_angular_velocity_rad_s", t_val)
        club_ang = _interp3d(result, time_arr, "club_angular_velocity_rad_s", t_val)
        normal_force = _interp1d(result, time_arr, "normal_force_n", t_val)
        friction_force = _interp1d(result, time_arr, "friction_force_n", t_val)
        compression = _interp1d(result, time_arr, "compression_m", t_val)
        face_angle = _interp1d(result, time_arr, "face_angle_deg", t_val)
        dynamic_loft = _interp1d(result, time_arr, "dynamic_loft_deg", t_val)
        twist_angle = _interp1d(result, time_arr, "twist_angle_rad", t_val)

        orient_arr = getattr(result, "club_orientation", None)
        if orient_arr is None and hasattr(result, "channel"):
            orient_arr = result.channel("club_orientation")
        if orient_arr is not None:
            orient_arr = np.asarray(orient_arr, dtype=np.float64)
            idx = int(
                np.clip(
                    np.searchsorted(time_arr, t_val),
                    0,
                    len(time_arr) - 1,
                )
            )
            club_orient = _mat3(orient_arr[idx])
        else:
            club_orient = _mat3(np.eye(3))

    return ContactPlaybackFrame(
        time_s=t_val,
        ball_position_m=ball_pos,
        club_position_m=club_pos,
        club_orientation=club_orient,
        ball_velocity_mps=ball_vel,
        club_velocity_mps=club_vel,
        ball_angular_velocity_rad_s=ball_ang,
        club_angular_velocity_rad_s=club_ang,
        normal_force_n=normal_force,
        friction_force_n=friction_force,
        compression_m=compression,
        face_angle_deg=face_angle,
        dynamic_loft_deg=dynamic_loft,
        twist_angle_rad=twist_angle,
    )


def interpolate_contact_trajectory_frames(
    result: Any,
    num_frames: int = 50,
    *,
    tolerances: ImpactAuditTolerances | None = None,
) -> list[ContactPlaybackFrame]:
    """Interpolate trajectory frames across the contact window [t_0, t_1] (#9549).

    Preconditions:
        ``result`` passes :func:`audit_impact_interval` tolerances and contact
        separated. ``num_frames`` >= 2.
    Postconditions:
        Returns a list of length ``num_frames`` containing time-resolved
        ball and club kinematics, forces, and angles across the contact window.
    Raises:
        ImpactAuditError: When the audit fails, contact is truncated (TIME_LIMIT)
            or contact was missed (NO_CONTACT). Fails closed.
    """
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}")

    verdict = audit_impact_interval(result, tolerances)
    if not verdict.qualified:
        raise ImpactAuditError(verdict)

    time_arr = getattr(result, "time_s", None)
    if time_arr is None or len(time_arr) < 2:
        raise ValueError("result must provide time_s array with at least 2 samples")

    t_0 = float(time_arr[0])
    duration = getattr(result, "contact_duration_s", None)
    t_1 = (
        float(duration)
        if duration is not None and duration > t_0
        else float(time_arr[-1])
    )
    if t_1 <= t_0:
        t_1 = float(time_arr[-1])
    if t_1 <= t_0:
        raise ValueError(
            f"invalid contact window: t_0={t_0} must be strictly less than t_1={t_1}"
        )

    query_times = np.linspace(t_0, t_1, num_frames)
    return [_interpolate_single_frame(result, time_arr, float(t)) for t in query_times]


def export_impact_run_playback_summary(
    result: Any,
    *,
    num_playback_frames: int = 50,
    tolerances: ImpactAuditTolerances | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Integrate contact-interval solve into live run, playback, and export summaries.

    Calls :func:`audit_and_export_impact_run` and augments qualified runs with
    interpolated playback frames across [t_0, t_1]. Fails closed when contact
    did not complete or was truncated/missed.
    """
    base_export = audit_and_export_impact_run(result, tolerances=tolerances)
    qualified = bool(base_export["qualified"])

    if qualified:
        frames = interpolate_contact_trajectory_frames(
            result, num_frames=num_playback_frames, tolerances=tolerances
        )
        t_0 = frames[0].time_s
        t_1 = frames[-1].time_s
        return {
            "format": UD_IMPACT_PLAYBACK_SCHEMA,
            "version": UD_IMPACT_PLAYBACK_VERSION,
            "qualified": True,
            "contact_completed": True,
            "termination": base_export["termination"],
            "completion_status": base_export["completion_status"],
            "contact_window_s": [float(t_0), float(t_1)],
            "contact_duration_s": float(t_1 - t_0),
            "frame_count": len(frames),
            "playback_frames": [frame.to_dict() for frame in frames],
            "post_impact_state": base_export["post_impact_state"],
            "audit": base_export["audit"],
        }

    if strict:
        verdict = audit_impact_interval(result, tolerances=tolerances)
        raise ImpactAuditError(verdict)

    return {
        "format": UD_IMPACT_PLAYBACK_SCHEMA,
        "version": UD_IMPACT_PLAYBACK_VERSION,
        "qualified": False,
        "contact_completed": base_export["contact_completed"],
        "termination": base_export["termination"],
        "completion_status": base_export["completion_status"],
        "contact_window_s": None,
        "contact_duration_s": None,
        "frame_count": 0,
        "playback_frames": None,
        "post_impact_state": None,
        "audit": base_export["audit"],
    }


__all__ = [
    "ContactPlaybackFrame",
    "UD_IMPACT_PLAYBACK_SCHEMA",
    "UD_IMPACT_PLAYBACK_VERSION",
    "UD_PRE_IMPACT_BUNDLE_FORMAT",
    "create_default_hand_wrenches",
    "create_default_modal_shaft",
    "create_default_provenance",
    "create_default_timebase",
    "export_impact_run_playback_summary",
    "export_pre_impact_kernel_bundle",
    "interpolate_contact_trajectory_frames",
    "kernel_bundle_to_pre_impact_bundle",
]
