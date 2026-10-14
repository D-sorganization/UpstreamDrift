"""Pre-impact kernel bundle export and contact-interval playback handoff.

Provides:
- ``export_pre_impact_kernel_bundle``: Exports versioned ``ud.pre_impact_bundle/1``
  compatible swing, grip-wrench, and shaft-elastic states for Tools impact kernels
  (Issue #9703, child of #9700).
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
    PreImpactBundleError,
    Quantity,
    com_inertia,
    fail,
    unit_vector,
)
from src.shared.python.physics._pre_impact_frames import (
    Pose,
    wrench_to_parent,
)
from src.shared.python.physics.impact_interval_audit import (
    ImpactAuditError,
    ImpactAuditTolerances,
    audit_and_export_impact_run,
    audit_impact_interval,
)
from src.shared.python.physics.pre_impact_bundle import (
    CONVENTIONS,
    HEAD_FRAME_ID,
    PRE_IMPACT_BUNDLE_SCHEMA,
    PRE_IMPACT_BUNDLE_VERSION,
    UD_PRE_IMPACT_BUNDLE_SCHEMA,
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

UD_PRE_IMPACT_BUNDLE_FORMAT = UD_PRE_IMPACT_BUNDLE_SCHEMA
UD_PRE_IMPACT_BUNDLE_VERSION = PRE_IMPACT_BUNDLE_VERSION
UD_IMPACT_PLAYBACK_SCHEMA = "ud.impact_playback/1"
UD_IMPACT_PLAYBACK_VERSION = 1

_CHANNELS_1D = (
    "normal_force_n",
    "friction_force_n",
    "compression_m",
    "face_angle_deg",
    "dynamic_loft_deg",
    "twist_angle_rad",
)
_CHANNELS_VEC3 = (
    "ball_position_m",
    "club_position_m",
    "ball_velocity_mps",
    "club_velocity_mps",
    "ball_angular_velocity_rad_s",
    "club_angular_velocity_rad_s",
)


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
        np.array(sample_times_s, dtype=float)
        if sample_times_s is not None
        else np.array([0.0, 0.0005, 0.001, 0.0015], dtype=float)
    )
    return TimeBase(
        sample_times_s=times,
        event_time_s=event_time_s,
        interval_s=interval_s,
        interpolant="cubic_hermite",
        time_uncertainty_s=Quantity(FieldOrigin.SYNTHETIC, 1.0e-6),
        interpolation_error_m=Quantity(FieldOrigin.SYNTHETIC, 1.0e-5),
    )


def create_default_modal_shaft(
    *,
    dimension: int = 2,
    basis_id: str = "synthetic-shaft-modal",
    version: str = "1",
    amplitudes: Sequence[float] | None = None,
    velocities: Sequence[float] | None = None,
    axial_stations_m: Sequence[float] | None = None,
    axial_force_n: Sequence[float] | None = None,
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
        np.asarray(amplitudes, dtype=float)
        if amplitudes is not None
        else np.zeros(dimension)
    )
    qd = (
        np.asarray(velocities, dtype=float)
        if velocities is not None
        else np.zeros(dimension)
    )
    stations = (
        np.asarray(axial_stations_m, dtype=float)
        if axial_stations_m is not None
        else np.array([0.0, 0.45, 0.90])
    )
    force = (
        np.asarray(axial_force_n, dtype=float)
        if axial_force_n is not None
        else np.array([120.0, 70.0, 20.0])
    )
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
        np.asarray(lead_force_n, dtype=float)
        if lead_force_n is not None
        else np.array([10.0, -30.0, 120.0])
    )
    moment = (
        np.asarray(lead_moment_n_m, dtype=float)
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


def _head_state_from_swing(swing: Any) -> HeadState:
    """Extract and validate HeadState from a SwingState, PreImpactState or dict."""
    if isinstance(swing, Mapping):
        velocity = np.asarray(swing["clubhead_velocity"], dtype=float)
        angular_vel = np.asarray(
            swing.get("clubhead_angular_velocity", [0.0, 0.0, 0.0]), dtype=float
        )
        mass = float(swing.get("clubhead_mass", 0.200))
        normal = np.asarray(
            swing.get("clubhead_orientation", [1.0, 0.0, 0.0]), dtype=float
        )
        if normal.ndim == 2:
            normal = normal[0]
        normal = normal / np.linalg.norm(normal)
        moi = float(swing.get("clubhead_moi", 4.5e-4))
        offset_val = swing.get("impact_offset")
        offset = (
            np.zeros(3)
            if offset_val is None
            else np.array(
                [
                    0.0,
                    float(offset_val[0]) if len(offset_val) > 0 else 0.0,
                    float(offset_val[1]) if len(offset_val) > 1 else 0.0,
                ]
            )
        )
    else:
        velocity = np.asarray(swing.clubhead_velocity, dtype=float)
        angular_vel = np.asarray(
            getattr(swing, "clubhead_angular_velocity", np.zeros(3)), dtype=float
        )
        mass = float(getattr(swing, "clubhead_mass", 0.200))
        orient = np.asarray(
            getattr(swing, "clubhead_orientation", np.array([1.0, 0.0, 0.0])),
            dtype=float,
        )
        normal = orient[0] if orient.ndim == 2 else orient
        norm_val = np.linalg.norm(normal)
        normal = normal / norm_val if norm_val > 0.0 else np.array([1.0, 0.0, 0.0])
        moi = float(getattr(swing, "clubhead_moi", 4.5e-4))
        offset_val = getattr(swing, "impact_offset", None)
        offset = (
            np.zeros(3)
            if offset_val is None
            else np.array(
                [
                    0.0,
                    float(offset_val[0]) if len(offset_val) > 0 else 0.0,
                    float(offset_val[1]) if len(offset_val) > 1 else 0.0,
                ]
            )
        )

    if mass <= 0.0 or not math.isfinite(mass):
        fail("out_of_range", f"clubhead mass must be positive, got {mass}")
    if moi <= 0.0 or not math.isfinite(moi):
        fail("out_of_range", f"clubhead MOI must be positive, got {moi}")

    inertia = np.diag([moi, moi, moi])
    com_inertia(inertia, "head.inertia_com_kg_m2")
    unit_vector(normal, "head.contact_normal")

    return HeadState(
        mass_kg=Quantity(FieldOrigin.MEASURED, mass),
        inertia_com_kg_m2=Quantity(FieldOrigin.IDENTIFIED, inertia),
        linear_velocity_mps=Quantity(FieldOrigin.MEASURED, velocity),
        angular_velocity_rad_s=Quantity(FieldOrigin.MEASURED, angular_vel),
        contact_offset_m=Quantity(FieldOrigin.PRESCRIBED, offset),
        contact_normal=Quantity(FieldOrigin.PRESCRIBED, normal),
        face_curvature_1_m=Quantity.absent(),
    )


def export_pre_impact_kernel_bundle(
    *,
    bundle: PreImpactBundle | Mapping[str, Any] | None = None,
    swing: Any | None = None,
    head: HeadState | Mapping[str, Any] | None = None,
    ball: BallState | Mapping[str, Any] | None = None,
    shaft: ShaftState | Mapping[str, Any] | None = None,
    hands: tuple[HandWrench, ...]
    | Sequence[HandWrench]
    | Sequence[Mapping[str, Any]]
    | None = None,
    head_pose: BundlePose | Mapping[str, Any] | None = None,
    grip_pose: BundlePose | Mapping[str, Any] | None = None,
    timebase: TimeBase | Mapping[str, Any] | None = None,
    provenance: Provenance | Mapping[str, Any] | None = None,
    constraints: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export versioned ``ud.pre_impact_bundle/1`` compatible kernel bundle (#9703).

    Preconditions:
        Either a valid :class:`PreImpactBundle` (or dict) is provided, or
        valid swing kinematics/components are supplied. All physical invariants
        (positive mass, SPD inertia, proper rotations, monotonic time) are enforced.
    Postconditions:
        Returns a JSON-serialisable dictionary whose schema and format match
        ``ud.pre_impact_bundle/1`` and which passes bundle round-trip checks.
    """
    if bundle is not None:
        if isinstance(bundle, PreImpactBundle):
            pib = bundle
        elif isinstance(bundle, Mapping):
            pib = PreImpactBundle.from_dict(bundle)
        else:
            raise TypeError(
                f"bundle must be PreImpactBundle or Mapping, got {type(bundle).__name__}"
            )
    else:
        # Construct HeadState
        if head is not None:
            if isinstance(head, HeadState):
                head_state = head
            else:
                head_state = _head_state_from_swing(head)
        elif swing is not None:
            head_state = _head_state_from_swing(swing)
        else:
            head_state = _head_state_from_swing(
                {
                    "clubhead_velocity": [45.0, 0.0, 0.0],
                    "clubhead_angular_velocity": [0.0, 0.0, 0.0],
                    "clubhead_mass": 0.200,
                    "clubhead_moi": 4.5e-4,
                }
            )

        # BallState
        if ball is not None:
            if isinstance(ball, BallState):
                ball_state = ball
            else:
                ball_state = BallState(
                    position_m=Quantity(
                        FieldOrigin.MEASURED,
                        np.asarray(ball.get("position_m", [0.0, 0.0, 0.0])),
                    ),
                    velocity_mps=Quantity(
                        FieldOrigin.MEASURED,
                        np.asarray(ball.get("velocity_mps", [0.0, 0.0, 0.0])),
                    ),
                    spin_rad_s=(
                        Quantity(
                            FieldOrigin.MEASURED,
                            np.asarray(ball["spin_rad_s"]),
                        )
                        if ball.get("spin_rad_s") is not None
                        else Quantity.absent()
                    ),
                )
        else:
            ball_state = BallState(
                position_m=Quantity(FieldOrigin.MEASURED, np.zeros(3)),
                velocity_mps=Quantity(FieldOrigin.MEASURED, np.zeros(3)),
                spin_rad_s=Quantity.absent(),
            )

        # ShaftState
        if shaft is not None:
            if isinstance(shaft, ShaftState):
                shaft_state = shaft
            else:
                raise TypeError(f"shaft must be ShaftState, got {type(shaft).__name__}")
        else:
            shaft_state = create_default_modal_shaft()

        # Hands
        if hands is not None:
            hand_list: list[HandWrench] = []
            for h in hands:
                if isinstance(h, HandWrench):
                    hand_list.append(h)
                else:
                    raise TypeError(f"hand must be HandWrench, got {type(h).__name__}")
            hand_tuple = tuple(hand_list)
        else:
            hand_tuple = create_default_hand_wrenches()

        # Poses
        h_pose = (
            head_pose
            if isinstance(head_pose, BundlePose)
            else (
                BundlePose(
                    WORLD_FRAME_ID,
                    HEAD_FRAME_ID,
                    np.eye(3),
                    np.zeros(3),
                )
                if head_pose is None
                else BundlePose(
                    WORLD_FRAME_ID,
                    HEAD_FRAME_ID,
                    np.asarray(head_pose["rotation"]),
                    np.asarray(head_pose["translation_m"]),
                )
            )
        )
        g_pose = (
            grip_pose
            if isinstance(grip_pose, BundlePose)
            else (
                BundlePose(
                    WORLD_FRAME_ID,
                    "grip",
                    np.eye(3),
                    np.array([0.0, 0.90, 0.0]),
                )
                if grip_pose is None
                else BundlePose(
                    WORLD_FRAME_ID,
                    "grip",
                    np.asarray(grip_pose["rotation"]),
                    np.asarray(grip_pose["translation_m"]),
                )
            )
        )

        tb = timebase if isinstance(timebase, TimeBase) else create_default_timebase()
        prov = (
            provenance
            if isinstance(provenance, Provenance)
            else create_default_provenance()
        )
        c_tuple = (
            tuple(constraints)
            if constraints is not None
            else ("free detached head during contact interval",)
        )

        pib = PreImpactBundle(
            schema=UD_PRE_IMPACT_BUNDLE_SCHEMA,
            version=UD_PRE_IMPACT_BUNDLE_VERSION,
            provenance=prov,
            timebase=tb,
            head_pose=h_pose,
            grip_pose=g_pose,
            head=head_state,
            ball=ball_state,
            shaft=shaft_state,
            hands=hand_tuple,
            constraints=c_tuple,
        )

    # Convert to dict and ensure format and schema match ud.pre_impact_bundle/1
    raw_dict = pib.to_dict()
    raw_dict["format"] = UD_PRE_IMPACT_BUNDLE_SCHEMA
    raw_dict["schema"] = UD_PRE_IMPACT_BUNDLE_SCHEMA
    raw_dict["version"] = UD_PRE_IMPACT_BUNDLE_VERSION
    return raw_dict


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
    arr = np.asarray(arr, dtype=float)
    res = [float(np.interp(t_val, time_arr, arr[:, col])) for col in range(3)]
    return (res[0], res[1], res[2])


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

    # Determine contact window [t_0, t_1]
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
    frames: list[ContactPlaybackFrame] = []

    for t in query_times:
        t_val = float(t)
        sample = None
        if hasattr(result, "at_time") and callable(result.at_time):
            try:
                sample = result.at_time(t_val)
            except Exception:
                sample = None

        if sample is not None:
            ball_pos = tuple(
                float(x) for x in np.asarray(sample.ball_position_m).flat[:3]
            )
            club_pos = tuple(
                float(x) for x in np.asarray(sample.club_position_m).flat[:3]
            )
            orient = np.asarray(sample.club_orientation, dtype=float)
            club_orient = tuple(
                tuple(float(val) for val in row) for row in orient.reshape((3, 3))
            )
            ball_vel = tuple(
                float(x) for x in np.asarray(sample.ball_velocity_mps).flat[:3]
            )
            club_vel = tuple(
                float(x) for x in np.asarray(sample.club_velocity_mps).flat[:3]
            )
            ball_ang = tuple(
                float(x)
                for x in np.asarray(sample.ball_angular_velocity_rad_s).flat[:3]
            )
            club_ang = tuple(
                float(x)
                for x in np.asarray(sample.club_angular_velocity_rad_s).flat[:3]
            )
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
                orient_arr = np.asarray(orient_arr, dtype=float)
                idx = int(
                    np.clip(
                        np.searchsorted(time_arr, t_val),
                        0,
                        len(time_arr) - 1,
                    )
                )
                mat = orient_arr[idx].reshape((3, 3))
                club_orient = tuple(tuple(float(v) for v in row) for row in mat)
            else:
                club_orient = (
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                )

        frames.append(
            ContactPlaybackFrame(
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
        )

    return frames


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
]
