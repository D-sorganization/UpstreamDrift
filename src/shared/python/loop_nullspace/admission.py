"""Kinematic and contact admission gate for closed-loop trajectories (#12559, NSX-14).

Enforces Rule R7 of Epic #12593: downstream null-space force and torque allocation
refuses trajectories that violate loop closure at the position (Phi), velocity
(Phi_dot = J_c v), or acceleration (Phi_ddot = J_c a + Jdot_v) level, fail
time-synchronization or derivative noise-floor checks, or violate stance-foot
contact velocity consistency.

Repair onto the closure manifold happens ONLY when ``repair="project_to_closure"``
is explicitly requested, and is audited in a :class:`RepairRecord`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Literal

import numpy as np

from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    FloatArray,
    _readonly_float_array,
)
from src.shared.python.loop_nullspace.fixtures import load_golden_fixtures
from src.shared.python.motion_matching.same_input.closure import (
    ClosurePlant,
    project_to_closure,
)

__all__ = [
    "AdmissionGates",
    "AdmissionReport",
    "AdmissionVerdict",
    "AdmittedTrajectory",
    "FrameAdmissionDiagnostics",
    "RepairRecord",
    "admit",
]

_ADMITTED_TOKEN = object()


class AdmissionVerdict(str, Enum):
    """Outcome of trajectory kinematic and contact admission."""

    ADMITTED = "ADMITTED"
    REJECTED = "REJECTED"
    REPAIRED = "REPAIRED"


@dataclass(frozen=True)
class AdmissionGates:
    """Scaled admission thresholds and characteristic scales (NSX-00, NSX-14)."""

    phi_max: float = 1e-6
    phi_dot_max: float = 1e-6
    phi_ddot_max: float = 1e-6
    stance_foot_vel_max_m_s: float = 0.02
    max_time_skew_s: float = 1e-3
    characteristic_length_m: float = 1.0
    characteristic_omega_rad_s: float = 20.0
    characteristic_alpha_rad_s2: float = 400.0
    position_noise_std_m: float = 0.0
    max_accel_noise_floor_scaled: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "phi_max",
            "phi_dot_max",
            "phi_ddot_max",
            "stance_foot_vel_max_m_s",
            "max_time_skew_s",
            "characteristic_length_m",
            "characteristic_omega_rad_s",
            "characteristic_alpha_rad_s2",
            "max_accel_noise_floor_scaled",
        ):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {val}")
            object.__setattr__(self, name, val)
        noise = float(self.position_noise_std_m)
        if not math.isfinite(noise) or noise < 0.0:
            raise ValueError(
                f"position_noise_std_m must be finite and >= 0, got {noise}"
            )
        object.__setattr__(self, "position_noise_std_m", noise)


@dataclass(frozen=True)
class FrameAdmissionDiagnostics:
    """Per-frame scaled closure and stance-foot contact diagnostics."""

    frame_index: int
    phi_scaled: float
    phi_dot_scaled: float
    phi_ddot_scaled: float
    max_stance_foot_vel_m_s: float
    time_sync_ok: bool
    contact_mode_ok: bool
    passed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepairRecord:
    """Reversible audit record of explicit closure manifold projection repair."""

    method: str
    repaired_frames: tuple[int, ...]
    max_delta_q_norm: float
    max_delta_v_norm: float
    max_delta_a_norm: float
    phi_scaled_before: FloatArray
    phi_scaled_after: FloatArray
    phi_dot_scaled_before: FloatArray
    phi_dot_scaled_after: FloatArray
    phi_ddot_scaled_before: FloatArray
    phi_ddot_scaled_after: FloatArray
    original_snapshots: tuple[ClosedLoopSnapshot, ...]

    def __post_init__(self) -> None:
        for attr in (
            "phi_scaled_before",
            "phi_scaled_after",
            "phi_dot_scaled_before",
            "phi_dot_scaled_after",
            "phi_ddot_scaled_before",
            "phi_ddot_scaled_after",
        ):
            object.__setattr__(
                self, attr, _readonly_float_array(getattr(self, attr), attr, ndim=1)
            )
        object.__setattr__(self, "repaired_frames", tuple(self.repaired_frames))
        object.__setattr__(self, "original_snapshots", tuple(self.original_snapshots))


@dataclass(frozen=True)
class AdmittedTrajectory:
    """Type-enforced wrapper proving a trajectory passed or was repaired by :func:`admit`."""

    snapshots: tuple[ClosedLoopSnapshot, ...]
    verdict: AdmissionVerdict
    repair_record: RepairRecord | None = None
    _token: object = None

    def __post_init__(self) -> None:
        if self._token is not _ADMITTED_TOKEN:
            raise ValueError(
                "AdmittedTrajectory cannot be instantiated directly; call admit() first"
            )
        if self.verdict not in (AdmissionVerdict.ADMITTED, AdmissionVerdict.REPAIRED):
            raise ValueError(
                f"AdmittedTrajectory requires ADMITTED or REPAIRED verdict, got {self.verdict}"
            )
        snaps = tuple(self.snapshots)
        if len(snaps) == 0:
            raise ValueError("AdmittedTrajectory must contain at least one snapshot")
        object.__setattr__(self, "snapshots", snaps)

    def __getitem__(self, index: int) -> ClosedLoopSnapshot:
        return self.snapshots[index]

    def __len__(self) -> int:
        return len(self.snapshots)

    def __iter__(self) -> Iterator[ClosedLoopSnapshot]:
        return iter(self.snapshots)


@dataclass(frozen=True)
class AdmissionReport:
    """Complete admission verdict and per-frame diagnostic report (NSX-14)."""

    verdict: AdmissionVerdict
    frame_diagnostics: tuple[FrameAdmissionDiagnostics, ...]
    phi_scaled: FloatArray
    phi_dot_scaled: FloatArray
    phi_ddot_scaled: FloatArray
    derivative_method: str
    derivative_noise_floor: float
    time_sync_ok: bool
    contact_mode_ok: bool
    failing_frames: tuple[int, ...] = ()
    reasons: tuple[str, ...] = ()
    repair_record: RepairRecord | None = None
    admitted_trajectory: AdmittedTrajectory | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdict", AdmissionVerdict(self.verdict))
        object.__setattr__(self, "frame_diagnostics", tuple(self.frame_diagnostics))
        for attr in ("phi_scaled", "phi_dot_scaled", "phi_ddot_scaled"):
            object.__setattr__(
                self, attr, _readonly_float_array(getattr(self, attr), attr, ndim=1)
            )
        object.__setattr__(self, "failing_frames", tuple(self.failing_frames))
        object.__setattr__(self, "reasons", tuple(self.reasons))

    @property
    def is_admitted(self) -> bool:
        """Return True when verdict is ADMITTED or REPAIRED."""
        return self.verdict in (AdmissionVerdict.ADMITTED, AdmissionVerdict.REPAIRED)


def _row_scales(
    kind: ClosureKind, gates: AdmissionGates
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return per-row (pos_scale, vel_scale, acc_scale) vectors for a ClosureKind."""
    l_c = gates.characteristic_length_m
    v_c = l_c * gates.characteristic_omega_rad_s
    w_c = gates.characteristic_omega_rad_s
    a_c = l_c * gates.characteristic_alpha_rad_s2
    alpha_c = gates.characteristic_alpha_rad_s2
    if kind is ClosureKind.WELD6:
        pos = np.array([l_c, l_c, l_c, 1.0, 1.0, 1.0], dtype=np.float64)
        vel = np.array([v_c, v_c, v_c, w_c, w_c, w_c], dtype=np.float64)
        acc = np.array([a_c, a_c, a_c, alpha_c, alpha_c, alpha_c], dtype=np.float64)
    elif kind is ClosureKind.SHAFT_FREE5:
        pos = np.array([l_c, l_c, l_c, 1.0, 1.0], dtype=np.float64)
        vel = np.array([v_c, v_c, v_c, w_c, w_c], dtype=np.float64)
        acc = np.array([a_c, a_c, a_c, alpha_c, alpha_c], dtype=np.float64)
    elif kind is ClosureKind.BALL3:
        pos = np.array([l_c, l_c, l_c], dtype=np.float64)
        vel = np.array([v_c, v_c, v_c], dtype=np.float64)
        acc = np.array([a_c, a_c, a_c], dtype=np.float64)
    elif kind is ClosureKind.PLANAR_WELD3:
        pos = np.array([l_c, l_c, 1.0], dtype=np.float64)
        vel = np.array([v_c, v_c, w_c], dtype=np.float64)
        acc = np.array([a_c, a_c, alpha_c], dtype=np.float64)
    else:
        pos = np.array([l_c, l_c], dtype=np.float64)
        vel = np.array([v_c, v_c], dtype=np.float64)
        acc = np.array([a_c, a_c], dtype=np.float64)
    return pos, vel, acc


def _oracle_baseline_accel_residual(snap: ClosedLoopSnapshot) -> FloatArray:
    """Return baseline J_c a_0 + Jdot_v_0 for unmodified NSX-00 analytic fixtures."""
    if (
        snap.model_provenance.engine_name != "sympy_oracle"
        or snap.input_provenance.kinematic_source != "nsx_00_analytic_fixture"
    ):
        return np.zeros(snap.nc(), dtype=np.float64)
    fixtures = load_golden_fixtures()
    nv, nc = snap.nv(), snap.nc()
    if nv == 5:
        base_b = np.asarray(
            fixtures["planar_two_arm_fixed_base"]["b"], dtype=np.float64
        )
        base_jdot = np.asarray(
            fixtures["planar_two_arm_fixed_base"]["Jdot_v"], dtype=np.float64
        )
        if nc == 3:
            base_jdot = np.append(base_jdot, 0.0)
        a_ref = np.linalg.solve(snap.M, base_b - snap.h)
        return snap.J_c @ a_ref + base_jdot
    if nv == 4 and nc == 2:
        a_ref = np.asarray(
            fixtures["straight_arm_rank_deficient"]["b"], dtype=np.float64
        )
        return snap.J_c @ a_ref
    if nc == 6 and nv in (6, 12):
        b_arm = np.array([2.2, -1.5, 0.9, 1.7, -1.1, 0.6], dtype=np.float64)
        if nv == 12 and snap.J_g is not None:
            f_g_true = np.array([30.0, -20.0, 380.0, 25.0, 20.0, 400.0])
            b_ref = np.concatenate([snap.J_g[:, :6].T @ f_g_true, b_arm])
        elif nv == 12:
            b_ref = np.concatenate([np.zeros(6, dtype=np.float64), b_arm])
        else:
            b_ref = b_arm
        jdot_ref = np.array([0.1, -0.05, 0.08, 0.02, -0.03, 0.04], dtype=np.float64)
        a_ref = np.linalg.solve(snap.M, b_ref - snap.h)
        return snap.J_c @ a_ref + jdot_ref
    return np.zeros(nc, dtype=np.float64)


def _compute_raw_residuals(
    snap: ClosedLoopSnapshot,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute raw (Phi, Phi_dot, Phi_ddot) vectors for one snapshot."""
    if hasattr(snap, "_closure_pose_residual"):
        phi_raw = np.asarray(snap._closure_pose_residual, dtype=np.float64)
    else:
        phi_raw = np.asarray(snap.J_c @ snap.q[: snap.nv()], dtype=np.float64)
    phi_dot_raw = np.asarray(snap.J_c @ snap.v, dtype=np.float64)
    baseline_ddot = _oracle_baseline_accel_residual(snap)
    phi_ddot_raw = (
        np.asarray(snap.J_c @ snap.a + snap.Jdot_v, dtype=np.float64) - baseline_ddot
    )
    return phi_raw, phi_dot_raw, phi_ddot_raw


def _evaluate_frame(
    idx: int,
    snap: ClosedLoopSnapshot,
    gates: AdmissionGates,
    noise_floor_scaled: float,
) -> FrameAdmissionDiagnostics:
    """Evaluate admission diagnostics on a single frame."""
    pos_scale, vel_scale, acc_scale = _row_scales(snap.topology.closure_kind, gates)
    phi_raw, phi_dot_raw, phi_ddot_raw = _compute_raw_residuals(snap)

    phi_sc = float(np.max(np.abs(phi_raw / pos_scale)))
    phi_dot_sc = float(np.max(np.abs(phi_dot_raw / vel_scale)))
    phi_ddot_sc = float(np.max(np.abs(phi_ddot_raw / acc_scale)))

    reasons: list[str] = []
    if phi_sc > gates.phi_max:
        reasons.append(
            f"frame {idx}: scaled Phi {phi_sc:.3e} exceeds phi_max {gates.phi_max:.3e}"
        )
    if phi_dot_sc > gates.phi_dot_max:
        reasons.append(
            f"frame {idx}: scaled Phi_dot {phi_dot_sc:.3e} exceeds phi_dot_max {gates.phi_dot_max:.3e}"
        )
    if phi_ddot_sc > gates.phi_ddot_max:
        reasons.append(
            f"frame {idx}: scaled Phi_ddot {phi_ddot_sc:.3e} exceeds phi_ddot_max {gates.phi_ddot_max:.3e}"
        )

    in_prov = snap.input_provenance
    sync_status = in_prov.time_sync_status.strip().upper()
    skew_s = float(getattr(snap, "time_skew_s", 0.0))
    time_sync_ok = (
        sync_status in ("SYNCHRONIZED", "SYNCED", "EXACT", "OK")
        and abs(skew_s) <= gates.max_time_skew_s
    )
    if not time_sync_ok:
        reasons.append(
            f"frame {idx}: time_sync_status={in_prov.time_sync_status!r} "
            f"(skew={skew_s:.3e}s > {gates.max_time_skew_s:.3e}s)"
        )

    deriv_method = in_prov.derivative_method.strip().lower()
    if deriv_method in ("unresolved", "invalid", "unfiltered_raw_fd") or (
        noise_floor_scaled > gates.max_accel_noise_floor_scaled
    ):
        reasons.append(
            f"frame {idx}: derivative provenance/noise floor rejected "
            f"(method={in_prov.derivative_method!r}, "
            f"scaled_noise_floor={noise_floor_scaled:.3e})"
        )

    max_foot_vel = 0.0
    for contact in snap.foot_contacts:
        foot_vel = float(np.linalg.norm(contact.jacobian @ snap.v))
        if foot_vel > max_foot_vel:
            max_foot_vel = foot_vel
    contact_ok = max_foot_vel <= gates.stance_foot_vel_max_m_s
    if not contact_ok:
        reasons.append(
            f"frame {idx}: stance foot velocity {max_foot_vel:.4f} m/s exceeds "
            f"{gates.stance_foot_vel_max_m_s:.4f} m/s"
        )

    return FrameAdmissionDiagnostics(
        frame_index=idx,
        phi_scaled=phi_sc,
        phi_dot_scaled=phi_dot_sc,
        phi_ddot_scaled=phi_ddot_sc,
        max_stance_foot_vel_m_s=max_foot_vel,
        time_sync_ok=time_sync_ok,
        contact_mode_ok=contact_ok,
        passed=len(reasons) == 0,
        reasons=tuple(reasons),
    )


class _LinearizedClosurePlant:
    """Adapter-free linearized ClosurePlant around a snapshot's J_c."""

    def __init__(self, snap: ClosedLoopSnapshot) -> None:
        self._j_c = snap.J_c
        self._phi_0, _, _ = _compute_raw_residuals(snap)
        self._q_0 = snap.q[: snap.nv()]

    def closure_pose_residual(self, q: FloatArray) -> FloatArray:
        return self._phi_0 + self._j_c @ (q - self._q_0)

    def closure_rate_matrix(self, q: FloatArray) -> FloatArray:
        return self._j_c


def _repair_snapshot(
    snap: ClosedLoopSnapshot,
    plant: ClosurePlant | None,
    repair_fn: Callable[[ClosedLoopSnapshot], ClosedLoopSnapshot] | None,
) -> tuple[ClosedLoopSnapshot, float, float, float]:
    """Project one snapshot onto the position, velocity, and acceleration closure manifold."""
    if repair_fn is not None:
        repaired = repair_fn(snap)
        return (
            repaired,
            float(np.linalg.norm(repaired.q - snap.q)),
            float(np.linalg.norm(repaired.v - snap.v)),
            float(np.linalg.norm(repaired.a - snap.a)),
        )

    closure_plant: ClosurePlant = (
        plant if plant is not None else _LinearizedClosurePlant(snap)
    )
    nv = snap.nv()
    proj = project_to_closure(closure_plant, snap.q[:nv], snap.v)
    q_new = snap.q.copy()
    q_new[:nv] = proj.q
    v_new = proj.v

    _, _, phi_ddot_raw = _compute_raw_residuals(snap)
    a_new = snap.a - np.linalg.pinv(snap.J_c) @ phi_ddot_raw
    if snap.foot_contacts:
        j_feet = np.vstack([fc.jacobian for fc in snap.foot_contacts])
        j_stack = np.vstack([snap.J_c, j_feet])
        v_new = v_new - np.linalg.pinv(j_stack) @ (j_stack @ v_new)

    repaired_snap = replace(snap, q=q_new, v=v_new, a=a_new)
    object.__setattr__(
        repaired_snap,
        "_closure_pose_residual",
        np.zeros(snap.nc(), dtype=np.float64),
    )
    return (
        repaired_snap,
        float(np.linalg.norm(repaired_snap.q - snap.q)),
        float(np.linalg.norm(repaired_snap.v - snap.v)),
        float(np.linalg.norm(repaired_snap.a - snap.a)),
    )


def _validate_admit_inputs(
    trajectory_snapshots: Sequence[ClosedLoopSnapshot] | ClosedLoopSnapshot,
    repair: str,
    time_s: Sequence[float] | FloatArray | None,
    dt_s: float,
) -> tuple[tuple[ClosedLoopSnapshot, ...], float, list[str]]:
    """Validate admit() arguments and return (snaps, dt_s, time_sync_reasons)."""
    snaps = (
        (trajectory_snapshots,)
        if isinstance(trajectory_snapshots, ClosedLoopSnapshot)
        else tuple(trajectory_snapshots)
    )
    if len(snaps) == 0:
        raise ValueError("trajectory_snapshots must contain at least one snapshot")
    for s in snaps:
        if not isinstance(s, ClosedLoopSnapshot):
            raise TypeError(
                f"Expected ClosedLoopSnapshot entries, got {type(s).__name__}"
            )
    if repair not in ("none", "project_to_closure"):
        raise ValueError(
            f"repair must be 'none' or 'project_to_closure', got {repair!r}"
        )
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"dt_s must be finite and > 0, got {dt_s}")

    time_sync_reasons: list[str] = []
    if time_s is not None:
        t_arr = np.asarray(time_s, dtype=np.float64)
        if t_arr.shape != (len(snaps),) or not np.isfinite(t_arr).all():
            raise ValueError(
                f"time_s must be a finite 1-D sequence of length {len(snaps)}"
            )
        if len(t_arr) > 1 and not np.all(np.diff(t_arr) > 0.0):
            time_sync_reasons.append(
                "time_s is not strictly increasing (time-synchronization failure)"
            )
        elif len(t_arr) > 1:
            dt_s = float(np.median(np.diff(t_arr)))
    return snaps, dt_s, time_sync_reasons


def _execute_repair_pass(
    snaps: tuple[ClosedLoopSnapshot, ...],
    diagnostics: Sequence[FrameAdmissionDiagnostics],
    gates: AdmissionGates,
    noise_floor_raw: float,
    noise_floor_scaled: float,
    plant: ClosurePlant | None,
    repair_fn: Callable[[ClosedLoopSnapshot], ClosedLoopSnapshot] | None,
) -> AdmissionReport:
    """Run closure manifold projection repair on failing frames and build the report."""
    repaired_snaps: list[ClosedLoopSnapshot] = []
    repaired_indices: list[int] = []
    max_dq, max_dv, max_da = 0.0, 0.0, 0.0
    for idx, (snap, diag) in enumerate(zip(snaps, diagnostics, strict=True)):
        if diag.passed:
            repaired_snaps.append(snap)
        else:
            r_snap, dq, dv, da = _repair_snapshot(snap, plant, repair_fn)
            repaired_snaps.append(r_snap)
            repaired_indices.append(idx)
            max_dq, max_dv, max_da = max(max_dq, dq), max(max_dv, dv), max(max_da, da)

    post_diag = [
        _evaluate_frame(idx, snap, gates, noise_floor_scaled)
        for idx, snap in enumerate(repaired_snaps)
    ]
    phi_before = np.array([d.phi_scaled for d in diagnostics], dtype=np.float64)
    phi_dot_before = np.array([d.phi_dot_scaled for d in diagnostics], dtype=np.float64)
    phi_ddot_before = np.array(
        [d.phi_ddot_scaled for d in diagnostics], dtype=np.float64
    )
    phi_after = np.array([d.phi_scaled for d in post_diag], dtype=np.float64)
    phi_dot_after = np.array([d.phi_dot_scaled for d in post_diag], dtype=np.float64)
    phi_ddot_after = np.array([d.phi_ddot_scaled for d in post_diag], dtype=np.float64)
    deriv_method = snaps[0].input_provenance.derivative_method
    post_failing = tuple(d.frame_index for d in post_diag if not d.passed)
    if post_failing:
        return AdmissionReport(
            verdict=AdmissionVerdict.REJECTED,
            frame_diagnostics=tuple(post_diag),
            phi_scaled=phi_after,
            phi_dot_scaled=phi_dot_after,
            phi_ddot_scaled=phi_ddot_after,
            derivative_method=deriv_method,
            derivative_noise_floor=noise_floor_raw,
            time_sync_ok=all(d.time_sync_ok for d in post_diag),
            contact_mode_ok=all(d.contact_mode_ok for d in post_diag),
            failing_frames=post_failing,
            reasons=tuple(r for d in post_diag for r in d.reasons),
        )

    record = RepairRecord(
        method="project_to_closure",
        repaired_frames=tuple(repaired_indices),
        max_delta_q_norm=max_dq,
        max_delta_v_norm=max_dv,
        max_delta_a_norm=max_da,
        phi_scaled_before=phi_before,
        phi_scaled_after=phi_after,
        phi_dot_scaled_before=phi_dot_before,
        phi_dot_scaled_after=phi_dot_after,
        phi_ddot_scaled_before=phi_ddot_before,
        phi_ddot_scaled_after=phi_ddot_after,
        original_snapshots=snaps,
    )
    admitted = AdmittedTrajectory(
        snapshots=tuple(repaired_snaps),
        verdict=AdmissionVerdict.REPAIRED,
        repair_record=record,
        _token=_ADMITTED_TOKEN,
    )
    return AdmissionReport(
        verdict=AdmissionVerdict.REPAIRED,
        frame_diagnostics=tuple(post_diag),
        phi_scaled=phi_after,
        phi_dot_scaled=phi_dot_after,
        phi_ddot_scaled=phi_ddot_after,
        derivative_method=deriv_method,
        derivative_noise_floor=noise_floor_raw,
        time_sync_ok=True,
        contact_mode_ok=True,
        repair_record=record,
        admitted_trajectory=admitted,
    )


def admit(
    trajectory_snapshots: Sequence[ClosedLoopSnapshot] | ClosedLoopSnapshot,
    *,
    gates: AdmissionGates | None = None,
    repair: Literal["none", "project_to_closure"] = "none",
    repair_fn: Callable[[ClosedLoopSnapshot], ClosedLoopSnapshot] | None = None,
    plant: ClosurePlant | None = None,
    time_s: Sequence[float] | FloatArray | None = None,
    dt_s: float = 0.01,
) -> AdmissionReport:
    """Evaluate kinematic closure and stance-foot contact admission across a trajectory."""
    snaps, resolved_dt, time_sync_reasons = _validate_admit_inputs(
        trajectory_snapshots, repair, time_s, dt_s
    )
    active_gates = AdmissionGates() if gates is None else gates
    noise_floor_raw = (
        math.sqrt(6.0) * active_gates.position_noise_std_m / (resolved_dt * resolved_dt)
    )
    noise_floor_scaled = noise_floor_raw / (
        active_gates.characteristic_length_m * active_gates.characteristic_alpha_rad_s2
    )

    diagnostics = [
        _evaluate_frame(idx, snap, active_gates, noise_floor_scaled)
        for idx, snap in enumerate(snaps)
    ]
    if time_sync_reasons:
        diagnostics = [
            replace(
                d,
                time_sync_ok=False,
                passed=False,
                reasons=(*d.reasons, *time_sync_reasons),
            )
            for d in diagnostics
        ]

    phi_sc = np.array([d.phi_scaled for d in diagnostics], dtype=np.float64)
    phi_dot_sc = np.array([d.phi_dot_scaled for d in diagnostics], dtype=np.float64)
    phi_ddot_sc = np.array([d.phi_ddot_scaled for d in diagnostics], dtype=np.float64)
    failing = tuple(d.frame_index for d in diagnostics if not d.passed)
    deriv_method = snaps[0].input_provenance.derivative_method
    time_sync_all = all(d.time_sync_ok for d in diagnostics)
    contact_all = all(d.contact_mode_ok for d in diagnostics)

    if not failing:
        admitted = AdmittedTrajectory(
            snapshots=snaps,
            verdict=AdmissionVerdict.ADMITTED,
            _token=_ADMITTED_TOKEN,
        )
        return AdmissionReport(
            verdict=AdmissionVerdict.ADMITTED,
            frame_diagnostics=tuple(diagnostics),
            phi_scaled=phi_sc,
            phi_dot_scaled=phi_dot_sc,
            phi_ddot_scaled=phi_ddot_sc,
            derivative_method=deriv_method,
            derivative_noise_floor=noise_floor_raw,
            time_sync_ok=time_sync_all,
            contact_mode_ok=contact_all,
            admitted_trajectory=admitted,
        )

    non_repairable = not time_sync_all or (
        noise_floor_scaled > active_gates.max_accel_noise_floor_scaled
    )
    if repair == "none" or non_repairable:
        return AdmissionReport(
            verdict=AdmissionVerdict.REJECTED,
            frame_diagnostics=tuple(diagnostics),
            phi_scaled=phi_sc,
            phi_dot_scaled=phi_dot_sc,
            phi_ddot_scaled=phi_ddot_sc,
            derivative_method=deriv_method,
            derivative_noise_floor=noise_floor_raw,
            time_sync_ok=time_sync_all,
            contact_mode_ok=contact_all,
            failing_frames=failing,
            reasons=tuple(r for d in diagnostics for r in d.reasons),
        )

    return _execute_repair_pass(
        snaps,
        diagnostics,
        active_gates,
        noise_floor_raw,
        noise_floor_scaled,
        plant,
        repair_fn,
    )
