"""Full-body forward dynamics simulation, polynomial control, and contact audit (#10069).

Provides continuous zero-feedback forward simulation across the 41-coordinate
full-body skeletal models with compliant Hunt-Crossley ground contact and dual-grip
weld loop-closure constraints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import math
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.contact_law import (
    GroundPlane,
    calibrate_ground_height_at_address,
)
from src.shared.python.motion_matching.polynomial_torque import (
    evaluate_polynomial_torque,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture
from src.shared.python.motion_matching.tour_metrics import (
    SharedMetrics,
    compute_shared_metrics,
)

logger = logging.getLogger(__name__)

Array: TypeAlias = NDArray[np.float64]
DEFAULT_UNACTUATED: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5})


@dataclass(frozen=True)
class ContactAuditResult:
    """Comprehensive ground contact force and penetration summary."""

    max_normal_force_n: float
    max_friction_force_n: float
    max_penetration_m: float
    per_sphere_max_force_n: dict[str, float]
    per_sphere_contact_ratio: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_normal_force_n": self.max_normal_force_n,
            "max_friction_force_n": self.max_friction_force_n,
            "max_penetration_m": self.max_penetration_m,
            "per_sphere_max_force_n": self.per_sphere_max_force_n,
            "per_sphere_contact_ratio": self.per_sphere_contact_ratio,
        }


@dataclass(frozen=True)
class ForwardRolloutResult:
    """Outcome of an uninterrupted forward dynamics simulation from t=0."""

    time_s: Array
    q: Array
    qd: Array
    predicted_markers_m: Array
    shared_metrics: SharedMetrics
    contact_audit: ContactAuditResult
    max_closure_translation_m: float
    max_closure_rotation_rad: float | None
    max_closure_residual_m: float
    status: str

    def is_closure_accepted(
        self,
        *,
        max_translation_tol_m: float = 1e-3,
        max_rotation_tol_rad: float = 0.05,
    ) -> bool:
        """Rollout numerical closure acceptance predicate.

        Gating execution status, finite state validity, strictly increasing time,
        consistent sequence shapes, and separated loop-closure tolerances.
        Fails closed on missing or non-finite measurements, negative metrics,
        infinite or non-positive profile limits, or unexecuted/failed runs.

        Note: this predicate verifies numerical rollout closure and validity only;
        it does NOT imply or grant full physical/scientific qualification, contact
        audit compliance, cross-engine replay parity, or downstream model acceptance.
        """
        if self.status != "success":
            return False
        if len(self.time_s) == 0:
            return False
        # Profile limits (tolerances) must be finite and strictly positive
        if not (
            math.isfinite(max_translation_tol_m)
            and max_translation_tol_m > 0.0
            and math.isfinite(max_rotation_tol_rad)
            and max_rotation_tol_rad > 0.0
        ):
            return False
        # Time array must be finite and strictly increasing
        time_arr = np.asarray(self.time_s)
        if not np.all(np.isfinite(time_arr)):
            return False
        if len(time_arr) > 1 and not np.all(np.diff(time_arr) > 0.0):
            return False
        # Shapes and time alignment
        q_arr = np.asarray(self.q)
        qd_arr = np.asarray(self.qd)
        pred_arr = np.asarray(self.predicted_markers_m)
        n_frames = len(time_arr)
        if (
            q_arr.shape[0] != n_frames
            or qd_arr.shape[0] != n_frames
            or pred_arr.shape[0] != n_frames
        ):
            return False
        # Finite state check
        if (
            not np.all(np.isfinite(q_arr))
            or not np.all(np.isfinite(qd_arr))
            or not np.all(np.isfinite(pred_arr))
        ):
            return False
        # Reject zero-filled / unexecuted output
        if not np.any(q_arr) and not np.any(qd_arr) and not np.any(pred_arr):
            return False
        # Measured closure metrics must be finite and non-negative
        if (
            self.max_closure_translation_m is None
            or not math.isfinite(self.max_closure_translation_m)
            or self.max_closure_translation_m < 0.0
        ):
            return False
        # Rotation closure evidence must be present, finite, and non-negative
        if (
            self.max_closure_rotation_rad is None
            or not math.isfinite(self.max_closure_rotation_rad)
            or self.max_closure_rotation_rad < 0.0
        ):
            return False
        if (
            self.max_closure_residual_m is None
            or not math.isfinite(self.max_closure_residual_m)
            or self.max_closure_residual_m < 0.0
        ):
            return False
        return bool(
            self.max_closure_translation_m <= max_translation_tol_m
            and self.max_closure_rotation_rad <= max_rotation_tol_rad
        )

    def is_accepted(
        self,
        *,
        max_translation_tol_m: float = 1e-3,
        max_rotation_tol_rad: float = 0.05,
    ) -> bool:
        """Rollout closure acceptance predicate (delegates to is_closure_accepted).

        Checks numerical integration and separated loop-closure tolerances. Does
        not imply full physical/scientific qualification or multi-engine replay.
        """
        return self.is_closure_accepted(
            max_translation_tol_m=max_translation_tol_m,
            max_rotation_tol_rad=max_rotation_tol_rad,
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize rollout result summary with separated units and legacy labels."""
        return {
            "status": self.status,
            "max_closure_translation_m": self.max_closure_translation_m,
            "max_closure_rotation_rad": self.max_closure_rotation_rad,
            "legacy_mixed_unit_closure_value": self.max_closure_residual_m,
            "max_closure_residual_m": self.max_closure_residual_m,
            "contact_audit": self.contact_audit.as_dict(),
            "shared_metrics": self.shared_metrics.as_dict(),
        }


def evaluate_polynomial_torques(
    theta: Array,
    t: float,
    duration_s: float,
    coordinate_names: Sequence[str],
    unactuated_indices: set[int] | frozenset[int] | None = None,
    *,
    normalize_time: bool = False,
) -> dict[str, float]:
    """Evaluate degree-6 polynomial torques across coordinates.

    tau_i(t) = sum_{k=0}^6 c_{i, k} * t^k (or normalized time if normalize_time=True).
    """
    th = np.asarray(theta, dtype=np.float64)
    n_coords = len(coordinate_names)
    if th.shape[0] != n_coords:
        raise ValueError(f"Expected {n_coords} rows in theta, got {th.shape[0]}")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be strictly positive")

    t_eval = float(np.clip(t / duration_s, 0.0, 1.0)) if normalize_time else float(t)
    raw_torques = evaluate_polynomial_torque(th, t_eval)

    unact = DEFAULT_UNACTUATED if unactuated_indices is None else unactuated_indices
    torques: dict[str, float] = {}
    for i, name in enumerate(coordinate_names):
        if i in unact:
            torques[name] = 0.0
        else:
            torques[name] = float(raw_torques[i])
    return torques


def _compute_frame_markers(
    ik_adapter: Any,
    q: Array,
    marker_offsets: Mapping[str, Any],
    labels: Sequence[str],
) -> Array:
    """Compute (markers, 3) 3D world positions for a single configuration."""
    poses = ik_adapter.pose_fn(q)
    frame_points = np.full((len(labels), 3), np.nan, dtype=np.float64)
    for i, label in enumerate(labels):
        if label in marker_offsets:
            info = marker_offsets[label]
            body_name = info["body"]
            if body_name in poses:
                r_mat, t_vec = poses[body_name]
                frame_points[i] = r_mat @ info["offset_m"] + t_vec
    return frame_points


def _audit_contact_samples(
    all_samples: Sequence[dict[str, Any]],
    n_frames: int,
) -> ContactAuditResult:
    """Aggregate ground contact audit statistics across the simulation timeline."""
    max_fn = 0.0
    max_ft = 0.0
    max_d = 0.0
    sphere_max_f: dict[str, float] = {}
    sphere_contacts: dict[str, int] = {}

    for samples in all_samples:
        for s_name, sample in samples.items():
            fn = float(np.linalg.norm(sample.normal_force_n))
            ft = float(np.linalg.norm(sample.friction_force_n))
            d = float(sample.penetration_m)
            max_fn = max(max_fn, fn)
            max_ft = max(max_ft, ft)
            max_d = max(max_d, d)

            prev_max = sphere_max_f.get(s_name, 0.0)
            sphere_max_f[s_name] = max(prev_max, fn)
            if d > 0.0 or fn > 1e-3:
                sphere_contacts[s_name] = sphere_contacts.get(s_name, 0) + 1

    ratios = {
        s_name: (sphere_contacts.get(s_name, 0) / max(1, n_frames))
        for s_name in sphere_max_f
    }

    return ContactAuditResult(
        max_normal_force_n=max_fn,
        max_friction_force_n=max_ft,
        max_penetration_m=max_d,
        per_sphere_max_force_n=sphere_max_f,
        per_sphere_contact_ratio=ratios,
    )


@dataclass(frozen=True)
class RolloutOptions:
    """Configuration options for full-body forward dynamics rollout."""

    substeps: int = 2
    unactuated_indices: frozenset[int] = DEFAULT_UNACTUATED
    integrator: str = "rk45"
    normalize_time: bool = False
    rtol: float = 1e-5
    atol: float = 1e-7


def _slice_capture(capture: TourCapture, n_frames: int) -> TourCapture:
    """Return capture truncated to n_frames if sub-horizon rollout."""
    if capture.frames == n_frames:
        return capture
    return TourCapture(
        time_s=capture.time_s[:n_frames] - capture.time_s[0],
        labels=capture.labels,
        points_m=capture.points_m[:n_frames],
        valid=capture.valid[:n_frames],
        source_sha256=capture.source_sha256,
    )


def _accumulate_max(current: float | None, new_val: float | None) -> float | None:
    """Accumulate maximum without hiding NaNs behind 0.0 or overwriting None."""
    if new_val is None:
        return current
    if current is None:
        return new_val
    if np.isnan(current) or np.isnan(new_val):
        return float("nan")
    return max(current, new_val)


def _build_failed_rollout(
    times: Array,
    n_coords: int,
    capture: TourCapture,
    marker_offsets: Mapping[str, Any],
    *,
    status: str = "failed",
    max_closure_translation_m: float = float("nan"),
    max_closure_rotation_rad: float | None = None,
    max_closure_residual_m: float = float("nan"),
) -> ForwardRolloutResult:
    """Construct a fallback ForwardRolloutResult when numerical integration fails or is invalid."""
    n_frames = len(times)
    eval_cap = _slice_capture(capture, n_frames)
    zero_markers = np.zeros((n_frames, len(eval_cap.labels), 3), dtype=np.float64)
    return ForwardRolloutResult(
        time_s=times,
        q=np.zeros((n_frames, n_coords)),
        qd=np.zeros((n_frames, n_coords)),
        predicted_markers_m=zero_markers,
        shared_metrics=compute_shared_metrics(
            capture=eval_cap,
            predicted_points_m=zero_markers,
            tracked_labels=list(marker_offsets.keys()),
        ),
        contact_audit=_audit_contact_samples([], n_frames),
        max_closure_translation_m=max_closure_translation_m,
        max_closure_rotation_rad=max_closure_rotation_rad,
        max_closure_residual_m=max_closure_residual_m,
        status=status,
    )


def _assemble_rollout_result(
    times: Array,
    data: tuple[
        Array,
        Array,
        Array,
        list[dict[str, Any]],
        float,
        float | None,
        float,
        str,
    ],
    capture: TourCapture,
    marker_offsets: Mapping[str, Any],
) -> ForwardRolloutResult:
    """Construct a ForwardRolloutResult from trajectory data, contact audit, and metrics."""
    q_traj, qd_traj, pred_m, samples, trans_err, rot_err, mixed_err, status = data
    if status != "success":
        return _build_failed_rollout(
            times,
            q_traj.shape[1],
            capture,
            marker_offsets,
            status=status,
            max_closure_translation_m=trans_err,
            max_closure_rotation_rad=rot_err,
            max_closure_residual_m=mixed_err,
        )
    eval_cap = _slice_capture(capture, len(times))
    return ForwardRolloutResult(
        time_s=times,
        q=q_traj,
        qd=qd_traj,
        predicted_markers_m=pred_m,
        shared_metrics=compute_shared_metrics(
            capture=eval_cap,
            predicted_points_m=pred_m,
            tracked_labels=list(marker_offsets.keys()),
        ),
        contact_audit=_audit_contact_samples(samples, len(times)),
        max_closure_translation_m=trans_err,
        max_closure_rotation_rad=rot_err,
        max_closure_residual_m=mixed_err,
        status="success",
    )


def _extract_closure_errors(model: Any) -> tuple[float, float | None, float]:
    """Extract translation, rotation, and mixed closure error norms from model.

    Validates declared 1D closure residual shape (size 3 for translation-only or
    size 6 for translation + rotation). Returns rot_err as None when rotation is
    not measured, preserving missing evidence distinct from measured 0.0.
    """
    err_p, _ = model.closure_errors()
    err_p_arr = np.asarray(err_p, dtype=np.float64)
    if err_p_arr.ndim != 1 or err_p_arr.size not in (3, 6):
        raise ValueError(
            f"Expected 1D closure residual of size 3 or 6, got shape {err_p_arr.shape}"
        )
    trans_err = float(np.linalg.norm(err_p_arr[:3]))
    rot_err = float(np.linalg.norm(err_p_arr[3:6])) if err_p_arr.size == 6 else None
    mixed_err = float(np.linalg.norm(err_p_arr))
    return trans_err, rot_err, mixed_err


def _evaluate_frame_dynamics(
    model: Any,
    theta: Array,
    t: float,
    duration_s: float,
    coord_names: Sequence[str],
    q_vec: Array,
    qd_vec: Array,
    options: RolloutOptions,
) -> tuple[dict[str, Any], dict[str, float], float, float | None, float]:
    """Evaluate contact samples, torques, accelerations, and closure errors."""
    q_dict = {name: float(q_vec[i]) for i, name in enumerate(coord_names)}
    qd_dict = {name: float(qd_vec[i]) for i, name in enumerate(coord_names)}
    c_samples = model.evaluate_contact_samples(q_dict, qd_dict)
    tau_dict = evaluate_polynomial_torques(
        theta,
        t,
        duration_s,
        coord_names,
        options.unactuated_indices,
        normalize_time=options.normalize_time,
    )
    acc_dict = model.accelerations(q_dict, qd_dict, tau_dict)
    trans_err, rot_err, mixed_err = _extract_closure_errors(model)
    return c_samples, acc_dict, trans_err, rot_err, mixed_err


def _check_trajectory_finite(
    q_traj: Array,
    qd_traj: Array,
    pred_markers: Array,
    tracked_labels: Sequence[str],
    marker_offsets: Mapping[str, Any],
    max_trans_err: float | None,
    max_rot_err: float | None,
    max_mixed_err: float | None,
) -> str:
    """Validate that state trajectories, tracked markers, and closure errors are finite."""
    tracked_indices = [
        i for i, label in enumerate(tracked_labels) if label in marker_offsets
    ]
    has_nan = (
        not np.all(np.isfinite(q_traj))
        or not np.all(np.isfinite(qd_traj))
        or (
            len(tracked_indices) > 0
            and not np.all(np.isfinite(pred_markers[:, tracked_indices, :]))
        )
        or (max_trans_err is not None and not np.isfinite(max_trans_err))
        or (max_rot_err is not None and not np.isfinite(max_rot_err))
        or (max_mixed_err is not None and not np.isfinite(max_mixed_err))
    )
    return "invalid" if has_nan else "success"


@dataclass(frozen=True)
class _EulerStepContext:
    """Context for Euler substep integration."""

    model: Any
    theta: Array
    duration_s: float
    coord_names: Sequence[str]
    options: RolloutOptions


def _step_euler_substeps(
    ctx: _EulerStepContext,
    t_curr: float,
    curr_q: Array,
    curr_qd: Array,
    dt_frame: float,
    all_contact_samples: list[dict[str, Any]],
    closure_errs: list[float | None],
) -> None:
    """Execute substeps of semi-implicit Euler integration for one frame step."""
    dt_sub = dt_frame / max(1, ctx.options.substeps)
    for sub in range(ctx.options.substeps):
        t_sub = t_curr + sub * dt_sub
        c_samples, acc_dict, trans_err, rot_err, mixed_err = _evaluate_frame_dynamics(
            ctx.model,
            ctx.theta,
            t_sub,
            ctx.duration_s,
            ctx.coord_names,
            curr_q,
            curr_qd,
            ctx.options,
        )
        all_contact_samples.append(c_samples)
        closure_errs[0] = _accumulate_max(closure_errs[0], trans_err)
        closure_errs[1] = _accumulate_max(closure_errs[1], rot_err)
        closure_errs[2] = _accumulate_max(closure_errs[2], mixed_err)

        acc = np.array([acc_dict[name] for name in ctx.coord_names], dtype=np.float64)
        curr_qd += acc * dt_sub
        curr_q += curr_qd * dt_sub


def _audit_rollout_trajectory(
    ctx: _EulerStepContext,
    ik_adapter: Any,
    times: Array,
    q_traj: Array,
    qd_traj: Array,
    marker_offsets: Mapping[str, Any],
    labels: Sequence[str],
) -> tuple[Array, list[dict[str, Any]], float, float | None, float, str]:
    """Audit kinematics, contact samples, and loop closure along a simulated trajectory."""
    n_frames = len(times)
    pred_markers = np.full((n_frames, len(labels), 3), np.nan, dtype=np.float64)
    all_contact_samples: list[dict[str, Any]] = []
    max_trans: float | None = None
    max_rot: float | None = None
    max_mixed: float | None = None

    for k in range(n_frames):
        pred_markers[k] = _compute_frame_markers(
            ik_adapter, q_traj[k], marker_offsets, labels
        )
        c_samples, _, trans, rot, mixed = _evaluate_frame_dynamics(
            ctx.model,
            ctx.theta,
            float(times[k]),
            ctx.duration_s,
            ctx.coord_names,
            q_traj[k],
            qd_traj[k],
            ctx.options,
        )
        all_contact_samples.append(c_samples)
        max_trans = _accumulate_max(max_trans, trans)
        max_rot = _accumulate_max(max_rot, rot)
        max_mixed = _accumulate_max(max_mixed, mixed)

    status = _check_trajectory_finite(
        q_traj,
        qd_traj,
        pred_markers,
        labels,
        marker_offsets,
        max_trans,
        max_rot,
        max_mixed,
    )
    return (
        pred_markers,
        all_contact_samples,
        max_trans if max_trans is not None else 0.0,
        max_rot,
        max_mixed if max_mixed is not None else 0.0,
        status,
    )


def _simulate_rk45(
    model: Any,
    ik_adapter: Any,
    theta: Array,
    times: Array,
    initial_state: tuple[Array, Array],
    marker_offsets: Mapping[str, Any],
    capture: TourCapture,
    options: RolloutOptions,
) -> tuple[
    Array,
    Array,
    Array,
    list[dict[str, Any]],
    float,
    float | None,
    float,
    str,
]:
    """Execute forward simulation via adaptive RK45 integration."""
    from scipy.integrate import solve_ivp

    n_frames = len(times)
    coord_names = list(model.coordinate_order)
    n_coords = len(coord_names)
    duration_s = float(times[-1]) if times[-1] > 0.0 else 1.0
    init_state = np.concatenate([initial_state[0], initial_state[1]])

    def deriv(t: float, state: Array) -> Array:
        q_d = {name: float(state[i]) for i, name in enumerate(coord_names)}
        qd_d = {name: float(state[n_coords + i]) for i, name in enumerate(coord_names)}
        tau = evaluate_polynomial_torques(
            theta,
            t,
            duration_s,
            coord_names,
            options.unactuated_indices,
            normalize_time=options.normalize_time,
        )
        acc_dict = model.accelerations(q_d, qd_d, tau)
        acc = np.array([acc_dict[name] for name in coord_names], dtype=np.float64)
        return np.concatenate([state[n_coords:], acc])

    sol = solve_ivp(
        deriv,
        (float(times[0]), float(times[-1])),
        init_state,
        t_eval=times,
        method="RK45",
        rtol=options.rtol,
        atol=options.atol,
    )
    if not sol.success:
        zero_markers = np.zeros((n_frames, len(capture.labels), 3), dtype=np.float64)
        return (
            np.zeros((n_frames, n_coords)),
            np.zeros((n_frames, n_coords)),
            zero_markers,
            [],
            float("nan"),
            None,
            float("nan"),
            "failed",
        )

    q_traj = sol.y[:n_coords, :].T
    qd_traj = sol.y[n_coords:, :].T
    ctx = _EulerStepContext(model, theta, duration_s, coord_names, options)
    pred_m, samples, max_trans, max_rot, max_mixed, status = _audit_rollout_trajectory(
        ctx, ik_adapter, times, q_traj, qd_traj, marker_offsets, capture.labels
    )
    return q_traj, qd_traj, pred_m, samples, max_trans, max_rot, max_mixed, status


def _simulate_euler(
    model: Any,
    ik_adapter: Any,
    theta: Array,
    times: Array,
    initial_state: tuple[Array, Array],
    marker_offsets: Mapping[str, Any],
    capture: TourCapture,
    options: RolloutOptions,
) -> tuple[
    Array,
    Array,
    Array,
    list[dict[str, Any]],
    float,
    float | None,
    float,
    str,
]:
    """Execute forward simulation via semi-implicit Euler integration."""
    n_frames = len(times)
    coord_names = list(model.coordinate_order)
    n_coords = len(coord_names)
    duration_s = float(times[-1]) if times[-1] > 0.0 else 1.0

    curr_q = np.array(initial_state[0], dtype=np.float64, copy=True)
    curr_qd = np.array(initial_state[1], dtype=np.float64, copy=True)
    q_traj = np.zeros((n_frames, n_coords), dtype=np.float64)
    qd_traj = np.zeros((n_frames, n_coords), dtype=np.float64)
    pred_markers = np.full((n_frames, len(capture.labels), 3), np.nan, dtype=np.float64)

    q_traj[0] = curr_q.copy()
    qd_traj[0] = curr_qd.copy()
    pred_markers[0] = _compute_frame_markers(
        ik_adapter, curr_q, marker_offsets, capture.labels
    )

    all_contact_samples: list[dict[str, Any]] = []
    closure_errs: list[float | None] = [None, None, None]
    ctx = _EulerStepContext(model, theta, duration_s, coord_names, options)

    for step in range(n_frames - 1):
        dt_frame = float(times[step + 1]) - float(times[step])
        _step_euler_substeps(
            ctx,
            float(times[step]),
            curr_q,
            curr_qd,
            dt_frame,
            all_contact_samples,
            closure_errs,
        )
        q_traj[step + 1] = curr_q.copy()
        qd_traj[step + 1] = curr_qd.copy()
        pred_markers[step + 1] = _compute_frame_markers(
            ik_adapter, curr_q, marker_offsets, capture.labels
        )

    # Audit terminal Euler state (frame n_frames - 1)
    c_samples, _, trans, rot, mixed = _evaluate_frame_dynamics(
        model,
        theta,
        float(times[-1]),
        duration_s,
        coord_names,
        curr_q,
        curr_qd,
        options,
    )
    all_contact_samples.append(c_samples)
    closure_errs[0] = _accumulate_max(closure_errs[0], trans)
    closure_errs[1] = _accumulate_max(closure_errs[1], rot)
    closure_errs[2] = _accumulate_max(closure_errs[2], mixed)

    status = _check_trajectory_finite(
        q_traj,
        qd_traj,
        pred_markers,
        capture.labels,
        marker_offsets,
        closure_errs[0],
        closure_errs[1],
        closure_errs[2],
    )
    return (
        q_traj,
        qd_traj,
        pred_markers,
        all_contact_samples,
        closure_errs[0] if closure_errs[0] is not None else 0.0,
        closure_errs[1],
        closure_errs[2] if closure_errs[2] is not None else 0.0,
        status,
    )


def simulate_full_body_forward(
    model: Any,
    ik_adapter: Any,
    theta: Array,
    time_grid: Array,
    initial_state: tuple[Array, Array],
    marker_offsets: Mapping[str, Any],
    capture: TourCapture,
    options: RolloutOptions | None = None,
) -> ForwardRolloutResult:
    """Simulate uninterrupted forward dynamics from initial state over time_grid."""
    opts = RolloutOptions() if options is None else options
    times = np.asarray(time_grid, dtype=np.float64)
    q0 = np.asarray(initial_state[0], dtype=np.float64)

    if hasattr(model, "ground_plane") and abs(model.ground_plane.height_m) < 1e-6:
        calib_z = calibrate_ground_height_at_address(model, q0)
        model.ground_plane = GroundPlane(
            normal=model.ground_plane.normal, height_m=calib_z
        )
    elif hasattr(model, "ground") and abs(model.ground.height_m) < 1e-6:
        calib_z = calibrate_ground_height_at_address(model, q0)
        model.ground = GroundPlane(normal=model.ground.normal, height_m=calib_z)
    elif hasattr(model, "_ground_plane") and abs(model._ground_plane.height_m) < 1e-6:
        calib_z = calibrate_ground_height_at_address(model, q0)
        model._ground_plane = GroundPlane(
            normal=model._ground_plane.normal, height_m=calib_z
        )

    if opts.integrator == "rk45" and len(times) > 1:
        data = _simulate_rk45(
            model,
            ik_adapter,
            theta,
            times,
            initial_state,
            marker_offsets,
            capture,
            opts,
        )
    else:
        data = _simulate_euler(
            model,
            ik_adapter,
            theta,
            times,
            initial_state,
            marker_offsets,
            capture,
            opts,
        )
    return _assemble_rollout_result(times, data, capture, marker_offsets)
