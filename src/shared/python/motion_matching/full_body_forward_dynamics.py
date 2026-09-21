"""Full-body forward dynamics simulation, polynomial control, and contact audit (#10069).

Provides continuous zero-feedback forward simulation across the 41-coordinate
full-body skeletal models with compliant Hunt-Crossley ground contact and dual-grip
weld loop-closure constraints.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import logging
import math
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.contact_law import (
    GroundPlane,
    calibrate_ground_height_at_address,
)
from src.shared.python.motion_matching.ground_support import (
    SupportReport,
    convex_hull_contains,
    support_report,
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
Controller: TypeAlias = Callable[[float, Array, Array], Array]
InverseDynamicsFn: TypeAlias = Callable[
    ["FullBodySimulator", Array, Array, Array], Array
]
DEFAULT_UNACTUATED: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5})
MIN_SINGULAR_VALUE: float = 1e-2
ROOT_COORDINATES: tuple[str, ...] = (
    "TranslationInputX",
    "TranslationInputY",
    "TranslationInputZ",
    "HipInputX",
    "HipInputY",
    "HipInputZ",
)


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
    shared_metrics: SharedMetrics | None
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
            or (pred_arr.size > 0 and not np.all(np.isfinite(pred_arr)))
        ):
            return False
        # Reject zero-filled / unexecuted output
        if (
            not np.any(q_arr)
            and not np.any(qd_arr)
            and (pred_arr.size == 0 or not np.any(pred_arr))
        ):
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
            "shared_metrics": (
                self.shared_metrics.as_dict()
                if self.shared_metrics is not None
                else None
            ),
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
    auto_calibrate_ground: bool = True
    preserve_ground_calibration: bool = True


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
    capture: TourCapture | None,
    marker_offsets: Mapping[str, Any] | None,
    *,
    status: str = "failed",
    max_closure_translation_m: float = float("nan"),
    max_closure_rotation_rad: float | None = None,
    max_closure_residual_m: float = float("nan"),
) -> ForwardRolloutResult:
    """Construct a fallback ForwardRolloutResult when numerical integration fails or is invalid."""
    n_frames = len(times)
    if capture is not None and marker_offsets is not None:
        eval_cap = _slice_capture(capture, n_frames)
        zero_markers = np.zeros((n_frames, len(eval_cap.labels), 3), dtype=np.float64)
        shared_metrics = compute_shared_metrics(
            capture=eval_cap,
            predicted_points_m=zero_markers,
            tracked_labels=list(marker_offsets.keys()),
        )
    else:
        zero_markers = np.empty((n_frames, 0, 3), dtype=np.float64)
        shared_metrics = None
    return ForwardRolloutResult(
        time_s=times,
        q=np.zeros((n_frames, n_coords)),
        qd=np.zeros((n_frames, n_coords)),
        predicted_markers_m=zero_markers,
        shared_metrics=shared_metrics,
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
    capture: TourCapture | None,
    marker_offsets: Mapping[str, Any] | None,
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
    if capture is not None and marker_offsets is not None:
        eval_cap = _slice_capture(capture, len(times))
        shared_metrics = compute_shared_metrics(
            capture=eval_cap,
            predicted_points_m=pred_m,
            tracked_labels=list(marker_offsets.keys()),
        )
    else:
        shared_metrics = None
    return ForwardRolloutResult(
        time_s=times,
        q=q_traj,
        qd=qd_traj,
        predicted_markers_m=pred_m,
        shared_metrics=shared_metrics,
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
    ik_adapter: Any | None,
    times: Array,
    q_traj: Array,
    qd_traj: Array,
    marker_offsets: Mapping[str, Any] | None,
    labels: Sequence[str],
) -> tuple[Array, list[dict[str, Any]], float, float | None, float, str]:
    """Audit kinematics, contact samples, and loop closure along a simulated trajectory."""
    n_frames = len(times)
    has_markers = (
        ik_adapter is not None and marker_offsets is not None and len(labels) > 0
    )
    mo = marker_offsets if marker_offsets is not None else {}
    if has_markers:
        pred_markers = np.full((n_frames, len(labels), 3), np.nan, dtype=np.float64)
    else:
        pred_markers = np.empty((n_frames, 0, 3), dtype=np.float64)
    all_contact_samples: list[dict[str, Any]] = []
    max_trans: float | None = None
    max_rot: float | None = None
    max_mixed: float | None = None

    for k in range(n_frames):
        if has_markers and ik_adapter is not None:
            pred_markers[k] = _compute_frame_markers(ik_adapter, q_traj[k], mo, labels)
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
        labels if has_markers else (),
        mo if has_markers else {},
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
    ik_adapter: Any | None,
    theta: Array,
    times: Array,
    initial_state: tuple[Array, Array],
    marker_offsets: Mapping[str, Any] | None,
    capture: TourCapture | None,
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
    labels = capture.labels if capture is not None else ()

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
        zero_markers = np.zeros((n_frames, len(labels), 3), dtype=np.float64)
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
        ctx, ik_adapter, times, q_traj, qd_traj, marker_offsets, labels
    )
    return q_traj, qd_traj, pred_m, samples, max_trans, max_rot, max_mixed, status


def _audit_terminal_euler(
    ctx: _EulerStepContext,
    t_end: float,
    curr_q: Array,
    curr_qd: Array,
    closure_errs: list[float | None],
    all_contact_samples: list[dict[str, Any]],
) -> None:
    """Audit terminal Euler state and accumulate closure errors."""
    c_samples, _, trans, rot, mixed = _evaluate_frame_dynamics(
        ctx.model,
        ctx.theta,
        t_end,
        ctx.duration_s,
        ctx.coord_names,
        curr_q,
        curr_qd,
        ctx.options,
    )
    all_contact_samples.append(c_samples)
    closure_errs[0] = _accumulate_max(closure_errs[0], trans)
    closure_errs[1] = _accumulate_max(closure_errs[1], rot)
    closure_errs[2] = _accumulate_max(closure_errs[2], mixed)


def _simulate_euler(
    model: Any,
    ik_adapter: Any | None,
    theta: Array,
    times: Array,
    initial_state: tuple[Array, Array],
    marker_offsets: Mapping[str, Any] | None,
    capture: TourCapture | None,
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
    labels = capture.labels if capture is not None else ()
    has_markers = (
        ik_adapter is not None and marker_offsets is not None and len(labels) > 0
    )
    mo = marker_offsets if marker_offsets is not None else {}

    curr_q = np.array(initial_state[0], dtype=np.float64, copy=True)
    curr_qd = np.array(initial_state[1], dtype=np.float64, copy=True)
    q_traj = np.zeros((n_frames, n_coords), dtype=np.float64)
    qd_traj = np.zeros((n_frames, n_coords), dtype=np.float64)
    if has_markers and ik_adapter is not None:
        pred_markers = np.full((n_frames, len(labels), 3), np.nan, dtype=np.float64)
        pred_markers[0] = _compute_frame_markers(ik_adapter, curr_q, mo, labels)
    else:
        pred_markers = np.empty((n_frames, 0, 3), dtype=np.float64)

    q_traj[0] = curr_q.copy()
    qd_traj[0] = curr_qd.copy()

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
        if has_markers and ik_adapter is not None:
            pred_markers[step + 1] = _compute_frame_markers(
                ik_adapter, curr_q, mo, labels
            )

    _audit_terminal_euler(
        ctx,
        float(times[-1]),
        curr_q,
        curr_qd,
        closure_errs,
        all_contact_samples,
    )

    status = _check_trajectory_finite(
        q_traj,
        qd_traj,
        pred_markers,
        labels if has_markers else (),
        mo if has_markers else {},
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
    ik_adapter: Any | None = None,
    theta: Array | None = None,
    time_grid: Array | Sequence[float] | None = None,
    initial_state: tuple[Array, Array] | None = None,
    marker_offsets: Mapping[str, Any] | None = None,
    capture: TourCapture | None = None,
    options: RolloutOptions | None = None,
) -> ForwardRolloutResult:
    """Simulate uninterrupted forward dynamics from initial state over time_grid."""
    opts = RolloutOptions() if options is None else options
    if time_grid is None:
        raise ValueError("time_grid must be provided")
    times = np.asarray(time_grid, dtype=np.float64)
    if initial_state is None:
        raise ValueError("initial_state must be provided")
    q0 = np.asarray(initial_state[0], dtype=np.float64)
    coord_count = len(getattr(model, "coordinate_order", [])) or len(q0)
    th = (
        np.zeros((coord_count, 7), dtype=np.float64)
        if theta is None
        else np.asarray(theta, dtype=np.float64)
    )

    if opts.auto_calibrate_ground:
        for attr in ("ground_plane", "ground", "_ground_plane"):
            if hasattr(model, attr):
                gp = getattr(model, attr)
                if abs(gp.height_m) < 1e-6:
                    calib_z = calibrate_ground_height_at_address(model, q0)
                    setattr(
                        model, attr, GroundPlane(normal=gp.normal, height_m=calib_z)
                    )
                break

    if opts.integrator == "rk45" and len(times) > 1:
        data = _simulate_rk45(
            model,
            ik_adapter,
            th,
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
            th,
            times,
            initial_state,
            marker_offsets,
            capture,
            opts,
        )
    return _assemble_rollout_result(times, data, capture, marker_offsets)


@dataclass(frozen=True)
class SimulationRecord:
    """Sampled state, torque and support history of one run."""

    time_s: Array
    q: Array
    v: Array
    tau: Array
    normal_force_n: Array
    weight_fraction: Array
    centre_of_pressure_m: Array
    inside_support_polygon: Array
    lowest_sphere_height_m: Array


@dataclass(frozen=True)
class ComputedTorqueGains:
    """Gains and references for computed-torque control."""

    omega_rad_s: float | Array
    zeta: float = 1.0
    balance: tuple[float, float] | None = None
    root_regulation: tuple[float, float] | None = None
    inverse_fn: InverseDynamicsFn | None = None


class FullBodySimulator:
    """RK4 forward dynamics with unactuated root and shared ground contact."""

    def __init__(self, adapter: Any) -> None:
        names = tuple(adapter.coordinate_order)
        if names[:6] != ROOT_COORDINATES:
            raise ValueError("The first six coordinates must be the pelvis root")
        self.adapter = adapter
        self.names = names
        self.nv = len(names)
        self.root = np.arange(6)
        self.actuated = np.arange(6, self.nv)
        self.lower_limb = np.arange(adapter.upper_body_coordinates, self.nv)
        if hasattr(adapter, "mass_kg"):
            self.mass_kg = float(adapter.mass_kg)
            self.gravity = np.asarray(
                getattr(adapter, "gravity", [0.0, 0.0, -9.81]), dtype=float
            )
            self._dof = np.arange(self.nv)
        elif hasattr(adapter, "model") and hasattr(adapter.model, "opt"):
            model = adapter.model
            opt = model.opt
            self.mass_kg = float(np.sum(model.body_mass))
            self.gravity = np.array(opt.gravity, dtype=float)
            # Adapter force vectors follow MuJoCo DOF order; states follow spec order.
            self._dof = np.array([model.joint(name).dofadr[0] for name in names])
        else:
            spec = getattr(adapter, "specification", {})
            bodies = spec.get("bodies", {})
            body_list = bodies.values() if isinstance(bodies, dict) else bodies
            self.mass_kg = float(
                sum(b.get("mass_kg", 0.0) for b in body_list)
                or spec.get("subject", {}).get("mass_kg", 75.0)
            )
            self.gravity = np.asarray(
                spec.get("gravity_m_s2", [0.0, 0.0, -9.81]), dtype=float
            )
            self._dof = np.arange(self.nv)

    def _map(self, values: Array) -> dict[str, float]:
        return dict(zip(self.names, values.tolist(), strict=True))

    def root_translation_axes(self, q: Array) -> Array:
        """World directions (columns) of the three root slide coordinates at ``q``."""
        adapter = self.adapter
        if not hasattr(adapter, "model") or not hasattr(adapter.model, "joint"):
            return np.eye(3)
        adapter.frame_poses(self._map(q))
        model = adapter.model
        return np.column_stack(
            [adapter.data.xaxis[model.joint(name).id] for name in self.names[:3]]
        )

    def acceleration(self, q: Array, v: Array, tau: Array) -> Array:
        """Constrained acceleration with contact; root torques are forced to zero."""
        effort = np.asarray(tau, dtype=float).copy()
        if effort.shape != (self.nv,):
            raise ValueError("Torque vector must match the coordinate count")
        effort[self.root] = 0.0
        result = self.adapter.accelerations(
            self._map(q), self._map(v), self._map(effort)
        )
        return np.array([result[name] for name in self.names])

    def feedforward(
        self, q: Array, v: Array, *, compensate_contact: bool = True
    ) -> Array:
        """Torque cancelling the bias (and, optionally, contact) generalized forces."""
        bias, contact, _ = self.adapter.generalized_forces(self._map(q), self._map(v))
        tau = np.zeros(self.nv)
        force = bias - contact if compensate_contact else bias
        tau[self.actuated] = force[self._dof][self.actuated]
        return tau

    def static_penetration_m(self) -> float:
        """Penetration at which equally loaded spheres carry the weight at rest."""
        contact_params = self.adapter.contact_parameters
        stiffness = float(contact_params.stiffness_n_m)
        spheres = self.adapter._spheres
        return (
            self.mass_kg
            * float(np.linalg.norm(self.gravity))
            / (stiffness * len(spheres))
        )

    def affine_dynamics(self, q: Array, v: Array) -> tuple[Array, Array]:
        """Return ``(A, b)`` with ``a = A @ tau_actuated + b`` at the state."""
        adapter = self.adapter
        if hasattr(adapter, "affine_dynamics"):
            return adapter.affine_dynamics(q, v)
        bias, contact, _ = adapter.generalized_forces(self._map(q), self._map(v))
        mj, model, data = adapter._mj, adapter.model, adapter.data
        mass = np.zeros((model.nv, model.nv))
        mj.mj_fullM(model, mass, data.qM)
        if hasattr(adapter, "evaluate_weld_closure"):
            jac, drift = adapter.evaluate_weld_closure()
        elif hasattr(adapter, "_evaluate_weld_closure"):
            jac, drift = adapter._evaluate_weld_closure(
                mj, model, data, adapter._closure
            )
        else:
            from src.engines.physics_engines.mujoco.python.native_model import (
                _evaluate_weld_closure,
            )

            jac, drift = _evaluate_weld_closure(mj, model, data, adapter._closure)
        m = jac.shape[0]
        kkt = np.block([[mass, -jac.T], [jac, np.zeros((m, m))]])
        rhs = np.zeros((model.nv + m, 1 + self.actuated.size))
        rhs[: model.nv, 0] = contact - bias
        rhs[model.nv :, 0] = -drift
        rhs[self._dof[self.actuated], 1:] = np.eye(self.actuated.size)
        try:
            solution = np.linalg.solve(kkt, rhs)[: model.nv]
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(kkt, rhs, rcond=1e-7)[0][: model.nv]
        ordered = solution[self._dof]
        return ordered[:, 1:], ordered[:, 0]

    def inverse_dynamics(self, q: Array, v: Array, joint_acceleration: Array) -> Array:
        """Torques giving the actuated joints exactly ``joint_acceleration``."""
        target = np.asarray(joint_acceleration, dtype=float)
        if target.shape != (self.actuated.size,) or not np.isfinite(target).all():
            raise ValueError("Joint acceleration must be finite, one per joint")
        affine, offset = self.affine_dynamics(q, v)
        rows = self.actuated
        u, sv, vt = np.linalg.svd(affine[rows], full_matrices=False)
        keep = sv > MIN_SINGULAR_VALUE
        inverse = (vt[keep].T / sv[keep]) @ u[:, keep].T
        tau = np.zeros(self.nv)
        tau[rows] = inverse @ (target - offset[rows])
        return tau

    def centre_of_mass(self, q: Array) -> tuple[Array, Array]:
        """Whole-body centre of mass and its Jacobian (spec coordinate order)."""
        adapter = self.adapter
        if hasattr(adapter, "centre_of_mass"):
            return adapter.centre_of_mass(q)
        adapter.frame_poses(self._map(q))
        mj, model, data = adapter._mj, adapter.model, adapter.data
        mj.mj_comPos(model, data)
        jac = np.zeros((3, model.nv))
        mj.mj_jacSubtreeCom(model, data, jac, 1)
        return data.subtree_com[1].copy(), jac[:, self._dof]

    def support(self, q: Array, v: Array) -> tuple[SupportReport, float]:
        """Support report and the lowest sphere height at a state."""
        samples = self.adapter.evaluate_contact_samples(self._map(q), self._map(v))
        plane = self.adapter.ground_plane
        n = np.asarray(plane.normal, dtype=float)
        n = n / np.linalg.norm(n)
        points: dict[str, Array] = {}
        lowest = np.inf
        if hasattr(self.adapter, "get_sphere_kinematics"):
            zero_rates = dict.fromkeys(self.names, 0.0)
            coords = self._map(q)
            for name in self.adapter._spheres:
                pos, _, radius = self.adapter.get_sphere_kinematics(
                    name, coords, zero_rates
                )
                height = float(pos @ n - plane.height_m)
                lowest = min(lowest, height - radius)
                points[name] = pos - height * n
        else:
            adapter_spheres = self.adapter._spheres
            adapter_data = self.adapter.data
            for name, info in adapter_spheres.items():
                site_id = info["site_id"]
                centre = adapter_data.site_xpos[site_id].copy()
                height = float(centre @ n - plane.height_m)
                lowest = min(lowest, height - info["radius"])
                points[name] = centre - (height) * n
        report = support_report(
            samples, points, plane, self.mass_kg, self.gravity.tolist()
        )
        return report, float(lowest)

    def step(
        self, t: float, q: Array, v: Array, controller: Controller, dt: float
    ) -> tuple[Array, Array, Array]:
        """One RK4 step; returns the new state and the torque used at the start."""

        def rate(t_k: float, q_k: Array, v_k: Array) -> tuple[Array, Array, Array]:
            tau_k = np.asarray(controller(t_k, q_k, v_k), dtype=float)
            return v_k, self.acceleration(q_k, v_k, tau_k), tau_k

        k1q, k1v, tau = rate(t, q, v)
        k2q, k2v, _ = rate(t + dt / 2, q + dt / 2 * k1q, v + dt / 2 * k1v)
        k3q, k3v, _ = rate(t + dt / 2, q + dt / 2 * k2q, v + dt / 2 * k2v)
        k4q, k4v, _ = rate(t + dt, q + dt * k3q, v + dt * k3v)
        q_next = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        v_next = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        if not (np.isfinite(q_next).all() and np.isfinite(v_next).all()):
            raise FloatingPointError(f"Nonfinite state at t={t:.4f}s")
        return q_next, v_next, tau

    def run(
        self,
        q0: Array,
        v0: Array,
        controller: Controller,
        *,
        duration_s: float,
        dt_s: float,
        record_every: int = 1,
    ) -> SimulationRecord:
        """Integrate from ``(q0, v0)`` and sample every ``record_every`` steps."""
        q_curr: Array = np.asarray(q0, dtype=float).copy()
        v_curr: Array = np.asarray(v0, dtype=float).copy()
        if tuple(q_curr.shape) != (self.nv,) or tuple(v_curr.shape) != (self.nv,):
            raise ValueError("Initial state must match the coordinate count")
        if not (np.isfinite(q_curr).all() and np.isfinite(v_curr).all()):
            raise ValueError("Initial state must be finite")
        if duration_s <= 0 or dt_s <= 0 or record_every < 1:
            raise ValueError("Duration, step and record interval must be positive")
        steps = int(round(duration_s / dt_s))
        times, qs, vs, taus = [0.0], [q_curr.copy()], [v_curr.copy()], []
        supports: list[tuple[SupportReport, float]] = [self.support(q_curr, v_curr)]
        tau_prev: Array = np.zeros(self.nv)
        for k in range(1, steps + 1):
            q_curr, v_curr, tau_prev = self.step(
                (k - 1) * dt_s, q_curr, v_curr, controller, dt_s
            )
            if k % record_every == 0 or k == steps:
                times.append(k * dt_s)
                qs.append(q_curr.copy())
                vs.append(v_curr.copy())
                taus.append(tau_prev.copy())
                supports.append(self.support(q_curr, v_curr))
        taus.insert(0, taus[0] if taus else tau_prev)
        cops = np.array(
            [
                (
                    r.centre_of_pressure_m
                    if r.centre_of_pressure_m is not None
                    else (np.nan,) * 3
                )
                for r, _ in supports
            ]
        )
        return SimulationRecord(
            time_s=np.array(times),
            q=np.array(qs),
            v=np.array(vs),
            tau=np.array(taus),
            normal_force_n=np.array([r.total_normal_force_n for r, _ in supports]),
            weight_fraction=np.array([r.weight_fraction for r, _ in supports]),
            centre_of_pressure_m=cops,
            inside_support_polygon=np.array(
                [r.inside_support_polygon for r, _ in supports]
            ),
            lowest_sphere_height_m=np.array([h for _, h in supports]),
        )


def preload_feet(
    simulator: FullBodySimulator, q: Array, *, preload: bool = True
) -> Array:
    """Translate the root along the ground normal so the feet rest on the plane."""
    q_out = np.asarray(q, dtype=float).copy()
    depth = simulator.static_penetration_m() if preload else 0.0
    adapter = simulator.adapter
    plane = adapter.ground_plane
    n = np.asarray(plane.normal, dtype=float)
    n = n / np.linalg.norm(n)
    if hasattr(adapter, "get_sphere_kinematics"):
        zero_rates = dict.fromkeys(simulator.names, 0.0)
        coords = simulator._map(q_out)
        centres = np.array(
            [
                adapter.get_sphere_kinematics(s, coords, zero_rates)[0]
                for s in adapter._spheres
            ]
        )
        radii = np.array(
            [
                adapter.get_sphere_kinematics(s, coords, zero_rates)[2]
                for s in adapter._spheres
            ]
        )
    else:
        adapter.frame_poses(simulator._map(q_out))
        centres = np.array(
            [adapter.data.site_xpos[i["site_id"]] for i in adapter._spheres.values()]
        )
        radii = np.array([i["radius"] for i in adapter._spheres.values()])
    lowest = float(np.min(centres @ n - radii)) - plane.height_m + depth
    q_out[:3] += np.linalg.solve(simulator.root_translation_axes(q_out), -lowest * n)
    return q_out


def _check_gains(
    omega_rad_s: float | Array, zeta: float, balance: tuple[float, float] | None
) -> None:
    omega = np.asarray(omega_rad_s, dtype=float)
    if not np.isfinite(omega).all() or np.any(omega <= 0) or zeta <= 0:
        raise ValueError("Natural frequency and damping ratio must be positive")
    if balance is not None and (len(balance) != 2 or min(balance) < 0):
        raise ValueError("Balance gains must be two nonnegative numbers")


def joint_natural_frequencies(
    simulator: FullBodySimulator, *, upper_body: float, lower_limb: float
) -> Array:
    """Per-coordinate natural frequency vector: stiff upper body, compliant legs."""
    if upper_body <= 0 or lower_limb <= 0:
        raise ValueError("Natural frequencies must be positive")
    omega = np.full(simulator.nv, float(upper_body))
    omega[simulator.lower_limb] = float(lower_limb)
    return omega


def _planted_coupling(simulator: FullBodySimulator) -> Array:
    """``S`` with ``v_root = S v_legs`` when every contact sphere is held still."""
    adapter = simulator.adapter
    if hasattr(adapter, "_planted_coupling"):
        return adapter._planted_coupling(simulator)
    if hasattr(adapter, "sphere_jacobians"):
        jac_feet = adapter.sphere_jacobians()
        legs = simulator.lower_limb
        return -np.linalg.pinv(jac_feet[:, simulator.root]) @ jac_feet[:, legs]
    mj, model, data = adapter._mj, adapter.model, adapter.data
    rows = []
    for info in adapter._spheres.values():
        buffer = np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, buffer, None, info["site_id"])
        rows.append(buffer[:, simulator._dof])
    jac_feet = np.concatenate(rows)
    legs = simulator.lower_limb
    return -np.linalg.pinv(jac_feet[:, simulator.root]) @ jac_feet[:, legs]


def _root_regulation_acceleration(
    simulator: FullBodySimulator,
    q: Array,
    v: Array,
    q_ref: Array,
    v_ref: Array,
    gains: tuple[float, float],
) -> Array:
    """Lower-limb acceleration steering the root (pelvis) pose to its reference."""
    adapter = simulator.adapter
    if hasattr(adapter, "_mj"):
        adapter.frame_poses(simulator._map(q))
        mj = adapter._mj
        mj.mj_comPos(adapter.model, adapter.data)
    coupling = _planted_coupling(simulator)
    root = simulator.root
    wanted = gains[0] * (q_ref[root] - q[root]) + gains[1] * (v_ref[root] - v[root])
    out = np.zeros(simulator.nv)
    out[simulator.lower_limb] = np.linalg.pinv(coupling) @ wanted
    return out


def _planted_com_jacobian(
    simulator: FullBodySimulator, q: Array
) -> tuple[Array, Array]:
    """CoM position and its Jacobian over the lower-limb joints with feet planted."""
    com, jac_com = simulator.centre_of_mass(q)
    legs = simulator.lower_limb
    coupling = _planted_coupling(simulator)
    return com, jac_com[:, legs] + jac_com[:, simulator.root] @ coupling


def _balance_acceleration(
    simulator: FullBodySimulator,
    q: Array,
    v: Array,
    com_ref: Array,
    gains: tuple[float, float],
) -> Array:
    """Actuated-joint acceleration steering the CoM over ``com_ref`` (legs only)."""
    com, jac = _planted_com_jacobian(simulator, q)
    ground_plane = simulator.adapter.ground_plane
    n = np.asarray(ground_plane.normal, dtype=float)
    n = n / np.linalg.norm(n)
    error = com_ref - com
    error -= (error @ n) * n
    velocity = jac @ v[simulator.lower_limb]
    velocity -= (velocity @ n) * n
    wanted = gains[0] * error - gains[1] * velocity
    out = np.zeros(simulator.nv)
    out[simulator.lower_limb] = np.linalg.pinv(jac) @ wanted
    return out[simulator.actuated]


def _computed_torque(
    simulator: FullBodySimulator,
    q: Array,
    v: Array,
    q_ref: Array,
    v_ref: Array,
    a_ref: Array,
    gains: ComputedTorqueGains,
    com_ref: Array | None = None,
) -> Array:
    act = simulator.actuated
    omega = np.broadcast_to(
        np.asarray(gains.omega_rad_s, dtype=float), (simulator.nv,)
    )[act]
    wanted = (
        a_ref[act]
        + 2.0 * gains.zeta * omega * (v_ref[act] - v[act])
        + omega**2 * (q_ref[act] - q[act])
    )
    if gains.balance is not None and com_ref is not None:
        wanted = wanted + _balance_acceleration(simulator, q, v, com_ref, gains.balance)
    if gains.root_regulation is not None:
        wanted = (
            wanted
            + _root_regulation_acceleration(
                simulator, q, v, q_ref, v_ref, gains.root_regulation
            )[act]
        )
    inv = gains.inverse_fn or (
        lambda sim, q_i, v_i, w: sim.inverse_dynamics(q_i, v_i, w)
    )
    return inv(simulator, q, v, wanted)


def _tracking_gains(
    omega_rad_s: float | Array,
    zeta: float,
    balance: tuple[float, float] | None,
    root_regulation: tuple[float, float] | None,
    *,
    inverse_fn: InverseDynamicsFn | None = None,
) -> ComputedTorqueGains:
    _check_gains(omega_rad_s, zeta, balance)
    _check_gains(1.0, 1.0, root_regulation)
    return ComputedTorqueGains(
        omega_rad_s=omega_rad_s,
        zeta=zeta,
        balance=balance,
        root_regulation=root_regulation,
        inverse_fn=inverse_fn,
    )


def _tracking_controller_from_gains(
    simulator: FullBodySimulator,
    time_ref: Sequence[float] | Array,
    q_ref: Array,
    gains: ComputedTorqueGains,
    *,
    acceleration_feedforward: float = 1.0,
) -> Controller:
    """Computed-torque tracking with pre-built gains."""
    if not 0.0 <= acceleration_feedforward <= 1.0:
        raise ValueError("acceleration_feedforward must lie in [0, 1]")
    times = np.asarray(time_ref, dtype=float)
    reference = np.asarray(q_ref, dtype=float)
    if (
        times.ndim != 1
        or np.any(np.diff(times) <= 0)
        or reference.shape != (times.size, simulator.nv)
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Reference times must increase with one finite q row each")
    if times.size > 1:
        velocity = np.gradient(reference, times, axis=0)
        acceleration = acceleration_feedforward * np.gradient(velocity, times, axis=0)
    else:
        velocity = np.zeros_like(reference)
        acceleration = np.zeros_like(reference)

    def sample(table: Array, t: float) -> Array:
        return np.array([np.interp(t, times, table[:, k]) for k in range(simulator.nv)])

    def controller(t: float, q: Array, v: Array) -> Array:
        q_t, v_t, a_t = (
            sample(reference, t),
            sample(velocity, t),
            sample(acceleration, t),
        )
        com_ref = (
            simulator.centre_of_mass(q_t)[0] if gains.balance is not None else None
        )
        return _computed_torque(simulator, q, v, q_t, v_t, a_t, gains, com_ref)

    return controller


def hold_pose_controller(
    simulator: FullBodySimulator,
    q_ref: Array,
    *,
    omega_rad_s: float | Array,
    zeta: float = 1.0,
    balance: tuple[float, float] | None = None,
    root_regulation: tuple[float, float] | None = None,
) -> Controller:
    """Computed-torque hold of a posture with optional centre-of-mass balance."""
    reference = np.asarray(q_ref, dtype=float).copy()
    if reference.shape != (simulator.nv,) or not np.isfinite(reference).all():
        raise ValueError("Reference posture must be finite with model size")
    _check_gains(omega_rad_s, zeta, balance)
    _check_gains(1.0, 1.0, root_regulation)
    com_ref = simulator.centre_of_mass(reference)[0] if balance is not None else None
    zero = np.zeros(simulator.nv)
    gains = ComputedTorqueGains(
        omega_rad_s=omega_rad_s,
        zeta=zeta,
        balance=balance,
        root_regulation=root_regulation,
    )

    def controller(t: float, q: Array, v: Array) -> Array:
        return _computed_torque(simulator, q, v, reference, zero, zero, gains, com_ref)

    return controller


def tracking_controller(
    simulator: FullBodySimulator,
    time_ref: Sequence[float] | Array,
    q_ref: Array,
    *,
    omega_rad_s: float | Array,
    zeta: float = 1.0,
    balance: tuple[float, float] | None = None,
    root_regulation: tuple[float, float] | None = None,
    acceleration_feedforward: float = 1.0,
) -> Controller:
    """Computed-torque tracking of a reference trajectory (linear interpolation)."""
    return _tracking_controller_from_gains(
        simulator,
        time_ref,
        q_ref,
        _tracking_gains(omega_rad_s, zeta, balance, root_regulation),
        acceleration_feedforward=acceleration_feedforward,
    )


def _distance_outside(point_xy: Array, hull_xy: Array) -> float:
    """Distance from a point to a convex polygon, zero inside."""
    if convex_hull_contains(point_xy, hull_xy):
        return 0.0
    best = np.inf
    n = len(hull_xy)
    for i in range(n):
        a, b = hull_xy[i], hull_xy[(i + 1) % n]
        ab = b - a
        s = float(np.clip((point_xy - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(point_xy - (a + s * ab))))
    return best


def _compute_frame_momentum(
    adapter: Any, simulator: FullBodySimulator, q: Array, v: Array
) -> tuple[Array, Array, Array]:
    """Whole-body CoM position, linear momentum, and angular momentum."""
    if hasattr(adapter, "_compute_frame_momentum"):
        return adapter._compute_frame_momentum(simulator, q, v)
    adapter.frame_poses(simulator._map(q))
    adapter.data.qvel[simulator._dof] = v
    adapter._mj.mj_forward(adapter.model, adapter.data)
    adapter._mj.mj_subtreeVel(adapter.model, adapter.data)
    linear = adapter.data.subtree_linvel[1] * adapter.model.body_subtreemass[1]
    return (
        adapter.data.subtree_com[1].copy(),
        linear.copy(),
        adapter.data.subtree_angmom[1].copy(),
    )


def _compute_zmp_point(
    c0: Array, reaction: Array, moment: Array, ground: GroundPlane
) -> Array:
    """Point on ground plane where reaction gives zero moment about CoM."""
    n = np.asarray(ground.normal, dtype=float)
    n = n / np.linalg.norm(n)
    normal_load = max(float(reaction @ n), 1e-6)
    height = float(c0 @ n) - ground.height_m
    r_t = reaction - n * float(reaction @ n)
    return c0 - n * height + (np.cross(n, moment) - height * r_t) / normal_load


def reference_zmp(
    simulator: FullBodySimulator,
    time_ref: Sequence[float] | Array,
    q_ref: Array,
    ground: GroundPlane,
    *,
    contact_tolerance_m: float = 0.005,
    min_load_fraction: float = 0.1,
) -> dict[str, Array]:
    """Zero-moment point a reference trajectory demands of this model."""
    times = np.asarray(time_ref, dtype=float)
    ref = np.asarray(q_ref, dtype=float)
    if (
        times.ndim != 1
        or times.size < 3
        or np.any(np.diff(times) <= 0)
        or ref.shape != (times.size, simulator.nv)
        or not np.isfinite(ref).all()
    ):
        raise ValueError(
            "Reference needs at least three increasing times and finite rows"
        )
    if contact_tolerance_m < 0 or not 0.0 < min_load_fraction < 1.0:
        raise ValueError(
            "contact tolerance must be nonnegative, load fraction in (0, 1)"
        )
    adapter = simulator.adapter
    n = np.asarray(ground.normal, dtype=float)
    n = n / np.linalg.norm(n)
    g = float(np.linalg.norm(simulator.gravity))
    mass = simulator.mass_kg
    velocity = np.gradient(ref, times, axis=0)
    acceleration = np.gradient(velocity, times, axis=0)
    dt = 1e-4
    frames = times.size
    zmp, com, grf = np.empty((frames, 2)), np.empty((frames, 3)), np.empty((frames, 3))
    outside, unloaded = np.empty(frames), np.zeros(frames, dtype=bool)
    hulls = np.empty(frames, dtype=object)
    sphere_names = list(adapter._spheres)

    for k in range(frames):
        c0, p0, l0 = _compute_frame_momentum(adapter, simulator, ref[k], velocity[k])
        c1, p1, l1 = _compute_frame_momentum(
            adapter,
            simulator,
            ref[k] + velocity[k] * dt,
            velocity[k] + acceleration[k] * dt,
        )
        reaction = (p1 - p0) / dt - simulator.gravity * mass
        moment = (l1 - l0) / dt
        point = _compute_zmp_point(c0, reaction, moment, ground)
        if hasattr(adapter, "get_sphere_kinematics"):
            coords = simulator._map(ref[k])
            zero_rates = dict.fromkeys(simulator.names, 0.0)
            centres = np.array(
                [
                    adapter.get_sphere_kinematics(s, coords, zero_rates)[0]
                    for s in sphere_names
                ]
            )
            radii = np.array(
                [
                    adapter.get_sphere_kinematics(s, coords, zero_rates)[2]
                    for s in sphere_names
                ]
            )
        else:
            adapter.frame_poses(simulator._map(ref[k]))
            centres = np.array(
                [
                    adapter.data.site_xpos[adapter._spheres[s]["site_id"]]
                    for s in sphere_names
                ]
            )
            radii = np.array([adapter._spheres[s]["radius"] for s in sphere_names])
        touching = (centres @ n - radii - ground.height_m) <= contact_tolerance_m
        feet = centres[touching] if touching.sum() >= 3 else centres
        hull = feet[:, :2]
        zmp[k], com[k], grf[k] = point[:2], c0, reaction / (mass * g)
        unloaded[k] = float(reaction @ n) < min_load_fraction * mass * g
        outside[k] = 0.0 if unloaded[k] else _distance_outside(point[:2], hull)
        hulls[k] = hull.copy()
    return {
        "zmp_xy": zmp,
        "com": com,
        "grf_over_weight": grf,
        "outside_m": outside,
        "unloaded": unloaded,
        "hull_xy": hulls,
    }
