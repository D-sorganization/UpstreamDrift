"""Full-body forward dynamics simulation, polynomial control, and contact audit (#10069).

Provides continuous zero-feedback forward simulation across the 41-coordinate
full-body skeletal models with compliant Hunt-Crossley ground contact and dual-grip
weld loop-closure constraints.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.contact_law import GroundPlane
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
    max_closure_rotation_rad: float
    max_closure_residual_m: float
    status: str

    def is_accepted(
        self,
        *,
        max_translation_tol_m: float = 1e-3,
        max_rotation_tol_rad: float = 0.05,
    ) -> bool:
        """Physical rollout acceptance gating execution status and separated closure tolerances.

        Rejects failed or zero-filled output. Both translation and rotation errors must be
        within explicit physical limits.
        """
        if self.status != "success" or len(self.time_s) == 0:
            return False
        if (
            not np.any(self.q)
            and not np.any(self.qd)
            and not np.any(self.predicted_markers_m)
        ):
            return False
        return bool(
            self.max_closure_translation_m <= max_translation_tol_m
            and self.max_closure_rotation_rad <= max_rotation_tol_rad
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


def calibrate_ground_height_at_address(model: Any, address_q: Array) -> float:
    """Calibrate ground plane height to the lowest contact sphere surface at address."""
    q_dict = {
        name: float(address_q[i]) for i, name in enumerate(model.coordinate_order)
    }
    if hasattr(model, "_mj"):
        model.data.qpos[:] = model._vector(q_dict)
        model._mj.mj_fwdPosition(model.model, model.data)
        min_z = min(
            model.data.site_xpos[s_info["site_id"]][2] - s_info["radius"]
            for s_info in model._spheres.values()
        )
        return float(min_z)
    if hasattr(model, "_pin"):
        q = model.configuration(q_dict)
        model._pin.forwardKinematics(model.model, model.data, q)
        model._pin.updateFramePlacements(model.model, model.data)
        min_z = min(
            float(
                model.data.oMf[model._contact_frames[s.name]].translation[2]
                - s.radius_m
            )
            for s in model.contact_spheres
        )
        return float(min_z)
    if hasattr(model, "plant"):
        model.plant.SetPositions(model.context, model._vector(q_dict, model._q_indices))
        min_z = min(
            float(
                model.plant.CalcPointsPositions(
                    model.context,
                    s_info["frame"],
                    np.zeros(3),
                    model.plant.world_frame(),
                )[2, 0]
                - s_info["radius_m"]
            )
            for s_info in model._spheres.values()
        )
        return float(min_z)
    return 0.0


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


def _build_failed_rollout(
    times: Array,
    n_coords: int,
    capture: TourCapture,
    marker_offsets: Mapping[str, Any],
) -> ForwardRolloutResult:
    """Construct a fallback ForwardRolloutResult when numerical integration fails."""
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
        max_closure_translation_m=0.0,
        max_closure_rotation_rad=0.0,
        max_closure_residual_m=0.0,
        status="failed",
    )


def _assemble_rollout_result(
    times: Array,
    data: tuple[Array, Array, Array, list[dict[str, Any]], float, float, float, str],
    capture: TourCapture,
    marker_offsets: Mapping[str, Any],
) -> ForwardRolloutResult:
    """Construct a ForwardRolloutResult from trajectory data, contact audit, and metrics."""
    q_traj, qd_traj, pred_m, samples, trans_err, rot_err, mixed_err, status = data
    if status != "success":
        return _build_failed_rollout(times, q_traj.shape[1], capture, marker_offsets)
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


def _extract_closure_errors(model: Any) -> tuple[float, float, float]:
    """Extract translation, rotation, and mixed closure error norms from model."""
    err_p, _ = model.closure_errors()
    err_p_arr = np.asarray(err_p, dtype=np.float64)
    trans_err = float(np.linalg.norm(err_p_arr[:3]))
    rot_err = float(np.linalg.norm(err_p_arr[3:])) if err_p_arr.size > 3 else 0.0
    mixed_err = float(np.linalg.norm(err_p_arr))
    return trans_err, rot_err, mixed_err


def _simulate_rk45(
    model: Any,
    ik_adapter: Any,
    theta: Array,
    times: Array,
    initial_state: tuple[Array, Array],
    marker_offsets: Mapping[str, Any],
    capture: TourCapture,
    options: RolloutOptions,
) -> tuple[Array, Array, Array, list[dict[str, Any]], float, float, float, str]:
    """Execute forward simulation via adaptive RK45 integration."""
    from scipy.integrate import solve_ivp

    n_frames = len(times)
    coord_names = list(model.coordinate_order)
    n_coords = len(coord_names)
    duration_s = float(times[-1]) if times[-1] > 0.0 else 1.0
    init_state = np.concatenate([initial_state[0], initial_state[1]])

    def deriv(t: float, state: Array) -> Array:
        q_dict = {name: float(state[i]) for i, name in enumerate(coord_names)}
        qd_dict = {
            name: float(state[n_coords + i]) for i, name in enumerate(coord_names)
        }
        tau_dict = evaluate_polynomial_torques(
            theta,
            t,
            duration_s,
            coord_names,
            options.unactuated_indices,
            normalize_time=options.normalize_time,
        )
        acc_dict = model.accelerations(q_dict, qd_dict, tau_dict)
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
            0.0,
            0.0,
            0.0,
            "failed",
        )

    q_traj = sol.y[:n_coords, :].T
    qd_traj = sol.y[n_coords:, :].T
    pred_markers = np.full((n_frames, len(capture.labels), 3), np.nan, dtype=np.float64)
    all_contact_samples = []
    max_trans_err = 0.0
    max_rot_err = 0.0
    max_mixed_err = 0.0

    for k in range(n_frames):
        q_k = q_traj[k]
        qd_k = qd_traj[k]
        pred_markers[k] = _compute_frame_markers(
            ik_adapter, q_k, marker_offsets, capture.labels
        )
        q_dict = {name: float(q_k[i]) for i, name in enumerate(coord_names)}
        qd_dict = {name: float(qd_k[i]) for i, name in enumerate(coord_names)}
        all_contact_samples.append(model.evaluate_contact_samples(q_dict, qd_dict))
        tau_dict = evaluate_polynomial_torques(
            theta,
            float(times[k]),
            duration_s,
            coord_names,
            options.unactuated_indices,
            normalize_time=options.normalize_time,
        )
        model.accelerations(q_dict, qd_dict, tau_dict)
        trans_err, rot_err, mixed_err = _extract_closure_errors(model)
        max_trans_err = max(max_trans_err, trans_err)
        max_rot_err = max(max_rot_err, rot_err)
        max_mixed_err = max(max_mixed_err, mixed_err)

    return (
        q_traj,
        qd_traj,
        pred_markers,
        all_contact_samples,
        max_trans_err,
        max_rot_err,
        max_mixed_err,
        "success",
    )


def _simulate_euler(
    model: Any,
    ik_adapter: Any,
    theta: Array,
    times: Array,
    initial_state: tuple[Array, Array],
    marker_offsets: Mapping[str, Any],
    capture: TourCapture,
    options: RolloutOptions,
) -> tuple[Array, Array, Array, list[dict[str, Any]], float, float, float, str]:
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

    all_contact_samples = []
    max_trans_err = 0.0
    max_rot_err = 0.0
    max_mixed_err = 0.0

    for step in range(n_frames - 1):
        t_curr = float(times[step])
        t_next = float(times[step + 1])
        dt_frame = t_next - t_curr
        dt_sub = dt_frame / max(1, options.substeps)

        for sub in range(options.substeps):
            t_sub = t_curr + sub * dt_sub
            tau_dict = evaluate_polynomial_torques(
                theta,
                t_sub,
                duration_s,
                coord_names,
                options.unactuated_indices,
                normalize_time=options.normalize_time,
            )
            q_dict = {name: float(curr_q[i]) for i, name in enumerate(coord_names)}
            qd_dict = {name: float(curr_qd[i]) for i, name in enumerate(coord_names)}

            c_samples = model.evaluate_contact_samples(q_dict, qd_dict)
            all_contact_samples.append(c_samples)

            acc_dict = model.accelerations(q_dict, qd_dict, tau_dict)
            trans_err, rot_err, mixed_err = _extract_closure_errors(model)
            max_trans_err = max(max_trans_err, trans_err)
            max_rot_err = max(max_rot_err, rot_err)
            max_mixed_err = max(max_mixed_err, mixed_err)

            acc = np.array([acc_dict[name] for name in coord_names], dtype=np.float64)
            curr_qd += acc * dt_sub
            curr_q += curr_qd * dt_sub

        q_traj[step + 1] = curr_q.copy()
        qd_traj[step + 1] = curr_qd.copy()
        pred_markers[step + 1] = _compute_frame_markers(
            ik_adapter, curr_q, marker_offsets, capture.labels
        )

    return (
        q_traj,
        qd_traj,
        pred_markers,
        all_contact_samples,
        max_trans_err,
        max_rot_err,
        max_mixed_err,
        "success",
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
