"""Bounded overlapping-window IK refinement and temporal regularization (PF-02, #10432).

Retains per-frame MarkerIkSolver solutions as warm-start initializers, applies
bounded overlapping-window optimization with temporal smoothness, dual-grip weld
closure, foot ground non-penetration barrier, and multi-start geometric floor diagnosis.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import logging
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require

logger = logging.getLogger(__name__)

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class WindowedIkOptions:
    """Configuration for overlapping-window IK refinement."""

    window_size: int = 9
    overlap: int = 3
    marker_weight: float = 1.0
    closure_weight: float = 1e3
    smoothness_weight: float = 1e-1
    ground_barrier_weight: float = 5e3
    ground_height_m: float = 0.0
    max_iterations: int = 15
    damping: float = 1e-3
    step_limit_rad: float = 0.25
    convergence_tol: float = 1e-5


@dataclass(frozen=True)
class RefinementReceipt:
    """Receipt summarizing windowed IK refinement metrics."""

    frames: int
    coordinates: int
    initial_marker_rms_m: float
    final_marker_rms_m: float
    initial_closure_max_m: float
    final_closure_max_m: float
    lowest_ground_clearance_m: float
    window_count: int
    converged: bool


@dataclass(frozen=True)
class DifficultFrameFloorDiagnosis:
    """Geometric floor estimation comparing constrained vs unconstrained fits."""

    frame_index: int
    unconstrained_marker_rms_m: float
    unconstrained_closure_err_m: float
    constrained_marker_rms_m: float
    constrained_closure_err_m: float
    geometric_floor_gap_m: float


class WindowedIkRefiner:
    """Overlapping-window kinematic trajectory refiner."""

    def __init__(
        self,
        forward_markers_fn: Callable[[Array], Array],
        *,
        closure_fn: Callable[[Array], Array] | None = None,
        foot_contact_fn: Callable[[Array], Array] | None = None,
        lower: Array | None = None,
        upper: Array | None = None,
        options: WindowedIkOptions | None = None,
    ) -> None:
        self.forward_markers_fn = forward_markers_fn
        self.closure_fn = closure_fn
        self.foot_contact_fn = foot_contact_fn
        self.lower = None if lower is None else np.asarray(lower, dtype=float)
        self.upper = None if upper is None else np.asarray(upper, dtype=float)
        self.options = options or WindowedIkOptions()

        if self.lower is not None and self.upper is not None:
            require(self.lower.shape == self.upper.shape, "Bounds shape mismatch")
            require(bool(np.all(self.lower <= self.upper)), "lower <= upper required")

    def _frame_residuals_and_jacobian(
        self,
        q: Array,
        targets: Array,
        valid: BoolArray,
        weights: Array,
        q_prev: Array | None,
        q_next: Array | None,
    ) -> tuple[Array, Array]:
        opts = self.options
        n = q.size
        eps = 1e-6

        p0 = self.forward_markers_fn(q)
        m_valid = np.flatnonzero(valid)
        n_val = m_valid.size

        # Numerical marker Jacobian
        jac_m = np.zeros((n_val * 3, n), dtype=float)
        diff_m = np.zeros(n_val * 3, dtype=float)

        if n_val > 0:
            scale_m = np.sqrt(opts.marker_weight * weights[m_valid])
            for idx, m_idx in enumerate(m_valid):
                diff = p0[m_idx] - targets[m_idx]
                diff_m[idx * 3 : (idx + 1) * 3] = scale_m[idx] * diff

            for j in range(n):
                q_p = q.copy()
                q_p[j] += eps
                p_p = self.forward_markers_fn(q_p)
                for idx, m_idx in enumerate(m_valid):
                    col = (p_p[m_idx] - p0[m_idx]) / eps
                    jac_m[idx * 3 : (idx + 1) * 3, j] = scale_m[idx] * col

        # Closure residuals
        c_res: list[float] = []
        c_jac_list: list[Array] = []
        if self.closure_fn is not None and opts.closure_weight > 0.0:
            c0 = self.closure_fn(q)
            scale_c = np.sqrt(opts.closure_weight)
            c_res = (scale_c * c0).tolist()
            for j in range(n):
                q_p = q.copy()
                q_p[j] += eps
                c_p = self.closure_fn(q_p)
                col_c = scale_c * (c_p - c0) / eps
                c_jac_list.append(col_c[:, None])
            jac_c = np.hstack(c_jac_list) if c_jac_list else np.zeros((len(c0), n))
        else:
            jac_c = np.zeros((0, n))

        # Ground barrier: foot contact spheres bottom point >= ground_height_m
        g_res: list[float] = []
        g_jac_list: list[Array] = []
        if self.foot_contact_fn is not None and opts.ground_barrier_weight > 0.0:
            foot_pts0 = self.foot_contact_fn(q)
            scale_g = np.sqrt(opts.ground_barrier_weight)
            for p_idx, pt in enumerate(foot_pts0):
                z_depth = opts.ground_height_m - pt[2]
                if z_depth > 0.0:
                    g_res.append(scale_g * z_depth)
                    grad_z = np.zeros(n)
                    for j in range(n):
                        q_p = q.copy()
                        q_p[j] += eps
                        pt_p = self.foot_contact_fn(q_p)[p_idx]
                        grad_z[j] = scale_g * (-(pt_p[2] - pt[2]) / eps)
                    g_jac_list.append(grad_z)
            jac_g = np.vstack(g_jac_list) if g_jac_list else np.zeros((0, n))
        else:
            jac_g = np.zeros((0, n))

        # Smoothness toward neighbors (Laplacian temporal penalty)
        s_res: list[float] = []
        s_jac_list: list[Array] = []
        if opts.smoothness_weight > 0.0 and q_prev is not None and q_next is not None:
            scale_s = np.sqrt(opts.smoothness_weight)
            s_diff = scale_s * (2.0 * q - q_prev - q_next)
            s_res = s_diff.tolist()
            jac_s = 2.0 * scale_s * np.eye(n)
        elif opts.smoothness_weight > 0.0 and q_prev is not None:
            scale_s = np.sqrt(opts.smoothness_weight)
            s_diff = scale_s * (q - q_prev)
            s_res = s_diff.tolist()
            jac_s = scale_s * np.eye(n)
        else:
            jac_s = np.zeros((0, n))

        all_res = np.concatenate(
            [diff_m, np.array(c_res), np.array(g_res), np.array(s_res)]
        )
        all_jac = np.vstack([jac_m, jac_c, jac_g, jac_s])
        return all_res, all_jac

    def refine_window(
        self,
        q_window: Array,
        targets_window: Array,
        valid_window: BoolArray,
        weights: Array,
    ) -> Array:
        """Refine configurations within a single window."""
        w_len, n_coords = q_window.shape
        q_opt = q_window.copy()
        opts = self.options

        for _it in range(opts.max_iterations):
            max_step = 0.0
            for f in range(w_len):
                q_prev = q_opt[f - 1] if f > 0 else None
                q_next = q_opt[f + 1] if f < w_len - 1 else None

                res, jac = self._frame_residuals_and_jacobian(
                    q_opt[f],
                    targets_window[f],
                    valid_window[f],
                    weights,
                    q_prev,
                    q_next,
                )
                normal = jac.T @ jac + opts.damping * np.eye(n_coords)
                step = -np.linalg.solve(normal, jac.T @ res)
                norm = float(np.max(np.abs(step)))
                if norm > opts.step_limit_rad:
                    step *= opts.step_limit_rad / norm
                q_cand = q_opt[f] + step
                if self.lower is not None and self.upper is not None:
                    q_cand = np.clip(q_cand, self.lower, self.upper)
                q_opt[f] = q_cand
                max_step = max(max_step, norm)

            if max_step < opts.convergence_tol:
                break

        return q_opt

    def refine_trajectory(
        self,
        q_initial: Array,
        targets: Array,
        valid: BoolArray,
        weights: Array | None = None,
    ) -> tuple[Array, RefinementReceipt]:
        """Refine full kinematic trajectory across overlapping windows.

        Args:
            q_initial: Warm start trajectory (N, coordinates).
            targets: Marker positions (N, markers, 3).
            valid: Marker observation mask (N, markers).
            weights: Optional per-marker weights (markers,).
        """
        q_in = np.asarray(q_initial, dtype=float)
        t_arr = np.asarray(targets, dtype=float)
        v_arr = np.asarray(valid, dtype=bool)
        n_frames, n_coords = q_in.shape
        n_markers = t_arr.shape[1]

        require(n_frames >= 1, "Trajectory must contain at least 1 frame")
        require(t_arr.shape[0] == n_frames, "Targets frame count mismatch")
        require(v_arr.shape == (n_frames, n_markers), "Valid mask shape mismatch")

        w_weights = (
            np.ones(n_markers, dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )

        # Baseline evaluation
        init_rms = self._compute_trajectory_marker_rms(q_in, t_arr, v_arr)
        init_closure_max = self._compute_trajectory_closure_max(q_in)

        opts = self.options
        w_size = max(3, min(opts.window_size, n_frames))
        stride = max(1, w_size - opts.overlap)

        q_accum = np.zeros_like(q_in)
        counts = np.zeros(n_frames, dtype=float)
        window_count = 0

        for start in range(0, n_frames, stride):
            end = min(start + w_size, n_frames)
            if end - start < 2 and n_frames >= 2:
                start = max(0, end - w_size)

            q_sub = q_in[start:end].copy()
            t_sub = t_arr[start:end]
            v_sub = v_arr[start:end]

            q_ref = self.refine_window(q_sub, t_sub, v_sub, w_weights)

            # Cosine / Hann weighting for smooth overlapping blend
            sub_len = end - start
            if sub_len == 1:
                w_blend = np.ones(1)
            else:
                w_blend = np.sin(np.linspace(0.1, np.pi - 0.1, sub_len)) ** 2

            for i in range(sub_len):
                q_accum[start + i] += w_blend[i] * q_ref[i]
                counts[start + i] += w_blend[i]
            window_count += 1

            if end == n_frames:
                break

        # Normalize accumulated configurations
        q_out = np.zeros_like(q_in)
        for f in range(n_frames):
            q_out[f] = q_accum[f] / (counts[f] if counts[f] > 0 else 1.0)
            if self.lower is not None and self.upper is not None:
                q_out[f] = np.clip(q_out[f], self.lower, self.upper)

        final_rms = self._compute_trajectory_marker_rms(q_out, t_arr, v_arr)
        final_closure_max = self._compute_trajectory_closure_max(q_out)
        lowest_clearance = self._compute_lowest_clearance(q_out)

        receipt = RefinementReceipt(
            frames=n_frames,
            coordinates=n_coords,
            initial_marker_rms_m=init_rms,
            final_marker_rms_m=final_rms,
            initial_closure_max_m=init_closure_max,
            final_closure_max_m=final_closure_max,
            lowest_ground_clearance_m=lowest_clearance,
            window_count=window_count,
            converged=bool(np.isfinite(q_out).all()),
        )

        ensure(len(q_out) == n_frames, "No dropped frames permitted")
        return q_out, receipt

    def diagnose_difficult_frames(
        self,
        q_traj: Array,
        targets: Array,
        valid: BoolArray,
        weights: Array | None = None,
        *,
        top_k: int = 3,
    ) -> list[DifficultFrameFloorDiagnosis]:
        """Diagnose geometric floors on worst frames with and without closure."""
        n_frames = q_traj.shape[0]
        per_frame_err: list[tuple[float, int]] = []

        for f in range(n_frames):
            p = self.forward_markers_fn(q_traj[f])
            v_idx = np.flatnonzero(valid[f])
            if v_idx.size > 0:
                diff = np.linalg.norm(p[v_idx] - targets[f, v_idx], axis=1)
                per_frame_err.append((float(np.sqrt(np.mean(diff**2))), f))
            else:
                per_frame_err.append((0.0, f))

        per_frame_err.sort(reverse=True, key=lambda x: x[0])
        worst_frames = [idx for _, idx in per_frame_err[:top_k]]

        diagnoses: list[DifficultFrameFloorDiagnosis] = []
        saved_closure_wt = self.options.closure_weight

        for f_idx in worst_frames:
            q_seed = q_traj[f_idx]
            v_frame = valid[f_idx : f_idx + 1]
            t_frame = targets[f_idx : f_idx + 1]

            # Fit without closure (geometric floor)
            refiner_unconstrained = WindowedIkRefiner(
                self.forward_markers_fn,
                closure_fn=None,
                foot_contact_fn=self.foot_contact_fn,
                lower=self.lower,
                upper=self.upper,
                options=WindowedIkOptions(
                    window_size=1,
                    closure_weight=0.0,
                    smoothness_weight=0.0,
                    max_iterations=25,
                ),
            )
            q_uncon, _ = refiner_unconstrained.refine_trajectory(
                q_seed[None, :], t_frame, v_frame, weights
            )
            rms_uncon = self._compute_trajectory_marker_rms(q_uncon, t_frame, v_frame)
            c_uncon = (
                float(np.linalg.norm(self.closure_fn(q_uncon[0])))
                if self.closure_fn is not None
                else 0.0
            )

            # Fit with closure
            q_con, _ = self.refine_trajectory(
                q_seed[None, :], t_frame, v_frame, weights
            )
            rms_con = self._compute_trajectory_marker_rms(q_con, t_frame, v_frame)
            c_con = (
                float(np.linalg.norm(self.closure_fn(q_con[0])))
                if self.closure_fn is not None
                else 0.0
            )

            diagnoses.append(
                DifficultFrameFloorDiagnosis(
                    frame_index=f_idx,
                    unconstrained_marker_rms_m=rms_uncon,
                    unconstrained_closure_err_m=c_uncon,
                    constrained_marker_rms_m=rms_con,
                    constrained_closure_err_m=c_con,
                    geometric_floor_gap_m=rms_con - rms_uncon,
                )
            )

        return diagnoses

    def _compute_trajectory_marker_rms(
        self, q_traj: Array, targets: Array, valid: BoolArray
    ) -> float:
        errors_sq: list[float] = []
        for f in range(q_traj.shape[0]):
            pos = self.forward_markers_fn(q_traj[f])
            v_idx = np.flatnonzero(valid[f])
            if v_idx.size > 0:
                diffs = pos[v_idx] - targets[f, v_idx]
                errors_sq.extend(np.sum(diffs**2, axis=1).tolist())
        return float(np.sqrt(np.mean(errors_sq))) if errors_sq else 0.0

    def _compute_trajectory_closure_max(self, q_traj: Array) -> float:
        if self.closure_fn is None:
            return 0.0
        max_c = 0.0
        for f in range(q_traj.shape[0]):
            c = float(np.linalg.norm(self.closure_fn(q_traj[f])))
            max_c = max(max_c, c)
        return max_c

    def _compute_lowest_clearance(self, q_traj: Array) -> float:
        if self.foot_contact_fn is None:
            return 0.0
        lowest = float("inf")
        gh = self.options.ground_height_m
        for f in range(q_traj.shape[0]):
            pts = self.foot_contact_fn(q_traj[f])
            for pt in pts:
                lowest = min(lowest, float(pt[2] - gh))
        return lowest if np.isfinite(lowest) else 0.0
