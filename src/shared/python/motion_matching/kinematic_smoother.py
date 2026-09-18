"""Joint kinematic smoother with derivative consistency and weld closure preservation (PF-02, #10432).

Provides Quintic B-spline / Hermite smoothing of generalized coordinates q(t),
generating mathematically consistent joint rates v(t) = dq/dt and accelerations
a(t) = d2q/dt2 without boundary jerk spikes, while preserving dual-grip weld
closure constraints and respecting coordinate bounds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
import scipy.interpolate as sp_interp
import scipy.signal as sp_signal

from src.shared.python.contracts import ensure, require

logger = logging.getLogger(__name__)

Array: TypeAlias = NDArray[np.float64]
ClosureFn: TypeAlias = Callable[[Array], Array]
ClosureJacFn: TypeAlias = Callable[[Array], Array]


@dataclass(frozen=True)
class SmoothingAudit:
    """Diagnostic metrics evaluating kinematic trajectory smoothing quality."""

    frame_count: int
    duration_s: float
    sample_rate_hz: float
    cutoff_hz: float
    max_acceleration_step: float
    boundary_spike_start_ratio: float
    boundary_spike_end_ratio: float
    max_jerk: float
    position_rms_diff: float
    max_closure_error_m: float
    max_closure_rate_m_s: float
    bounds_violation_count: int
    no_dropped_frames: bool


@dataclass(frozen=True)
class SmoothKinematicTrajectory:
    """Smooth kinematic state trajectory with consistent rates and accelerations."""

    time_s: Array
    q: Array
    v: Array
    a: Array
    closure_error_m: Array
    closure_rate_m_s: Array
    audit: SmoothingAudit


@dataclass(frozen=True)
class KinematicSmootherOptions:
    """Options for kinematic trajectory smoothing."""

    cutoff_hz: float = 12.0
    dt: float = 1.0 / 360.0
    filter_order: int = 4
    boundary_padding_frames: int = 15
    closure_tolerance_m: float = 1e-3
    closure_velocity_tolerance_m_s: float = 1e-2
    closure_max_iterations: int = 5
    closure_damping: float = 1e-6
    enforce_bounds: bool = True


class KinematicSmoother:
    """Engine-agnostic kinematic smoother producing consistent q, v, a."""

    def __init__(
        self,
        options: KinematicSmootherOptions | None = None,
        *,
        lower: Array | None = None,
        upper: Array | None = None,
        closure_fn: ClosureFn | None = None,
        closure_jacobian_fn: ClosureJacFn | None = None,
    ) -> None:
        self.options = options or KinematicSmootherOptions()
        self.lower = None if lower is None else np.asarray(lower, dtype=float)
        self.upper = None if upper is None else np.asarray(upper, dtype=float)
        self.closure_fn = closure_fn
        self.closure_jacobian_fn = closure_jacobian_fn

        if self.lower is not None and self.upper is not None:
            require(
                self.lower.shape == self.upper.shape,
                "lower and upper bounds must share dimensions",
            )
            require(
                bool(np.all(self.lower <= self.upper)),
                "lower bounds must not exceed upper bounds",
            )

    def _numerical_closure_jacobian(
        self, q: Array, closure_fn: ClosureFn, eps: float = 1e-6
    ) -> Array:
        """Compute finite-difference Jacobian of closure constraint."""
        c0 = closure_fn(q)
        m = c0.size
        n = q.size
        jac = np.zeros((m, n), dtype=float)
        for i in range(n):
            q_plus = q.copy()
            q_plus[i] += eps
            c_plus = closure_fn(q_plus)
            jac[:, i] = (c_plus - c0) / eps
        return jac

    def _project_closure_and_bounds(
        self,
        q_traj: Array,
        v_traj: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Project trajectory onto closure manifold and coordinate bounds."""
        n_frames, n_coords = q_traj.shape
        q_proj = q_traj.copy()
        v_proj = v_traj.copy()
        closure_errs = np.zeros(n_frames, dtype=float)
        closure_rates = np.zeros(n_frames, dtype=float)

        closure_fn = self.closure_fn
        closure_jac_fn = self.closure_jacobian_fn
        opts = self.options

        for k in range(n_frames):
            q_curr = q_proj[k]
            v_curr = v_proj[k]

            if closure_fn is not None:
                for _ in range(opts.closure_max_iterations):
                    c_val = closure_fn(q_curr)
                    c_norm = float(np.linalg.norm(c_val))
                    if c_norm <= opts.closure_tolerance_m:
                        break
                    jac = (
                        closure_jac_fn(q_curr)
                        if closure_jac_fn is not None
                        else self._numerical_closure_jacobian(q_curr, closure_fn)
                    )
                    # Damped least-squares correction: dq = -J^T (J J^T + lambda I)^-1 c
                    gram = jac @ jac.T + opts.closure_damping * np.eye(jac.shape[0])
                    step = -jac.T @ np.linalg.solve(gram, c_val)
                    q_curr = q_curr + step
                    if (
                        opts.enforce_bounds
                        and self.lower is not None
                        and self.upper is not None
                    ):
                        q_curr = np.clip(q_curr, self.lower, self.upper)

                c_final = closure_fn(q_curr)
                closure_errs[k] = float(np.linalg.norm(c_final))

                # Project velocity into closure nullspace: v_proj = v - J^# (J v)
                jac_v = (
                    closure_jac_fn(q_curr)
                    if closure_jac_fn is not None
                    else self._numerical_closure_jacobian(q_curr, closure_fn)
                )
                j_v = jac_v @ v_curr
                gram_v = jac_v @ jac_v.T + opts.closure_damping * np.eye(jac_v.shape[0])
                v_corr = jac_v.T @ np.linalg.solve(gram_v, j_v)
                v_curr = v_curr - v_corr
                closure_rates[k] = float(np.linalg.norm(jac_v @ v_curr))
            else:
                if (
                    opts.enforce_bounds
                    and self.lower is not None
                    and self.upper is not None
                ):
                    q_curr = np.clip(q_curr, self.lower, self.upper)

            q_proj[k] = q_curr
            v_proj[k] = v_curr

        return q_proj, v_proj, closure_errs, closure_rates

    def smooth(
        self,
        q_raw: Array,
        *,
        dt: float | None = None,
        time_s: Array | None = None,
    ) -> SmoothKinematicTrajectory:
        """Smooth raw coordinates and compute consistent rates and accelerations.

        Args:
            q_raw: (N, coordinates) raw joint position trajectory.
            dt: Time step in seconds. If None, derived from time_s or options.
            time_s: Monotonic timestamps array (N,).
        """
        q_arr = np.asarray(q_raw, dtype=float)
        require(
            q_arr.ndim == 2, "q_raw must be a 2D array of shape (frames, coordinates)"
        )
        n_frames, n_coords = q_arr.shape
        require(n_frames >= 2, "Trajectory must have at least 2 frames")

        opts = self.options
        step_dt = float(dt if dt is not None else opts.dt)
        require(step_dt > 0.0, "Time step must be positive")

        if time_s is not None:
            t_grid = np.asarray(time_s, dtype=float)
            require(t_grid.shape == (n_frames,), "time_s must match frame count")
        else:
            t_grid = np.arange(n_frames, dtype=float) * step_dt

        # Filter design: zero-phase Butterworth
        fs = 1.0 / step_dt
        nyquist = 0.5 * fs
        norm_cutoff = min(opts.cutoff_hz / nyquist, 0.95)
        require(norm_cutoff > 0.0, "Cutoff frequency must be positive")

        if n_frames < 6:
            # Fallback for ultra-short sequences
            q_filtered = q_arr.copy()
            v_smooth = np.gradient(q_filtered, step_dt, axis=0)
            a_smooth = np.gradient(v_smooth, step_dt, axis=0)
        else:
            b, a_filter = sp_signal.butter(
                opts.filter_order, norm_cutoff, btype="low", analog=False
            )

            # Reflection padding to eliminate boundary acceleration and jerk spikes
            pad_len = min(opts.boundary_padding_frames, n_frames - 1)
            q_pad_left = 2.0 * q_arr[0] - q_arr[1 : pad_len + 1][::-1]
            q_pad_right = 2.0 * q_arr[-1] - q_arr[-pad_len - 1 : -1][::-1]
            q_padded = np.vstack([q_pad_left, q_arr, q_pad_right])

            # Zero-phase bidirectional filtering
            q_filt_padded = sp_signal.filtfilt(b, a_filter, q_padded, axis=0)
            q_filtered = q_filt_padded[pad_len : pad_len + n_frames]

            # Quintic B-spline fit on padded points to ensure C4 smoothness and eliminate boundary derivative spikes
            t_pad_left = t_grid[0] - np.arange(pad_len, 0, -1, dtype=float) * step_dt
            t_pad_right = t_grid[-1] + np.arange(1, pad_len + 1, dtype=float) * step_dt
            t_padded = np.concatenate([t_pad_left, t_grid, t_pad_right])

            v_smooth = np.zeros((n_frames, n_coords), dtype=float)
            a_smooth = np.zeros((n_frames, n_coords), dtype=float)
            k_spline = min(5, t_padded.size - 1)

            for col in range(n_coords):
                spl = sp_interp.splrep(t_padded, q_filt_padded[:, col], k=k_spline, s=0)
                q_filtered[:, col] = sp_interp.splev(t_grid, spl, der=0)
                v_smooth[:, col] = sp_interp.splev(t_grid, spl, der=1)
                a_smooth[:, col] = sp_interp.splev(t_grid, spl, der=2)

        # Enforce bounds and weld loop-closure
        q_proj, v_proj, closure_errs, closure_rates = self._project_closure_and_bounds(
            q_filtered, v_smooth
        )

        # If closure or bounds projection nudged q, recompute compatible acceleration
        if self.closure_fn is not None or (
            opts.enforce_bounds and self.lower is not None
        ):
            a_final = np.gradient(v_proj, step_dt, axis=0)
        else:
            a_final = a_smooth

        # Audit smoothing metrics
        delta_a = np.linalg.norm(np.diff(a_final, axis=0), axis=1)
        max_acc_step = float(np.max(delta_a)) if delta_a.size else 0.0
        median_acc_step = float(np.median(delta_a)) if delta_a.size else 1e-9
        start_ratio = (
            float(delta_a[0] / (delta_a[1] + 1e-9)) if delta_a.size > 1 else 1.0
        )
        end_ratio = (
            float(delta_a[-1] / (delta_a[-2] + 1e-9)) if delta_a.size > 1 else 1.0
        )
        jerk = delta_a / step_dt if delta_a.size else np.zeros(0)
        max_jerk = float(np.max(jerk)) if jerk.size else 0.0
        rms_diff = float(np.sqrt(np.mean((q_proj - q_arr) ** 2)))

        bounds_violations = 0
        if self.lower is not None and self.upper is not None:
            below = q_proj < (self.lower - 1e-6)
            above = q_proj > (self.upper + 1e-6)
            bounds_violations = int(np.sum(below | above))

        audit = SmoothingAudit(
            frame_count=n_frames,
            duration_s=float(t_grid[-1] - t_grid[0]),
            sample_rate_hz=fs,
            cutoff_hz=opts.cutoff_hz,
            max_acceleration_step=max_acc_step,
            boundary_spike_start_ratio=start_ratio,
            boundary_spike_end_ratio=end_ratio,
            max_jerk=max_jerk,
            position_rms_diff=rms_diff,
            max_closure_error_m=(
                float(np.max(closure_errs)) if closure_errs.size else 0.0
            ),
            max_closure_rate_m_s=(
                float(np.max(closure_rates)) if closure_rates.size else 0.0
            ),
            bounds_violation_count=bounds_violations,
            no_dropped_frames=(len(q_proj) == n_frames),
        )

        ensure(len(q_proj) == n_frames, "Smoothed trajectory must not drop frames")
        ensure(
            bool(
                np.isfinite(q_proj).all()
                and np.isfinite(v_proj).all()
                and np.isfinite(a_final).all()
            ),
            "Kinematic states must be fully finite",
        )

        return SmoothKinematicTrajectory(
            time_s=t_grid,
            q=q_proj,
            v=v_proj,
            a=a_final,
            closure_error_m=closure_errs,
            closure_rate_m_s=closure_rates,
            audit=audit,
        )
