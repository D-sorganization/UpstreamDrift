"""Constrained multiple-shooting motion matching with global degree-6 polynomial parameterization (#9921 / #9964).

Decomposes the forward integration horizon into multiple shooting sub-windows
connected by state defect constraints, while evaluating every sub-window under
the EXACT SAME global degree-6 polynomial control law.
The defect vector d_k = x_end(k) - x_start(k+1) is constrained or penalized to zero.
Final candidate acceptance requires an unsegmented forward simulation from t=0
with zero state resets.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from src.shared.python.motion_matching.prefix_fit import (
    MarkerTarget,
    _readonly,
)

Array: TypeAlias = NDArray[np.float64]
SegmentedForward: TypeAlias = Callable[
    [Array, Array, Array | None], tuple[Array, Array]
]
UnsegmentedForward: TypeAlias = Callable[[Array, Array], Array]


@dataclass(frozen=True)
class MultipleShootingOptions:
    """Configuration for constrained multiple-shooting fitting."""

    shooting_nodes: tuple[float, ...]
    state_dim: int = 0
    defect_weight: float = 100.0
    defect_tolerance: float = 1e-3
    max_nfev: int = 150
    finite_difference_step: float | None = None
    terminal_weight: float = 0.0
    pelvis_indices: tuple[int, int] | None = None
    pelvis_yaw_weight: float = 0.0
    pelvis_yaw_max_error_pct: float = 5.0
    regularization: Callable[[Array], Array] | None = None

    def __post_init__(self) -> None:
        nodes = np.asarray(self.shooting_nodes, dtype=float)
        if (
            nodes.ndim != 1
            or nodes.size == 0
            or not np.isfinite(nodes).all()
            or nodes[0] <= 0
            or np.any(np.diff(nodes) <= 0)
        ):
            raise ValueError(
                "shooting_nodes must be positive, finite, and strictly increasing"
            )
        if self.defect_weight < 0:
            raise ValueError("defect_weight must be non-negative")
        if self.defect_tolerance <= 0:
            raise ValueError("defect_tolerance must be positive")


@dataclass(frozen=True)
class MultipleShootingFit:
    """Result of multiple-shooting optimization."""

    theta: Array
    intermediate_states: dict[float, Array]
    max_defect_norm: float
    segmented_rmse_m: float
    unsegmented_rmse_m: float
    optimizer_converged: bool
    accepted: bool
    message: str


def fit_multiple_shooting(
    target: MarkerTarget,
    segmented_forward: SegmentedForward,
    unsegmented_forward: UnsegmentedForward,
    *,
    initial_theta: Array,
    lower_theta: Array,
    upper_theta: Array,
    initial_states: Mapping[float, Array],
    state_bounds: Mapping[float, tuple[Array, Array]],
    options: MultipleShootingOptions,
) -> MultipleShootingFit:
    """Optimize global polynomial parameters theta and intermediate states via multiple shooting."""
    nodes = options.shooting_nodes
    n_windows = len(nodes)
    theta_dim = len(initial_theta)

    # Windows: [0, nodes[0]], [nodes[0], nodes[1]], ..., [nodes[-2], nodes[-1]]
    window_boundaries = (0.0,) + nodes

    # Intermediate state nodes are nodes[:-1] (exclude terminal end)
    internal_nodes = nodes[:-1]

    # Pack decision variables: [theta, state_1, state_2, ...]
    var_list = [initial_theta]
    lower_list = [lower_theta]
    upper_list = [upper_theta]

    state_offsets = {}
    curr_offset = theta_dim
    for node in internal_nodes:
        if node not in initial_states or node not in state_bounds:
            raise ValueError(
                f"Missing initial state or bounds for shooting node {node}"
            )
        s_init = np.asarray(initial_states[node], dtype=float)
        s_lo, s_hi = state_bounds[node]
        var_list.append(s_init)
        lower_list.append(s_lo)
        upper_list.append(s_hi)
        state_offsets[node] = (curr_offset, curr_offset + len(s_init))
        curr_offset += len(s_init)

    x0 = np.concatenate(var_list)
    x_lower = np.concatenate(lower_list)
    x_upper = np.concatenate(upper_list)

    # Prepare observation masks per window
    window_data = []
    for i in range(n_windows):
        t_start = window_boundaries[i]
        t_end = window_boundaries[i + 1]
        mask = (target.time >= t_start - 1e-12) & (target.time <= t_end + 1e-12)
        w_time = target.time[mask]
        w_points = target.points[mask]
        w_obs = np.isfinite(w_points).all(axis=2) & (target.weights > 0)
        window_data.append((t_start, t_end, w_time, w_points, w_obs))

    window_cache: dict[int, tuple[bytes, bytes | None, Array, Array]] = {}

    def residual(p: Array) -> Array:
        theta = p[:theta_dim]
        theta_bytes = theta.tobytes()
        states = {}
        for node in internal_nodes:
            s_lo, s_hi = state_offsets[node]
            states[node] = p[s_lo:s_hi]

        res_parts = []
        end_states = []

        # 1. Evaluate segmented forward dynamics on each window
        for i in range(n_windows):
            t_start, t_end, w_time, w_points, w_obs = window_data[i]
            init_s = None if i == 0 else states[t_start]
            init_s_bytes = None if init_s is None else init_s.tobytes()

            cached = window_cache.get(i)
            if (
                cached is not None
                and cached[0] == theta_bytes
                and cached[1] == init_s_bytes
            ):
                pred_markers, end_s = cached[2], cached[3]
            else:
                pred_markers, end_s = segmented_forward(theta, w_time, init_s)
                window_cache[i] = (theta_bytes, init_s_bytes, pred_markers, end_s)

            end_states.append(end_s)

            # Marker residuals
            delta = (pred_markers - w_points)[w_obs]
            w_grid = np.broadcast_to(target.weights[None, :], w_obs.shape)
            w_root = np.sqrt(w_grid[w_obs])
            marker_res = (delta * w_root[:, None]).ravel()
            res_parts.append(marker_res)

        # 2. Defect residuals: d_i = end_state[i] - states[node_i]
        for i, node in enumerate(internal_nodes):
            defect = end_states[i] - states[node]
            res_parts.append(options.defect_weight * defect)

        # 3. Terminal frame & pelvis yaw residuals
        if options.terminal_weight > 0:
            last_w_time, last_w_points, last_w_obs = (
                window_data[-1][2],
                window_data[-1][3],
                window_data[-1][4],
            )
            if len(last_w_points) > 0 and np.any(last_w_obs[-1]):
                term_delta = (pred_markers[-1] - last_w_points[-1])[last_w_obs[-1]]
                res_parts.append(options.terminal_weight * term_delta.ravel())

        if options.pelvis_indices is not None and options.pelvis_yaw_weight > 0:
            wl_i, wr_i = options.pelvis_indices
            if pred_markers.shape[1] > max(wl_i, wr_i) and len(window_data[-1][3]) > 0:
                obs_term = window_data[-1][3][-1]
                v_p = pred_markers[-1, wr_i, :2] - pred_markers[-1, wl_i, :2]
                v_t = obs_term[wr_i, :2] - obs_term[wl_i, :2]
                norm_p = float(np.linalg.norm(v_p))
                norm_t = float(np.linalg.norm(v_t))
                if norm_p > 1e-6 and norm_t > 1e-6:
                    sin_yaw = (v_p[1] * v_t[0] - v_p[0] * v_t[1]) / (norm_p * norm_t)
                    res_parts.append(np.array([options.pelvis_yaw_weight * sin_yaw]))

        # 4. Regularization
        if options.regularization is not None:
            reg_res = options.regularization(theta)
            if reg_res is not None and len(reg_res) > 0:
                res_parts.append(reg_res)

        return np.concatenate(res_parts)

    diff_step = (
        np.full_like(x0, options.finite_difference_step)
        if options.finite_difference_step is not None
        else None
    )

    optimum = least_squares(
        residual,
        x0,
        bounds=(x_lower, x_upper),
        max_nfev=options.max_nfev,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        x_scale="jac",
        diff_step=diff_step,
    )

    opt_theta = optimum.x[:theta_dim].copy()
    opt_states = {}
    for node in internal_nodes:
        idx_lo, idx_hi = state_offsets[node]
        opt_states[node] = optimum.x[idx_lo:idx_hi].copy()

    # Compute final defect norm and segmented RMSE across windows
    defects = []
    seg_dists_list = []
    for i in range(n_windows):
        t_start, t_end, w_time, w_points, w_obs = window_data[i]
        init_s = None if i == 0 else opt_states[t_start]
        pred_markers, end_s = segmented_forward(opt_theta, w_time, init_s)
        if i < len(internal_nodes):
            node = internal_nodes[i]
            defects.append(float(np.linalg.norm(end_s - opt_states[node])))
        if np.any(w_obs):
            seg_dists_list.append((pred_markers - w_points)[w_obs])

    max_defect = float(max(defects)) if len(defects) > 0 else 0.0
    if seg_dists_list:
        seg_dists = np.concatenate(seg_dists_list, axis=0)
        segmented_rmse = float(np.sqrt(np.mean(np.sum(seg_dists**2, axis=-1))))
    else:
        segmented_rmse = 0.0

    # Evaluate unsegmented forward simulation across all time
    unsegmented_pred = unsegmented_forward(opt_theta, target.time)
    obs_all = np.isfinite(target.points).all(axis=2) & (target.weights > 0)
    unseg_dists = np.linalg.norm((unsegmented_pred - target.points)[obs_all], axis=1)

    is_finite = bool(
        np.isfinite(unsegmented_pred).all() and np.isfinite(unseg_dists).all()
    )
    if is_finite and len(unseg_dists) > 0:
        unsegmented_rmse = float(np.sqrt(np.mean(unseg_dists**2)))
    else:
        unsegmented_rmse = float("inf")

    accepted = bool(
        optimum.success
        and is_finite
        and max_defect <= options.defect_tolerance
        and unsegmented_rmse < float("inf")
    )

    return MultipleShootingFit(
        theta=_readonly(opt_theta),
        intermediate_states=opt_states,
        max_defect_norm=max_defect,
        segmented_rmse_m=segmented_rmse,
        unsegmented_rmse_m=unsegmented_rmse,
        optimizer_converged=bool(optimum.success),
        accepted=accepted,
        message=str(optimum.message),
    )


def verify_unsegmented_forward_rollout(
    target: MarkerTarget,
    unsegmented_forward: UnsegmentedForward,
    theta: Array,
) -> dict[str, Any]:
    """Verify that candidate theta produces a valid 100% continuous forward rollout from t=0."""
    pred = unsegmented_forward(theta, target.time)
    obs = np.isfinite(target.points).all(axis=2) & (target.weights > 0)
    dists = np.linalg.norm((pred - target.points)[obs], axis=1)
    rmse_m = float(np.sqrt(np.mean(dists**2)))

    # Terminal metrics
    term_obs = obs[-1]
    term_dists = np.linalg.norm((pred[-1] - target.points[-1])[term_obs], axis=1)
    term_rmse = float(np.sqrt(np.mean(term_dists**2)))
    term_max = float(np.max(term_dists))

    return {
        "rmse_m": rmse_m,
        "terminal_rmse_m": term_rmse,
        "terminal_max_m": term_max,
        "prediction": pred,
    }
