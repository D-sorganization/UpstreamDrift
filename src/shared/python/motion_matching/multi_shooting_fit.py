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
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag
from scipy.optimize import least_squares

from src.shared.python.motion_matching.equality_least_squares import (
    solve_equality_least_squares,
)
from src.shared.python.motion_matching.marker_replay_report import observed_rms
from src.shared.python.motion_matching.residual_regularization import (
    regularization_derivative,
    regularization_residual,
)
from src.shared.python.motion_matching.prefix_fit import (
    MarkerTarget,
    _readonly,
)

Array: TypeAlias = NDArray[np.float64]
SegmentedForward: TypeAlias = Callable[
    [Array, Array, Array | None], tuple[Array, Array]
]
SegmentedForwardBatch: TypeAlias = Callable[
    [Sequence[tuple[Array, Array, Array | None]]],
    Sequence[tuple[Array, Array]],
]
UnsegmentedForward: TypeAlias = Callable[[Array, Array], Array]


@dataclass(frozen=True)
class MultipleShootingOptions:
    """Configuration for multiple-shooting fitting.

    slsqp uses max_iterations and a separate hard max_nfev residual budget.
    Its fixed constraint_projection maps scaled physical defect rows to
    independent equalities. defect_weight is ignored for that backend.
    Callbacks still observe the assembled residual (including unweighted
    physical defects); the constrained objective excludes those defect rows.
    Optional variable_scales apply only to slsqp, ordered as theta followed by
    each internal node's optimization coordinates in increasing time order.
    Physical node transforms, callback values and defect scales are unchanged.
    """

    shooting_nodes: tuple[float, ...]
    state_dim: int = 0
    defect_weight: float = 100.0
    defect_tolerance: float = 1e-3
    max_nfev: int = 150
    step_tolerance: float | None = 1e-8
    finite_difference_step: float | None = None
    terminal_weight: float = 0.0
    pelvis_indices: tuple[int, int] | None = None
    pelvis_yaw_weight: float = 0.0
    pelvis_yaw_max_error_pct: float = 5.0
    regularization: Callable[[Array], Array] | None = None
    callback: Callable[[Array, Array, float], None] | None = None
    checkpoint_callback: (
        Callable[[Array, Mapping[float, Array], float], None] | None
    ) = None
    state_transform: Callable[[float, Array], Array] | None = None
    defect_scales: Array | None = None
    acceptance: Callable[[Array], bool] | None = None
    window_jacobian: (
        Callable[[Array, Array, Array | None], tuple[Array, Array]] | None
    ) = None
    state_transform_jacobian: Callable[[float, Array], Array] | None = None
    solver: Literal["least_squares", "slsqp"] = "least_squares"
    max_iterations: int = 50
    equality_tolerance: float = 1e-8
    constraint_projection: Callable[[float], Array] | None = None
    variable_scales: Array | None = None
    segmented_forward_batch: SegmentedForwardBatch | None = None
    regularization_jacobian: Callable[[Array], Array] | None = None

    def __post_init__(self) -> None:
        if self.regularization_jacobian is not None and self.regularization is None:
            raise ValueError("regularization_jacobian requires regularization")
        if self.solver not in ("least_squares", "slsqp"):
            raise ValueError("Unknown shooting solver")
        if self.segmented_forward_batch is not None and not callable(
            self.segmented_forward_batch
        ):
            raise ValueError("Segmented forward batch must be callable")
        if self.variable_scales is not None and self.solver != "slsqp":
            raise ValueError("Explicit variable scales require the slsqp backend")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations <= 0
            or not np.isfinite(self.equality_tolerance)
            or self.equality_tolerance <= 0
        ):
            raise ValueError("Invalid constrained iteration budget or tolerance")
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
        if self.step_tolerance is not None and (
            isinstance(self.step_tolerance, bool)
            or not isinstance(self.step_tolerance, (int, float, np.floating))
            or not np.isfinite(self.step_tolerance)
            or self.step_tolerance <= np.finfo(float).eps
        ):
            raise ValueError(
                "Invalid step tolerance; use None to disable step stopping"
            )


@dataclass(frozen=True)
class MultipleShootingFit:
    """Result with physical intermediate states and scaled defect norms.

    accepted requires the explicit application acceptance callback as well as
    finite continuous replay, optimizer convergence and bounded state defects.
    A missing callback never qualifies a fit.
    terminal_replay_gap_m is observed-marker pointwise RMS between the last
    shooting window and continuous replay. None means absent endpoint evidence;
    infinity means a nonfinite observed endpoint. It is diagnostic, not a gate.
    optimality is None for SLSQP: its result has no comparable scaled
    least_squares optimality measure. Full physical defects still gate acceptance.
    """

    theta: Array
    intermediate_states: dict[float, Array]
    max_defect_norm: float
    segmented_rmse_m: float
    unsegmented_rmse_m: float
    optimizer_converged: bool
    accepted: bool
    message: str
    defect_norms: dict[float, float]
    optimality: float | None
    function_evaluations: int
    active_bound_count: int
    terminal_replay_gap_m: float | None = None


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

    def physical_state(time: float, variables: Array) -> Array:
        state = np.asarray(
            variables
            if options.state_transform is None
            else options.state_transform(time, variables),
            dtype=float,
        )
        if state.ndim != 1 or not state.size or not np.isfinite(state).all():
            raise ValueError("Node transform must return a finite state vector")
        return state.copy()

    def scaled_defect(end: Array, start: Array) -> Array:
        if end.shape != start.shape or not np.isfinite(end).all():
            raise ValueError("Shooting endpoint state shape or finiteness differs")
        scales = (
            np.ones_like(start)
            if options.defect_scales is None
            else np.asarray(options.defect_scales)
        )
        if (
            scales.shape != start.shape
            or not np.isfinite(scales).all()
            or np.any(scales <= 0)
        ):
            raise ValueError("Defect scales must match state and be finite positive")
        return (end - start) / scales

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
    defect_factor = 1.0 if options.solver == "slsqp" else options.defect_weight

    def residual(p: Array) -> Array:
        theta = p[:theta_dim]
        theta_bytes = theta.tobytes()
        states = {}
        for node in internal_nodes:
            s_lo, s_hi = state_offsets[node]
            states[node] = physical_state(node, p[s_lo:s_hi])

        res_parts = []
        end_states = []

        # 1. Evaluate segmented forward dynamics on each window. The optional
        # batch boundary receives only cache misses; it is responsible solely
        # for independent window execution. Cache ownership and residual
        # assembly stay in this coordinating process.
        evaluated: dict[int, tuple[Array, Array]] = {}
        misses: list[tuple[int, Array, Array, Array | None, bytes, bytes | None]] = []
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
                evaluated[i] = (cached[2], cached[3])
            else:
                misses.append((i, theta, w_time, init_s, theta_bytes, init_s_bytes))

        if not misses:
            fresh: list[tuple[Array, Array]] = []
        elif options.segmented_forward_batch is None:
            fresh = [
                segmented_forward(theta, clock, state)
                for _index, theta, clock, state, _theta_bytes, _state_bytes in misses
            ]
        else:
            fresh = list(
                options.segmented_forward_batch(
                    [
                        (theta, clock, state)
                        for _index, theta, clock, state, *_ in misses
                    ]
                )
            )
        if len(fresh) != len(misses):
            raise ValueError("Segmented batch result count differs from cache misses")
        for miss, result in zip(misses, fresh, strict=True):
            i, _theta, _clock, _state, miss_theta_bytes, miss_state_bytes = miss
            pred_markers, end_s = (np.asarray(part, dtype=float) for part in result)
            if not np.isfinite(pred_markers).all() or not np.isfinite(end_s).all():
                raise ValueError("Segmented batch returned nonfinite values")
            window_cache[i] = (
                miss_theta_bytes,
                miss_state_bytes,
                pred_markers,
                end_s,
            )
            evaluated[i] = (pred_markers, end_s)

        # Existing marker/defect assembly intentionally stays ordered and local.
        for i in range(n_windows):
            t_start, t_end, w_time, w_points, w_obs = window_data[i]
            pred_markers, end_s = evaluated[i]
            end_states.append(end_s)

            # Marker residuals
            delta = (pred_markers - w_points)[w_obs]
            w_grid = np.broadcast_to(target.weights[None, :], w_obs.shape)
            w_root = np.sqrt(w_grid[w_obs])
            marker_res = (delta * w_root[:, None]).ravel()
            res_parts.append(marker_res)

        # 2. Defect residuals: d_i = end_state[i] - states[node_i]
        for i, node in enumerate(internal_nodes):
            defect = scaled_defect(end_states[i], states[node])
            res_parts.append(defect_factor * defect)

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
            reg_res = regularization_residual(theta, options.regularization)
            if reg_res.size > 0:
                res_parts.append(reg_res)

        full_res = np.concatenate(res_parts)
        # Every evaluation, including rejected trials and finite differences, may
        # be observed. These snapshots are not acceptance or convergence events.
        cost = float(full_res @ full_res)
        if options.callback is not None:
            options.callback(_readonly(theta), _readonly(full_res), cost)
        if options.checkpoint_callback is not None:
            options.checkpoint_callback(
                _readonly(theta),
                {time: _readonly(state) for time, state in states.items()},
                cost,
            )
        return full_res

    def analytic_jacobian(p: Array) -> Array:
        if options.window_jacobian is None:
            raise ValueError("Missing window Jacobian")
        parts, ends, node_jacobians = [], [], {}
        for node, (lo, hi) in state_offsets.items():
            physical = physical_state(node, p[lo:hi])
            callback = options.state_transform_jacobian
            if options.state_transform is not None and callback is None:
                raise ValueError("Missing node transform Jacobian")
            transform = (
                np.eye(hi - lo)
                if callback is None
                else np.asarray(callback(node, p[lo:hi]))
            )
            if (
                transform.shape != (physical.size, hi - lo)
                or not np.isfinite(transform).all()
            ):
                raise ValueError("Invalid node transform Jacobian")
            full = np.zeros((physical.size, p.size))
            full[:, lo:hi] = transform
            node_jacobians[node] = full
        terminal_jac = None
        for i, (start, _end, clock, points, observed) in enumerate(window_data):
            state = (
                None
                if i == 0
                else physical_state(start, p[slice(*state_offsets[start])])
            )
            marker, endpoint = options.window_jacobian(p[:theta_dim], clock, state)
            marker, endpoint = np.asarray(marker), np.asarray(endpoint)
            local_size = theta_dim + (0 if state is None else state.size)
            if (
                marker.shape != (*points.shape, local_size)
                or endpoint.ndim != 2
                or endpoint.shape[1] != local_size
                or not np.isfinite(marker).all()
                or not np.isfinite(endpoint).all()
            ):
                raise ValueError("Invalid window Jacobian shape or values")
            chain = np.zeros((local_size, p.size))
            chain[:theta_dim, :theta_dim] = np.eye(theta_dim)
            if state is not None:
                chain[theta_dim:] = node_jacobians[start]
            global_marker = marker @ chain
            ends.append(endpoint @ chain)
            weights = np.broadcast_to(target.weights[None, :], observed.shape)
            parts.append(
                (
                    global_marker[observed] * np.sqrt(weights[observed])[:, None, None]
                ).reshape(-1, p.size)
            )
            if (
                i == n_windows - 1
                and options.terminal_weight > 0
                and observed[-1].any()
            ):
                terminal_jac = options.terminal_weight * global_marker[
                    -1, observed[-1]
                ].reshape(-1, p.size)
        for i, node in enumerate(internal_nodes):
            delta = ends[i] - node_jacobians[node]
            scales = (
                np.ones(delta.shape[0])
                if options.defect_scales is None
                else options.defect_scales
            )
            parts.append(defect_factor * delta / np.asarray(scales)[:, None])
        if terminal_jac is not None:
            parts.append(terminal_jac)
        if options.regularization is not None:
            if options.regularization_jacobian is None:
                raise ValueError("Missing regularization_jacobian")
            local = regularization_derivative(
                p[:theta_dim], options.regularization, options.regularization_jacobian
            )
            penalty = np.zeros((local.shape[0], p.size))
            penalty[:, :theta_dim] = local
            parts.append(penalty)
        return np.concatenate(parts)

    if options.window_jacobian is not None:
        if options.pelvis_yaw_weight > 0:
            raise ValueError("Analytic shooting Jacobian does not yet support yaw")
        if (
            options.regularization is not None
            and options.regularization_jacobian is None
        ):
            raise ValueError(
                "Analytic shooting requires regularization_jacobian for penalties"
            )
        if (
            options.state_transform is not None
            and options.state_transform_jacobian is None
        ):
            raise ValueError("Analytic transformed nodes require a transform Jacobian")

    diff_step = (
        np.full_like(x0, options.finite_difference_step)
        if options.finite_difference_step is not None
        else None
    )

    if options.solver == "slsqp":
        if options.window_jacobian is None or not internal_nodes:
            raise ValueError(
                "Constrained shooting requires analytic windows and internal nodes"
            )
        projections = []
        for node in internal_nodes:
            size = physical_state(node, np.asarray(initial_states[node])).size
            projection = (
                np.eye(size)
                if options.constraint_projection is None
                else np.asarray(options.constraint_projection(node), dtype=float)
            )
            if projection.ndim != 2 or projection.shape[1] != size:
                raise ValueError(
                    "Constraint projection must map scaled physical defects"
                )
            projections.append(projection)
        optimum = solve_equality_least_squares(
            residual,
            analytic_jacobian,
            x0,
            x_lower,
            x_upper,
            equality_start=sum(int(w[4].sum()) * 3 for w in window_data),
            projection=block_diag(*projections),
            max_iterations=options.max_iterations,
            max_evaluations=options.max_nfev,
            constraint_tolerance=options.equality_tolerance,
            variable_scales=options.variable_scales,
        )
    else:
        optimum = least_squares(
            residual,
            x0,
            jac=analytic_jacobian if options.window_jacobian is not None else "2-point",
            bounds=(x_lower, x_upper),
            max_nfev=options.max_nfev,
            ftol=1e-8,
            xtol=options.step_tolerance,
            gtol=1e-8,
            x_scale="jac",
            diff_step=diff_step,
        )

    opt_theta = optimum.x[:theta_dim].copy()
    opt_states = {}
    for node in internal_nodes:
        idx_lo, idx_hi = state_offsets[node]
        opt_states[node] = physical_state(node, optimum.x[idx_lo:idx_hi])

    # Compute final defect norm and segmented RMSE across windows
    defects = []
    seg_dists_list = []
    for i in range(n_windows):
        t_start, t_end, w_time, w_points, w_obs = window_data[i]
        init_s = None if i == 0 else opt_states[t_start]
        pred_markers, end_s = segmented_forward(opt_theta, w_time, init_s)
        if i < len(internal_nodes):
            node = internal_nodes[i]
            defects.append(
                float(np.linalg.norm(scaled_defect(end_s, opt_states[node])))
            )
        if np.any(w_obs):
            seg_dists_list.append((pred_markers - w_points)[w_obs])

    max_defect = float(max(defects)) if len(defects) > 0 else 0.0
    if seg_dists_list:
        seg_dists = np.concatenate(seg_dists_list, axis=0)
        segmented_rmse = float(np.sqrt(np.mean(np.sum(seg_dists**2, axis=-1))))
    else:
        segmented_rmse = 0.0

    terminal_segmented = pred_markers[-1].copy()
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

    # Pointwise disagreement is not the difference of two RMS-to-target values.
    # Missing observations or unequal endpoint clocks provide no gap evidence.
    terminal_replay_gap = None
    if w_time[-1] == target.time[-1] and np.any(obs_all[-1]):
        gap_errors = np.linalg.norm(terminal_segmented - unsegmented_pred[-1], axis=1)
        terminal_replay_gap = (
            observed_rms(gap_errors, obs_all[-1])
            if np.isfinite(gap_errors[obs_all[-1]]).all()
            else float("inf")
        )

    accepted = bool(
        optimum.success
        and is_finite
        and max_defect <= options.defect_tolerance
        and unsegmented_rmse < float("inf")
        and options.acceptance is not None
        and options.acceptance(unsegmented_pred)
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
        defect_norms=dict(zip(internal_nodes, defects, strict=True)),
        optimality=float(optimum.optimality)
        if hasattr(optimum, "optimality")
        else None,
        function_evaluations=int(optimum.nfev),
        active_bound_count=int(np.count_nonzero(optimum.active_mask)),
        terminal_replay_gap_m=terminal_replay_gap,
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
