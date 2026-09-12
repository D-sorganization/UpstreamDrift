"""Progressive, single-initial-state marker fitting for a forward oracle (#9921).

This is an optimizer, not a dynamics engine or a C3D loader. The caller adapts
the existing BodyTarget and engine outputs into a weighted numerical view, and
owns physical bounds, marker attachments and a fixed initial-state contract.
All prefixes start at zero; no measured state is injected between windows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import comb

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from typing import TypeAlias

from .polynomial_torque import COEFFS_PER_JOINT

Array: TypeAlias = NDArray[np.float64]
Forward: TypeAlias = Callable[[Array, Array], Array]


def _readonly(values: Array) -> Array:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def normalized_to_simscape(coefficients: Array, *, duration_s: float) -> Array:
    """Convert ascending powers of t/duration to native A..G powers of seconds.

    Input and result have shape (joints, 7). The result is a fresh array; its
    last column is the constant torque. This preserves one global time origin.
    """
    values = np.asarray(coefficients, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != COEFFS_PER_JOINT
        or not np.isfinite(values).all()
    ):
        raise ValueError("coefficients must be finite with shape (joints, 7)")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        converted = values / duration_s ** np.arange(COEFFS_PER_JOINT)
    if not np.isfinite(converted).all():
        raise ValueError("duration_s causes non-finite polynomial coefficients")
    return converted[:, ::-1].copy()


def bernstein_to_simscape(control_torques: Array, *, duration_s: float) -> Array:
    """Convert degree-zero through six Bernstein torques to native A..G.

    Bounding all control values bounds the entire continuous torque on
    [0, duration_s], by the nonnegative partition of unity. Bounds are on torque,
    not on native power coefficients. No bound is asserted outside that interval.
    """
    values = np.asarray(control_torques, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or not 1 <= values.shape[1] <= COEFFS_PER_JOINT
        or not np.isfinite(values).all()
    ):
        raise ValueError("control torques must be finite with 1 through 7 columns")
    degree = values.shape[1] - 1
    power = np.zeros((values.shape[0], COEFFS_PER_JOINT))
    for k in range(degree + 1):
        for j in range(k, degree + 1):
            power[:, j] += (
                values[:, k]
                * comb(degree, k)
                * comb(degree - k, j - k)
                * (-1) ** (j - k)
            )
    return normalized_to_simscape(power, duration_s=duration_s)


def reexpress_bernstein_basis(
    control_torques: Array,
    *,
    source_duration_s: float,
    target_duration_s: float,
) -> Array:
    """Re-express degree-6 Bernstein controls from one basis duration to another.

    Guarantees that the continuous physical torque tau(t) is strictly identical
    across [0, min(source_duration_s, target_duration_s)]:
        tau_target(t) == tau_source(t) for all t.
    """
    c_src = np.asarray(control_torques, dtype=float)
    if c_src.ndim != 2 or c_src.shape[1] != COEFFS_PER_JOINT:
        raise ValueError(
            f"control_torques must have shape (joints, {COEFFS_PER_JOINT})"
        )
    if source_duration_s <= 0 or target_duration_s <= 0:
        raise ValueError("Basis durations must be strictly positive")

    native_descending = bernstein_to_simscape(c_src, duration_s=source_duration_s)
    native_ascending = native_descending[:, ::-1]

    powers_target = native_ascending * (
        target_duration_s ** np.arange(COEFFS_PER_JOINT)
    )

    degree = COEFFS_PER_JOINT - 1
    m_mat = np.zeros((COEFFS_PER_JOINT, COEFFS_PER_JOINT), dtype=float)
    for k in range(degree + 1):
        for j in range(k, degree + 1):
            m_mat[j, k] = comb(degree, k) * comb(degree - k, j - k) * ((-1) ** (j - k))

    inv_m = np.linalg.inv(m_mat.T)
    c_target = powers_target @ inv_m
    return np.ascontiguousarray(c_target, dtype=float)


def bernstein_effort_range(control_torques: Array) -> Array:
    """Numerical min/max on the full basis interval, including stationary points.

    Uses the existing coefficient conversion on normalized time for conditioning.
    These extrema describe applied efforts, not the conservative control bounds.
    """
    coefficients = bernstein_to_simscape(control_torques, duration_s=1.0)
    result = np.empty((len(coefficients), 2))
    for index in range(len(coefficients)):
        row = coefficients[index]
        roots = np.roots(np.polyder(row))
        interior = roots.real[
            (np.abs(roots.imag) < 1e-10) & (roots.real > 0) & (roots.real < 1)
        ]
        values = np.polyval(row, np.r_[0.0, interior, 1.0])
        if not np.isfinite(values).all():
            raise ValueError("non-finite polynomial extrema")
        result[index] = [values.min(), values.max()]
    return result


DEFAULT_ANATOMICAL_WEIGHTS: dict[str, float] = {
    "waist": 100.0,
    "hip": 100.0,
    "pelvis": 100.0,
    "back": 80.0,
    "spine": 80.0,
    "torso": 80.0,
    "head": 40.0,
    "shoulder": 20.0,
    "scap": 20.0,
    "arm": 5.0,
    "elbow": 5.0,
    "wrist": 5.0,
    "marker_2": 25.0,
    "marker_3": 25.0,
    "club": 25.0,
    "marker": 1.0,
}


def build_anatomical_marker_weights(
    labels: Sequence[str],
    *,
    custom_weights: Mapping[str, float] | None = None,
    default_weight: float = 1.0,
) -> Array:
    """Compute per-marker tracking weights based on anatomical hierarchy.

    Proximal / trunk markers receive higher weights to prevent the base from
    drifting or spinning to satisfy distal extremity errors.
    """
    weights = np.empty(len(labels), dtype=float)
    overrides = custom_weights or {}
    for i, label in enumerate(labels):
        if label in overrides:
            weights[i] = overrides[label]
            continue
        lower_name = label.lower()
        matched_weight = default_weight
        for pattern, weight in DEFAULT_ANATOMICAL_WEIGHTS.items():
            if pattern in lower_name:
                matched_weight = weight
                break
        weights[i] = matched_weight
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("marker weights must be finite and positive")
    return weights


def bernstein_curvature_regularizer(
    control_count: int,
    *,
    weight: float = 0.1,
    scales: Array | None = None,
) -> Callable[[Array], Array]:
    """Return a regularizer penalizing second differences (curvature) of controls.

    For control points c_0, ..., c_{K-1} per joint, second differences
    Delta^2 c_k = c_{k+2} - 2 c_{k+1} + c_k are directly proportional to
    torque acceleration (d^2 tau / dt^2). High-frequency wiggles and rapid
    fluctuations are penalized. Linear and constant controls produce zero penalty.
    """
    if control_count < 1:
        raise ValueError("control_count must be a positive integer")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and non-negative")
    root_weight = np.sqrt(weight)

    def regularizer(parameters: Array) -> Array:
        if weight == 0 or control_count < 3:
            return np.zeros(0, dtype=float)
        p = np.asarray(parameters, dtype=float)
        if p.size % control_count != 0:
            raise ValueError("parameters length must be a multiple of control_count")
        n_channels = p.size // control_count
        grid = p.reshape(n_channels, control_count)
        curv = grid[:, 2:] - 2.0 * grid[:, 1:-1] + grid[:, :-2]
        if scales is not None:
            s = np.asarray(scales, dtype=float).reshape(n_channels, control_count)[:, 0]
            curv = curv * s[:, None] / 100.0
        return (root_weight * curv).ravel()

    return regularizer


@dataclass(frozen=True)
class MarkerTarget:
    """Weighted numerical view of mapped observations, in metres and seconds.

    NaN XYZ samples are unobserved, never filled. Every positively weighted
    marker needs an observation. Arrays are copied and made read-only.
    This view does not infer labels, units, registration or attachments.
    """

    time: Array
    points: Array
    weights: Array

    def __post_init__(self) -> None:
        time, points, weights = map(_readonly, (self.time, self.points, self.weights))
        if (
            time.ndim != 1
            or len(time) < 2
            or not np.isfinite(time).all()
            or time[0] != 0
            or np.any(np.diff(time) <= 0)
        ):
            raise ValueError("time must be finite, start at zero and strictly increase")
        if points.ndim != 3 or points.shape[0] != len(time) or points.shape[2] != 3:
            raise ValueError("points must have shape (times, markers, 3)")
        if (
            weights.shape != (points.shape[1],)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.any(weights > 0)
        ):
            raise ValueError("weights must match markers, be nonnegative and nonzero")
        if np.isinf(points).any():
            raise ValueError("observations may be NaN but not infinite")
        observed = np.isfinite(points).all(axis=2)
        if np.any((weights > 0) & ~np.any(observed, axis=0)):
            raise ValueError("every weighted marker needs an observation")
        for name, value in (("time", time), ("points", points), ("weights", weights)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PrefixStage:
    """One independently evaluated optimizer candidate; errors are distances."""

    end_s: float
    parameters: Array
    rmse_m: float
    p95_m: float
    max_m: float
    evaluations: int
    optimizer_converged: bool
    message: str
    terminal_rmse_m: float = 0.0
    terminal_max_m: float = 0.0
    pelvis_yaw_diff_deg: float = 0.0
    pelvis_yaw_error_pct: float = 0.0


@dataclass(frozen=True)
class PrefixFit:
    """Numerical fit result, not a claim of qualified Simscape execution."""

    parameters: Array
    stages: tuple[PrefixStage, ...]
    accepted: bool


def _predicted(forward: Forward, parameters: Array, time: Array, shape: tuple) -> Array:
    prediction = np.asarray(
        forward(_readonly(parameters), _readonly(time)), dtype=float
    )
    if prediction.shape != shape or not np.isfinite(prediction).all():
        raise ValueError(
            "forward output must be finite and match requested marker shape"
        )
    return prediction


def _validate_fit_inputs(
    initial: Array,
    lower: Array,
    upper: Array,
    prefix_end_s: Sequence[float],
    target: MarkerTarget,
    acceptance_rmse_m: float,
    max_nfev: int,
) -> tuple[Array, Array, Array, Array]:
    values, lo, hi = (np.asarray(x, dtype=float) for x in (initial, lower, upper))
    if (
        values.ndim != 1
        or values.size == 0
        or lo.shape != values.shape
        or hi.shape != values.shape
        or not np.isfinite([values, lo, hi]).all()
        or np.any(lo >= hi)
        or np.any(values < lo)
        or np.any(values > hi)
    ):
        raise ValueError(
            "parameters need matching finite bounds and a feasible initial"
        )
    ends = np.asarray(prefix_end_s, dtype=float)
    if (
        ends.ndim != 1
        or ends.size == 0
        or not np.isfinite(ends).all()
        or ends[0] <= 0
        or np.any(np.diff(ends) <= 0)
        or ends[-1] != target.time[-1]
    ):
        raise ValueError("prefix schedule must increase and end at the target duration")
    if not np.isfinite(acceptance_rmse_m) or acceptance_rmse_m <= 0:
        raise ValueError("acceptance_rmse_m must be finite and positive")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    return values.copy(), lo, hi, ends


@dataclass(frozen=True)
class PrefixFitOptions:
    """Optional configuration for prefix fitting execution."""

    max_nfev: int = 100
    checkpoint: Callable[[PrefixStage], None] | None = None
    finite_difference_step: float | None = None
    regularization: Callable[[Array], Array] | None = None
    terminal_weight: float = 0.0
    time_weight_power: float = 0.0
    time_weight_scale: float = 0.0
    pelvis_indices: tuple[int, int] | None = None
    pelvis_yaw_weight: float = 0.0
    pelvis_yaw_max_error_pct: float = 5.0
    acceptance_terminal_rmse_m: float | None = None


def _run_prefix_stage(
    target: MarkerTarget,
    forward: Forward,
    end: float,
    values: Array,
    bounds: tuple[Array, Array],
    options: PrefixFitOptions,
) -> tuple[PrefixStage, Array]:
    """Optimize a single growing prefix stage and compute evaluation metrics."""
    lo, hi = bounds
    mask = target.time <= end
    time, measured = target.time[mask], target.points[mask]
    observed = np.isfinite(measured).all(axis=2) & (target.weights > 0)
    if len(time) < 2 or not observed.any():
        raise ValueError("prefix needs at least two times and an observed marker")

    # Time weighting: weight increases towards the horizon
    if options.time_weight_scale > 0 and options.time_weight_power > 0:
        s = np.clip(time / end, 0.0, 1.0)
        time_mult = 1.0 + options.time_weight_scale * (s**options.time_weight_power)
    else:
        time_mult = np.ones(len(time), dtype=float)

    base_weights = time_mult[:, None] * target.weights[None, :]
    root_weights = np.sqrt(base_weights[observed])
    evaluations = 0

    def residual(
        parameters: Array,
        stage_time: Array = time,
        stage_measured: Array = measured,
        stage_observed: NDArray[np.bool_] = observed,
        stage_weights: Array = root_weights,
    ) -> Array:
        nonlocal evaluations
        prediction = _predicted(forward, parameters, stage_time, stage_measured.shape)
        evaluations += 1
        delta = (prediction - stage_measured)[stage_observed]
        marker_residuals = (delta * stage_weights[:, None]).ravel()
        res_list = [marker_residuals]

        # Dedicated terminal-frame penalty
        if options.terminal_weight > 0 and stage_observed[-1].any():
            term_obs = stage_observed[-1]
            term_w = np.sqrt(target.weights[term_obs]) * options.terminal_weight
            term_delta = (prediction[-1] - stage_measured[-1])[term_obs]
            res_list.append((term_delta * term_w[:, None]).ravel())

        # Dedicated pelvis yaw penalty (< 5% error enforcement)
        if options.pelvis_indices is not None and options.pelvis_yaw_weight > 0:
            wl_i, wr_i = options.pelvis_indices
            v_p = prediction[:, wr_i, :2] - prediction[:, wl_i, :2]
            v_t = stage_measured[:, wr_i, :2] - stage_measured[:, wl_i, :2]
            norm_p = np.linalg.norm(v_p, axis=1, keepdims=True) + 1e-9
            norm_t = np.linalg.norm(v_t, axis=1, keepdims=True) + 1e-9
            u_p = v_p / norm_p
            u_t = v_t / norm_t
            sin_diff = u_p[:, 1] * u_t[:, 0] - u_p[:, 0] * u_t[:, 1]
            yaw_weights = options.pelvis_yaw_weight * (
                time_mult if options.time_weight_scale > 0 else 1.0
            )
            res_list.append((sin_diff * yaw_weights).ravel())
            if options.terminal_weight > 0:
                res_list.append(
                    sin_diff[-1:]
                    * (options.pelvis_yaw_weight * options.terminal_weight)
                )

        if options.regularization is not None:
            reg_residuals = np.asarray(
                options.regularization(parameters), dtype=float
            ).ravel()
            if reg_residuals.size > 0:
                if not np.isfinite(reg_residuals).all():
                    raise ValueError("regularization residuals must be finite")
                res_list.append(reg_residuals)

        return np.concatenate(res_list) if len(res_list) > 1 else marker_residuals

    diff_step_arr = (
        np.full_like(values, options.finite_difference_step)
        if options.finite_difference_step is not None
        else None
    )
    optimum = least_squares(
        residual,
        values,
        bounds=(lo, hi),
        max_nfev=options.max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        x_scale="jac",
        diff_step=diff_step_arr,
    )
    new_values = optimum.x.copy()
    prediction = _predicted(forward, new_values, time, measured.shape)
    distances = np.linalg.norm((prediction - measured)[observed], axis=1)

    # Compute terminal frame metrics
    term_obs = observed[-1]
    if term_obs.any():
        term_dists = np.linalg.norm((prediction[-1] - measured[-1])[term_obs], axis=1)
        terminal_rmse_m = float(np.sqrt(np.mean(term_dists**2)))
        terminal_max_m = float(np.max(term_dists))
    else:
        terminal_rmse_m = 0.0
        terminal_max_m = 0.0

    # Compute pelvis yaw metrics (< 5% error requirement)
    pelvis_yaw_diff_deg = 0.0
    pelvis_yaw_error_pct = 0.0
    if options.pelvis_indices is not None:
        wl_i, wr_i = options.pelvis_indices
        v_p_term = prediction[-1, wr_i, :2] - prediction[-1, wl_i, :2]
        v_t_term = measured[-1, wr_i, :2] - measured[-1, wl_i, :2]
        yaw_target = np.degrees(np.arctan2(v_t_term[1], v_t_term[0]))
        yaw_pred = np.degrees(np.arctan2(v_p_term[1], v_p_term[0]))
        diff_deg = float((yaw_pred - yaw_target + 180) % 360 - 180)
        pelvis_yaw_diff_deg = diff_deg
        pelvis_yaw_error_pct = float(abs(diff_deg) / max(abs(yaw_target), 1.0) * 100.0)

    stage = PrefixStage(
        end_s=float(time[-1]),
        parameters=_readonly(new_values),
        rmse_m=float(np.sqrt(np.mean(distances**2))),
        p95_m=float(np.percentile(distances, 95)),
        max_m=float(distances.max()),
        evaluations=evaluations + 1,
        optimizer_converged=bool(optimum.success),
        message=str(optimum.message),
        terminal_rmse_m=terminal_rmse_m,
        terminal_max_m=terminal_max_m,
        pelvis_yaw_diff_deg=pelvis_yaw_diff_deg,
        pelvis_yaw_error_pct=pelvis_yaw_error_pct,
    )
    return stage, new_values


def fit_prefixes(
    target: MarkerTarget,
    forward: Forward,
    *,
    initial: Array,
    lower: Array,
    upper: Array,
    prefix_end_s: Sequence[float],
    acceptance_rmse_m: float,
    options: PrefixFitOptions | None = None,
) -> PrefixFit:
    """Fit growing prefixes using bounded least squares and prior warm starts.

    ``forward(parameters, time)`` must integrate from the same initial state
    and return finite, registered marker positions exactly at requested times.
    Parameters may include constant geometry, subject to caller-supplied bounds.
    All previous observations remain in each objective. Errors from the oracle
    propagate: a failed simulation must not become a finite penalty plateau.
    ``finite_difference_step`` is SciPy's relative parameter perturbation;
    choose parameter scaling and a step resolvable by the native solver.
    None preserves SciPy's default. At zero SciPy uses its default fallback.

    Each stage is re-evaluated outside the optimizer and optionally checkpointed.
    Acceptance requires optimizer convergence and a full-duration distance RMSE
    below the supplied threshold; physical qualification remains the caller's job.
    """
    opt = options if options is not None else PrefixFitOptions()
    if opt.finite_difference_step is not None and (
        isinstance(opt.finite_difference_step, bool)
        or not np.isfinite(opt.finite_difference_step)
        or opt.finite_difference_step <= 0
    ):
        raise ValueError("finite_difference_step must be finite and positive")
    values, lo, hi, ends = _validate_fit_inputs(
        initial,
        lower,
        upper,
        prefix_end_s,
        target,
        acceptance_rmse_m,
        opt.max_nfev,
    )
    stages: list[PrefixStage] = []
    for end in ends:
        stage, values = _run_prefix_stage(target, forward, end, values, (lo, hi), opt)
        stages.append(stage)
        if opt.checkpoint is not None:
            opt.checkpoint(stage)
    final = stages[-1]
    accepted = bool(final.optimizer_converged and final.rmse_m <= acceptance_rmse_m)
    if opt.pelvis_indices is not None:
        accepted = accepted and bool(
            final.pelvis_yaw_error_pct <= opt.pelvis_yaw_max_error_pct
        )
    if opt.acceptance_terminal_rmse_m is not None:
        accepted = accepted and bool(
            final.terminal_rmse_m <= opt.acceptance_terminal_rmse_m
        )
    return PrefixFit(
        _readonly(values),
        tuple(stages),
        accepted,
    )
