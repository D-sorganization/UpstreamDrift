# ruff: noqa: E501
"""Optimization helpers for the advanced plots tab."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, TypedDict, cast

import numpy as np

from .analysis_utils import evaluate_output

__all__ = [
    "OPTIMIZATION_PENALTY_VALUE",
    "OptimizationHistoryEntry",
    "OptimizationResults",
    "find_optimal_on_surface",
    "run_adam_optimization",
]


@dataclass
class OptimizationHistoryEntry:
    """Container tracking the progress of each optimization iteration."""

    iteration: int
    objective: float
    parameters: dict[str, float]


class OptimizationResults(TypedDict):
    """Results from an Adam-based optimization session."""

    best_output: float
    best_parameters: dict[str, float]
    best_state: dict[str, float]
    best_composition: dict[str, float]
    history: list[OptimizationHistoryEntry]
    final_parameters: dict[str, float]
    iterations: int


def _build_override_mapping(
    parameter_names: Sequence[str],
    values: Sequence[float],
) -> dict[str, float]:
    """Create a mapping of parameter names to their associated values."""

    if parameter_names is None:
        raise ValueError("parameter_names must be provided")
    override: dict[str, float] = {}
    for name, value in zip(parameter_names, values, strict=False):
        if name in {"Temperature", "O2/Feed Ratio", "Steam/Feed Ratio", "Pressure"}:
            override[name] = value
    return override


def _compute_gradient_component(
    index: int,
    cfg: dict[str, Any],
    values: np.ndarray,
    objective: float,
    gradient_step: float,
    parameter_names: Sequence[str],
    engine: Any,
    base_params: dict[str, float],
    manual_hhv: float,
    output_name: str,
) -> float:
    """Compute a single component of the finite-difference gradient.

    Selects forward, backward, or central differencing depending on
    whether the current value lies at a parameter bound.
    """
    if index is None:
        raise ValueError("index must be provided")
    lower = float(cfg["min"])
    upper = float(cfg["max"])
    if np.isclose(lower, upper):
        return 0.0

    step = max(gradient_step, (upper - lower) * 1e-3)
    if step <= 0:
        return 0.0

    current = values[index]
    at_lower = np.isclose(current, lower)
    at_upper = np.isclose(current, upper)

    if at_lower and at_upper:
        return 0.0

    def _eval_at(offset: float) -> float:
        perturbed = values.copy()
        perturbed[index] = np.clip(current + offset, lower, upper)
        if np.isclose(perturbed[index], current):
            return float("nan")
        overrides = _build_override_mapping(
            parameter_names,
            perturbed.tolist(),
        )
        val, _, _ = evaluate_output(
            engine,
            base_params,
            manual_hhv,
            output_name,
            overrides,
        )
        return val

    # Forward-only at lower bound
    if at_lower:
        fwd = _eval_at(step)
        if not np.isfinite(fwd):
            return 0.0
        fwd_val = np.clip(current + step, lower, upper)
        return float((fwd - objective) / (fwd_val - current))

    # Backward-only at upper bound
    if at_upper:
        bwd = _eval_at(-step)
        if not np.isfinite(bwd):
            return 0.0
        bwd_val = np.clip(current - step, lower, upper)
        return float((objective - bwd) / (current - bwd_val))

    # Central difference in interior
    plus_val = np.clip(current + step, lower, upper)
    minus_val = np.clip(current - step, lower, upper)
    if np.isclose(plus_val, minus_val):
        return 0.0

    fwd = _eval_at(step)
    bwd = _eval_at(-step)
    if not (np.isfinite(fwd) and np.isfinite(bwd)):
        return 0.0
    return float((fwd - bwd) / (plus_val - minus_val))


@dataclass
class _AdamState:
    """Mutable state for the Adam optimizer loop."""

    parameter_names: list[str]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    values: np.ndarray
    m: np.ndarray  # 1st moment estimate
    v: np.ndarray  # 2nd moment estimate
    best_output: float
    best_parameters: dict[str, float]
    best_state: dict[str, float]
    best_composition: dict[str, float]
    history: list[OptimizationHistoryEntry]
    previous_values: np.ndarray
    base_params: dict[str, float]
    output_name: str


def _init_adam_state(
    analysis_params: dict[str, object],
    parameter_configs: Sequence[dict[str, Any]],
    maximize: bool,
) -> _AdamState:
    """Extract parameters, build bounds, and initialise Adam moment vectors."""
    if analysis_params is None:
        raise ValueError("analysis_params must be provided")
    parameter_names = [cfg["name"] for cfg in parameter_configs]
    lower_bounds = np.array([cfg["min"] for cfg in parameter_configs], dtype=float)
    upper_bounds = np.array([cfg["max"] for cfg in parameter_configs], dtype=float)
    values = np.array([cfg["initial"] for cfg in parameter_configs], dtype=float)
    return _AdamState(
        parameter_names=parameter_names,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        values=values,
        m=np.zeros_like(values),
        v=np.zeros_like(values),
        best_output=-np.inf if maximize else np.inf,
        best_parameters={},
        best_state={},
        best_composition={},
        history=[],
        previous_values=values.copy(),
        base_params=cast(dict[str, float], analysis_params["base_params"]),
        output_name=cast(str, analysis_params["output_variable"]),
    )


def _evaluate_and_record(
    st: _AdamState,
    iteration: int,
    engine: Any,
    manual_hhv: float,
    maximize: bool,
) -> float:
    """Evaluate the objective, update best tracking, and record history.

    Returns the (possibly clamped) objective value.
    """
    if st is None:
        raise ValueError("st must be provided")
    overrides = _build_override_mapping(st.parameter_names, st.values.tolist())
    objective, composition, state = evaluate_output(
        engine, st.base_params, manual_hhv, st.output_name, overrides
    )

    if not np.isfinite(objective):
        objective = -np.inf if maximize else np.inf
        composition, state = {}, {}

    if np.isfinite(objective) and (
        (maximize and objective > st.best_output)
        or (not maximize and objective < st.best_output)
    ):
        st.best_output = objective
        st.best_parameters = overrides.copy()
        st.best_state = state
        st.best_composition = composition

    st.history.append(
        OptimizationHistoryEntry(
            iteration=iteration,
            objective=objective,
            parameters=overrides.copy(),
        ),
    )
    return objective


def _adam_update(
    st: _AdamState,
    gradient: np.ndarray,
    iteration: int,
    *,
    maximize: bool,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> None:
    """Apply one Adam parameter update in-place."""
    if st is None:
        raise ValueError("st must be provided")
    st.m = beta1 * st.m + (1 - beta1) * gradient
    st.v = beta2 * st.v + (1 - beta2) * (gradient**2)
    m_hat = st.m / (1 - beta1**iteration)
    v_hat = st.v / (1 - beta2**iteration)

    sign = 1.0 if maximize else -1.0
    update = sign * learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    st.values = np.clip(st.values + update, st.lower_bounds, st.upper_bounds)


def run_adam_optimization(
    engine: Any,
    analysis_params: dict[str, object],
    manual_hhv: float,
    parameter_configs: Sequence[dict[str, Any]],
    *,
    maximize: bool,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    gradient_step: float,
    max_iterations: int,
    tolerance: float,
    gradient_tolerance: float | None = None,
) -> OptimizationResults:
    """Run an Adam-based search across operating parameters.

    Parameters
    ----------
    engine:
        Calculation engine.
    analysis_params:
        Dict with ``"base_params"`` and ``"output_variable"``.
    manual_hhv:
        User-specified HHV [Btu/lb].
    parameter_configs:
        Sequence of dicts with ``"name"``, ``"min"``, ``"max"``,
        ``"initial"``.
    maximize:
        ``True`` to maximize, ``False`` to minimize.
    learning_rate, beta1, beta2, epsilon:
        Adam hyperparameters (Kingma & Ba, 2014).
    gradient_step:
        Finite-difference step size.
    max_iterations:
        Maximum iterations.
    tolerance:
        Convergence tolerance on parameter updates.
    gradient_tolerance:
        Convergence tolerance on gradient norm (defaults to
        *tolerance*).

    Returns
    -------
    OptimizationResults
        Best objective, parameters, state, composition, history,
        final parameters, and iteration count.
    """
    if not parameter_configs:
        raise ValueError("At least one parameter must be provided")

    st = _init_adam_state(analysis_params, parameter_configs, maximize)

    if gradient_tolerance is None:
        gradient_tolerance = tolerance

    for iteration in range(1, max_iterations + 1):
        objective = _evaluate_and_record(st, iteration, engine, manual_hhv, maximize)

        # Finite-difference gradient
        gradient = np.zeros_like(st.values)
        if np.isfinite(objective):
            for idx, cfg in enumerate(parameter_configs):
                gradient[idx] = _compute_gradient_component(
                    idx,
                    cfg,
                    st.values,
                    objective,
                    gradient_step,
                    st.parameter_names,
                    engine,
                    st.base_params,
                    manual_hhv,
                    st.output_name,
                )

        if np.linalg.norm(gradient) < gradient_tolerance:
            break

        _adam_update(
            st,
            gradient,
            iteration,
            maximize=maximize,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )

        if np.linalg.norm(st.values - st.previous_values) < tolerance:
            break
        st.previous_values = st.values.copy()

    return {
        "best_output": st.best_output,
        "best_parameters": st.best_parameters,
        "best_state": st.best_state,
        "best_composition": st.best_composition,
        "history": st.history,
        "final_parameters": _build_override_mapping(
            st.parameter_names,
            st.values.tolist(),
        ),
        "iterations": len(st.history),
    }


# =============================================================================
# SURFACE OPTIMIZATION FUNCTIONS (Interpolation-based)
# =============================================================================

OPTIMIZATION_PENALTY_VALUE: Final[float] = 1e10


def find_optimal_on_surface(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    method: str = "Grid Search",
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    callback: Any | None = None,
) -> dict[str, Any]:
    """
    Find the optimal (maximum) point on a surface defined by grid data.

    Args:
        x_grid: 1D or 2D array of X coordinates.
        y_grid: 1D or 2D array of Y coordinates.
        z_grid: 2D array of Z values (shape matching grid).
        method: Optimization method ("Grid Search", "L-BFGS-B", "Differential Evolution").
        bounds: Optional ((min_x, max_x), (min_y, max_y)). Calculated from grid if None.
        callback: Optional callback(evaluations, total) for progress tracking.

    Returns:
        Dictionary containing 'optimal_x', 'optimal_y', 'optimal_z', etc.
    """
    # Prepare data for interpolation
    # Ensure 1D unique sorted arrays for RegularGridInterpolator
    if x_grid is None:
        raise ValueError("x_grid must be provided")
    x_vals = np.unique(x_grid) if x_grid.ndim > 1 else x_grid

    y_vals = np.unique(y_grid) if y_grid.ndim > 1 else y_grid

    # z_grid should be (x, y) or (y, x) depending on meshgrid convention
    # Usually meshgrid(x, y) produces shape (len(y), len(x)).
    # RegularGridInterpolator expects (len(x), len(y)) if x is first dim.
    # The AdvancedPlotsTab passed Z.T, implying Z was (y, x).
    # We will try to infer or assume Z corresponds to shape (len(y_vals), len(x_vals)).

    # If Z shape is (len(y), len(x)), we transpose for Interp((x, y), Z.T)
    from scipy.interpolate import RegularGridInterpolator  # lazy import

    if z_grid.shape == (len(y_vals), len(x_vals)):
        interpolator = RegularGridInterpolator(
            (x_vals, y_vals),
            z_grid.T,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
    elif z_grid.shape == (len(x_vals), len(y_vals)):
        interpolator = RegularGridInterpolator(
            (x_vals, y_vals),
            z_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
    else:
        raise ValueError(f"Z grid shape {z_grid.shape} does not match X/Y dimensions.")

    if bounds is None:
        bounds = (
            (float(x_vals.min()), float(x_vals.max())),
            (float(y_vals.min()), float(y_vals.max())),
        )

    # Evaluation wrapper
    def objective(p: Any) -> float:  # Maximization -> Negative for minimization
        try:
            val = interpolator(p)
            # Handle potential 1-element array return
            if np.ndim(val) > 0:
                val = val.item()

            if np.isnan(val):
                return OPTIMIZATION_PENALTY_VALUE
            return -float(val)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return OPTIMIZATION_PENALTY_VALUE

    if method == "Grid Search":
        return _run_grid_search(interpolator, bounds, callback)
    if method == "L-BFGS-B":
        return _run_lbfgsb(objective, bounds, callback)
    if method == "Differential Evolution":
        return _run_differential_evolution(objective, bounds, callback)
    raise ValueError(f"Unknown optimization method: {method}")


def _run_grid_search(interpolator: Any, bounds: Any, callback: Any) -> dict:
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # Use 15x15 grid (225 points)
    n_x, n_y = 15, 15
    xs = np.linspace(x_min, x_max, n_x)
    ys = np.linspace(y_min, y_max, n_y)

    best_x, best_y, best_z = None, None, float("-inf")
    total_points = n_x * n_y
    count = 0

    for x in xs:
        for y in ys:
            count += 1
            if callback and count % 10 == 0:
                callback(count, total_points)

            try:
                val = interpolator([x, y])
                if np.ndim(val) > 0:
                    val = val.item()
                z = float(val)

                if not np.isnan(z) and z > best_z:
                    best_z = z
                    best_x = x
                    best_y = y
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                pass

    return {
        "success": best_x is not None,
        "optimal_x": best_x,
        "optimal_y": best_y,
        "optimal_z": best_z,
        "evaluations": count,
    }


def _run_lbfgsb(objective: Any, bounds: Any, callback: Any) -> dict:
    x_mean = (bounds[0][0] + bounds[0][1]) / 2
    y_mean = (bounds[1][0] + bounds[1][1]) / 2
    x0 = [x_mean, y_mean]

    eval_count = [0]

    def tracked_obj(p: Any) -> float:
        eval_count[0] += 1
        if callback and eval_count[0] % 5 == 0:
            callback(eval_count[0], 100)  # Unknown total
        return float(objective(p))

    from scipy.optimize import minimize  # lazy import

    res = minimize(
        tracked_obj,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 100, "ftol": 1e-6},
    )

    return {
        "success": res.success,
        "optimal_x": res.x[0],
        "optimal_y": res.x[1],
        "optimal_z": -res.fun,
        "evaluations": eval_count[0],
        "message": res.message,
    }


def _run_differential_evolution(objective: Any, bounds: Any, callback: Any) -> dict:
    eval_count = [0]

    def tracked_obj(p: Any) -> float:
        eval_count[0] += 1
        if callback and eval_count[0] % 10 == 0:
            callback(eval_count[0], 100)
        return float(objective(p))

    from scipy.optimize import differential_evolution  # lazy import

    res = differential_evolution(
        tracked_obj, bounds, maxiter=50, popsize=10, atol=0.01, tol=0.01
    )

    return {
        "success": res.success,
        "optimal_x": res.x[0],
        "optimal_y": res.x[1],
        "optimal_z": -res.fun,
        "evaluations": eval_count[0],
        "message": res.message,
    }
