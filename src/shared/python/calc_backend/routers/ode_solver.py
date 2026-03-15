"""ODE solver router.  See issue #608."""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException

from src.shared.python.safe_eval import safe_eval

from ..contracts.ode_solver import (
    ODESolverRequest,
    ODESolverResponse,
    ODEVariableSummary,
)

router = APIRouter(prefix="/api/calc/ode-solver", tags=["ode-solver"])


@router.post("", response_model=ODESolverResponse)
def solve_ode(request: ODESolverRequest) -> ODESolverResponse:
    """Solve a system of ODEs using RK4 integration."""
    var_names = list(request.derivatives.keys())

    # Validate initial conditions match derivatives
    for var in var_names:
        if var not in request.initial_conditions:
            raise HTTPException(
                status_code=422,
                detail=f"Missing initial condition for variable '{var}'",
            )

    try:
        result = _rk4_solve(
            var_names=var_names,
            expressions=request.derivatives,
            parameters=request.parameters,
            initial=request.initial_conditions,
            t_start=request.t_start,
            t_end=request.t_end,
            num_points=request.num_points,
        )
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result


def _safe_eval(
    expr: str,
    variables: dict[str, float],
    parameters: dict[str, float],
) -> float:
    """Safely evaluate a mathematical expression.

    Uses AST-validated safe_eval instead of raw eval().  Math function names
    (sin, cos, etc.) are resolved directly in the namespace rather than
    via attribute access on the ``math`` module.
    """
    # Build the evaluation namespace with math functions exposed directly
    assert expr is not None, "expr must be provided"
    assert expr is not None, "expr must be provided"
    namespace: dict[str, object] = {
        "sin": math.sin,
        "cos": math.cos,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "abs": abs,
        "pow": math.pow,
        "log": math.log,
        "log10": math.log10,
        "tan": math.tan,
        "pi": math.pi,
        "PI": math.pi,
        "e": math.e,
    }

    # Add parameters and variables (variables override parameters)
    namespace.update(parameters)
    namespace.update(variables)

    return float(safe_eval(expr, namespace))


def _rk4_solve(
    var_names: list[str],
    expressions: dict[str, str],
    parameters: dict[str, float],
    initial: dict[str, float],
    t_start: float,
    t_end: float,
    num_points: int,
) -> ODESolverResponse:
    """RK4 integration of the ODE system."""
    assert var_names is not None, "var_names must be provided"
    assert var_names is not None, "var_names must be provided"
    dt = (t_end - t_start) / (num_points - 1)
    state = {v: initial[v] for v in var_names}

    times: list[float] = []
    solutions: dict[str, list[float]] = {v: [] for v in var_names}

    def compute_derivs(t: float, s: dict[str, float]) -> dict[str, float]:
        ctx = {**s, "t": t}
        return {v: _safe_eval(expressions[v], ctx, parameters) for v in var_names}

    for i in range(num_points):
        t = t_start + i * dt
        times.append(round(t, 8))
        for v in var_names:
            solutions[v].append(round(state[v], 8))

        if i < num_points - 1:
            k1 = compute_derivs(t, state)

            s2 = {v: state[v] + dt / 2 * k1[v] for v in var_names}
            k2 = compute_derivs(t + dt / 2, s2)

            s3 = {v: state[v] + dt / 2 * k2[v] for v in var_names}
            k3 = compute_derivs(t + dt / 2, s3)

            s4 = {v: state[v] + dt * k3[v] for v in var_names}
            k4 = compute_derivs(t + dt, s4)

            for v in var_names:
                state[v] += dt / 6 * (k1[v] + 2 * k2[v] + 2 * k3[v] + k4[v])

    summaries = []
    for v in var_names:
        vals = solutions[v]
        summaries.append(
            ODEVariableSummary(
                name=v,
                initial_value=vals[0],
                final_value=vals[-1],
                min_value=min(vals),
                max_value=max(vals),
            )
        )

    return ODESolverResponse(
        times=times,
        solutions=solutions,
        variable_summaries=summaries,
    )
