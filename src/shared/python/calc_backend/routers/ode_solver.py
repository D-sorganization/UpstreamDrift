"""ODE solver router.  See issue #608."""

from __future__ import annotations  # noqa: E402, F404

import math  # noqa: E402

from fastapi import APIRouter, HTTPException  # noqa: E402

from src.shared.python.safe_eval import safe_eval  # noqa: E402

from ..contracts.ode_solver import (  # noqa: E402
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
    if expr is None:
        raise ValueError("expr must be provided")
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
    """Integration of the ODE system using unified ODESolver."""
    if var_names is None:
        raise ValueError("var_names must be provided")
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if t_end <= t_start:
        raise ValueError(
            f"t_end ({t_end}) must be strictly greater than t_start ({t_start})"
        )

    import numpy as np

    from shared.python.sidekick.process_calculators.ode_solver import ODESolver

    solver = ODESolver(derivatives=expressions, parameters=parameters)
    y0 = [initial[v] for v in var_names]
    t_eval = np.linspace(t_start, t_end, num_points)

    sol = solver.solve((t_start, t_end), y0, t_eval=t_eval)
    if not sol.success:
        raise ValueError(f"ODE solver failed: {sol.message}")
    if not np.all(np.isfinite(sol.y)):
        raise ValueError(
            "ODE solution diverged with non-finite values; reduce the time span "
            "or check the derivative system"
        )

    times = [round(float(t), 8) for t in sol.t]
    solutions = {
        v: [round(float(val), 8) for val in sol.y[i]] for i, v in enumerate(var_names)
    }

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
