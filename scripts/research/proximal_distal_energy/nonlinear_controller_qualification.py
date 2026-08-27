"""Manufactured nonlinear solver qualification (#9126).

Qualifies candidate nonlinear MPC and projected iLQR solver kernels on a
manufactured mathematical fixture before plant transport.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from typing import Any
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

INFERENCE_BOUNDARY = (
    "This qualification establishes solver convergence, bound adherence, "
    "monotonicity, and replay mechanics on a manufactured mathematical benchmark. "
    "It does not evaluate golf swing performance, human intent, or controller ranking."
)


def manufactured_dynamics(x: FloatArray, u: FloatArray, dt: float = 0.01) -> FloatArray:
    """Nonlinear manufactured 2D benchmark dynamics with trigonometric and cubic coupling."""
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)):
        return np.full_like(x, np.nan)
    x0, x1 = x[0], x[1]
    u0 = u[0]
    next_x0 = x0 + dt * math.sin(x1)
    next_x1 = x1 + dt * (u0 - 0.5 * (x1**3))
    return np.array([next_x0, next_x1], dtype=np.float64)


def manufactured_cost(x: FloatArray, u: FloatArray, x_target: FloatArray) -> float:
    """Quadratic running cost."""
    dx = x - x_target
    return float(np.dot(dx, dx) + 0.1 * np.dot(u, u))


@dataclass(frozen=True, slots=True)
class SolverQualificationResult:
    """Results from testing candidate solver mechanics on manufactured fixture."""

    solver_name: str
    derivatives_passed: bool
    bounds_respected: bool
    cost_monotonicity_passed: bool
    deterministic_replay_passed: bool
    warm_start_benefit_detected: bool
    typed_nonfinite_failure_passed: bool
    iterations_run: int
    final_cost: float


class ProjectedILQRSolver:
    """Projected first-order iLQR solver with box control constraints."""

    def __init__(
        self,
        horizon: int = 20,
        dt: float = 0.01,
        u_bounds: tuple[float, float] = (-2.0, 2.0),
        max_iter: int = 25,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.u_min, self.u_max = u_bounds
        self.max_iter = max_iter

    def solve(
        self,
        x0: FloatArray,
        x_target: FloatArray,
        warm_start_u: FloatArray | None = None,
    ) -> tuple[FloatArray, list[float], bool]:
        """Run projected gradient / iLQR iterations."""
        H = self.horizon
        dt = self.dt
        u = (
            warm_start_u.copy()
            if warm_start_u is not None
            else np.zeros((H, 1), dtype=np.float64)
        )
        u = np.clip(u, self.u_min, self.u_max)

        def rollout(controls: FloatArray) -> tuple[FloatArray, float, bool]:
            x_traj = [x0.copy()]
            total_cost = 0.0
            x_cur = x0.copy()
            for k in range(H):
                total_cost += manufactured_cost(x_cur, controls[k], x_target)
                x_next = manufactured_dynamics(x_cur, controls[k], dt)
                if not np.all(np.isfinite(x_next)):
                    return np.array(x_traj), float("inf"), False
                x_cur = x_next
                x_traj.append(x_cur.copy())
            total_cost += float(np.dot(x_cur - x_target, x_cur - x_target) * 5.0)
            return np.array(x_traj), total_cost, True

        _, current_cost, valid = rollout(u)
        if not valid or not math.isfinite(current_cost):
            return u, [float("inf")], False

        cost_history = [current_cost]
        learning_rate = 0.1

        for _ in range(self.max_iter):
            # Compute numerical gradient with respect to u
            grad = np.zeros_like(u)
            eps = 1e-5
            for k in range(H):
                u_pert = u.copy()
                u_pert[k, 0] += eps
                u_pert = np.clip(u_pert, self.u_min, self.u_max)
                _, cost_p, val_p = rollout(u_pert)
                if val_p:
                    grad[k, 0] = (cost_p - current_cost) / eps

            # Line search with projection
            alpha = learning_rate
            accepted = False
            best_u = u.copy()
            best_cost = current_cost

            for _ in range(8):
                u_step = np.clip(u - alpha * grad, self.u_min, self.u_max)
                _, step_cost, val_step = rollout(u_step)
                if val_step and step_cost <= current_cost + 1e-12:
                    accepted = True
                    best_u = u_step
                    best_cost = step_cost
                    break
                alpha *= 0.5

            if accepted:
                u = best_u
                current_cost = best_cost
                cost_history.append(current_cost)
            else:
                cost_history.append(current_cost)
                break

        return u, cost_history, True


def qualify_solver_kernel(
    solver_name: str = "bounded_projected_ilqr",
) -> SolverQualificationResult:
    """Run manufactured qualification battery on candidate solver."""
    solver = ProjectedILQRSolver(horizon=15, dt=0.01, u_bounds=(-1.5, 1.5), max_iter=20)
    x0 = np.array([0.5, -0.2], dtype=np.float64)
    x_target = np.array([0.0, 0.0], dtype=np.float64)

    # 1. Monotonicity and bounds check (cold start)
    u_cold, cost_hist_cold, success_cold = solver.solve(x0, x_target)
    bounds_ok = bool(np.all(u_cold >= -1.50001) and np.all(u_cold <= 1.50001))

    monotonic_ok = True
    for i in range(len(cost_hist_cold) - 1):
        if cost_hist_cold[i + 1] > cost_hist_cold[i] + 1e-10:
            monotonic_ok = False
            break

    # 2. Deterministic replay check
    u_replay, cost_hist_replay, _ = solver.solve(x0, x_target)
    replay_ok = bool(
        np.allclose(u_cold, u_replay, atol=1e-14) and cost_hist_cold == cost_hist_replay
    )

    # 3. Warm start sensitivity check
    u_warm, cost_hist_warm, _ = solver.solve(x0, x_target, warm_start_u=u_cold)
    warm_start_ok = bool(cost_hist_warm[0] <= cost_hist_cold[0] + 1e-10)

    # 4. Typed nonfinite-dynamics failure check
    x_nan = np.array([np.nan, 0.0], dtype=np.float64)
    _, cost_nan, success_nan = solver.solve(x_nan, x_target)
    nonfinite_handled_ok = not success_nan or math.isinf(cost_nan[0])

    # 5. Derivatives check on manufactured dynamics
    eps = 1e-6
    x_test = np.array([0.3, 0.4])
    u_test = np.array([0.5])
    f_nominal = manufactured_dynamics(x_test, u_test, dt=0.01)
    df_dx0_num = (
        manufactured_dynamics(x_test + np.array([eps, 0.0]), u_test, 0.01)[0]
        - f_nominal[0]
    ) / eps
    derivatives_ok = bool(abs(df_dx0_num - 1.0) < 1e-4)

    return SolverQualificationResult(
        solver_name=solver_name,
        derivatives_passed=derivatives_ok,
        bounds_respected=bounds_ok,
        cost_monotonicity_passed=monotonic_ok,
        deterministic_replay_passed=replay_ok,
        warm_start_benefit_detected=warm_start_ok,
        typed_nonfinite_failure_passed=nonfinite_handled_ok,
        iterations_run=int(len(cost_hist_cold)),
        final_cost=float(cost_hist_cold[-1]),
    )


def validate_qualification() -> dict[str, Any]:
    res = qualify_solver_kernel()
    all_passed = (
        res.derivatives_passed
        and res.bounds_respected
        and res.cost_monotonicity_passed
        and res.deterministic_replay_passed
        and res.warm_start_benefit_detected
        and res.typed_nonfinite_failure_passed
    )
    data = asdict(res)
    data["status"] = "PASSED" if all_passed else "FAILED"
    data["inference_boundary"] = INFERENCE_BOUNDARY
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate #9126 solver qualification.")
    parser.add_argument(
        "action", nargs="?", default="validate", choices=["validate", "report"]
    )
    args = parser.parse_args()

    evidence = validate_qualification()
    print(json.dumps(evidence, indent=2))

    if evidence["status"] != "PASSED":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
