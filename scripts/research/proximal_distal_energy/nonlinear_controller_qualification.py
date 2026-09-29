"""Qualify prospective nonlinear-control kernels on a manufactured fixture.

The held-out double-pendulum grid is deliberately not imported or executed.
Passing this module qualifies solver mechanics only, not golf performance.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .nonlinear_controller_kernels import solve_projected_ilqr
from .nonlinear_controller_numerics import (
    BoxBounds,
    FloatArray,
    QuadraticTrackingCost,
    SolverResult,
    central_dynamics_jacobians,
    manufactured_step,
    monotonic,
)

Solver = Callable[..., SolverResult]
ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REPORT_PATH = ARTICLE / "data/nonlinear_controller_solver_qualification.json"
REGISTRATION_PATH = ARTICLE / "data/nonlinear_controller_comparison_registration.json"
ENVIRONMENT_LOCK_PATH = ROOT / "requirements.lock"
SCHEMA_VERSION = "proximal-distal-nonlinear-controller-qualification/v1"
SOURCE_RELATIVE_PATHS = (
    REGISTRATION_PATH.relative_to(ROOT),
    Path("scripts/research/proximal_distal_energy/nonlinear_controller_numerics.py"),
    Path("scripts/research/proximal_distal_energy/nonlinear_controller_kernels.py"),
)
DERIVATIVE_ERROR_LIMIT = 1.0e-8
BOUND_VIOLATION_LIMIT = 1.0e-12
COLD_WARM_COST_LIMIT = 0.02
ACCEPTED_COST_INCREASE_TOLERANCE = 1.0e-10
PROJECTED_ILQR_SOLVER_NAME = "bounded_projected_first_order_ilqr_kernel"
UNAVAILABLE_SOLVERS = (
    "bounded_nmpc_collocation",
    "second_order_ddp",
    "risk_sensitive_control",
    "scenario_stochastic_mpc",
)


def build_qualification(root: Path = ROOT) -> dict[str, object]:
    """Build deterministic manufactured-fixture solver evidence."""
    root = root.resolve()
    registration = root / REGISTRATION_PATH.relative_to(ROOT)
    environment_lock = root / ENVIRONMENT_LOCK_PATH.relative_to(ROOT)
    initial = np.array([0.55, -0.10])
    horizon = 24
    cost = _fixture_cost()
    bounds = BoxBounds(np.array([-0.40]), np.array([0.40]))
    solvers = [
        _qualify_solver(
            PROJECTED_ILQR_SOLVER_NAME,
            solve_projected_ilqr,
            initial,
            horizon,
            cost,
            bounds,
        )
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "manufactured_nonlinear_solver_qualification",
        "evidence_status": "solver_mechanics_only_no_golf_evaluation",
        "registration_authority": {
            "path": registration.relative_to(root).as_posix(),
            "sha256": _sha256(registration),
        },
        "source_authorities": [
            _authority(root, path) for path in SOURCE_RELATIVE_PATHS
        ],
        "environment_authority": {
            "path": environment_lock.relative_to(root).as_posix(),
            "sha256": _sha256(environment_lock),
            "supported_python_lanes": ["3.11", "3.12"],
            "replay_rule": (
                "exact equality is required in the locked supported environment"
            ),
        },
        "fixture": _fixture_record(initial, horizon, bounds),
        "qualification_thresholds": {
            "directional_derivative_max_abs_error": DERIVATIVE_ERROR_LIMIT,
            "maximum_bound_violation": BOUND_VIOLATION_LIMIT,
            "cold_warm_relative_cost_difference": COLD_WARM_COST_LIMIT,
            "accepted_cost_increase_tolerance": (ACCEPTED_COST_INCREASE_TOLERANCE),
        },
        "directional_derivative_max_abs_error": _directional_error(),
        "solvers": solvers,
        "unavailable_solvers": list(UNAVAILABLE_SOLVERS),
        "qualified_solver_count": len(solvers),
        "double_pendulum_evaluation_count": 0,
        "ranking_eligible_method_count": 0,
        "ranking_rule": (
            "no solver is ranking-eligible before frozen held-out execution"
        ),
        "inference_boundary": (
            "Manufactured-fixture qualification checks solver mechanics only. It "
            "does not qualify the double-pendulum comparison, establish golf "
            "performance, identify human control, or support coaching guidance."
        ),
    }


def validate_qualification(
    report: dict[str, object], root: Path = ROOT
) -> dict[str, int]:
    """Fail closed on replay, bounds, descent, sensitivity, or scope drift."""
    if report != build_qualification(root):
        raise ValueError("qualification differs from deterministic authority")
    solvers = report.get("solvers")
    if not isinstance(solvers, list) or len(solvers) != 1:
        raise ValueError("exactly one implemented solver kernel must be qualified")
    if int(report.get("qualified_solver_count", -1)) != len(solvers):
        raise ValueError("qualified solver count drifted")
    derivative_error = float(
        report.get("directional_derivative_max_abs_error", math.inf)
    )
    if derivative_error > DERIVATIVE_ERROR_LIMIT:
        raise ValueError("manufactured derivative agreement gate failed")
    for solver in solvers:
        if not isinstance(solver, dict) or not _solver_gates_pass(solver):
            name = solver.get("name") if isinstance(solver, dict) else "unknown"
            raise ValueError(f"{name}: qualification gate failed")
    evaluations = int(report.get("double_pendulum_evaluation_count", -1))
    eligible = sum(solver.get("eligible_for_ranking") is True for solver in solvers)
    if evaluations != 0 or eligible != 0:
        raise ValueError("manufactured qualification cannot rank golf controllers")
    if int(report.get("ranking_eligible_method_count", -1)) != eligible:
        raise ValueError("ranking-eligible method count drifted")
    return {
        "solver_count": len(solvers),
        "qualified_solver_count": len(solvers),
        "double_pendulum_evaluation_count": evaluations,
        "ranking_eligible_count": eligible,
    }


def qualify_solver_kernel(
    solver_name: str = PROJECTED_ILQR_SOLVER_NAME,
) -> dict[str, object]:
    """Qualify the sole implemented kernel and reject unavailable identities."""

    if solver_name != PROJECTED_ILQR_SOLVER_NAME:
        raise ValueError(
            f"unsupported solver {solver_name!r}; the only implemented solver is "
            f"{PROJECTED_ILQR_SOLVER_NAME!r}"
        )
    initial = np.array([0.55, -0.10])
    return _qualify_solver(
        solver_name,
        solve_projected_ilqr,
        initial,
        24,
        _fixture_cost(),
        BoxBounds(np.array([-0.40]), np.array([0.40])),
    )


def _qualify_solver(
    name: str,
    solver: Solver,
    initial: FloatArray,
    horizon: int,
    cost: QuadraticTrackingCost,
    bounds: BoxBounds,
) -> dict[str, object]:
    zero = np.zeros((horizon, 1))
    cold = _solve(solver, initial, horizon, cost, bounds, zero)
    replay = _solve(solver, initial, horizon, cost, bounds, zero)
    if not cold.success or cold.controls is None:
        raise ValueError(f"{name}: cold qualification failed")
    warm_seed = np.vstack((np.zeros((1, 1)), cold.controls[:-1]))
    warm = _solve(solver, initial, horizon, cost, bounds, warm_seed)
    return _solver_record(name, cold, replay, warm, bounds)


def _solve(
    solver: Solver,
    initial: FloatArray,
    horizon: int,
    cost: QuadraticTrackingCost,
    bounds: BoxBounds,
    seed: FloatArray,
) -> SolverResult:
    return solver(
        manufactured_step,
        initial,
        horizon=horizon,
        cost=cost,
        bounds=bounds,
        initial_controls=seed,
    )


def _solver_record(
    name: str,
    cold: SolverResult,
    replay: SolverResult,
    warm: SolverResult,
    bounds: BoxBounds,
) -> dict[str, object]:
    if cold.controls is None or replay.controls is None or warm.controls is None:
        raise ValueError(f"{name}: missing successful controls")
    if cold.cost is None or warm.cost is None:
        raise ValueError(f"{name}: missing successful cost")
    scale = max(abs(cold.cost), 1.0)
    sensitivity = abs(warm.cost - cold.cost) / scale
    violation = max(
        float(np.max(bounds.lower - cold.controls)),
        float(np.max(cold.controls - bounds.upper)),
        0.0,
    )
    return {
        "name": name,
        "status": "manufactured_fixture_qualified",
        "eligible_for_ranking": False,
        "cold_cost": cold.cost,
        "warm_cost": warm.cost,
        "cold_warm_relative_cost_difference": sensitivity,
        "cold_warm_sensitivity_passed": bool(
            warm.success and sensitivity <= COLD_WARM_COST_LIMIT
        ),
        "accepted_costs": list(cold.accepted_costs),
        "accepted_costs_monotonic": monotonic(cold.accepted_costs),
        "maximum_bound_violation": violation,
        "deterministic_replay_passed": _replay_matches(cold, replay),
        "control_sha256": hashlib.sha256(cold.controls.tobytes()).hexdigest(),
        "remaining_gate": (
            "matched_double_pendulum_typed_outcomes_replay_and_held_out_execution"
        ),
    }


def _replay_matches(cold: SolverResult, replay: SolverResult) -> bool:
    return bool(
        replay.success
        and np.array_equal(cold.controls, replay.controls)
        and np.array_equal(cold.states, replay.states)
        and cold.cost == replay.cost
    )


def _solver_gates_pass(solver: dict[str, object]) -> bool:
    gates = (
        solver.get("status") == "manufactured_fixture_qualified",
        solver.get("deterministic_replay_passed") is True,
        solver.get("cold_warm_sensitivity_passed") is True,
        solver.get("accepted_costs_monotonic") is True,
        float(solver.get("maximum_bound_violation", math.inf)) <= BOUND_VIOLATION_LIMIT,
    )
    return all(gates)


def _fixture_record(
    initial: FloatArray, horizon: int, bounds: BoxBounds
) -> dict[str, object]:
    return {
        "name": "damped_nonlinear_pendulum_discrete_fixture",
        "state": ["angle_rad", "rate_rad_s"],
        "control": ["torque_like_input"],
        "step_s": 0.04,
        "horizon_steps": horizon,
        "initial_state": initial.tolist(),
        "control_lower": bounds.lower.tolist(),
        "control_upper": bounds.upper.tolist(),
    }


def _directional_error() -> float:
    state = np.array([0.31, -0.22])
    control = np.array([0.17])
    state_map, control_map = central_dynamics_jacobians(
        manufactured_step,
        state,
        control,
        state_steps=np.array([1.0e-5, 2.0e-5]),
        control_steps=np.array([1.0e-5]),
    )
    state_direction = np.array([0.6, -0.8])
    control_direction = np.array([0.35])
    step = 2.0e-6
    observed = (
        manufactured_step(
            state + step * state_direction, control + step * control_direction
        )
        - manufactured_step(
            state - step * state_direction, control - step * control_direction
        )
    ) / (2.0 * step)
    predicted = state_map @ state_direction + control_map @ control_direction
    return float(np.max(np.abs(predicted - observed)))


def _fixture_cost() -> QuadraticTrackingCost:
    return QuadraticTrackingCost(
        np.diag([4.0, 0.4]),
        np.diag([0.08]),
        np.diag([18.0, 2.0]),
        np.zeros(2),
        np.zeros(1),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority(root: Path, relative_path: Path) -> dict[str, str]:
    return {
        "path": relative_path.as_posix(),
        "sha256": _sha256(root / relative_path),
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("write", "validate"), nargs="?", default="validate"
    )
    args = parser.parse_args()
    if args.command == "write":
        report = build_qualification(ROOT)
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(validate_qualification(report, ROOT), indent=2))


__all__ = [
    "REPORT_PATH",
    "BoxBounds",
    "QuadraticTrackingCost",
    "build_qualification",
    "central_dynamics_jacobians",
    "manufactured_step",
    "PROJECTED_ILQR_SOLVER_NAME",
    "qualify_solver_kernel",
    "solve_projected_ilqr",
    "validate_qualification",
]


if __name__ == "__main__":
    main()
