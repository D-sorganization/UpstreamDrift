"""Qualification gates for prospective nonlinear-controller solver kernels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.nonlinear_controller_qualification import (
    PROJECTED_ILQR_SOLVER_NAME,
    REPORT_PATH,
    BoxBounds,
    QuadraticTrackingCost,
    build_qualification,
    central_dynamics_jacobians,
    manufactured_step,
    qualify_solver_kernel,
    solve_projected_ilqr,
    validate_qualification,
)

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.scientific


def _problem() -> tuple[np.ndarray, QuadraticTrackingCost, BoxBounds]:
    return (
        np.array([0.55, -0.10]),
        QuadraticTrackingCost(
            state_weight=np.diag([4.0, 0.4]),
            control_weight=np.diag([0.08]),
            terminal_weight=np.diag([18.0, 2.0]),
            reference_state=np.zeros(2),
            reference_control=np.zeros(1),
        ),
        BoxBounds(lower=np.array([-0.40]), upper=np.array([0.40])),
    )


def test_manufactured_derivatives_match_independent_direction() -> None:
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
    predicted = state_map @ state_direction + control_map @ control_direction
    step = 2.0e-6
    observed = (
        manufactured_step(
            state + step * state_direction, control + step * control_direction
        )
        - manufactured_step(
            state - step * state_direction, control - step * control_direction
        )
    ) / (2.0 * step)
    assert np.max(np.abs(predicted - observed)) < 1.0e-8


def test_solver_enforces_bounds_and_accepts_only_descent() -> None:
    initial, cost, bounds = _problem()
    result = solve_projected_ilqr(
        manufactured_step,
        initial,
        horizon=24,
        cost=cost,
        bounds=bounds,
        initial_controls=np.zeros((24, 1)),
    )
    assert result.success is True
    assert result.controls is not None
    assert np.all(result.controls >= bounds.lower - 1.0e-12)
    assert np.all(result.controls <= bounds.upper + 1.0e-12)
    assert len(result.accepted_costs) >= 2
    assert np.all(np.diff(result.accepted_costs) <= 1.0e-12)
    assert result.accepted_costs[-1] < result.accepted_costs[0]


def test_nonfinite_rollout_is_typed_without_fabricated_trajectory() -> None:
    initial, cost, bounds = _problem()

    def invalid_dynamics(state: np.ndarray, control: np.ndarray) -> np.ndarray:
        del state, control
        return np.array([np.nan, 0.0])

    result = solve_projected_ilqr(
        invalid_dynamics,
        initial,
        horizon=4,
        cost=cost,
        bounds=bounds,
        initial_controls=np.zeros((4, 1)),
    )
    assert result.success is False
    assert result.status == "dynamics_failure"
    assert result.states is None
    assert result.controls is None


def test_committed_qualification_is_deterministic_and_scope_limited() -> None:
    expected = build_qualification(ROOT)
    committed = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert committed == expected
    assert validate_qualification(committed, ROOT) == {
        "solver_count": 1,
        "qualified_solver_count": 1,
        "double_pendulum_evaluation_count": 0,
        "ranking_eligible_count": 0,
    }
    assert committed["registration_authority"]["sha256"] == next(
        item["sha256"]
        for item in committed["source_authorities"]
        if item["path"].endswith("nonlinear_controller_comparison_registration.json")
    )


def test_unimplemented_collocation_identity_fails_closed() -> None:
    with pytest.raises(ValueError, match="only implemented solver"):
        qualify_solver_kernel("bounded_nmpc_collocation")
    assert qualify_solver_kernel()["name"] == PROJECTED_ILQR_SOLVER_NAME
