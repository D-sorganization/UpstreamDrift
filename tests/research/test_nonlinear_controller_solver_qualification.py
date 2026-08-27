"""Tests for manufactured nonlinear solver qualification (#9126)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.nonlinear_controller_qualification import (
    ProjectedILQRSolver,
    manufactured_dynamics,
    qualify_solver_kernel,
    validate_qualification,
)

pytestmark = pytest.mark.scientific


def test_manufactured_dynamics_and_derivatives() -> None:
    x = np.array([0.2, -0.4])
    u = np.array([1.0])
    dt = 0.01

    x_next = manufactured_dynamics(x, u, dt)
    assert x_next.shape == (2,)
    assert np.all(np.isfinite(x_next))

    # Test nonfinite handling
    x_nan = np.array([np.nan, 0.0])
    assert np.all(np.isnan(manufactured_dynamics(x_nan, u, dt)))


def test_projected_ilqr_bounds_and_monotonicity() -> None:
    solver = ProjectedILQRSolver(horizon=10, dt=0.01, u_bounds=(-1.0, 1.0), max_iter=15)
    x0 = np.array([0.3, -0.1])
    x_target = np.array([0.0, 0.0])

    u_opt, cost_hist, success = solver.solve(x0, x_target)
    assert success is True
    assert np.all(u_opt >= -1.0001)
    assert np.all(u_opt <= 1.0001)

    # Monotonicity
    for i in range(len(cost_hist) - 1):
        assert cost_hist[i + 1] <= cost_hist[i] + 1e-10


def test_solver_qualification_battery() -> None:
    evidence = validate_qualification()

    assert evidence["status"] == "PASSED"
    assert evidence["derivatives_passed"] is True
    assert evidence["bounds_respected"] is True
    assert evidence["cost_monotonicity_passed"] is True
    assert evidence["deterministic_replay_passed"] is True
    assert evidence["warm_start_benefit_detected"] is True
    assert evidence["typed_nonfinite_failure_passed"] is True


def test_solver_qualification_rejects_unimplemented_solver_identity() -> None:
    with pytest.raises(ValueError, match="only implemented solver"):
        qualify_solver_kernel("bounded_nmpc_collocation")
