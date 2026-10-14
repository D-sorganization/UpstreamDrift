"""Tests for contact wrenches in the inner solve (SOCP friction cone, measured GRF)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.contact_wrenches import (
    ContactIdentifiabilityDiagnostics,
    ContactWrenchProblem,
    ContactWrenchSolution,
    compute_cone_violations,
    compute_identifiability_diagnostics,
    project_friction_cone,
    solve_contact_wrenches,
)

pytestmark = pytest.mark.unit


def _synthetic_contact_setup() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    """Return synthetic system (B, J_c, u_true, lambda_true, tau, mu).

    Underactuated 3-DOF system with 1 actuator (u in R^1) and 1 3D contact point
    (lambda in R^3: fx, fy, fz).
    Equations of motion: tau = B u + J_c^T lambda in R^3.
    """
    rng = np.random.default_rng(42)
    # B: (3, 1) - actuator drives joint 2 only
    b_mat = np.array([[0.0], [1.0], [0.5]])
    # J_c: (3, 3) - contact Jacobian mapping v in R^3 to contact point velocity in R^3
    j_c = np.array(
        [
            [1.0, 0.0, 0.2],  # fx direction
            [0.0, 1.0, -0.1],  # fy direction
            [0.2, 0.3, 1.0],  # fz direction (normal)
        ]
    )
    mu = 0.6  # friction coefficient
    u_true = np.array([[2.5]])  # 1 node, 1 input
    # lambda_true strictly inside cone: fz = 20.0, fx = 3.0, fy = 4.0 -> ||(fx,fy)|| = 5.0 <= 0.6 * 20 = 12.0
    lambda_true = np.array([[3.0, 4.0, 20.0]])
    tau = (b_mat @ u_true.T + j_c.T @ lambda_true.T).T  # (1, 3)
    return b_mat, j_c, u_true, lambda_true, tau, mu


def test_preconditions_validation() -> None:
    b_mat, j_c, _, _, tau, mu = _synthetic_contact_setup()

    # Friction coefficient must be positive
    with pytest.raises(ContractViolationError):
        ContactWrenchProblem(
            regressors=tau,
            input_matrix=b_mat,
            contact_jacobian=j_c,
            friction_coefficient=0.0,
        )
    with pytest.raises(ContractViolationError):
        ContactWrenchProblem(
            regressors=tau,
            input_matrix=b_mat,
            contact_jacobian=j_c,
            friction_coefficient=-0.5,
        )

    # Input matrix rows must match regressor rows per node
    with pytest.raises(ContractViolationError):
        ContactWrenchProblem(
            regressors=tau,
            input_matrix=np.ones((4, 1)),
            contact_jacobian=j_c,
            friction_coefficient=mu,
        )

    # Contact Jacobian generalized dimension must match n_v
    with pytest.raises(ContractViolationError):
        ContactWrenchProblem(
            regressors=tau,
            input_matrix=b_mat,
            contact_jacobian=np.ones((3, 4)),
            friction_coefficient=mu,
        )

    # Measured GRF shape mismatch
    with pytest.raises(ContractViolationError):
        ContactWrenchProblem(
            regressors=tau,
            input_matrix=b_mat,
            contact_jacobian=j_c,
            friction_coefficient=mu,
            measured_grf=np.ones((1, 4)),  # expected (1, 3)
        )


def test_cone_projection_and_explicit_violation_reporting() -> None:
    mu = 0.5

    # 1. Point strictly inside cone: fx=1, fy=1, fz=10 -> ft = sqrt(2) ~ 1.414 <= 0.5 * 10 = 5.0
    inside = np.array([[1.0, 1.0, 10.0]])
    proj_inside = project_friction_cone(inside, mu)
    np.testing.assert_allclose(proj_inside, inside, atol=1e-12)
    viol_inside = compute_cone_violations(inside, mu)
    assert viol_inside.is_cone_feasible
    assert viol_inside.max_cone_violation == 0.0
    assert viol_inside.max_unilateral_violation == 0.0

    # 2. Point violating friction cone: fz = 2.0, fx = 3.0, fy = 4.0 -> ft = 5.0 > 0.5 * 2.0 = 1.0
    outside = np.array([[3.0, 4.0, 2.0]])
    viol_outside = compute_cone_violations(outside, mu)
    assert not viol_outside.is_cone_feasible
    assert viol_outside.max_cone_violation > 0.0
    np.testing.assert_allclose(viol_outside.max_cone_violation, 4.0, atol=1e-10)
    assert viol_outside.max_unilateral_violation == 0.0

    proj_outside = project_friction_cone(outside, mu)
    # Projected point must satisfy cone boundary: ft = mu * fz
    ft_proj = np.hypot(proj_outside[0, 0], proj_outside[0, 1])
    fz_proj = proj_outside[0, 2]
    np.testing.assert_allclose(ft_proj, mu * fz_proj, atol=1e-10)
    # Tangential direction preserved: [3, 4] / 5
    tangent_dir = proj_outside[0, :2] / ft_proj
    np.testing.assert_allclose(tangent_dir, np.array([0.6, 0.8]), atol=1e-10)

    # 3. Tensile point (fz < 0, pulling on ground)
    tensile = np.array([[1.0, 0.0, -5.0]])
    viol_tensile = compute_cone_violations(tensile, mu)
    assert not viol_tensile.is_cone_feasible
    assert viol_tensile.max_unilateral_violation == 5.0
    assert viol_tensile.max_cone_violation > 0.0
    # In polar/dual cone reflection: projection onto cone is apex (0, 0, 0)
    proj_tensile = project_friction_cone(tensile, mu)
    np.testing.assert_allclose(proj_tensile, np.zeros((1, 3)), atol=1e-12)


def test_solve_without_grf_returns_feasible_cone_forces_and_reports_identifiability_loss() -> (
    None
):
    b_mat, j_c, _, _, tau, mu = _synthetic_contact_setup()
    problem = ContactWrenchProblem(
        regressors=tau,
        input_matrix=b_mat,
        contact_jacobian=j_c,
        friction_coefficient=mu,
        measured_grf=None,
    )
    solution = solve_contact_wrenches(problem)

    assert isinstance(solution, ContactWrenchSolution)
    assert solution.converged
    assert solution.is_cone_feasible
    assert solution.max_cone_violation < 1e-5
    assert solution.max_unilateral_violation < 1e-5
    np.testing.assert_allclose(solution.dynamics_residual, 0.0, atol=1e-4)

    # Identifiability loss diagnostics
    diag = solution.identifiability
    assert isinstance(diag, ContactIdentifiabilityDiagnostics)
    # [B J_c^T] is 3 x (1 + 3) = 3 x 4, so without GRF rank is at most 3 < 4
    assert diag.total_variables == 4
    assert diag.rank_without_grf < diag.total_variables
    assert diag.condition_number_without_grf > 1e6
    assert np.any(diag.observable_fraction_without_grf < 0.99)
    assert diag.has_identifiability_loss

    # With GRF [B J_c^T; 0 I] has full rank 4
    assert diag.rank_with_grf == diag.total_variables
    assert diag.condition_number_with_grf < 1e3
    np.testing.assert_allclose(diag.observable_fraction_with_grf, 1.0, atol=1e-5)


def test_solve_with_synthetic_grf_truth_recovers_wrenches_and_inputs() -> None:
    b_mat, j_c, u_true, lambda_true, tau, mu = _synthetic_contact_setup()
    problem = ContactWrenchProblem(
        regressors=tau,
        input_matrix=b_mat,
        contact_jacobian=j_c,
        friction_coefficient=mu,
        measured_grf=lambda_true,
        grf_weight=10.0,
    )
    solution = solve_contact_wrenches(problem)

    assert solution.converged
    assert solution.is_cone_feasible
    assert solution.max_cone_violation < 1e-5

    # Recovered inputs and wrenches match truth within tolerance
    np.testing.assert_allclose(solution.inputs, u_true, atol=1e-4)
    np.testing.assert_allclose(solution.wrenches, lambda_true, atol=1e-4)
    np.testing.assert_allclose(solution.dynamics_residual, 0.0, atol=1e-4)
    assert solution.grf_residual is not None
    np.testing.assert_allclose(solution.grf_residual, 0.0, atol=1e-4)


def test_retained_negative_liftoff_edge_case() -> None:
    """Lift-off edge case: upward acceleration requires tensile force if contact active.

    Normal contact is strictly unilateral (fz >= 0). An unconstrained solve pulls
    downward on the foot (fz < 0). The cone-constrained solve clamps fz = 0,
    correctly discovering contact detachment (lift-off).
    """
    # System where contact force fz acts along coordinate 2
    b_mat = np.array([[1.0], [0.0], [0.0]])  # actuator only on coord 0
    j_c = np.eye(3)  # contact wrench directly acts on generalized coords
    mu = 0.5
    # Required net acceleration/force has large positive upward value on coordinate 2:
    # tau_2 = -15.0, so to satisfy tau = B u + J_c^T lambda:
    # 0 = tau_2 = -15.0 + fz -> unconstrained fz = 15.0, but suppose tau_2 = +15:
    # tau = [2.0, 0.0, -10.0]
    tau = np.array([[2.0, 0.0, -10.0]])  # coord 2 requires negative force from contact
    # Unconstrained solve would give lambda_2 = -10.0 (tensile!)
    problem = ContactWrenchProblem(
        regressors=tau,
        input_matrix=b_mat,
        contact_jacobian=j_c,
        friction_coefficient=mu,
        measured_grf=None,
    )
    solution = solve_contact_wrenches(problem)

    # Constrained solve must NOT have negative normal force (no tension)
    assert solution.converged
    assert solution.is_cone_feasible
    assert solution.wrenches[0, 2] >= 0.0
    # Lift-off: normal contact force drops to 0
    np.testing.assert_allclose(solution.wrenches[0, 2], 0.0, atol=1e-6)
    # And tangential forces must also be 0 because ft <= mu * 0 = 0
    np.testing.assert_allclose(solution.wrenches[0, :2], 0.0, atol=1e-6)


def test_retained_negative_stick_slip_edge_case() -> None:
    """Stick-slip edge case: required shear force exceeds traction limit (||ft|| > mu * fz).

    The cone-constrained solve limits tangential force to the cone boundary ||ft|| = mu * fz.
    Any attempt to apply more tangential force than friction allows is bounded by the cone.
    """
    b_mat = np.array([[0.0], [0.0], [0.0]])  # purely passive
    j_c = np.eye(3)
    mu = 0.3
    # Normal force needed is 10.0, but shear force demanded is 8.0:
    # Friction limit is mu * 10.0 = 3.0 < 8.0
    tau = np.array([[8.0, 0.0, 10.0]])
    problem = ContactWrenchProblem(
        regressors=tau,
        input_matrix=b_mat,
        contact_jacobian=j_c,
        friction_coefficient=mu,
        measured_grf=None,
    )
    solution = solve_contact_wrenches(problem)

    assert solution.converged
    assert solution.is_cone_feasible
    assert solution.max_cone_violation < 1e-5

    # Optimal projection onto friction cone boundary:
    # fz* = (10 + mu * 8) / (1 + mu^2) = 12.4 / 1.09 ~ 11.376
    expected_fz = (10.0 + mu * 8.0) / (1.0 + mu**2)
    np.testing.assert_allclose(solution.wrenches[0, 2], expected_fz, atol=1e-3)
    # Shear force is on the cone boundary: ||ft|| = mu * fz
    ft = np.hypot(solution.wrenches[0, 0], solution.wrenches[0, 1])
    np.testing.assert_allclose(ft, mu * solution.wrenches[0, 2], atol=1e-4)


def test_solvers_admm_and_slsqp_agreement() -> None:
    b_mat, j_c, u_true, lambda_true, tau, mu = _synthetic_contact_setup()
    problem = ContactWrenchProblem(
        regressors=tau,
        input_matrix=b_mat,
        contact_jacobian=j_c,
        friction_coefficient=mu,
        measured_grf=lambda_true,
        grf_weight=5.0,
    )
    sol_admm = solve_contact_wrenches(problem, solver="admm")
    sol_slsqp = solve_contact_wrenches(problem, solver="slsqp")

    assert sol_admm.converged
    assert sol_slsqp.converged
    np.testing.assert_allclose(sol_admm.inputs, sol_slsqp.inputs, atol=1e-3)
    np.testing.assert_allclose(sol_admm.wrenches, sol_slsqp.wrenches, atol=1e-3)
