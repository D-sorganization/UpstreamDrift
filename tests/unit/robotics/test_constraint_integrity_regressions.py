"""Regression tests for silently-violated constraints in the robotics stack.

* #8022 - ``NullspaceQPSolver`` dropped inequality constraints and post-clamped
  to variable bounds, reporting ``success=True`` for solutions that violated
  the constraints it was given.
* #8007 - ``check_force_closure`` accepted rank-deficient grasps and always
  returned a quality margin of exactly 0.0.
* #8017 - the ZMP angular-momentum term was multiplied by the CoM height, which
  is dimensionally inconsistent and left a residual horizontal moment at the
  reported ZMP.
"""

from __future__ import annotations

import numpy as np
import pytest

from model_generation.core.constants import DEFAULT_MASS_KG
from src.robotics.contact.grasp_analysis import check_force_closure
from src.robotics.control.whole_body.qp_solver import (
    NullspaceQPSolver,
    QPProblem,
    ScipyQPSolver,
)
from src.robotics.core.types import ContactState
from src.robotics.locomotion.zmp_computer import ZMPComputer

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# #8022 - NullspaceQPSolver
# ---------------------------------------------------------------------------


class TestNullspaceQPSolverConstraintIntegrity:
    """The solver must never report success while ignoring a constraint."""

    def test_inequality_constraints_are_rejected_not_ignored(self) -> None:
        """`x[2] >= 0` (no pulling contact force) must not be silently dropped."""
        problem = QPProblem(
            H=np.eye(3),
            g=np.array([0.0, 0.0, 5.0]),  # unconstrained optimum x[2] = -5
            A_ineq=np.array([[0.0, 0.0, -1.0]]),
            ub_ineq=np.array([0.0]),
        )
        solution = NullspaceQPSolver().solve(problem)

        assert solution.success is False
        assert solution.x is None
        assert "inequality" in solution.status.lower()

    def test_scipy_solver_still_handles_the_same_problem(self) -> None:
        problem = QPProblem(
            H=np.eye(3),
            g=np.array([0.0, 0.0, 5.0]),
            A_ineq=np.array([[0.0, 0.0, -1.0]]),
            ub_ineq=np.array([0.0]),
        )
        solver = ScipyQPSolver()
        if not solver.is_available():  # pragma: no cover - scipy present in CI
            pytest.skip("scipy not available")
        solution = solver.solve(problem)
        assert solution.success
        assert solution.x is not None
        assert np.all(problem.A_ineq @ solution.x <= 1e-6)

    def test_infeasible_bounds_plus_equality_is_not_reported_successful(self) -> None:
        """x0 + x1 = 4 with both variables capped at 1 has no solution."""
        problem = QPProblem(
            H=np.eye(2),
            g=np.zeros(2),
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([4.0]),
            x_lb=np.array([-1.0, -1.0]),
            x_ub=np.array([1.0, 1.0]),
        )
        solution = NullspaceQPSolver().solve(problem)
        assert solution.success is False

    def test_equality_is_preserved_when_bounds_are_inactive(self) -> None:
        problem = QPProblem(
            H=np.eye(2),
            g=np.zeros(2),
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([1.5]),
            x_lb=np.array([-1.0, -1.0]),
            x_ub=np.array([1.0, 1.0]),
        )
        solution = NullspaceQPSolver().solve(problem)
        assert solution.success
        assert solution.x is not None
        assert np.allclose(problem.A_eq @ solution.x, problem.b_eq, atol=1e-9)
        assert np.allclose(solution.x, [0.75, 0.75], atol=1e-9)

    def test_active_bound_keeps_equality_and_matches_scipy(self) -> None:
        """A bound-active solution must still satisfy A_eq @ x = b_eq exactly."""
        problem = QPProblem(
            H=np.eye(2),
            g=np.array([-10.0, 0.0]),
            A_eq=np.array([[1.0, 1.0]]),
            b_eq=np.array([1.0]),
            x_lb=np.array([-5.0, -5.0]),
            x_ub=np.array([0.5, 5.0]),
        )
        solution = NullspaceQPSolver().solve(problem)
        assert solution.success
        assert solution.x is not None
        assert np.allclose(problem.A_eq @ solution.x, problem.b_eq, atol=1e-9)
        assert np.all(solution.x <= problem.x_ub + 1e-9)
        assert np.all(solution.x >= problem.x_lb - 1e-9)
        assert np.allclose(solution.x, [0.5, 0.5], atol=1e-9)

        reference = ScipyQPSolver()
        if reference.is_available():
            assert np.allclose(solution.x, reference.solve(problem).x, atol=1e-6)

    def test_bound_only_problem_still_respects_bounds(self) -> None:
        problem = QPProblem(
            H=np.eye(2),
            g=np.array([100.0, -100.0]),
            x_lb=np.array([-1.0, -1.0]),
            x_ub=np.array([1.0, 1.0]),
        )
        solution = NullspaceQPSolver().solve(problem)
        assert solution.success
        assert np.allclose(solution.x, [-1.0, 1.0], atol=1e-9)


# ---------------------------------------------------------------------------
# #8007 - check_force_closure
# ---------------------------------------------------------------------------


def _contact(contact_id: int, position, normal, mu: float = 0.5) -> ContactState:
    return ContactState(
        contact_id=contact_id,
        body_a=f"finger{contact_id}",
        body_b="object",
        position=np.asarray(position, dtype=float),
        normal=np.asarray(normal, dtype=float),
        normal_force=10.0,
        friction_coefficient=mu,
    )


def _enveloping_contacts(n: int) -> list[ContactState]:
    directions = np.vstack([np.eye(3), -np.eye(3)])[:n]
    return [_contact(i, 0.05 * d, -d) for i, d in enumerate(directions)]


class TestForceClosure:
    def test_two_antipodal_contacts_have_no_force_closure(self) -> None:
        """Rank 5 of 6: no contact force can torque about the contact line."""
        contacts = [
            _contact(0, [-0.05, 0.0, 0.0], [1.0, 0.0, 0.0]),
            _contact(1, [0.05, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        ]
        has_closure, margin = check_force_closure(contacts)
        assert has_closure is False
        assert margin == 0.0

    def test_enveloping_grasp_has_positive_margin(self) -> None:
        has_closure, margin = check_force_closure(_enveloping_contacts(6))
        assert has_closure is True
        assert margin > 0.0

    def test_margin_discriminates_between_grasps(self) -> None:
        """The margin used to be exactly 0.0 for every grasp."""
        _, margin_four = check_force_closure(_enveloping_contacts(4))
        _, margin_six = check_force_closure(_enveloping_contacts(6))
        assert margin_six > margin_four > 0.0

    def test_margin_is_bounded_by_the_wrench_hull_support_function(self) -> None:
        """epsilon = min_{||d||=1} max_i <g_i, d>; sampling gives an upper bound."""
        from src.robotics.contact.grasp_analysis import _build_wrench_generators

        contacts = _enveloping_contacts(6)
        _, margin = check_force_closure(contacts)
        generators = _build_wrench_generators(contacts, 8)

        rng = np.random.default_rng(3)
        directions = rng.normal(size=(20000, 6))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        upper_bound = float(np.min(np.max(directions @ generators, axis=1)))
        assert 0.0 < margin <= upper_bound + 1e-9

    def test_closure_implies_positive_margin(self) -> None:
        for n in (3, 4, 5, 6):
            has_closure, margin = check_force_closure(_enveloping_contacts(n))
            assert has_closure == (margin > 0.0)


# ---------------------------------------------------------------------------
# #8017 - ZMP
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Non-humanoid engine stub; ZMPComputer falls back to DEFAULT_MASS_KG."""

    def get_com_velocity(self) -> np.ndarray:
        return np.zeros(3)


def _residual_horizontal_moment(
    com: np.ndarray,
    acceleration: np.ndarray,
    angular_momentum_rate: np.ndarray,
    zmp: np.ndarray,
    mass: float,
    gravity: float,
) -> np.ndarray:
    """dL/dt + (c - p) x m (a + g z_hat); must vanish at the true ZMP."""
    force = mass * (acceleration + np.array([0.0, 0.0, gravity]))
    return (angular_momentum_rate + np.cross(com - zmp, force))[:2]


class TestZMPAngularMomentumTerm:
    @pytest.mark.parametrize("height", [0.30, 0.60, 0.90, 1.20])
    def test_no_residual_moment_at_reported_zmp(self, height: float) -> None:
        com = np.array([0.05, -0.02, height])
        acceleration = np.array([1.5, -0.8, 0.3])
        angular_momentum_rate = np.array([30.0, 20.0, 0.0])

        computer = ZMPComputer(_FakeEngine())
        result = computer.compute_zmp(
            com_position=com,
            com_acceleration=acceleration,
            angular_momentum_rate=angular_momentum_rate,
        )

        residual = _residual_horizontal_moment(
            com,
            acceleration,
            angular_momentum_rate,
            result.zmp_position,
            DEFAULT_MASS_KG,
            computer.GRAVITY,
        )
        assert np.allclose(residual, 0.0, atol=1e-9)

    def test_matches_closed_form_definition(self) -> None:
        com = np.array([0.05, -0.02, 0.90])
        acceleration = np.array([1.5, -0.8, 0.3])
        angular_momentum_rate = np.array([12.0, 8.0, 0.0])

        computer = ZMPComputer(_FakeEngine())
        result = computer.compute_zmp(
            com_position=com,
            com_acceleration=acceleration,
            angular_momentum_rate=angular_momentum_rate,
        )

        denom = acceleration[2] + computer.GRAVITY
        expected = np.array(
            [
                com[0]
                - com[2] * acceleration[0] / denom
                - angular_momentum_rate[1] / (DEFAULT_MASS_KG * denom),
                com[1]
                - com[2] * acceleration[1] / denom
                + angular_momentum_rate[0] / (DEFAULT_MASS_KG * denom),
                0.0,
            ]
        )
        assert np.allclose(result.zmp_position, expected, atol=1e-12)

    def test_zero_angular_momentum_is_unchanged(self) -> None:
        """The defect only bit callers supplying a non-zero dL/dt."""
        com = np.array([0.05, -0.02, 0.90])
        acceleration = np.array([1.5, -0.8, 0.3])
        computer = ZMPComputer(_FakeEngine())
        result = computer.compute_zmp(com_position=com, com_acceleration=acceleration)
        denom = acceleration[2] + computer.GRAVITY
        expected = np.array(
            [
                com[0] - com[2] * acceleration[0] / denom,
                com[1] - com[2] * acceleration[1] / denom,
                0.0,
            ]
        )
        assert np.allclose(result.zmp_position, expected, atol=1e-12)
