"""Tests for src/research/mpc/*."""

from __future__ import annotations

import numpy as np
import pytest

from src.research.mpc.controller import (
    Constraint,
    CostFunction,
    ModelPredictiveController,
    MPCResult,
)
from src.research.mpc.specialized import (
    CentroidalMPC,
    CentroidalState,
    WholeBodyMPC,
)


class TestCostFunction:
    def test_running_cost_no_ref(self) -> None:
        cf = CostFunction(Q=np.eye(2), R=np.eye(1))
        c = cf.evaluate_running_cost(np.array([1.0, 0.0]), np.array([2.0]))
        assert c == pytest.approx(1.0 + 4.0)

    def test_running_cost_with_ref_1d(self) -> None:
        cf = CostFunction(
            Q=np.eye(2),
            R=np.eye(1),
            x_ref=np.array([1.0, 1.0]),
            u_ref=np.array([0.5]),
        )
        c = cf.evaluate_running_cost(np.array([2.0, 1.0]), np.array([0.5]))
        assert c == pytest.approx(1.0)

    def test_running_cost_with_traj_ref(self) -> None:
        cf = CostFunction(
            Q=np.eye(2),
            R=np.eye(1),
            x_ref=np.array([[0.0, 0.0], [1.0, 1.0]]),
            u_ref=np.array([[0.0], [1.0]]),
        )
        c = cf.evaluate_running_cost(np.array([1.0, 1.0]), np.array([1.0]), k=1)
        assert c == pytest.approx(0.0)

    def test_running_cost_ref_past_horizon(self) -> None:
        cf = CostFunction(
            Q=np.eye(2),
            R=np.eye(1),
            x_ref=np.array([[1.0, 1.0]]),
            u_ref=np.array([[0.0]]),
        )
        # k beyond, uses last
        c = cf.evaluate_running_cost(np.array([1.0, 1.0]), np.array([0.0]), k=5)
        assert c == pytest.approx(0.0)

    def test_running_cost_linear_terms(self) -> None:
        cf = CostFunction(
            Q=np.eye(2),
            R=np.eye(1),
            q=np.array([1.0, 0.0]),
            r=np.array([2.0]),
        )
        c = cf.evaluate_running_cost(np.array([1.0, 0.0]), np.array([1.0]))
        # x^T Q x = 1; q.x = 1; u^T R u = 1; r.u = 2 -> total 5
        assert c == pytest.approx(5.0)

    def test_terminal_cost_zero_when_no_P(self) -> None:
        cf = CostFunction(Q=np.eye(2), R=np.eye(1))
        assert cf.evaluate_terminal_cost(np.array([3.0, 4.0])) == 0.0

    def test_terminal_cost_with_P_and_ref(self) -> None:
        cf = CostFunction(
            Q=np.eye(2),
            R=np.eye(1),
            P=2 * np.eye(2),
            x_ref=np.array([1.0, 1.0]),
            p=np.array([1.0, 0.0]),
        )
        c = cf.evaluate_terminal_cost(np.array([2.0, 1.0]))
        # x_err = [1,0]; x.P.x = 2; p.x_err = 1 -> 3
        assert c == pytest.approx(3.0)

    def test_terminal_cost_traj_ref(self) -> None:
        cf = CostFunction(
            Q=np.eye(2),
            R=np.eye(1),
            P=np.eye(2),
            x_ref=np.array([[0.0, 0.0], [1.0, 1.0]]),
        )
        c = cf.evaluate_terminal_cost(np.array([1.0, 1.0]))
        assert c == pytest.approx(0.0)


class TestConstraint:
    def test_default(self) -> None:
        c = Constraint()
        assert c.constraint_type == "mixed"
        assert c.A is None and c.B is None


class TestMPCResult:
    def test_defaults(self) -> None:
        r = MPCResult(
            success=True, optimal_states=None, optimal_controls=None, cost=0.0
        )
        assert r.solve_time == 0.0
        assert r.iterations == 0


class TestModelPredictiveController:
    def test_construction(self, fake_engine) -> None:
        mpc = ModelPredictiveController(fake_engine, horizon=5, dt=0.05)
        assert mpc.n_states == 4
        assert mpc.n_controls == 2
        assert mpc.horizon == 5

    def test_no_n_q_attr(self) -> None:
        class Bare:
            pass

        mpc = ModelPredictiveController(Bare(), horizon=2)
        assert mpc.n_states == 14
        assert mpc.n_controls == 7

    def test_set_cost_and_constraints(self, fake_engine) -> None:
        mpc = ModelPredictiveController(fake_engine, horizon=3)
        cf = CostFunction(Q=np.eye(4), R=np.eye(2), P=np.eye(4))
        mpc.set_cost_function(cf)
        assert mpc._cost is cf

        c1 = Constraint(B=np.eye(2), ub=np.ones(2))
        mpc.add_constraint(c1)
        assert len(mpc._constraints) == 1
        mpc.set_constraints([c1, c1])
        assert len(mpc._constraints) == 2
        mpc.clear_constraints()
        assert mpc._constraints == []

    def test_solve_raises_without_cost(self, fake_engine) -> None:
        mpc = ModelPredictiveController(fake_engine, horizon=3)
        with pytest.raises(ValueError, match="Cost function not set"):
            mpc.solve(np.zeros(4))

    def test_solve_runs(self, fake_engine) -> None:
        mpc = ModelPredictiveController(fake_engine, horizon=3, dt=0.05)
        mpc._max_iterations = 2  # speed
        cf = CostFunction(
            Q=np.eye(4),
            R=np.eye(2) * 0.01,
            P=np.eye(4),
        )
        mpc.set_cost_function(cf)
        result = mpc.solve(np.array([1.0, 0.0, 0.0, 0.0]))
        assert isinstance(result, MPCResult)
        assert result.optimal_states.shape == (4, 4)
        assert result.optimal_controls.shape == (3, 2)
        assert result.solve_time >= 0

    def test_solve_with_reference(self, fake_engine) -> None:
        mpc = ModelPredictiveController(fake_engine, horizon=2, dt=0.05)
        mpc._max_iterations = 1
        cf = CostFunction(Q=np.eye(4), R=np.eye(2) * 0.01)
        mpc.set_cost_function(cf)
        ref = np.zeros((3, 4))
        res = mpc.solve(np.zeros(4), reference_trajectory=ref)
        assert res.optimal_states is not None

    def test_get_first_control(self, fake_engine) -> None:
        mpc = ModelPredictiveController(fake_engine)
        r = MPCResult(
            success=True,
            optimal_states=None,
            optimal_controls=np.array([[1.0, 2.0], [3.0, 4.0]]),
            cost=0.0,
        )
        u = mpc.get_first_control(r)
        np.testing.assert_allclose(u, [1.0, 2.0])

    def test_get_first_control_empty(self, fake_engine) -> None:
        mpc = ModelPredictiveController(fake_engine)
        r = MPCResult(success=False, optimal_states=None, optimal_controls=None, cost=0)
        u = mpc.get_first_control(r)
        assert u.shape == (2,)
        np.testing.assert_allclose(u, 0.0)

    def test_constraint_violation_bounds(self, fake_engine) -> None:
        mpc = ModelPredictiveController(fake_engine, horizon=2, dt=0.05)
        mpc._max_iterations = 1
        cf = CostFunction(Q=np.eye(4), R=np.eye(2) * 0.01)
        mpc.set_cost_function(cf)
        # Constraint that will be violated: u <= -100
        mpc.add_constraint(Constraint(B=np.eye(2)[:1, :], ub=np.array([-100.0])))
        res = mpc.solve(np.zeros(4))
        assert res.constraint_violations >= 0

    def test_dynamics_fallback_no_attrs(self) -> None:
        class Bare:
            n_q = 2
            n_v = 2

        mpc = ModelPredictiveController(Bare(), horizon=1, dt=0.1)
        # _dynamics on a bare engine uses fallback q+v*dt
        x_next = mpc._dynamics(np.array([0, 0, 1.0, 1.0]), np.zeros(2))
        np.testing.assert_allclose(x_next[:2], [0.1, 0.1])
        np.testing.assert_allclose(x_next[2:], [1.0, 1.0])


class TestCentroidalState:
    def test_construction(self) -> None:
        s = CentroidalState(
            com_position=np.zeros(3),
            com_velocity=np.zeros(3),
            angular_momentum=np.zeros(3),
            contact_forces={"l": np.zeros(3)},
        )
        assert s.com_position.shape == (3,)
        assert "l" in s.contact_forces


class TestCentroidalMPC:
    def test_construction(self, fake_engine) -> None:
        cmpc = CentroidalMPC(fake_engine, horizon=4, dt=0.05, n_contacts=2)
        assert cmpc.n_states == 9
        assert cmpc.n_controls == 6
        assert cmpc._cost is not None
        assert cmpc._mass == 50.0

    def test_set_mass(self, fake_engine) -> None:
        cmpc = CentroidalMPC(fake_engine, horizon=2)
        cmpc.set_mass(80.0)
        assert cmpc._mass == 80.0

    def test_update_contact_positions(self, fake_engine) -> None:
        cmpc = CentroidalMPC(fake_engine, horizon=2)
        new_pos = [np.array([1, 0, 0.0]), np.array([0, 1, 0.0])]
        cmpc.update_contact_positions(new_pos)
        assert cmpc._contact_positions[0][0] == 1

    def test_dynamics(self, fake_engine) -> None:
        cmpc = CentroidalMPC(fake_engine, horizon=2, dt=0.01, n_contacts=2)
        x = np.zeros(9)
        x[2] = 1.0  # com z
        u = np.zeros(6)
        u[2] = 50.0 * 9.81  # gravity-cancelling normal force
        x_next = cmpc._dynamics(x, u)
        assert x_next.shape == (9,)
        # acceleration ~ 0 -> velocity ~0
        np.testing.assert_allclose(x_next[3:6], 0.0, atol=1e-6)

    def test_friction_cone_constraints(self, fake_engine) -> None:
        cmpc = CentroidalMPC(fake_engine, horizon=2, n_contacts=2)
        cmpc.add_friction_cone_constraints()
        # 2 contacts * (1 normal + 2 axes * 1 constraint pair) = 6 constraints
        assert len(cmpc._constraints) == 6

    def test_set_gait_reference(self, fake_engine) -> None:
        cmpc = CentroidalMPC(fake_engine, horizon=5)
        cmpc.set_gait_reference(np.array([0.5, 0.0]), target_height=1.0)
        assert cmpc._cost.x_ref is not None
        assert cmpc._cost.x_ref.shape == (6, 9)
        assert cmpc._cost.x_ref[0, 2] == 1.0
        # CoM moves
        assert cmpc._cost.x_ref[-1, 0] > cmpc._cost.x_ref[0, 0]


class TestWholeBodyMPC:
    def test_construction(self, fake_engine) -> None:
        wb = WholeBodyMPC(fake_engine, horizon=3, dt=0.01)
        assert wb._cost is not None
        assert wb.n_states == 4

    def test_set_end_effector_target(self, fake_engine) -> None:
        wb = WholeBodyMPC(fake_engine)
        target = np.array([0.5, 0, 0.5, 1, 0, 0, 0.0])
        wb.set_end_effector_target("ee", target)
        assert "ee" in wb._end_effector_targets

    def test_set_joint_targets_with_vel(self, fake_engine) -> None:
        wb = WholeBodyMPC(fake_engine)
        wb.set_joint_targets(np.array([0.1, 0.2]), np.array([0.0, 0.0]))
        assert wb._cost.x_ref.shape == (4,)

    def test_set_joint_targets_default_vel(self, fake_engine) -> None:
        wb = WholeBodyMPC(fake_engine)
        wb.set_joint_targets(np.array([0.5, 0.5]))
        assert wb._cost.x_ref[2] == 0.0

    def test_joint_limit_constraints(self, fake_engine) -> None:
        wb = WholeBodyMPC(fake_engine)
        wb.add_joint_limit_constraints(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        assert len(wb._constraints) == 1

    def test_torque_limit_constraints(self, fake_engine) -> None:
        wb = WholeBodyMPC(fake_engine)
        wb.add_torque_limit_constraints(np.array([5.0, 5.0]))
        assert len(wb._constraints) == 1

    def test_solve_with_ee_tracking_no_ik(self, fake_engine) -> None:
        wb = WholeBodyMPC(fake_engine, horizon=2, dt=0.05)
        wb._max_iterations = 1
        res = wb.solve_with_ee_tracking(np.zeros(4))
        assert isinstance(res, MPCResult)

    def test_solve_with_ee_tracking_with_ik(self, fake_engine) -> None:
        class IkEngine:
            n_q = 2
            n_v = 2

            def solve_ik(self, name, target):  # noqa: ANN001
                return np.array([0.3, 0.3]), True

        wb = WholeBodyMPC(IkEngine(), horizon=2, dt=0.05)
        wb._max_iterations = 1
        wb.set_end_effector_target("ee", np.array([0.0, 0, 0, 1, 0, 0, 0.0]))
        res = wb.solve_with_ee_tracking(np.zeros(4))
        assert isinstance(res, MPCResult)
