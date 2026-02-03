"""Unit tests for control module.

Tests cover:
    - Task creation and validation
    - QP problem formulation and solving
    - Whole-body controller functionality
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src.robotics.control.whole_body.task import (
    Task,
    TaskType,
    create_com_task,
    create_posture_task,
    create_ee_task,
    create_contact_constraint,
)
from src.robotics.control.whole_body.qp_solver import (
    QPProblem,
    QPSolution,
    QPSolver,
    ScipyQPSolver,
    NullspaceQPSolver,
    create_default_solver,
)
from src.robotics.control.whole_body.wbc_controller import (
    WholeBodyController,
    WBCConfig,
    WBCSolution,
)


class TestTask:
    """Tests for Task class."""

    def test_create_task_basic(self) -> None:
        """Test creating a basic task."""
        J = np.eye(3)
        target = np.array([1.0, 0.0, 0.0])

        task = Task(
            name="test_task",
            task_type=TaskType.SOFT,
            priority=1,
            jacobian=J,
            target=target,
            weight=np.ones(3),
        )

        assert task.name == "test_task"
        assert task.task_type == TaskType.SOFT
        assert task.priority == 1
        assert_array_equal(task.jacobian, J)

    def test_task_validation_target_mismatch(self) -> None:
        """Test task validates target dimensions."""
        J = np.eye(3)
        target = np.zeros(2)  # Wrong size

        with pytest.raises(ValueError, match="doesn't match"):
            Task(
                name="bad_task",
                task_type=TaskType.SOFT,
                priority=1,
                jacobian=J,
                target=target,
            )

    def test_task_priority(self) -> None:
        """Test task priority setting."""
        task = Task(
            name="high_priority",
            task_type=TaskType.SOFT,
            priority=10,
            jacobian=np.eye(3),
            target=np.zeros(3),
        )

        assert task.priority == 10

    def test_task_type_equality(self) -> None:
        """Test equality task type."""
        task = Task(
            name="eq_task",
            task_type=TaskType.EQUALITY,
            priority=0,
            jacobian=np.eye(3),
            target=np.zeros(3),
        )

        assert task.task_type == TaskType.EQUALITY

    def test_task_type_inequality_requires_bounds(self) -> None:
        """Test inequality task requires bounds."""
        with pytest.raises(ValueError, match="bound"):
            Task(
                name="ineq_task",
                task_type=TaskType.INEQUALITY,
                priority=1,
                jacobian=np.eye(3),
                target=np.zeros(3),
            )

    def test_create_com_task(self) -> None:
        """Test COM task factory."""
        J_com = np.random.randn(3, 6)
        com_target = np.array([0.0, 0.0, 0.5])
        com_current = np.array([0.0, 0.0, 0.4])
        com_vel = np.array([0.0, 0.0, 0.01])

        task = create_com_task(
            jacobian_com=J_com,
            com_target=com_target,
            com_current=com_current,
            com_velocity=com_vel,
            gain_p=100.0,
            gain_d=20.0,
            weight=1.0,
        )

        assert task.name == "com_tracking"
        assert task.task_type == TaskType.SOFT
        assert task.jacobian.shape == (3, 6)

    def test_create_posture_task(self) -> None:
        """Test posture task factory."""
        n_v = 7
        q_target = np.zeros(n_v)
        q_current = 0.1 * np.ones(n_v)
        v_current = np.zeros(n_v)

        task = create_posture_task(
            n_v=n_v,
            q_target=q_target,
            q_current=q_current,
            v_current=v_current,
            gain_p=50.0,
            gain_d=10.0,
            weight=0.1,
        )

        assert task.name == "posture"
        assert task.task_type == TaskType.SOFT
        assert task.jacobian.shape == (n_v, n_v)

    def test_create_ee_task(self) -> None:
        """Test end-effector task factory."""
        J_ee = np.random.randn(6, 7)
        ee_target = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])
        ee_current = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0])
        ee_velocity = np.zeros(6)

        task = create_ee_task(
            jacobian_ee=J_ee,
            ee_target=ee_target,
            ee_current=ee_current,
            ee_velocity=ee_velocity,
            gain_p=100.0,
            gain_d=20.0,
            weight=1.0,
        )

        assert task.name == "end_effector"
        assert task.task_type == TaskType.SOFT

    def test_create_contact_constraint(self) -> None:
        """Test contact constraint factory."""
        J_c = np.random.randn(3, 6)
        contact_vel = np.zeros(3)

        task = create_contact_constraint(
            jacobian_contact=J_c,
            contact_velocity=contact_vel,
        )

        assert task.name == "contact"
        assert task.task_type == TaskType.EQUALITY
        assert task.priority == 0  # Highest priority

    def test_task_get_weight_matrix(self) -> None:
        """Test weight matrix generation."""
        weights = np.array([1.0, 2.0, 3.0])
        task = Task(
            name="weighted",
            task_type=TaskType.SOFT,
            priority=1,
            jacobian=np.eye(3),
            target=np.zeros(3),
            weight=weights,
        )

        W = task.get_weight_matrix()
        assert W.shape == (3, 3)
        assert_array_equal(np.diag(W), weights)


class TestQPProblem:
    """Tests for QPProblem class."""

    def test_create_qp_problem(self) -> None:
        """Test creating a QP problem."""
        H = np.eye(3)
        g = np.array([1.0, 2.0, 3.0])

        problem = QPProblem(H=H, g=g)

        assert problem.n_vars == 3
        assert problem.n_eq == 0
        assert problem.n_ineq == 0

    def test_qp_problem_with_equality(self) -> None:
        """Test QP problem with equality constraints."""
        H = np.eye(3)
        g = np.zeros(3)
        A_eq = np.array([[1.0, 1.0, 1.0]])
        b_eq = np.array([1.0])

        problem = QPProblem(H=H, g=g, A_eq=A_eq, b_eq=b_eq)

        assert problem.n_vars == 3
        assert problem.n_eq == 1

    def test_qp_problem_validation(self) -> None:
        """Test QP problem validates dimensions."""
        H = np.eye(3)
        g = np.array([1.0, 2.0])  # Wrong size

        with pytest.raises(ValueError, match="doesn't match"):
            QPProblem(H=H, g=g)

    def test_qp_problem_nonsquare_h(self) -> None:
        """Test QP problem rejects non-square H."""
        H = np.ones((3, 4))
        g = np.zeros(3)

        with pytest.raises(ValueError, match="square"):
            QPProblem(H=H, g=g)


class TestQPSolvers:
    """Tests for QP solver implementations."""

    def test_scipy_solver_available(self) -> None:
        """Test scipy solver availability check."""
        solver = ScipyQPSolver()
        # Should be True if scipy is installed
        assert isinstance(solver.is_available(), bool)

    def test_nullspace_solver_always_available(self) -> None:
        """Test nullspace solver is always available."""
        solver = NullspaceQPSolver()
        assert solver.is_available() is True

    def test_nullspace_solve_unconstrained(self) -> None:
        """Test nullspace solver on unconstrained problem."""
        # min 0.5 * x^T @ H @ x + g^T @ x
        # H = I, g = [1, 2, 3]
        # Solution: x = -g = [-1, -2, -3]
        H = np.eye(3)
        g = np.array([1.0, 2.0, 3.0])

        problem = QPProblem(H=H, g=g)
        solver = NullspaceQPSolver()
        solution = solver.solve(problem)

        assert solution.success
        assert solution.x is not None
        # Use looser tolerance due to regularization
        assert_allclose(solution.x, -g, atol=1e-4)

    def test_nullspace_solve_with_equality(self) -> None:
        """Test nullspace solver with equality constraints."""
        # min 0.5 * ||x||^2
        # s.t. x[0] + x[1] + x[2] = 3
        # Solution: x = [1, 1, 1]
        H = np.eye(3)
        g = np.zeros(3)
        A_eq = np.array([[1.0, 1.0, 1.0]])
        b_eq = np.array([3.0])

        problem = QPProblem(H=H, g=g, A_eq=A_eq, b_eq=b_eq)
        solver = NullspaceQPSolver()
        solution = solver.solve(problem)

        assert solution.success
        assert solution.x is not None
        assert_allclose(solution.x, [1, 1, 1], atol=1e-4)

    def test_scipy_solve_unconstrained(self) -> None:
        """Test scipy solver on unconstrained problem."""
        solver = ScipyQPSolver()
        if not solver.is_available():
            pytest.skip("scipy not available")

        H = np.eye(3)
        g = np.array([1.0, 2.0, 3.0])

        problem = QPProblem(H=H, g=g)
        solution = solver.solve(problem)

        assert solution.success
        assert solution.x is not None
        assert_allclose(solution.x, -g, atol=1e-3)

    def test_scipy_solve_with_bounds(self) -> None:
        """Test scipy solver with variable bounds."""
        solver = ScipyQPSolver()
        if not solver.is_available():
            pytest.skip("scipy not available")

        H = np.eye(3)
        g = np.array([1.0, 2.0, 3.0])
        x_lb = np.array([0.0, 0.0, 0.0])  # Non-negative
        x_ub = np.array([np.inf, np.inf, np.inf])

        problem = QPProblem(H=H, g=g, x_lb=x_lb, x_ub=x_ub)
        solution = solver.solve(problem)

        assert solution.success
        assert solution.x is not None
        # Solution should be at origin (constrained by non-negativity)
        assert_allclose(solution.x, [0, 0, 0], atol=1e-6)

    def test_create_default_solver(self) -> None:
        """Test default solver factory."""
        solver = create_default_solver()
        assert isinstance(solver, QPSolver)
        assert solver.is_available()

    def test_solver_returns_iterations(self) -> None:
        """Test solver returns iteration count."""
        solver = NullspaceQPSolver()
        problem = QPProblem(H=np.eye(2), g=np.zeros(2))
        solution = solver.solve(problem)

        assert solution.iterations >= 0

    def test_solver_returns_solve_time(self) -> None:
        """Test solver returns solve time."""
        solver = NullspaceQPSolver()
        problem = QPProblem(H=np.eye(2), g=np.zeros(2))
        solution = solver.solve(problem)

        assert solution.solve_time >= 0.0


class MockEngine:
    """Mock engine for testing WBC.

    Implements RoboticsCapable protocol.
    """

    def __init__(self, n_v: int = 6) -> None:
        self.n_v = n_v
        self._q = np.zeros(n_v)
        self._qd = np.zeros(n_v)
        self._time = 0.0

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q.copy(), self._qd.copy()

    def set_state(self, q: np.ndarray, qd: np.ndarray) -> None:
        self._q = q.copy()
        self._qd = qd.copy()

    def compute_mass_matrix(self) -> np.ndarray:
        return np.eye(self.n_v)

    def compute_bias_forces(self) -> np.ndarray:
        return np.zeros(self.n_v)

    def compute_gravity_forces(self) -> np.ndarray:
        return np.zeros(self.n_v)

    def compute_jacobian(self, body_name: str) -> dict | None:
        J_lin = np.random.randn(3, self.n_v)
        J_ang = np.random.randn(3, self.n_v)
        return {"linear": J_lin, "angular": J_ang}

    def get_time(self) -> float:
        return self._time


class TestWholeBodyController:
    """Tests for WholeBodyController class."""

    def test_create_controller(self) -> None:
        """Test creating a WBC."""
        engine = MockEngine()
        wbc = WholeBodyController(engine)

        assert wbc.engine is engine
        assert wbc.n_tasks == 0

    def test_controller_requires_robotics_capable(self) -> None:
        """Test controller validates engine type."""
        with pytest.raises(TypeError, match="RoboticsCapable"):
            WholeBodyController("not_an_engine")  # type: ignore

    def test_add_task(self) -> None:
        """Test adding tasks to controller."""
        engine = MockEngine()
        wbc = WholeBodyController(engine)

        task = create_posture_task(
            n_v=6,
            q_target=np.zeros(6),
            q_current=np.zeros(6),
            v_current=np.zeros(6),
        )

        wbc.add_task(task)

        assert wbc.n_tasks == 1
        assert wbc.get_task("posture") is not None

    def test_add_duplicate_task_raises(self) -> None:
        """Test adding duplicate task name raises."""
        engine = MockEngine()
        wbc = WholeBodyController(engine)

        task1 = create_posture_task(
            n_v=6,
            q_target=np.zeros(6),
            q_current=np.zeros(6),
            v_current=np.zeros(6),
        )
        task2 = create_posture_task(
            n_v=6,
            q_target=np.ones(6),
            q_current=np.zeros(6),
            v_current=np.zeros(6),
        )

        wbc.add_task(task1)

        with pytest.raises(ValueError, match="already exists"):
            wbc.add_task(task2)

    def test_remove_task(self) -> None:
        """Test removing tasks."""
        engine = MockEngine()
        wbc = WholeBodyController(engine)

        task = create_posture_task(
            n_v=6,
            q_target=np.zeros(6),
            q_current=np.zeros(6),
            v_current=np.zeros(6),
        )

        wbc.add_task(task)
        assert wbc.n_tasks == 1

        result = wbc.remove_task("posture")
        assert result is True
        assert wbc.n_tasks == 0

    def test_remove_nonexistent_task(self) -> None:
        """Test removing non-existent task returns False."""
        engine = MockEngine()
        wbc = WholeBodyController(engine)

        result = wbc.remove_task("nonexistent")
        assert result is False

    def test_clear_tasks(self) -> None:
        """Test clearing all tasks."""
        engine = MockEngine()
        wbc = WholeBodyController(engine)

        for i in range(3):
            task = Task(
                name=f"task_{i}",
                task_type=TaskType.SOFT,
                priority=1,
                jacobian=np.eye(6),
                target=np.zeros(6),
            )
            wbc.add_task(task)

        assert wbc.n_tasks == 3
        wbc.clear_tasks()
        assert wbc.n_tasks == 0

    def test_solve_no_tasks(self) -> None:
        """Test solve with no tasks returns failure."""
        engine = MockEngine()
        wbc = WholeBodyController(engine)

        solution = wbc.solve()

        assert solution.success is False
        assert "No tasks" in solution.status

    def test_solve_single_task(self) -> None:
        """Test solve with single posture task."""
        engine = MockEngine()
        wbc = WholeBodyController(engine)

        # Add posture task to stay at current position
        task = create_posture_task(
            n_v=6,
            q_target=np.zeros(6),
            q_current=np.zeros(6),
            v_current=np.zeros(6),
            gain_p=100.0,
            gain_d=20.0,
            weight=1.0,
        )
        wbc.add_task(task)

        solution = wbc.solve()

        assert solution.success
        assert solution.joint_accelerations is not None
        assert solution.joint_torques is not None
        assert len(solution.joint_accelerations) == 6
        assert len(solution.joint_torques) == 6

    def test_solve_multiple_tasks(self) -> None:
        """Test solve with multiple tasks."""
        engine = MockEngine()
        wbc = WholeBodyController(engine)

        # High priority task
        task1 = Task(
            name="high_priority",
            task_type=TaskType.SOFT,
            priority=10,
            jacobian=np.eye(6)[:3],  # First 3 DOFs
            target=np.zeros(3),
            weight=np.full(3, 10.0),
        )

        # Low priority task
        task2 = Task(
            name="low_priority",
            task_type=TaskType.SOFT,
            priority=1,
            jacobian=np.eye(6)[3:],  # Last 3 DOFs
            target=np.zeros(3),
            weight=np.ones(3),
        )

        wbc.add_task(task1)
        wbc.add_task(task2)

        solution = wbc.solve()

        assert solution.success
        assert "high_priority" in solution.task_errors
        assert "low_priority" in solution.task_errors

    def test_solve_with_config(self) -> None:
        """Test solve with custom configuration."""
        engine = MockEngine()
        config = WBCConfig(
            dt=0.002,
            regularization=1e-4,
            use_hierarchical=False,
        )
        wbc = WholeBodyController(engine, config=config)

        task = create_posture_task(
            n_v=6,
            q_target=np.zeros(6),
            q_current=np.zeros(6),
            v_current=np.zeros(6),
        )
        wbc.add_task(task)

        solution = wbc.solve()
        assert solution.success

    def test_solve_hierarchical_vs_weighted(self) -> None:
        """Test hierarchical and weighted modes produce results."""
        engine = MockEngine()

        # Hierarchical mode
        config_hqp = WBCConfig(use_hierarchical=True)
        wbc_hqp = WholeBodyController(engine, config=config_hqp)

        task = create_posture_task(
            n_v=6,
            q_target=np.zeros(6),
            q_current=np.zeros(6),
            v_current=np.zeros(6),
        )
        wbc_hqp.add_task(task)
        solution_hqp = wbc_hqp.solve()

        # Weighted mode
        config_weighted = WBCConfig(use_hierarchical=False)
        wbc_weighted = WholeBodyController(engine, config=config_weighted)
        wbc_weighted.add_task(task)
        solution_weighted = wbc_weighted.solve()

        assert solution_hqp.success
        assert solution_weighted.success

    def test_solve_with_contact_jacobians(self) -> None:
        """Test solve with contact Jacobians."""
        engine = MockEngine()
        wbc = WholeBodyController(engine)

        task = create_posture_task(
            n_v=6,
            q_target=np.zeros(6),
            q_current=np.zeros(6),
            v_current=np.zeros(6),
        )
        wbc.add_task(task)

        # Add contact Jacobians
        J_contact = np.random.randn(3, 6)
        wbc.set_contact_jacobians([J_contact])

        solution = wbc.solve()

        # May or may not succeed depending on constraints
        # Just verify it doesn't crash
        assert isinstance(solution.success, bool)

    def test_task_sorting_by_priority(self) -> None:
        """Test tasks are sorted by priority."""
        engine = MockEngine()
        wbc = WholeBodyController(engine)

        task_low = Task(
            name="low",
            task_type=TaskType.SOFT,
            priority=1,
            jacobian=np.eye(6),
            target=np.zeros(6),
        )
        task_high = Task(
            name="high",
            task_type=TaskType.SOFT,
            priority=10,
            jacobian=np.eye(6),
            target=np.zeros(6),
        )

        # Add low priority first
        wbc.add_task(task_low)
        wbc.add_task(task_high)

        # High priority should be first after sorting
        tasks = wbc.tasks
        assert tasks[0].name == "high"
        assert tasks[1].name == "low"


class TestWBCConfig:
    """Tests for WBCConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = WBCConfig()

        assert config.dt == 0.001
        assert config.regularization == 1e-6
        assert config.use_hierarchical is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        limits = np.array([100.0, 100.0, 50.0, 50.0, 20.0, 20.0])
        config = WBCConfig(
            dt=0.002,
            torque_limits=limits,
            use_hierarchical=False,
        )

        assert config.dt == 0.002
        assert_array_equal(config.torque_limits, limits)
        assert config.use_hierarchical is False


class TestWBCSolution:
    """Tests for WBCSolution dataclass."""

    def test_failed_solution(self) -> None:
        """Test failed solution defaults."""
        solution = WBCSolution(success=False, status="Failed")

        assert solution.success is False
        assert solution.joint_accelerations is None
        assert solution.cost == float("inf")

    def test_successful_solution(self) -> None:
        """Test successful solution."""
        qdd = np.array([1.0, 2.0, 3.0])
        tau = np.array([10.0, 20.0, 30.0])

        solution = WBCSolution(
            success=True,
            joint_accelerations=qdd,
            joint_torques=tau,
            cost=0.5,
            status="Optimal",
        )

        assert solution.success is True
        assert_array_equal(solution.joint_accelerations, qdd)
        assert_array_equal(solution.joint_torques, tau)
        assert solution.cost == 0.5


class TestIntegration:
    """Integration tests for control module."""

    def test_full_wbc_pipeline(self) -> None:
        """Test complete WBC pipeline from task creation to solve."""
        # Setup
        engine = MockEngine(n_v=7)
        config = WBCConfig(dt=0.001, regularization=1e-5)
        wbc = WholeBodyController(engine, config=config)

        # Create tasks
        posture_task = create_posture_task(
            n_v=7,
            q_target=np.zeros(7),
            q_current=0.1 * np.ones(7),
            v_current=np.zeros(7),
            gain_p=100.0,
            gain_d=20.0,
            weight=1.0,
        )

        wbc.add_task(posture_task)

        # Solve
        solution = wbc.solve()

        # Verify
        assert solution.success
        assert solution.joint_accelerations is not None
        assert len(solution.joint_accelerations) == 7
        assert "posture" in solution.task_errors

    def test_tracking_task_generates_acceleration(self) -> None:
        """Test that tracking error generates corrective acceleration."""
        engine = MockEngine(n_v=3)
        wbc = WholeBodyController(engine, WBCConfig(use_hierarchical=False))

        # Target is at 1.0, current is at 0.0 -> should accelerate positive
        task = create_posture_task(
            n_v=3,
            q_target=np.array([1.0, 0.0, 0.0]),
            q_current=np.array([0.0, 0.0, 0.0]),
            v_current=np.zeros(3),
            gain_p=100.0,
            gain_d=20.0,
        )
        wbc.add_task(task)

        solution = wbc.solve()

        assert solution.success
        # First joint should have positive acceleration toward target
        assert solution.joint_accelerations[0] > 0

    def test_solver_fallback(self) -> None:
        """Test solver fallback when scipy unavailable."""
        # Force use of nullspace solver
        solver = NullspaceQPSolver()
        engine = MockEngine()
        wbc = WholeBodyController(engine, solver=solver)

        task = create_posture_task(
            n_v=6,
            q_target=np.zeros(6),
            q_current=np.zeros(6),
            v_current=np.zeros(6),
        )
        wbc.add_task(task)

        solution = wbc.solve()
        assert solution.success
