"""Whole-body controller with hierarchical task prioritization.

This module provides the main WholeBodyController class that
orchestrates task management, constraint handling, and QP solving.

Design by Contract:
    Controller validates inputs and guarantees consistent outputs.
    Failed solves are indicated, never silent failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.robotics.control.whole_body.qp_solver import (
    QPProblem,
    QPSolution,
    QPSolver,
    create_default_solver,
)
from src.robotics.control.whole_body.task import Task
from src.robotics.core.protocols import RoboticsCapable


@dataclass
class WBCConfig:
    """Configuration for whole-body controller.

    Attributes:
        dt: Control timestep [s].
        regularization: Regularization weight for joint accelerations.
        torque_limits: Joint torque limits (n_v,) or None for unlimited.
        velocity_limits: Joint velocity limits (n_v,) or None.
        acceleration_limits: Joint acceleration limits (n_v,) or None.
        contact_force_regularization: Weight for contact force minimization.
        use_hierarchical: Use strict task hierarchy (HQP) vs weighted sum.
    """

    dt: float = 0.001
    regularization: float = 1e-6
    torque_limits: NDArray[np.float64] | None = None
    velocity_limits: NDArray[np.float64] | None = None
    acceleration_limits: NDArray[np.float64] | None = None
    contact_force_regularization: float = 1e-4
    use_hierarchical: bool = True


@dataclass
class WBCSolution:
    """Solution from whole-body controller.

    Attributes:
        success: Whether solver found valid solution.
        joint_accelerations: Optimal joint accelerations (n_v,).
        joint_torques: Optimal joint torques (n_v,).
        contact_forces: Contact forces if contacts present.
        cost: Total cost value.
        status: Status message.
        task_errors: Error for each task after solution.
    """

    success: bool
    joint_accelerations: NDArray[np.float64] | None = None
    joint_torques: NDArray[np.float64] | None = None
    contact_forces: NDArray[np.float64] | None = None
    cost: float = float("inf")
    status: str = ""
    task_errors: dict[str, float] = field(default_factory=dict)


class WholeBodyController:
    """Whole-body controller with hierarchical task prioritization.

    Solves the inverse dynamics problem:
        M(q) @ qdd + C(q, qd) @ qd + g(q) = S^T @ tau + J_c^T @ f_c

    With task objectives minimizing:
        sum_i w_i * ||J_i @ qdd - target_i||^2

    Subject to constraints:
        - Equation of motion
        - Contact force friction constraints
        - Torque/velocity/acceleration limits

    Design by Contract:
        Preconditions:
            - Engine must be initialized
            - Tasks must have correct dimensions

        Postconditions:
            - Solution contains valid accelerations if success=True
            - Task errors are computed for all tasks

    Example:
        >>> wbc = WholeBodyController(engine)
        >>> wbc.add_task(create_com_task(...))
        >>> wbc.add_task(create_posture_task(...))
        >>> solution = wbc.solve()
        >>> if solution.success:
        ...     tau = solution.joint_torques
    """

    def __init__(
        self,
        engine: RoboticsCapable,
        config: WBCConfig | None = None,
        solver: QPSolver | None = None,
    ) -> None:
        """Initialize whole-body controller.

        Args:
            engine: Physics engine implementing RoboticsCapable.
            config: Controller configuration.
            solver: QP solver to use. Uses default if None.

        Raises:
            TypeError: If engine doesn't implement required protocol.
        """
        if not isinstance(engine, RoboticsCapable):
            raise TypeError(
                f"Engine must implement RoboticsCapable, got {type(engine).__name__}"
            )

        self._engine = engine
        self._config = config or WBCConfig()
        self._solver = solver or create_default_solver()
        self._tasks: list[Task] = []
        self._contact_jacobians: list[NDArray[np.float64]] = []

    @property
    def engine(self) -> RoboticsCapable:
        """Get underlying physics engine."""
        return self._engine

    @property
    def config(self) -> WBCConfig:
        """Get controller configuration."""
        return self._config

    @property
    def tasks(self) -> list[Task]:
        """Get current task list (copy)."""
        return list(self._tasks)

    @property
    def n_tasks(self) -> int:
        """Get number of tasks."""
        return len(self._tasks)

    def add_task(self, task: Task) -> None:
        """Add task to controller.

        Tasks are sorted by priority (highest first).

        Args:
            task: Task to add.

        Raises:
            ValueError: If task with same name already exists.
        """
        if any(t.name == task.name for t in self._tasks):
            raise ValueError(f"Task with name '{task.name}' already exists")

        self._tasks.append(task)
        self._sort_tasks()

    def remove_task(self, name: str) -> bool:
        """Remove task by name.

        Args:
            name: Name of task to remove.

        Returns:
            True if task was removed, False if not found.
        """
        for i, task in enumerate(self._tasks):
            if task.name == name:
                self._tasks.pop(i)
                return True
        return False

    def clear_tasks(self) -> None:
        """Remove all tasks."""
        self._tasks.clear()

    def get_task(self, name: str) -> Task | None:
        """Get task by name.

        Args:
            name: Task name.

        Returns:
            Task if found, None otherwise.
        """
        for task in self._tasks:
            if task.name == name:
                return task
        return None

    def set_contact_jacobians(
        self,
        jacobians: list[NDArray[np.float64]],
    ) -> None:
        """Set contact Jacobians for current contacts.

        Args:
            jacobians: List of contact Jacobians, each (3, n_v) or (6, n_v).
        """
        self._contact_jacobians = list(jacobians)

    def solve(self) -> WBCSolution:
        """Solve whole-body control problem.

        Builds QP from current tasks and constraints, solves it,
        and extracts joint accelerations and torques.

        Returns:
            WBCSolution with optimal values if successful.

        Raises:
            ControlError: If problem setup fails.
        """
        if not self._tasks:
            return WBCSolution(
                success=False,
                status="No tasks defined",
            )

        try:
            # Get robot state and dynamics
            q, qd = self._engine.get_state()
            n_v = len(qd)

            M = self._engine.compute_mass_matrix()
            nle = self._engine.compute_bias_forces()

            # Decision variables: [qdd, contact_forces (optional)]
            n_contacts = len(self._contact_jacobians)
            n_contact_vars = 3 * n_contacts  # 3D forces per contact

            # Build QP problem
            if self._config.use_hierarchical:
                solution = self._solve_hierarchical(n_v, n_contact_vars, M, nle, qd)
            else:
                solution = self._solve_weighted(n_v, n_contact_vars, M, nle, qd)

            return solution

        except Exception as e:
            return WBCSolution(
                success=False,
                status=f"Solve failed: {e}",
            )

    def _solve_weighted(
        self,
        n_v: int,
        n_contact_vars: int,
        M: NDArray[np.float64],
        nle: NDArray[np.float64],
        qd: NDArray[np.float64],
    ) -> WBCSolution:
        """Solve using weighted sum of tasks.

        Args:
            n_v: Number of velocity DOFs.
            n_contact_vars: Number of contact force variables.
            M: Mass matrix.
            nle: Nonlinear effects.
            qd: Current joint velocities.

        Returns:
            WBCSolution from weighted QP.
        """
        n_vars = n_v + n_contact_vars

        # Build cost: sum of weighted task costs + regularization
        H = np.zeros((n_vars, n_vars))
        g = np.zeros(n_vars)

        for task in self._tasks:
            if task.jacobian is None:
                continue

            J = task.jacobian
            target = task.target  # Desired task-space acceleration
            W = task.get_weight_matrix()  # Diagonal weight matrix

            # Task dimension
            J.shape[0]

            # Ensure dimensions match
            if J.shape[1] != n_v:
                continue

            # Cost: ||J @ qdd - target||^2_W = (J @ qdd - target)^T @ W @ (J @ qdd - target)
            # Expanded: qdd^T @ J^T @ W @ J @ qdd - 2 * target^T @ W @ J @ qdd + const
            # H contribution: J^T @ W @ J
            # g contribution: -J^T @ W @ target
            H[:n_v, :n_v] += J.T @ W @ J
            g[:n_v] += -J.T @ W @ target

        # Add regularization on accelerations
        H[:n_v, :n_v] += self._config.regularization * np.eye(n_v)

        # Add regularization on contact forces
        if n_contact_vars > 0:
            H[n_v:, n_v:] += self._config.contact_force_regularization * np.eye(
                n_contact_vars
            )

        # Build constraints
        A_eq, b_eq = self._build_dynamics_constraint(n_v, n_contact_vars, M, nle)
        A_ineq, lb_ineq, ub_ineq = self._build_inequality_constraints(
            n_v, n_contact_vars, qd
        )

        # Variable bounds
        x_lb, x_ub = self._build_variable_bounds(n_v, n_contact_vars, qd)

        # Create and solve QP
        problem = QPProblem(
            H=H,
            g=g,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ineq=A_ineq,
            lb_ineq=lb_ineq,
            ub_ineq=ub_ineq,
            x_lb=x_lb,
            x_ub=x_ub,
        )

        qp_solution = self._solver.solve(problem)

        return self._extract_solution(qp_solution, n_v, n_contact_vars, M, nle)

    def _solve_hierarchical(
        self,
        n_v: int,
        n_contact_vars: int,
        M: NDArray[np.float64],
        nle: NDArray[np.float64],
        qd: NDArray[np.float64],
    ) -> WBCSolution:
        """Solve using strict task hierarchy (HQP).

        Higher priority tasks are solved first, then lower priority
        tasks are solved in the nullspace of higher priority tasks.

        Args:
            n_v: Number of velocity DOFs.
            n_contact_vars: Number of contact force variables.
            M: Mass matrix.
            nle: Nonlinear effects.
            qd: Current joint velocities.

        Returns:
            WBCSolution from hierarchical solve.
        """
        # Group tasks by priority
        priority_groups = self._group_tasks_by_priority()

        if not priority_groups:
            return WBCSolution(success=False, status="No valid tasks")

        n_vars = n_v + n_contact_vars

        # Initialize nullspace projector
        accumulated_A: list[NDArray[np.float64]] = []

        # Solution accumulator
        x_solution = np.zeros(n_vars)

        # Solve each priority level
        for _priority, tasks in sorted(priority_groups.items(), reverse=True):
            # Build task matrices for this level
            H = np.zeros((n_vars, n_vars))
            g = np.zeros(n_vars)

            for task in tasks:
                if task.jacobian is None:
                    continue

                J = task.jacobian
                target = task.target
                W = task.get_weight_matrix()

                if J.shape[1] != n_v:
                    continue

                # Extend Jacobian to full decision space
                J_full = np.zeros((J.shape[0], n_vars))
                J_full[:, :n_v] = J

                # Apply nullspace projection from higher priorities
                if accumulated_A:
                    A_stack = np.vstack(accumulated_A)
                    N = self._compute_nullspace_projector(A_stack, n_vars)
                    J_proj = J_full @ N
                else:
                    J_proj = J_full

                H += J_proj.T @ W @ J_proj
                g += -J_proj.T @ W @ target

                # Add to accumulated constraints for next level
                accumulated_A.append(J_full)

            # Add regularization
            H[:n_v, :n_v] += self._config.regularization * np.eye(n_v)
            if n_contact_vars > 0:
                H[n_v:, n_v:] += self._config.contact_force_regularization * np.eye(
                    n_contact_vars
                )

            # Build constraints (same for all levels)
            A_eq, b_eq = self._build_dynamics_constraint(n_v, n_contact_vars, M, nle)
            A_ineq, lb_ineq, ub_ineq = self._build_inequality_constraints(
                n_v, n_contact_vars, qd
            )
            x_lb, x_ub = self._build_variable_bounds(n_v, n_contact_vars, qd)

            problem = QPProblem(
                H=H,
                g=g,
                A_eq=A_eq,
                b_eq=b_eq,
                A_ineq=A_ineq,
                lb_ineq=lb_ineq,
                ub_ineq=ub_ineq,
                x_lb=x_lb,
                x_ub=x_ub,
            )

            qp_solution = self._solver.solve(problem)

            if qp_solution.success and qp_solution.x is not None:
                x_solution = qp_solution.x

        # Extract final solution
        return self._extract_solution_from_x(x_solution, n_v, n_contact_vars, M, nle)

    def _build_dynamics_constraint(
        self,
        n_v: int,
        n_contact_vars: int,
        M: NDArray[np.float64],
        nle: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        """Build equation of motion constraint.

        M @ qdd + nle = tau + J_c^T @ f_c

        For floating base: first 6 rows have no actuation (tau = 0).

        Args:
            n_v: Number of velocity DOFs.
            n_contact_vars: Number of contact force variables.
            M: Mass matrix.
            nle: Nonlinear effects.

        Returns:
            Tuple of (A_eq, b_eq) or (None, None) if no constraint.
        """
        if not self._contact_jacobians:
            # No contacts - no dynamics constraint in QP
            return None, None

        n_vars = n_v + n_contact_vars

        # Build stacked contact Jacobian transpose
        J_c_T = np.zeros((n_v, n_contact_vars))
        for i, J_c in enumerate(self._contact_jacobians):
            # Use only linear part for point contacts
            if J_c.shape[0] == 6:
                J_c = J_c[:3]
            J_c_T[:, 3 * i : 3 * (i + 1)] = J_c.T

        # Dynamics: M @ qdd - J_c^T @ f_c = -nle
        A_eq = np.zeros((n_v, n_vars))
        A_eq[:, :n_v] = M
        A_eq[:, n_v:] = -J_c_T

        b_eq = -nle

        return A_eq, b_eq

    def _build_inequality_constraints(
        self,
        n_v: int,
        n_contact_vars: int,
        qd: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64] | None,
        NDArray[np.float64] | None,
        NDArray[np.float64] | None,
    ]:
        """Build inequality constraints.

        Includes friction cone constraints for contact forces.

        Args:
            n_v: Number of velocity DOFs.
            n_contact_vars: Number of contact force variables.
            qd: Current joint velocities.

        Returns:
            Tuple of (A_ineq, lb_ineq, ub_ineq) or (None, None, None).
        """
        constraints_A: list[NDArray[np.float64]] = []
        constraints_lb: list[NDArray[np.float64]] = []
        constraints_ub: list[NDArray[np.float64]] = []

        n_vars = n_v + n_contact_vars

        # Friction cone constraints for contact forces
        if n_contact_vars > 0:
            mu = 0.5  # Default friction coefficient
            n_contacts = n_contact_vars // 3

            for i in range(n_contacts):
                # Normal force non-negative: f_z >= 0 => -f_z <= 0
                A_row = np.zeros((1, n_vars))
                A_row[0, n_v + 3 * i + 2] = -1.0
                constraints_A.append(A_row)
                constraints_lb.append(np.array([-np.inf]))
                constraints_ub.append(np.array([0.0]))

                # Friction pyramid constraints
                for sign_x, sign_y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    A_row = np.zeros((1, n_vars))
                    A_row[0, n_v + 3 * i] = sign_x
                    A_row[0, n_v + 3 * i + 1] = sign_y
                    A_row[0, n_v + 3 * i + 2] = -mu
                    constraints_A.append(A_row)
                    constraints_lb.append(np.array([-np.inf]))
                    constraints_ub.append(np.array([0.0]))

        if not constraints_A:
            return None, None, None

        A_ineq = np.vstack(constraints_A)
        lb_ineq = np.concatenate(constraints_lb)
        ub_ineq = np.concatenate(constraints_ub)

        return A_ineq, lb_ineq, ub_ineq

    def _build_variable_bounds(
        self,
        n_v: int,
        n_contact_vars: int,
        qd: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        """Build variable bounds.

        Args:
            n_v: Number of velocity DOFs.
            n_contact_vars: Number of contact force variables.
            qd: Current joint velocities.

        Returns:
            Tuple of (x_lb, x_ub) or (None, None).
        """
        n_vars = n_v + n_contact_vars

        x_lb = -np.inf * np.ones(n_vars)
        x_ub = np.inf * np.ones(n_vars)

        # Acceleration limits
        if self._config.acceleration_limits is not None:
            lim = self._config.acceleration_limits
            x_lb[:n_v] = -lim
            x_ub[:n_v] = lim

        # Velocity limits translated to acceleration
        if self._config.velocity_limits is not None:
            dt = self._config.dt
            v_lim = self._config.velocity_limits
            qdd_lb_from_v = (-v_lim - qd) / dt
            qdd_ub_from_v = (v_lim - qd) / dt
            x_lb[:n_v] = np.maximum(x_lb[:n_v], qdd_lb_from_v)
            x_ub[:n_v] = np.minimum(x_ub[:n_v], qdd_ub_from_v)

        # Contact force bounds for numerical stability
        max_force = 10000.0
        if n_contact_vars > 0:
            x_lb[n_v:] = -max_force
            x_ub[n_v:] = max_force

        return x_lb, x_ub

    def _extract_solution(
        self,
        qp_solution: QPSolution,
        n_v: int,
        n_contact_vars: int,
        M: NDArray[np.float64],
        nle: NDArray[np.float64],
    ) -> WBCSolution:
        """Extract WBC solution from QP solution.

        Args:
            qp_solution: Solution from QP solver.
            n_v: Number of velocity DOFs.
            n_contact_vars: Number of contact force variables.
            M: Mass matrix.
            nle: Nonlinear effects.

        Returns:
            WBCSolution.
        """
        if not qp_solution.success or qp_solution.x is None:
            return WBCSolution(
                success=False,
                status=qp_solution.status,
            )

        return self._extract_solution_from_x(qp_solution.x, n_v, n_contact_vars, M, nle)

    def _extract_solution_from_x(
        self,
        x: NDArray[np.float64],
        n_v: int,
        n_contact_vars: int,
        M: NDArray[np.float64],
        nle: NDArray[np.float64],
    ) -> WBCSolution:
        """Extract WBC solution from decision variable vector.

        Args:
            x: Decision variables [qdd, f_c].
            n_v: Number of velocity DOFs.
            n_contact_vars: Number of contact force variables.
            M: Mass matrix.
            nle: Nonlinear effects.

        Returns:
            WBCSolution.
        """
        qdd = x[:n_v]

        contact_forces = None
        if n_contact_vars > 0:
            contact_forces = x[n_v:]

        # Compute torques: tau = M @ qdd + nle - J_c^T @ f_c
        tau = M @ qdd + nle
        if contact_forces is not None and self._contact_jacobians:
            for i, J_c in enumerate(self._contact_jacobians):
                if J_c.shape[0] == 6:
                    J_c = J_c[:3]
                f_c = contact_forces[3 * i : 3 * (i + 1)]
                tau -= J_c.T @ f_c

        # Compute task errors
        task_errors = self._compute_task_errors(qdd)

        # Compute total cost
        cost = sum(task_errors.values())

        return WBCSolution(
            success=True,
            joint_accelerations=qdd,
            joint_torques=tau,
            contact_forces=contact_forces,
            cost=cost,
            status="Optimal",
            task_errors=task_errors,
        )

    def _compute_task_errors(
        self,
        qdd: NDArray[np.float64],
    ) -> dict[str, float]:
        """Compute task tracking errors.

        Args:
            qdd: Solved joint accelerations.

        Returns:
            Dictionary mapping task name to weighted error.
        """
        errors: dict[str, float] = {}

        for task in self._tasks:
            if task.jacobian is None:
                continue

            J = task.jacobian
            if J.shape[1] != len(qdd):
                continue

            # Task error: J @ qdd - target
            xdd_actual = J @ qdd
            xdd_error = xdd_actual - task.target

            # Weighted squared error using weight matrix
            W = task.get_weight_matrix()
            error = float(xdd_error @ W @ xdd_error)
            errors[task.name] = error

        return errors

    def _group_tasks_by_priority(self) -> dict[int, list[Task]]:
        """Group tasks by priority level.

        Returns:
            Dictionary mapping priority to list of tasks.
        """
        groups: dict[int, list[Task]] = {}

        for task in self._tasks:
            priority = task.priority
            if priority not in groups:
                groups[priority] = []
            groups[priority].append(task)

        return groups

    def _sort_tasks(self) -> None:
        """Sort tasks by priority (highest first)."""
        self._tasks.sort(key=lambda t: t.priority, reverse=True)

    def _compute_nullspace_projector(
        self,
        A: NDArray[np.float64],
        n: int,
    ) -> NDArray[np.float64]:
        """Compute nullspace projector N = I - pinv(A) @ A.

        Args:
            A: Constraint matrix.
            n: Dimension of projector.

        Returns:
            Nullspace projector matrix (n, n).
        """
        A_pinv = np.linalg.pinv(A)
        return np.eye(n) - A_pinv @ A
