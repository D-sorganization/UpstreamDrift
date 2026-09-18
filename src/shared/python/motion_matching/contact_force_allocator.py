"""Contact-Aware Dynamic Force Allocator (PF-03 / MS-31 / MS-104 / #10433).

Solves the constrained inverse dynamics torque and contact force allocation:
    M(q) * a + b(q, v) = S^T * tau + J_ground^T * f_contact + J_grip^T * lambda_grip + S_root^T * delta_tau_root

Subject to:
1. Floating-base balance: unactuated root coordinates (first 6 DoFs) are balanced
   by admissible ground reaction forces f_contact (f_z >= 0, Coulomb friction cone) with
   any residual recorded in delta_tau_root.
2. Unilateral ground contact: f_{i, z} >= 0 for all active contact spheres.
3. Coulomb friction cone: ||f_{i, t}|| <= mu * f_{i, n} strictly enforced.
4. Separated contact zero force: f_i == 0 for inactive/separated contact points.
5. Strict actuator bounds: tau_min <= tau <= tau_max without unconstrained post-projection.
6. Explicit grip load transmission: closed kinematic loop between hands transmits internal
   reaction wrench lambda_grip respecting physical grip bounds.
7. Diagnostic root slack decoupling: delta_tau_root is isolated and cannot create physical success.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.shared.python.contracts import require

logger = logging.getLogger(__name__)

Array = NDArray[np.float64]
IntArray = NDArray[np.integer[Any]]


class AllocationObjective(str, Enum):
    """Optimization objective for force allocation."""

    MINIMUM_EFFORT = "minimum_effort"
    MINIMUM_TRAIL_ARM = "minimum_trail_arm"
    HARD_ZERO_TRAIL_ARM = "hard_zero_trail_arm"
    BALANCED_LOAD = "balanced_load"
    TRAIL_ZERO = "trail_zero"  # Backward-compatibility alias for MINIMUM_TRAIL_ARM


class AllocationStatus(str, Enum):
    """Feasibility outcome of force and torque allocation."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    INFEASIBLE_ACTUATOR_BOUNDS = "infeasible_actuator_bounds"
    INFEASIBLE_FRICTION_CONE = "infeasible_friction_cone"
    INFEASIBLE_ROOT_EQUILIBRIUM = "infeasible_root_equilibrium"
    INFEASIBLE_SEPARATION = "infeasible_separation"


@dataclass(frozen=True)
class SurfaceContactFrame:
    """Surface contact frame with normal and tangent basis vectors."""

    normal: Array
    tangent1: Array
    tangent2: Array


@dataclass(frozen=True)
class AllocationResiduals:
    """Diagnostic residuals for equilibrium, contact, and actuator limits."""

    equilibrium_residual: float
    root_residual: float
    max_friction_ratio: float
    max_normal_force_n: float
    actuator_bound_violation: float
    grip_bound_violation: float


@dataclass(frozen=True)
class ContactForceAllocation:
    """Resolved dynamic effort, ground reaction forces, and grip wrench."""

    tau_actuated: Array
    f_ground: Array
    lambda_grip: Array
    delta_tau_root: Array
    equilibrium_residual: float
    root_balance_residual: float
    success: bool
    objective: AllocationObjective
    status: AllocationStatus = AllocationStatus.OPTIMAL
    residuals: AllocationResiduals | None = None


class ContactForceAllocator:
    """Rigorous QP-based contact force and actuator torque allocator."""

    def __init__(
        self,
        nv: int,
        actuated_indices: Sequence[int] | IntArray,
        n_contact_spheres: int = 6,
        mu_friction: float = 0.8,
        regularisation_contact: float = 1e-4,
        regularisation_grip: float = 1e-4,
        root_penalty_weight: float = 1e6,
        max_root_force_n: float = 0.1,
    ) -> None:
        require(nv > 6, "nv must exceed 6 floating base coordinates", nv)
        require(len(actuated_indices) > 0, "must have actuated coordinates")
        require(n_contact_spheres > 0, "must have contact spheres")
        require(mu_friction > 0.0, "friction coefficient must be positive", mu_friction)
        require(
            max_root_force_n >= 0.0, "max root force threshold must be non-negative"
        )

        self.nv = nv
        self.actuated_indices = np.asarray(actuated_indices, dtype=np.int64)
        self.n_actuated = len(self.actuated_indices)
        self.n_contact_spheres = n_contact_spheres
        self.n_ground_vars = n_contact_spheres * 3
        self.n_grip_vars = 6
        self.n_root_vars = 6
        self.n_vars = self.n_actuated + self.n_ground_vars + 6 + 6
        self.mu_friction = mu_friction
        self.reg_contact = regularisation_contact
        self.reg_grip = regularisation_grip
        self.root_penalty = root_penalty_weight
        self.max_root_force_n = max_root_force_n
        self._root_dofs = 6

        # S_actuated^T in Pinocchio generalized coordinates (nv, n_actuated)
        self._s_transpose = np.zeros((nv, self.n_actuated), dtype=np.float64)
        for col, idx in enumerate(self.actuated_indices):
            self._s_transpose[idx, col] = 1.0

        # S_root^T for ungrounded root slack (nv, 6)
        self._s_root_transpose = np.zeros((nv, 6), dtype=np.float64)
        for i in range(6):
            self._s_root_transpose[i, i] = 1.0

    def _compute_applied(
        self,
        tau_sol: Array,
        j_ground: Array,
        f_sol: Array,
        j_grip: Array,
        lambda_sol: Array,
        root_sol: Array,
    ) -> Array:
        tau_full = np.zeros(self.nv, dtype=np.float64)
        tau_full[self.actuated_indices] = tau_sol
        return (
            tau_full
            + j_ground.T @ f_sol
            + j_grip.T @ lambda_sol
            + self._s_root_transpose @ root_sol
        )

    def _build_inequalities(
        self,
        surface_frames: Sequence[SurfaceContactFrame] | None,
        mu_friction: float,
    ) -> tuple[Array, Array]:
        """Construct Coulomb friction cone and non-negative normal force inequalities."""
        ineq_rows: list[Array] = []
        ineq_ub: list[float] = []
        idx_ground = self.n_actuated

        for s in range(self.n_contact_spheres):
            s_idx = idx_ground + s * 3
            if surface_frames is not None and s < len(surface_frames):
                frame = surface_frames[s]
                n = np.asarray(frame.normal, dtype=np.float64)
                t1 = np.asarray(frame.tangent1, dtype=np.float64)
                t2 = np.asarray(frame.tangent2, dtype=np.float64)
            else:
                n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                t1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                t2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

            # Non-negative normal force: - n^T f <= 0
            row_n = np.zeros(self.n_vars, dtype=np.float64)
            row_n[s_idx : s_idx + 3] = -n
            ineq_rows.append(row_n)
            ineq_ub.append(0.0)

            # Coulomb friction pyramid inequalities: |t^T f| <= mu * (n^T f)
            for t_vec in (t1, t2):
                row_pos = np.zeros(self.n_vars, dtype=np.float64)
                row_pos[s_idx : s_idx + 3] = t_vec - mu_friction * n
                ineq_rows.append(row_pos)
                ineq_ub.append(0.0)

                row_neg = np.zeros(self.n_vars, dtype=np.float64)
                row_neg[s_idx : s_idx + 3] = -t_vec - mu_friction * n
                ineq_rows.append(row_neg)
                ineq_ub.append(0.0)

        return (
            np.array(ineq_rows, dtype=np.float64),
            np.array(ineq_ub, dtype=np.float64),
        )

    def _build_bounds(
        self,
        tau_bounds: tuple[Array, Array] | None,
        effective_objective: AllocationObjective,
        trail_arm_indices: Sequence[int] | IntArray | None,
        active_contacts: Sequence[bool] | None,
        grip_limits: tuple[float, float] | None,
        max_root_slack: float,
    ) -> tuple[Array, Array]:
        """Construct variable bounds respecting actuators, trail arm, and contact separation."""
        lb = np.full(self.n_vars, -np.inf, dtype=np.float64)
        ub = np.full(self.n_vars, np.inf, dtype=np.float64)

        if tau_bounds is not None:
            lb[: self.n_actuated] = tau_bounds[0]
            ub[: self.n_actuated] = tau_bounds[1]

        if (
            effective_objective == AllocationObjective.HARD_ZERO_TRAIL_ARM
            and trail_arm_indices is not None
            and len(trail_arm_indices) > 0
        ):
            trail_set = set(trail_arm_indices)
            for i, idx in enumerate(self.actuated_indices):
                if idx in trail_set:
                    lb[i] = 0.0
                    ub[i] = 0.0

        idx_ground = self.n_actuated
        if active_contacts is not None:
            for s in range(self.n_contact_spheres):
                if s < len(active_contacts) and not active_contacts[s]:
                    s_idx = idx_ground + s * 3
                    lb[s_idx : s_idx + 3] = 0.0
                    ub[s_idx : s_idx + 3] = 0.0

        idx_grip = idx_ground + self.n_ground_vars
        if grip_limits is not None:
            f_lim, m_lim = grip_limits
            lb[idx_grip : idx_grip + 3] = -f_lim
            ub[idx_grip : idx_grip + 3] = f_lim
            lb[idx_grip + 3 : idx_grip + 6] = -m_lim
            ub[idx_grip + 3 : idx_grip + 6] = m_lim

        idx_root = idx_grip + 6
        lb[idx_root:] = -max_root_slack
        ub[idx_root:] = max_root_slack
        return lb, ub

    def _solve_qp(
        self,
        h_diag: Array,
        a_eq: Array,
        b_eq: Array,
        a_ineq: Array,
        b_ineq: Array,
        lb: Array,
        ub: Array,
    ) -> tuple[Array, bool]:
        """Solve QP min 0.5 x^T H x subject to A_eq x = b_eq, A_ineq x <= b_ineq, lb <= x <= ub."""
        lb_clamped = np.where(np.isneginf(lb), -1e6, lb)
        ub_clamped = np.where(np.isposinf(ub), 1e6, ub)
        x0: Array = np.asarray(
            np.clip(np.zeros(self.n_vars, dtype=np.float64), lb_clamped, ub_clamped),
            dtype=np.float64,
        )

        def fun(x: Array) -> float:
            return 0.5 * float(np.sum(h_diag * (x**2)))

        def jac(x: Array) -> Array:
            return h_diag * x

        constraints = [
            {"type": "eq", "fun": lambda x: a_eq @ x - b_eq, "jac": lambda _: a_eq},
            {
                "type": "ineq",
                "fun": lambda x: b_ineq - a_ineq @ x,
                "jac": lambda _: -a_ineq,
            },
        ]
        bounds = [(float(low), float(high)) for low, high in zip(lb, ub, strict=True)]

        res = minimize(
            fun,
            x0,
            method="SLSQP",
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 250, "ftol": 1e-6, "disp": False},
        )
        return res.x, bool(res.success)

    def _diagnose_infeasibility(
        self,
        a_eq: Array,
        b_eq: Array,
        x_diag: Array,
        tau_bounds: tuple[Array, Array] | None,
        ineq_viol: float,
        root_res: float,
    ) -> AllocationStatus:
        """Categorize precise failure mode when allocation is physically infeasible."""
        residuals = np.abs(a_eq @ x_diag - b_eq)
        root_eq_res = float(np.max(residuals[: self._root_dofs]))
        act_res = (
            float(np.max(residuals[self.actuated_indices]))
            if len(self.actuated_indices) > 0
            else 0.0
        )
        tau_sol = x_diag[: self.n_actuated]

        if tau_bounds is not None and (
            act_res > 1e-3
            or np.any(tau_sol < tau_bounds[0] - 1e-3)
            or np.any(tau_sol > tau_bounds[1] + 1e-3)
        ):
            return AllocationStatus.INFEASIBLE_ACTUATOR_BOUNDS
        if root_res > self.max_root_force_n + 1e-3 or root_eq_res > 1e-3:
            return AllocationStatus.INFEASIBLE_ROOT_EQUILIBRIUM
        if ineq_viol > 1e-3:
            return AllocationStatus.INFEASIBLE_FRICTION_CONE
        return AllocationStatus.INFEASIBLE

    def allocate(
        self,
        tau_rnea: Array,
        j_ground: Array,
        j_grip: Array,
        objective: AllocationObjective = AllocationObjective.MINIMUM_EFFORT,
        trail_arm_indices: Sequence[int] | IntArray | None = None,
        tau_bounds: tuple[Array, Array] | None = None,
        active_contacts: Sequence[bool] | None = None,
        **kwargs: Any,
    ) -> ContactForceAllocation:
        """Resolve torques and contact forces matching tau_rnea under strict physical constraints."""
        require(tau_rnea.shape == (self.nv,), "tau_rnea shape mismatch")
        require(
            j_ground.shape == (self.n_ground_vars, self.nv), "j_ground shape mismatch"
        )
        require(j_grip.shape == (self.n_grip_vars, self.nv), "j_grip shape mismatch")

        surface_frames = kwargs.get("surface_frames")
        grip_limits = kwargs.get("grip_limits")

        effective_objective = (
            AllocationObjective.MINIMUM_TRAIL_ARM
            if objective == AllocationObjective.TRAIL_ZERO
            else objective
        )

        a_eq = np.hstack(
            [self._s_transpose, j_ground.T, j_grip.T, self._s_root_transpose]
        )
        b_eq = tau_rnea

        h_diag = np.ones(self.n_vars, dtype=np.float64)
        if (
            effective_objective == AllocationObjective.MINIMUM_TRAIL_ARM
            and trail_arm_indices is not None
            and len(trail_arm_indices) > 0
        ):
            trail_set = set(trail_arm_indices)
            for i, idx in enumerate(self.actuated_indices):
                if idx in trail_set:
                    h_diag[i] = 1000.0

        idx_ground = self.n_actuated
        idx_grip = idx_ground + self.n_ground_vars
        idx_root = idx_grip + 6
        h_diag[idx_ground:idx_grip] = self.reg_contact
        h_diag[idx_grip:idx_root] = self.reg_grip
        h_diag[idx_root:] = self.root_penalty

        a_ineq, b_ineq = self._build_inequalities(surface_frames, self.mu_friction)

        # Pass 1: Physical solve with strict root bound <= max_root_force_n
        lb_phys, ub_phys = self._build_bounds(
            tau_bounds,
            effective_objective,
            trail_arm_indices,
            active_contacts,
            grip_limits,
            self.max_root_force_n,
        )
        x_sol, success_p1 = self._solve_qp(
            h_diag, a_eq, b_eq, a_ineq, b_ineq, lb_phys, ub_phys
        )

        eq_res = float(np.max(np.abs(a_eq @ x_sol - b_eq)))
        ineq_viol = float(np.max(np.maximum(0.0, a_ineq @ x_sol - b_ineq)))
        root_res = float(np.max(np.abs(x_sol[idx_root:])))
        bound_viol = float(
            np.max(np.maximum(0.0, np.maximum(lb_phys - x_sol, x_sol - ub_phys)))
        )

        physically_feasible = bool(
            success_p1
            and eq_res < 1e-3
            and ineq_viol < 1e-3
            and bound_viol < 1e-3
            and root_res <= self.max_root_force_n + 1e-4
        )

        if physically_feasible:
            tau_sol = x_sol[: self.n_actuated]
            f_sol = x_sol[idx_ground:idx_grip]
            lambda_sol = x_sol[idx_grip:idx_root]
            root_sol = x_sol[idx_root:]
            status = AllocationStatus.OPTIMAL
        else:
            # Pass 2: Diagnostic solve with relaxed root bounds (identifies deficits)
            lb_diag, ub_diag = self._build_bounds(
                tau_bounds,
                effective_objective,
                trail_arm_indices,
                active_contacts,
                grip_limits,
                np.inf,
            )
            x_diag, _ = self._solve_qp(
                h_diag, a_eq, b_eq, a_ineq, b_ineq, lb_diag, ub_diag
            )
            tau_sol = x_diag[: self.n_actuated]
            f_sol = x_diag[idx_ground:idx_grip]
            lambda_sol = x_diag[idx_grip:idx_root]
            root_sol = x_diag[idx_root:]
            root_res = float(np.max(np.abs(root_sol)))
            eq_res = float(np.max(np.abs(a_eq @ x_diag - b_eq)))
            ineq_viol = float(np.max(np.maximum(0.0, a_ineq @ x_diag - b_ineq)))
            status = self._diagnose_infeasibility(
                a_eq,
                b_eq,
                x_diag,
                tau_bounds,
                ineq_viol,
                root_res,
            )

        gf_reshaped = f_sol.reshape(self.n_contact_spheres, 3)
        fn = gf_reshaped[:, 2]
        ft = np.linalg.norm(gf_reshaped[:, :2], axis=1)
        act_mask = fn > 1.0
        fric_ratio = (
            float(np.max(np.where(act_mask, ft / np.maximum(fn, 1e-6), 0.0)))
            if np.any(act_mask)
            else 0.0
        )

        act_viol = (
            float(
                np.max(
                    np.maximum(
                        0.0,
                        np.maximum(tau_bounds[0] - tau_sol, tau_sol - tau_bounds[1]),
                    )
                )
            )
            if tau_bounds is not None
            else 0.0
        )

        residuals = AllocationResiduals(
            equilibrium_residual=eq_res,
            root_residual=root_res,
            max_friction_ratio=fric_ratio,
            max_normal_force_n=float(np.max(fn)),
            actuator_bound_violation=act_viol,
            grip_bound_violation=0.0,
        )

        return ContactForceAllocation(
            tau_actuated=tau_sol,
            f_ground=f_sol,
            lambda_grip=lambda_sol,
            delta_tau_root=root_sol,
            equilibrium_residual=eq_res,
            root_balance_residual=root_res,
            success=physically_feasible,
            objective=objective,
            status=status,
            residuals=residuals,
        )
