"""Contact-Aware Dynamic Force Allocator (MS-31 / MS-104 / PF-03 / #10433).

Solves the constrained inverse dynamics torque and contact force allocation:
    M(q) * a + b(q, v) = S^T * tau + J_ground^T * f_contact + J_grip^T * lambda_grip + S_root^T * delta_tau_root

Subject to:
1. Exact dynamic equilibrium: acceleration parity error is mathematically zero (< 1e-4).
2. Friction cone enforcement: 8-faceted polyhedral friction pyramid on active contact points.
3. Unilateral ground contact: non-negative normal contact forces (f_n >= 0) along arbitrary
   surface normals (supports rotated / inclined ground frames).
4. Contact separation: separated or airborne contact spheres exert strictly zero force.
5. Actuator bounds: lb <= tau <= ub enforced without unconstrained post-projection.
6. Floating-base root balance: unactuated root coordinates (first 6 DoFs) balanced by
   admissible ground reaction forces; diagnostic root slack delta_tau_root is strictly
   isolated and cannot create false physical success.
7. Trail-arm reduction: supports both soft minimization (MINIMUM_TRAIL_ARM) and hard-zero
   trail mode (HARD_ZERO_TRAIL) which rejects infeasible dynamics cleanly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.shared.python.contracts import require

Array: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.integer[Any]]


class AllocationObjective(str, Enum):
    """Optimization objective for force allocation."""

    MINIMUM_EFFORT = "minimum_effort"
    MINIMUM_TRAIL_ARM = "minimum_trail_arm"
    HARD_ZERO_TRAIL = "hard_zero_trail"
    BALANCED_LOAD = "balanced_load"

    @classmethod
    def from_string(cls, val: str | AllocationObjective) -> AllocationObjective:
        """Resolve string or enum with backwards compatibility for legacy names."""
        if isinstance(val, cls):
            return val
        s = str(val).lower()
        if s in (
            "trail_zero",
            "soft_trail_zero",
            "minimum_trail",
            "soft_trail_reduction",
        ):
            return cls.MINIMUM_TRAIL_ARM
        if s in ("hard_zero_trail", "hard_trail_zero", "zero_trail"):
            return cls.HARD_ZERO_TRAIL
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown allocation objective: {val}")


class FeasibilityStatus(str, Enum):
    """Feasibility state of the dynamic force allocation."""

    FEASIBLE = "feasible"
    INFEASIBLE_FRICTION_CONE = "infeasible_friction_cone"
    INFEASIBLE_ACTUATOR_BOUNDS = "infeasible_actuator_bounds"
    INFEASIBLE_ROOT_BALANCE = "infeasible_root_balance"
    INFEASIBLE_GROUND_UNILATERAL = "infeasible_ground_unilateral"
    INFEASIBLE_SEPARATION = "infeasible_separation"
    INFEASIBLE_HARD_ZERO_TRAIL = "infeasible_hard_zero_trail"
    UNSUPPORTED = "unsupported"


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
    objective: AllocationObjective = AllocationObjective.MINIMUM_EFFORT
    is_physically_feasible: bool = True
    feasibility_status: FeasibilityStatus = FeasibilityStatus.FEASIBLE
    dual_residuals: float = 0.0
    friction_violations: float = 0.0
    actuator_violations: float = 0.0
    root_slack_norm: float = 0.0
    max_friction_ratio: float = 0.0


def _get_tangent_basis(normal: Array) -> tuple[Array, Array, Array]:
    """Compute orthonormal coordinate frame (n, t1, t2) for a given surface normal."""
    n = np.asarray(normal, dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    require(n_norm > 1e-8, "Surface normal cannot be zero vector")
    n = n / n_norm

    # Choose helper vector not parallel to n
    u = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])

    t1 = u - np.dot(u, n) * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return n, t1, t2


def verify_torque_and_rate_bounds(
    tau_trajectory: Array,
    dt: float,
    tau_bounds: tuple[Array, Array],
    rate_bounds: tuple[Array, Array] | float | None = None,
    atol: float = 1e-4,
) -> tuple[bool, dict[str, Any]]:
    """Verify that torque trajectory and its discrete rate satisfy limits.

    Parameters:
        tau_trajectory: Array of shape (T, n_actuated) or (T, nv).
        dt: Time step in seconds (must be positive).
        tau_bounds: (lb, ub) torque limits.
        rate_bounds: (lb_rate, ub_rate) or max absolute rate scalar.
        atol: Numerical tolerance.

    Returns:
        (is_valid, diagnostics_dict)
    """
    require(dt > 0.0, "dt must be positive", dt)
    tau_arr = np.asarray(tau_trajectory, dtype=np.float64)
    require(tau_arr.ndim == 2, "tau_trajectory must be 2D", tau_arr.shape)

    lb_tau, ub_tau = np.asarray(tau_bounds[0]), np.asarray(tau_bounds[1])
    lower_violation = np.maximum(0.0, lb_tau - tau_arr)
    upper_violation = np.maximum(0.0, tau_arr - ub_tau)
    max_tau_violation = float(max(np.max(lower_violation), np.max(upper_violation)))
    is_torque_bounded = max_tau_violation <= atol

    is_rate_bounded = True
    max_rate_violation = 0.0
    if rate_bounds is not None and len(tau_arr) > 1:
        dtau_dt = np.diff(tau_arr, axis=0) / dt
        if isinstance(rate_bounds, tuple):
            lb_rate, ub_rate = np.asarray(rate_bounds[0]), np.asarray(rate_bounds[1])
            rate_low_viol = np.maximum(0.0, lb_rate - dtau_dt)
            rate_up_viol = np.maximum(0.0, dtau_dt - ub_rate)
            max_rate_violation = float(max(np.max(rate_low_viol), np.max(rate_up_viol)))
        else:
            max_rate = float(rate_bounds)
            max_rate_violation = float(
                np.max(np.maximum(0.0, np.abs(dtau_dt) - max_rate))
            )
        is_rate_bounded = max_rate_violation <= atol

    is_valid = bool(is_torque_bounded and is_rate_bounded)
    diagnostics = {
        "is_valid": is_valid,
        "is_torque_bounded": is_torque_bounded,
        "is_rate_bounded": is_rate_bounded,
        "max_tau_violation": max_tau_violation,
        "max_rate_violation": max_rate_violation,
    }
    return is_valid, diagnostics


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
    ) -> None:
        require(nv > 6, "nv must exceed 6 floating base coordinates", nv)
        require(len(actuated_indices) > 0, "must have actuated coordinates")
        require(n_contact_spheres > 0, "must have contact spheres")
        require(mu_friction > 0.0, "friction coefficient must be positive", mu_friction)

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
        self._root_dofs = 6

        # S_actuated^T in Pinocchio generalized coordinates (nv, n_actuated)
        self._s_transpose = np.zeros((nv, self.n_actuated))
        for col, idx in enumerate(self.actuated_indices):
            self._s_transpose[idx, col] = 1.0

        # S_root^T for ungrounded root slack (nv, 6)
        self._s_root_transpose = np.zeros((nv, 6))
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
        tau_full = np.zeros(self.nv)
        tau_full[self.actuated_indices] = tau_sol
        return (
            tau_full
            + j_ground.T @ f_sol
            + j_grip.T @ lambda_sol
            + self._s_root_transpose @ root_sol
        )

    def allocate(
        self,
        tau_rnea: Array,
        j_ground: Array,
        j_grip: Array,
        objective: AllocationObjective | str = AllocationObjective.MINIMUM_EFFORT,
        trail_arm_indices: Sequence[int] | IntArray | None = None,
        tau_bounds: tuple[Array, Array] | None = None,
        grip_bounds: tuple[Array, Array] | None = None,
        contact_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
        ground_normal: Array | None = None,
        atol: float = 1e-4,
    ) -> ContactForceAllocation:
        """Resolve torques and contact forces matching tau_rnea exactly under constraints."""
        obj = AllocationObjective.from_string(objective)
        require(
            tau_rnea.shape == (self.nv,),
            "tau_rnea shape mismatch",
            (tau_rnea.shape, self.nv),
        )
        require(
            j_ground.shape == (self.n_ground_vars, self.nv),
            "j_ground shape mismatch",
            (j_ground.shape, (self.n_ground_vars, self.nv)),
        )
        require(
            j_grip.shape == (self.n_grip_vars, self.nv),
            "j_grip shape mismatch",
            (j_grip.shape, (self.n_grip_vars, self.nv)),
        )

        idx_ground = self.n_actuated
        idx_grip = idx_ground + self.n_ground_vars
        idx_root = idx_grip + self.n_grip_vars

        # Parse ground normal and tangent frame
        if ground_normal is None:
            n_hat = np.array([0.0, 0.0, 1.0])
            t1_hat = np.array([1.0, 0.0, 0.0])
            t2_hat = np.array([0.0, 1.0, 0.0])
        else:
            n_hat, t1_hat, t2_hat = _get_tangent_basis(ground_normal)

        # Contact separation mask
        if contact_mask is not None:
            mask = np.asarray(contact_mask, dtype=bool)
            require(
                len(mask) == self.n_contact_spheres,
                "contact_mask length must match n_contact_spheres",
            )
        else:
            mask = np.ones(self.n_contact_spheres, dtype=bool)

        # Check if actuator bounds or hard zero trail require actuator slack variables
        has_act_limits = (
            tau_bounds is not None or obj == AllocationObjective.HARD_ZERO_TRAIL
        )
        n_act_slack = self.n_actuated if has_act_limits else 0
        n_total_vars = self.n_vars + n_act_slack

        # Build equality matrix: A_eq @ x = tau_rnea
        # x = [tau_actuated, f_ground, lambda_grip, delta_tau_root, (delta_tau_act)]
        blocks = [
            self._s_transpose,
            j_ground.T,
            j_grip.T,
            self._s_root_transpose,
        ]
        if has_act_limits:
            blocks.append(self._s_transpose)
        a_eq = np.hstack(blocks)

        # Objective diagonal weights H_diag
        h_diag = np.ones(n_total_vars, dtype=np.float64)

        # Actuator weighting
        trail_set: set[int] = (
            set(trail_arm_indices) if trail_arm_indices is not None else set()
        )
        if obj == AllocationObjective.MINIMUM_TRAIL_ARM and len(trail_set) > 0:
            for i, idx in enumerate(self.actuated_indices):
                if idx in trail_set:
                    h_diag[i] = 1000.0
                else:
                    h_diag[i] = 1.0
        else:
            h_diag[: self.n_actuated] = 1.0

        # Ground force weights
        h_diag[idx_ground:idx_grip] = self.reg_contact
        # Grip wrench weights
        h_diag[idx_grip:idx_root] = self.reg_grip
        # Root slack penalty
        h_diag[idx_root : idx_root + self.n_root_vars] = self.root_penalty
        # Actuator slack penalty (heavily penalize violating actuator limits)
        if has_act_limits:
            h_diag[self.n_vars :] = 1e6

        # Variable bounds (lb, ub)
        lb_x = np.full(n_total_vars, -np.inf)
        ub_x = np.full(n_total_vars, np.inf)

        # Apply actuator bounds
        if tau_bounds is not None:
            lb_x[: self.n_actuated] = np.asarray(tau_bounds[0], dtype=np.float64)
            ub_x[: self.n_actuated] = np.asarray(tau_bounds[1], dtype=np.float64)

        # Hard-zero trail mode: force trail arm coordinates to strictly 0
        if obj == AllocationObjective.HARD_ZERO_TRAIL and len(trail_set) > 0:
            for i, idx in enumerate(self.actuated_indices):
                if idx in trail_set:
                    lb_x[i] = 0.0
                    ub_x[i] = 0.0

        # Separated contacts exert strictly zero force
        for s in range(self.n_contact_spheres):
            if not mask[s]:
                s_idx = idx_ground + s * 3
                lb_x[s_idx : s_idx + 3] = 0.0
                ub_x[s_idx : s_idx + 3] = 0.0

        # Grip bounds if specified
        if grip_bounds is not None:
            lb_x[idx_grip:idx_root] = np.asarray(grip_bounds[0], dtype=np.float64)
            ub_x[idx_grip:idx_root] = np.asarray(grip_bounds[1], dtype=np.float64)

        # Polyhedral friction pyramid & non-negative normal constraints
        # 8 facets: theta_k = 2 * pi * k / 8
        k_facets = 8
        scale = math.cos(math.pi / k_facets)
        c_ineq_rows: list[Array] = []

        for s in range(self.n_contact_spheres):
            if not mask[s]:
                continue
            s_idx = idx_ground + s * 3
            # Non-negative normal force: f_s . n_hat >= 0
            row_n = np.zeros(n_total_vars)
            row_n[s_idx : s_idx + 3] = n_hat
            c_ineq_rows.append(row_n)

            # Inscribed friction pyramid: scale * mu * (f_s . n) - (cos(th) t1 + sin(th) t2) . f_s >= 0
            for k in range(k_facets):
                theta = 2.0 * math.pi * k / k_facets
                facet_dir = (
                    scale * self.mu_friction * n_hat
                    - math.cos(theta) * t1_hat
                    - math.sin(theta) * t2_hat
                )
                row_f = np.zeros(n_total_vars)
                row_f[s_idx : s_idx + 3] = facet_dir
                c_ineq_rows.append(row_f)

        c_ineq = (
            np.array(c_ineq_rows)
            if c_ineq_rows
            else np.zeros((0, n_total_vars), dtype=np.float64)
        )

        # Scale variables so that quadratic Hessian becomes identity (condition number = 1.0)
        # y = D * x  where D = sqrt(H_diag)
        d_vec = np.sqrt(h_diag)
        d_inv = 1.0 / d_vec

        a_eq_scaled = a_eq * d_inv[np.newaxis, :]
        c_ineq_scaled = (
            c_ineq * d_inv[np.newaxis, :]
            if len(c_ineq) > 0
            else np.zeros((0, n_total_vars))
        )

        lb_y = lb_x * d_vec
        ub_y = ub_x * d_vec
        bounds_y = list(zip(lb_y, ub_y, strict=True))

        constraints: list[dict[str, Any]] = [
            {
                "type": "eq",
                "fun": lambda y: a_eq_scaled @ y - tau_rnea,
                "jac": lambda y: a_eq_scaled,
            }
        ]
        if len(c_ineq) > 0:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda y: c_ineq_scaled @ y,
                    "jac": lambda y: c_ineq_scaled,
                }
            )

        # Initial point
        y0 = np.zeros(n_total_vars)

        res = minimize(  # type: ignore[call-overload]
            fun=lambda y: 0.5 * np.sum(y**2),
            x0=y0,
            jac=lambda y: y,
            method="SLSQP",
            bounds=bounds_y,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8},
        )

        # Unscale solution: x = y * d_inv
        x_sol = res.x * d_inv
        tau_sol = x_sol[: self.n_actuated]
        f_sol = x_sol[idx_ground:idx_grip]
        lambda_sol = x_sol[idx_grip:idx_root]
        root_sol = x_sol[idx_root : idx_root + self.n_root_vars]
        act_slack_sol = x_sol[self.n_vars :] if has_act_limits else np.zeros(0)

        # Residuals and violations
        applied = self._compute_applied(
            tau_sol, j_ground, f_sol, j_grip, lambda_sol, root_sol
        )
        eq_res = float(np.max(np.abs(applied - tau_rnea)))
        root_res = float(np.max(np.abs(root_sol)))
        act_slack_res = (
            float(np.max(np.abs(act_slack_sol))) if len(act_slack_sol) > 0 else 0.0
        )

        # Check friction cone violations and measure Coulomb friction ratio
        f_reshaped = f_sol.reshape(self.n_contact_spheres, 3)
        friction_viol_max = 0.0
        max_friction_ratio = 0.0
        for s in range(self.n_contact_spheres):
            fx = float(f_sol[s * 3])
            fy = float(f_sol[s * 3 + 1])
            fz = float(f_sol[s * 3 + 2])
            f_tan = float(np.hypot(fx, fy))
            if fz > 1e-4:
                ratio = f_tan / fz
                if ratio > max_friction_ratio:
                    max_friction_ratio = ratio
            if not mask[s]:
                continue
            fn_s = float(f_reshaped[s] @ n_hat)
            ft_vec = f_reshaped[s] - fn_s * n_hat
            ft_s = float(np.linalg.norm(ft_vec))
            viol_s = max(0.0, ft_s - self.mu_friction * max(0.0, fn_s))
            if viol_s > friction_viol_max:
                friction_viol_max = viol_s

        # Feasibility assessment
        is_physically_feasible = True
        feasibility_status = FeasibilityStatus.FEASIBLE

        # Actuator violations
        if act_slack_res > 1e-3:
            is_physically_feasible = False
            if (
                obj == AllocationObjective.HARD_ZERO_TRAIL
                and trail_arm_indices is not None
            ):
                trail_set = set(trail_arm_indices)
                trail_slack = [
                    abs(act_slack_sol[col])
                    for col, idx in enumerate(self.actuated_indices)
                    if idx in trail_set
                ]
                if trail_slack and max(trail_slack) > 1e-3:
                    feasibility_status = FeasibilityStatus.INFEASIBLE_HARD_ZERO_TRAIL
                else:
                    feasibility_status = FeasibilityStatus.INFEASIBLE_ACTUATOR_BOUNDS
            else:
                feasibility_status = FeasibilityStatus.INFEASIBLE_ACTUATOR_BOUNDS

        # Root balance violations (diagnostic slack cannot create physical success)
        if root_res > 0.01:
            is_physically_feasible = False
            if feasibility_status == FeasibilityStatus.FEASIBLE:
                # Determine primary cause of root ungroundedness
                fn_vals = [
                    float(f_reshaped[s] @ n_hat)
                    for s in range(self.n_contact_spheres)
                    if mask[s]
                ]
                total_fn = sum(fn_vals) if fn_vals else 0.0
                if tau_rnea[2] < -1.0 or (total_fn < 1.0 and abs(tau_rnea[2]) > 10.0):
                    feasibility_status = FeasibilityStatus.INFEASIBLE_ROOT_BALANCE
                elif abs(root_sol[0]) > 1.0 or abs(root_sol[1]) > 1.0:
                    feasibility_status = FeasibilityStatus.INFEASIBLE_FRICTION_CONE
                else:
                    feasibility_status = FeasibilityStatus.INFEASIBLE_ROOT_BALANCE

        # Overall equilibrium error check
        if eq_res > 1e-3:
            is_physically_feasible = False
            if feasibility_status == FeasibilityStatus.FEASIBLE:
                feasibility_status = FeasibilityStatus.INFEASIBLE_ROOT_BALANCE

        bounds_satisfied = True
        if tau_bounds is not None:
            bounds_satisfied = bool(
                np.all(tau_sol >= tau_bounds[0] - 1e-5)
                and np.all(tau_sol <= tau_bounds[1] + 1e-5)
            )

        success = bool(is_physically_feasible and eq_res < 1e-4 and bounds_satisfied)

        return ContactForceAllocation(
            tau_actuated=tau_sol,
            f_ground=f_sol,
            lambda_grip=lambda_sol,
            delta_tau_root=root_sol,
            equilibrium_residual=eq_res,
            root_balance_residual=root_res,
            success=success,
            objective=obj,
            is_physically_feasible=is_physically_feasible,
            feasibility_status=feasibility_status,
            dual_residuals=eq_res,
            friction_violations=friction_viol_max,
            actuator_violations=act_slack_res,
            root_slack_norm=root_res,
            max_friction_ratio=max_friction_ratio,
        )

    def is_friction_feasible(self, allocation: ContactForceAllocation) -> bool:
        """Check whether resolved ground reaction forces obey Coulomb friction cone."""
        if (
            not allocation.is_physically_feasible
            and allocation.feasibility_status
            == FeasibilityStatus.INFEASIBLE_FRICTION_CONE
        ):
            return False
        return bool(
            allocation.max_friction_ratio <= self.mu_friction + 1e-3
            and allocation.friction_violations < 1e-3
        )
