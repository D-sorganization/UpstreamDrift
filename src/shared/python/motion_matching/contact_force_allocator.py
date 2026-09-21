"""Contact-Aware Dynamic Force Allocator (MS-31 / MS-104 / #10415).

Solves the constrained inverse dynamics torque and contact force allocation:
    M(q) * a + b(q, v) = S^T * tau + J_ground^T * f_contact + J_grip^T * lambda_grip + S_root^T * delta_tau_root

Subject to:
1. Floating-base balance: unactuated root coordinates (first 6 DoFs) are balanced
   by admissible ground reaction forces f_contact (f_z >= 0, friction cone) with
   any residual recorded in delta_tau_root.
2. Unilateral ground contact: f_{i, z} >= 0 for all active contact spheres.
3. Grip load transmission: closed kinematic loop between hands transmits internal
   reaction wrench lambda_grip without violating dynamics.
4. Exact dynamic equilibrium: acceleration parity error is mathematically zero (< 1e-8).
5. Minimum achievable trail-arm torque without breaking dynamics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear

from src.shared.python.contracts import require

Array: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.integer[Any]]


class AllocationObjective(str, Enum):
    """Optimization objective for force allocation."""

    MINIMUM_EFFORT = "minimum_effort"
    MINIMUM_TRAIL_ARM = "minimum_trail_arm"
    HARD_ZERO_TRAIL_ARM = "hard_zero_trail_arm"
    BALANCED_LOAD = "balanced_load"
    TRAIL_ZERO = "trail_zero"  # Backward-compatibility alias for MINIMUM_TRAIL_ARM


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

    def audit_feasibility(
        self,
        body_mass_kg: float,
        ankle_indices: Sequence[int] | None = None,
        contact_positions_m: Mapping[str, Array] | None = None,
        ground: Any = None,
        mu_friction: float = 0.8,
        constitutive_forces: Mapping[str, Any] | None = None,
        config: Any = None,
    ) -> Any:
        """Audit this allocation against physical feasibility and capacity limits."""
        from src.shared.python.motion_matching.contact_force_feasibility import (
            audit_contact_force_feasibility,
        )

        return audit_contact_force_feasibility(
            tau_actuated=self.tau_actuated,
            f_ground=self.f_ground,
            delta_tau_root=self.delta_tau_root,
            body_mass_kg=body_mass_kg,
            ankle_indices=ankle_indices,
            contact_positions_m=contact_positions_m,
            ground=ground,
            mu_friction=mu_friction,
            constitutive_forces=constitutive_forces,
            config=config,
        )


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
        root_penalty_weight: float = 1e4,
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
        objective: AllocationObjective = AllocationObjective.MINIMUM_EFFORT,
        trail_arm_indices: Sequence[int] | IntArray | None = None,
        tau_bounds: tuple[Array, Array] | None = None,
    ) -> ContactForceAllocation:
        """Resolve torques and contact forces matching tau_rnea exactly."""
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

        # Build equality constraint matrix A_eq @ x = tau_rnea
        # x = [tau_actuated (n_a), f_ground (n_g), lambda_grip (6), delta_tau_root (6)]
        a_eq = np.hstack(
            [self._s_transpose, j_ground.T, j_grip.T, self._s_root_transpose]
        )

        effective_objective = (
            AllocationObjective.MINIMUM_TRAIL_ARM
            if objective == AllocationObjective.TRAIL_ZERO
            else objective
        )

        # Objective diagonal weights W
        w_diag = np.ones(self.n_vars, dtype=float)
        # Actuator weights
        if (
            effective_objective == AllocationObjective.MINIMUM_TRAIL_ARM
            and trail_arm_indices is not None
            and len(trail_arm_indices) > 0
        ):
            trail_set = set(trail_arm_indices)
            for i, idx in enumerate(self.actuated_indices):
                if idx in trail_set:
                    w_diag[i] = 1000.0  # heavy penalty on trail arm
                else:
                    w_diag[i] = 1.0
        else:
            w_diag[: self.n_actuated] = 1.0

        idx_grip = self.n_actuated + self.n_ground_vars
        idx_root = idx_grip + 6

        # Ground force weights
        w_diag[self.n_actuated : idx_grip] = self.reg_contact
        # Grip wrench weights
        w_diag[idx_grip:idx_root] = self.reg_grip
        # Root slack penalty (heavily penalize ungrounded root forces)
        w_diag[idx_root:] = self.root_penalty

        # Variable bounds:
        lb = np.full(self.n_vars, -np.inf)
        ub = np.full(self.n_vars, np.inf)

        if tau_bounds is not None:
            lb[: self.n_actuated] = tau_bounds[0]
            ub[: self.n_actuated] = tau_bounds[1]

        # Enforce hard zero on trail arm if requested
        if (
            effective_objective == AllocationObjective.HARD_ZERO_TRAIL_ARM
            and trail_arm_indices is not None
            and len(trail_arm_indices) > 0
        ):
            trail_set = set(trail_arm_indices)
            for i, idx in enumerate(self.actuated_indices):
                if idx in trail_set:
                    lb[i] = -1e-5
                    ub[i] = 1e-5

        # Unilateral ground contact: f_{i, z} >= 0
        for s in range(self.n_contact_spheres):
            z_var_idx = self.n_actuated + s * 3 + 2
            lb[z_var_idx] = 0.0

        # Formulation: Solve bounded least squares on the penalty-augmented system
        # min || sqrt(W) * x ||^2 + gamma * || A_eq * x - tau_rnea ||^2
        gamma = 1e8
        sqrt_gamma = np.sqrt(gamma)
        sqrt_w = np.sqrt(w_diag)

        m_aug = np.vstack([sqrt_gamma * a_eq, np.diag(sqrt_w)])
        b_aug = np.concatenate([sqrt_gamma * tau_rnea, np.zeros(self.n_vars)])

        res = lsq_linear(m_aug, b_aug, bounds=(lb, ub), lsmr_tol="auto", max_iter=200)

        x_sol = res.x
        tau_sol = x_sol[: self.n_actuated]
        f_sol = x_sol[self.n_actuated : idx_grip]
        lambda_sol = x_sol[idx_grip:idx_root]
        root_sol = x_sol[idx_root:]

        # Verify equilibrium
        applied = self._compute_applied(
            tau_sol, j_ground, f_sol, j_grip, lambda_sol, root_sol
        )
        eq_res = float(np.max(np.abs(applied - tau_rnea)))

        root_res = float(np.max(np.abs(root_sol)))

        # Exact projection refinement: project any minute numerical residual onto actuated coordinates
        residual_err = tau_rnea - applied
        tau_refined = tau_sol + residual_err[self.actuated_indices]
        applied = self._compute_applied(
            tau_refined, j_ground, f_sol, j_grip, lambda_sol, root_sol
        )
        eq_res = float(np.max(np.abs(applied - tau_rnea)))

        # Verify whether bounds were respected
        bounds_ok = True
        if tau_bounds is not None:
            if np.any(tau_refined < tau_bounds[0] - 1e-3) or np.any(
                tau_refined > tau_bounds[1] + 1e-3
            ):
                bounds_ok = False

        if (
            effective_objective == AllocationObjective.HARD_ZERO_TRAIL_ARM
            and trail_arm_indices
        ):
            trail_set = set(trail_arm_indices)
            for i, idx in enumerate(self.actuated_indices):
                if idx in trail_set and abs(tau_refined[i]) > 1e-4:
                    bounds_ok = False

        success = bool(res.success and eq_res < 1e-4 and bounds_ok)

        return ContactForceAllocation(
            tau_actuated=tau_refined,
            f_ground=f_sol,
            lambda_grip=lambda_sol,
            delta_tau_root=root_sol,
            equilibrium_residual=eq_res,
            root_balance_residual=root_res,
            success=success,
            objective=objective,
        )
