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
import logging
import math
import warnings
from typing import Any, TypeAlias

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.loop_nullspace.qp_core import QPConstraints, solve_qp
from src.shared.python.loop_nullspace.solvers import ConicSettings, SolveStatus
from src.shared.python.loop_nullspace.zero_side_shim import (
    ZeroSideConsistency,
    check_zero_side_consistency,
)

logger = logging.getLogger(__name__)

# Clarabel tolerance for the scaled allocation QP (the legacy SLSQP used ftol=1e-8).
_SOLVER_TOL = 1e-9

_WORLD_X, _WORLD_Y, _WORLD_Z = np.eye(3)
_FRICTION_FACETS = 8  # facets of the inscribed friction pyramid per contact
_TRAIL_ARM_WEIGHT = 1000.0  # objective weight on trail-arm torques
_ACTUATOR_SLACK_WEIGHT = 1e6  # objective weight on actuator-limit slack

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
        warnings.warn(
            "ContactForceAllocator is a compatibility shim over the conic core; "
            "use src.shared.python.loop_nullspace (allocate_trajectory / "
            "qp_core.solve_qp) for new code (NSX-24, #12567).",
            DeprecationWarning,
            stacklevel=2,
        )
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

    def actuation_blocks(self) -> tuple[Array, Array]:
        """Return ``(S_actuated^T, S_root^T)``, each of shape ``(nv, .)``."""
        return self._s_transpose, self._s_root_transpose

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
        warnings.warn(
            "ContactForceAllocator.allocate is a legacy shim; prefer "
            "loop_nullspace.allocate_trajectory over the conic core.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        fs = feasible_set_from_frame(
            self,
            FrameInputs(
                tau_rnea=tau_rnea,
                j_ground=j_ground,
                j_grip=j_grip,
                objective=obj,
                trail_arm_indices=trail_arm_indices,
                tau_bounds=tau_bounds,
                grip_bounds=grip_bounds,
                contact_mask=contact_mask,
                ground_normal=ground_normal,
            ),
        )
        mask, n_hat, d_inv = fs.mask, fs.n_hat, fs.d_inv
        has_act_limits, n_total_vars = fs.has_actuator_slack, fs.n_vars_total
        idx_ground = self.n_actuated
        idx_grip = idx_ground + self.n_ground_vars
        idx_root = idx_grip + self.n_grip_vars

        trail_set: set[int] = (
            set(trail_arm_indices) if trail_arm_indices is not None else set()
        )
        if obj == AllocationObjective.HARD_ZERO_TRAIL and trail_set:
            zero_check = check_zero_side_consistency(
                tau_rnea,
                self.actuated_indices,
                list(trail_set),
                j_ground=j_ground,
                j_grip=j_grip,
                atol=atol,
            )
            if zero_check.status is ZeroSideConsistency.INCONSISTENT:
                logger.info(
                    "HARD_ZERO_TRAIL inconsistent: scaled residual %.3g",
                    zero_check.residual_scaled,
                )
                return ContactForceAllocation(
                    tau_actuated=np.zeros(self.n_actuated, dtype=np.float64),
                    f_ground=np.zeros(self.n_ground_vars, dtype=np.float64),
                    lambda_grip=np.zeros(self.n_grip_vars, dtype=np.float64),
                    delta_tau_root=np.zeros(self.n_root_vars, dtype=np.float64),
                    equilibrium_residual=zero_check.residual_norm,
                    root_balance_residual=zero_check.residual_norm,
                    success=False,
                    objective=obj,
                    is_physically_feasible=False,
                    feasibility_status=FeasibilityStatus.INFEASIBLE_HARD_ZERO_TRAIL,
                )

        # Convex QP in the scaled variables y (Hessian = I): exact conic solve
        # through the shared core (NSX-24).  Never falls back to another
        # optimizer; a missing backend raises ImportError (R11).
        res = solve_qp(
            sp.identity(n_total_vars, format="csr"),
            np.zeros(n_total_vars),
            fs.qp_constraints(),
            settings=ConicSettings(tol=_SOLVER_TOL),
        )
        solver_converged = res.status is SolveStatus.OPTIMAL
        if not solver_converged:
            logger.warning(
                "Contact force allocation not solved (%s %s); returning the "
                "zero iterate with success=False",
                res.status.value,
                res.reason,
            )
        if solver_converged and res.x is None:
            raise ValueError("optimal solve returned no solution vector")
        y_sol: np.ndarray = (
            res.x if solver_converged and res.x is not None else np.zeros(n_total_vars)
        )

        # Unscale solution: x = y * d_inv
        x_sol = y_sol * d_inv
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

        success = bool(solver_converged and eq_res < 1e-4 and bounds_satisfied)

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


@dataclass(frozen=True)
class FrameInputs:
    """Per-frame arguments of :meth:`ContactForceAllocator.allocate` (shapes checked there)."""

    tau_rnea: Array
    j_ground: Array
    j_grip: Array
    objective: AllocationObjective | str = AllocationObjective.MINIMUM_EFFORT
    trail_arm_indices: Sequence[int] | IntArray | None = None
    tau_bounds: tuple[Array, Array] | None = None
    grip_bounds: tuple[Array, Array] | None = None
    contact_mask: Sequence[bool] | NDArray[np.bool_] | None = None
    ground_normal: Array | None = None


@dataclass(frozen=True, eq=False)
class FrameFeasibleSet:
    """Scaled constraint set of one frame in the allocator's variable layout.

    ``x = [tau_actuated, f_ground, lambda_grip, delta_tau_root, (actuator slack)]``
    and the solver works in ``y = x / d_inv``, where the objective Hessian is the
    identity; ``a_ub y <= 0`` holds the unilateral and friction-pyramid rows.
    Not a ``loop_nullspace.FeasibleSet``: the legacy objective weights every
    variable, which ``allocate_trajectory`` does not (NSX-24).
    """

    a_eq: Array
    b_eq: Array
    a_ub: Array
    lo: Array
    hi: Array
    d_inv: Array
    mask: NDArray[np.bool_]
    n_hat: Array
    has_actuator_slack: bool

    @property
    def n_vars_total(self) -> int:
        """Scaled variable count, including any actuator slack."""
        return int(self.d_inv.size)

    def qp_constraints(self) -> QPConstraints:
        """Constraints in the form :func:`solve_qp` consumes."""
        has_rows = self.a_ub.shape[0] > 0
        return QPConstraints(
            A_eq=self.a_eq,
            b_eq=self.b_eq,
            A_ub=self.a_ub if has_rows else None,
            b_ub=np.zeros(self.a_ub.shape[0]) if has_rows else None,
            lo=self.lo,
            hi=self.hi,
        )


def _contact_mask(
    allocator: ContactForceAllocator,
    contact_mask: Sequence[bool] | NDArray[np.bool_] | None,
) -> NDArray[np.bool_]:
    if contact_mask is None:
        return np.ones(allocator.n_contact_spheres, dtype=bool)
    mask = np.asarray(contact_mask, dtype=bool)
    require(
        len(mask) == allocator.n_contact_spheres,
        "contact_mask length must match n_contact_spheres",
    )
    return mask


def _objective_weights(
    allocator: ContactForceAllocator,
    obj: AllocationObjective,
    trail_set: set[int],
    n_total_vars: int,
) -> Array:
    """Diagonal ``H`` of the objective ``1/2 x^T H x``."""
    n_act = allocator.n_actuated
    idx_grip = n_act + allocator.n_ground_vars
    idx_root = idx_grip + allocator.n_grip_vars
    h_diag = np.ones(n_total_vars, dtype=np.float64)
    if obj == AllocationObjective.MINIMUM_TRAIL_ARM:
        for i, idx in enumerate(allocator.actuated_indices):
            if idx in trail_set:
                h_diag[i] = _TRAIL_ARM_WEIGHT
    h_diag[n_act:idx_grip] = allocator.reg_contact
    h_diag[idx_grip:idx_root] = allocator.reg_grip
    h_diag[idx_root : idx_root + allocator.n_root_vars] = allocator.root_penalty
    h_diag[allocator.n_vars :] = _ACTUATOR_SLACK_WEIGHT
    return h_diag


def _variable_bounds(
    allocator: ContactForceAllocator,
    frame: FrameInputs,
    obj: AllocationObjective,
    trail_set: set[int],
    mask: NDArray[np.bool_],
    n_total_vars: int,
) -> tuple[Array, Array]:
    n_act = allocator.n_actuated
    idx_grip = n_act + allocator.n_ground_vars
    lb_x = np.full(n_total_vars, -np.inf)
    ub_x = np.full(n_total_vars, np.inf)
    if frame.tau_bounds is not None:
        lb_x[:n_act] = np.asarray(frame.tau_bounds[0], dtype=np.float64)
        ub_x[:n_act] = np.asarray(frame.tau_bounds[1], dtype=np.float64)
    if obj == AllocationObjective.HARD_ZERO_TRAIL:
        for i, idx in enumerate(allocator.actuated_indices):
            if idx in trail_set:
                lb_x[i] = 0.0
                ub_x[i] = 0.0
    for s in np.flatnonzero(~mask):
        s_idx = n_act + int(s) * 3
        lb_x[s_idx : s_idx + 3] = 0.0
        ub_x[s_idx : s_idx + 3] = 0.0
    if frame.grip_bounds is not None:
        idx_root = idx_grip + allocator.n_grip_vars
        lb_x[idx_grip:idx_root] = np.asarray(frame.grip_bounds[0], dtype=np.float64)
        ub_x[idx_grip:idx_root] = np.asarray(frame.grip_bounds[1], dtype=np.float64)
    return lb_x, ub_x


def _friction_rows(
    allocator: ContactForceAllocator,
    mask: NDArray[np.bool_],
    frame_dirs: tuple[Array, Array, Array],
    n_total_vars: int,
) -> Array:
    """Rows ``c`` of ``c x >= 0``: ``f.n >= 0`` and the friction pyramid."""
    n_hat, t1_hat, t2_hat = frame_dirs
    scale = math.cos(math.pi / _FRICTION_FACETS)
    rows: list[Array] = []
    for s in range(allocator.n_contact_spheres):
        if not mask[s]:
            continue
        s_idx = allocator.n_actuated + s * 3
        row_n = np.zeros(n_total_vars)
        row_n[s_idx : s_idx + 3] = n_hat
        rows.append(row_n)
        for k in range(_FRICTION_FACETS):
            theta = 2.0 * math.pi * k / _FRICTION_FACETS
            facet_dir = (
                scale * allocator.mu_friction * n_hat
                - math.cos(theta) * t1_hat
                - math.sin(theta) * t2_hat
            )
            row_f = np.zeros(n_total_vars)
            row_f[s_idx : s_idx + 3] = facet_dir
            rows.append(row_f)
    return np.array(rows) if rows else np.zeros((0, n_total_vars), dtype=np.float64)


def feasible_set_from_frame(
    allocator: ContactForceAllocator, frame: FrameInputs
) -> FrameFeasibleSet:
    """Assemble the scaled constraint set of one frame (the NSX-24 adapter seam).

    Both legacy public APIs take their per-frame constraints from here and pass
    them unchanged to :func:`solve_qp`.  Preconditions: shapes validated by the
    caller; ``ground_normal`` non-zero and ``contact_mask`` one entry per sphere
    (``ValueError`` otherwise).  Postcondition: ``a_eq @ y = tau_rnea`` is the
    only equality and the scaled Hessian is the identity.
    """
    obj = AllocationObjective.from_string(frame.objective)
    normal = frame.ground_normal
    n_hat, t1_hat, t2_hat = (
        (_WORLD_Z, _WORLD_X, _WORLD_Y) if normal is None else _get_tangent_basis(normal)
    )
    mask = _contact_mask(allocator, frame.contact_mask)
    trail_set = (
        set(frame.trail_arm_indices) if frame.trail_arm_indices is not None else set()
    )
    has_act_limits = (
        frame.tau_bounds is not None or obj == AllocationObjective.HARD_ZERO_TRAIL
    )
    n_total_vars = allocator.n_vars + (allocator.n_actuated if has_act_limits else 0)

    s_act, s_root = allocator.actuation_blocks()
    blocks = [s_act, frame.j_ground.T, frame.j_grip.T, s_root]
    if has_act_limits:
        blocks.append(s_act)
    a_eq = np.hstack(blocks)

    h_diag = _objective_weights(allocator, obj, trail_set, n_total_vars)
    lb_x, ub_x = _variable_bounds(allocator, frame, obj, trail_set, mask, n_total_vars)
    c_ineq = _friction_rows(allocator, mask, (n_hat, t1_hat, t2_hat), n_total_vars)

    # y = D x with D = sqrt(H): the quadratic Hessian becomes the identity.
    d_vec = np.sqrt(h_diag)
    d_inv = 1.0 / d_vec
    return FrameFeasibleSet(
        a_eq=a_eq * d_inv[np.newaxis, :],
        b_eq=np.asarray(frame.tau_rnea, dtype=np.float64),
        a_ub=-c_ineq * d_inv[np.newaxis, :],
        lo=lb_x * d_vec,
        hi=ub_x * d_vec,
        d_inv=d_inv,
        mask=mask,
        n_hat=n_hat,
        has_actuator_slack=has_act_limits,
    )
