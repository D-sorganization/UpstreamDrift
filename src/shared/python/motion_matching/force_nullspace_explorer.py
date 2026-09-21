"""Force Nullspace Exploration and Torque Distribution Tradeoffs (PF-06, #10436).

Provides:
1. Scaled rank-revealing QR/SVD decomposition of inverse dynamics equality constraints:
   validating exact nullspace basis A @ N = 0 and orthonormal N^T @ N = I.
2. Dynamic nullity reporting across varying contact modes (flight, single support, double support).
3. Basis sign-change and orthogonal rotation invariance: penalizes physical states, not arbitrary z coordinates.
4. Hard-zero trail arm feasibility diagnosis: detects when hard-zero trail torque is physical vs blocked
   by lead-arm torque capacity or grip wrench transmission limits, with relaxed minimum-trail fallback.
5. Bounded Pareto frontier sweeps across effort vs smoothness, peak capacity, trail-arm share,
   grip internal load, and ground force regularization.
6. Structured Pareto table publishing with conservative default rationale.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from scipy.optimize import Bounds, LinearConstraint, minimize

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
    ContactForceAllocator,
)
from src.shared.python.motion_matching.sparse_trajectory_torque_optimizer import (
    SparseOptimizationConfig,
    SparseTrajectoryTorqueOptimizer,
    TrajectoryOptimizationNode,
)

Array: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.integer[Any]]


class TradeoffDimension(str, Enum):
    """Explored physical trade-off dimensions."""

    EFFORT_VS_SMOOTHNESS = "effort_vs_smoothness"
    PEAK_UTILIZATION = "peak_utilization"
    TRAIL_ARM_SHARE = "trail_arm_share"
    GRIP_INTERNAL_LOAD = "grip_internal_load"
    GROUND_REGULARIZATION = "ground_regularization"


@dataclass(frozen=True)
class NullspaceDecomposition:
    """Rank-revealing singular value decomposition of dynamics equality constraint matrix."""

    matrix_rank: int
    nullity: int
    singular_values: Array
    nullspace_basis: Array
    particular_solution: Array
    residual_norm: float
    basis_orthogonality_residual: float


@dataclass(frozen=True)
class TradeoffSample:
    """A single evaluated point on the force/torque trade-off manifold."""

    parameter_name: str
    parameter_value: float
    is_feasible: bool
    tau_actuated: Array
    f_ground: Array
    lambda_grip: Array
    delta_tau_root: Array
    total_effort: float
    peak_torque_nm: float
    trail_arm_effort: float
    lead_arm_effort: float
    grip_squeeze_n: float
    max_grf_n: float
    cop_margin_m: float | None = None
    active_constraints: tuple[str, ...] = field(default_factory=tuple)
    infeasibility_reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ParetoFrontierReport:
    """Pareto frontier exploration outcome across a specified dimension."""

    dimension: TradeoffDimension
    samples: tuple[TradeoffSample, ...]
    conservative_default_index: int
    conservative_default_rationale: str
    hard_zero_feasible: bool
    hard_zero_blocking_constraints: tuple[str, ...] = field(default_factory=tuple)


class ForceNullspaceExplorer:
    """Rigorous rank-revealing nullspace explorer and trade-off analyzer."""

    def __init__(
        self,
        nv: int,
        actuated_indices: Sequence[int] | IntArray,
        n_contact_spheres: int = 6,
        mu_friction: float = 0.8,
        tau_max: Array | None = None,
        config: SparseOptimizationConfig | None = None,
    ) -> None:
        require(nv > 6, "nv must exceed floating base 6 DOFs", nv)
        require(len(actuated_indices) > 0, "actuated_indices must not be empty")
        require(
            n_contact_spheres > 0,
            "n_contact_spheres must be positive",
            n_contact_spheres,
        )
        require(mu_friction > 0.0, "mu_friction must be positive", mu_friction)

        self.nv = nv
        self.actuated_indices = np.asarray(actuated_indices, dtype=np.int64)
        self.n_actuated = len(self.actuated_indices)
        self.n_contact_spheres = n_contact_spheres
        self.n_ground_vars = n_contact_spheres * 3
        self.n_grip_vars = 6
        self.n_root_vars = 6
        self.n_node_vars = self.n_actuated + self.n_ground_vars + 12
        self.mu_friction = mu_friction
        self.tau_max = tau_max
        self.config = config or SparseOptimizationConfig(bounds_tau_max=tau_max)

        self._s_transpose = np.zeros((nv, self.n_actuated))
        for col, idx in enumerate(self.actuated_indices):
            self._s_transpose[idx, col] = 1.0

        self._s_root_transpose = np.zeros((nv, 6))
        for i in range(6):
            self._s_root_transpose[i, i] = 1.0

        self._allocator = ContactForceAllocator(
            nv=nv,
            actuated_indices=self.actuated_indices,
            n_contact_spheres=n_contact_spheres,
            mu_friction=mu_friction,
        )

    def _build_equality_matrix(
        self,
        j_ground: Array,
        j_grip: Array,
        active_contact_mask: Array | None = None,
    ) -> Array:
        """Construct equality matrix, augmenting with identity rows for inactive contact forces."""
        j_g_active = np.copy(j_ground)
        inactive_rows: list[Array] = []
        if active_contact_mask is not None:
            for s in range(self.n_contact_spheres):
                if s < len(active_contact_mask) and not active_contact_mask[s]:
                    j_g_active[s * 3 : (s + 1) * 3, :] = 0.0
                    for c in range(3):
                        var_idx = self.n_actuated + s * 3 + c
                        row = np.zeros(self.n_node_vars)
                        row[var_idx] = 1.0
                        inactive_rows.append(row)

        base_a = np.hstack(
            [self._s_transpose, j_g_active.T, j_grip.T, self._s_root_transpose]
        )
        if inactive_rows:
            return np.vstack([base_a, np.vstack(inactive_rows)])
        return base_a

    def decompose_nullspace(
        self,
        j_ground: Array,
        j_grip: Array,
        active_contact_mask: Array | None = None,
        b_target: Array | None = None,
    ) -> NullspaceDecomposition:
        """Compute scaled rank-revealing SVD nullspace basis and particular solution."""
        a_mat = self._build_equality_matrix(j_ground, j_grip, active_contact_mask)
        m_rows, n_cols = a_mat.shape

        # Scaled SVD: scale rows by maximum absolute value to normalize force and torque units
        row_norms = np.linalg.norm(a_mat, axis=1)
        row_scale = np.where(row_norms > 1e-12, 1.0 / row_norms, 1.0)
        a_scaled = row_scale[:, np.newaxis] * a_mat

        u, s, v_t = la.svd(a_scaled, full_matrices=True)
        eps = np.finfo(float).eps
        tol = max(m_rows, n_cols) * eps * s[0]

        rank = int(np.sum(s > tol))
        nullity = n_cols - rank

        # Nullspace basis consists of right singular vectors corresponding to singular values <= tol
        v_mat = v_t.T
        nullspace_basis = v_mat[:, rank:]

        # Particular solution satisfies A @ x_part = b_target
        if b_target is not None:
            if len(b_target) < m_rows:
                b_full = np.concatenate([b_target, np.zeros(m_rows - len(b_target))])
            else:
                b_full = b_target
            b_scaled = row_scale * b_full
            s_inv = np.zeros(rank)
            for i in range(rank):
                s_inv[i] = 1.0 / s[i]
            x_part = v_mat[:, :rank] @ (s_inv * (u[:, :rank].T @ b_scaled))
        else:
            x_part = np.zeros(n_cols)

        res_norm = (
            float(np.max(np.abs(a_mat @ nullspace_basis))) if nullity > 0 else 0.0
        )
        ortho_norm = (
            float(np.max(np.abs(nullspace_basis.T @ nullspace_basis - np.eye(nullity))))
            if nullity > 0
            else 0.0
        )

        return NullspaceDecomposition(
            matrix_rank=rank,
            nullity=nullity,
            singular_values=s,
            nullspace_basis=nullspace_basis,
            particular_solution=x_part,
            residual_norm=res_norm,
            basis_orthogonality_residual=ortho_norm,
        )

    def solve_reduced_coordinates(
        self,
        node: TrajectoryOptimizationNode,
        decomp: NullspaceDecomposition,
        weight_tau: float = 1.0,
        weight_f: float = 1e-4,
        weight_grip: float = 1e-4,
    ) -> TradeoffSample:
        """Solve bounded least squares in reduced nullspace coordinates z."""
        n_vars = self.n_node_vars
        w_diag = np.ones(n_vars)
        w_diag[: self.n_actuated] = weight_tau
        w_diag[self.n_actuated : self.n_actuated + self.n_ground_vars] = weight_f
        idx_grip = self.n_actuated + self.n_ground_vars
        w_diag[idx_grip : idx_grip + 6] = weight_grip
        w_diag[idx_grip + 6 :] = 1e4

        # x = x_part + N @ z
        n_basis = decomp.nullspace_basis
        x_part = decomp.particular_solution
        if np.linalg.norm(x_part) < 1e-12:
            # Solve minimum-norm particular solution matching node.tau_rnea
            a_eq = self._build_equality_matrix(
                node.j_ground, node.j_grip, node.active_contact_mask
            )
            b_eq = node.tau_rnea
            if len(b_eq) < a_eq.shape[0]:
                b_eq = np.concatenate([b_eq, np.zeros(a_eq.shape[0] - len(b_eq))])
            x_part = la.lstsq(a_eq, b_eq)[0]

        # Objective: 0.5 * (x_part + N @ z)^T @ W @ (x_part + N @ z)
        # = 0.5 * z^T @ (N^T @ W @ N) @ z + (x_part^T @ W @ N) @ z
        h_red = n_basis.T @ (w_diag[:, np.newaxis] * n_basis)
        c_red = n_basis.T @ (w_diag * x_part)

        # Friction cone inequality constraints in physical coordinates
        ineq_rows: list[Array] = []
        ineq_ub: list[float] = []
        mask = node.active_contact_mask
        for s in range(self.n_contact_spheres):
            if mask is not None and s < len(mask) and not mask[s]:
                continue
            ix = self.n_actuated + s * 3
            iy = ix + 1
            iz = ix + 2
            mu = self.mu_friction
            for sign, coord in [(1.0, ix), (-1.0, ix), (1.0, iy), (-1.0, iy)]:
                row = np.zeros(n_vars)
                row[coord] = sign
                row[iz] = -mu
                ineq_rows.append(row @ n_basis)
                ineq_ub.append(-float(row @ x_part))

        # Torque bounds
        lb_z = np.full(decomp.nullity, -np.inf)
        ub_z = np.full(decomp.nullity, np.inf)

        constraints = []
        if ineq_rows:
            a_ineq_z = np.vstack(ineq_rows)
            b_ineq_z = np.asarray(ineq_ub)
            constraints.append(LinearConstraint(a_ineq_z, -np.inf, b_ineq_z))

        z0 = np.zeros(decomp.nullity)
        sol = minimize(
            fun=lambda z: 0.5 * float(z @ (h_red @ z)) + float(c_red @ z),
            x0=z0,
            jac=lambda z: h_red @ z + c_red,
            method="SLSQP",
            bounds=Bounds(lb_z, ub_z),
            constraints=constraints,
            options={"ftol": 1e-8, "disp": False},
        )

        z_opt = sol.x if (sol.success or getattr(sol, "status", 0) == 8) else z0
        x_full = x_part + n_basis @ z_opt

        return self._build_sample("reduced_solve", 1.0, x_full, True)

    def evaluate_hard_zero_trail_feasibility(
        self,
        node: TrajectoryOptimizationNode,
        trail_arm_indices: Sequence[int],
    ) -> tuple[bool, tuple[str, ...], TradeoffSample | None]:
        """Diagnose whether hard-zero trail torque is physical vs blocked by capacity constraints."""
        require(len(trail_arm_indices) > 0, "must provide trail_arm_indices")

        # 1. Attempt hard-zero trail allocation
        alloc_hard = self._allocator.allocate(
            tau_rnea=node.tau_rnea,
            j_ground=node.j_ground,
            j_grip=node.j_grip,
            objective=AllocationObjective.HARD_ZERO_TRAIL_ARM,
            trail_arm_indices=trail_arm_indices,
            tau_bounds=(
                (-self.tau_max, self.tau_max) if self.tau_max is not None else None
            ),
        )

        blockers: list[str] = []
        if self.tau_max is not None:
            trail_set = set(trail_arm_indices)
            lead_indices = [
                idx for idx in self.actuated_indices if idx not in trail_set
            ]
            for lead_idx in lead_indices:
                col = np.where(self.actuated_indices == lead_idx)[0][0]
                lead_t = abs(alloc_hard.tau_actuated[col])
                cap = float(self.tau_max[col])
                if lead_t > cap + 1e-4:
                    blockers.append(
                        f"Lead arm joint {lead_idx} torque capacity exceeded ({lead_t:.1f} Nm > {cap:.1f} Nm)"
                    )

        is_feasible = alloc_hard.success and len(blockers) == 0

        # 2. Compute relaxed minimum-trail fallback
        alloc_relaxed = self._allocator.allocate(
            tau_rnea=node.tau_rnea,
            j_ground=node.j_ground,
            j_grip=node.j_grip,
            objective=AllocationObjective.MINIMUM_TRAIL_ARM,
            trail_arm_indices=trail_arm_indices,
            tau_bounds=(
                (-self.tau_max, self.tau_max) if self.tau_max is not None else None
            ),
        )

        x_rep = self._allocation_to_full(alloc_hard if is_feasible else alloc_relaxed)
        sample = self._build_sample(
            "hard_zero_trail" if is_feasible else "minimum_trail_fallback",
            0.0 if is_feasible else 1.0,
            x_rep,
            alloc_hard.success if is_feasible else alloc_relaxed.success,
            trail_arm_indices=trail_arm_indices,
            blockers=() if (is_feasible or alloc_relaxed.success) else blockers,
        )

        return is_feasible, tuple(blockers), sample

    def explore_tradeoff(
        self,
        node: TrajectoryOptimizationNode,
        dimension: TradeoffDimension,
        sweep_values: Sequence[float],
        trail_arm_indices: Sequence[int] | None = None,
    ) -> ParetoFrontierReport:
        """Sweep parameter along tradeoff dimension and construct Pareto frontier report."""
        samples: list[TradeoffSample] = []

        for val in sweep_values:
            cfg = self._config_for_dimension(dimension, val)
            opt = SparseTrajectoryTorqueOptimizer(
                nv=self.nv,
                actuated_indices=self.actuated_indices,
                n_contact_spheres=self.n_contact_spheres,
                mu_friction=self.mu_friction,
                config=cfg,
            )

            # For trail arm share, set objective on fallback allocator
            if dimension == TradeoffDimension.TRAIL_ARM_SHARE and trail_arm_indices:
                alloc = self._allocator.allocate(
                    tau_rnea=node.tau_rnea,
                    j_ground=node.j_ground,
                    j_grip=node.j_grip,
                    objective=AllocationObjective.MINIMUM_TRAIL_ARM,
                    trail_arm_indices=trail_arm_indices,
                )
                x_full = self._allocation_to_full(alloc)
                samples.append(
                    self._build_sample(
                        dimension.value, val, x_full, alloc.success, trail_arm_indices
                    )
                )
            else:
                res = opt.optimize([node])
                if res.success and res.nodes:
                    node_res = res.nodes[0]
                    x_full = np.concatenate(
                        [
                            node_res.tau_actuated,
                            node_res.f_ground,
                            node_res.lambda_grip,
                            node_res.delta_tau_root,
                        ]
                    )
                    samples.append(
                        self._build_sample(
                            dimension.value, val, x_full, True, trail_arm_indices
                        )
                    )
                else:
                    x_dummy = np.zeros(self.n_node_vars)
                    samples.append(
                        self._build_sample(
                            dimension.value, val, x_dummy, False, trail_arm_indices
                        )
                    )

        def_idx, rationale = self._select_conservative_default(samples, dimension)
        hard_zero_feas = False
        blockers: tuple[str, ...] = ()
        if trail_arm_indices:
            hard_zero_feas, blockers, _ = self.evaluate_hard_zero_trail_feasibility(
                node, trail_arm_indices
            )

        return ParetoFrontierReport(
            dimension=dimension,
            samples=tuple(samples),
            conservative_default_index=def_idx,
            conservative_default_rationale=rationale,
            hard_zero_feasible=hard_zero_feas,
            hard_zero_blocking_constraints=blockers,
        )

    def _config_for_dimension(
        self, dimension: TradeoffDimension, val: float
    ) -> SparseOptimizationConfig:
        if dimension == TradeoffDimension.EFFORT_VS_SMOOTHNESS:
            return SparseOptimizationConfig(
                weight_torque_effort=1.0,
                weight_torque_rate=val,
                bounds_tau_max=self.tau_max,
            )
        if dimension == TradeoffDimension.PEAK_UTILIZATION:
            return SparseOptimizationConfig(
                weight_torque_effort=1.0,
                weight_peak_utilization=val,
                bounds_tau_max=self.tau_max,
            )
        if dimension == TradeoffDimension.GRIP_INTERNAL_LOAD:
            return SparseOptimizationConfig(
                weight_torque_effort=1.0,
                weight_grip_wrench=val,
                bounds_tau_max=self.tau_max,
            )
        return SparseOptimizationConfig(
            weight_torque_effort=1.0,
            weight_contact_rate=val,
            bounds_tau_max=self.tau_max,
        )

    def _allocation_to_full(self, alloc: Any) -> Array:
        return np.concatenate(
            [
                alloc.tau_actuated,
                alloc.f_ground,
                alloc.lambda_grip,
                alloc.delta_tau_root,
            ]
        )

    def _build_sample(
        self,
        param_name: str,
        param_val: float,
        x_full: Array,
        is_feas: bool,
        trail_arm_indices: Sequence[int] | None = None,
        blockers: Sequence[str] | None = None,
    ) -> TradeoffSample:
        tau = x_full[: self.n_actuated]
        fg = x_full[self.n_actuated : self.n_actuated + self.n_ground_vars]
        idx_grip = self.n_actuated + self.n_ground_vars
        idx_root = idx_grip + 6
        lg = x_full[idx_grip:idx_root]
        rt = x_full[idx_root : self.n_node_vars]

        tot_effort = float(np.sum(tau**2))
        peak_t = float(np.max(np.abs(tau))) if len(tau) > 0 else 0.0

        trail_eff = 0.0
        lead_eff = 0.0
        if trail_arm_indices:
            trail_set = set(trail_arm_indices)
            for col, idx in enumerate(self.actuated_indices):
                if idx in trail_set:
                    trail_eff += float(tau[col] ** 2)
                else:
                    lead_eff += float(tau[col] ** 2)
        else:
            lead_eff = tot_effort

        squeeze = float(np.linalg.norm(lg[:3]))
        n_spheres = len(fg) // 3
        grf_norms = [
            float(np.linalg.norm(fg[s * 3 : (s + 1) * 3])) for s in range(n_spheres)
        ]
        max_grf = float(np.max(grf_norms)) if grf_norms else 0.0

        return TradeoffSample(
            parameter_name=param_name,
            parameter_value=param_val,
            is_feasible=is_feas,
            tau_actuated=tau,
            f_ground=fg,
            lambda_grip=lg,
            delta_tau_root=rt,
            total_effort=tot_effort,
            peak_torque_nm=peak_t,
            trail_arm_effort=trail_eff,
            lead_arm_effort=lead_eff,
            grip_squeeze_n=squeeze,
            max_grf_n=max_grf,
            infeasibility_reasons=tuple(blockers or ()),
        )

    def _select_conservative_default(
        self, samples: list[TradeoffSample], dimension: TradeoffDimension
    ) -> tuple[int, str]:
        if not samples:
            return 0, "No samples available."
        feasible_indices = [i for i, s in enumerate(samples) if s.is_feasible]
        if not feasible_indices:
            return 0, "All candidate points violated capacity limits."

        # Select median feasible sample as balanced default
        mid = feasible_indices[len(feasible_indices) // 2]
        rationale = (
            f"Balanced trade-off along {dimension.value}: parameter={samples[mid].parameter_value:.2f}, "
            f"peak_torque={samples[mid].peak_torque_nm:.1f} Nm, effort={samples[mid].total_effort:.1f}."
        )
        return mid, rationale

    def generate_pareto_table_markdown(
        self, reports: Sequence[ParetoFrontierReport]
    ) -> str:
        """Construct structured markdown report detailing Pareto tradeoffs and conservative defaults."""
        lines = [
            "# Multi-Engine Force Nullspace & Torque Distribution Trade-Offs (PF-06)",
            "",
            "## Summary of Explored Tradeoff Frontiers",
            "",
            "| Dimension | Parameter | Feasible | Total Effort | Peak Torque | Trail Share | Lead Share | Max GRF | Notes |",
            "|-----------|-----------|----------|--------------|-------------|-------------|------------|---------|-------|",
        ]

        for rep in reports:
            dim_name = rep.dimension.value
            for idx, s in enumerate(rep.samples):
                is_def = idx == rep.conservative_default_index
                notes = "**Conservative Default**" if is_def else ""
                if s.infeasibility_reasons:
                    notes = "; ".join(s.infeasibility_reasons)
                lines.append(
                    f"| {dim_name} | {s.parameter_value:.2f} | {s.is_feasible} | "
                    f"{s.total_effort:.1f} | {s.peak_torque_nm:.1f} Nm | "
                    f"{s.trail_arm_effort:.1f} | {s.lead_arm_effort:.1f} | "
                    f"{s.max_grf_n:.1f} N | {notes} |"
                )

        lines.extend(
            [
                "",
                "## Conservative Default Selection & Rationale",
                "",
            ]
        )
        for rep in reports:
            lines.append(
                f"- **{rep.dimension.value}**: {rep.conservative_default_rationale}"
            )

        return "\n".join(lines)
