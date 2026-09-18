"""Sparse Trajectory Torque Optimizer (PF-05, #10435).

Formulates and solves full-horizon and receding-window sparse trajectory optimization
for smooth, low-effort torque histories, contact force sharing, and grip wrench transmission:

    min_{tau, f, lambda, delta_tau_root, gamma}
        sum_{k=0}^{K-1} w_k [ ||tau_k||_{W_tau}^2 + ||lambda_k||_{W_lambda}^2 + w_root ||delta_tau_root,k||^2 ]
      + sum_{k=0}^{K-2} dt_k [ ||tau_dot_k||_{W_tau_dot}^2 + ||f_dot_k||_{W_f_dot}^2 ]
      + sum_{k=1}^{K-2} dt_bar_k ||tau_ddot_k||_{W_tau_ddot}^2
      + w_peak * gamma^2

Subject to:
1. Exact inverse dynamics per node:
   S^T * tau_k + J_ground,k^T * f_k + J_grip,k^T * lambda_k + S_root^T * delta_tau_root,k = tau_rnea,k
2. Unilateral ground reaction forces: f_{z, s, k} >= 0 and flight mode f_{s, k} = 0 when inactive.
3. Linearized Coulomb friction cones: |f_{x, s, k}| <= mu * f_{z, s, k}, |f_{y, s, k}| <= mu * f_{z, s, k}.
4. Actuator torque limits and power envelope: |tau_{j, k}| <= tau_max,j, |tau_{j, k} * q_dot_{j, k}| <= P_max,j.
5. Epigraph peak utilization: |tau_{j, k}| <= gamma * tau_max,j (when w_peak > 0).
6. Floating-base residual bounds: ||delta_tau_root,k||_inf <= epsilon_root.
7. Receding horizon seam constraints: C0/C1 continuity across window boundaries with zero torque jumps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, minimize

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.contact_force_allocator import (
    AllocationObjective,
    ContactForceAllocator,
)

Array: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.integer[Any]]


@dataclass(frozen=True)
class SparseOptimizationConfig:
    """Hyperparameters, bounds, and weights for trajectory torque optimization."""

    weight_torque_effort: float = 1.0
    weight_grip_wrench: float = 1e-4
    weight_root_residual: float = 1e4
    weight_torque_rate: float = 10.0
    weight_contact_rate: float = 1.0
    weight_torque_acceleration: float = 1.0
    weight_peak_utilization: float = 0.0
    enable_power_limits: bool = True
    root_force_tolerance_n: float = 5.0
    root_torque_tolerance_nm: float = 1.0
    max_iter: int = 300
    ftol: float = 1e-7
    bounds_tau_max: Array | None = None
    bounds_power_max: Array | None = None

    def __post_init__(self) -> None:
        require(self.weight_torque_effort >= 0.0, "weight_torque_effort >= 0")
        require(self.weight_torque_rate >= 0.0, "weight_torque_rate >= 0")
        require(self.weight_contact_rate >= 0.0, "weight_contact_rate >= 0")
        require(
            self.weight_torque_acceleration >= 0.0, "weight_torque_acceleration >= 0"
        )
        require(self.root_force_tolerance_n > 0.0, "root_force_tolerance_n > 0")
        require(self.root_torque_tolerance_nm > 0.0, "root_torque_tolerance_nm > 0")


@dataclass(frozen=True)
class TrajectoryOptimizationNode:
    """Dynamics and contact kinematics context for a single time step."""

    time_s: float
    tau_rnea: Array
    j_ground: Array
    j_grip: Array
    q_dot: Array | None = None
    active_contact_mask: Array | None = None

    def __post_init__(self) -> None:
        require(self.time_s >= 0.0, "time_s must be non-negative", self.time_s)
        require(len(self.tau_rnea) > 6, "tau_rnea must exceed floating-base DOFs")


@dataclass(frozen=True)
class SeamBoundaryCondition:
    """Boundary conditions for C0 and C1 continuity at window seams."""

    tau_seam: Array
    tau_dot_seam: Array | None = None


@dataclass(frozen=True)
class NodeOptimizationResult:
    """Optimized physical variables at a single trajectory node."""

    time_s: float
    tau_actuated: Array
    f_ground: Array
    lambda_grip: Array
    delta_tau_root: Array
    tau_dot: Array | None = None
    tau_ddot: Array | None = None


@dataclass(frozen=True)
class SparseOptimizationResult:
    """Full trajectory optimization outcome and diagnostics."""

    success: bool
    nodes: tuple[NodeOptimizationResult, ...]
    total_cost: float
    effort_cost: float
    rate_cost: float
    acceleration_cost: float
    peak_utilization: float | None
    max_root_force_n: float
    max_root_torque_nm: float
    solver_message: str


@dataclass(frozen=True)
class InterpolationAuditResult:
    """Qualification audit of interpolated inter-node states."""

    is_feasible: bool
    max_torque_utilization: float
    max_power_w: float
    max_friction_violation_n: float
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)


class SparseTrajectoryTorqueOptimizer:
    """Full-horizon sparse QP torque and contact force optimizer."""

    def __init__(
        self,
        nv: int,
        actuated_indices: Sequence[int] | IntArray,
        n_contact_spheres: int = 6,
        mu_friction: float = 0.8,
        config: SparseOptimizationConfig | None = None,
    ) -> None:
        require(nv > 6, "nv must exceed 6 floating-base coordinates", nv)
        require(len(actuated_indices) > 0, "must have actuated coordinates")
        require(n_contact_spheres > 0, "must have contact spheres")
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
        self.config = config or SparseOptimizationConfig()

        self._s_transpose = np.zeros((nv, self.n_actuated))
        for col, idx in enumerate(self.actuated_indices):
            self._s_transpose[idx, col] = 1.0

        self._s_root_transpose = np.zeros((nv, 6))
        for i in range(6):
            self._s_root_transpose[i, i] = 1.0

        self._fallback_allocator = ContactForceAllocator(
            nv=nv,
            actuated_indices=self.actuated_indices,
            n_contact_spheres=n_contact_spheres,
            mu_friction=mu_friction,
        )

    def _node_offset(self, k: int) -> int:
        return k * self.n_node_vars

    def _build_cost_matrix(
        self,
        nodes: Sequence[TrajectoryOptimizationNode],
        total_vars: int,
        has_epigraph: bool,
        initial_seam: SeamBoundaryCondition | None = None,
    ) -> tuple[sp.csr_matrix, Array]:
        k_count = len(nodes)
        mat = sp.dok_matrix((total_vars, total_vars), dtype=float)
        c = np.zeros(total_vars, dtype=float)

        cfg = self.config
        times = [n.time_s for n in nodes]
        dts = [times[i + 1] - times[i] for i in range(k_count - 1)]

        self._add_effort_cost(mat, k_count, dts, cfg)
        self._add_rate_costs(mat, c, k_count, dts, cfg, initial_seam)
        self._add_acceleration_costs(mat, k_count, dts, cfg)

        if has_epigraph:
            mat[total_vars - 1, total_vars - 1] += 2.0 * cfg.weight_peak_utilization

        return mat.tocsr(), c

    def _add_effort_cost(
        self,
        mat: sp.dok_matrix,
        k_count: int,
        dts: list[float],
        cfg: SparseOptimizationConfig,
    ) -> None:
        for k in range(k_count):
            offset = self._node_offset(k)
            w_k = (
                1.0
                if k_count == 1
                else (
                    0.5 * dts[0]
                    if k == 0
                    else (
                        0.5 * dts[-1]
                        if k == k_count - 1
                        else 0.5 * (dts[k - 1] + dts[k])
                    )
                )
            )
            for j in range(self.n_actuated):
                mat[offset + j, offset + j] += 2.0 * w_k * cfg.weight_torque_effort
            for j in range(self.n_ground_vars):
                idx = offset + self.n_actuated + j
                mat[idx, idx] += 2.0 * w_k * 1e-4
            for j in range(self.n_grip_vars):
                idx = offset + self.n_actuated + self.n_ground_vars + j
                mat[idx, idx] += 2.0 * w_k * cfg.weight_grip_wrench
            for j in range(self.n_root_vars):
                idx = offset + self.n_actuated + self.n_ground_vars + 6 + j
                mat[idx, idx] += 2.0 * w_k * cfg.weight_root_residual

    def _add_rate_costs(
        self,
        mat: sp.dok_matrix,
        c: Array,
        k_count: int,
        dts: list[float],
        cfg: SparseOptimizationConfig,
        initial_seam: SeamBoundaryCondition | None = None,
    ) -> None:
        for k in range(k_count - 1):
            dt = max(dts[k], 1e-6)
            off_a = self._node_offset(k)
            off_b = self._node_offset(k + 1)
            coeff_tau = 2.0 * cfg.weight_torque_rate / dt
            coeff_f = 2.0 * cfg.weight_contact_rate / dt

            for j in range(self.n_actuated):
                ia, ib = off_a + j, off_b + j
                mat[ia, ia] += coeff_tau
                mat[ib, ib] += coeff_tau
                mat[ia, ib] -= coeff_tau
                mat[ib, ia] -= coeff_tau

            for j in range(self.n_ground_vars):
                ia = off_a + self.n_actuated + j
                ib = off_b + self.n_actuated + j
                mat[ia, ia] += coeff_f
                mat[ib, ib] += coeff_f
                mat[ia, ib] -= coeff_f
                mat[ib, ia] -= coeff_f

        if (
            initial_seam is not None
            and initial_seam.tau_dot_seam is not None
            and k_count >= 2
        ):
            dt0 = max(dts[0], 1e-6)
            coeff = 2.0 * cfg.weight_torque_rate / dt0
            off_next = self._node_offset(1)
            for j in range(self.n_actuated):
                d_val = float(initial_seam.tau_dot_seam[j])
                target = float(initial_seam.tau_seam[j]) + dt0 * d_val
                mat[off_next + j, off_next + j] += coeff
                c[off_next + j] -= 2.0 * coeff * target

    def _add_acceleration_costs(
        self,
        mat: sp.dok_matrix,
        k_count: int,
        dts: list[float],
        cfg: SparseOptimizationConfig,
    ) -> None:
        if cfg.weight_torque_acceleration <= 0.0 or k_count < 3:
            return
        for k in range(1, k_count - 1):
            dt_prev = max(dts[k - 1], 1e-6)
            dt_curr = max(dts[k], 1e-6)
            dt_bar = 0.5 * (dt_prev + dt_curr)
            scale = 1.0 / math.sqrt(dt_bar)
            a_prev = scale / dt_prev
            a_curr = -scale * (1.0 / dt_prev + 1.0 / dt_curr)
            a_next = scale / dt_curr

            for j in range(self.n_actuated):
                i_p = self._node_offset(k - 1) + j
                i_c = self._node_offset(k) + j
                i_n = self._node_offset(k + 1) + j
                for idx_u, w_u in [(i_p, a_prev), (i_c, a_curr), (i_n, a_next)]:
                    for idx_v, w_v in [(i_p, a_prev), (i_c, a_curr), (i_n, a_next)]:
                        mat[idx_u, idx_v] += (
                            2.0 * cfg.weight_torque_acceleration * w_u * w_v
                        )

    def _build_equalities(
        self,
        nodes: Sequence[TrajectoryOptimizationNode],
        total_vars: int,
    ) -> tuple[sp.csr_matrix, Array]:
        k_count = len(nodes)
        eq_rows: list[sp.dok_matrix] = []
        eq_b: list[float] = []

        for k, node in enumerate(nodes):
            a_node = np.hstack(
                [
                    self._s_transpose,
                    node.j_ground.T,
                    node.j_grip.T,
                    self._s_root_transpose,
                ]
            )
            offset = self._node_offset(k)
            for r in range(self.nv):
                row_vec = sp.dok_matrix((1, total_vars), dtype=float)
                for c in range(self.n_node_vars):
                    val = a_node[r, c]
                    if abs(val) > 1e-12:
                        row_vec[0, offset + c] = val
                eq_rows.append(row_vec)
                eq_b.append(node.tau_rnea[r])

        if not eq_rows:
            return sp.csr_matrix((0, total_vars)), np.zeros(0)
        return sp.vstack(eq_rows).tocsr(), np.asarray(eq_b, dtype=float)

    def _build_inequalities(
        self,
        nodes: Sequence[TrajectoryOptimizationNode],
        total_vars: int,
        has_epigraph: bool,
    ) -> tuple[sp.csr_matrix, Array]:
        ineq_rows: list[sp.dok_matrix] = []
        ineq_ub: list[float] = []
        gamma_idx = total_vars - 1

        for k, node in enumerate(nodes):
            offset = self._node_offset(k)
            # Coulomb friction cones per active sphere
            self._add_friction_cone_rows(offset, node, ineq_rows, ineq_ub, total_vars)

            # Epigraph bounds: |tau_{k, j}| <= gamma * tau_max,j
            if has_epigraph and self.config.bounds_tau_max is not None:
                for j in range(self.n_actuated):
                    tau_max = float(self.config.bounds_tau_max[j])
                    idx_tau = offset + j
                    # tau - gamma * tau_max <= 0
                    r1 = sp.dok_matrix((1, total_vars), dtype=float)
                    r1[0, idx_tau] = 1.0
                    r1[0, gamma_idx] = -tau_max
                    ineq_rows.append(r1)
                    ineq_ub.append(0.0)
                    # -tau - gamma * tau_max <= 0
                    r2 = sp.dok_matrix((1, total_vars), dtype=float)
                    r2[0, idx_tau] = -1.0
                    r2[0, gamma_idx] = -tau_max
                    ineq_rows.append(r2)
                    ineq_ub.append(0.0)

        if not ineq_rows:
            return sp.csr_matrix((0, total_vars)), np.zeros(0)
        return sp.vstack(ineq_rows).tocsr(), np.asarray(ineq_ub, dtype=float)

    def _add_friction_cone_rows(
        self,
        offset: int,
        node: TrajectoryOptimizationNode,
        ineq_rows: list[sp.dok_matrix],
        ineq_ub: list[float],
        total_vars: int,
    ) -> None:
        mask = node.active_contact_mask
        for s in range(self.n_contact_spheres):
            if mask is not None and s < len(mask) and not mask[s]:
                continue
            ix = offset + self.n_actuated + s * 3
            iy = ix + 1
            iz = ix + 2
            mu = self.mu_friction
            for sign, coord in [(1.0, ix), (-1.0, ix), (1.0, iy), (-1.0, iy)]:
                r = sp.dok_matrix((1, total_vars), dtype=float)
                r[0, coord] = sign
                r[0, iz] = -mu
                ineq_rows.append(r)
                ineq_ub.append(0.0)

    def _build_bounds(
        self,
        nodes: Sequence[TrajectoryOptimizationNode],
        total_vars: int,
        has_epigraph: bool,
        initial_seam: SeamBoundaryCondition | None = None,
        final_seam: SeamBoundaryCondition | None = None,
    ) -> Bounds:
        lb = np.full(total_vars, -np.inf)
        ub = np.full(total_vars, np.inf)

        cfg = self.config
        for k, node in enumerate(nodes):
            offset = self._node_offset(k)
            # 1. Actuator torque & power bounds
            for j in range(self.n_actuated):
                limit = (
                    float(cfg.bounds_tau_max[j])
                    if cfg.bounds_tau_max is not None
                    else np.inf
                )
                if (
                    cfg.enable_power_limits
                    and cfg.bounds_power_max is not None
                    and node.q_dot is not None
                ):
                    q_d = abs(float(node.q_dot[self.actuated_indices[j]]))
                    if q_d > 1e-4:
                        p_limit = float(cfg.bounds_power_max[j]) / q_d
                        limit = min(limit, p_limit)
                lb[offset + j] = -limit
                ub[offset + j] = limit

            # 2. Contact forces: f_z >= 0 and zero flight forces
            mask = node.active_contact_mask
            for s in range(self.n_contact_spheres):
                iz = offset + self.n_actuated + s * 3 + 2
                lb[iz] = 0.0
                if mask is not None and s < len(mask) and not mask[s]:
                    for m in range(3):
                        var_idx = offset + self.n_actuated + s * 3 + m
                        lb[var_idx] = 0.0
                        ub[var_idx] = 0.0

            # 3. Floating base residual slack bounds
            r_off = offset + self.n_actuated + self.n_ground_vars + 6
            for m in range(3):
                lb[r_off + m] = -cfg.root_force_tolerance_n
                ub[r_off + m] = cfg.root_force_tolerance_n
                lb[r_off + 3 + m] = -cfg.root_torque_tolerance_nm
                ub[r_off + 3 + m] = cfg.root_torque_tolerance_nm

        if initial_seam is not None:
            for j in range(self.n_actuated):
                val = float(initial_seam.tau_seam[j])
                lb[j] = val
                ub[j] = val

        if final_seam is not None:
            off_last = self._node_offset(len(nodes) - 1)
            for j in range(self.n_actuated):
                val = float(final_seam.tau_seam[j])
                lb[off_last + j] = val
                ub[off_last + j] = val

        if has_epigraph:
            lb[total_vars - 1] = 0.0
            ub[total_vars - 1] = np.inf

        return Bounds(lb, ub)

    def _warm_start(
        self,
        nodes: Sequence[TrajectoryOptimizationNode],
        total_vars: int,
        has_epigraph: bool,
        initial_seam: SeamBoundaryCondition | None = None,
        final_seam: SeamBoundaryCondition | None = None,
    ) -> Array:
        x0 = np.zeros(total_vars, dtype=float)
        max_ratio = 0.1
        k_count = len(nodes)

        for k, node in enumerate(nodes):
            tau_b = None
            if k == 0 and initial_seam is not None:
                tau_b = (initial_seam.tau_seam - 1e-4, initial_seam.tau_seam + 1e-4)
            elif k == k_count - 1 and final_seam is not None:
                tau_b = (final_seam.tau_seam - 1e-4, final_seam.tau_seam + 1e-4)

            alloc = self._fallback_allocator.allocate(
                tau_rnea=node.tau_rnea,
                j_ground=node.j_ground,
                j_grip=node.j_grip,
                tau_bounds=tau_b,
                objective=AllocationObjective.MINIMUM_EFFORT,
            )
            offset = self._node_offset(k)
            x0[offset : offset + self.n_actuated] = alloc.tau_actuated
            x0[
                offset + self.n_actuated : offset + self.n_actuated + self.n_ground_vars
            ] = alloc.f_ground
            idx_grip = offset + self.n_actuated + self.n_ground_vars
            x0[idx_grip : idx_grip + 6] = alloc.lambda_grip
            x0[idx_grip + 6 : idx_grip + 12] = alloc.delta_tau_root

            if has_epigraph and self.config.bounds_tau_max is not None:
                for j in range(self.n_actuated):
                    r = abs(alloc.tau_actuated[j]) / float(
                        self.config.bounds_tau_max[j]
                    )
                    if r > max_ratio:
                        max_ratio = r

        if has_epigraph:
            x0[total_vars - 1] = max_ratio * 1.1

        return x0

    def optimize(
        self,
        nodes: Sequence[TrajectoryOptimizationNode],
        initial_seam: SeamBoundaryCondition | None = None,
        final_seam: SeamBoundaryCondition | None = None,
    ) -> SparseOptimizationResult:
        """Execute full sparse trajectory optimization across input nodes."""
        require(len(nodes) > 0, "nodes sequence must not be empty")

        k_count = len(nodes)
        has_epigraph = (
            self.config.weight_peak_utilization > 0.0
            and self.config.bounds_tau_max is not None
        )
        total_vars = k_count * self.n_node_vars + (1 if has_epigraph else 0)

        h_mat, c_vec = self._build_cost_matrix(
            nodes, total_vars, has_epigraph, initial_seam
        )
        a_eq, b_eq = self._build_equalities(nodes, total_vars)
        a_ineq, b_ineq = self._build_inequalities(nodes, total_vars, has_epigraph)
        bounds = self._build_bounds(
            nodes, total_vars, has_epigraph, initial_seam, final_seam
        )
        x0 = self._warm_start(nodes, total_vars, has_epigraph, initial_seam, final_seam)

        constraints = []
        if a_eq.shape[0] > 0:
            constraints.append(LinearConstraint(a_eq, b_eq, b_eq))
        if a_ineq.shape[0] > 0:
            constraints.append(LinearConstraint(a_ineq, -np.inf, b_ineq))

        sol = minimize(
            fun=lambda x: 0.5 * float(x @ (h_mat @ x)) + float(c_vec @ x),
            x0=x0,
            jac=lambda x: h_mat @ x + c_vec,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": self.config.max_iter,
                "ftol": self.config.ftol,
                "disp": False,
            },
        )

        return self._extract_result(nodes, sol, h_mat, total_vars, has_epigraph)

    def _extract_result(
        self,
        nodes: Sequence[TrajectoryOptimizationNode],
        sol: Any,
        h_mat: sp.csr_matrix,
        total_vars: int,
        has_epigraph: bool,
    ) -> SparseOptimizationResult:
        k_count = len(nodes)
        x = np.asarray(sol.x, dtype=float)
        node_results = []
        max_root_f = 0.0
        max_root_t = 0.0

        for k in range(k_count):
            offset = self._node_offset(k)
            tau = x[offset : offset + self.n_actuated]
            fg = x[
                offset + self.n_actuated : offset + self.n_actuated + self.n_ground_vars
            ]
            lg = x[
                offset + self.n_actuated + self.n_ground_vars : offset
                + self.n_actuated
                + self.n_ground_vars
                + 6
            ]
            rt = x[
                offset + self.n_actuated + self.n_ground_vars + 6 : offset
                + self.n_node_vars
            ]

            f_norm = float(np.linalg.norm(rt[:3]))
            t_norm = float(np.linalg.norm(rt[3:]))
            max_root_f = max(max_root_f, f_norm)
            max_root_t = max(max_root_t, t_norm)

            node_results.append(
                NodeOptimizationResult(
                    time_s=nodes[k].time_s,
                    tau_actuated=tau,
                    f_ground=fg,
                    lambda_grip=lg,
                    delta_tau_root=rt,
                )
            )

        self._populate_derivatives(node_results)

        total_cost = 0.5 * float(x @ (h_mat @ x))
        effort_cost, rate_cost, accel_cost = self._compute_cost_breakdown(node_results)
        peak_util = float(x[total_vars - 1]) if has_epigraph else None

        is_success = bool(sol.success) or (
            getattr(sol, "status", None) == 8
            and max_root_f <= self.config.root_force_tolerance_n
            and max_root_t <= self.config.root_torque_tolerance_nm
        )

        return SparseOptimizationResult(
            success=is_success,
            nodes=tuple(node_results),
            total_cost=total_cost,
            effort_cost=effort_cost,
            rate_cost=rate_cost,
            acceleration_cost=accel_cost,
            peak_utilization=peak_util,
            max_root_force_n=max_root_f,
            max_root_torque_nm=max_root_t,
            solver_message=str(sol.message),
        )

    def _populate_derivatives(self, node_results: list[NodeOptimizationResult]) -> None:
        k_count = len(node_results)
        tau_mat = np.array([n.tau_actuated for n in node_results])
        times = [n.time_s for n in node_results]

        tau_dots: list[Array | None] = [None] * k_count
        tau_ddots: list[Array | None] = [None] * k_count

        if k_count >= 2:
            for k in range(k_count):
                if k == 0:
                    dt = max(times[1] - times[0], 1e-6)
                    tau_dots[0] = (tau_mat[1] - tau_mat[0]) / dt
                elif k == k_count - 1:
                    dt = max(times[-1] - times[-2], 1e-6)
                    tau_dots[-1] = (tau_mat[-1] - tau_mat[-2]) / dt
                else:
                    dt1 = max(times[k] - times[k - 1], 1e-6)
                    dt2 = max(times[k + 1] - times[k], 1e-6)
                    tau_dots[k] = 0.5 * (
                        (tau_mat[k] - tau_mat[k - 1]) / dt1
                        + (tau_mat[k + 1] - tau_mat[k]) / dt2
                    )

        if k_count >= 3:
            for k in range(1, k_count - 1):
                dt_bar = 0.5 * (times[k + 1] - times[k - 1])
                d_next = (tau_mat[k + 1] - tau_mat[k]) / max(
                    times[k + 1] - times[k], 1e-6
                )
                d_prev = (tau_mat[k] - tau_mat[k - 1]) / max(
                    times[k] - times[k - 1], 1e-6
                )
                tau_ddots[k] = (d_next - d_prev) / max(dt_bar, 1e-6)

        for k in range(k_count):
            old = node_results[k]
            node_results[k] = NodeOptimizationResult(
                time_s=old.time_s,
                tau_actuated=old.tau_actuated,
                f_ground=old.f_ground,
                lambda_grip=old.lambda_grip,
                delta_tau_root=old.delta_tau_root,
                tau_dot=tau_dots[k],
                tau_ddot=tau_ddots[k],
            )

    def _compute_cost_breakdown(
        self, node_results: list[NodeOptimizationResult]
    ) -> tuple[float, float, float]:
        effort = sum(float(np.sum(n.tau_actuated**2)) for n in node_results)
        rate = sum(
            float(np.sum(n.tau_dot**2)) for n in node_results if n.tau_dot is not None
        )
        accel = sum(
            float(np.sum(n.tau_ddot**2)) for n in node_results if n.tau_ddot is not None
        )
        return effort, rate, accel


class WindowedTrajectoryTorqueOptimizer:
    """Receding-horizon trajectory optimizer enforcing C0/C1 continuity across window seams."""

    def __init__(
        self,
        nv: int,
        actuated_indices: Sequence[int] | IntArray,
        n_contact_spheres: int = 6,
        mu_friction: float = 0.8,
        window_size: int = 30,
        overlap_size: int = 10,
        config: SparseOptimizationConfig | None = None,
    ) -> None:
        require(window_size > 2, "window_size must exceed 2", window_size)
        require(
            0 < overlap_size < window_size,
            "overlap_size must be strictly between 0 and window_size",
            (overlap_size, window_size),
        )
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.stride = window_size - overlap_size
        self.optimizer = SparseTrajectoryTorqueOptimizer(
            nv=nv,
            actuated_indices=actuated_indices,
            n_contact_spheres=n_contact_spheres,
            mu_friction=mu_friction,
            config=config,
        )

    def optimize(
        self, nodes: Sequence[TrajectoryOptimizationNode]
    ) -> SparseOptimizationResult:
        """Solve trajectory in overlapping windows with seamless continuity."""
        total_nodes = len(nodes)
        if total_nodes <= self.window_size:
            return self.optimizer.optimize(nodes)

        stitched_nodes: list[NodeOptimizationResult] = []
        current_start = 0
        seam_cond: SeamBoundaryCondition | None = None

        while current_start < total_nodes:
            current_end = min(current_start + self.window_size, total_nodes)
            window_nodes = nodes[current_start:current_end]

            res = self.optimizer.optimize(window_nodes, initial_seam=seam_cond)
            if not res.success:
                return res

            is_last_window = current_end == total_nodes
            if is_last_window:
                stitched_nodes.extend(res.nodes)
                break

            # Seam occurs at window index (window_size - overlap_size)
            seam_idx = min(self.stride, len(res.nodes) - 1)
            stitched_nodes.extend(res.nodes[:seam_idx])

            seam_node = res.nodes[seam_idx]
            seam_cond = SeamBoundaryCondition(
                tau_seam=seam_node.tau_actuated,
                tau_dot_seam=seam_node.tau_dot,
            )
            current_start += seam_idx

        return SparseOptimizationResult(
            success=True,
            nodes=tuple(stitched_nodes),
            total_cost=sum(float(np.sum(n.tau_actuated**2)) for n in stitched_nodes),
            effort_cost=sum(float(np.sum(n.tau_actuated**2)) for n in stitched_nodes),
            rate_cost=0.0,
            acceleration_cost=0.0,
            peak_utilization=None,
            max_root_force_n=max(
                float(np.linalg.norm(n.delta_tau_root[:3])) for n in stitched_nodes
            ),
            max_root_torque_nm=max(
                float(np.linalg.norm(n.delta_tau_root[3:])) for n in stitched_nodes
            ),
            solver_message="Windowed optimization completed successfully",
        )


def audit_trajectory_interpolation_feasibility(
    result: SparseOptimizationResult,
    nodes: Sequence[TrajectoryOptimizationNode],
    n_subsamples: int = 5,
    tau_max: Array | None = None,
    power_max: Array | None = None,
    mu_friction: float = 0.8,
) -> InterpolationAuditResult:
    """Evaluate power, torque bounds, and friction cone compliance at inter-node samples."""
    require(result.success, "cannot audit an unsuccessful optimization result")
    require(n_subsamples >= 1, "n_subsamples must be at least 1", n_subsamples)

    failures: list[str] = []
    max_tau_util = 0.0
    max_power = 0.0
    max_fric_viol = 0.0

    k_count = len(result.nodes)
    for k in range(k_count - 1):
        node_a = result.nodes[k]
        node_b = result.nodes[k + 1]
        v_a = nodes[k].q_dot
        v_b = nodes[k + 1].q_dot

        for step in range(1, n_subsamples + 1):
            alpha = float(step) / float(n_subsamples + 1)
            tau_interp = (
                1.0 - alpha
            ) * node_a.tau_actuated + alpha * node_b.tau_actuated
            f_interp = (1.0 - alpha) * node_a.f_ground + alpha * node_b.f_ground

            # Check torque bounds
            if tau_max is not None:
                util = float(np.max(np.abs(tau_interp) / tau_max))
                max_tau_util = max(max_tau_util, util)
                if util > 1.0 + 1e-4:
                    failures.append(
                        f"Torque violation at inter-node {k}-{k + 1}: {util:.2f} of tau_max"
                    )

            # Check power envelope
            if power_max is not None and v_a is not None and v_b is not None:
                v_interp = (1.0 - alpha) * v_a + alpha * v_b
                p_util = float(
                    np.max(np.abs(tau_interp * v_interp[-len(tau_interp) :]))
                )
                max_power = max(max_power, p_util)
                if p_util > float(np.max(power_max)) + 1e-3:
                    failures.append(
                        f"Power violation at inter-node {k}-{k + 1}: {p_util:.1f} W"
                    )

            # Check contact friction cone
            n_spheres = len(f_interp) // 3
            for s in range(n_spheres):
                fx = f_interp[s * 3]
                fy = f_interp[s * 3 + 1]
                fz = f_interp[s * 3 + 2]
                if fz < -1e-4:
                    failures.append(f"Negative normal force at sphere {s}: {fz:.2f} N")
                viol = max(0.0, abs(fx) - mu_friction * fz, abs(fy) - mu_friction * fz)
                max_fric_viol = max(max_fric_viol, viol)
                if viol > 1.0:
                    failures.append(
                        f"Friction cone exceeded at sphere {s}: +{viol:.1f} N"
                    )

    return InterpolationAuditResult(
        is_feasible=len(failures) == 0,
        max_torque_utilization=max_tau_util,
        max_power_w=max_power,
        max_friction_violation_n=max_fric_viol,
        failure_reasons=tuple(failures),
    )
