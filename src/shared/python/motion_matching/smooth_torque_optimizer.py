"""Smooth Low-Effort Torque Optimizer (PF-05 / #10435).

Solves the multi-node trajectory-level sparse QP over actuator torques, ground contact
forces, internal grip wrenches, and diagnostic root slack across an entire motion horizon.

Formulation:
    min_X  sum_k  [ w_effort * ||tau_k / tau_max||^2 * dt_k
                   + w_dtau * ||dtau_k / tau_max||^2 * dt_k
                   + w_ddtau * ||ddtau_k / tau_max||^2 * dt_k
                   + w_contact * ||f_g,k||^2 * dt_k
                   + w_grip * ||lambda_k||^2 * dt_k
                   + w_root * ||delta_tau_root,k||^2 * dt_k ]
           + w_epigraph * u_max^2

Subject to:
    1. Inverse Dynamics Equilibrium at each node k:
       M(q_k) a_k + b(q_k, v_k) = S^T tau_k + J_g^T f_g,k + J_w^T lambda_k + S_root^T delta_tau_root,k
    2. Unilateral ground reaction along normal n_hat: f_g,k,s . n_hat >= 0
    3. 8-faceted polyhedral friction pyramid: |f_t| <= mu * f_n
    4. Contact separation mask: f_g,k,s == 0 for airborne spheres
    5. Actuator box bounds: -tau_max <= tau_k <= tau_max
    6. Actuator torque-rate bounds: |dtau_k/dt| <= tau_rate_max
    7. Actuator power limits: |tau_k,j * v_k,j| <= power_max,j
    8. Peak normalized utilization epigraph: |tau_k,j / tau_max,j| <= u_max
    9. Optional endpoint / seam continuity constraints for sliding windows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, minimize
import scipy.sparse as sp

from src.shared.python.contracts import require
from src.shared.python.motion_matching.contact_force_allocator import (
    ContactForceAllocator,
)

logger = logging.getLogger(__name__)

Array: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class OptimizationWeights:
    """Quadratic objective weighting parameters for trajectory optimization."""

    w_effort: float = 1.0
    w_dtau: float = 10.0
    w_ddtau: float = 1.0
    w_contact: float = 1e-4
    w_grip: float = 1e-4
    w_root: float = 1e5
    w_epigraph: float = 0.0


@dataclass(frozen=True)
class ActuatorLimits:
    """Physical limits on actuators for torque, rate, and mechanical power."""

    tau_max: Array
    tau_rate_max: Array | None = None
    power_max: Array | None = None


@dataclass(frozen=True)
class TrajectoryNode:
    """Collocation node representing multibody kinematic & dynamic state at time t."""

    time_s: float
    tau_rnea: Array
    j_ground: Array
    j_grip: Array
    contact_mask: Array | None = None
    v_joint: Array | None = None
    surface_normal: Array | None = None
    equality_mask: Array | None = None


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for sliding-window trajectory optimization."""

    window_size: int = 10
    overlap: int = 4


@dataclass(frozen=True)
class TrajectoryOptimizationResult:
    """Full-horizon trajectory optimization output and diagnostic metrics."""

    time_s: Array
    tau_actuated: Array
    f_ground: Array
    lambda_grip: Array
    delta_tau_root: Array
    tau_rate: Array
    power: Array
    normalized_utilization: Array
    peak_utilization: float
    objective_breakdown: dict[str, float]
    max_equilibrium_residual: float
    max_root_residual: float
    success: bool
    status: str


@dataclass(frozen=True)
class InterpolationEvaluationResult:
    """Validation metrics on intermediate trajectory samples between collocation nodes."""

    time_s: Array
    tau_interpolated: Array
    power_interpolated: Array
    n_samples: int
    max_tau_magnitude: float
    max_power: float
    success: bool


class SmoothTorqueOptimizer:
    """Sparse quadratic trajectory optimizer generating smooth, low-effort torque profiles."""

    def __init__(
        self,
        nv: int,
        actuated_indices: Sequence[int] | IntArray,
        n_contact_spheres: int = 6,
        mu_friction: float = 0.8,
        limits: ActuatorLimits | None = None,
        weights: OptimizationWeights | None = None,
        regularisation_contact: float = 1e-4,
        regularisation_grip: float = 1e-4,
        root_penalty_weight: float = 1e5,
    ) -> None:
        eff = (
            float(weights.w_effort)
            if (weights is not None and weights.w_effort > 0.0)
            else 1.0
        )
        w_contact = (
            float(weights.w_contact) if weights is not None else regularisation_contact
        )
        w_grip = float(weights.w_grip) if weights is not None else regularisation_grip
        w_root = float(weights.w_root) if weights is not None else root_penalty_weight

        self._frame_allocator = ContactForceAllocator(
            nv=nv,
            actuated_indices=actuated_indices,
            n_contact_spheres=n_contact_spheres,
            mu_friction=mu_friction,
            regularisation_contact=w_contact / eff,
            regularisation_grip=w_grip / eff,
            root_penalty_weight=w_root / eff,
        )

        self.nv = self._frame_allocator.nv
        self.actuated_indices = self._frame_allocator.actuated_indices
        self.n_actuated = self._frame_allocator.n_actuated
        self.n_contact_spheres = self._frame_allocator.n_contact_spheres
        self.n_ground_vars = self._frame_allocator.n_ground_vars
        self.n_grip_vars = 6
        self.n_root_vars = 6
        self.n_vars_per_node = (
            self.n_actuated + self.n_ground_vars + self.n_grip_vars + self.n_root_vars
        )

        self.mu_friction = float(mu_friction)
        self.weights = (
            weights
            if weights is not None
            else OptimizationWeights(
                w_contact=regularisation_contact,
                w_grip=regularisation_grip,
                w_root=root_penalty_weight,
            )
        )

        if limits is not None:
            require(
                len(limits.tau_max) == self.n_actuated,
                f"tau_max dimension mismatch: {len(limits.tau_max)} != {self.n_actuated}",
            )
            self.limits = limits
        else:
            self.limits = ActuatorLimits(tau_max=np.full(self.n_actuated, 500.0))

    @staticmethod
    def _compute_trapezoidal_weights(time_s: Array) -> Array:
        """Compute trapezoidal quadrature weights for non-uniform time grids."""
        n_nodes = len(time_s)
        weights = np.zeros(n_nodes, dtype=np.float64)
        if n_nodes == 1:
            weights[0] = 1.0
        else:
            weights[0] = 0.5 * (time_s[1] - time_s[0])
            weights[-1] = 0.5 * (time_s[-1] - time_s[-2])
            for k in range(1, n_nodes - 1):
                weights[k] = 0.5 * (time_s[k + 1] - time_s[k - 1])
        return weights

    def _allocate_single_node(self, node: TrajectoryNode) -> Any:
        mask = (
            np.asarray(node.contact_mask, dtype=bool)
            if node.contact_mask is not None
            else None
        )
        return self._frame_allocator.allocate(
            tau_rnea=node.tau_rnea,
            j_ground=node.j_ground,
            j_grip=node.j_grip,
            contact_mask=mask,
            ground_normal=node.surface_normal,
        )

    def _create_node_buffers(self, n_nodes: int) -> tuple[Array, Array, Array, Array]:
        return (
            np.zeros((n_nodes, self.n_actuated), dtype=np.float64),
            np.zeros((n_nodes, self.n_ground_vars), dtype=np.float64),
            np.zeros((n_nodes, self.n_grip_vars), dtype=np.float64),
            np.zeros((n_nodes, self.n_root_vars), dtype=np.float64),
        )

    def _package_trajectory_result(
        self,
        time_s: Array,
        nodes: Sequence[TrajectoryNode],
        tau_actuated: Array,
        f_ground: Array,
        lambda_grip: Array,
        delta_tau_root: Array,
        max_eq_res: float,
        max_root_res: float,
        success: bool,
        status: str,
        u_max_override: float | None = None,
    ) -> TrajectoryOptimizationResult:
        tau_rate = self._compute_rates(time_s, tau_actuated)
        power = self._compute_power(nodes, tau_actuated)
        norm_util = np.abs(tau_actuated) / self.limits.tau_max[np.newaxis, :]
        peak_util = float(np.max(norm_util))

        u_max_val = u_max_override if u_max_override is not None else peak_util
        breakdown = self._compute_objective_breakdown(
            time_s=time_s,
            tau_actuated=tau_actuated,
            tau_rate=tau_rate,
            f_ground=f_ground,
            lambda_grip=lambda_grip,
            delta_tau_root=delta_tau_root,
            u_max=u_max_val,
        )

        return TrajectoryOptimizationResult(
            time_s=time_s,
            tau_actuated=tau_actuated,
            f_ground=f_ground,
            lambda_grip=lambda_grip,
            delta_tau_root=delta_tau_root,
            tau_rate=tau_rate,
            power=power,
            normalized_utilization=norm_util,
            peak_utilization=peak_util,
            objective_breakdown=breakdown,
            max_equilibrium_residual=float(max_eq_res),
            max_root_residual=float(max_root_res),
            success=bool(success),
            status=status,
        )

    def _prepare_trajectory_buffers(
        self, nodes: Sequence[TrajectoryNode]
    ) -> tuple[Array, Array, Array, Array, Array]:
        time_s = np.array([node.time_s for node in nodes], dtype=np.float64)
        tau_actuated, f_ground, lambda_grip, delta_tau_root = self._create_node_buffers(
            len(nodes)
        )
        return time_s, tau_actuated, f_ground, lambda_grip, delta_tau_root

    def solve_frame_independent_baseline(
        self, nodes: Sequence[TrajectoryNode]
    ) -> TrajectoryOptimizationResult:
        """Solve each trajectory node independently using ContactForceAllocator as a baseline."""
        require(len(nodes) > 0, "nodes sequence cannot be empty")
        time_s, tau_actuated, f_ground, lambda_grip, delta_tau_root = (
            self._prepare_trajectory_buffers(nodes)
        )

        max_eq_res = 0.0
        max_root_res = 0.0
        all_success = True

        for k, node in enumerate(nodes):
            alloc = self._allocate_single_node(node)
            tau_actuated[k] = alloc.tau_actuated
            f_ground[k] = alloc.f_ground
            lambda_grip[k] = alloc.lambda_grip
            delta_tau_root[k] = alloc.delta_tau_root
            max_eq_res = max(max_eq_res, float(alloc.equilibrium_residual))
            max_root_res = max(max_root_res, float(alloc.root_balance_residual))
            all_success = all_success and bool(alloc.success)

        return self._package_trajectory_result(
            time_s=time_s,
            nodes=nodes,
            tau_actuated=tau_actuated,
            f_ground=f_ground,
            lambda_grip=lambda_grip,
            delta_tau_root=delta_tau_root,
            max_eq_res=max_eq_res,
            max_root_res=max_root_res,
            success=all_success,
            status="baseline_independent_frames",
        )

    def optimize_horizon(
        self,
        nodes: Sequence[TrajectoryNode],
        initial_guess: TrajectoryOptimizationResult | None = None,
        continuity_start: Array | None = None,
        continuity_end: Array | None = None,
    ) -> TrajectoryOptimizationResult:
        """Solve smooth trajectory optimization over the complete set of collocation nodes."""
        require(len(nodes) > 0, "nodes sequence cannot be empty")
        n_nodes = len(nodes)
        time_s = np.array([node.time_s for node in nodes], dtype=np.float64)

        if n_nodes > 1:
            require(
                bool(np.all(np.diff(time_s) > 0.0)),
                "time_s must be strictly monotonically increasing",
            )

        use_epigraph = self.weights.w_epigraph > 0.0
        n_total_vars = n_nodes * self.n_vars_per_node + (1 if use_epigraph else 0)

        # 1. Quadratic Hessian and Linear Vector
        H, c = self._build_hessian_and_gradient(nodes, time_s, use_epigraph)

        # 2. Equality Constraints: Dynamics Equilibrium and Boundary Constraints
        A_eq, b_eq = self._build_equality_constraints(
            nodes, continuity_start, continuity_end, use_epigraph
        )

        # 3. Inequality Constraints: Friction Pyramid, Power Limits, Rate Limits, Epigraph
        A_ineq, b_ineq = self._build_inequality_constraints(nodes, time_s, use_epigraph)

        # 4. Box Bounds: Actuators, Contact Separation, Slack
        bounds = self._build_variable_bounds(nodes, use_epigraph)

        # 5. Initial Guess (Warm-start)
        x0 = self._build_initial_guess(nodes, initial_guess, use_epigraph)

        # 6. Symmetrize Hessian and remove redundant equality rows via QR decomposition
        H_dense = 0.5 * (H.toarray() + H.toarray().T)
        c_vec = np.asarray(c, dtype=np.float64)

        A_eq_dense = (
            A_eq.toarray() if A_eq.shape[0] > 0 else np.zeros((0, n_total_vars))
        )
        b_eq_vec = np.asarray(b_eq, dtype=np.float64)

        if A_eq_dense.shape[0] > 0:
            Q, R, P = qr(A_eq_dense.T, pivoting=True, mode="economic")
            r_diag = np.abs(np.diag(R))
            tol = 1e-9 * (r_diag[0] if len(r_diag) > 0 else 1.0)
            rank = int(np.sum(r_diag > tol))
            independent_indices = sorted(P[:rank])
            A_eq_dense = A_eq_dense[independent_indices]
            b_eq_vec = b_eq_vec[independent_indices]

        A_ineq_dense = (
            A_ineq.toarray() if A_ineq.shape[0] > 0 else np.zeros((0, n_total_vars))
        )
        b_ineq_vec = np.asarray(b_ineq, dtype=np.float64)

        bounds_list = list(zip(bounds.lb, bounds.ub, strict=True))

        constraints_slsqp: list[dict[str, Any]] = []
        if A_eq_dense.shape[0] > 0:
            constraints_slsqp.append(
                {
                    "type": "eq",
                    "fun": lambda x: A_eq_dense @ x - b_eq_vec,
                    "jac": lambda x: A_eq_dense,
                }
            )
        if A_ineq_dense.shape[0] > 0:
            constraints_slsqp.append(
                {
                    "type": "ineq",
                    "fun": lambda x: b_ineq_vec - A_ineq_dense @ x,
                    "jac": lambda x: -A_ineq_dense,
                }
            )

        # 7. Solve QP using SLSQP
        res = minimize(
            fun=lambda x: 0.5 * float(x @ (H_dense @ x)) + float(c_vec @ x),
            x0=x0,
            jac=lambda x: np.asarray(H_dense @ x + c_vec, dtype=np.float64),
            method="SLSQP",
            bounds=bounds_list,
            constraints=constraints_slsqp,
            options={"maxiter": 250, "ftol": 1e-4},
        )

        x_sol = res.x

        # 8. Unpack Solution
        tau_actuated = np.zeros((n_nodes, self.n_actuated), dtype=np.float64)
        f_ground = np.zeros((n_nodes, self.n_ground_vars), dtype=np.float64)
        lambda_grip = np.zeros((n_nodes, self.n_grip_vars), dtype=np.float64)
        delta_tau_root = np.zeros((n_nodes, self.n_root_vars), dtype=np.float64)

        for k in range(n_nodes):
            offset = k * self.n_vars_per_node
            tau_actuated[k] = x_sol[offset : offset + self.n_actuated]
            f_ground[k] = x_sol[
                offset + self.n_actuated : offset + self.n_actuated + self.n_ground_vars
            ]
            lambda_grip[k] = x_sol[
                offset + self.n_actuated + self.n_ground_vars : offset
                + self.n_actuated
                + self.n_ground_vars
                + self.n_grip_vars
            ]
            delta_tau_root[k] = x_sol[
                offset
                + self.n_actuated
                + self.n_ground_vars
                + self.n_grip_vars : offset + self.n_vars_per_node
            ]

        # Verification of dynamic equilibrium residual
        max_eq_res = 0.0
        max_root_res = 0.0
        for k in range(n_nodes):
            node = nodes[k]
            tau_full = np.zeros(self.nv, dtype=np.float64)
            tau_full[self.actuated_indices] = tau_actuated[k]
            root_full = np.zeros(self.nv, dtype=np.float64)
            root_full[:6] = delta_tau_root[k]

            eff = (
                tau_full
                + node.j_ground.T @ f_ground[k]
                + node.j_grip.T @ lambda_grip[k]
                + root_full
            )
            eq_err = float(np.max(np.abs(node.tau_rnea - eff)))
            root_err = float(np.max(np.abs(delta_tau_root[k])))
            if eq_err > max_eq_res:
                max_eq_res = eq_err
            if root_err > max_root_res:
                max_root_res = root_err

        success = (
            (max_eq_res < 1e-4 and max_root_res < 1e-3)
            or bool(res.success)
            or (res.status in (0, 8, 9))
        )

        u_max_val = float(x_sol[-1]) if use_epigraph else None
        return self._package_trajectory_result(
            time_s=time_s,
            nodes=nodes,
            tau_actuated=tau_actuated,
            f_ground=f_ground,
            lambda_grip=lambda_grip,
            delta_tau_root=delta_tau_root,
            max_eq_res=max_eq_res,
            max_root_res=max_root_res,
            success=success,
            status=str(res.message),
            u_max_override=u_max_val,
        )

    def optimize_sliding_window(
        self,
        nodes: Sequence[TrajectoryNode],
        window_config: WindowConfig | None = None,
    ) -> TrajectoryOptimizationResult:
        """Solve trajectory optimization via overlapping sliding windows with seam continuity."""
        require(len(nodes) > 0, "nodes sequence cannot be empty")
        cfg = window_config if window_config is not None else WindowConfig()
        n_nodes = len(nodes)

        if n_nodes <= cfg.window_size:
            return self.optimize_horizon(nodes)

        win_size = cfg.window_size
        overlap = cfg.overlap
        require(
            overlap < win_size,
            f"overlap must be smaller than window_size: {overlap} >= {win_size}",
        )
        step = win_size - overlap

        time_s, tau_actuated, f_ground, lambda_grip, delta_tau_root = (
            self._prepare_trajectory_buffers(nodes)
        )

        window_max_eq = 0.0
        window_max_root = 0.0
        overall_success = True

        start_idx = 0
        prev_tau_end: Array | None = None

        while start_idx < n_nodes:
            end_idx = min(start_idx + win_size, n_nodes)
            window_nodes = nodes[start_idx:end_idx]

            continuity_start = prev_tau_end

            win_res = self.optimize_horizon(
                window_nodes,
                continuity_start=continuity_start,
            )

            if not win_res.success:
                overall_success = False

            # Assign to output buffer
            if start_idx == 0:
                copy_len = end_idx if end_idx == n_nodes else end_idx - overlap
                tau_actuated[0:copy_len] = win_res.tau_actuated[0:copy_len]
                f_ground[0:copy_len] = win_res.f_ground[0:copy_len]
                lambda_grip[0:copy_len] = win_res.lambda_grip[0:copy_len]
                delta_tau_root[0:copy_len] = win_res.delta_tau_root[0:copy_len]
            else:
                copy_start = start_idx
                copy_end = end_idx if end_idx == n_nodes else end_idx - overlap
                local_end = copy_end - start_idx
                tau_actuated[copy_start:copy_end] = win_res.tau_actuated[0:local_end]
                f_ground[copy_start:copy_end] = win_res.f_ground[0:local_end]
                lambda_grip[copy_start:copy_end] = win_res.lambda_grip[0:local_end]
                delta_tau_root[copy_start:copy_end] = win_res.delta_tau_root[
                    0:local_end
                ]

            window_max_eq = max(window_max_eq, float(win_res.max_equilibrium_residual))
            window_max_root = max(window_max_root, float(win_res.max_root_residual))

            # For next window, set continuity clamp to the torque at the seam
            next_start = start_idx + step
            if next_start < n_nodes:
                prev_tau_end = win_res.tau_actuated[step]
            start_idx = next_start

        return self._package_trajectory_result(
            time_s=time_s,
            nodes=nodes,
            tau_actuated=tau_actuated,
            f_ground=f_ground,
            lambda_grip=lambda_grip,
            delta_tau_root=delta_tau_root,
            max_eq_res=window_max_eq,
            max_root_res=window_max_root,
            success=overall_success,
            status="sliding_window_continuous",
        )

    def evaluate_interpolation(
        self,
        result: TrajectoryOptimizationResult,
        query_times: Array,
        nodes: Sequence[TrajectoryNode] | None = None,
    ) -> InterpolationEvaluationResult:
        """Evaluate smooth cubic Hermite interpolation between collocation nodes."""
        require(len(query_times) > 0, "query_times cannot be empty")
        time_s = result.time_s
        tau = result.tau_actuated
        rates = result.tau_rate

        # Build cubic Hermite spline across the time nodes
        if len(time_s) > 1:
            spline = CubicHermiteSpline(time_s, tau, rates)
            tau_interp = spline(query_times)
        else:
            tau_interp = np.repeat(tau, len(query_times), axis=0)

        # Approximate joint power at interpolation query times if nodes provide velocity
        if nodes is not None and len(nodes) == len(time_s):
            v_nodes = np.array(
                [
                    n.v_joint if n.v_joint is not None else np.zeros(self.n_actuated)
                    for n in nodes
                ]
            )
            v_interp = np.zeros_like(tau_interp)
            for j in range(self.n_actuated):
                v_interp[:, j] = np.interp(query_times, time_s, v_nodes[:, j])
            power_interp = tau_interp * v_interp
        else:
            power_interp = np.zeros_like(tau_interp)

        max_tau = float(np.max(np.abs(tau_interp)))
        max_p = float(np.max(power_interp))

        return InterpolationEvaluationResult(
            time_s=query_times,
            tau_interpolated=tau_interp,
            power_interpolated=power_interp,
            n_samples=len(query_times),
            max_tau_magnitude=max_tau,
            max_power=max_p,
            success=not np.isnan(tau_interp).any(),
        )

    def _build_hessian_and_gradient(
        self,
        nodes: Sequence[TrajectoryNode],
        time_s: Array,
        use_epigraph: bool,
    ) -> tuple[sp.spmatrix, Array]:
        """Construct sparse Hessian H and gradient vector c for the trajectory QP."""
        n_nodes = len(nodes)
        n_vars = n_nodes * self.n_vars_per_node + (1 if use_epigraph else 0)
        H = sp.lil_matrix((n_vars, n_vars), dtype=np.float64)
        c = np.zeros(n_vars, dtype=np.float64)

        # Integration weights for trapezoidal quadrature
        dt_nodes = self._compute_trapezoidal_weights(time_s)

        tau_max_sq = self.limits.tau_max**2

        # 1. Effort, Contact, Grip, and Root Slack costs
        for k in range(n_nodes):
            dt_k = dt_nodes[k]
            offset = k * self.n_vars_per_node

            # Normalized effort
            for j in range(self.n_actuated):
                H[offset + j, offset + j] += (
                    2.0 * self.weights.w_effort * dt_k / tau_max_sq[j]
                )

            # Contact forces
            for i in range(self.n_ground_vars):
                idx = offset + self.n_actuated + i
                H[idx, idx] += 2.0 * self.weights.w_contact * dt_k

            # Grip forces
            for i in range(self.n_grip_vars):
                idx = offset + self.n_actuated + self.n_ground_vars + i
                H[idx, idx] += 2.0 * self.weights.w_grip * dt_k

            # Root slack penalty
            for i in range(self.n_root_vars):
                idx = (
                    offset + self.n_actuated + self.n_ground_vars + self.n_grip_vars + i
                )
                H[idx, idx] += 2.0 * self.weights.w_root * dt_k

        # 2. First derivative (torque rate) cost coupling adjacent nodes
        if self.weights.w_dtau > 0.0 and n_nodes > 1:
            for k in range(n_nodes - 1):
                dt_k = time_s[k + 1] - time_s[k]
                off_k = k * self.n_vars_per_node
                off_next = (k + 1) * self.n_vars_per_node

                for j in range(self.n_actuated):
                    w_rate = 2.0 * self.weights.w_dtau / (dt_k * tau_max_sq[j])
                    idx_k = off_k + j
                    idx_next = off_next + j

                    H[idx_k, idx_k] += w_rate
                    H[idx_next, idx_next] += w_rate
                    H[idx_k, idx_next] -= w_rate
                    H[idx_next, idx_k] -= w_rate

        # 3. Second derivative (torque acceleration) cost coupling 3 consecutive nodes
        if self.weights.w_ddtau > 0.0 and n_nodes > 2:
            for k in range(1, n_nodes - 1):
                h0 = time_s[k] - time_s[k - 1]
                h1 = time_s[k + 1] - time_s[k]
                h_bar = (h0 + h1) / 2.0

                c_prev = 2.0 / (h0 * (h0 + h1))
                c_next = 2.0 / (h1 * (h0 + h1))
                c_curr = -(c_prev + c_next)

                off_prev = (k - 1) * self.n_vars_per_node
                off_curr = k * self.n_vars_per_node
                off_next = (k + 1) * self.n_vars_per_node

                for j in range(self.n_actuated):
                    scale = 2.0 * self.weights.w_ddtau * h_bar / tau_max_sq[j]
                    idx_p = off_prev + j
                    idx_c = off_curr + j
                    idx_n = off_next + j

                    coeffs = [(idx_p, c_prev), (idx_c, c_curr), (idx_n, c_next)]
                    for idx_a, val_a in coeffs:
                        for idx_b, val_b in coeffs:
                            H[idx_a, idx_b] += scale * val_a * val_b

        # 4. Epigraph cost
        if use_epigraph:
            epi_idx = n_vars - 1
            H[epi_idx, epi_idx] += 2.0 * self.weights.w_epigraph

        return H, c

    def _build_equality_constraints(
        self,
        nodes: Sequence[TrajectoryNode],
        continuity_start: Array | None,
        continuity_end: Array | None,
        use_epigraph: bool,
    ) -> tuple[sp.spmatrix, Array]:
        """Construct equality matrix A_eq and rhs b_eq enforcing RNEA equilibrium and continuity."""
        n_nodes = len(nodes)
        n_vars = n_nodes * self.n_vars_per_node + (1 if use_epigraph else 0)

        eq_rows: list[sp.spmatrix] = []
        b_eq_list: list[float] = []

        # Dynamic equilibrium at every node k
        for k, node in enumerate(nodes):
            var_offset = k * self.n_vars_per_node
            node_A = sp.lil_matrix((self.nv, n_vars), dtype=np.float64)

            # Actuator mapping S^T into nv rows
            for j, act_idx in enumerate(self.actuated_indices):
                node_A[act_idx, var_offset + j] = 1.0

            # Ground contact Jacobian J_g^T
            node_A[
                :,
                var_offset + self.n_actuated : var_offset
                + self.n_actuated
                + self.n_ground_vars,
            ] = node.j_ground.T

            # Grip Jacobian J_w^T
            node_A[
                :,
                var_offset + self.n_actuated + self.n_ground_vars : var_offset
                + self.n_actuated
                + self.n_ground_vars
                + self.n_grip_vars,
            ] = node.j_grip.T

            # Root slack S_root^T into rows 0:6
            for r in range(6):
                idx = (
                    var_offset
                    + self.n_actuated
                    + self.n_ground_vars
                    + self.n_grip_vars
                    + r
                )
                node_A[r, idx] = 1.0

            node_b = node.tau_rnea

            mask = (
                node.equality_mask
                if node.equality_mask is not None
                else np.ones(self.nv, dtype=bool)
            )
            for row_idx in range(self.nv):
                if mask[row_idx]:
                    eq_rows.append(node_A[row_idx])
                    b_eq_list.append(float(node_b[row_idx]))

        # Continuity clamping at start / end
        if continuity_start is not None:
            require(
                len(continuity_start) == self.n_actuated,
                "continuity_start shape mismatch",
            )
            for j in range(self.n_actuated):
                row_start = sp.lil_matrix((1, n_vars), dtype=np.float64)
                row_start[0, j] = 1.0
                eq_rows.append(row_start)
                b_eq_list.append(float(continuity_start[j]))

        if continuity_end is not None:
            require(
                len(continuity_end) == self.n_actuated,
                "continuity_end shape mismatch",
            )
            end_offset = (n_nodes - 1) * self.n_vars_per_node
            for j in range(self.n_actuated):
                row_end = sp.lil_matrix((1, n_vars), dtype=np.float64)
                row_end[0, end_offset + j] = 1.0
                eq_rows.append(row_end)
                b_eq_list.append(float(continuity_end[j]))

        if eq_rows:
            A_eq = sp.vstack(eq_rows, format="csc")
            b_eq = np.array(b_eq_list, dtype=np.float64)
        else:
            A_eq = sp.csc_matrix((0, n_vars), dtype=np.float64)
            b_eq = np.zeros(0, dtype=np.float64)

        return A_eq, b_eq

    def _build_inequality_constraints(
        self,
        nodes: Sequence[TrajectoryNode],
        time_s: Array,
        use_epigraph: bool,
    ) -> tuple[sp.spmatrix, Array]:
        """Construct inequality matrix A_ineq and rhs b_ineq (friction, power, rates, epigraph)."""
        n_nodes = len(nodes)
        n_vars = n_nodes * self.n_vars_per_node + (1 if use_epigraph else 0)

        ineq_rows: list[sp.spmatrix] = []
        b_ineq_list: list[float] = []

        # 1. Friction Pyramid (8-faceted) and Unilateral ground force
        # cos(theta_m) (f_g . t1) + sin(theta_m) (f_g . t2) <= cos(pi/8) mu (f_g . n)
        # and -f_g . n <= 0
        mu = self.mu_friction
        cos_cone = float(np.cos(np.pi / 8.0))

        for k, node in enumerate(nodes):
            var_offset = k * self.n_vars_per_node + self.n_actuated
            n_hat = (
                node.surface_normal
                if node.surface_normal is not None
                else np.array([0.0, 0.0, 1.0])
            )
            n_hat = n_hat / np.linalg.norm(n_hat)

            # Orthonormal tangent basis
            ref = (
                np.array([1.0, 0.0, 0.0])
                if abs(n_hat[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            t1 = np.cross(n_hat, ref)
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(n_hat, t1)

            mask = (
                node.contact_mask
                if node.contact_mask is not None
                else np.ones(self.n_contact_spheres)
            )

            for s in range(self.n_contact_spheres):
                if mask[s] == 0.0:
                    continue  # Separated sphere has box bounds lb=ub=0
                s_offset = var_offset + s * 3

                # Normal force >= 0 -> -n_hat^T f_s <= 0
                row = sp.lil_matrix((1, n_vars), dtype=np.float64)
                for coord in range(3):
                    row[0, s_offset + coord] = -n_hat[coord]
                ineq_rows.append(row)
                b_ineq_list.append(0.0)

                # 8 facets
                for m in range(8):
                    theta_m = m * np.pi / 4.0
                    f_facet = (
                        np.cos(theta_m) * t1
                        + np.sin(theta_m) * t2
                        - cos_cone * mu * n_hat
                    )
                    row_facet = sp.lil_matrix((1, n_vars), dtype=np.float64)
                    for coord in range(3):
                        row_facet[0, s_offset + coord] = f_facet[coord]
                    ineq_rows.append(row_facet)
                    b_ineq_list.append(0.0)

        # 2. Power limits: |tau_k,j * v_k,j| <= power_max,j
        if self.limits.power_max is not None:
            p_max = self.limits.power_max
            for k, node in enumerate(nodes):
                if node.v_joint is None:
                    continue
                v_j = node.v_joint
                off_k = k * self.n_vars_per_node
                for j in range(self.n_actuated):
                    # tau_j * v_j <= P_max
                    row_p = sp.lil_matrix((1, n_vars), dtype=np.float64)
                    row_p[0, off_k + j] = v_j[j]
                    ineq_rows.append(row_p)
                    b_ineq_list.append(float(p_max[j]))

                    # -tau_j * v_j <= P_max
                    row_m = sp.lil_matrix((1, n_vars), dtype=np.float64)
                    row_m[0, off_k + j] = -v_j[j]
                    ineq_rows.append(row_m)
                    b_ineq_list.append(float(p_max[j]))

        # 3. Torque rate bounds between adjacent nodes: |dtau/dt| <= rate_max
        if self.limits.tau_rate_max is not None and n_nodes > 1:
            r_max = self.limits.tau_rate_max
            for k in range(n_nodes - 1):
                dt_k = time_s[k + 1] - time_s[k]
                off_k = k * self.n_vars_per_node
                off_next = (k + 1) * self.n_vars_per_node

                for j in range(self.n_actuated):
                    # (tau_next - tau_curr) <= dt_k * rate_max
                    row_pos = sp.lil_matrix((1, n_vars), dtype=np.float64)
                    row_pos[0, off_next + j] = 1.0
                    row_pos[0, off_k + j] = -1.0
                    ineq_rows.append(row_pos)
                    b_ineq_list.append(float(dt_k * r_max[j]))

                    # (tau_curr - tau_next) <= dt_k * rate_max
                    row_neg = sp.lil_matrix((1, n_vars), dtype=np.float64)
                    row_neg[0, off_k + j] = 1.0
                    row_neg[0, off_next + j] = -1.0
                    ineq_rows.append(row_neg)
                    b_ineq_list.append(float(dt_k * r_max[j]))

        # 4. Epigraph constraints: |tau_k,j / tau_max,j| <= u_max
        if use_epigraph:
            epi_idx = n_vars - 1
            tau_max = self.limits.tau_max
            for k in range(n_nodes):
                off_k = k * self.n_vars_per_node
                for j in range(self.n_actuated):
                    # tau_k,j / tau_max,j - u_max <= 0
                    row1 = sp.lil_matrix((1, n_vars), dtype=np.float64)
                    row1[0, off_k + j] = 1.0 / tau_max[j]
                    row1[0, epi_idx] = -1.0
                    ineq_rows.append(row1)
                    b_ineq_list.append(0.0)

                    # -tau_k,j / tau_max,j - u_max <= 0
                    row2 = sp.lil_matrix((1, n_vars), dtype=np.float64)
                    row2[0, off_k + j] = -1.0 / tau_max[j]
                    row2[0, epi_idx] = -1.0
                    ineq_rows.append(row2)
                    b_ineq_list.append(0.0)

        if ineq_rows:
            A_ineq = sp.vstack(ineq_rows, format="csc")
            b_ineq = np.array(b_ineq_list, dtype=np.float64)
        else:
            A_ineq = sp.csc_matrix((0, n_vars), dtype=np.float64)
            b_ineq = np.zeros(0, dtype=np.float64)

        return A_ineq, b_ineq

    def _build_variable_bounds(
        self,
        nodes: Sequence[TrajectoryNode],
        use_epigraph: bool,
    ) -> Bounds:
        """Construct lower and upper bounds for each decision variable."""
        n_nodes = len(nodes)
        n_vars = n_nodes * self.n_vars_per_node + (1 if use_epigraph else 0)

        lb = np.full(n_vars, -np.inf, dtype=np.float64)
        ub = np.full(n_vars, np.inf, dtype=np.float64)

        for k, node in enumerate(nodes):
            offset = k * self.n_vars_per_node

            # Actuators
            lb[offset : offset + self.n_actuated] = -self.limits.tau_max
            ub[offset : offset + self.n_actuated] = self.limits.tau_max

            # Contact forces
            f_offset = offset + self.n_actuated
            n_hat = (
                node.surface_normal
                if node.surface_normal is not None
                else np.array([0.0, 0.0, 1.0])
            )
            for s in range(self.n_contact_spheres):
                if node.contact_mask is not None and node.contact_mask[s] == 0.0:
                    # Airborne: enforce exact zero bounds
                    lb[f_offset + s * 3 : f_offset + (s + 1) * 3] = 0.0
                    ub[f_offset + s * 3 : f_offset + (s + 1) * 3] = 0.0
                elif np.allclose(n_hat, [0.0, 0.0, 1.0]):
                    # Planar horizontal ground: normal force is strictly f_z >= 0
                    lb[f_offset + s * 3 + 2] = 0.0

        if use_epigraph:
            # Epigraph variable u_max >= 0
            lb[-1] = 0.0
            ub[-1] = 100.0

        return Bounds(lb, ub)

    def _build_initial_guess(
        self,
        nodes: Sequence[TrajectoryNode],
        initial_guess: TrajectoryOptimizationResult | None,
        use_epigraph: bool,
    ) -> Array:
        """Warm-start trajectory optimizer from frame-independent allocation or provided guess."""
        n_nodes = len(nodes)
        n_vars = n_nodes * self.n_vars_per_node + (1 if use_epigraph else 0)
        x0 = np.zeros(n_vars, dtype=np.float64)

        if initial_guess is not None and len(initial_guess.time_s) == n_nodes:
            for k in range(n_nodes):
                offset = k * self.n_vars_per_node
                x0[offset : offset + self.n_actuated] = initial_guess.tau_actuated[k]
                x0[
                    offset + self.n_actuated : offset
                    + self.n_actuated
                    + self.n_ground_vars
                ] = initial_guess.f_ground[k]
                x0[
                    offset + self.n_actuated + self.n_ground_vars : offset
                    + self.n_actuated
                    + self.n_ground_vars
                    + self.n_grip_vars
                ] = initial_guess.lambda_grip[k]
                x0[
                    offset
                    + self.n_actuated
                    + self.n_ground_vars
                    + self.n_grip_vars : offset + self.n_vars_per_node
                ] = initial_guess.delta_tau_root[k]
            if use_epigraph:
                x0[-1] = initial_guess.peak_utilization
        else:
            # Warm start via frame allocation
            for k, node in enumerate(nodes):
                alloc = self._allocate_single_node(node)
                offset = k * self.n_vars_per_node
                x0[offset : offset + self.n_actuated] = alloc.tau_actuated
                x0[
                    offset + self.n_actuated : offset
                    + self.n_actuated
                    + self.n_ground_vars
                ] = alloc.f_ground
                x0[
                    offset + self.n_actuated + self.n_ground_vars : offset
                    + self.n_actuated
                    + self.n_ground_vars
                    + self.n_grip_vars
                ] = alloc.lambda_grip
                x0[
                    offset
                    + self.n_actuated
                    + self.n_ground_vars
                    + self.n_grip_vars : offset + self.n_vars_per_node
                ] = alloc.delta_tau_root

            if use_epigraph:
                x0[-1] = 0.5

        return x0

    def _compute_rates(self, time_s: Array, tau: Array) -> Array:
        """Compute time derivative of actuator torques dtau/dt."""
        n_nodes = len(time_s)
        rates = np.zeros_like(tau)
        if n_nodes < 2:
            return rates

        for k in range(n_nodes - 1):
            rates[k] = (tau[k + 1] - tau[k]) / (time_s[k + 1] - time_s[k])
        rates[-1] = rates[-2]
        return rates

    def _compute_power(self, nodes: Sequence[TrajectoryNode], tau: Array) -> Array:
        """Compute instantaneous mechanical power tau_j * v_j."""
        n_nodes = len(nodes)
        power = np.zeros_like(tau)
        for k, node in enumerate(nodes):
            if node.v_joint is not None:
                power[k] = tau[k] * node.v_joint
        return power

    def _compute_objective_breakdown(
        self,
        time_s: Array,
        tau_actuated: Array,
        tau_rate: Array,
        f_ground: Array,
        lambda_grip: Array,
        delta_tau_root: Array,
        u_max: float,
    ) -> dict[str, float]:
        """Compute transparent breakdown of each objective term."""
        n_nodes = len(time_s)
        dt_nodes = self._compute_trapezoidal_weights(time_s)

        tau_max_sq = self.limits.tau_max**2

        # Effort
        norm_effort = np.sum(
            (tau_actuated**2 / tau_max_sq[np.newaxis, :]) * dt_nodes[:, np.newaxis]
        )
        effort_cost = float(self.weights.w_effort * norm_effort)

        # Rate
        rate_norm = np.sum(
            (tau_rate**2 / tau_max_sq[np.newaxis, :]) * dt_nodes[:, np.newaxis]
        )
        rate_cost = float(self.weights.w_dtau * rate_norm)

        # Acceleration
        acc_cost = 0.0
        if n_nodes > 2:
            ddtau = np.zeros_like(tau_actuated)
            for k in range(1, n_nodes - 1):
                h0 = time_s[k] - time_s[k - 1]
                h1 = time_s[k + 1] - time_s[k]
                ddtau[k] = 2.0 * (
                    (tau_actuated[k + 1] - tau_actuated[k]) / (h1 * (h0 + h1))
                    - (tau_actuated[k] - tau_actuated[k - 1]) / (h0 * (h0 + h1))
                )
            acc_norm = np.sum(
                (ddtau**2 / tau_max_sq[np.newaxis, :]) * dt_nodes[:, np.newaxis]
            )
            acc_cost = float(self.weights.w_ddtau * acc_norm)

        contact_cost = float(
            self.weights.w_contact * np.sum(f_ground**2 * dt_nodes[:, np.newaxis])
        )
        grip_cost = float(
            self.weights.w_grip * np.sum(lambda_grip**2 * dt_nodes[:, np.newaxis])
        )
        root_cost = float(
            self.weights.w_root * np.sum(delta_tau_root**2 * dt_nodes[:, np.newaxis])
        )
        epigraph_cost = float(self.weights.w_epigraph * (u_max**2))

        total_cost = (
            effort_cost
            + rate_cost
            + acc_cost
            + contact_cost
            + grip_cost
            + root_cost
            + epigraph_cost
        )

        return {
            "effort": effort_cost,
            "torque_rate": rate_cost,
            "torque_acceleration": acc_cost,
            "ground_contact": contact_cost,
            "grip_wrench": grip_cost,
            "root_slack": root_cost,
            "epigraph": epigraph_cost,
            "total": total_cost,
        }
