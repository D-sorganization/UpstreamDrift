"""Contact wrenches in the inner solve: unilateral and friction-cone SOCP.

For a rigid multibody system interacting with the ground, the inverse-dynamics
equation of motion at time node ``n`` is::

    Y(q_n, v_n, a_n) pi = B u_n + J_c(q_n)^T lambda_n

where ``u_n`` are actuator joint inputs, ``lambda_n`` are contact forces /
wrenches, and ``J_c`` is the contact Jacobian mapping generalized velocities to
contact point velocities.

Physical contact constraints require:
1. Unilateral normal force bounds: ``f_z >= 0`` (the ground can only push, never pull).
2. Coulomb friction cone constraints (second-order cone / SOCP):
   ``||(f_x, f_y)||_2 <= mu * f_z``, where ``mu > 0`` is the coefficient of friction.

Ground reaction forces (GRF) measured via force plates enter as observations
tracking ``lambda_n`` rather than fixed decision variables.  When GRF measurements
are absent, the system experiences identifiability loss (kinematics alone cannot
distinguish internal contact forces from actuator torques); this module provides
explicit observability diagnostics (condition numbers and observable fractions)
and guarantees that cone violations and non-convergence are never silently clipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.linalg as la
import scipy.optimize as opt
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from src.shared.python.core.contracts import ensure, require

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ConeViolationReport:
    """Explicit report of unilateral and friction-cone violations."""

    unilateral_violations: FloatArray
    cone_violations: FloatArray
    max_unilateral_violation: float
    max_cone_violation: float
    is_cone_feasible: bool


@dataclass(frozen=True)
class ContactIdentifiabilityDiagnostics:
    """Observability and conditioning of contact wrenches and inputs."""

    observable_fraction_with_grf: FloatArray
    observable_fraction_without_grf: FloatArray
    condition_number_with_grf: float
    condition_number_without_grf: float
    rank_with_grf: int
    rank_without_grf: int
    total_variables: int
    tolerance: float

    @property
    def has_identifiability_loss(self) -> bool:
        """True if the system loses observability or conditioning without GRF."""
        return bool(
            self.rank_without_grf < self.total_variables
            or self.condition_number_without_grf > 1e6
            or np.any(self.observable_fraction_without_grf < 0.99)
        )

    @property
    def observable_fraction_loss(self) -> FloatArray:
        """Drop in observable fraction when GRF observations are omitted."""
        return self.observable_fraction_with_grf - self.observable_fraction_without_grf


@dataclass(frozen=True)
class ContactWrenchProblem:
    """Problem specification for contact wrenches in the inner solve."""

    regressors: FloatArray
    input_matrix: FloatArray
    contact_jacobian: FloatArray
    friction_coefficient: float
    parameters: FloatArray | None = None
    measured_grf: FloatArray | None = None
    grf_weight: float = 1.0
    dynamics_weight: FloatArray | None = None
    input_effort_weight: float = 1e-6
    contact_effort_weight: float = 1e-6
    input_smoothness_weight: float = 0.0
    trial_index: IntArray | None = None
    contact_dim: int = 3

    def __post_init__(self) -> None:
        require(
            self.friction_coefficient > 0.0,
            "friction coefficient must be strictly positive",
            self.friction_coefficient,
        )
        require(
            self.contact_dim in (2, 3), "contact_dim must be 2 or 3", self.contact_dim
        )

        # Regressors: either (N, n_v) or (N, n_v, n_pi)
        require(
            self.regressors.ndim in (2, 3),
            "regressors must be 2D or 3D",
            self.regressors.shape,
        )
        n_nodes = self.regressors.shape[0]
        n_v = self.regressors.shape[1]

        if self.regressors.ndim == 3:
            require(
                self.parameters is not None,
                "parameters required when regressors is 3D",
            )
            n_pi = self.regressors.shape[2]
            assert self.parameters is not None
            require(
                self.parameters.shape == (n_pi,),
                "parameters must have shape (n_pi,)",
                self.parameters.shape,
            )

        require(
            self.input_matrix.ndim == 2 and self.input_matrix.shape[0] == n_v,
            "input_matrix rows must match n_v",
            self.input_matrix.shape,
        )

        # Contact Jacobian: either (n_lambda, n_v) or (N, n_lambda, n_v)
        require(
            self.contact_jacobian.ndim in (2, 3),
            "contact_jacobian must be 2D or 3D",
            self.contact_jacobian.shape,
        )
        if self.contact_jacobian.ndim == 2:
            n_lambda, j_nv = self.contact_jacobian.shape
        else:
            require(
                self.contact_jacobian.shape[0] == n_nodes,
                "batched contact_jacobian leading dim must match n_nodes",
            )
            n_lambda = self.contact_jacobian.shape[1]
            j_nv = self.contact_jacobian.shape[2]

        require(j_nv == n_v, "contact_jacobian columns must match n_v", (j_nv, n_v))
        require(
            n_lambda % self.contact_dim == 0,
            f"n_lambda ({n_lambda}) must be a multiple of contact_dim ({self.contact_dim})",
        )

        if self.measured_grf is not None:
            require(
                self.measured_grf.shape == (n_nodes, n_lambda),
                "measured_grf must have shape (n_nodes, n_lambda)",
                self.measured_grf.shape,
            )
            require(self.grf_weight >= 0.0, "grf_weight must be non-negative")

        if self.dynamics_weight is not None:
            require(
                self.dynamics_weight.shape == (n_v,),
                "dynamics_weight shape must match n_v",
                self.dynamics_weight.shape,
            )
            require(
                bool(np.all(self.dynamics_weight >= 0.0)),
                "dynamics_weight entries must be non-negative",
            )

        require(self.input_effort_weight >= 0.0, "input_effort_weight >= 0")
        require(self.contact_effort_weight >= 0.0, "contact_effort_weight >= 0")
        require(self.input_smoothness_weight >= 0.0, "input_smoothness_weight >= 0")

        if self.trial_index is not None:
            require(
                self.trial_index.shape == (n_nodes,),
                "trial_index per node",
                self.trial_index.shape,
            )
            require(
                bool(np.all(np.diff(self.trial_index) >= 0)),
                "trial_index must be sorted",
            )

    @property
    def n_nodes(self) -> int:
        return int(self.regressors.shape[0])

    @property
    def n_rows_per_node(self) -> int:
        return int(self.regressors.shape[1])

    @property
    def n_inputs(self) -> int:
        return int(self.input_matrix.shape[1])

    @property
    def n_contact_forces(self) -> int:
        return int(self.contact_jacobian.shape[-2])

    @property
    def n_contact_points(self) -> int:
        return self.n_contact_forces // self.contact_dim

    def get_torques(self) -> FloatArray:
        """Return inverse dynamics torques (N, n_v)."""
        if self.regressors.ndim == 2:
            return np.asarray(self.regressors, dtype=np.float64)
        assert self.parameters is not None
        return np.asarray(
            np.einsum("nvi,i->nv", self.regressors, self.parameters),
            dtype=np.float64,
        )

    def get_jacobian(self, node: int) -> FloatArray:
        """Return contact Jacobian (n_lambda, n_v) for node."""
        if self.contact_jacobian.ndim == 2:
            return np.asarray(self.contact_jacobian, dtype=np.float64)
        return np.asarray(self.contact_jacobian[node], dtype=np.float64)


@dataclass(frozen=True)
class ContactWrenchSolution:
    """Solution of the contact-wrench inner solve."""

    inputs: FloatArray
    wrenches: FloatArray
    cost: float
    dynamics_residual: FloatArray
    grf_residual: FloatArray | None
    converged: bool
    iterations: int
    status: str
    cone_violations: FloatArray
    max_cone_violation: float
    unilateral_violations: FloatArray
    max_unilateral_violation: float
    is_cone_feasible: bool
    identifiability: ContactIdentifiabilityDiagnostics


def project_friction_cone(
    forces: FloatArray, mu: float, contact_dim: int = 3
) -> FloatArray:
    """Project contact forces onto the Coulomb second-order cone (SOC).

    For each contact point:
    If ``contact_dim == 3``:
        forces are ``(fx, fy, fz)``.
        Unilateral constraint: ``fz >= 0``.
        Friction cone: ``||(fx, fy)||_2 <= mu * fz``.
    If ``contact_dim == 2``:
        forces are ``(fx, fz)``.
        Unilateral constraint: ``fz >= 0``.
        Friction cone: ``|fx| <= mu * fz``.
    """
    require(mu > 0.0, "friction coefficient mu must be positive", mu)
    require(contact_dim in (2, 3), "contact_dim must be 2 or 3", contact_dim)
    orig_shape = forces.shape
    require(
        orig_shape[-1] % contact_dim == 0,
        "last dimension must be a multiple of contact_dim",
        orig_shape,
    )

    flat = np.asarray(forces, dtype=np.float64).reshape(-1, contact_dim)
    projected = np.zeros_like(flat)

    if contact_dim == 3:
        fx = flat[:, 0]
        fy = flat[:, 1]
        fz = flat[:, 2]
        ft = np.hypot(fx, fy)
    else:
        fx = flat[:, 0]
        fy = np.zeros_like(fx)
        fz = flat[:, 1]
        ft = np.abs(fx)

    # Case 1: In the cone
    in_cone = (ft <= mu * fz) & (fz >= 0.0)
    projected[in_cone] = flat[in_cone]

    # Case 2: In polar/dual cone reflection (fz + mu * ft <= 0)
    # The closest point on the cone is the apex (0, 0, 0)
    in_polar = (fz + mu * ft <= 0.0) & ~in_cone
    projected[in_polar] = 0.0

    # Case 3: Outside cone and polar cone: project onto cone boundary ft* = mu * fz*
    outside = ~in_cone & ~in_polar
    if np.any(outside):
        fz_star = (fz[outside] + mu * ft[outside]) / (1.0 + mu**2)
        ft_star = mu * fz_star
        ft_safe = np.maximum(ft[outside], np.finfo(np.float64).tiny)
        scale = ft_star / ft_safe

        if contact_dim == 3:
            projected[outside, 0] = fx[outside] * scale
            projected[outside, 1] = fy[outside] * scale
            projected[outside, 2] = fz_star
        else:
            projected[outside, 0] = fx[outside] * scale
            projected[outside, 1] = fz_star

    return projected.reshape(orig_shape)


def compute_cone_violations(
    forces: FloatArray, mu: float, contact_dim: int = 3, tolerance: float = 1e-6
) -> ConeViolationReport:
    """Quantify unilateral and friction cone violations explicitly without clipping."""
    require(mu > 0.0, "mu must be positive", mu)
    flat = np.asarray(forces, dtype=np.float64).reshape(-1, contact_dim)

    if contact_dim == 3:
        fx = flat[:, 0]
        fy = flat[:, 1]
        fz = flat[:, 2]
        ft = np.hypot(fx, fy)
    else:
        fx = flat[:, 0]
        fz = flat[:, 1]
        ft = np.abs(fx)

    unilateral_viols = np.maximum(0.0, -fz)
    cone_viols = np.maximum(0.0, ft - mu * fz)

    max_unilateral = float(np.max(unilateral_viols)) if unilateral_viols.size else 0.0
    max_cone = float(np.max(cone_viols)) if cone_viols.size else 0.0
    is_feasible = bool(max_unilateral <= tolerance and max_cone <= tolerance)

    return ConeViolationReport(
        unilateral_violations=unilateral_viols,
        cone_violations=cone_viols,
        max_unilateral_violation=max_unilateral,
        max_cone_violation=max_cone,
        is_cone_feasible=is_feasible,
    )


def compute_identifiability_diagnostics(
    problem: ContactWrenchProblem, relative_tolerance: float = 1e-7
) -> ContactIdentifiabilityDiagnostics:
    """Compute observable fractions and condition numbers with vs without GRF."""
    n_u = problem.n_inputs
    n_lambda = problem.n_contact_forces
    n_v = problem.n_rows_per_node
    total_vars = n_u + n_lambda

    w_dyn = (
        problem.dynamics_weight if problem.dynamics_weight is not None else np.ones(n_v)
    )
    w_mat = np.diag(w_dyn)

    # Average or representative contact Jacobian
    if problem.contact_jacobian.ndim == 2:
        j_c = problem.contact_jacobian
    else:
        j_c = np.mean(problem.contact_jacobian, axis=0)

    # Design matrix without GRF: [W * B, W * J_c^T] (n_v x total_vars)
    a_dyn = np.hstack([w_mat @ problem.input_matrix, w_mat @ j_c.T])

    # Design matrix with GRF: [W * B, W * J_c^T; 0, w_grf * I_lambda]
    w_grf = problem.grf_weight if problem.grf_weight > 0.0 else 1.0
    grf_block = np.hstack([np.zeros((n_lambda, n_u)), w_grf * np.eye(n_lambda)])
    a_with_grf = np.vstack([a_dyn, grf_block])

    tiny = float(np.finfo(np.float64).tiny)

    # SVD without GRF
    _, s_no, vt_no = np.linalg.svd(a_dyn, full_matrices=True)
    tol_no = relative_tolerance * max(float(s_no[0]), tiny)
    rank_no = int(np.count_nonzero(s_no > tol_no))

    if rank_no < total_vars:
        cond_no = float("inf")
    else:
        cond_no = float(s_no[0] / max(float(s_no[-1]), tiny))

    v_no = vt_no.T
    obs_frac_no = np.sum(v_no[:, :rank_no] ** 2, axis=1)

    # SVD with GRF
    _, s_with, vt_with = np.linalg.svd(a_with_grf, full_matrices=True)
    tol_with = relative_tolerance * max(float(s_with[0]), tiny)
    rank_with = int(np.count_nonzero(s_with > tol_with))
    cond_with = float(s_with[0] / max(float(s_with[-1]), tiny))

    v_with = vt_with.T
    obs_frac_with = np.sum(v_with[:, :rank_with] ** 2, axis=1)

    return ContactIdentifiabilityDiagnostics(
        observable_fraction_with_grf=obs_frac_with,
        observable_fraction_without_grf=obs_frac_no,
        condition_number_with_grf=cond_with,
        condition_number_without_grf=cond_no,
        rank_with_grf=rank_with,
        rank_without_grf=rank_no,
        total_variables=total_vars,
        tolerance=tol_no,
    )


def _build_node_cost(
    problem: ContactWrenchProblem,
    n: int,
    w_mat: FloatArray,
    tau_n: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Build quadratic cost matrix H and linear term g for node n."""
    n_u = problem.n_inputs
    n_lambda = problem.n_contact_forces
    j_c = problem.get_jacobian(n)
    a_n = np.hstack([w_mat @ problem.input_matrix, w_mat @ j_c.T])
    b_n = w_mat @ tau_n

    grf_meas = problem.measured_grf[n] if problem.measured_grf is not None else None
    w_grf = problem.grf_weight if grf_meas is not None else 0.0

    h_mat = a_n.T @ a_n
    h_mat[:n_u, :n_u] += (problem.input_effort_weight**2) * np.eye(n_u)
    h_mat[n_u:, n_u:] += (problem.contact_effort_weight**2 + w_grf**2) * np.eye(
        n_lambda
    )

    g_vec = a_n.T @ b_n
    if grf_meas is not None:
        g_vec[n_u:] += (w_grf**2) * grf_meas

    return h_mat, g_vec


def _solve_node_slsqp(
    problem: ContactWrenchProblem,
    n: int,
    w_mat: FloatArray,
    tau_n: FloatArray,
    max_iterations: int,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, bool, int, float]:
    """Solve the inner wrench optimization at node n using SLSQP."""
    n_u = problem.n_inputs
    n_lambda = problem.n_contact_forces
    n_c = problem.n_contact_points
    dim = problem.contact_dim
    mu = problem.friction_coefficient

    h_mat, g_vec = _build_node_cost(problem, n, w_mat, tau_n)

    def fun(x: FloatArray) -> float:
        return float(0.5 * x @ (h_mat @ x) - g_vec @ x)

    def jac(x: FloatArray) -> FloatArray:
        return h_mat @ x - g_vec

    bounds: list[tuple[float | None, float | None]] = [(None, None)] * (n_u + n_lambda)
    for c in range(n_c):
        idx = n_u + (c * dim + (dim - 1))
        bounds[idx] = (0.0, None)

    def cone_constraint(x: FloatArray) -> FloatArray:
        lam = x[n_u:].reshape(n_c, dim)
        if dim == 3:
            ft = np.hypot(lam[:, 0], lam[:, 1])
            fz = lam[:, 2]
        else:
            ft = np.abs(lam[:, 0])
            fz = lam[:, 1]
        return mu * fz - ft

    x0 = la.solve(h_mat + 1e-4 * np.eye(n_u + n_lambda), g_vec, assume_a="pos")
    x0[n_u:] = project_friction_cone(x0[n_u:], mu, contact_dim=dim)

    res = opt.minimize(  # type: ignore[call-overload]
        fun,
        x0,
        jac=jac,
        bounds=bounds,
        constraints={"type": "ineq", "fun": cone_constraint},
        method="SLSQP",
        options={"maxiter": max_iterations, "ftol": min(tolerance**2, 1e-12)},
    )

    x_sol = res.x
    viol_check = compute_cone_violations(
        x_sol[n_u:], mu, contact_dim=dim, tolerance=tolerance
    )
    is_success = bool(res.success or (res.status == 8 and viol_check.is_cone_feasible))
    cost = fun(x_sol)
    return x_sol[:n_u], x_sol[n_u:], is_success, int(res.nit), cost


def _solve_contact_wrenches_slsqp(
    problem: ContactWrenchProblem,
    w_mat: FloatArray,
    torques: FloatArray,
    max_iterations: int,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, bool, int, float, str]:
    """Solve all nodes independently via SLSQP."""
    inputs_list: list[FloatArray] = []
    wrenches_list: list[FloatArray] = []
    converged_all = True
    total_iters = 0
    total_cost = 0.0

    for n in range(problem.n_nodes):
        u_n, lam_n, success, iters, cost = _solve_node_slsqp(
            problem, n, w_mat, torques[n], max_iterations, tolerance
        )
        inputs_list.append(u_n)
        wrenches_list.append(lam_n)
        converged_all = converged_all and success
        total_iters += iters
        total_cost += cost

    inputs = np.array(inputs_list, dtype=np.float64)
    wrenches = np.array(wrenches_list, dtype=np.float64)
    status_msg = "converged" if converged_all else "slsqp_iteration_limit"
    return inputs, wrenches, converged_all, total_iters, total_cost, status_msg


def _solve_contact_wrenches_admm(
    problem: ContactWrenchProblem,
    w_mat: FloatArray,
    torques: FloatArray,
    max_iterations: int,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, bool, int, float, str]:
    """Solve the coupled contact wrench problem via ADMM."""
    n_nodes = problem.n_nodes
    n_u = problem.n_inputs
    n_lambda = problem.n_contact_forces
    dim = problem.contact_dim
    mu = problem.friction_coefficient
    w_smooth = problem.input_smoothness_weight
    has_smooth = w_smooth > 0.0 and n_nodes > 2

    q_blocks: list[FloatArray] = []
    g_blocks: list[FloatArray] = []
    for n in range(n_nodes):
        hn, gn = _build_node_cost(problem, n, w_mat, torques[n])
        q_blocks.append(hn)
        g_blocks.append(gn)

    q_diag: Any = sp.block_diag(q_blocks, format="csc").tocsc()
    g_vec = np.concatenate(g_blocks)

    if has_smooth:
        diff_rows = []
        for n in range(1, n_nodes - 1):
            row = np.zeros(n_nodes)
            row[n - 1] = 1.0
            row[n] = -2.0
            row[n + 1] = 1.0
            diff_rows.append(row)
        diff_mat = sp.csr_matrix(np.array(diff_rows))
        smooth_u = sp.kron(diff_mat, sp.eye(n_u))
        col_indices = np.concatenate(
            [
                np.arange(n * (n_u + n_lambda), n * (n_u + n_lambda) + n_u)
                for n in range(n_nodes)
            ]
        )
        smooth_full = sp.csr_matrix(
            (smooth_u.data, (smooth_u.indices, col_indices)),
            shape=(smooth_u.shape[0], n_nodes * (n_u + n_lambda)),
        )
        q_diag = q_diag + (w_smooth**2) * (smooth_full.T @ smooth_full).tocsc()

    lambda_cols = np.concatenate(
        [
            np.arange(n * (n_u + n_lambda) + n_u, (n + 1) * (n_u + n_lambda))
            for n in range(n_nodes)
        ]
    )
    s_lambda = sp.csr_matrix(
        (np.ones(n_nodes * n_lambda), (np.arange(n_nodes * n_lambda), lambda_cols)),
        shape=(n_nodes * n_lambda, n_nodes * (n_u + n_lambda)),
    )

    rho = 10.0
    h_aug = (q_diag + rho * (s_lambda.T @ s_lambda)).tocsc()
    lu_solver = splu(h_aug)

    z: FloatArray = np.zeros(n_nodes * n_lambda)
    y: FloatArray = np.zeros(n_nodes * n_lambda)
    converged_all = False
    iter_count = 0
    x_sol: FloatArray = np.zeros(n_nodes * (n_u + n_lambda))

    for k in range(max_iterations):
        iter_count = k + 1
        rhs = g_vec + s_lambda.T @ (rho * z - y)
        x_sol = lu_solver.solve(rhs)
        lam = s_lambda @ x_sol
        z_prev = z.copy()
        z = project_friction_cone(lam + (1.0 / rho) * y, mu, contact_dim=dim).reshape(
            -1
        )
        r_pri = lam - z
        y = y + rho * r_pri
        r_dual = rho * (z - z_prev)
        if (
            float(np.max(np.abs(r_pri))) <= tolerance
            and float(np.max(np.abs(r_dual))) <= tolerance
        ):
            converged_all = True
            break

    status_msg = "converged" if converged_all else "admm_iteration_limit"
    stacked_x = x_sol.reshape(n_nodes, n_u + n_lambda)
    inputs = stacked_x[:, :n_u]
    wrenches = stacked_x[:, n_u:]
    total_cost = float(0.5 * x_sol @ (q_diag @ x_sol) - g_vec @ x_sol)
    return inputs, wrenches, converged_all, iter_count, total_cost, status_msg


def solve_contact_wrenches(
    problem: ContactWrenchProblem,
    solver: Literal["admm", "slsqp", "auto"] = "auto",
    max_iterations: int = 400,
    tolerance: float = 1e-6,
) -> ContactWrenchSolution:
    """Solve the inner contact wrench problem with unilateral and friction-cone SOCP."""
    n_nodes = problem.n_nodes
    n_u = problem.n_inputs
    n_lambda = problem.n_contact_forces
    n_v = problem.n_rows_per_node
    dim = problem.contact_dim
    mu = problem.friction_coefficient

    w_dyn = (
        problem.dynamics_weight if problem.dynamics_weight is not None else np.ones(n_v)
    )
    w_mat = np.diag(w_dyn)
    torques = problem.get_torques()  # (N, n_v)

    identifiability = compute_identifiability_diagnostics(problem)

    # Choose solver backend
    use_slsqp = solver == "slsqp" or (
        solver == "auto" and problem.input_smoothness_weight == 0.0
    )

    if use_slsqp:
        inputs, wrenches, converged_all, total_iters, total_cost, status_msg = (
            _solve_contact_wrenches_slsqp(
                problem, w_mat, torques, max_iterations, tolerance
            )
        )
    else:
        inputs, wrenches, converged_all, total_iters, total_cost, status_msg = (
            _solve_contact_wrenches_admm(
                problem, w_mat, torques, max_iterations, tolerance
            )
        )

    # Compute dynamics residuals
    dyn_residuals = []
    for n in range(n_nodes):
        j_c = problem.get_jacobian(n)
        tau_pred = problem.input_matrix @ inputs[n] + j_c.T @ wrenches[n]
        dyn_residuals.append(w_mat @ (torques[n] - tau_pred))
    dynamics_residual = np.array(dyn_residuals, dtype=np.float64)

    grf_residual = None
    if problem.measured_grf is not None:
        grf_residual = wrenches - problem.measured_grf

    # Evaluate cone violations explicitly
    viol_report = compute_cone_violations(
        wrenches, mu, contact_dim=dim, tolerance=tolerance
    )

    ensure(inputs.shape == (n_nodes, n_u), "inputs shape")
    ensure(wrenches.shape == (n_nodes, n_lambda), "wrenches shape")

    return ContactWrenchSolution(
        inputs=inputs,
        wrenches=wrenches,
        cost=total_cost,
        dynamics_residual=dynamics_residual,
        grf_residual=grf_residual,
        converged=converged_all,
        iterations=total_iters,
        status=status_msg,
        cone_violations=viol_report.cone_violations,
        max_cone_violation=viol_report.max_cone_violation,
        unilateral_violations=viol_report.unilateral_violations,
        max_unilateral_violation=viol_report.max_unilateral_violation,
        is_cone_feasible=viol_report.is_cone_feasible,
        identifiability=identifiability,
    )
