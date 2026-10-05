"""Variable-projection inner solve for inertial parameters, inputs and template.

For a fixed kinematic trajectory the inverse-dynamics collocation residual

    r_n = W [ Y(q_n, v_n, a_n) pi - B u_n ]

is *linear* in the inner variables ``w = (pi, u_1..N, ubar)``, as are all the
priors (anthropometric prior on ``pi``, effort and smoothness on ``u``, a shared
phase-indexed torque template ``ubar`` coupling trials, and linear anchors such
as a known total mass).  The inner problem is therefore a sparse linear least
squares solved exactly by normal equations; the outer Gauss-Newton step over
the kinematic variables then uses the reduced (Schur-complement / Kaufman)
Hessian, see :func:`reduced_gauss_newton`.  This is the Golub-Pereyra
variable-projection method applied to dynamics matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.linalg import SuperLU, splu

from src.shared.python.core.contracts import ensure, require

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
_NORMAL_RIDGE = 1e-14


@dataclass(frozen=True)
class LinearAnchor:
    """Soft linear constraint ``weight * (coefficients . pi - value) = 0``."""

    coefficients: FloatArray
    value: float
    weight: float

    def __post_init__(self) -> None:
        require(self.coefficients.ndim == 1, "anchor coefficients must be 1-D")
        require(self.weight > 0.0, "anchor weight must be positive", self.weight)


@dataclass(frozen=True)
class InnerProblem:
    """Data of one inner solve; ``trial_index`` must be sorted (contiguous trials)."""

    regressors: FloatArray
    input_matrix: FloatArray
    trial_index: IntArray
    phase_index: IntArray | None
    dynamics_weight: FloatArray
    parameter_prior_mean: FloatArray
    parameter_prior_weight: FloatArray
    input_effort_weight: float
    input_smoothness_weight: float
    template_weight: float
    anchors: tuple[LinearAnchor, ...] = ()
    fixed_parameters: FloatArray | None = None

    def __post_init__(self) -> None:
        n_nodes, n_v, n_pi = self.regressors.shape
        if self.fixed_parameters is not None:
            require(
                self.fixed_parameters.shape == (n_pi,), "fixed_parameters per parameter"
            )
            require(
                len(self.anchors) == 0,
                "anchors are outer rows when parameters are fixed",
            )
        require(
            self.input_matrix.shape[0] == n_v,
            "input matrix rows = n_v",
            self.input_matrix.shape,
        )
        require(self.trial_index.shape == (n_nodes,), "trial_index per node")
        require(
            bool(np.all(np.diff(self.trial_index) >= 0)), "trial_index must be sorted"
        )
        require(self.dynamics_weight.shape == (n_v,), "dynamics_weight per row")
        require(self.parameter_prior_mean.shape == (n_pi,), "prior mean per parameter")
        require(
            self.parameter_prior_weight.shape == (n_pi,), "prior weight per parameter"
        )
        require(self.input_effort_weight >= 0.0, "effort weight >= 0")
        require(self.input_smoothness_weight >= 0.0, "smoothness weight >= 0")
        require(self.template_weight >= 0.0, "template weight >= 0")
        if self.template_weight > 0.0:
            require(self.phase_index is not None, "template coupling needs phase_index")
        for anchor in self.anchors:
            require(anchor.coefficients.shape == (n_pi,), "anchor width = n_pi")

    @property
    def n_nodes(self) -> int:
        return int(self.regressors.shape[0])

    @property
    def n_rows_per_node(self) -> int:
        return int(self.regressors.shape[1])

    @property
    def n_parameters(self) -> int:
        return int(self.regressors.shape[2])

    @property
    def n_inputs(self) -> int:
        return int(self.input_matrix.shape[1])

    @property
    def parameters_are_free(self) -> bool:
        return self.fixed_parameters is None

    @property
    def n_parameter_columns(self) -> int:
        return self.n_parameters if self.parameters_are_free else 0

    @property
    def n_phases(self) -> int:
        if self.template_weight <= 0.0 or self.phase_index is None:
            return 0
        return int(self.phase_index.max()) + 1


@dataclass(frozen=True)
class InnerSolution:
    """Exact minimiser of the inner linear least-squares problem."""

    parameters: FloatArray
    inputs: FloatArray
    template: FloatArray | None
    stacked: FloatArray
    residual: FloatArray
    rhs: FloatArray
    design: sp.csr_matrix
    normal_lu: SuperLU
    n_dynamics_rows: int
    parameters_free: bool

    @property
    def parameter_prior_rows(self) -> slice:
        """Rows of ``design`` holding the parameter prior (directly after dynamics)."""
        require(
            self.parameters_free, "no parameter prior rows when parameters are fixed"
        )
        start = self.n_dynamics_rows
        return slice(start, start + self.parameters.size)

    @property
    def cost(self) -> float:
        return float(self.residual @ self.residual)


def _second_difference_rows(trial_index: IntArray) -> sp.csr_matrix:
    """Rows ``u[n-1] - 2 u[n] + u[n+1]`` for consecutive nodes of the same trial."""
    n_nodes = trial_index.size
    interior = np.flatnonzero(
        (np.r_[False, trial_index[1:] == trial_index[:-1]])
        & (np.r_[trial_index[:-1] == trial_index[1:], False])
    )
    rows = np.repeat(np.arange(interior.size), 3)
    cols = (interior[:, None] + np.array([-1, 0, 1])[None, :]).ravel()
    data = np.tile(np.array([1.0, -2.0, 1.0]), interior.size)
    return sp.csr_matrix((data, (rows, cols)), shape=(interior.size, n_nodes))


def _dynamics_blocks(
    problem: InnerProblem,
) -> tuple[sp.csr_matrix | None, sp.csr_matrix, FloatArray]:
    """Return ``(pi block or None, u block, rhs)`` of the weighted dynamics rows."""
    weights = problem.dynamics_weight[None, :, None]
    weighted = (weights * problem.regressors).reshape(-1, problem.n_parameters)
    u_block = sp.kron(
        sp.eye(problem.n_nodes),
        -(problem.dynamics_weight[:, None] * problem.input_matrix),
    )
    if problem.parameters_are_free:
        return (
            sp.csr_matrix(weighted),
            sp.csr_matrix(u_block),
            np.zeros(weighted.shape[0]),
        )
    fixed = problem.fixed_parameters
    require(fixed is not None, "fixed parameters expected")
    assert fixed is not None  # narrowed for the type checker after the contract
    return None, sp.csr_matrix(u_block), -(weighted @ fixed)


def _input_prior_blocks(
    problem: InnerProblem,
) -> list[tuple[sp.csr_matrix, sp.csr_matrix | None]]:
    """Return ``(u-block, template-block)`` row groups for effort/smoothness/template."""
    n_u, n_nodes = problem.n_inputs, problem.n_nodes
    eye_u = sp.eye(n_u)
    groups: list[tuple[sp.csr_matrix, sp.csr_matrix | None]] = []
    if problem.input_effort_weight > 0.0:
        groups.append(
            (sp.csr_matrix(problem.input_effort_weight * sp.eye(n_nodes * n_u)), None)
        )
    if problem.input_smoothness_weight > 0.0:
        diff = _second_difference_rows(problem.trial_index)
        groups.append(
            (
                sp.csr_matrix(problem.input_smoothness_weight * sp.kron(diff, eye_u)),
                None,
            )
        )
    if problem.n_phases > 0 and problem.phase_index is not None:
        onehot = sp.csr_matrix(
            (np.ones(n_nodes), (np.arange(n_nodes), problem.phase_index)),
            shape=(n_nodes, problem.n_phases),
        )
        weight = problem.template_weight
        groups.append(
            (
                sp.csr_matrix(weight * sp.eye(n_nodes * n_u)),
                sp.csr_matrix(-weight * sp.kron(onehot, eye_u)),
            )
        )
    return groups


def _assemble(problem: InnerProblem) -> tuple[sp.csr_matrix, FloatArray, int]:
    """Stack all residual rows into ``(design, rhs, n_dynamics_rows)``."""
    n_pi, n_u = problem.n_parameter_columns, problem.n_inputs
    n_u_cols, n_t_cols = problem.n_nodes * n_u, problem.n_phases * n_u
    pi_dyn, u_dyn, dyn_rhs = _dynamics_blocks(problem)
    n_dyn = u_dyn.shape[0]
    blocks: list[list[sp.spmatrix | None]] = [[pi_dyn, u_dyn, None]]
    rhs: list[FloatArray] = [dyn_rhs]
    if problem.parameters_are_free:
        blocks.append([sp.diags(problem.parameter_prior_weight), None, None])
        rhs.append(problem.parameter_prior_weight * problem.parameter_prior_mean)
    for u_block, t_block in _input_prior_blocks(problem):
        blocks.append([None, u_block, t_block])
        rhs.append(np.zeros(u_block.shape[0]))
    for anchor in problem.anchors:
        blocks.append(
            [sp.csr_matrix(anchor.weight * anchor.coefficients[None, :]), None, None]
        )
        rhs.append(np.array([anchor.weight * anchor.value]))
    widths = (n_pi, n_u_cols, n_t_cols)
    rows = [_fill_row(row, widths) for row in blocks]
    design = sp.vstack(rows, format="csr")
    return design, np.concatenate(rhs), n_dyn


def _fill_row(
    row: list[sp.spmatrix | None], widths: tuple[int, int, int]
) -> sp.csr_matrix:
    height = next(block.shape[0] for block in row if block is not None)
    filled = [
        block if block is not None else sp.csr_matrix((height, width))
        for block, width in zip(row, widths, strict=True)
        if width > 0 or block is not None
    ]
    return sp.hstack(filled, format="csr")


def solve_inner(problem: InnerProblem) -> InnerSolution:
    """Solve the inner linear least squares exactly via sparse normal equations.

    Postcondition: the normal-equation residual ``A^T r`` vanishes to solver
    precision, which :func:`reduced_gauss_newton` relies on.
    """
    design, rhs, n_dyn = _assemble(problem)
    normal = (design.T @ design).tocsc() + _NORMAL_RIDGE * sp.eye(
        design.shape[1], format="csc"
    )
    lu = splu(normal)
    stacked = lu.solve(design.T @ rhs)
    residual = design @ stacked - rhs
    ensure(
        bool(np.linalg.norm(design.T @ residual) <= 1e-6 * (1.0 + np.linalg.norm(rhs))),
        "normal eq",
    )
    n_pi, n_u = problem.n_parameter_columns, problem.n_inputs
    inputs = stacked[n_pi : n_pi + problem.n_nodes * n_u].reshape(problem.n_nodes, n_u)
    template = None
    if problem.n_phases > 0:
        template = stacked[n_pi + problem.n_nodes * n_u :].reshape(
            problem.n_phases, n_u
        )
    parameters = (
        stacked[:n_pi]
        if problem.parameters_are_free
        else np.array(problem.fixed_parameters)
    )
    return InnerSolution(
        parameters,
        inputs,
        template,
        stacked,
        residual,
        rhs,
        design,
        lu,
        n_dyn,
        problem.parameters_are_free,
    )


def reduced_gauss_newton(
    solution: InnerSolution, jacobian_xi_dynamics: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return the reduced Gauss-Newton ``(H, g)`` over the outer variables.

    ``jacobian_xi_dynamics`` is the derivative of the *dynamics residual rows*
    with respect to the outer (kinematic/geometric) variables, holding the inner
    variables fixed.  Eliminating the inner variables exactly gives
    ``H = J^T (I - A (A^T A)^{-1} A^T) J`` and ``g = J^T r`` (Kaufman form).
    """
    n_rows, n_xi = jacobian_xi_dynamics.shape
    require(n_rows == solution.n_dynamics_rows, "Jacobian rows = dynamics rows", n_rows)
    design_dyn = solution.design[: solution.n_dynamics_rows]
    cross = np.asarray(design_dyn.T @ jacobian_xi_dynamics)
    correction = solution.normal_lu.solve(cross)
    hessian = jacobian_xi_dynamics.T @ jacobian_xi_dynamics - cross.T @ correction
    gradient = jacobian_xi_dynamics.T @ solution.residual[: solution.n_dynamics_rows]
    ensure(hessian.shape == (n_xi, n_xi), "reduced Hessian square")
    return 0.5 * (hessian + hessian.T), gradient
