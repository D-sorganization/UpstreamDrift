"""Cross-engine Jacobian and constraint diagnostics (Issue #760).

Provides standardized task-point Jacobian analysis, constraint rank
monitoring, nullspace computation, and cross-engine consistency
validation for all physics engines.

Design by Contract:
    Preconditions:
        - Jacobian matrices must be well-formed (m x n) with n > 0
        - Engine must be initialized and in a valid state
    Postconditions:
        - Diagnostics are deterministic for a given state
        - Rank and nullspace dimensions are consistent
    Invariants:
        - rank(J) + nullspace_dim(J) == n (number of columns)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)

# Singular value threshold for rank determination
RANK_TOLERANCE = 1e-8

# Task-point names used across golf biomechanics
GOLF_TASK_POINTS = [
    "clubhead",
    "grip",
    "left_hand",
    "right_hand",
    "shaft_mid",
]


class TaskPointCategory(Enum):
    """Category of task points for Jacobian analysis."""

    END_EFFECTOR = auto()  # clubhead
    GRIP = auto()  # grip, hands
    SHAFT = auto()  # shaft points
    BODY = auto()  # arbitrary body


@dataclass
class JacobianDiagnostics:
    """Diagnostic report for a single Jacobian evaluation.

    Attributes:
        body_name: Body or task-point name
        shape: Jacobian matrix shape (m, n)
        rank: Numerical rank of the Jacobian
        nullspace_dim: Dimension of the null space (n - rank)
        condition_number: Condition number kappa = sigma_max / sigma_min
        singular_values: All singular values (descending)
        manipulability: Yoshikawa manipulability index (product of sigma)
        is_near_singular: True if condition_number > 1e6
    """

    body_name: str
    shape: tuple[int, int]
    rank: int
    nullspace_dim: int
    condition_number: float
    singular_values: np.ndarray
    manipulability: float
    is_near_singular: bool


@dataclass
class ConstraintDiagnostics:
    """Diagnostic report for constraint Jacobian analysis.

    Attributes:
        constraint_rank: Rank of the constraint Jacobian
        nullspace_dim: Dimension of the constraint null space
        nullspace_basis: Orthonormal basis for the null space (n x nullspace_dim)
        is_overconstrained: True if rank > expected DOF
        condition_number: Condition number of the constraint Jacobian
    """

    constraint_rank: int
    nullspace_dim: int
    nullspace_basis: np.ndarray
    is_overconstrained: bool
    condition_number: float


@dataclass
class CrossEngineJacobianReport:
    """Cross-engine Jacobian consistency report.

    Attributes:
        body_name: Task point being compared
        engines: Names of engines being compared
        shape_match: Whether Jacobian shapes match
        rank_match: Whether ranks match
        max_element_diff: Maximum element-wise difference
        frobenius_diff: Frobenius norm of difference
        condition_numbers: Condition numbers per engine
        passed: Whether the comparison passes tolerances
    """

    body_name: str
    engines: list[str]
    shape_match: bool
    rank_match: bool
    max_element_diff: float
    frobenius_diff: float
    condition_numbers: list[float]
    passed: bool


def compute_jacobian_diagnostics(
    J: np.ndarray,
    body_name: str = "",
    rank_tol: float = RANK_TOLERANCE,
) -> JacobianDiagnostics:
    """Compute comprehensive diagnostics for a Jacobian matrix.

    Args:
        J: Jacobian matrix (m x n)
        body_name: Name of body/task-point for reporting
        rank_tol: Tolerance for rank determination

    Returns:
        JacobianDiagnostics with rank, nullspace, conditioning info
    """
    if J.size == 0:
        return JacobianDiagnostics(
            body_name=body_name,
            shape=(0, 0),
            rank=0,
            nullspace_dim=0,
            condition_number=float("inf"),
            singular_values=np.array([]),
            manipulability=0.0,
            is_near_singular=True,
        )

    m, n = J.shape
    sigma = np.linalg.svd(J, compute_uv=False)

    # Numerical rank
    rank = int(np.sum(sigma > rank_tol))
    nullspace_dim = n - rank

    # Condition number
    if sigma[-1] > rank_tol:
        kappa = float(sigma[0] / sigma[-1])
    else:
        kappa = float("inf")

    # Manipulability (Yoshikawa)
    mu = float(np.prod(sigma[sigma > rank_tol])) if rank > 0 else 0.0

    is_near_singular = kappa > 1e6

    if is_near_singular:
        logger.warning(
            "%s: near-singular Jacobian (kappa=%.2e, rank=%d/%d)",
            body_name,
            kappa,
            rank,
            min(m, n),
        )

    return JacobianDiagnostics(
        body_name=body_name,
        shape=(m, n),
        rank=rank,
        nullspace_dim=nullspace_dim,
        condition_number=kappa,
        singular_values=sigma,
        manipulability=mu,
        is_near_singular=is_near_singular,
    )


def compute_constraint_diagnostics(
    J_constraint: np.ndarray,
    expected_dof: int | None = None,
) -> ConstraintDiagnostics:
    """Analyze a constraint Jacobian for rank and nullspace.

    The constraint Jacobian relates joint velocities to constraint
    violations: C_dot = J_c * qdot = 0. The nullspace of J_c gives
    the feasible motions.

    Args:
        J_constraint: Constraint Jacobian (n_constraints x n_dof)
        expected_dof: Expected number of unconstrained DOF

    Returns:
        ConstraintDiagnostics with rank, nullspace basis, and flags
    """
    if J_constraint.size == 0:
        return ConstraintDiagnostics(
            constraint_rank=0,
            nullspace_dim=0,
            nullspace_basis=np.array([]).reshape(0, 0),
            is_overconstrained=False,
            condition_number=float("inf"),
        )

    n_constraints, n_dof = J_constraint.shape

    # SVD for rank and nullspace
    U, sigma, Vt = np.linalg.svd(J_constraint, full_matrices=True)
    rank = int(np.sum(sigma > RANK_TOLERANCE))
    nullspace_dim = n_dof - rank

    # Nullspace basis: last (n_dof - rank) rows of Vt
    if nullspace_dim > 0:
        nullspace_basis = Vt[rank:].T  # (n_dof x nullspace_dim)
    else:
        nullspace_basis = np.zeros((n_dof, 0))

    # Overconstrained if rank > expected unconstrained DOF
    is_overconstrained = False
    if expected_dof is not None:
        is_overconstrained = rank > (n_dof - expected_dof)

    # Condition number of constraint Jacobian
    if len(sigma) > 0 and sigma[-1] > RANK_TOLERANCE:
        kappa = float(sigma[0] / sigma[-1])
    else:
        kappa = float("inf")

    return ConstraintDiagnostics(
        constraint_rank=rank,
        nullspace_dim=nullspace_dim,
        nullspace_basis=nullspace_basis,
        is_overconstrained=is_overconstrained,
        condition_number=kappa,
    )


def compute_nullspace_projection(J: np.ndarray) -> np.ndarray:
    """Compute the nullspace projection matrix P = I - J^+ J.

    Projects joint velocities onto the nullspace of the task Jacobian,
    enabling redundancy resolution.

    Args:
        J: Task Jacobian (m x n)

    Returns:
        Projection matrix P (n x n) such that J @ P = 0
    """
    n = J.shape[1]
    J_pinv = np.linalg.pinv(J)
    return np.eye(n) - J_pinv @ J


def validate_jacobians_cross_engine(
    jacobians: dict[str, np.ndarray],
    body_name: str,
    atol: float = 1e-3,
    rtol: float = 0.05,
) -> CrossEngineJacobianReport:
    """Validate Jacobian consistency across multiple engines.

    Compares Jacobian matrices from different engines at the same
    configuration to detect implementation discrepancies.

    Args:
        jacobians: Map of engine_name -> Jacobian matrix
        body_name: Task-point being compared
        atol: Absolute tolerance for element-wise comparison
        rtol: Relative tolerance for Frobenius norm comparison

    Returns:
        CrossEngineJacobianReport with comparison results
    """
    engines = list(jacobians.keys())
    matrices = list(jacobians.values())

    if len(matrices) < 2:
        return CrossEngineJacobianReport(
            body_name=body_name,
            engines=engines,
            shape_match=True,
            rank_match=True,
            max_element_diff=0.0,
            frobenius_diff=0.0,
            condition_numbers=[float(np.linalg.cond(m)) for m in matrices],
            passed=True,
        )

    # Shape check
    shapes = [m.shape for m in matrices]
    shape_match = all(s == shapes[0] for s in shapes)

    # Rank check
    ranks = [int(np.linalg.matrix_rank(m, tol=RANK_TOLERANCE)) for m in matrices]
    rank_match = all(r == ranks[0] for r in ranks)

    # Element-wise and Frobenius norm comparison
    max_diff = 0.0
    max_frob = 0.0
    if shape_match:
        for i in range(len(matrices)):
            for j in range(i + 1, len(matrices)):
                diff = np.abs(matrices[i] - matrices[j])
                max_diff = max(max_diff, float(np.max(diff)))
                frob = float(np.linalg.norm(matrices[i] - matrices[j], "fro"))
                max_frob = max(max_frob, frob)

    condition_numbers = [float(np.linalg.cond(m)) for m in matrices]

    # Pass if shapes match, ranks match, and differences are within tolerance
    ref_norm = max(float(np.linalg.norm(m, "fro")) for m in matrices)
    frob_ok = max_frob < atol + rtol * ref_norm if ref_norm > 0 else max_frob < atol
    passed = shape_match and rank_match and frob_ok

    return CrossEngineJacobianReport(
        body_name=body_name,
        engines=engines,
        shape_match=shape_match,
        rank_match=rank_match,
        max_element_diff=max_diff,
        frobenius_diff=max_frob,
        condition_numbers=condition_numbers,
        passed=passed,
    )


def diagnose_task_points(
    engine_compute_jacobian: Callable[[str], dict[str, np.ndarray] | None],
    task_points: list[str] | None = None,
) -> dict[str, JacobianDiagnostics]:
    """Run diagnostics on all golf task points for an engine.

    Args:
        engine_compute_jacobian: Engine's compute_jacobian(body_name) method
        task_points: List of body names to diagnose (default: GOLF_TASK_POINTS)

    Returns:
        Map of body_name -> JacobianDiagnostics
    """
    if task_points is None:
        task_points = GOLF_TASK_POINTS

    results: dict[str, JacobianDiagnostics] = {}
    for point in task_points:
        try:
            jac_dict = engine_compute_jacobian(point)
            if jac_dict is None:
                logger.info("%s: body not found in model, skipping", point)
                continue

            # Prefer spatial Jacobian
            if "spatial" in jac_dict:
                J = jac_dict["spatial"]
            elif "linear" in jac_dict:
                J = jac_dict["linear"]
            else:
                continue

            results[point] = compute_jacobian_diagnostics(J, body_name=point)
        except Exception as e:
            logger.warning("Failed to compute Jacobian for %s: %s", point, e)

    return results
