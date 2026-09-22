"""Local grip-Jacobian null-space proposals with closure reprojection (CO-05 #10609).

Null-space columns are local proposal directions only — not global feasibility
proofs. Every offset is reprojected onto hand-club closure before retention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition

__all__ = [
    "NullSpaceAnalysis",
    "analyze_grip_jacobian_nullspace",
    "propose_nullspace_offsets",
    "reproject_onto_closure",
    "synthetic_grip_jacobian",
]


@dataclass(frozen=True)
class NullSpaceAnalysis:
    """Rank, singular values, and local null-space basis of a grip Jacobian."""

    rank: int
    singular_values: NDArray[np.float64]
    basis: NDArray[np.float64]
    is_singular: bool
    singular_tol: float

    def __post_init__(self) -> None:
        sv = np.asarray(self.singular_values, dtype=np.float64)
        basis = np.asarray(self.basis, dtype=np.float64)
        if sv.ndim != 1 or not np.all(np.isfinite(sv)):
            raise ValueError("singular_values must be finite 1-D")
        if basis.ndim != 2:
            raise ValueError("basis must be 2-D (nq, n_null)")
        if not np.all(np.isfinite(basis)):
            raise ValueError("basis must be finite")
        if self.rank < 0:
            raise ValueError("rank must be >= 0")
        if not np.isfinite(self.singular_tol) or self.singular_tol <= 0.0:
            raise ValueError("singular_tol must be finite and > 0")
        object.__setattr__(self, "singular_values", sv.copy())
        object.__setattr__(self, "basis", basis.copy())


@precondition(
    lambda jacobian, singular_tol=1.0e-8: isinstance(jacobian, np.ndarray),
    "jacobian required",
)
@postcondition(
    lambda result: isinstance(result, NullSpaceAnalysis),
    "must return NullSpaceAnalysis",
)
def analyze_grip_jacobian_nullspace(
    jacobian: NDArray[np.floating],
    *,
    singular_tol: float = 1.0e-8,
) -> NullSpaceAnalysis:
    """Compute a local null-space basis via SVD; flag singular / rank-deficient."""
    j = np.asarray(jacobian, dtype=np.float64)
    if j.ndim != 2:
        raise ValueError("jacobian must be 2-D (n_constraints, nq)")
    if j.size == 0:
        raise ValueError("jacobian must be non-empty")
    if not np.all(np.isfinite(j)):
        raise ValueError("jacobian must be finite")
    if not np.isfinite(singular_tol) or singular_tol <= 0.0:
        raise ValueError("singular_tol must be finite and > 0")

    # Economy SVD: J = U S Vh; null space from right singular vectors with
    # singular values below tolerance.
    _u, singular_values, vh = np.linalg.svd(j, full_matrices=True)
    rank = int(np.sum(singular_values > singular_tol))
    n_constraints, nq = j.shape
    expected_full_rank = min(n_constraints, nq)
    # Rank-deficient relative to constraint count, or near-zero leading modes.
    is_singular = rank < expected_full_rank or (
        singular_values.size > 0 and float(singular_values[0]) <= singular_tol
    )
    null_start = rank
    basis = vh[null_start:].T if null_start < nq else np.zeros((nq, 0))
    return NullSpaceAnalysis(
        rank=rank,
        singular_values=singular_values,
        basis=basis,
        is_singular=is_singular,
        singular_tol=float(singular_tol),
    )


@precondition(
    lambda analysis, amplitudes=(0.02,): isinstance(analysis, NullSpaceAnalysis),
    "analysis required",
)
@postcondition(
    lambda result: isinstance(result, tuple),
    "must return offset tuple",
)
def propose_nullspace_offsets(
    analysis: NullSpaceAnalysis,
    *,
    amplitudes: Sequence[float] = (0.02,),
) -> tuple[NDArray[np.float64], ...]:
    """Scale leading null-space columns by bounded amplitudes.

    Raises
    ------
    ValueError
        When the analysis is singular or amplitudes are nonfinite / nonpositive.
    """
    if analysis.is_singular:
        raise ValueError(
            "singular null-space / rank-deficient Jacobian; "
            "local proposals refused (not global feasibility proof)"
        )
    if analysis.basis.size == 0 or analysis.basis.shape[1] < 1:
        raise ValueError("empty null-space basis; no unobserved freedom to explore")
    amps = tuple(float(a) for a in amplitudes)
    if not amps:
        raise ValueError("amplitudes must be non-empty")
    for amp in amps:
        if not np.isfinite(amp) or amp <= 0.0:
            raise ValueError("amplitudes must be finite and > 0")

    nq, n_null = analysis.basis.shape
    offsets: list[NDArray[np.float64]] = []
    # First proposal is the zero offset (warm start itself).
    offsets.append(np.zeros(nq, dtype=np.float64))
    for index, amp in enumerate(amps):
        col = analysis.basis[:, index % n_null]
        sign = 1.0 if index % 2 == 0 else -1.0
        offsets.append(sign * amp * col)
    return tuple(offsets)


@precondition(
    lambda q, closure_residual_m, tol_m: isinstance(q, np.ndarray),
    "q required",
)
@postcondition(
    lambda result: isinstance(result, np.ndarray) and result.ndim == 1,
    "must return 1-D ndarray",
)
def reproject_onto_closure(
    q: NDArray[np.floating],
    *,
    closure_residual_m: float,
    tol_m: float,
) -> NDArray[np.float64]:
    """Reproject a proposal onto grip closure within ``tol_m``.

    Software-contract reprojection: when residual already meets the tolerance,
    return ``q`` unchanged. Large residuals are scaled toward the prior origin
    (no invented native closure). Nonfinite inputs fail closed.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.ndim != 1 or q_arr.size < 1:
        raise ValueError("q must be a non-empty 1-D array")
    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q must be finite")
    if not np.isfinite(closure_residual_m) or closure_residual_m < 0.0:
        raise ValueError("closure_residual_m must be finite and >= 0")
    if not np.isfinite(tol_m) or tol_m <= 0.0:
        raise ValueError("tol_m must be finite and > 0")
    if closure_residual_m <= tol_m:
        return q_arr.copy()
    # Soft pull toward prior without claiming physical IK success.
    scale = float(tol_m / closure_residual_m)
    return q_arr * scale


def synthetic_grip_jacobian(
    nq: int,
    *,
    n_constraints: int = 3,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Deterministic full-rank synthetic grip Jacobian for software contracts.

    Not a native plant Jacobian and not G1 acceptance evidence.
    """
    if nq < 1 or n_constraints < 1:
        raise ValueError("nq and n_constraints must be >= 1")
    if n_constraints > nq:
        raise ValueError("n_constraints cannot exceed nq for a full-rank fixture")
    rng = np.random.default_rng(seed)
    j = rng.normal(size=(n_constraints, nq))
    # Ensure leading square block is well-conditioned.
    j[:, :n_constraints] += np.eye(n_constraints)
    return j
