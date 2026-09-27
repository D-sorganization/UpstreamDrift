"""Sparse residual discovery using sequentially thresholded least squares (STLSQ).

This module provides the residual-discovery core only (Brunton, Proctor & Kutz
2016, PNAS 113:3932). An NM-07 evaluator stays unwired until a measured residual
dataset exists (#10960 P0-3, #11007 Package 2, #11024).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.contracts import require

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "CandidateLibrary",
    "SparseResidualFit",
    "fit_sparse_residual",
]


def _format_monomial(comb: tuple[str, ...]) -> str:
    """Format a combination of state variable names as a monomial string."""
    counts: dict[str, int] = {}
    for var in comb:
        counts[var] = counts.get(var, 0) + 1
    parts: list[str] = []
    for var, count in counts.items():
        if count == 1:
            parts.append(var)
        else:
            parts.append(f"{var}^{count}")
    return " ".join(parts)


@dataclass(frozen=True)
class CandidateLibrary:
    """Candidate feature library for sparse polynomial and trigonometric regression.

    Generates monomials up to a given degree in a deterministic order:
    bias first (if include_bias is True), then degree 1 in state order,
    then degree 2 by itertools.combinations_with_replacement, and so on.
    With include_trig=True, appends sin(x_i) and cos(x_i) for each state variable.
    """

    degree: int = 2
    include_trig: bool = False
    include_bias: bool = True

    def __post_init__(self) -> None:
        require(self.degree >= 1, f"degree must be at least 1, got {self.degree}")

    def feature_names(self, state_names: Sequence[str]) -> tuple[str, ...]:
        """Return deterministic feature name strings for the given state names.

        Args:
            state_names: Sequence of variable names corresponding to state columns.

        Returns:
            Tuple of feature name strings.
        """
        require(len(state_names) > 0, "state_names must not be empty")
        names: list[str] = []
        if self.include_bias:
            names.append("1")
        for d in range(1, self.degree + 1):
            for comb in itertools.combinations_with_replacement(state_names, d):
                names.append(_format_monomial(comb))
        if self.include_trig:
            for s in state_names:
                names.append(f"sin({s})")
                names.append(f"cos({s})")
        return tuple(names)

    def transform(self, state: ArrayLike) -> np.ndarray:
        """Map state trajectory of shape (T, n) to candidate feature matrix (T, n_features).

        Args:
            state: Array-like state trajectory of shape (T, n).

        Returns:
            Feature matrix of shape (T, n_features).
        """
        X = np.asarray(state, dtype=np.float64)
        require(X.ndim == 2, f"state must be 2-D, got shape {X.shape}")
        require(bool(np.all(np.isfinite(X))), "state must contain only finite values")

        T, n = X.shape
        cols: list[np.ndarray] = []

        if self.include_bias:
            cols.append(np.ones((T, 1), dtype=np.float64))

        for d in range(1, self.degree + 1):
            for comb in itertools.combinations_with_replacement(range(n), d):
                col = np.prod(X[:, comb], axis=1, keepdims=True)
                cols.append(col)

        if self.include_trig:
            for i in range(n):
                cols.append(np.sin(X[:, [i]]))
                cols.append(np.cos(X[:, [i]]))

        if cols:
            return np.hstack(cols)
        return np.empty((T, 0), dtype=np.float64)


@dataclass(frozen=True)
class SparseResidualFit:
    """Fitted sparse residual regression model resulting from STLSQ.

    Attributes:
        coefficients: Coefficient matrix of shape (n_features, n_targets), read-only.
        active: Boolean support mask of shape (n_features, n_targets), read-only.
        feature_names: Tuple of feature name strings.
        library: CandidateLibrary instance used for the fit.
        threshold: Sparsity cutoff threshold applied during STLSQ.
        iterations: Number of STLSQ iterations executed.
        converged: Whether the active support converged before max_iter.
        training_rmse: Root mean squared error over the training data.
    """

    coefficients: np.ndarray
    active: np.ndarray
    feature_names: tuple[str, ...]
    library: CandidateLibrary
    threshold: float
    iterations: int
    converged: bool
    training_rmse: float

    def __post_init__(self) -> None:
        self.coefficients.setflags(write=False)
        self.active.setflags(write=False)

    def predict(self, state: ArrayLike) -> np.ndarray:
        """Predict residual correction for the given state trajectory.

        Args:
            state: Array-like state trajectory of shape (T, n).

        Returns:
            Residual prediction array of shape (T, n_targets).
        """
        X = np.asarray(state, dtype=np.float64)
        require(X.ndim == 2, f"state must be 2-D, got shape {X.shape}")
        require(bool(np.all(np.isfinite(X))), "state must contain only finite values")
        phi = self.library.transform(X)
        require(
            phi.shape[1] == self.coefficients.shape[0],
            f"feature count mismatch: library produced {phi.shape[1]} features, "
            f"expected {self.coefficients.shape[0]}",
        )
        return phi @ self.coefficients


def _solve_lstsq(A: np.ndarray, b: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Solve linear least squares with optional ridge regularisation."""
    if ridge > 0.0:
        n_features = A.shape[1]
        A_aug = np.vstack([A, np.sqrt(ridge) * np.eye(n_features)])
        if b.ndim == 1:
            b_aug = np.concatenate([b, np.zeros(n_features)])
        else:
            b_aug = np.vstack([b, np.zeros((n_features, b.shape[1]))])
        sol, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
        return sol
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return sol


def _stlsq(
    Theta: np.ndarray,
    Y: np.ndarray,
    *,
    threshold: float,
    max_iter: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Sequentially thresholded least squares over a fixed feature matrix.

    Returns (coefficients, active support, iterations, converged). Postcondition:
    coefficients are zero exactly where the support is False.
    """
    C = _solve_lstsq(Theta, Y, ridge=ridge).reshape(Theta.shape[1], -1)
    active = np.abs(C) >= threshold
    C[~active] = 0.0
    iterations = 0
    for iterations in range(1, max_iter + 1):
        for j in range(Y.shape[1]):
            col_active = active[:, j]
            C[:, j] = 0.0
            if np.any(col_active):
                C[col_active, j] = _solve_lstsq(
                    Theta[:, col_active], Y[:, j], ridge=ridge
                )
        new_active = np.abs(C) >= threshold
        C[~new_active] = 0.0
        if np.array_equal(new_active, active):
            return C, active, iterations, True
        active = new_active
    return C, active, iterations, False


def fit_sparse_residual(
    state: ArrayLike,
    target: ArrayLike,
    *,
    library: CandidateLibrary,
    threshold: float,
    max_iter: int = 10,
    ridge: float = 0.0,
    state_names: Sequence[str] | None = None,
) -> SparseResidualFit:
    """Fit a sparse residual by sequentially thresholded least squares (STLSQ).

    Brunton, Proctor & Kutz 2016, PNAS 113(15):3932-3937.

    Residual-discovery core only. An NM-07 evaluator stays unwired until a
    measured residual dataset exists (#10960 P0-3, #11007 Package 2).

    Args:
        state: State trajectory of shape (T, n).
        target: Target residual trajectory of shape (T, m).
        library: Candidate feature library.
        threshold: Sparsity cutoff threshold (>= 0).
        max_iter: Maximum STLSQ iterations (>= 1).
        ridge: Ridge regression parameter (>= 0).
        state_names: Optional state variable names. Defaults to ('x0', 'x1', ...).

    Returns:
        SparseResidualFit containing fitted coefficients, active support, and diagnostics.
    """
    X = np.asarray(state, dtype=np.float64)
    Y = np.asarray(target, dtype=np.float64)

    require(X.ndim == 2, f"state must be 2-D, got shape {X.shape}")
    require(Y.ndim == 2, f"target must be 2-D, got shape {Y.shape}")
    require(bool(np.all(np.isfinite(X))), "state must contain only finite values")
    require(bool(np.all(np.isfinite(Y))), "target must contain only finite values")
    require(
        X.shape[0] == Y.shape[0],
        f"row count mismatch: state has {X.shape[0]}, target has {Y.shape[0]}",
    )
    require(threshold >= 0.0, f"threshold must be non-negative, got {threshold}")
    require(ridge >= 0.0, f"ridge must be non-negative, got {ridge}")
    require(max_iter >= 1, f"max_iter must be >= 1, got {max_iter}")

    T, n_states = X.shape
    _, n_targets = Y.shape

    if state_names is None:
        state_names_seq = tuple(f"x{i}" for i in range(n_states))
    else:
        require(
            len(state_names) == n_states,
            f"state_names length ({len(state_names)}) must match state dimension ({n_states})",
        )
        state_names_seq = tuple(state_names)

    feat_names = library.feature_names(state_names_seq)
    Theta = library.transform(X)
    n_features = Theta.shape[1]

    require(
        n_features <= T,
        f"underdetermined fit refused: rows ({T}) < n_features ({n_features})",
    )

    C, active, iterations, converged = _stlsq(
        Theta, Y, threshold=threshold, max_iter=max_iter, ridge=ridge
    )

    pred = Theta @ C
    training_rmse = float(np.sqrt(np.mean((Y - pred) ** 2)))

    return SparseResidualFit(
        coefficients=C,
        active=active,
        feature_names=feat_names,
        library=library,
        threshold=threshold,
        iterations=iterations,
        converged=converged,
        training_rmse=training_rmse,
    )
