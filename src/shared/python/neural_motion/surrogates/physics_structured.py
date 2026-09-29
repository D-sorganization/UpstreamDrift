"""Physics-structured forward surrogate with analytical rigid prior and residual correction (NM-07 #10622)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .sparse_residual import SparseResidualFit

__all__ = ["PhysicsStructuredSurrogate"]


class PhysicsStructuredSurrogate:
    """Physics-structured forward surrogate for smooth rigid multibody models.

    Combines an analytical kinematic/rigid prior with a fitted sparse residual
    correction (STLSQ). When no fitted residual is provided, residual evaluation
    and forward rollout fail closed (#11007 / #10960).
    """

    def __init__(
        self,
        n_dof: int,
        n_coeffs: int = 7,
        *,
        residual: SparseResidualFit | None = None,
    ) -> None:
        if n_dof <= 0:
            raise ValueError("n_dof must be positive")
        if n_coeffs <= 0:
            raise ValueError("n_coeffs must be positive")
        self.n_dof = n_dof
        self.n_coeffs = n_coeffs
        self.total_coeffs = n_dof * n_coeffs
        self.residual = residual

    def analytical_prior(self, coeffs: np.ndarray, timegrid: np.ndarray) -> np.ndarray:
        """Evaluate analytical polynomial/harmonic kinematic prior."""
        c = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if c.size != self.total_coeffs:
            raise ValueError(f"expected {self.total_coeffs} coefficients, got {c.size}")
        t = np.asarray(timegrid, dtype=np.float64).reshape(-1)
        T = len(t)
        t_norm = (t - t[0]) / max(t[-1] - t[0], 1e-9)

        # Basis matrix (T, n_coeffs): t^0, t^1, t^2, ..., t^(n_coeffs-1)
        basis = np.vstack([t_norm**k for k in range(self.n_coeffs)]).T  # (T, n_coeffs)

        c_per_dof = c.reshape(self.n_dof, self.n_coeffs)
        # Prior trajectory: (T, n_dof)
        return basis @ c_per_dof.T

    def residual_correction(self, prior: np.ndarray) -> np.ndarray:
        """Evaluate fitted sparse residual correction on prior trajectory."""
        if self.residual is None:
            raise NotImplementedError(  # tracked: #11024 (fail closed, #10960)
                "PhysicsStructuredSurrogate requires a fitted residual (SparseResidualFit); "
                "the hand-chosen constant residual was removed in #11007 / #11024. "
                "Pass a fitted SparseResidualFit instance to __init__(..., residual=fit)."
            )
        p = np.asarray(prior, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != self.n_dof:
            raise ValueError(f"expected prior shape (T, {self.n_dof}), got {p.shape}")
        pred = self.residual.predict(p)
        if pred.shape != p.shape:
            raise ValueError(
                f"residual prediction shape {pred.shape} does not match prior shape {p.shape}"
            )
        return pred

    def forward_trajectory(
        self, coeffs: np.ndarray, timegrid: np.ndarray
    ) -> np.ndarray:
        """Full forward prediction: analytical prior + residual."""
        prior = self.analytical_prior(coeffs, timegrid)
        residual = self.residual_correction(prior)
        return prior + residual
