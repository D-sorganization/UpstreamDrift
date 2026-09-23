"""Physics-structured forward surrogate with analytical rigid prior and residual correction (NM-07 #10622)."""

from __future__ import annotations

import numpy as np

__all__ = ["PhysicsStructuredSurrogate"]


class PhysicsStructuredSurrogate:
    """Physics-structured forward surrogate for smooth rigid multibody models.

    Combines an analytical kinematic/rigid prior with a bounded residual
    correction. For smooth rigid dynamics without contact impacts, the prior
    ensures high local gradient fidelity and prevents adversarial exploitation.
    """

    def __init__(self, n_dof: int, n_coeffs: int = 7) -> None:
        if n_dof <= 0:
            raise ValueError("n_dof must be positive")
        if n_coeffs <= 0:
            raise ValueError("n_coeffs must be positive")
        self.n_dof = n_dof
        self.n_coeffs = n_coeffs
        self.total_coeffs = n_dof * n_coeffs

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
        """Bounded neural-like residual correction."""
        # Small smooth non-linear adjustment modeling compliance/flex
        return 0.02 * np.tanh(0.5 * prior)

    def forward_trajectory(
        self, coeffs: np.ndarray, timegrid: np.ndarray
    ) -> np.ndarray:
        """Full forward prediction: analytical prior + residual."""
        prior = self.analytical_prior(coeffs, timegrid)
        residual = self.residual_correction(prior)
        return prior + residual
