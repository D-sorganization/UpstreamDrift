"""Muscle Synergy Analysis using Non-negative Matrix Factorization (NMF).

This module provides tools to extract and analyze muscle synergies from
electromyography (EMG) or simulated muscle activation data.

Synergies represent coordinated groups of muscles activated together
by the central nervous system to simplify control.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

# sklearn is an optional dependency for muscle synergy analysis
try:
    from sklearn.decomposition import NMF

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    NMF = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


@dataclass
class SynergyResult:
    """Result of muscle synergy analysis."""

    weights: np.ndarray  # Muscle weights (W matrix), shape (n_muscles, n_synergies)
    activations: (
        np.ndarray
    )  # Temporal activation profiles (H matrix), shape (n_synergies, n_samples)
    reconstructed: np.ndarray  # Reconstructed data (W @ H)
    vaf: float  # Variance Accounted For (0-1)
    n_synergies: int
    muscle_names: list[str] | None = None


class MuscleSynergyAnalyzer:
    """Analyzes muscle synergies from activation data."""

    def __init__(
        self, activation_data: np.ndarray, muscle_names: list[str] | None = None
    ) -> None:
        """Initialize with activation data.

        Args:
            activation_data: Array of shape (n_samples, n_muscles).
                             Must be non-negative.
            muscle_names: Optional list of muscle names.
        """
        self.data = np.asarray(activation_data)
        if np.any(self.data < 0):
            logger.warning(
                "Activation data contains negative values. NMF requires non-negative data. Clipping to 0."
            )
            self.data = np.maximum(self.data, 0)

        self.n_samples, self.n_muscles = self.data.shape
        self.muscle_names = muscle_names or [
            f"Muscle {i}" for i in range(self.n_muscles)
        ]

    def extract_synergies(
        self, n_synergies: int, max_iter: int = 1000
    ) -> SynergyResult:
        """Extract a specific number of synergies using NMF.

        Factorizes V ≈ W @ H
        where V is (n_muscles, n_samples)
        W is (n_muscles, n_synergies) - Muscle Weights
        H is (n_synergies, n_samples) - Activation Profiles

        Note: sklearn NMF expects (n_samples, n_features).
        So we pass data as is (n_samples, n_muscles).
        sklearn decomposes X ≈ W_sklearn @ H_sklearn
        where W_sklearn is (n_samples, n_components) -> This corresponds to Transpose of H
        and H_sklearn is (n_components, n_features) -> This corresponds to Transpose of W

        Wait, standard synergy notation:
        V (muscles x time) = W (muscles x synergies) * H (synergies x time)

        sklearn: X (samples x features) = W (samples x components) * H (components x features)
        Here samples=time, features=muscles.
        So X = time x muscles.
        W_sklearn = time x synergies (This is H.T)
        H_sklearn = synergies x muscles (This is W.T)

        So:
        H_standard = W_sklearn.T
        W_standard = H_sklearn.T

        Args:
            n_synergies: Number of synergies to extract.
            max_iter: Max iterations for NMF solver.

        Returns:
            SynergyResult object.
        """
        if n_synergies < 1 or n_synergies > self.n_muscles:
            raise ValueError(
                f"Invalid number of synergies: {n_synergies}. Must be between 1 and {self.n_muscles}"
            )

        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "sklearn is required for muscle synergy analysis. "
                "Install with: pip install scikit-learn"
            )

        model = NMF(
            n_components=n_synergies, init="nndsvd", max_iter=max_iter, random_state=42
        )

        # X is (n_samples, n_muscles)
        W_sklearn = model.fit_transform(
            self.data
        )  # (n_samples, n_synergies) -> Temporal profiles (transposed)
        H_sklearn = (
            model.components_
        )  # (n_synergies, n_muscles) -> Muscle weights (transposed)

        # Convert to standard notation
        # W (muscle weights) = H_sklearn.T  -> (n_muscles, n_synergies)
        # H (temporal profiles) = W_sklearn.T -> (n_synergies, n_samples)

        W_standard = H_sklearn.T
        H_standard = W_sklearn.T

        # Reconstruction
        # V_recon = W @ H = H_sklearn.T @ W_sklearn.T = (W_sklearn @ H_sklearn).T
        X_recon = np.dot(W_sklearn, H_sklearn)  # (n_samples, n_muscles)

        # Calculate VAF (Variance Accounted For)
        # VAF = 1 - sum((V - V_recon)^2) / sum(V^2)
        # Here V is self.data

        sst = np.sum(self.data**2)
        sse = np.sum((self.data - X_recon) ** 2)
        vaf = 1.0 - (sse / sst)

        return SynergyResult(
            weights=W_standard,
            activations=H_standard,
            reconstructed=X_recon,
            vaf=float(vaf),
            n_synergies=n_synergies,
            muscle_names=self.muscle_names,
        )

    def find_optimal_synergies(
        self, max_synergies: int = 10, vaf_threshold: float = 0.90
    ) -> SynergyResult:
        """Find the minimum number of synergies to satisfy VAF threshold.

        Args:
            max_synergies: Maximum number to try.
            vaf_threshold: VAF threshold (e.g., 0.90 for 90%).

        Returns:
            SynergyResult for the optimal number of synergies.
        """
        limit = min(max_synergies, self.n_muscles)

        if limit < 1:
            raise ValueError("Cannot extract synergies: limit must be >= 1")

        best_result: SynergyResult | None = None

        for n in range(1, limit + 1):
            result = self.extract_synergies(n)
            best_result = result

            if result.vaf >= vaf_threshold:
                logger.info(
                    f"VAF threshold {vaf_threshold} met with {n} synergies (VAF={result.vaf:.4f})"
                )
                return result

        # best_result is guaranteed to be set since limit >= 1
        assert best_result is not None, "Loop should have set best_result"
        logger.warning(
            f"VAF threshold not met. Best VAF {best_result.vaf:.4f} with {limit} synergies."
        )
        return best_result
