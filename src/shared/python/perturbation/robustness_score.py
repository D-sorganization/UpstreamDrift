"""Robustness Score computation for perturbation analysis."""

from __future__ import annotations


def compute_robustness_score(cv_weighted: float) -> float:
    """Compute the Robustness Score (RS) from a weighted Coefficient of Variation.

    Parameters
    ----------
    cv_weighted : float
        Weighted coefficient of variation.

    Returns
    -------
    float
        Robustness Score between 0.0 and 1.0. Higher is more robust.

    Design by Contract
    ------------------
    Pre: cv_weighted >= 0.0
    Post: 0.0 <= result <= 1.0
    """
    if not (cv_weighted >= 0.0):
        raise ValueError(f"cv_weighted must be non-negative, got {cv_weighted}")
    return 1.0 / (1.0 + cv_weighted)
