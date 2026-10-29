"""Sensitivity-derived tolerance bands for cross-engine validation (issue #13794).

Replaces fixed heuristic tolerances (e.g. 20%) with rigorous bounds derived
from model mismatch and finite-time trajectory sensitivity:
    tol = sigma * (eps_model + eps_integrator) + floor

For pointwise evaluations (such as M(q) or inverse dynamics at rest on same-spec models),
tol collapses to the numerical floor (<= 1e-6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CrossEngineToleranceBand:
    """Sensitivity-derived tolerance band for cross-engine comparisons."""

    tolerance: float
    sigma: float
    epsilon_model: float
    epsilon_integrator: float
    floor: float
    is_pointwise: bool
    receipt_source: str
    documentation: str


def compute_cross_engine_tolerance_band(
    *,
    sigma: float | None = None,
    epsilon_model: float = 0.0,
    epsilon_integrator: float = 0.0,
    floor: float = 1e-6,
    is_pointwise: bool = False,
    receipt: Any | None = None,
) -> CrossEngineToleranceBand:
    """Compute sensitivity-derived tolerance band.

    Design by Contract:
    - Precondition: epsilon_model >= 0 and finite.
    - Precondition: epsilon_integrator >= 0 and finite.
    - Precondition: floor > 0 and finite.
    - Precondition: For dynamic comparisons (!is_pointwise), sigma must be
      provided directly (>0) or extracted from receipt.sigma_max.
    - Postcondition: Returns tolerance >= floor.
    """
    if not math.isfinite(epsilon_model) or epsilon_model < 0.0:
        raise ValueError(
            f"epsilon_model must be non-negative and finite, got {epsilon_model}"
        )
    if not math.isfinite(epsilon_integrator) or epsilon_integrator < 0.0:
        raise ValueError(
            f"epsilon_integrator must be non-negative and finite, got {epsilon_integrator}"
        )
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError(f"floor must be finite and positive, got {floor}")

    source = "direct"
    if receipt is not None:
        source = "receipt"
        if sigma is None:
            sigma = getattr(receipt, "sigma_max", None)

    if is_pointwise:
        effective_sigma = 1.0 if sigma is None else sigma
        if not math.isfinite(effective_sigma) or effective_sigma <= 0.0:
            raise ValueError(
                f"sigma must be finite and positive, got {effective_sigma}"
            )
        if epsilon_model == 0.0 and epsilon_integrator == 0.0:
            tol = floor
            doc = "pointwise same-spec numerical floor"
        else:
            tol = effective_sigma * (epsilon_model + epsilon_integrator) + floor
            doc = "pointwise model-mismatch band"
        return CrossEngineToleranceBand(
            tolerance=tol,
            sigma=effective_sigma,
            epsilon_model=epsilon_model,
            epsilon_integrator=epsilon_integrator,
            floor=floor,
            is_pointwise=True,
            receipt_source=source,
            documentation=doc,
        )

    if sigma is None:
        raise ValueError(
            "sigma must be provided directly or extracted from a SensitivityReceipt"
        )
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be finite and positive, got {sigma}")

    tol = sigma * (epsilon_model + epsilon_integrator) + floor
    doc = "ensemble/shadowing band derived from finite-time amplification"
    return CrossEngineToleranceBand(
        tolerance=tol,
        sigma=sigma,
        epsilon_model=epsilon_model,
        epsilon_integrator=epsilon_integrator,
        floor=floor,
        is_pointwise=False,
        receipt_source=source,
        documentation=doc,
    )
