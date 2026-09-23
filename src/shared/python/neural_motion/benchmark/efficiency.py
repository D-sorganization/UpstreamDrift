"""Data efficiency and learning curve analysis for NM-10.

Governing Issue: #10625
Parent Epic: #10603
"""

from __future__ import annotations

import math
from typing import Sequence

from .types import DataEfficiencyCurve


def evaluate_data_efficiency(
    native_simulation_budgets: Sequence[int],
    active_acquisition_acceptance: Sequence[float],
    random_acquisition_acceptance: Sequence[float],
) -> DataEfficiencyCurve:
    """Evaluate sample efficiency trajectories comparing active vs random acquisition.

    Design by Contract:
        All inputs must be non-empty, matching length, monotonically non-decreasing budgets,
        and acceptance values in [0.0, 1.0].
    """
    if not (
        len(native_simulation_budgets)
        == len(active_acquisition_acceptance)
        == len(random_acquisition_acceptance)
    ):
        raise ValueError("All input sequences must have identical lengths")
    if not native_simulation_budgets:
        raise ValueError("Input sequences cannot be empty")

    for b in native_simulation_budgets:
        if b <= 0:
            raise ValueError(f"Simulation budgets must be strictly positive, got {b}")

    for a_val, r_val in zip(
        active_acquisition_acceptance, random_acquisition_acceptance, strict=True
    ):
        if not (0.0 <= a_val <= 1.0) or not (0.0 <= r_val <= 1.0):
            raise ValueError(
                f"Acceptance values must be in [0.0, 1.0], got ({a_val}, {r_val})"
            )

    # Compute area under curve / efficiency ratio
    mean_active = sum(active_acquisition_acceptance) / len(
        active_acquisition_acceptance
    )
    mean_random = sum(random_acquisition_acceptance) / len(
        random_acquisition_acceptance
    )

    multiplier = (mean_active / mean_random) if mean_random > 1e-9 else 1.0
    confirmed = all(
        a >= r - 1e-6
        for a, r in zip(
            active_acquisition_acceptance, random_acquisition_acceptance, strict=True
        )
    )

    return DataEfficiencyCurve(
        budget_points=tuple(native_simulation_budgets),
        active_acceptance_curve=tuple(active_acquisition_acceptance),
        random_acceptance_curve=tuple(random_acquisition_acceptance),
        sample_efficiency_multiplier=float(multiplier),
        active_superiority_confirmed=confirmed,
    )
