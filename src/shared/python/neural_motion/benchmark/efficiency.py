"""Data efficiency and learning curve analysis for NM-10 (R12, #11155).

Governing Issue: #10625, #11155
Parent Epic: #10603
"""

from __future__ import annotations

import math
from typing import Sequence

from scipy import stats

from .types import DataEfficiencyCurve, MultiSeedEfficiencySummary


def trapezoidal_auc(x: Sequence[int | float], y: Sequence[float]) -> float:
    """Compute the trapezoidal area under the curve over simulation budgets.

    Design by Contract:
        x and y must have identical lengths >= 1.
    """
    if len(x) != len(y):
        raise ValueError("x and y sequences must have identical lengths")
    if len(x) < 2:
        return 0.0

    return sum((x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2.0 for i in range(len(x) - 1))


def _interpolate_budget_for_target(
    budgets: Sequence[int],
    curve: Sequence[float],
    target: float,
) -> float | None:
    """Interpolate the simulation budget required to attain target acceptance."""
    if not curve or not budgets:
        return None
    if target <= curve[0]:
        return float(budgets[0])

    for i in range(len(curve) - 1):
        y0, y1 = curve[i], curve[i + 1]
        if (y0 <= target <= y1) or (y1 <= target <= y0):
            if abs(y1 - y0) < 1e-12:
                return float(budgets[i])
            fraction = (target - y0) / (y1 - y0)
            return float(budgets[i] + fraction * (budgets[i + 1] - budgets[i]))

    return None


def compute_budget_to_target_ratio(
    native_simulation_budgets: Sequence[int],
    active_acceptance_curve: Sequence[float],
    random_acceptance_curve: Sequence[float],
    target_acceptance: float,
) -> float | None:
    """Calculate the ratio of random to active budget needed to reach target acceptance.

    Returns None if target is unattained by either method, representing an
    undefined/inconclusive sample-budget savings estimand.
    """
    if not (0.0 <= target_acceptance <= 1.0):
        raise ValueError(
            f"target_acceptance must be in [0.0, 1.0], got {target_acceptance}"
        )
    if not active_acceptance_curve or not random_acceptance_curve:
        return None

    if target_acceptance > max(active_acceptance_curve) or target_acceptance > max(
        random_acceptance_curve
    ):
        return None

    budget_active = _interpolate_budget_for_target(
        native_simulation_budgets, active_acceptance_curve, target_acceptance
    )
    budget_random = _interpolate_budget_for_target(
        native_simulation_budgets, random_acceptance_curve, target_acceptance
    )

    if budget_active is None or budget_random is None or budget_active <= 0:
        return None

    return float(budget_random / budget_active)


def _validate_budgets_and_curves(
    native_simulation_budgets: Sequence[int],
    active_acquisition_acceptance: Sequence[float],
    random_acquisition_acceptance: Sequence[float],
) -> None:
    """Enforce Design-by-Contract preconditions on simulation budgets and curves."""
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

    for i in range(len(native_simulation_budgets) - 1):
        curr_b, next_b = native_simulation_budgets[i], native_simulation_budgets[i + 1]
        if next_b == curr_b:
            raise ValueError(
                f"Simulation budgets must be strictly increasing without duplicates, found duplicate {curr_b}"
            )
        if next_b < curr_b:
            raise ValueError(
                f"Simulation budgets must be strictly increasing, got unsorted budgets [{curr_b}, {next_b}]"
            )

    for a_val, r_val in zip(
        active_acquisition_acceptance, random_acquisition_acceptance, strict=True
    ):
        if (
            not (0.0 <= a_val <= 1.0)
            or not (0.0 <= r_val <= 1.0)
            or not math.isfinite(a_val)
            or not math.isfinite(r_val)
        ):
            raise ValueError(
                f"Acceptance values must be finite and in [0.0, 1.0], got ({a_val}, {r_val})"
            )


def evaluate_data_efficiency(
    native_simulation_budgets: Sequence[int],
    active_acquisition_acceptance: Sequence[float],
    random_acquisition_acceptance: Sequence[float],
    target_acceptance: float | None = None,
) -> DataEfficiencyCurve:
    """Evaluate sample efficiency trajectories comparing active vs random acquisition.

    Distinguishes descriptive acceptance gain (arithmetic mean and trapezoidal AUC)
    from horizontal sample-budget savings. Rejects bad budgets fail-closed, avoids
    false superiority on equal curves, and marks zero-reference cases inconclusive.
    """
    _validate_budgets_and_curves(
        native_simulation_budgets,
        active_acquisition_acceptance,
        random_acquisition_acceptance,
    )

    mean_active = sum(active_acquisition_acceptance) / len(
        active_acquisition_acceptance
    )
    mean_random = sum(random_acquisition_acceptance) / len(
        random_acquisition_acceptance
    )

    active_auc = trapezoidal_auc(
        native_simulation_budgets, active_acquisition_acceptance
    )
    random_auc = trapezoidal_auc(
        native_simulation_budgets, random_acquisition_acceptance
    )

    if mean_random <= 1e-9 or random_auc <= 1e-9:
        mean_ratio: float | None = None
        auc_ratio: float | None = None
        sample_multiplier: float | None = None
        is_inconclusive = True
        status_msg = "Zero-reference random curve; descriptive ratios are undefined."
    else:
        mean_ratio = float(mean_active / mean_random)
        auc_ratio = float(active_auc / random_auc)
        sample_multiplier = mean_ratio
        is_inconclusive = False
        status_msg = "Evaluated"

    non_inferior = all(
        a >= r - 1e-6
        for a, r in zip(
            active_acquisition_acceptance, random_acquisition_acceptance, strict=True
        )
    )
    strictly_superior = any(
        a > r + 1e-6
        for a, r in zip(
            active_acquisition_acceptance, random_acquisition_acceptance, strict=True
        )
    )
    confirmed = non_inferior and strictly_superior and not is_inconclusive

    budget_savings: float | None = None
    if target_acceptance is not None:
        budget_savings = compute_budget_to_target_ratio(
            native_simulation_budgets,
            active_acquisition_acceptance,
            random_acquisition_acceptance,
            target_acceptance,
        )

    return DataEfficiencyCurve(
        budget_points=tuple(native_simulation_budgets),
        active_acceptance_curve=tuple(active_acquisition_acceptance),
        random_acceptance_curve=tuple(random_acquisition_acceptance),
        mean_acceptance_ratio=mean_ratio,
        auc_acceptance_ratio=auc_ratio,
        budget_to_target_ratio=budget_savings,
        sample_efficiency_multiplier=sample_multiplier,
        active_superiority_confirmed=confirmed,
        is_inconclusive=is_inconclusive,
        status_message=status_msg,
    )


def evaluate_multi_seed_data_efficiency(
    native_simulation_budgets: Sequence[int],
    active_seed_runs: Sequence[Sequence[float]],
    random_seed_runs: Sequence[Sequence[float]],
    confidence_level: float = 0.95,
) -> MultiSeedEfficiencySummary:
    """Aggregate multi-seed curves and validate statistical superiority under uncertainty."""
    if len(active_seed_runs) < 2 or len(random_seed_runs) < 2:
        raise ValueError("Multi-seed evaluation requires at least 2 seed runs")
    if len(active_seed_runs) != len(random_seed_runs):
        raise ValueError("Active and random runs must have identical seed counts")

    for run in list(active_seed_runs) + list(random_seed_runs):
        _validate_budgets_and_curves(native_simulation_budgets, run, run)

    n_seeds = len(active_seed_runs)
    n_budgets = len(native_simulation_budgets)

    mean_active = []
    std_active = []
    mean_random = []
    std_random = []

    for j in range(n_budgets):
        act_vals = [run[j] for run in active_seed_runs]
        rnd_vals = [run[j] for run in random_seed_runs]

        m_act = sum(act_vals) / n_seeds
        s_act = math.sqrt(sum((v - m_act) ** 2 for v in act_vals) / (n_seeds - 1))
        mean_active.append(m_act)
        std_active.append(s_act)

        m_rnd = sum(rnd_vals) / n_seeds
        s_rnd = math.sqrt(sum((v - m_rnd) ** 2 for v in rnd_vals) / (n_seeds - 1))
        mean_random.append(m_rnd)
        std_random.append(s_rnd)

    auc_active = [
        trapezoidal_auc(native_simulation_budgets, run) for run in active_seed_runs
    ]
    auc_random = [
        trapezoidal_auc(native_simulation_budgets, run) for run in random_seed_runs
    ]

    mean_auc_act = sum(auc_active) / n_seeds
    mean_auc_rnd = sum(auc_random) / n_seeds

    if mean_auc_rnd <= 1e-9:
        mean_auc_ratio = None
    else:
        mean_auc_ratio = float(mean_auc_act / mean_auc_rnd)

    # Paired difference testing on learning curve AUC
    diffs = [a - r for a, r in zip(auc_active, auc_random, strict=True)]
    mean_diff = sum(diffs) / n_seeds
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n_seeds - 1)
    std_diff = math.sqrt(var_diff)

    if std_diff < 1e-12:
        p_val = 0.0 if mean_diff > 0 else 1.0
    else:
        t_stat = mean_diff / (std_diff / math.sqrt(n_seeds))
        p_val = float(stats.t.sf(t_stat, df=n_seeds - 1))

    alpha = 1.0 - confidence_level
    pointwise_superior = all(
        (ma - sa) >= (mr - sr)
        for ma, sa, mr, sr in zip(
            mean_active, std_active, mean_random, std_random, strict=True
        )
    )
    statistically_superior = bool(
        mean_diff > 0 and p_val < alpha and pointwise_superior
    )

    return MultiSeedEfficiencySummary(
        budget_points=tuple(native_simulation_budgets),
        mean_active_curve=tuple(mean_active),
        std_active_curve=tuple(std_active),
        mean_random_curve=tuple(mean_random),
        std_random_curve=tuple(std_random),
        mean_auc_ratio=mean_auc_ratio,
        active_statistically_superior=statistically_superior,
        p_value=p_val,
        confidence_level=confidence_level,
        is_inconclusive=not statistically_superior and mean_diff > 0,
        status_message="Statistically superior"
        if statistically_superior
        else "Inconclusive or non-superior",
    )
