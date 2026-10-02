"""Statistical validation and contract tests for neural data-efficiency curves (R12, #11155).

Tests cover:
- Rejection of unsorted, non-positive, and duplicate simulation budgets.
- Disallowing active superiority confirmation on equal curves ([0.5, 0.5] vs [0.5, 0.5]).
- Undefined / inconclusive status on zero-reference or unattained-target curves (no 1.0 default).
- Distinct estimands: descriptive mean ratio, budget-weighted trapezoidal AUC ratio,
  and horizontal budget-to-target efficiency (sample savings).
- Multi-seed statistical uncertainty aggregation and superiority decision rules.
"""

from __future__ import annotations

import math
import pytest

from src.shared.python.neural_motion.benchmark.efficiency import (
    compute_budget_to_target_ratio,
    evaluate_data_efficiency,
    evaluate_multi_seed_data_efficiency,
    trapezoidal_auc,
)
from src.shared.python.neural_motion.benchmark.types import (
    DataEfficiencyCurve,
    MultiSeedEfficiencySummary,
)

pytestmark = pytest.mark.unit


def test_reject_unsorted_budgets() -> None:
    """Unsorted budgets like [100, 1] must be rejected fail-closed with ValueError."""
    with pytest.raises(
        ValueError, match=r"strictly increasing|monotonically increasing|unsorted"
    ):
        evaluate_data_efficiency(
            native_simulation_budgets=[100, 1],
            active_acquisition_acceptance=[0.5, 0.5],
            random_acquisition_acceptance=[0.5, 0.5],
        )


def test_reject_duplicate_budgets() -> None:
    """Duplicate budgets like [100, 100, 200] must be rejected fail-closed with ValueError."""
    with pytest.raises(ValueError, match=r"strictly increasing|duplicate"):
        evaluate_data_efficiency(
            native_simulation_budgets=[100, 100, 200],
            active_acquisition_acceptance=[0.3, 0.5, 0.7],
            random_acquisition_acceptance=[0.2, 0.4, 0.5],
        )


def test_reject_non_positive_budgets() -> None:
    """Budgets with 0 or negative numbers must be rejected."""
    with pytest.raises(ValueError, match=r"strictly positive"):
        evaluate_data_efficiency(
            native_simulation_budgets=[0, 100],
            active_acquisition_acceptance=[0.3, 0.5],
            random_acquisition_acceptance=[0.2, 0.4],
        )

    with pytest.raises(ValueError, match=r"strictly positive"):
        evaluate_data_efficiency(
            native_simulation_budgets=[-50, 100],
            active_acquisition_acceptance=[0.3, 0.5],
            random_acquisition_acceptance=[0.2, 0.4],
        )


def test_reject_mismatched_lengths_and_empty() -> None:
    """Mismatched sequence lengths or empty sequences must raise ValueError."""
    with pytest.raises(ValueError, match=r"empty"):
        evaluate_data_efficiency([], [], [])

    with pytest.raises(ValueError, match=r"identical lengths"):
        evaluate_data_efficiency([100, 200], [0.5], [0.5, 0.6])


def test_reject_out_of_bounds_acceptance() -> None:
    """Acceptance values < 0 or > 1 must raise ValueError."""
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        evaluate_data_efficiency([100], [1.05], [0.5])

    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        evaluate_data_efficiency([100], [0.5], [-0.01])


def test_equal_curves_cannot_confirm_superiority() -> None:
    """Equal curves [0.5, 0.5] vs [0.5, 0.5] must NOT assert active superiority."""
    curve = evaluate_data_efficiency(
        native_simulation_budgets=[100, 250],
        active_acquisition_acceptance=[0.5, 0.5],
        random_acquisition_acceptance=[0.5, 0.5],
    )
    assert curve.active_superiority_confirmed is False
    # Descriptive ratio of equal curves is 1.0, but superiority is False
    assert curve.mean_acceptance_ratio == pytest.approx(1.0)
    assert curve.auc_acceptance_ratio == pytest.approx(1.0)


def test_zero_reference_remains_undefined_inconclusive() -> None:
    """When random acquisition has zero acceptance everywhere, ratio is undefined/inconclusive, not 1.0."""
    curve = evaluate_data_efficiency(
        native_simulation_budgets=[100, 250],
        active_acquisition_acceptance=[0.4, 0.6],
        random_acquisition_acceptance=[0.0, 0.0],
    )
    # Must NOT default to 1.0
    assert curve.mean_acceptance_ratio is None
    assert curve.auc_acceptance_ratio is None
    assert curve.sample_efficiency_multiplier is None
    assert curve.is_inconclusive is True
    assert curve.active_superiority_confirmed is False
    assert (
        "undefined" in curve.status_message.lower()
        or "zero" in curve.status_message.lower()
    )


def test_distinguish_mean_auc_and_budget_to_target_estimands() -> None:
    """Analytically verify distinction between mean ratio, trapezoidal AUC ratio, and budget-to-target savings."""
    # Budgets with unequal spacing to demonstrate AUC vs arithmetic mean divergence
    budgets = [100, 200, 1000]
    active_acc = [0.50, 0.80, 0.90]
    random_acc = [0.20, 0.30, 0.60]

    # Arithmetic means:
    # mean_active = (0.50 + 0.80 + 0.90) / 3 = 2.20 / 3 = 0.733333
    # mean_random = (0.20 + 0.30 + 0.60) / 3 = 1.10 / 3 = 0.366667
    # mean_ratio = 2.20 / 1.10 = 2.0

    # Trapezoidal AUC:
    # interval 1 (100 to 200, dx=100):
    #   active: 100 * (0.50 + 0.80) / 2 = 65.0
    #   random: 100 * (0.20 + 0.30) / 2 = 25.0
    # interval 2 (200 to 1000, dx=800):
    #   active: 800 * (0.80 + 0.90) / 2 = 680.0
    #   random: 800 * (0.30 + 0.60) / 2 = 360.0
    # active_auc = 65.0 + 680.0 = 745.0
    # random_auc = 25.0 + 360.0 = 385.0
    # auc_ratio = 745.0 / 385.0 = 1.93506...

    curve = evaluate_data_efficiency(
        native_simulation_budgets=budgets,
        active_acquisition_acceptance=active_acc,
        random_acquisition_acceptance=random_acc,
    )
    assert curve.mean_acceptance_ratio == pytest.approx(2.0, rel=1e-4)
    assert curve.auc_acceptance_ratio == pytest.approx(745.0 / 385.0, rel=1e-4)
    # Arithmetic mean ratio and AUC ratio are quantitatively distinct
    assert curve.mean_acceptance_ratio != pytest.approx(
        curve.auc_acceptance_ratio, rel=1e-3
    )
    assert curve.active_superiority_confirmed is True

    # Budget-to-target savings at target = 0.60 (which random achieves at budget 1000):
    # On active curve:
    # at budget 100: 0.50, at budget 200: 0.80.
    # Linear interpolation for active to reach 0.60:
    # frac = (0.60 - 0.50) / (0.80 - 0.50) = 0.10 / 0.30 = 1/3
    # B_active = 100 + (1/3) * (200 - 100) = 133.333
    # B_random to reach 0.60 is 1000.
    # Budget savings ratio = 1000 / 133.333 = 7.5x!
    savings_at_60 = compute_budget_to_target_ratio(
        native_simulation_budgets=budgets,
        active_acceptance_curve=active_acc,
        random_acceptance_curve=random_acc,
        target_acceptance=0.60,
    )
    assert savings_at_60 == pytest.approx(7.5, rel=1e-3)
    # The budget savings multiplier (7.5x) is vastly different from the descriptive ratio (2.0x or 1.93x)!


def test_budget_to_target_unattained_target_returns_none() -> None:
    """When target accuracy is not attained by either curve, return None rather than a misleading number."""
    budgets = [100, 250, 500]
    active_acc = [0.4, 0.6, 0.8]
    random_acc = [0.2, 0.3, 0.5]

    # Target 0.90 is unattained by both
    assert (
        compute_budget_to_target_ratio(
            budgets, active_acc, random_acc, target_acceptance=0.90
        )
        is None
    )

    # Target 0.70 is attained by active (at budget ~375), but NEVER attained by random (max 0.5)
    # Random cannot reach target within budget -> inconclusive / undefined budget savings
    assert (
        compute_budget_to_target_ratio(
            budgets, active_acc, random_acc, target_acceptance=0.70
        )
        is None
    )


def test_trapezoidal_auc_helper() -> None:
    """trapezoidal_auc computes accurate integration over arbitrary positive intervals."""
    budgets = [10, 20, 50]
    values = [0.2, 0.4, 0.8]
    # (20 - 10) * (0.2 + 0.4) / 2 = 10 * 0.3 = 3.0
    # (50 - 20) * (0.4 + 0.8) / 2 = 30 * 0.6 = 18.0
    # Total = 21.0
    assert trapezoidal_auc(budgets, values) == pytest.approx(21.0)


def test_multi_seed_uncertainty_and_statistical_superiority() -> None:
    """Multiple seeds with variance aggregate mean curves, standard errors, and test superiority."""
    budgets = [100, 250, 500, 1000]
    # 3 seeds for active and random
    active_runs = [
        [0.46, 0.66, 0.83, 0.93],
        [0.44, 0.64, 0.81, 0.91],
        [0.45, 0.65, 0.82, 0.92],
    ]
    random_runs = [
        [0.31, 0.49, 0.63, 0.75],
        [0.29, 0.47, 0.61, 0.73],
        [0.30, 0.48, 0.62, 0.74],
    ]

    summary = evaluate_multi_seed_data_efficiency(
        native_simulation_budgets=budgets,
        active_seed_runs=active_runs,
        random_seed_runs=random_runs,
        confidence_level=0.95,
    )
    assert isinstance(summary, MultiSeedEfficiencySummary)
    assert len(summary.mean_active_curve) == 4
    assert len(summary.mean_random_curve) == 4
    assert summary.active_statistically_superior is True
    assert summary.mean_auc_ratio is not None and summary.mean_auc_ratio > 1.2
    assert summary.p_value is not None and summary.p_value < 0.05
    assert summary.is_inconclusive is False


def test_multi_seed_overlapping_variance_is_inconclusive() -> None:
    """When seeds have large variance such that active is not consistently superior, mark inconclusive."""
    budgets = [100, 200]
    active_runs = [
        [0.55, 0.70],
        [0.40, 0.50],
        [0.45, 0.55],
    ]
    random_runs = [
        [0.50, 0.65],
        [0.48, 0.60],
        [0.42, 0.58],
    ]

    summary = evaluate_multi_seed_data_efficiency(
        native_simulation_budgets=budgets,
        active_seed_runs=active_runs,
        random_seed_runs=random_runs,
        confidence_level=0.95,
    )
    # The curves overlap substantially and active is not uniformly or statistically significantly superior
    assert summary.active_statistically_superior is False
