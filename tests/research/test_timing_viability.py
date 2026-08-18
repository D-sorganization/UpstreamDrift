from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.timing_viability import (
    ViabilityLimits,
    largest_contiguous_width_s,
    summarize_timing_viability,
    viability_mask,
)

pytestmark = pytest.mark.scientific


METRIC_NAMES = (
    "delivery_speed_m_s",
    "face_path_error_deg",
    "peak_hand_force_n",
    "effort_proxy_nms",
    "returned_to_viable_set",
    "normalized_energy_residual",
    "realized_event_time_s",
)


def test_viability_mask_applies_common_performance_load_and_recovery_guards() -> None:
    baseline = np.array([10.0, 3.0, 100.0, 20.0, 1.0, 1e-8, 0.14])
    outcomes = np.array(
        [
            [9.6, 4.5, 108.0, 21.0, 1.0, 2e-8, 0.13],
            [9.4, 4.5, 108.0, 21.0, 1.0, 2e-8, 0.14],
            [9.6, 6.0, 108.0, 21.0, 1.0, 2e-8, 0.15],
            [9.6, 4.5, 112.0, 21.0, 1.0, 2e-8, 0.16],
            [9.6, 4.5, 108.0, 21.0, 0.0, 2e-8, 0.17],
        ]
    )
    limits = ViabilityLimits()

    result = viability_mask(outcomes, baseline, METRIC_NAMES, limits)

    assert result.tolist() == [True, False, False, False, False]


def test_largest_contiguous_width_uses_declared_phase_coordinate() -> None:
    offsets = np.array([-0.03, -0.015, 0.0, 0.015, 0.03])
    assert largest_contiguous_width_s(
        offsets, np.array([0, 1, 1, 1, 0])
    ) == pytest.approx(0.03)
    assert largest_contiguous_width_s(offsets, np.ones(5, dtype=bool)) == pytest.approx(
        0.06
    )
    assert largest_contiguous_width_s(offsets, np.zeros(5, dtype=bool)) == 0.0


def test_summary_retains_each_load_and_requires_robust_intersection() -> None:
    offsets = np.array([-0.02, 0.0, 0.02])
    baseline = np.array(
        [
            [10.0, 2.0, 100.0, 20.0, 1.0, 0.0, 0.14],
            [9.0, 3.0, 110.0, 22.0, 1.0, 0.0, 0.15],
        ]
    )
    outcomes = np.repeat(baseline[:, None, :], 3, axis=1)
    outcomes[0, 0, 0] = 8.0
    outcomes[1, 2, 0] = 7.0

    summary = summarize_timing_viability(
        offsets,
        outcomes,
        baseline,
        load_names=("nominal", "adverse"),
        metric_names=METRIC_NAMES,
        limits=ViabilityLimits(),
    )

    assert summary["per_load"]["nominal"]["viable_mask"] == [False, True, True]
    assert summary["per_load"]["adverse"]["viable_mask"] == [True, True, False]
    assert summary["robust_viable_mask"] == [False, True, False]
    assert summary["robust_viable_fraction"] == pytest.approx(1 / 3)
    assert summary["robust_contiguous_width_s"] == 0.0


def test_contracts_fail_closed_on_invalid_shapes_and_limits() -> None:
    with pytest.raises(ValueError, match="speed_fraction_min"):
        ViabilityLimits(speed_fraction_min=1.1)
    with pytest.raises(ValueError, match="strictly increasing"):
        largest_contiguous_width_s(np.array([0.0, 0.0]), np.ones(2, dtype=bool))
    with pytest.raises(ValueError, match="metric_names"):
        viability_mask(np.ones((2, 3)), np.ones(3), ("a", "b"), ViabilityLimits())
