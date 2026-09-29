"""Contracts for finite command rise-time sensitivity."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.e1e_smooth_command_sensitivity import (
    TIME_CONSTANTS_S,
    evaluate_smooth_command_sensitivity,
    first_order_command_filter,
)

pytestmark = pytest.mark.scientific


def test_filter_preserves_preload_and_smooths_later_switches() -> None:
    raw = np.zeros((100, 2))
    raw[:, 0] = 60.0
    raw[:50, 1] = -10.0
    raw[50:, 1] = 15.0
    filtered = first_order_command_filter(raw, dt=0.001, time_constant_s=0.020)

    np.testing.assert_array_equal(filtered[0], raw[0])
    assert filtered[49, 1] == pytest.approx(-10.0)
    assert -10.0 < filtered[50, 1] < 15.0
    assert np.max(np.abs(np.diff(filtered[:, 1]))) < 25.0


def test_registered_ordering_survives_declared_rise_times() -> None:
    result = evaluate_smooth_command_sensitivity()
    assert result["time_constants_s"] == list(TIME_CONSTANTS_S)
    for by_shoulder in result["summaries"].values():
        for summary in by_shoulder.values():
            assert summary["attempted_programs"] == 46
            assert summary["registered_ordering_preserved"] is True
