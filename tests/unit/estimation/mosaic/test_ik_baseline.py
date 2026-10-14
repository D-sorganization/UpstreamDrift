"""Tests for the IK+smoothing baseline on the planar fixture."""

from __future__ import annotations

import pytest

from src.shared.python.estimation.mosaic.ik_baseline import (
    IKBaselineResult,
    run_ik_smoothing_baseline,
)

pytestmark = pytest.mark.unit


def test_ik_smoothing_baseline_measures_time_and_fails_open_loop_replay() -> None:
    # Measure time on the planar fixture without estimating any figure
    result = run_ik_smoothing_baseline()
    assert isinstance(result, IKBaselineResult)

    # Time to match is measured strictly positive, never None or estimated
    assert result.time_to_match_s > 0.0
    assert isinstance(result.time_to_match_s, float)

    # Standard IK + smoothing (uncalibrated priors, no dynamics collocation)
    # fails open-loop replay divergence gate
    assert result.open_loop_accepted is False or result.open_loop_rms_rad > 0.05
