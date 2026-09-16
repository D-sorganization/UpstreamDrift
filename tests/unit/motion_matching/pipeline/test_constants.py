"""Unit tests for motion matching pipeline constants."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_pipeline_constants_present_and_valid() -> None:
    from src.shared.python.motion_matching.pipeline import constants

    assert constants.DT_S > 0
    assert constants.RATE_HZ > 0
    assert constants.OMEGA_RAD_S > 0
    assert constants.BALANCE == (60.0, 15.0)
    assert constants.REFERENCE_CUTOFF_HZ > 0
    assert constants.TRACKING_CUTOFF_HZ > 0
    assert constants.ZMP_MARGIN_M > 0
    assert constants.ZMP_FILTER_ITERATIONS >= 1
    assert constants.ZMP_COM_WEIGHT > 0
    assert 0 < constants.SHOOTING_RELAXATION <= 1.0
    assert len(constants.SHOOTING_LOCKED) == 5
    assert len(constants.TOE_SPHERES) == 2
    assert len(constants.LEG_SEEDS) == 8
    assert "RKneeOut" in constants.LEG_SEEDS
    assert "LToeOut" in constants.LEG_SEEDS
    assert constants.STATIC_FRAMES > 0
    assert constants.CALIBRATION_STRIDE > 0
    assert constants.BOUND_WIDENING >= 1.0
    assert (
        frozenset(
            {"LWInputX", "RWInputX", "LWInputY", "RWInputY", "LFInput", "RFInput"}
        )
        == constants.IK_UNBOUNDED
    )
