"""Unit tests for motion matching pipeline constants."""

from __future__ import annotations

import numpy as np
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


def test_pipeline_default_paths_and_captures() -> None:
    from src.shared.python.motion_matching.pipeline import constants

    assert "driver" in constants.CAPTURES
    assert "iron" in constants.CAPTURES
    assert constants.CAPTURES["driver"].name == "C3D_TA_Driver.c3d"
    assert constants.CAPTURES["iron"].name == "C3D_TA_Iron.c3d"
    assert constants.SPEC.name == "full_body_spec_v2.json"
    assert constants.BUILD_RECEIPT.name == "build_receipt_v2.json"
    assert constants.CANDIDATE.name == "returned81_candidate.json"


def test_captures_lazy_owner() -> None:
    from unittest.mock import patch
    import os
    from src.shared.python.motion_matching.pipeline import constants
    from src.motion_capture.capture_registry import CaptureDataUnavailable

    assert constants.CAPTURE_NAMES == ("driver", "iron", "owner")
    # CAPTURES itself never grows an 'owner' key: it is a plain dict of the
    # two public fixtures, so dict(), copy(), and json all behave normally.
    assert set(constants.CAPTURES) == {"driver", "iron"}
    assert dict(constants.CAPTURES) == constants.CAPTURES

    # Resolving owner when CAPTURE_DATA_DIR is unset raises without breaking import
    with patch.dict(os.environ, {"CAPTURE_DATA_DIR": ""}, clear=False):
        with pytest.raises(CaptureDataUnavailable):
            _ = constants.capture_path("owner")

    # driver and iron still resolve normally
    assert constants.capture_path("driver").is_file()
    assert constants.capture_path("iron").is_file()


def test_capture_path_rejects_unknown_name() -> None:
    from src.shared.python.motion_matching.pipeline import constants

    with pytest.raises(ValueError, match="Unknown capture name"):
        constants.capture_path("nonexistent")


def test_rate_from_times_driver_and_owner_rates() -> None:
    from src.shared.python.motion_matching.pipeline.constants import rate_from_times

    driver_times = np.arange(654) / 360.0
    assert rate_from_times(driver_times) == pytest.approx(360.0)

    owner_times = np.arange(367) / 240.0
    assert rate_from_times(owner_times) == pytest.approx(240.0)


def test_rate_from_times_rejects_too_few_samples() -> None:
    from src.shared.python.motion_matching.pipeline.constants import rate_from_times

    with pytest.raises(ValueError, match="at least 2 samples"):
        rate_from_times([0.0])
    with pytest.raises(ValueError, match="at least 2 samples"):
        rate_from_times([])


def test_rate_from_times_rejects_nonpositive_spacing() -> None:
    from src.shared.python.motion_matching.pipeline.constants import rate_from_times

    with pytest.raises(ValueError, match="positive"):
        rate_from_times([0.0, 0.0, 0.1])
    with pytest.raises(ValueError, match="positive"):
        rate_from_times([0.0, -0.1])


def test_rate_from_times_rejects_nonuniform_spacing() -> None:
    from src.shared.python.motion_matching.pipeline.constants import rate_from_times

    with pytest.raises(ValueError, match="uniform"):
        rate_from_times([0.0, 0.1, 0.25])


def test_rate_from_times_tolerates_float_roundoff() -> None:
    from src.shared.python.motion_matching.pipeline.constants import rate_from_times

    times = np.cumsum([0.0] + [1.0 / 360.0] * 653)
    assert rate_from_times(times) == pytest.approx(360.0, rel=1e-3)
