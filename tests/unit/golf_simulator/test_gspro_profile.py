"""Unit tests for the frozen GSPro compatibility profile (GS-00 #10189).

Follows TDD, DbC, LoD and DRY principles.
"""

from __future__ import annotations

import pytest

from src.shared.python.golf_simulator.adapters.gspro.profile import (
    DEFAULT_GSPRO_PROFILE,
    FieldObservationStatus,
    GSProProfile,
    ResponseCategory,
)

pytestmark = pytest.mark.unit


def test_default_gspro_profile_frozen() -> None:
    profile = DEFAULT_GSPRO_PROFILE
    assert profile.profile_id == "gspro_open_connect_v1"
    assert profile.port == 921
    assert profile.host == "127.0.0.1"
    assert profile.speed_unit == "mph"
    assert profile.distance_unit in ("Yards", "Meters")
    assert profile.hla_sign_positive == "right"
    assert profile.max_message_bytes == 65536


def test_field_observation_matrix() -> None:
    profile = DEFAULT_GSPRO_PROFILE
    # Observed and documented fields
    assert profile.get_field_status("BallData.Speed") == FieldObservationStatus.OBSERVED
    assert (
        profile.get_field_status("BallData.TotalSpin")
        == FieldObservationStatus.OBSERVED
    )
    assert profile.get_field_status("BallData.VLA") == FieldObservationStatus.OBSERVED
    assert profile.get_field_status("BallData.HLA") == FieldObservationStatus.OBSERVED
    assert (
        profile.get_field_status("BallData.SpinAxis") == FieldObservationStatus.OBSERVED
    )

    # Unverified / unsupported fields
    assert (
        profile.get_field_status("Course.AimAngle")
        == FieldObservationStatus.UNSUPPORTED
    )
    assert (
        profile.get_field_status("NativeAvatar") == FieldObservationStatus.UNSUPPORTED
    )


def test_response_code_categorization() -> None:
    profile = DEFAULT_GSPRO_PROFILE
    assert profile.categorize_code(200) == ResponseCategory.CONFIRMED_ACCEPTED
    assert profile.categorize_code(201) == ResponseCategory.PLAYER_UPDATE
    assert profile.categorize_code(501) == ResponseCategory.ERROR_REJECTED
    assert profile.categorize_code(502) == ResponseCategory.ERROR_REJECTED
    assert profile.categorize_code(999) == ResponseCategory.UNKNOWN
