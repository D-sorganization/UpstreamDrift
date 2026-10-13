"""Shared, explicit club-hypothesis presets for GUI and headless runs."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_presets_publish_illustrative_driver_iron_and_wedge_inputs() -> None:
    from src.tools.shot_pattern_analysis.presets import CLUB_PRESETS

    assert set(CLUB_PRESETS) == {"driver", "seven_iron", "pitching_wedge"}
    driver = CLUB_PRESETS["driver"]
    iron = CLUB_PRESETS["seven_iron"]
    wedge = CLUB_PRESETS["pitching_wedge"]
    assert (driver.club_speed_mps, driver.loft_deg, driver.attack_angle_deg) == (
        45.0,
        12.8,
        -0.9,
    )
    assert (iron.club_speed_mps, iron.loft_deg, iron.attack_angle_deg) == (
        36.0,
        24.0,
        -4.0,
    )
    assert (wedge.club_speed_mps, wedge.loft_deg, wedge.attack_angle_deg) == (
        32.0,
        36.7,
        -5.0,
    )
    assert all(p.illustrative for p in CLUB_PRESETS.values())
    assert all(p.source_assumption for p in CLUB_PRESETS.values())


def test_preset_lookup_rejects_unknown_ids() -> None:
    from src.tools.shot_pattern_analysis.presets import get_club_preset

    with pytest.raises(ValueError, match="unknown club preset"):
        get_club_preset("tour_average")
