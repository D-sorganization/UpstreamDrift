"""Launch-spin validity envelope per club class (#15443).

The driver cap (3500 rpm) refused every iron flight; the envelope is now read
per club class from ``ball_flight_validity_settings.json``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.impact_parameters.flight_report import spin_envelope_reason
from src.shared.python.physics.ball_flight_validity import (
    MAX_DRIVER_LAUNCH_SPIN_RPM,
    LaunchSpinEnvelope,
    clamp_launch_spin_rate,
    launch_spin_envelope,
)
from src.shared.python.physics.ball_launch_conditions import LaunchConditions
from src.shared.python.physics.swing_ball_flight_pipeline import (
    SwingBallFlightPipeline,
    SwingState,
)

pytestmark = pytest.mark.unit

IRON_7_RPM = 7500.0  # a 7-iron launches near here (tour average 7,097 rpm)


def _launch(spin_rpm: float) -> LaunchConditions:
    return LaunchConditions(
        velocity=60.0,
        launch_angle=0.3,
        azimuth_angle=0.0,
        spin_rate=spin_rpm,
        spin_axis=np.array([0.0, 1.0, 0.0]),
    )


def _iron_swing() -> SwingState:
    loft_deg = 34.0
    loft = math.radians(loft_deg)
    normal = np.array([math.cos(loft), 0.0, math.sin(loft)])
    return SwingState(
        clubhead_velocity=np.array([40.0, 0.0, 0.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=normal,
        clubhead_loft_deg=loft_deg,
    )


@pytest.mark.parametrize(
    ("club", "expected"),
    [
        ("Driver", "driver"),
        ("driver", "driver"),
        ("1W", "driver"),
        ("3 Wood", "wood"),
        ("5w", "wood"),
        ("4 Hybrid", "hybrid"),
        ("iron7", "iron"),
        ("7 Iron", "iron"),
        ("7i", "iron"),
        ("Pitching Wedge", "wedge"),
        ("SW", "wedge"),
        ("mystery club", "driver"),
        ("", "driver"),
        (None, "driver"),
    ],
)
def test_club_name_decides_the_envelope_class(club, expected) -> None:
    assert launch_spin_envelope(club).club_class == expected


def test_driver_and_wood_keep_the_original_cap() -> None:
    assert MAX_DRIVER_LAUNCH_SPIN_RPM == 3500.0
    assert launch_spin_envelope("Driver").max_rpm == 3500.0
    assert launch_spin_envelope("3 Wood").max_rpm == 3500.0


def test_iron_and_wedge_envelopes_cover_tour_averages_with_margin() -> None:
    assert launch_spin_envelope("7 Iron").max_rpm > 1.05 * 7097.0
    assert launch_spin_envelope("Pitching Wedge").max_rpm > 1.05 * 9300.0
    assert launch_spin_envelope("7 Iron").max_rpm > IRON_7_RPM


def test_unknown_club_uses_the_conservative_driver_envelope() -> None:
    assert launch_spin_envelope("mystery club").max_rpm == MAX_DRIVER_LAUNCH_SPIN_RPM


def test_envelope_rejects_a_non_string_club() -> None:
    with pytest.raises(TypeError, match="club"):
        launch_spin_envelope(7)  # type: ignore[arg-type]


def test_driver_at_3600_rpm_is_still_clamped_and_invalid() -> None:
    envelope = launch_spin_envelope("Driver")
    clamped, was_clamped = clamp_launch_spin_rate(_launch(3600.0), envelope.max_rpm)
    assert was_clamped and clamped.spin_rate == pytest.approx(3500.0)
    meta = {"launch_spin_clamped": True, "pre_clamp_launch_spin_rpm": 3600.0}
    reason = spin_envelope_reason(meta, 3500.0, "Driver")
    assert "driver validity envelope" in reason


def test_pipeline_leaves_a_7_iron_at_7500_rpm_unclamped_and_valid() -> None:
    pipeline = SwingBallFlightPipeline()
    base = pipeline.run(_iron_swing(), club="7 Iron").launch_conditions.spin_rate
    result = pipeline.run(_iron_swing(), spin_factor=IRON_7_RPM / base, club="7 Iron")
    assert result.launch_conditions.spin_rate == pytest.approx(IRON_7_RPM)
    assert not result.metadata.get("launch_spin_clamped")
    assert result.metadata["flight_valid"] is True
    meta = dict(result.metadata)
    assert spin_envelope_reason(meta, IRON_7_RPM, "7 Iron") == ""


def test_pipeline_clamps_a_7_iron_above_its_cap_to_the_iron_cap() -> None:
    pipeline = SwingBallFlightPipeline()
    cap = launch_spin_envelope("7 Iron").max_rpm
    base = pipeline.run(_iron_swing(), club="7 Iron").launch_conditions.spin_rate
    result = pipeline.run(_iron_swing(), spin_factor=1.2 * cap / base, club="7 Iron")
    assert result.metadata["launch_spin_clamped"] is True
    assert result.launch_conditions.spin_rate == pytest.approx(cap)


def test_pipeline_without_a_club_clamps_at_the_driver_cap() -> None:
    pipeline = SwingBallFlightPipeline()
    base = pipeline.run(_iron_swing()).launch_conditions.spin_rate
    result = pipeline.run(_iron_swing(), spin_factor=4000.0 / base)
    assert result.metadata["launch_spin_clamped"] is True
    assert result.launch_conditions.spin_rate == pytest.approx(3500.0)


def test_iron_above_its_cap_is_invalid_and_names_the_iron_envelope() -> None:
    cap = launch_spin_envelope("7 Iron").max_rpm
    meta = {"launch_spin_clamped": True, "pre_clamp_launch_spin_rpm": cap + 500.0}
    reason = spin_envelope_reason(meta, cap, "7 Iron")
    assert "iron validity envelope" in reason
    assert "driver" not in reason
    assert f"{cap + 500.0:.0f} rpm" in reason


def test_report_only_flags_spin_beyond_the_margin() -> None:
    meta = {"launch_spin_clamped": True, "pre_clamp_launch_spin_rpm": 3500.5}
    assert spin_envelope_reason(meta, 3500.0, "Driver") == ""


def test_envelope_dataclass_validates_its_cap() -> None:
    with pytest.raises(ValueError, match="max_rpm"):
        LaunchSpinEnvelope(club_class="iron", label="iron", max_rpm=0.0)
