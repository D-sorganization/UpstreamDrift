"""Tests for the launch-monitor ingestion API."""

from __future__ import annotations

import math
import textwrap

import numpy as np
import pytest

from src.shared.python.launch_monitor import (
    FlightScopeAdapter,
    LaunchMonitorShot,
    TrackManAdapter,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fixture CSV strings
# ---------------------------------------------------------------------------

_TRACKMAN_CSV = textwrap.dedent("""\
    Club,Ball Speed,Club Speed,Smash Factor,Launch Angle,Launch Direction,Back Spin,Side Spin,Spin Axis,Total Spin,Carry,Total,Height,Descent Angle,Time of Flight,Attack Angle,Dyn. Loft,Club Path,Face Angle
    Driver,174.3,112.5,1.55,10.4,1.2,2545,-120,2.7,2548,282.0,305.0,105.0,38.0,6.5,-1.5,14.2,2.1,0.8
    7-Iron,123.0,90.5,1.36,16.3,-0.5,7097,200,-1.6,7100,176.0,180.0,102.0,50.0,5.8,-4.2,21.5,-0.3,-0.2
    """)

_FLIGHTSCOPE_CSV = textwrap.dedent("""\
    Club Name,Ball Speed (mph),Club Speed (mph),Smash Factor,Launch Angle (deg),Launch Direction (deg),Backspin (rpm),Sidespin (rpm),Spin Axis (deg),Total Spin (rpm),Carry (yds),Total (yds),Max Height (yds),Descent Angle (deg),Hang Time (s),Attack Angle (deg)
    Driver,172.0,110.0,1.56,10.8,0.5,2600,-80,1.8,2601,278.0,300.0,100.0,37.0,6.3,-1.2
    PW,102.0,75.0,1.36,24.0,-1.0,9300,150,-0.9,9301,141.0,143.0,93.0,52.0,5.5,-5.1
    """)

_TRACKMAN_MINIMAL_CSV = textwrap.dedent("""\
    Club,Ball Speed,Total Spin,Carry
    Driver,174.3,2548,282.0
    """)

_TRACKMAN_EMPTY_CSV = "Club,Ball Speed,Total Spin,Carry\n"


# ---------------------------------------------------------------------------
# TrackMan adapter
# ---------------------------------------------------------------------------


class TestTrackManAdapter:
    def test_parses_two_shots(self):
        shots = TrackManAdapter.from_string(_TRACKMAN_CSV)
        assert len(shots) == 2

    def test_driver_ball_speed_converted(self):
        shots = TrackManAdapter.from_string(_TRACKMAN_CSV)
        driver = shots[0]
        expected = 174.3 * 0.44704
        assert abs(driver.ball_speed_mps - expected) < 0.01

    def test_driver_carry_converted(self):
        shots = TrackManAdapter.from_string(_TRACKMAN_CSV)
        driver = shots[0]
        expected = 282.0 * 0.9144
        assert abs(driver.carry_m - expected) < 0.01

    def test_club_name_preserved(self):
        shots = TrackManAdapter.from_string(_TRACKMAN_CSV)
        assert shots[0].club == "Driver"
        assert shots[1].club == "7-Iron"

    def test_source_is_trackman(self):
        shots = TrackManAdapter.from_string(_TRACKMAN_CSV)
        assert all(s.source == "TrackMan" for s in shots)

    def test_spin_fields(self):
        shots = TrackManAdapter.from_string(_TRACKMAN_CSV)
        driver = shots[0]
        assert driver.back_spin_rpm == pytest.approx(2545.0)
        assert driver.side_spin_rpm == pytest.approx(-120.0)
        assert driver.total_spin_rpm == pytest.approx(2548.0)

    def test_optional_fields_parsed(self):
        shots = TrackManAdapter.from_string(_TRACKMAN_CSV)
        driver = shots[0]
        assert driver.max_height_m is not None
        assert driver.landing_angle_deg == pytest.approx(38.0)
        assert driver.flight_time_s == pytest.approx(6.5)
        assert driver.dynamic_loft_deg == pytest.approx(14.2)

    def test_minimal_csv_works(self):
        shots = TrackManAdapter.from_string(_TRACKMAN_MINIMAL_CSV)
        assert len(shots) == 1
        assert shots[0].total_m is None

    def test_empty_csv_returns_empty_list(self):
        shots = TrackManAdapter.from_string(_TRACKMAN_EMPTY_CSV)
        assert shots == []

    def test_smash_computed_when_missing(self):
        csv = "Club,Ball Speed,Club Speed,Carry\nDriver,174.3,112.5,282.0\n"
        shots = TrackManAdapter.from_string(csv)
        expected = (174.3 * 0.44704) / (112.5 * 0.44704)
        assert shots[0].smash_factor == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# FlightScope adapter
# ---------------------------------------------------------------------------


class TestFlightScopeAdapter:
    def test_parses_two_shots(self):
        shots = FlightScopeAdapter.from_string(_FLIGHTSCOPE_CSV)
        assert len(shots) == 2

    def test_driver_ball_speed(self):
        shots = FlightScopeAdapter.from_string(_FLIGHTSCOPE_CSV)
        expected = 172.0 * 0.44704
        assert abs(shots[0].ball_speed_mps - expected) < 0.01

    def test_source_is_flightscope(self):
        shots = FlightScopeAdapter.from_string(_FLIGHTSCOPE_CSV)
        assert all(s.source == "FlightScope" for s in shots)

    def test_pw_smash_factor(self):
        shots = FlightScopeAdapter.from_string(_FLIGHTSCOPE_CSV)
        pw = shots[1]
        assert pw.smash_factor == pytest.approx(1.36)


# ---------------------------------------------------------------------------
# LaunchMonitorShot type
# ---------------------------------------------------------------------------


class TestLaunchMonitorShot:
    @pytest.fixture()
    def driver_shot(self) -> LaunchMonitorShot:
        return LaunchMonitorShot(
            club="Driver",
            ball_speed_mps=77.9,
            club_speed_mps=50.3,
            smash_factor=1.549,
            launch_angle_deg=10.4,
            launch_direction_deg=1.2,
            back_spin_rpm=2545,
            side_spin_rpm=-120,
            spin_axis_deg=2.7,
            total_spin_rpm=2548,
            carry_m=257.8,
        )

    def test_negative_ball_speed_raises(self):
        with pytest.raises(ValueError, match="ball_speed_mps"):
            LaunchMonitorShot(
                club="Driver",
                ball_speed_mps=-1.0,
                club_speed_mps=50.0,
                smash_factor=1.4,
                launch_angle_deg=10.0,
                launch_direction_deg=0.0,
                back_spin_rpm=2500,
                side_spin_rpm=0,
                spin_axis_deg=0.0,
                total_spin_rpm=2500,
                carry_m=250.0,
            )

    def test_to_launch_conditions_velocity(self, driver_shot):
        lc = driver_shot.to_launch_conditions()
        assert lc.velocity == pytest.approx(driver_shot.ball_speed_mps)

    def test_to_launch_conditions_launch_angle(self, driver_shot):
        lc = driver_shot.to_launch_conditions()
        assert lc.launch_angle == pytest.approx(math.radians(10.4), rel=1e-4)

    def test_to_launch_conditions_spin_axis_unit_vector(self, driver_shot):
        lc = driver_shot.to_launch_conditions()
        norm = float(np.linalg.norm(lc.spin_axis))
        assert abs(norm - 1.0) < 1e-10

    def test_to_launch_conditions_spin_rate(self, driver_shot):
        lc = driver_shot.to_launch_conditions()
        assert lc.spin_rate == pytest.approx(2548.0)

    def test_extra_fields_stored(self):
        shots = TrackManAdapter.from_string(_TRACKMAN_CSV)
        # All known columns should be consumed; no extra expected for this CSV
        assert isinstance(shots[0].extra, dict)
