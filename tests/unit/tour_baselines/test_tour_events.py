"""Unit tests for Tour Baselines biomechanical swing events and native clocks (TB-01 #10586)."""

from pathlib import Path
import pytest
import numpy as np

from src.shared.python.motion_matching.tour_capture_contract import load_tour_capture
from src.shared.python.tour_baselines.events import (
    SWING_EVENTS_DRIVER,
    SWING_EVENTS_IRON,
    DetectionMethod,
    detect_tour_events,
)

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER_C3D = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"
IRON_C3D = REPO_ROOT / "data" / "C3D_TA_Iron.c3d"


def test_driver_and_iron_native_clocks_are_independent() -> None:
    driver = SWING_EVENTS_DRIVER
    iron = SWING_EVENTS_IRON

    # Driver is 360.0 Hz; Iron is 359.0 Hz
    assert driver.rate_hz == 360.0
    assert iron.rate_hz == 359.0

    # Impact frames and timestamps on native clocks
    assert driver.impact.frame_index == 476
    assert driver.impact.time_s == pytest.approx(476 / 360.0)

    assert iron.impact.frame_index == 480
    assert iron.impact.time_s == pytest.approx(480 / 359.0)

    # Both impacts are explicitly marked as inferred
    assert driver.impact.is_inferred is True
    assert iron.impact.is_inferred is True
    assert driver.impact.detection_method == DetectionMethod.TRAJECTORY_SPEED_PEAK
    assert iron.impact.detection_method == DetectionMethod.TRAJECTORY_SPEED_PEAK


def test_swing_phases_and_intervals() -> None:
    driver = SWING_EVENTS_DRIVER
    # Address interval: 0 to takeaway
    t0, t1 = driver.address_interval()
    assert t0 == 0.0
    assert t1 == driver.takeaway.time_s

    # Backswing interval: takeaway to top_of_backswing
    t0, t1 = driver.backswing_interval()
    assert t0 == driver.takeaway.time_s
    assert t1 == driver.top_of_backswing.time_s

    # Downswing interval: top_of_backswing to impact
    t0, t1 = driver.downswing_interval()
    assert t0 == driver.top_of_backswing.time_s
    assert t1 == driver.impact.time_s

    # Follow-through interval: impact to finish
    t0, t1 = driver.follow_through_interval()
    assert t0 == driver.impact.time_s
    assert t1 == driver.finish.time_s


def test_manual_override_event_detection() -> None:
    cap = load_tour_capture(DRIVER_C3D)
    overrides = {"impact_frame": 470, "top_frame": 390}
    events = detect_tour_events(cap, kind="driver", overrides=overrides)

    assert events.impact.frame_index == 470
    assert events.impact.time_s == pytest.approx(470 / 360.0)
    assert events.impact.detection_method == DetectionMethod.MANUAL_OVERRIDE
    assert events.impact.manual_override is True

    assert events.top_of_backswing.frame_index == 390
    assert events.top_of_backswing.manual_override is True


def test_detect_tour_events_from_capture() -> None:
    cap_driver = load_tour_capture(DRIVER_C3D)
    events_driver = detect_tour_events(cap_driver, kind="driver")
    assert events_driver.impact.frame_index == 476

    cap_iron = load_tour_capture(IRON_C3D)
    events_iron = detect_tour_events(cap_iron, kind="iron")
    assert events_iron.impact.frame_index == 480
