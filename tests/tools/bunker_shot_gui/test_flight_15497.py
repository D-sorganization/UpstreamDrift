"""Flight handoff must preserve the bunker launch and refusal contract."""

from dataclasses import replace
import math

import pytest

from bunkershot3d.ball import compute_bunker_launch
from tests.bunkershot3d.ball.test_pipeline_handoff import nominal_state
from src.tools.bunker_shot_gui.flight import bunker_flight

pytestmark = pytest.mark.integration


def test_valid_launch_reaches_green_with_consistent_landing() -> None:
    launch = compute_bunker_launch(nominal_state())
    trace = bunker_flight(launch, refused=False)
    assert trace is not None
    assert len(trace.flight_m) >= 2
    assert trace.flight_m[-1][2] == pytest.approx(0.0)
    assert trace.carry_m == pytest.approx(math.hypot(*trace.flight_m[-1][:2]))
    assert trace.rollout_m[0][:2] == pytest.approx(trace.flight_m[-1][:2])
    assert len(trace.rollout_m) > 1


def test_refusal_and_invalid_launch_never_fly() -> None:
    launch = compute_bunker_launch(nominal_state())
    assert bunker_flight(launch, refused=True) is None
    assert (
        bunker_flight(replace(launch, ball_speed_m_s=float("nan")), refused=False)
        is None
    )


def test_simulator_receives_release_spin_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.shared.python.physics.ball_simulator import BallFlightSimulator

    launch = compute_bunker_launch(nominal_state())
    seen = []
    original = BallFlightSimulator.simulate_trajectory

    def record(self, conditions, *args, **kwargs):
        seen.append(conditions)
        return original(self, conditions, *args, **kwargs)

    monkeypatch.setattr(BallFlightSimulator, "simulate_trajectory", record)
    assert bunker_flight(launch, refused=False) is not None
    assert seen[0].velocity == launch.ball_speed_m_s
    assert seen[0].launch_angle == launch.launch_angle_rad
    assert seen[0].azimuth_angle == launch.azimuth_rad
    assert seen[0].spin_rate == launch.spin_rate_rpm
    assert seen[0].spin_axis == pytest.approx(launch.spin_axis)
