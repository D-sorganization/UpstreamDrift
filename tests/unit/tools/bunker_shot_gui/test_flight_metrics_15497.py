"""Display-neutral readout numbers derived from a coherent BunkerFlight."""

import dataclasses
import math

import pytest

from src.tools.bunker_shot_gui.flight import BunkerFlight
from src.tools.bunker_shot_gui.flight_metrics import FlightMetrics, flight_metrics

pytestmark = pytest.mark.unit


@pytest.fixture
def straight() -> BunkerFlight:
    """Down-the-line flight: apex 6 m at x=10, lands at x=20, rolls to x=26."""
    return BunkerFlight(
        flight_m=(
            (0.0, 0.0, 0.0),
            (5.0, 0.0, 4.0),
            (10.0, 0.0, 6.0),
            (15.0, 0.0, 4.0),
            (20.0, 0.0, 0.0),
        ),
        rollout_m=((20.0, 0.0, 0.0), (23.0, 0.0, 0.5), (26.0, 0.0, 0.0)),
        carry_m=20.0,
    )


def test_straight_flight_numbers(straight: BunkerFlight) -> None:
    m = flight_metrics(straight)
    assert m is not None
    assert m.carry_m == pytest.approx(20.0)
    assert m.apex_height_m == pytest.approx(6.0)
    assert m.apex_downrange_m == pytest.approx(10.0)
    assert m.descent_angle_deg == pytest.approx(math.degrees(math.atan2(4.0, 5.0)))
    assert m.rollout_m == pytest.approx(6.0)
    assert m.total_m == pytest.approx(26.0)
    assert m.lateral_m == pytest.approx(0.0, abs=1e-12)


def test_offline_shot_reports_signed_lateral_and_radial_distances() -> None:
    # y is left-positive, so a rest point at y=-4.8 is 4.8 m right of the target line.
    flight = BunkerFlight(
        flight_m=((0.0, 0.0, 0.0), (1.5, -2.0, 1.0), (3.0, -4.0, 0.0)),
        rollout_m=((3.0, -4.0, 0.0), (3.3, -4.4, 0.1), (3.6, -4.8, 0.0)),
        carry_m=5.0,
    )
    m = flight_metrics(flight)
    assert m is not None
    assert m.lateral_m == pytest.approx(-4.8)
    assert m.total_m == pytest.approx(6.0)
    assert m.rollout_m == pytest.approx(1.0)  # ground path (0.6, -0.8) length
    assert m.apex_downrange_m == pytest.approx(math.hypot(1.5, -2.0))


def test_rollout_length_follows_the_ground_path_not_the_bounce_height(
    straight: BunkerFlight,
) -> None:
    taller = dataclasses.replace(
        straight,
        rollout_m=((20.0, 0.0, 0.0), (23.0, 0.0, 9.0), (26.0, 0.0, 0.0)),
    )
    m = flight_metrics(taller)
    assert m is not None
    assert m.rollout_m == pytest.approx(6.0)


def test_rest_at_landing_has_zero_rollout(straight: BunkerFlight) -> None:
    stuck = dataclasses.replace(straight, rollout_m=((20.0, 0.0, 0.0),))
    m = flight_metrics(stuck)
    assert m is not None
    assert m.rollout_m == 0.0
    assert m.total_m == pytest.approx(m.carry_m)


def test_no_flight_gives_no_metrics() -> None:
    """ADR-0032: a refused or unflown shot has nothing to read out."""
    assert flight_metrics(None) is None


def test_non_flight_input_is_a_contract_violation() -> None:
    with pytest.raises(TypeError):
        flight_metrics({"carry_m": 20.0})  # type: ignore[arg-type]


def test_vertical_final_segment_is_90_degrees_and_stalled_is_unknown() -> None:
    plunge = BunkerFlight(
        flight_m=((0.0, 0.0, 0.0), (5.0, 0.0, 3.0), (5.0, 0.0, 0.0)),
        rollout_m=((5.0, 0.0, 0.0),),
        carry_m=5.0,
    )
    m = flight_metrics(plunge)
    assert m is not None
    assert m.descent_angle_deg == pytest.approx(90.0)

    stalled = BunkerFlight(
        flight_m=((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 0.0, 0.0)),
        rollout_m=((5.0, 0.0, 0.0),),
        carry_m=5.0,
    )
    s = flight_metrics(stalled)
    assert s is not None
    assert s.descent_angle_deg is None


def test_metrics_are_immutable(straight: BunkerFlight) -> None:
    m = flight_metrics(straight)
    assert isinstance(m, FlightMetrics)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.carry_m = 1.0  # type: ignore[misc]
