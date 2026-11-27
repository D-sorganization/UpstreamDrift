"""Only coherent SI traces may cross the bunker visualization boundary."""

from dataclasses import replace
import math

import pytest

from src.tools.bunker_shot_gui.flight import BunkerFlight

pytestmark = pytest.mark.unit


@pytest.fixture
def trace() -> BunkerFlight:
    return BunkerFlight(
        flight_m=((0.0, 0.0, 0.0), (1.5, -2.0, 1.0), (3.0, -4.0, 0.0)),
        rollout_m=((3.0, -4.0, 0.0), (3.3, -4.4, 0.1), (3.6, -4.8, 0.0)),
        carry_m=5.0,
    )


def test_valid_trace_preserves_bounce_and_offline_coordinates(
    trace: BunkerFlight,
) -> None:
    assert trace.carry_m == math.hypot(*trace.flight_m[-1][:2])
    assert trace.rollout_m[0] == trace.flight_m[-1]
    assert trace.rollout_m[1][2] > 0.0
    assert replace(trace, rollout_m=trace.rollout_m[:1]).rollout_m == (
        trace.flight_m[-1],
    )


def test_allows_floating_point_landing_residue() -> None:
    arc = tuple((float(x), 0.0, math.sin(math.pi * x / 9)) for x in range(10))
    assert BunkerFlight(arc, ((9.0, 0.0, 0.0),), carry_m=9.0).flight_m == arc


def test_contract_cannot_be_disabled(
    trace: BunkerFlight, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DBC_LEVEL", "off")
    with pytest.raises(ValueError, match="carry"):
        replace(trace, carry_m=float("nan"))


@pytest.mark.parametrize("carry", [float("nan"), float("inf"), -1.0, 4.0])
def test_rejects_invalid_or_mismatched_carry(trace: BunkerFlight, carry: float) -> None:
    with pytest.raises(ValueError, match="carry"):
        replace(trace, carry_m=carry)


@pytest.mark.parametrize("field", ["flight_m", "rollout_m"])
@pytest.mark.parametrize("coordinate", range(3))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_interior_coordinates(
    trace: BunkerFlight, field: str, coordinate: int, value: float
) -> None:
    path = list(getattr(trace, field))
    point = list(path[1])
    point[coordinate] = value
    path[1] = tuple(point)
    with pytest.raises(ValueError, match=field):
        replace(trace, **{field: tuple(path)})


@pytest.mark.parametrize("field", ["flight_m", "rollout_m"])
@pytest.mark.parametrize("point", [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0), (1.0, 2.0, -0.1)])
def test_rejects_wrong_shape_or_below_green_points(
    trace: BunkerFlight, field: str, point: tuple[float, ...]
) -> None:
    path = list(getattr(trace, field))
    path[1] = point
    with pytest.raises(ValueError, match=field):
        replace(trace, **{field: tuple(path)})


@pytest.mark.parametrize(
    "field,path",
    [("flight_m", ()), ("flight_m", ((0.0, 0.0, 0.0),)), ("rollout_m", ())],
)
def test_rejects_incomplete_paths(trace: BunkerFlight, field: str, path: tuple) -> None:
    with pytest.raises(ValueError, match=field):
        replace(trace, **{field: path})


@pytest.mark.parametrize(
    "field,path",
    [
        ("flight_m", ((0.0, 0.0, 0.0), (3.0, -4.0, 0.1))),
        ("rollout_m", ((3.1, -4.0, 0.0), (3.6, -4.8, 0.0))),
        ("rollout_m", ((3.0, -4.0, 0.0), (3.6, -4.8, 0.1))),
    ],
)
def test_rejects_missing_landing_disconnected_rollout_or_airborne_rest(
    trace: BunkerFlight, field: str, path: tuple
) -> None:
    with pytest.raises(ValueError, match="landing|rollout|rest"):
        replace(trace, **{field: path})
