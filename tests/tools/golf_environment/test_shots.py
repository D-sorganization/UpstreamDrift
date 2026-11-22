"""Tests for the pure-data shot placement on a hole (epic #15150)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.tools.golf_environment import shots as sh

pytestmark = pytest.mark.unit

RED = (1.0, 0.0, 0.0, 1.0)


@dataclass
class _Hole:
    tee_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pin_position: tuple[float, float, float] = (380.0, 0.0, 0.0)
    fairway_width: float = 35.0
    green_radius: float = 15.0


def _flight(carry: float, lateral: float) -> sh.TargetFlight:
    """Parabola in target-frame axes (down-range, left, up); lateral is right."""
    s = np.linspace(0.0, 1.0, 21)
    path = np.column_stack((carry * s, -lateral * s, 30.0 * 4.0 * s * (1.0 - s)))
    return sh.TargetFlight(path, carry, lateral)


def _place(flights, hole=None):
    n = len(flights)
    return sh.shots_on_hole(flights, hole or _Hole(), ["s"] * n, [RED] * n)


def test_straight_shot_lands_on_centre_line_at_carry() -> None:
    (shot,) = _place([_flight(190.0, 0.0)])
    assert shot.path_m[-1] == pytest.approx([190.0, 0.0, 0.0], abs=1e-9)
    assert shot.carry_m == 190.0 and shot.side_m == 0.0


def test_right_shot_lands_right_of_line_seen_from_tee() -> None:
    (shot,) = _place([_flight(200.0, 15.0)])
    forward = np.array([1.0, 0.0])
    right = np.array([forward[1], -forward[0]])  # looking tee to pin
    assert np.dot(shot.path_m[-1, :2], right) == pytest.approx(15.0)


def test_pin_direction_and_tee_offset_are_followed() -> None:
    hole = _Hole(tee_position=(10.0, 20.0, 0.5), pin_position=(10.0, 420.0, 0.5))
    (shot,) = _place([_flight(100.0, 10.0)], hole)
    # tee to pin is +y, so the golfer's right is +x
    assert shot.path_m[0] == pytest.approx([10.0, 20.0, 0.5])
    assert shot.path_m[-1] == pytest.approx([20.0, 120.0, 0.5], abs=1e-9)


def test_n_in_n_out_keeps_labels_and_colours() -> None:
    flights = [_flight(160.0 + 20 * i, i - 1.0) for i in range(3)]
    out = sh.shots_on_hole(flights, _Hole(), list("abc"), [RED] * 3)
    assert [s.label for s in out] == list("abc")
    assert len(out) == 3


@pytest.mark.parametrize("bad", [np.zeros((5, 2)), np.zeros(5), np.zeros((1, 3))])
def test_bad_shapes_raise(bad) -> None:
    with pytest.raises(ValueError):
        _place([sh.TargetFlight(bad, 1.0, 0.0)])


def test_non_finite_and_mismatched_inputs_raise() -> None:
    nan_path = _flight(100.0, 0.0).trajectory_m.copy()
    nan_path[3, 0] = np.nan
    with pytest.raises(ValueError):
        _place([sh.TargetFlight(nan_path, 100.0, 0.0)])
    with pytest.raises(ValueError):
        sh.shots_on_hole([_flight(100.0, 0.0)], _Hole(), ["a", "b"], [RED])
    with pytest.raises(ValueError):
        _place([_flight(100.0, 0.0)], _Hole(pin_position=(0.0, 0.0, 5.0)))


def test_add_shots_to_renderer_uses_add_trajectory() -> None:
    calls = []

    class _R:
        def add_trajectory(self, points, color=None):
            calls.append((points.shape, color))

    sh.add_shots_to_renderer(_R(), _place([_flight(100.0, 0.0)] * 2))
    assert calls == [((21, 3), RED)] * 2


def test_settings_load_with_sources_and_colours() -> None:
    settings = sh.load_shots_settings()
    assert settings["sources"] and settings["hole"]["length_m"] > 200.0
    assert all(len(c) == 4 for c in settings["shot_colours_rgba"])

