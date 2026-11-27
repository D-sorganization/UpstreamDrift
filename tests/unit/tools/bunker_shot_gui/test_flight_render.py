"""Ball-flight and green-rollout drawing (issue #15497, slice 2 desktop)."""

from __future__ import annotations

import math

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.tools.bunker_shot_gui.flight import BunkerFlight
from src.tools.bunker_shot_gui.render_flight import (
    FlightUnits,
    flight_figure,
    flight_readout,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def flight() -> BunkerFlight:
    arc = tuple(
        (float(x), 0.1 * x, 4.0 * math.sin(math.pi * x / 10.0)) for x in range(11)
    )
    roll = tuple((10.0 + 0.5 * i, 1.0, 0.0) for i in range(5))
    return BunkerFlight(arc, roll, carry_m=math.hypot(10.0, 1.0))


def _texts(figure: Figure) -> str:
    return " ".join(t.get_text() for ax in figure.axes for t in ax.texts)


def test_draws_side_and_overhead_panels(flight: BunkerFlight) -> None:
    figure = flight_figure(flight)
    assert len(figure.axes) == 2
    side, top = figure.axes
    assert any(len(line.get_xdata()) == len(flight.flight_m) for line in side.lines)
    assert any(len(line.get_xdata()) == len(flight.rollout_m) for line in top.lines)


def test_marks_landing_and_rest_points(flight: BunkerFlight) -> None:
    top = flight_figure(flight).axes[1]
    landing = flight.flight_m[-1]
    rest = flight.rollout_m[-1]
    points = {
        (round(float(x), 6), round(float(y), 6))
        for line in top.lines
        if len(line.get_xdata()) == 1
        for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True)
    }
    assert (round(landing[0], 6), round(landing[1], 6)) in points
    assert (round(rest[0], 6), round(rest[1], 6)) in points


def test_yards_scale_the_data_and_label_the_axes(flight: BunkerFlight) -> None:
    figure = flight_figure(flight, units=FlightUnits.YARDS)
    side = figure.axes[0]
    assert "yd" in side.get_xlabel()
    line = max(side.lines, key=lambda ln: len(ln.get_xdata()))
    assert max(line.get_xdata()) == pytest.approx(10.0 / 0.9144)
    assert "m]" in flight_figure(flight).axes[0].get_xlabel()


def test_assumption_is_stamped_inside_the_axes(flight: BunkerFlight) -> None:
    assert flight.assumption in _texts(flight_figure(flight))


def test_refused_or_missing_flight_draws_banner_and_no_trajectory() -> None:
    figure = flight_figure(None, reason="refused: outside 3D-RFT envelope")
    assert "refused: outside 3D-RFT envelope" in _texts(figure)
    assert all(len(ax.lines) == 0 for ax in figure.axes)


def test_readout_states_carry_with_units_and_assumption(flight: BunkerFlight) -> None:
    text = flight_readout(flight, FlightUnits.METRES)
    assert f"{flight.carry_m:.1f} m" in text
    assert flight.assumption in text
    assert "no flight" in flight_readout(None, FlightUnits.METRES, reason="refused")


def test_narrow_canvas_keeps_labels_on_screen_and_banner_off_legend(
    flight: BunkerFlight,
) -> None:
    figure = flight_figure(flight, figure=Figure(figsize=(3.9, 6.0), dpi=100))
    renderer = FigureCanvasAgg(figure).get_renderer()
    figure.draw(renderer)
    for axes in figure.axes:
        assert axes.yaxis.label.get_window_extent(renderer).x0 >= 0
        for text in axes.texts:
            legend = figure.legends[0].get_window_extent(renderer)
            assert not text.get_window_extent(renderer).overlaps(legend)
