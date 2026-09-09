"""Frame and renderer boundary contracts for force coloring."""

from dataclasses import replace

import pytest

from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.body_part_viz.force_display import (
    ForceColorDisplay,
    SegmentLoadSeries,
)

pytestmark = pytest.mark.unit


class Renderer:
    def __init__(self):
        self.colors = {}

    def set_color(self, handle, color):
        self.colors[handle] = color


def test_seek_toggle_missing_and_model_independence():
    renderer = Renderer()
    display = ForceColorDisplay(renderer, {"arbitrary-link": "handle"})
    loads = SegmentLoadSeries(
        (0.0, 1.0, 2.0),
        {"arbitrary-link": (1000.0, -1000.0, None)},
        "analytical fixture",
    )
    display.set_loads(loads)
    display.configure(ForceColorScale(enabled=True))
    assert renderer.colors == {"handle": "#0000ff"}
    display.update_frame(1)
    assert renderer.colors["handle"] == "#ff0000"
    display.update_frame(2)
    assert renderer.colors["handle"] is None
    display.update_frame(0)
    display.configure(ForceColorScale())
    assert renderer.colors["handle"] is None
    display.configure(ForceColorScale(enabled=True))
    display.set_loads(None)
    assert renderer.colors["handle"] is None


def test_source_copies_inputs_and_rejects_unaligned_or_ambiguous_data():
    values = {"segment": [1.0, 2.0]}
    series = SegmentLoadSeries([0.0, 1.0], values, "fixture")
    values["segment"][0] = 999
    assert series.values_n["segment"][0] == 1
    for kwargs in [
        {"time_s": (1.0, 0.0)},
        {"values_n": {"segment": (1.0,)}},
        {"source": ""},
        {"units": "lbf"},
        {"sign_convention": "compression-positive"},
        {"time_s": (0.0, float("nan"))},
    ]:
        with pytest.raises(ValueError):
            replace(series, **kwargs)


def test_invalid_seek_is_rejected_without_color_mutation():
    renderer = Renderer()
    display = ForceColorDisplay(renderer, {"s": "h"})
    display.set_loads(SegmentLoadSeries((0.0,), {"s": (1000.0,)}, "fixture"))
    display.configure(ForceColorScale(enabled=True))
    with pytest.raises(IndexError):
        display.update_frame(1)
    assert renderer.colors["h"] == "#0000ff"
    with pytest.raises(TypeError):
        display.update_frame(True)


def test_unknown_segment_restores_and_nonfinite_is_unavailable():
    renderer = Renderer()
    display = ForceColorDisplay(renderer, {"s": "h", "other": "h2"})
    display.set_loads(SegmentLoadSeries((0.0,), {"s": (float("nan"),)}, "fixture"))
    display.configure(ForceColorScale(enabled=True))
    assert renderer.colors == {"h": None, "h2": None}


def test_display_rejects_duplicate_handles_and_wrong_inputs():
    with pytest.raises(ValueError):
        ForceColorDisplay(Renderer(), {"a": "h", "b": "h"})
    with pytest.raises(TypeError):
        ForceColorDisplay(object(), {})
    display = ForceColorDisplay(Renderer(), {})
    with pytest.raises(TypeError):
        display.configure({})
    with pytest.raises(TypeError):
        display.set_loads({})
