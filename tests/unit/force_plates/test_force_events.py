"""Force-event namespace and impact anchor (OF-04)."""

from __future__ import annotations

import json

import pytest
from matplotlib.figure import Figure

from src.shared.python.biomechanics.ground_reaction_plot_model import (
    plot_series_to_json,
    series_from_payload,
)
from src.shared.python.force_plates import (
    FORCE_EVENT_PREFIX,
    PlateTimeAlignment,
    VerticalPlateTrial,
    force_event_markers,
    impact_anchor,
    merge_event_markers,
)
from src.shared.python.plotting.renderers.time_panels import draw_event_markers
from tests.unit.biomechanics.test_ground_reaction_plot_renderer import _plot
from tests.unit.force_plates._trial_factory import trial_kwargs

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _trial(**events: float | None) -> VerticalPlateTrial:
    return VerticalPlateTrial(**trial_kwargs(events_s=events))


def _alignment(offset_s: float = 1.0) -> PlateTimeAlignment:
    return PlateTimeAlignment(
        offset_s=offset_s, sigma_offset_s=0.002, kind="manual", note="hand aligned"
    )


def test_events_are_prefixed() -> None:
    assert force_event_markers(_trial(max_trail=0.005)) == {"force.max_trail": 0.005}
    assert FORCE_EVENT_PREFIX == "force."


def test_events_without_a_time_are_omitted_not_zero() -> None:
    markers = force_event_markers(_trial(max_trail=0.005, impact=None))
    assert "force.impact" not in markers


def test_alignment_shifts_to_the_model_clock() -> None:
    alignment = _alignment(1.0)
    markers = force_event_markers(_trial(max_trail=0.005), alignment=alignment)
    assert markers["force.max_trail"] == pytest.approx(1.005)
    assert alignment.trial_time(markers["force.max_trail"]) == pytest.approx(0.005)


def test_markers_reject_wrong_types() -> None:
    with pytest.raises(TypeError):
        force_event_markers("trial")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        force_event_markers(_trial(), alignment=1.0)  # type: ignore[arg-type]


def test_merge_returns_a_new_dict_with_both_sets() -> None:
    model = {"address": 0.0, "top": 0.4}
    merged = merge_event_markers(model, {"force.max_trail": 0.005})
    assert merged == {"address": 0.0, "top": 0.4, "force.max_trail": 0.005}
    assert merged is not model
    assert model == {"address": 0.0, "top": 0.4}


def test_merge_refuses_unprefixed_force_key() -> None:
    with pytest.raises(ValueError, match="max_trail"):
        merge_event_markers({}, {"max_trail": 0.1})


def test_merge_refuses_prefixed_model_key() -> None:
    with pytest.raises(ValueError, match=r"force\.top"):
        merge_event_markers({"force.top": 0.1}, {})


def test_model_and_measured_impact_never_overwrite_each_other() -> None:
    merged = merge_event_markers({"impact": 0.30}, {"force.impact": 0.31})
    assert merged == {"impact": 0.30, "force.impact": 0.31}


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_merge_refuses_non_finite_times(bad: float) -> None:
    with pytest.raises(ValueError, match=r"force\.max_trail"):
        merge_event_markers({}, {"force.max_trail": bad})
    with pytest.raises(ValueError, match="top"):
        merge_event_markers({"top": bad}, {})


@pytest.mark.parametrize("bad", ["0.1", True, None])
def test_merge_refuses_non_numeric_times(bad: object) -> None:
    with pytest.raises(TypeError, match=r"force\.max_trail"):
        merge_event_markers({}, {"force.max_trail": bad})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="top"):
        merge_event_markers({"top": bad}, {})  # type: ignore[dict-item]


def test_impact_anchor_none_without_impact_and_equal_with_it() -> None:
    assert impact_anchor(_trial(max_trail=0.005)) is None
    assert impact_anchor(_trial(max_trail=0.005, impact=None)) is None
    assert impact_anchor(_trial(max_trail=0.005, impact=0.008)) == 0.008


def test_merged_events_survive_the_plot_model_and_json_round_trip() -> None:
    merged = merge_event_markers(
        {"top": 0.1}, force_event_markers(_trial(max_trail=0.005))
    )
    plot = _plot(events=merged)
    assert plot.events == merged
    again = series_from_payload(json.loads(plot_series_to_json(plot)))
    assert again.events == merged
    assert "force.max_trail" in again.events
    assert "max_trail" not in again.events


def test_python_renderer_label_keeps_the_force_namespace() -> None:
    ax = Figure().subplots()
    draw_event_markers(ax, {"force.max_trail": 0.1, "top": 0.2})
    assert [t.get_text() for t in ax.texts] == ["force.max_trail", "top"]


def test_force_events_are_never_mapped_to_swing_positions() -> None:
    markers = force_event_markers(_trial(max_trail=0.005))
    assert not {"address", "top", "impact", "finish"} & set(markers)
