"""Drawing the linked trace panel (issue #8708, epic #8699).

Headless. What is checked is that the panel keeps the three properties the
issue turns on: a unit on every axis, a validity **band** rather than a
badge, and a cursor that is the shared one rather than a second time axis of
its own.
"""

from __future__ import annotations

import pytest
from matplotlib.figure import Figure

from src.tools.bunker_shot_gui.render_traces import (
    TracePanelArtists,
    draw_trace_panel,
    trace_panel_still,
)
from src.tools.bunker_shot_gui.traces import ShotTraces

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="session")
def nominal_traces(nominal_shot) -> ShotTraces:  # type: ignore[no-untyped-def]
    """The scalar traces of the nominal shot."""
    traces = nominal_shot.traces
    assert traces is not None, nominal_shot.unavailable
    return traces


def _texts(figure: Figure) -> str:
    """Every string drawn inside any axes of a figure."""
    return "\n".join(text.get_text() for axes in figure.axes for text in axes.texts)


class TestOnePanelPerGroup:
    """A group is a panel, and a panel has one unit."""

    def test_there_is_one_axes_per_non_empty_group(
        self, nominal_traces: ShotTraces
    ) -> None:
        figure = trace_panel_still(nominal_traces)
        assert len(figure.axes) == len(nominal_traces.groups())

    def test_every_trace_is_drawn(self, nominal_traces: ShotTraces) -> None:
        figure = trace_panel_still(nominal_traces)
        drawn = sum(len(axes.lines) for axes in figure.axes)
        # One line per trace, plus one cursor per panel.
        assert drawn == len(nominal_traces.traces) + len(figure.axes)

    def test_every_frame_can_be_scrubbed_to(self, nominal_traces: ShotTraces) -> None:
        figure = Figure(figsize=(6.0, 7.0))
        artists = draw_trace_panel(figure, nominal_traces, frame=0)
        for frame in range(nominal_traces.n_frames):
            artists.update(frame)

    def test_scrubbing_adds_no_artists(self, nominal_traces: ShotTraces) -> None:
        figure = Figure(figsize=(6.0, 7.0))
        artists = draw_trace_panel(figure, nominal_traces, frame=0)
        before = [
            (len(axes.lines), len(axes.texts), len(axes.patches))
            for axes in figure.axes
        ]
        for frame in range(nominal_traces.n_frames):
            artists.update(frame)
        after = [
            (len(axes.lines), len(axes.texts), len(axes.patches))
            for axes in figure.axes
        ]
        assert after == before

    def test_a_frame_outside_the_record_is_refused(
        self, nominal_traces: ShotTraces
    ) -> None:
        figure = Figure()
        artists = draw_trace_panel(figure, nominal_traces, frame=0)
        with pytest.raises(ValueError, match="outside the recorded shot"):
            artists.update(nominal_traces.n_frames)


class TestUnitsAreOnEveryAxis:
    """The demo report's standard, restated by #8708."""

    def test_every_panel_labels_its_y_axis_with_a_unit(
        self, nominal_traces: ShotTraces
    ) -> None:
        for axes in trace_panel_still(nominal_traces).axes:
            assert "[" in axes.get_ylabel()
            assert "]" in axes.get_ylabel()

    def test_the_shared_time_axis_is_labelled_in_milliseconds(
        self, nominal_traces: ShotTraces
    ) -> None:
        figure = trace_panel_still(nominal_traces)
        assert "[ms]" in figure.axes[-1].get_xlabel()

    def test_every_legend_entry_names_a_trace(self, nominal_traces: ShotTraces) -> None:
        figure = trace_panel_still(nominal_traces)
        labelled = {
            line.get_label()
            for axes in figure.axes
            for line in axes.lines
            if not str(line.get_label()).startswith("_")
        }
        assert set(nominal_traces.names) <= labelled


class TestTheValidityIsDrawnAsABand:
    """The requirement #8708 exists for."""

    def test_every_panel_carries_one_shaded_span_per_regime(
        self, nominal_traces: ShotTraces
    ) -> None:
        figure = trace_panel_still(nominal_traces)
        expected = len(nominal_traces.band.spans())
        for axes in figure.axes:
            assert len(axes.patches) == expected

    def test_the_band_spans_are_drawn_behind_the_traces(
        self, nominal_traces: ShotTraces
    ) -> None:
        figure = trace_panel_still(nominal_traces)
        for axes in figure.axes:
            lowest_line = min(line.get_zorder() for line in axes.lines)
            for patch in axes.patches:
                assert patch.get_zorder() < lowest_line

    def test_the_band_is_explained_in_the_frame(
        self, nominal_traces: ShotTraces
    ) -> None:
        """A shaded stripe nobody can decode is worse than no stripe."""
        drawn = _texts(trace_panel_still(nominal_traces)).lower()
        assert "validity" in drawn or "envelope" in drawn

    def test_a_shot_that_changes_regime_shows_more_than_one_span(
        self, nominal_traces: ShotTraces
    ) -> None:
        figure = trace_panel_still(nominal_traces)
        if nominal_traces.band.changes:
            assert len(figure.axes[0].patches) > 1


class TestTheCursorIsTheSharedOne:
    """One time cursor across the panels and the 3-D view."""

    def test_the_cursor_moves_with_the_frame(self, nominal_traces: ShotTraces) -> None:
        figure = Figure(figsize=(6.0, 7.0))
        artists = draw_trace_panel(figure, nominal_traces, frame=0)
        artists.update(0)
        opening = [axes.lines[-1].get_xdata()[0] for axes in figure.axes]
        artists.update(nominal_traces.n_frames - 1)
        closing = [axes.lines[-1].get_xdata()[0] for axes in figure.axes]
        assert opening != closing

    def test_every_panel_shares_one_cursor_position(
        self, nominal_traces: ShotTraces
    ) -> None:
        figure = Figure(figsize=(6.0, 7.0))
        artists = draw_trace_panel(figure, nominal_traces, frame=5)
        artists.update(5)
        positions = {float(axes.lines[-1].get_xdata()[0]) for axes in figure.axes}
        assert len(positions) == 1

    def test_the_cursor_sits_at_the_frames_own_moment(
        self, nominal_traces: ShotTraces
    ) -> None:
        figure = Figure(figsize=(6.0, 7.0))
        artists = draw_trace_panel(figure, nominal_traces, frame=7)
        artists.update(7)
        assert float(figure.axes[0].lines[-1].get_xdata()[0]) == pytest.approx(
            float(nominal_traces.time_display[7])
        )

    def test_the_readout_states_the_verdict_at_that_moment(
        self, nominal_traces: ShotTraces
    ) -> None:
        figure = Figure(figsize=(6.0, 7.0))
        artists = draw_trace_panel(figure, nominal_traces, frame=0)
        artists.update(0)
        expected = nominal_traces.band.status_at(0).value.replace("_", " ").upper()
        assert expected in _texts(figure)


class TestNothingIsAutoScaledBetweenFrames:
    """The #8728 defect again: a fixed y-axis, or the eye reads noise."""

    def test_the_y_limits_do_not_move_while_scrubbing(
        self, nominal_traces: ShotTraces
    ) -> None:
        figure = Figure(figsize=(6.0, 7.0))
        artists = draw_trace_panel(figure, nominal_traces, frame=0)
        before = [axes.get_ylim() for axes in figure.axes]
        for frame in range(nominal_traces.n_frames):
            artists.update(frame)
        assert [axes.get_ylim() for axes in figure.axes] == before

    def test_the_x_limits_span_the_whole_record(
        self, nominal_traces: ShotTraces
    ) -> None:
        figure = trace_panel_still(nominal_traces)
        low, high = figure.axes[0].get_xlim()
        assert low <= float(nominal_traces.time_display[0])
        assert high >= float(nominal_traces.time_display[-1])


class TestThePanelIsBuiltOnceAndMutated:
    """The transport interval is the renderer's cost, not the arithmetic's."""

    def test_the_artists_report_how_many_panels_they_own(
        self, nominal_traces: ShotTraces
    ) -> None:
        figure = Figure(figsize=(6.0, 7.0))
        artists = draw_trace_panel(figure, nominal_traces, frame=0)
        assert isinstance(artists, TracePanelArtists)
        assert artists.n_panels == len(nominal_traces.groups())
