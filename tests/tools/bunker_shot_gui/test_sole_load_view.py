"""The animated sole-load view in the Qt workbench (issues #8705, #8707).

The computation and the drawing are covered headlessly in
``tests/unit/tools/bunker_shot_gui``. What is pinned here is the part that
only exists once there is a window: the transport (play, pause, scrub), the
wiring from a run to the view, and the two properties that must survive
contact with a GUI --

* a refusal clears the view and stops playback, so no sole is ever painted
  beside a REFUSED verdict;
* an A/B comparison puts both designs on **one** colour scale, because two
  grinds auto-scaled to their own maxima look identical.

Qt is imported through ``pytest.importorskip`` for the same reason as
``test_gui``: PyQt6 fails to load on some development machines.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.bunker_shot_gui.design import SolverSetup  # noqa: E402
from src.tools.bunker_shot_gui.field import LoadComponent  # noqa: E402
from src.tools.bunker_shot_gui.gui import BunkerShotWidget  # noqa: E402
from src.tools.bunker_shot_gui.model import WorkbenchModel  # noqa: E402
from src.tools.bunker_shot_gui.widgets import (  # noqa: E402
    GridMapWidget,
    SoleLoadFieldWidget,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

COARSE = SolverSetup(
    n_profile_points=12, n_stations=5, playability_points=2, target_carry_m=12.0
)


def _coarse_model(_settings: SolverSetup) -> WorkbenchModel:
    """Factory that ignores the panel's resolution and runs the cheap one."""
    return WorkbenchModel(COARSE)


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    """One offscreen QApplication for the module."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv[:1])
    return application


@pytest.fixture(scope="module")
def shot():  # type: ignore[no-untyped-def]
    """One real F0 shot, run once for the module."""
    from src.tools.bunker_shot_gui.design import SandCondition, SwingSetup, WedgeDesign

    return WorkbenchModel(COARSE).run_shot(
        WedgeDesign(name="view").geometry(), SandCondition().sand_state(), SwingSetup()
    )


@pytest.fixture()
def view(shot) -> SoleLoadFieldWidget:  # type: ignore[no-untyped-def]
    """A field view holding that shot."""
    widget = SoleLoadFieldWidget("A: sole load")
    widget.set_shot(shot.sole_field, shot.contact_patch)
    return widget


@pytest.fixture()
def widget() -> BunkerShotWidget:
    """A workbench wired to a cheap but genuine model."""
    return BunkerShotWidget(model_factory=_coarse_model)


class TestTheEmptyView:
    def test_a_new_view_holds_no_shot(self) -> None:
        empty = SoleLoadFieldWidget("A")
        assert empty.has_shot is False
        assert empty.n_frames == 0

    def test_a_new_view_is_not_playing(self) -> None:
        assert SoleLoadFieldWidget("A").is_playing is False

    def test_an_empty_view_cannot_be_scrubbed(self) -> None:
        with pytest.raises(ValueError, match="frame"):
            SoleLoadFieldWidget("A").set_frame(0)

    def test_the_title_is_kept(self) -> None:
        assert SoleLoadFieldWidget("A: sole load").title == "A: sole load"


class TestTheTransport:
    def test_the_view_spans_the_whole_shot(
        self, view: SoleLoadFieldWidget, shot
    ) -> None:  # type: ignore[no-untyped-def]
        assert view.n_frames == shot.sole_field.n_frames
        assert view._slider.maximum() == view.n_frames - 1

    def test_it_opens_on_the_moment_the_sole_carried_most(
        self, view: SoleLoadFieldWidget, shot
    ) -> None:  # type: ignore[no-untyped-def]
        peak = int(shot.sole_field.resultant_force_N(LoadComponent.TOTAL).argmax())
        assert view.frame_index == peak

    def test_scrubbing_moves_the_frame(self, view: SoleLoadFieldWidget) -> None:
        view.set_frame(2)
        assert view.frame_index == 2

    def test_the_slider_drives_the_frame(self, view: SoleLoadFieldWidget) -> None:
        view._slider.setValue(3)
        assert view.frame_index == 3

    def test_advancing_wraps_at_the_end(self, view: SoleLoadFieldWidget) -> None:
        view.set_frame(view.n_frames - 1)
        view.advance()
        assert view.frame_index == 0

    def test_a_frame_outside_the_shot_is_refused(
        self, view: SoleLoadFieldWidget
    ) -> None:
        with pytest.raises(ValueError, match="frame"):
            view.set_frame(view.n_frames)

    def test_playing_and_pausing_run_the_timer(self, view: SoleLoadFieldWidget) -> None:
        view.play()
        assert view.is_playing is True
        assert view._timer.isActive() is True
        view.pause()
        assert view.is_playing is False
        assert view._timer.isActive() is False

    def test_the_play_button_toggles(self, view: SoleLoadFieldWidget) -> None:
        view._play_button.click()
        assert view.is_playing is True
        view._play_button.click()
        assert view.is_playing is False


class TestTheViewClearsRatherThanGoingStale:
    def test_clearing_drops_the_shot(self, view: SoleLoadFieldWidget) -> None:
        view.clear()
        assert view.has_shot is False
        assert view.n_frames == 0

    def test_clearing_stops_playback(self, view: SoleLoadFieldWidget) -> None:
        view.play()
        view.clear()
        assert view.is_playing is False
        assert view._timer.isActive() is False


class TestTheScaleIsInjectable:
    def test_the_view_reports_the_scales_it_is_drawing_on(
        self, view: SoleLoadFieldWidget
    ) -> None:
        assert view.scales is not None
        assert set(view.scales) == set(LoadComponent)

    def test_supplied_scales_are_used_verbatim(self, shot) -> None:  # type: ignore[no-untyped-def]
        from src.tools.bunker_shot_gui.render import field_scales

        shared = field_scales((shot.sole_field, shot.sole_field))
        widget = SoleLoadFieldWidget("A")
        widget.set_shot(shot.sole_field, shot.contact_patch, scales=shared)
        assert widget.scales == shared


class TestTheGridMapCanBePinned:
    """The pre-existing map auto-scaled every grid to its own extremes."""

    def test_a_pinned_grid_keeps_the_limits_it_was_given(self) -> None:
        widget = GridMapWidget("m")
        widget.set_grid(np.array([[0.0, 1.0]]), limits=(0.0, 10.0))
        assert widget.limits == (0.0, 10.0)

    def test_an_unpinned_grid_still_scales_itself(self) -> None:
        widget = GridMapWidget("m")
        widget.set_grid(np.array([[0.0, 1.0]]))
        assert widget.limits is None

    def test_an_inverted_limit_pair_is_refused(self) -> None:
        with pytest.raises(ValueError, match="limits"):
            GridMapWidget("m").set_grid(np.zeros((2, 2)), limits=(5.0, 1.0))

    def test_clearing_drops_the_limits(self) -> None:
        widget = GridMapWidget("m")
        widget.set_grid(np.zeros((2, 2)), limits=(0.0, 1.0))
        widget.clear()
        assert widget.limits is None


class TestTheWorkbenchWiring:
    def test_running_a_design_fills_the_field_view(
        self, widget: BunkerShotWidget
    ) -> None:
        widget.run_design_a()
        assert widget._field_a.has_shot is True
        assert widget._field_a.n_frames > 2

    def test_the_second_field_view_stays_empty_for_a_single_run(
        self, widget: BunkerShotWidget
    ) -> None:
        widget.run_design_a()
        assert widget._field_b.has_shot is False

    def test_a_refusal_clears_the_field_view(self, widget: BunkerShotWidget) -> None:
        widget.run_design_a()
        widget._field_a.play()
        widget._conditions._dynamic.setChecked(False)
        widget.run_design_a()
        assert widget._field_a.has_shot is False
        assert widget._field_a.is_playing is False

    def test_an_input_error_clears_the_field_view(
        self, widget: BunkerShotWidget
    ) -> None:
        widget.run_design_a()
        widget._design_a._heel_relief.setValue(0.6)
        widget.run_design_a()
        assert widget._field_a.has_shot is False

    def test_a_comparison_fills_both_field_views(
        self, widget: BunkerShotWidget
    ) -> None:
        widget.run_comparison()
        assert widget._field_a.has_shot is True
        assert widget._field_b.has_shot is True

    def test_a_comparison_puts_both_designs_on_one_scale(
        self, widget: BunkerShotWidget
    ) -> None:
        widget.run_comparison()
        assert widget._field_a.scales == widget._field_b.scales

    def test_a_comparison_pins_both_bounce_maps_to_one_ramp(
        self, widget: BunkerShotWidget
    ) -> None:
        """Two grinds each scaled to their own peak cannot be compared."""
        widget.run_comparison()
        assert widget._bounce_map_a.limits is not None
        assert widget._bounce_map_a.limits == widget._bounce_map_b.limits

    def test_the_renderer_in_use_is_stated(self, widget: BunkerShotWidget) -> None:
        widget.run_design_a()
        assert "matplotlib" in widget._field_a.renderer_note.lower()
