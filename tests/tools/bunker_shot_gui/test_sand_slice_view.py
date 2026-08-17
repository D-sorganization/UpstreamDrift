"""The sand cut as a Qt view, scrubbed by the shared transport (#8711).

The point of this module is that the cut does **not** grow a slider. It is
handed a frame index by :class:`SoleLoadFieldWidget` like the 3-D scene and
the trace panel are, and -- unlike them -- its own record is not the shot's,
so it maps the index and says so.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.bunker_shot_gui.slices import (  # noqa: E402
    CursorMap,
    PlanePreset,
    slice_scale,
    swing_plane,
)
from src.tools.bunker_shot_gui.viewport_widgets import SandSliceWidget  # noqa: E402
from tests.unit.tools.bunker_shot_gui.test_slices import analytic_field  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    """One application for the session, offscreen."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv[:1])
    return application


@pytest.fixture
def loaded() -> SandSliceWidget:
    """A slice view holding a four-frame field on a 53-sample transport."""
    view = SandSliceWidget("Sand cut")
    view.set_shot(analytic_field(n_frames=4), cursor=CursorMap(53, 4))
    return view


class TestItFollowsWithoutASliderOfItsOwn:
    """One transport, and this view is not it."""

    def test_it_implements_the_follower_protocol(self) -> None:
        """``FollowsFrame`` is structural, so the check is on the shape."""
        import inspect

        from src.tools.bunker_shot_gui.widgets import FollowsFrame

        follower: FollowsFrame = SandSliceWidget("Sand cut")
        assert list(inspect.signature(follower.set_frame).parameters) == ["frame"]

    def test_it_owns_no_transport_controls(self) -> None:
        """A second slider is exactly what linking the views prevents."""
        from PyQt6.QtWidgets import QPushButton, QSlider

        view = SandSliceWidget("Sand cut")
        assert not view.findChildren(QSlider)
        assert not view.findChildren(QPushButton)

    def test_a_transport_frame_maps_onto_a_field_frame(
        self, loaded: SandSliceWidget
    ) -> None:
        loaded.set_frame(0)
        assert loaded.frame_index == 0
        loaded.set_frame(52)
        assert loaded.frame_index == 3

    def test_the_field_frame_count_is_not_the_shots(
        self, loaded: SandSliceWidget
    ) -> None:
        assert loaded.n_frames == 4
        assert loaded.cursor_map is not None
        assert loaded.cursor_map.n_transport == 53

    def test_a_frame_outside_the_transport_is_refused_not_clamped(
        self, loaded: SandSliceWidget
    ) -> None:
        with pytest.raises(ValueError, match="outside the shot"):
            loaded.set_frame(53)

    def test_an_empty_view_ignores_a_frame(self) -> None:
        """Views are cleared independently; a tick can reach an empty one."""
        SandSliceWidget("Sand cut").set_frame(7)

    def test_a_linked_field_view_drives_it(self) -> None:
        """The wiring the workbench uses, end to end."""
        from src.tools.bunker_shot_gui.widgets import SoleLoadFieldWidget

        transport = SoleLoadFieldWidget("Sole load")
        view = SandSliceWidget("Sand cut")
        transport.link(view)
        view.set_shot(analytic_field(n_frames=4), cursor=CursorMap(9, 4))
        transport.frame_changed.emit(8)
        assert view.frame_index == 3


class TestTheCutIsTheOnlyControlItOwns:
    """Where you cut is not when you cut."""

    def test_the_presets_are_offered_by_name(self, loaded: SandSliceWidget) -> None:
        presets = [plane.preset for plane in loaded.planes]
        assert presets[0] is PlanePreset.SWING_PLANE
        assert presets[1] is PlanePreset.FACE_NORMAL
        assert PlanePreset.HEEL_TO_TOE in presets

    def test_changing_the_cut_leaves_the_cursor_alone(
        self, loaded: SandSliceWidget
    ) -> None:
        loaded.set_frame(52)
        before = loaded.frame_index
        loaded._plane_box.setCurrentIndex(1)  # noqa: SLF001 - the control under test
        assert loaded.frame_index == before

    def test_moving_the_cursor_leaves_the_cut_alone(
        self, loaded: SandSliceWidget
    ) -> None:
        loaded._plane_box.setCurrentIndex(2)  # noqa: SLF001 - the control under test
        chosen = loaded.plane
        loaded.set_frame(26)
        assert loaded.plane is chosen

    def test_an_arbitrary_plane_can_be_supplied(self) -> None:
        view = SandSliceWidget("Sand cut")
        view.set_shot(analytic_field(), planes=(swing_plane(offset_m=0.004),))
        assert view.plane is not None
        assert view.plane.offset_m == pytest.approx(0.004)

    def test_an_empty_plane_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one plane"):
            SandSliceWidget("Sand cut").set_shot(analytic_field(), planes=())


class TestWorkbenchWiring:
    """The tab, the loader, and the one thing that must not go stale."""

    @pytest.fixture
    def widget(self):  # type: ignore[no-untyped-def]
        """The workbench, with the cheap real model behind it."""
        from src.tools.bunker_shot_gui.design import SolverSetup
        from src.tools.bunker_shot_gui.gui import BunkerShotWidget
        from src.tools.bunker_shot_gui.model import WorkbenchModel

        coarse = SolverSetup(
            n_profile_points=12,
            n_stations=5,
            playability_points=2,
            target_carry_m=12.0,
        )
        return BunkerShotWidget(model_factory=lambda _settings: WorkbenchModel(coarse))

    @pytest.fixture
    def stored_field(self, tmp_path):  # type: ignore[no-untyped-def]
        """A saved sand field on disk, for the loader to open."""
        from bunkershot3d.fields.store import save_field

        return save_field(tmp_path / "field.h5", analytic_field(n_frames=4))

    def test_the_workbench_offers_a_sand_cut_tab(self, widget) -> None:  # type: ignore[no-untyped-def]
        titles = [
            widget._views.tabText(index)  # noqa: SLF001 - the tab bar under test
            for index in range(widget._views.count())  # noqa: SLF001
        ]
        assert any("Sand cut" in title for title in titles)

    def test_the_tab_starts_empty_and_says_why(self, widget) -> None:  # type: ignore[no-untyped-def]
        assert not widget._slice_a.has_shot  # noqa: SLF001
        note = widget._field_note.text()  # noqa: SLF001
        assert "computed offline" in note
        assert "digest" in note

    def test_loading_a_field_fills_the_cut(self, widget, stored_field) -> None:  # type: ignore[no-untyped-def]
        widget.load_sand_field(str(stored_field))
        assert widget._slice_a.has_shot  # noqa: SLF001
        assert widget._slice_a.n_frames == 4  # noqa: SLF001

    def test_the_note_restates_the_tier_and_the_kinematics(
        self, widget, stored_field
    ) -> None:  # type: ignore[no-untyped-def]
        widget.load_sand_field(str(stored_field))
        note = widget._field_note.text()  # noqa: SLF001
        assert "F1" in note
        assert "BEYOND VALIDATION" in note
        assert "kinematics:" in note

    def test_a_new_run_clears_a_field_from_a_different_march(
        self, widget, stored_field
    ) -> None:
        """A stale cut beside a fresh design would read as that design's sand."""
        widget.load_sand_field(str(stored_field))
        assert widget._slice_a.has_shot  # noqa: SLF001
        widget.run_design_a()
        assert not widget._slice_a.has_shot  # noqa: SLF001
        assert "different run" in widget._field_note.text()  # noqa: SLF001

    def test_the_cut_is_linked_to_the_shared_transport(
        self, widget, stored_field
    ) -> None:  # type: ignore[no-untyped-def]
        widget.load_sand_field(str(stored_field))
        cursor = widget._slice_a.cursor_map  # noqa: SLF001
        assert cursor is not None
        widget._field_a.frame_changed.emit(cursor.n_transport - 1)  # noqa: SLF001
        assert widget._slice_a.frame_index == 3  # noqa: SLF001

    def test_a_corrupted_field_is_reported_not_drawn(
        self, widget, stored_field
    ) -> None:  # type: ignore[no-untyped-def]
        """The digest refusal has to reach the user, not a traceback."""
        import h5py

        with h5py.File(stored_field, "r+") as handle:
            handle["sand_field/velocity"][0, 0, 0] += 1.0
        widget.load_sand_field(str(stored_field))
        assert not widget._slice_a.has_shot  # noqa: SLF001
        assert "Could not load" in widget._field_note.text()  # noqa: SLF001
        assert "digest" in widget._field_note.text()  # noqa: SLF001


class TestScalesAndOpeningFrame:
    """Injected limits, and the moment worth opening on."""

    def test_it_opens_on_the_fastest_reportable_sand(self) -> None:
        field = analytic_field(n_frames=4)
        view = SandSliceWidget("Sand cut")
        view.set_shot(field)
        speeds = field.occupied_speed_m_s()
        assert view.frame_index == int(np.nanmax(speeds, axis=1).argmax())

    def test_a_shared_scale_is_used_rather_than_the_fields_own(self) -> None:
        quiet = analytic_field(peak_m_s=5.0)
        loud = analytic_field(peak_m_s=25.0)
        shared = slice_scale([quiet, loud])
        view = SandSliceWidget("Sand cut")
        view.set_shot(quiet, scale=shared)
        assert view.scale == shared
        assert view.scale != slice_scale([quiet])

    def test_it_defaults_to_its_own_coverage_when_alone(self) -> None:
        field = analytic_field()
        view = SandSliceWidget("Sand cut")
        view.set_shot(field)
        assert view.scale == slice_scale([field])

    def test_clearing_drops_everything(self, loaded: SandSliceWidget) -> None:
        loaded.clear()
        assert not loaded.has_shot
        assert loaded.n_frames == 0
        assert loaded.scale is None
        assert loaded.planes == ()
        assert loaded.plane is None
