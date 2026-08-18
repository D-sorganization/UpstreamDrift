"""One cursor across the three views (issues #8706, #8708, epic #8699).

The computation and the drawing are covered headlessly in
``tests/unit/tools/bunker_shot_gui``. What is pinned here is the part that
only exists once there is a window: that the sole load field, the 3-D scene
and the trace panel move **together**, driven by the one transport the field
view already owned.

That is the whole value of #8708. A force peak at 6.2 ms is only traceable
to a location on the sole if the sole view is showing 6.2 ms too, and three
independent sliders is precisely how that stops being true.

Qt is imported through ``pytest.importorskip`` for the same reason as
``test_gui``: PyQt6 fails to load on some development machines.
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.bunker_shot_gui.design import (  # noqa: E402
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
)
from src.tools.bunker_shot_gui.model import DesignEvaluation, WorkbenchModel  # noqa: E402
from src.tools.bunker_shot_gui.shot3d import CameraPreset  # noqa: E402
from src.tools.bunker_shot_gui.viewport_widgets import (  # noqa: E402
    ShotViewportWidget,
    TracePanelWidget,
)
from src.tools.bunker_shot_gui.widgets import SoleLoadFieldWidget  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

COARSE = SolverSetup(
    n_profile_points=12, n_stations=5, playability_points=2, target_carry_m=12.0
)


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    """One offscreen QApplication for the module."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv[:1])
    return application


@pytest.fixture(scope="module")
def evaluation() -> DesignEvaluation:
    """One real evaluated design on the coarse settings."""
    return WorkbenchModel(COARSE).evaluate(
        WedgeDesign(name="nominal"), SandCondition(), SwingSetup()
    )


@pytest.fixture()
def linked(
    evaluation: DesignEvaluation,
) -> tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]:
    """The three views of one shot, wired to the field view's transport."""
    field_view = SoleLoadFieldWidget("field")
    scene_view = ShotViewportWidget("scene")
    trace_view = TracePanelWidget("traces")
    field_view.link(scene_view)
    field_view.link(trace_view)
    shot = evaluation.shot
    assert shot.sole_field is not None, shot.unavailable
    assert shot.scene is not None, shot.unavailable
    assert shot.traces is not None, shot.unavailable
    scene_view.set_shot(shot.scene, band=shot.traces.band)
    trace_view.set_shot(shot.traces)
    field_view.set_shot(shot.sole_field, shot.contact_patch)
    return field_view, scene_view, trace_view


class TestThereIsOnlyOneTransport:
    """#8708 asks for lockstep, and a second slider is how lockstep dies."""

    def test_the_scene_view_has_no_transport_of_its_own(self) -> None:
        view = ShotViewportWidget("scene")
        assert not hasattr(view, "play")
        assert not hasattr(view, "toggle_play")

    def test_the_trace_panel_has_no_transport_of_its_own(self) -> None:
        view = TracePanelWidget("traces")
        assert not hasattr(view, "play")
        assert not hasattr(view, "toggle_play")

    def test_the_field_view_still_owns_the_transport(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        field_view, _, _ = linked
        assert hasattr(field_view, "toggle_play")
        assert field_view.n_frames > 2


class TestTheCursorIsShared:
    """The property the linked views exist for."""

    def test_scrubbing_the_field_moves_the_scene_and_the_traces(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        field_view, scene_view, trace_view = linked
        field_view.set_frame(4)
        assert scene_view.frame_index == 4
        assert trace_view.frame_index == 4

    def test_the_three_views_agree_at_every_sample(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        field_view, scene_view, trace_view = linked
        for frame in range(field_view.n_frames):
            field_view.set_frame(frame)
            assert scene_view.frame_index == frame
            assert trace_view.frame_index == frame

    def test_the_transport_timer_moves_all_three(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        field_view, scene_view, trace_view = linked
        field_view.set_frame(0)
        field_view.advance()
        assert (scene_view.frame_index, trace_view.frame_index) == (1, 1)

    def test_the_three_views_span_the_same_record(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        field_view, scene_view, trace_view = linked
        assert field_view.n_frames == scene_view.n_frames == trace_view.n_frames


class TestAFollowerRefusesAMomentItCannotShow:
    """A clamped index would leave a view describing a different moment."""

    def test_the_scene_refuses_a_frame_outside_the_shot(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        _, scene_view, _ = linked
        with pytest.raises(ValueError, match="outside the shot"):
            scene_view.set_frame(scene_view.n_frames)

    def test_the_trace_panel_refuses_a_frame_outside_the_shot(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        _, _, trace_view = linked
        with pytest.raises(ValueError, match="outside the shot"):
            trace_view.set_frame(trace_view.n_frames)

    def test_an_empty_follower_ignores_a_frame_rather_than_raising(self) -> None:
        """Views are cleared independently, so a tick can reach an empty one."""
        ShotViewportWidget("scene").set_frame(3)
        TracePanelWidget("traces").set_frame(3)


class TestTheViewsClearRatherThanGoingStale:
    """A refusal must not leave a head animating under a red banner."""

    def test_clearing_the_scene_drops_the_shot(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        _, scene_view, _ = linked
        scene_view.clear()
        assert not scene_view.has_shot
        assert scene_view.n_frames == 0

    def test_clearing_the_traces_drops_the_shot(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        _, _, trace_view = linked
        trace_view.clear()
        assert not trace_view.has_shot
        assert trace_view.n_panels == 0


class TestTheCameraIsIndependentOfTheCursor:
    """A view direction is not a moment in time."""

    def test_changing_the_camera_leaves_the_frame_alone(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        field_view, scene_view, _ = linked
        field_view.set_frame(6)
        scene_view._camera_box.setCurrentIndex(2)  # noqa: SLF001 - no public setter
        assert scene_view.frame_index == 6

    def test_the_selector_offers_every_named_view(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        _, scene_view, _ = linked
        offered = {
            scene_view._camera_box.itemData(index)  # noqa: SLF001 - no public reader
            for index in range(scene_view._camera_box.count())  # noqa: SLF001
        }
        assert offered == {preset.value for preset in CameraPreset}

    def test_the_view_reports_which_renderer_it_degraded_to(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        _, scene_view, _ = linked
        assert scene_view.renderer_note


class TestTheWorldBoxIsInjectable:
    """The #8728 argument, applied to the 3-D frame."""

    def test_the_view_reports_the_box_it_is_drawing_in(
        self, linked: tuple[SoleLoadFieldWidget, ShotViewportWidget, TracePanelWidget]
    ) -> None:
        _, scene_view, _ = linked
        assert scene_view.scale is not None

    def test_a_supplied_box_is_used_verbatim(
        self, evaluation: DesignEvaluation
    ) -> None:
        scene = evaluation.shot.scene
        assert scene is not None
        view = ShotViewportWidget("scene")
        view.set_shot(scene)
        wider = view.scale
        assert wider is not None
        second = ShotViewportWidget("scene")
        second.set_shot(scene, scale=wider)
        assert second.scale == wider
