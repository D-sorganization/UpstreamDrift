"""Drawing the 3-D shot scene (issue #8706, epic #8699).

Headless: these tests build figures without a canvas and assert on the
artists, so they run where PyQt6 does not.

What is checked here is not "does it look right" -- a test cannot see -- but
the three properties that make the picture safe to show someone:

* the validity status and the fidelity tier are **inside the axes**, because
  a screenshot keeps its contents and loses its caption;
* nothing about the frame is auto-scaled. The world box and the depth ramp
  come from a :class:`~src.tools.bunker_shot_gui.render3d.SceneScale` fixed
  over the whole shot and, in a comparison, over both designs. Issue #8728
  fixed exactly this bug for the sole load field, where per-grid
  auto-scaling made two grinds look identical however far apart they were;
* the frame says the sand is a free-surface height rather than a grain bed.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.tools.bunker_shot_gui.render3d import (
    SceneScale,
    draw_scene_frame,
    scene_scale,
    shot_scene_still,
)
from src.tools.bunker_shot_gui.shot3d import CameraPreset, ShotScene

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="session")
def nominal_scene(nominal_shot) -> ShotScene:  # type: ignore[no-untyped-def]
    """The 3-D scene of the nominal shot."""
    scene = nominal_shot.scene
    assert scene is not None, nominal_shot.unavailable
    return scene


def _texts(figure: Figure) -> str:
    """Every string drawn inside any axes of a figure."""
    return "\n".join(
        text.get_text() for axes in figure.axes for text in axes.texts
    ) + "\n".join(axes.get_title() for axes in figure.axes)


class TestTheFrameIsDrawnInThreeDimensions:
    """#8706 asks for the head moving through the sand, not another plan."""

    def test_a_still_is_a_figure_with_a_three_dimensional_axes(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = shot_scene_still(nominal_scene)
        assert figure.axes
        assert hasattr(figure.axes[0], "get_zlim")

    def test_every_frame_of_the_shot_can_be_drawn(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = Figure(figsize=(6.0, 4.0))
        artists = draw_scene_frame(figure, nominal_scene, frame=0)
        for frame in range(nominal_scene.n_frames):
            artists.update(frame)

    def test_updating_a_frame_adds_no_artists(self, nominal_scene: ShotScene) -> None:
        """The transport mutates; it must not accumulate."""
        figure = Figure(figsize=(6.0, 4.0))
        artists = draw_scene_frame(figure, nominal_scene, frame=0)
        axes = figure.axes[0]
        before = (len(axes.lines), len(axes.texts), len(axes.collections))
        for frame in range(nominal_scene.n_frames):
            artists.update(frame)
        assert (len(axes.lines), len(axes.texts), len(axes.collections)) == before

    def test_a_frame_outside_the_shot_is_refused(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = Figure()
        artists = draw_scene_frame(figure, nominal_scene, frame=0)
        with pytest.raises(ValueError, match="outside the recorded shot"):
            artists.update(nominal_scene.n_frames)


class TestNothingIsAutoScaled:
    """The #8728 defect, in three dimensions."""

    def test_the_world_box_is_identical_at_every_frame(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = Figure()
        artists = draw_scene_frame(figure, nominal_scene, frame=0)
        axes = figure.axes[0]
        limits = (axes.get_xlim(), axes.get_ylim(), axes.get_zlim())
        artists.update(nominal_scene.n_frames - 1)
        assert (axes.get_xlim(), axes.get_ylim(), axes.get_zlim()) == limits

    def test_two_designs_can_be_put_on_one_scale(
        self, nominal_scene: ShotScene
    ) -> None:
        shared = scene_scale((nominal_scene, nominal_scene))
        assert isinstance(shared, SceneScale)
        first, second = Figure(), Figure()
        draw_scene_frame(first, nominal_scene, frame=0, scale=shared)
        draw_scene_frame(second, nominal_scene, frame=0, scale=shared)
        assert first.axes[0].get_xlim() == second.axes[0].get_xlim()
        assert first.axes[0].get_zlim() == second.axes[0].get_zlim()

    def test_a_shared_scale_covers_both_designs(self, nominal_scene: ShotScene) -> None:
        one = scene_scale((nominal_scene,))
        wider = SceneScale(
            x_m=(one.x_m[0] - 1.0, one.x_m[1] + 1.0),
            y_m=one.y_m,
            z_m=one.z_m,
            depth_m=one.depth_m,
        )
        merged = one.merged(wider)
        assert merged.x_m[0] <= one.x_m[0]
        assert merged.x_m[1] >= wider.x_m[1]

    def test_a_scale_needs_something_to_cover(self) -> None:
        with pytest.raises(ValueError, match="at least one scene"):
            scene_scale(())

    def test_a_degenerate_scale_is_refused(self) -> None:
        with pytest.raises(ValueError, match="increase"):
            SceneScale(
                x_m=(1.0, 0.0), y_m=(0.0, 1.0), z_m=(0.0, 1.0), depth_m=(0.0, 1.0)
            )


class TestTheFrameCarriesItsOwnValidity:
    """ADR-0032: status and tier in the frame, never caption-only."""

    def test_the_status_is_drawn_inside_the_axes(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = shot_scene_still(nominal_scene)
        assert nominal_scene.status.value.replace("_", " ").upper() in _texts(figure)

    def test_the_fidelity_tier_is_drawn_inside_the_axes(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = shot_scene_still(nominal_scene)
        assert nominal_scene.fidelity_tier.value.upper() in _texts(figure)

    def test_the_frame_says_the_model_is_not_calibrated_for_bunker_sand(
        self, nominal_scene: ShotScene
    ) -> None:
        assert "not calibrated" in _texts(shot_scene_still(nominal_scene)).lower()

    def test_the_stamp_follows_the_band_when_one_is_given(
        self, nominal_scene: ShotScene, nominal_shot
    ) -> None:  # type: ignore[no-untyped-def]
        """A shot that changes regime must not be stamped with one verdict."""
        band = nominal_shot.traces.band
        figure = Figure()
        artists = draw_scene_frame(figure, nominal_scene, frame=0, band=band)
        artists.update(0)
        opening = _texts(figure)
        artists.update(nominal_scene.n_frames - 1)
        assert band.status_at(0).value.replace("_", " ").upper() in opening


class TestTheFrameDoesNotImplyGrains:
    """The note the epic is explicit about."""

    def test_the_frame_says_the_sand_is_a_free_surface_height(
        self, nominal_scene: ShotScene
    ) -> None:
        drawn = _texts(shot_scene_still(nominal_scene)).lower()
        assert "free-surface" in drawn or "free surface" in drawn

    def test_the_frame_says_no_grains_are_resolved(
        self, nominal_scene: ShotScene
    ) -> None:
        assert "grain" in _texts(shot_scene_still(nominal_scene)).lower()

    def test_the_frame_says_the_divot_is_a_swept_envelope(
        self, nominal_scene: ShotScene
    ) -> None:
        assert "swept" in _texts(shot_scene_still(nominal_scene)).lower()


class TestUnitsAreOnEverything:
    """The demo report's standard."""

    def test_the_three_world_axes_are_labelled_in_millimetres(
        self, nominal_scene: ShotScene
    ) -> None:
        axes = shot_scene_still(nominal_scene).axes[0]
        for label in (
            axes.get_xlabel(),
            axes.get_ylabel(),
            axes.get_zlabel(),
        ):
            assert "[mm]" in label

    def test_the_title_states_the_moment_in_milliseconds(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = Figure()
        artists = draw_scene_frame(figure, nominal_scene, frame=3)
        artists.update(3)
        assert "ms" in figure.axes[0].get_title()


class TestTheCameraPresetsAreApplied:
    """#8706 names three; a preset that is not applied is decoration."""

    @pytest.mark.parametrize("preset", list(CameraPreset))
    def test_each_preset_sets_the_view(
        self, nominal_scene: ShotScene, preset: CameraPreset
    ) -> None:
        figure = Figure()
        draw_scene_frame(figure, nominal_scene, frame=0, camera=preset)
        axes = figure.axes[0]
        assert float(axes.elev) == pytest.approx(preset.elevation_deg)
        assert float(axes.azim) == pytest.approx(preset.azimuth_deg)

    def test_the_camera_can_be_changed_without_rebuilding(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = Figure()
        artists = draw_scene_frame(
            figure, nominal_scene, frame=0, camera=CameraPreset.FACE_ON
        )
        before = len(figure.axes[0].lines)
        artists.set_camera(CameraPreset.SOLE_LEVEL)
        assert float(figure.axes[0].elev) == pytest.approx(0.0)
        assert len(figure.axes[0].lines) == before

    def test_the_named_view_is_stated_in_the_frame(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = Figure()
        draw_scene_frame(
            figure, nominal_scene, frame=0, camera=CameraPreset.DOWN_THE_LINE
        )
        assert CameraPreset.DOWN_THE_LINE.label in _texts(figure)


class TestTheRendererIsTheDegradedOne:
    """ADR-0027: the fallback is reported, never a silent substitution."""

    def test_the_frame_names_the_renderer_actually_in_use(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = Figure()
        artists = draw_scene_frame(figure, nominal_scene, frame=0)
        assert artists.fallback.renderer == "matplotlib"

    def test_a_degraded_frame_says_which_providers_were_missing(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = Figure()
        artists = draw_scene_frame(figure, nominal_scene, frame=0)
        described = artists.fallback.describe()
        if artists.fallback.degraded:
            assert "unavailable" in described
        else:
            assert "ADR-0027" in described

    def test_the_scene_is_drawn_from_the_backend_neutral_payload(
        self, nominal_scene: ShotScene
    ) -> None:
        """The fallback consumes the same payload a real provider would.

        Otherwise the matplotlib view and a future MeshCat view could drift
        apart without anything noticing.
        """
        figure = Figure()
        artists = draw_scene_frame(figure, nominal_scene, frame=0)
        assert np.allclose(
            artists.payload.trajectory_xyz, nominal_scene.sole_reference_world_m
        )
