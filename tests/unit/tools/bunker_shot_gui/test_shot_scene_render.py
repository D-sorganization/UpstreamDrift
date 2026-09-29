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

Two more properties are pinned here for #8706's own two defects, and both
need a real rendered frame rather than an unrasterised one, since both
defects were only visible once pixels existed:

* the head is a solid :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`
  built from the lofted mesh, not a scatter of element centroids
  (defect 1) -- checked, after a real draw, by counting the collection's
  own rendered faces against the scene's;
* the footer caption never runs off the figure it is drawn on (defect 2) --
  checked by measuring the caption Text artist's actual window extent
  against the figure's, the same measurement
  :mod:`~src.tools.bunker_shot_gui.render3d` itself uses to wrap it, so this
  is a real regression test for "...not where sand h", not a guess at one.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.tools.bunker_shot_gui.render3d import (
    SceneScale,
    draw_scene_frame,
    scene_scale,
    shot_scene_still,
)
from src.tools.bunker_shot_gui.shot3d import CameraPreset, ShotScene

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_STILL_FIGSIZE = (9.0, 6.5)
"""The size the #8706 defect 2 bug report was filed against."""


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


def _draw(figure: Figure) -> None:
    """Force a real Agg draw, so paths and text extents are populated.

    Everything else in this file asserts on artists before a draw, which is
    enough for their properties -- but a ``Poly3DCollection``'s per-face
    paths and a ``Text``'s window extent are only computed by a draw, the
    same way :func:`~.render3d._figure_renderer` forces one for a bare
    figure rather than trusting an unrasterised state.
    """
    FigureCanvasAgg(figure).draw()


def _footer_caption(figure: Figure) -> Text:
    """Find the footer caption among a figure's texts, by its own content.

    Identified rather than reached for by attribute name: every other test
    in this file goes through the same public ``axes.texts``, and the
    caption is the only text carrying the divot's own wording.
    """
    for axes in figure.axes:
        for text in axes.texts:
            if "F0 moves no sand" in text.get_text():
                return text
    raise AssertionError("no footer caption text found in the figure")


def _head_mesh_collection(figure: Figure) -> Poly3DCollection:
    """Find the clubhead's solid mesh collection among a figure's axes."""
    for axes in figure.axes:
        for collection in axes.collections:
            if (
                isinstance(collection, Poly3DCollection)
                and collection.get_label() == "clubhead (lofted mesh)"
            ):
                return collection
    raise AssertionError("no clubhead mesh collection found in the figure")


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


class TestTheHeadIsDrawnAsASolid:
    """#8706 defect 1: the head used to render as a cloud of dots.

    ``render3d.py`` used to draw the element centroids as a marker-only
    line -- twice, once labelled "head surface" and once "sole elements" --
    which is what made the clubhead look like a scatter rather than a
    wedge. It is now a :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`
    built from ``scene.head_mesh_body``, the same watertight mesh the F0
    solver discretised, posed by :meth:`~.shot3d.ShotScene.head_mesh_world_m`.
    """

    def test_the_head_is_a_poly3dcollection_not_a_scatter(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = shot_scene_still(nominal_scene)
        assert _head_mesh_collection(figure) is not None

    def test_the_rendered_face_count_matches_the_scenes_mesh(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = shot_scene_still(nominal_scene)
        _draw(figure)
        collection = _head_mesh_collection(figure)
        assert len(collection.get_paths()) == nominal_scene.n_head_mesh_faces

    def test_the_mesh_is_translucent_so_the_divot_behind_it_still_reads(
        self, nominal_scene: ShotScene
    ) -> None:
        """An opaque solid would hide the divot floor behind it from every
        camera that looks along the swing -- the whole point of drawing a
        solid at all is to still be able to see past it."""
        figure = shot_scene_still(nominal_scene)
        collection = _head_mesh_collection(figure)
        alpha = collection.get_alpha()
        assert alpha is not None
        assert 0.0 < alpha < 1.0

    def test_updating_a_frame_moves_the_mesh_without_adding_collections(
        self, nominal_scene: ShotScene
    ) -> None:
        """Built once, mutated through ``set_verts`` -- never rebuilt per
        frame, matching the existing build-once/mutate pattern this module
        already used for the trail and the divot floor."""
        figure = Figure(figsize=(6.0, 4.0))
        artists = draw_scene_frame(figure, nominal_scene, frame=0)
        axes = figure.axes[0]
        before = len(axes.collections)
        for frame in range(nominal_scene.n_frames):
            artists.update(frame)
        assert len(axes.collections) == before

    def test_the_mesh_is_reposed_by_the_frame(self, nominal_scene: ShotScene) -> None:
        figure = Figure(figsize=(6.0, 4.0))
        artists = draw_scene_frame(figure, nominal_scene, frame=0)
        _draw(figure)
        first_paths = [
            path.vertices.copy() for path in _head_mesh_collection(figure).get_paths()
        ]
        artists.update(nominal_scene.n_frames - 1)
        _draw(figure)
        last_paths = [
            path.vertices.copy() for path in _head_mesh_collection(figure).get_paths()
        ]
        assert not all(
            np.array_equal(a, b) for a, b in zip(first_paths, last_paths, strict=True)
        )


class TestTheFooterCaptionFitsTheFigure:
    """#8706 defect 2: the caption used to clip mid-word at the right edge.

    ``"...not where sand h"`` was the literal symptom -- an unwrapped
    caption running past the figure's own right edge. Every check here
    renders a real frame and measures the caption's actual rendered extent,
    the same measurement ``render3d._recompute_note`` itself uses to decide
    where to wrap, so this is a regression test for the pixels rather than
    a guess about them.
    """

    @pytest.mark.parametrize("preset", list(CameraPreset))
    def test_the_caption_never_runs_past_the_figures_right_edge(
        self, nominal_scene: ShotScene, preset: CameraPreset
    ) -> None:
        figure = shot_scene_still(nominal_scene, camera=preset, figsize=_STILL_FIGSIZE)
        _draw(figure)
        renderer = figure.canvas.get_renderer()
        caption = _footer_caption(figure)
        bbox = caption.get_window_extent(renderer=renderer)
        assert bbox.x1 <= figure.bbox.x1 + 1.0

    @pytest.mark.parametrize("preset", list(CameraPreset))
    def test_the_caption_stays_within_the_figures_left_edge_too(
        self, nominal_scene: ShotScene, preset: CameraPreset
    ) -> None:
        figure = shot_scene_still(nominal_scene, camera=preset, figsize=_STILL_FIGSIZE)
        _draw(figure)
        renderer = figure.canvas.get_renderer()
        bbox = _footer_caption(figure).get_window_extent(renderer=renderer)
        assert bbox.x0 >= -1.0

    def test_a_long_line_actually_wraps_into_more_than_one_row(
        self, nominal_scene: ShotScene
    ) -> None:
        """Proof the wrap is doing something, not just that nothing clips:
        a figure narrow enough that even a short caption has to wrap."""
        figure = shot_scene_still(
            nominal_scene, camera=CameraPreset.SOLE_LEVEL, figsize=(4.0, 4.0)
        )
        caption = _footer_caption(figure)
        assert caption.get_text().count("\n") > 2

    def test_wrapping_never_drops_or_mangles_a_word(
        self, nominal_scene: ShotScene
    ) -> None:
        """Wrapping only ever breaks at a space (see
        ``render3d._wrap_to_pixels``), so collapsing the wrapped
        whitespace back down must reproduce the three source lines
        word for word -- proof nothing was clipped, not just that the
        box it was clipped against moved."""
        preset = CameraPreset.SOLE_LEVEL
        figure = shot_scene_still(nominal_scene, camera=preset, figsize=_STILL_FIGSIZE)
        caption = _footer_caption(figure)
        expected = " ".join(
            (
                f"{preset.label} - {preset.description}",
                nominal_scene.surface.describe(),
                nominal_scene.divot.describe(),
            )
        ).split()
        assert caption.get_text().split() == expected
