"""Drawing the sand volume in the matplotlib 3-D scene (issue #8729).

Headless: matplotlib with the Agg backend, no Qt.

What these tests hold down is not that pixels appeared. It is that the
frame cannot be read as more than it is:

* the extrusion labels itself in the frame, in the vocabulary #8711
  established, so nobody reads a 2-D solve as a solved volume;
* the tier that produced the sand is stated, so an F0 shot's flat plane
  and an F1 shot's grain bed are never confused;
* the colour ramp is fixed and merges across designs (#8728);
* direction is drawn, not just magnitude, because sand ahead of the sole
  and sand up the face reach the same speed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from matplotlib.figure import Figure

from bunkershot3d.fields.schema import FieldQuantity
from bunkershot3d.solvers.protocol import FidelityTier
from src.tools.bunker_shot_gui.render3d import draw_scene_frame, shot_scene_still
from src.tools.bunker_shot_gui.render3d_sand import SandVolumeArtists
from src.tools.bunker_shot_gui.sandvolume import sand_volume, sand_volume_scale
from src.tools.bunker_shot_gui.shot3d import CameraPreset, ShotScene

from .test_scene_sand import f1_scene, f1_volume

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _stamp_and_note(artists) -> str:  # type: ignore[no-untyped-def]
    """Every word the frame draws outside the axes furniture."""
    texts = [
        child.get_text()
        for child in artists._axes.get_children()
        if hasattr(child, "get_text")
    ]
    return "\n".join(texts)


class TestTheFrameSaysWhatTheSandIs:
    """A 3-D picture of a 2-D solve must announce itself."""

    def test_the_frame_says_extruded(self, nominal_scene: ShotScene) -> None:
        figure = Figure(figsize=(8.0, 6.0))
        artists = draw_scene_frame(figure, f1_scene(nominal_scene), frame=0)
        assert "extrud" in _stamp_and_note(artists).lower()

    def test_the_frame_names_the_tier_that_moved_the_sand(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = Figure(figsize=(8.0, 6.0))
        artists = draw_scene_frame(figure, f1_scene(nominal_scene), frame=0)
        assert "F1" in _stamp_and_note(artists)

    def test_an_f0_frame_still_denies_grains(self, nominal_scene: ShotScene) -> None:
        figure = Figure(figsize=(8.0, 6.0))
        artists = draw_scene_frame(figure, nominal_scene, frame=0)
        assert "resolves no grains" in _stamp_and_note(artists)

    def test_an_f0_frame_draws_no_sand_artists(self, nominal_scene: ShotScene) -> None:
        figure = Figure(figsize=(8.0, 6.0))
        artists = draw_scene_frame(figure, nominal_scene, frame=0)
        assert artists.sand is None

    def test_an_f1_frame_draws_sand_artists(self, nominal_scene: ShotScene) -> None:
        figure = Figure(figsize=(8.0, 6.0))
        artists = draw_scene_frame(figure, f1_scene(nominal_scene), frame=0)
        assert artists.sand is not None

    def test_the_validity_stamp_survives(self, nominal_scene: ShotScene) -> None:
        """MAX_VALIDATED_SPEED_M_S is 1.44; a greenside shot is far past it."""
        figure = Figure(figsize=(8.0, 6.0))
        artists = draw_scene_frame(figure, f1_scene(nominal_scene), frame=0)
        assert "not calibrated for bunker sand" in _stamp_and_note(artists)


class TestSandReadsAsMovingMaterial:
    """A blob with a colour is not a flow."""

    def test_direction_arrows_are_drawn(self, nominal_scene: ShotScene) -> None:
        figure = Figure(figsize=(8.0, 6.0))
        artists = draw_scene_frame(figure, f1_scene(nominal_scene), frame=0)
        assert artists.sand is not None
        assert artists.sand.n_arrows > 0

    def test_the_arrows_point_where_the_sand_goes(self) -> None:
        """The fixture's flow grows along x, so arrows lean along +x."""
        volume = f1_volume()
        figure = Figure(figsize=(6.0, 5.0))
        axes = figure.add_subplot(111, projection="3d")
        artists = SandVolumeArtists(axes, volume, sand_volume_scale((volume,)))
        artists.update(volume.n_frames - 1)
        segments = np.asarray(artists.arrow_segments)
        assert segments.size > 0
        # Shaft segments run from base to tip; the mean run must be +x.
        assert float(np.mean(segments[:, 1, 0] - segments[:, 0, 0])) > 0.0

    def test_still_frames_draw_fewer_cells_than_moving_ones(self) -> None:
        """Only moving sand is painted, so the picture is the motion."""
        volume = f1_volume()
        figure = Figure(figsize=(6.0, 5.0))
        axes = figure.add_subplot(111, projection="3d")
        artists = SandVolumeArtists(axes, volume, sand_volume_scale((volume,)))
        artists.update(0)
        first = artists.n_painted
        artists.update(volume.n_frames - 1)
        assert artists.n_painted >= first

    def test_every_sheet_is_drawn(self) -> None:
        volume = f1_volume()
        figure = Figure(figsize=(6.0, 5.0))
        axes = figure.add_subplot(111, projection="3d")
        artists = SandVolumeArtists(axes, volume, sand_volume_scale((volume,)))
        artists.update(volume.n_frames - 1)
        drawn = np.asarray(artists.painted_across_m)
        assert np.unique(np.round(drawn, 9)).size == volume.n_sheets

    def test_density_is_a_channel_too(self) -> None:
        volume = f1_volume()
        figure = Figure(figsize=(6.0, 5.0))
        axes = figure.add_subplot(111, projection="3d")
        artists = SandVolumeArtists(
            axes,
            volume,
            sand_volume_scale((volume,)),
            quantity=FieldQuantity.DENSITY,
        )
        artists.update(0)
        assert artists.quantity is FieldQuantity.DENSITY
        assert artists.n_painted > 0


class TestTheFrameSaysHowItIsCuttingTheExtrusion:
    """Square-on and edge-on are the same field and look nothing alike."""

    def test_a_square_on_camera_says_it_shows_the_section(
        self, nominal_scene: ShotScene
    ) -> None:
        figure = Figure(figsize=(8.0, 6.0))
        artists = draw_scene_frame(
            figure, f1_scene(nominal_scene), frame=0, camera=CameraPreset.FACE_ON
        )
        assert artists.sand is not None
        assert "square to the solved plane" in artists.sand.viewing_note()

    def test_sighting_along_the_line_says_the_sheets_are_edge_on(
        self, nominal_scene: ShotScene
    ) -> None:
        """Otherwise the stripes read as across-width structure."""
        figure = Figure(figsize=(8.0, 6.0))
        artists = draw_scene_frame(
            figure,
            f1_scene(nominal_scene),
            frame=0,
            camera=CameraPreset.DOWN_THE_LINE,
        )
        assert artists.sand is not None
        assert "edge-on" in artists.sand.viewing_note()

    def test_the_note_reaches_the_frame(self, nominal_scene: ShotScene) -> None:
        figure = Figure(figsize=(8.0, 6.0))
        artists = draw_scene_frame(
            figure,
            f1_scene(nominal_scene),
            frame=0,
            camera=CameraPreset.DOWN_THE_LINE,
        )
        assert "edge-on" in _stamp_and_note(artists)

    def test_changing_camera_changes_the_note(self, nominal_scene: ShotScene) -> None:
        figure = Figure(figsize=(8.0, 6.0))
        artists = draw_scene_frame(
            figure, f1_scene(nominal_scene), frame=0, camera=CameraPreset.FACE_ON
        )
        assert artists.sand is not None
        before = artists.sand.viewing_note()
        artists.set_camera(CameraPreset.DOWN_THE_LINE)
        assert artists.sand.viewing_note() != before

    def test_the_arrow_sheet_follows_the_eye(self, nominal_scene: ShotScene) -> None:
        """The sheets in front would otherwise hide the one with arrows."""
        figure = Figure(figsize=(8.0, 6.0))
        artists = draw_scene_frame(
            figure, f1_scene(nominal_scene), frame=0, camera=CameraPreset.FACE_ON
        )
        assert artists.sand is not None
        segments = np.asarray(artists.sand.arrow_segments)
        assert segments.size > 0
        volume = artists.sand.volume
        nearest_mm = float(volume.across_m.min()) * 1e3
        assert float(np.mean(segments[:, 0, 1])) < nearest_mm + 1.0


class TestTheCaptionFitsTheFrame:
    """A qualifier that is drawn over the axis label is not a qualifier."""

    def test_the_caption_clears_the_x_axis_label(
        self, nominal_scene: ShotScene
    ) -> None:
        """Issue #8729 took the caption from three lines to eight."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        figure = shot_scene_still(f1_scene(nominal_scene), figsize=(9.5, 7.0))
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        renderer = canvas.get_renderer()
        axes = figure.get_axes()[0]
        note = next(
            text
            for text in axes.texts
            if "extruded from the solved plane" in text.get_text()
            or "sand:" in text.get_text()
        )
        note_top = note.get_window_extent(renderer=renderer).y1
        label_bottom = axes.xaxis.label.get_window_extent(renderer=renderer).y0
        assert note_top <= label_bottom

    def test_the_caption_stays_inside_the_figure(
        self, nominal_scene: ShotScene
    ) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        figure = shot_scene_still(f1_scene(nominal_scene), figsize=(9.5, 7.0))
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        renderer = canvas.get_renderer()
        axes = figure.get_axes()[0]
        for text in axes.texts:
            assert text.get_window_extent(renderer=renderer).y0 >= -1.0

    def test_a_short_f0_caption_keeps_close_to_the_original_band(
        self, nominal_scene: ShotScene
    ) -> None:
        """An F0 frame must not lose plot area to a band it does not need."""
        figure = shot_scene_still(nominal_scene, figsize=(9.5, 7.0))
        assert 0.21 <= figure.subplotpars.bottom <= 0.24

    def test_a_sand_caption_is_given_more_room(self, nominal_scene: ShotScene) -> None:
        """The extra lines have to come from somewhere."""
        plain = shot_scene_still(nominal_scene, figsize=(9.5, 7.0))
        sandy = shot_scene_still(f1_scene(nominal_scene), figsize=(9.5, 7.0))
        assert sandy.subplotpars.bottom > plain.subplotpars.bottom

    def test_the_band_is_capped(self, nominal_scene: ShotScene) -> None:
        """Better a caption slightly tight than a picture with nowhere to be."""
        figure = shot_scene_still(f1_scene(nominal_scene), figsize=(9.5, 7.0))
        assert figure.subplotpars.bottom <= 0.34


class TestNothingAutoScales:
    """Issue #8728, in the volume."""

    def test_the_injected_scale_is_kept(self) -> None:
        slow = sand_volume(_scaled_field(5.0), n_sheets=3, max_cells=200)
        fast = sand_volume(_scaled_field(25.0), n_sheets=3, max_cells=200)
        shared = sand_volume_scale((slow, fast))
        figure = Figure(figsize=(6.0, 5.0))
        axes = figure.add_subplot(111, projection="3d")
        artists = SandVolumeArtists(axes, slow, shared)
        assert artists.scale == shared

    def test_two_designs_on_one_scale_paint_differently(self) -> None:
        """The #8728 defect: each normalised to its own peak looks identical."""
        slow = sand_volume(_scaled_field(5.0), n_sheets=3, max_cells=200)
        fast = sand_volume(_scaled_field(25.0), n_sheets=3, max_cells=200)
        shared = sand_volume_scale((slow, fast))
        colours = []
        for volume in (slow, fast):
            figure = Figure(figsize=(6.0, 5.0))
            axes = figure.add_subplot(111, projection="3d")
            artists = SandVolumeArtists(axes, volume, shared)
            artists.update(volume.n_frames - 1)
            colours.append(float(np.nanmean(artists.painted_values)))
        assert colours[0] < colours[1]

    def test_a_frame_outside_the_record_is_refused(self) -> None:
        volume = f1_volume()
        figure = Figure(figsize=(6.0, 5.0))
        axes = figure.add_subplot(111, projection="3d")
        artists = SandVolumeArtists(axes, volume, sand_volume_scale((volume,)))
        with pytest.raises(ValueError, match="outside"):
            artists.update(volume.n_frames)


class TestTheStillRenders:
    """The frame a batch sweep and the report actually write out."""

    @pytest.mark.parametrize("camera", list(CameraPreset))
    def test_every_camera_preset_renders(
        self, nominal_scene: ShotScene, camera: CameraPreset
    ) -> None:
        figure = shot_scene_still(f1_scene(nominal_scene), camera=camera)
        figure.canvas.draw()
        assert figure.get_axes()

    def test_the_scene_scale_holds_the_sand(self, nominal_scene: ShotScene) -> None:
        """The world box must not crop the volume it frames."""
        from src.tools.bunker_shot_gui.render3d import scene_scale

        scene = f1_scene(nominal_scene)
        limits = scene_scale((scene,))
        assert scene.sand is not None
        assert limits.y_m[0] <= float(scene.sand.across_m.min())
        assert limits.y_m[1] >= float(scene.sand.across_m.max())


def _scaled_field(peak_m_s: float):  # type: ignore[no-untyped-def]
    """The analytic slice field at a chosen peak speed."""
    from .test_slices import analytic_field

    return analytic_field(peak_m_s=peak_m_s)


def _f1(scene: ShotScene) -> ShotScene:
    """An F1-labelled scene, for the parametrised cases above."""
    return dataclasses.replace(
        scene,
        fidelity_tier=FidelityTier.F1,
        surface=dataclasses.replace(
            scene.surface, resolves_grains=True, tier=FidelityTier.F1
        ),
        sand=f1_volume(),
    )
