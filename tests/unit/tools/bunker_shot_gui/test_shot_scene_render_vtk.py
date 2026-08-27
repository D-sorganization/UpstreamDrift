"""Drawing the 3-D shot scene through PyVista (issue #8706, epic #8699).

This is the VTK/PyVista twin of ``test_shot_scene_render.py``: same scene,
same fixed :class:`~.render3d.SceneScale`, same in-frame validity stamp, and
the same three named cameras -- everything :mod:`~.render3d_vtk` promises to
keep identical to the matplotlib fallback. What is checked here is not "does
it look right" -- a test still cannot see -- but that a real depth-buffered
mesh, a real offscreen render, and the same honesty rules are all present:
the validity stamp is never optional, nothing is auto-scaled per frame, and
the clubhead is drawn from the lofted head's actual triangle mesh rather than
a point cloud.

Gated with ``pytest.importorskip("pyvista")`` and ``requires_pyvista``: a
stock CI runner without the ``viz3d`` extra skips this whole file, and
``test_render3d_vtk_degradation.py`` covers the fallback path it leaves
untested.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="the VTK/PyVista viz3d extra is optional")

from src.tools.bunker_shot_gui.render3d import scene_scale  # noqa: E402
from src.tools.bunker_shot_gui.render3d_vtk import (  # noqa: E402
    RENDERER,
    draw_scene_frame_vtk,
    shot_scene_still_vtk,
)
from src.tools.bunker_shot_gui.shot3d import CameraPreset, ShotScene  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_pyvista]


@pytest.fixture()
def rendered(nominal_scene: ShotScene, nominal_build):  # type: ignore[no-untyped-def]
    """One built-and-rendered frame, closed automatically after the test."""
    artists = shot_scene_still_vtk(nominal_scene, nominal_build)
    yield artists
    artists.close()


class TestTheFrameIsARealOffscreenRender:
    """#8706's VTK half: an actual depth-buffered render, not a stub."""

    def test_rendering_produces_a_non_blank_image(self, rendered) -> None:  # type: ignore[no-untyped-def]
        image = rendered.image_array()
        assert image.ndim == 3
        assert image.shape[2] in (3, 4)
        # A blank canvas has zero variance; the sand plane, the head and the
        # stamp between them guarantee this frame is not one.
        assert float(np.asarray(image, dtype=np.float64).std()) > 0.0

    def test_every_frame_of_the_shot_can_be_drawn(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        artists = draw_scene_frame_vtk(nominal_scene, nominal_build, frame=0)
        try:
            for frame in range(nominal_scene.n_frames):
                artists.update(frame)
                assert artists.image_array().size > 0
        finally:
            artists.close()

    def test_a_frame_outside_the_shot_is_refused(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        artists = draw_scene_frame_vtk(nominal_scene, nominal_build, frame=0)
        try:
            with pytest.raises(ValueError, match="outside the recorded shot"):
                artists.update(nominal_scene.n_frames)
        finally:
            artists.close()


class TestTheHeadIsASolidMeshNotAPointCloud:
    """The whole reason to reach for PyVista over the matplotlib scatter."""

    def test_the_head_mesh_keeps_the_lofted_headbuild_topology(
        self, rendered, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        mesh = rendered.head_mesh
        assert mesh.n_points == nominal_build.loft.mesh.vertices.shape[0]
        assert mesh.n_faces == nominal_build.loft.mesh.faces.shape[0]

    def test_the_head_moves_between_frames_with_the_recorded_pose(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        artists = draw_scene_frame_vtk(nominal_scene, nominal_build, frame=0)
        try:
            first = np.array(artists.head_mesh.points)
            artists.update(nominal_scene.n_frames - 1)
            last = np.array(artists.head_mesh.points)
            assert not np.allclose(first, last)
        finally:
            artists.close()

    def test_the_head_points_match_the_scenes_own_pose_transform(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        """Posed independently of :meth:`ShotScene.head_world_m`, but from
        the exact same rotation and position, so the mesh cannot silently
        drift from what the scene itself claims the pose is."""
        frame = 3
        artists = draw_scene_frame_vtk(nominal_scene, nominal_build, frame=frame)
        try:
            expected_mm = (
                nominal_build.loft.mesh.vertices @ nominal_scene.orientation[frame].T
                + nominal_scene.position_m[frame]
            ) * 1e3
            assert np.allclose(
                np.array(artists.head_mesh.points), expected_mm, atol=1e-6
            )
        finally:
            artists.close()


class TestNothingIsAutoScaled:
    """The #8728 defect, in the PyVista renderer too."""

    def test_the_scale_in_force_is_the_one_supplied(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        shared = scene_scale((nominal_scene, nominal_scene))
        artists = draw_scene_frame_vtk(
            nominal_scene, nominal_build, frame=0, scale=shared
        )
        try:
            assert artists.scale == shared
        finally:
            artists.close()

    def test_the_divot_colour_limits_stay_fixed_across_frames(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        """Recomputing scalars per frame must not touch the mapper's clim."""
        artists = draw_scene_frame_vtk(nominal_scene, nominal_build, frame=0)
        try:
            mapper = artists.plotter.actors["floor"].mapper
            before = mapper.scalar_range
            artists.update(nominal_scene.n_frames - 1)
            after = artists.plotter.actors["floor"].mapper.scalar_range
            assert before == after
        finally:
            artists.close()


class TestTheFrameCarriesItsOwnValidity:
    """ADR-0032: status and tier in the frame, never caption-only."""

    def test_the_status_is_stamped_in_frame(
        self, rendered, nominal_scene: ShotScene
    ) -> None:  # type: ignore[no-untyped-def]
        assert (
            nominal_scene.status.value.replace("_", " ").upper() in rendered.stamp_text
        )

    def test_the_fidelity_tier_is_stamped_in_frame(
        self, rendered, nominal_scene: ShotScene
    ) -> None:  # type: ignore[no-untyped-def]
        assert nominal_scene.fidelity_tier.value.upper() in rendered.stamp_text

    def test_the_stamp_says_the_model_is_not_calibrated_for_bunker_sand(
        self, rendered
    ) -> None:  # type: ignore[no-untyped-def]
        assert "not calibrated" in rendered.stamp_text.lower()

    def test_the_stamp_names_this_renderer(self, rendered) -> None:  # type: ignore[no-untyped-def]
        assert f"renderer: {RENDERER}" in rendered.stamp_text

    def test_a_prettier_picture_still_carries_the_stamp_every_frame(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        """The honesty stamp is not optional (task requirement, verbatim)."""
        artists = draw_scene_frame_vtk(nominal_scene, nominal_build, frame=0)
        try:
            for frame in range(nominal_scene.n_frames):
                artists.update(frame)
                assert artists.stamp_text
                assert "renderer: pyvista" in artists.stamp_text
        finally:
            artists.close()

    def test_the_stamp_follows_the_band_when_one_is_given(
        self, nominal_scene: ShotScene, nominal_build, nominal_shot
    ) -> None:  # type: ignore[no-untyped-def]
        band = nominal_shot.traces.band
        artists = draw_scene_frame_vtk(nominal_scene, nominal_build, frame=0, band=band)
        try:
            artists.update(0)
            opening = artists.stamp_text
            artists.update(nominal_scene.n_frames - 1)
            assert band.status_at(0).value.replace("_", " ").upper() in opening
        finally:
            artists.close()


class TestTheFrameDoesNotImplyGrains:
    """The note the epic is explicit about, carried in the qualifier text."""

    def test_the_note_says_the_sand_is_a_free_surface_height(self, rendered) -> None:  # type: ignore[no-untyped-def]
        note = rendered.note_text.lower()
        assert "free-surface" in note or "free surface" in note

    def test_the_note_says_no_grains_are_resolved(self, rendered) -> None:  # type: ignore[no-untyped-def]
        assert "grain" in rendered.note_text.lower()

    def test_the_note_says_the_divot_is_a_swept_envelope(self, rendered) -> None:  # type: ignore[no-untyped-def]
        assert "swept" in rendered.note_text.lower()


class TestTheCameraPresetsAreApplied:
    """#8706 names three; a preset that is not applied is decoration."""

    @pytest.mark.parametrize("preset", list(CameraPreset))
    def test_each_preset_renders(
        self, nominal_scene: ShotScene, nominal_build, preset: CameraPreset
    ) -> None:  # type: ignore[no-untyped-def]
        artists = draw_scene_frame_vtk(
            nominal_scene, nominal_build, frame=0, camera=preset
        )
        try:
            assert artists.camera == preset
            assert preset.label in artists.note_text
            assert artists.image_array().size > 0
        finally:
            artists.close()

    def test_the_camera_can_be_changed_without_rebuilding_the_head(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        artists = draw_scene_frame_vtk(
            nominal_scene, nominal_build, frame=0, camera=CameraPreset.FACE_ON
        )
        try:
            before = np.array(artists.head_mesh.points)
            artists.set_camera(CameraPreset.SOLE_LEVEL)
            after = np.array(artists.head_mesh.points)
            assert artists.camera == CameraPreset.SOLE_LEVEL
            assert np.allclose(before, after)
        finally:
            artists.close()

    def test_an_unknown_camera_name_is_refused(
        self, nominal_scene: ShotScene, nominal_build
    ) -> None:  # type: ignore[no-untyped-def]
        artists = draw_scene_frame_vtk(nominal_scene, nominal_build, frame=0)
        try:
            with pytest.raises(ValueError, match="unknown camera preset"):
                artists.set_camera("bird's eye")
        finally:
            artists.close()


class TestTheSceneIsDrawnFromTheBackendNeutralPayload:
    """ADR-0027: this renderer must not drift from what a real provider gets."""

    def test_the_payload_trajectory_matches_the_scenes_own_path(
        self, rendered, nominal_scene: ShotScene
    ) -> None:  # type: ignore[no-untyped-def]
        assert np.allclose(
            rendered.payload.trajectory_xyz, nominal_scene.sole_reference_world_m
        )


class TestTheRendererReportsItself:
    """ADR-0027: the fallback -- or its absence -- is reported, never silent."""

    def test_the_fallback_names_vtk_pyvista_and_is_not_degraded(self, rendered) -> None:  # type: ignore[no-untyped-def]
        assert rendered.fallback.degraded is False
        assert rendered.fallback.renderer == RENDERER
        assert "VTK/PyVista" in rendered.fallback.describe()

    def test_a_mismatched_band_is_refused_at_construction(
        self, nominal_scene: ShotScene, nominal_build, decelerating_traces
    ) -> None:  # type: ignore[no-untyped-def]
        """A band from a different shot must not silently qualify this one."""
        if decelerating_traces.band.n_frames == nominal_scene.n_frames:
            pytest.skip("fixtures happen to share a frame count on this build")
        with pytest.raises(ValueError, match="one shot"):
            draw_scene_frame_vtk(
                nominal_scene, nominal_build, frame=0, band=decelerating_traces.band
            )
