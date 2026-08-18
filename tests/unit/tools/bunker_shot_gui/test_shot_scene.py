"""The 3-D shot scene, resolved in time (issue #8706, epic #8699).

The workbench can already say *which part of the sole* carried load and
*when*. What it cannot show is the thing a designer pictures when they talk
about a bunker shot at all: the head going through the sand. That is what a
:class:`~src.tools.bunker_shot_gui.shot3d.ShotScene` carries -- pose over
time, the free surface, and the section the head sweeps out under it.

Everything here is headless: no Qt, no matplotlib, no display. The drawing is
tested in ``test_shot_scene_render``.

Two honesty properties are pinned here rather than left to the renderer,
because they are properties of the *data* and a renderer cannot restore them:

* the sand surface is a single free-surface **height**, and the scene says so
  in a machine-readable way -- F0 resolves no grains, so nothing downstream may
  present a grain bed;
* the divot section is the **swept lower envelope of the head**, not transported
  sand. F0 never moves a grain, so the only honest divot at this tier is a
  statement about where the head has been.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.solvers import EnvelopeStatus, FidelityTier
from src.shared.python.visualization.viewport import ViewportOverlayPayload
from src.tools.bunker_shot_gui.shot3d import (
    CameraPreset,
    DivotSection,
    SandSurface,
    ShotScene,
    viewport_payload,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="session")
def nominal_scene(nominal_shot) -> ShotScene:  # type: ignore[no-untyped-def]
    """The 3-D scene of the nominal shot."""
    scene = nominal_shot.scene
    assert scene is not None, nominal_shot.unavailable
    return scene


class TestThePoseComesFromTheRecordedShot:
    """#8706 asks for the delivered trajectory, not a re-simulation."""

    def test_the_scene_has_one_pose_per_solver_sample(
        self, nominal_scene: ShotScene, nominal_shot
    ) -> None:  # type: ignore[no-untyped-def]
        field = nominal_shot.sole_field
        assert nominal_scene.n_frames == field.n_frames
        assert np.allclose(nominal_scene.time_s, field.time_s)

    def test_every_pose_carries_a_position_and_an_orientation(
        self, nominal_scene: ShotScene
    ) -> None:
        assert nominal_scene.position_m.shape == (nominal_scene.n_frames, 3)
        assert nominal_scene.orientation.shape == (nominal_scene.n_frames, 3, 3)

    def test_the_orientations_are_rotations(self, nominal_scene: ShotScene) -> None:
        """A pose drawn from a non-orthonormal matrix would shear the head."""
        for rotation in nominal_scene.orientation:
            assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
            assert np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-9)

    def test_the_head_travels_toward_the_target(self, nominal_scene: ShotScene) -> None:
        path = nominal_scene.path_world_m
        assert path[-1, 0] > path[0, 0]

    def test_the_head_points_are_placed_by_the_pose(
        self, nominal_scene: ShotScene
    ) -> None:
        """World points are ``p + R c``, so a moved head moves its cloud."""
        first = nominal_scene.head_world_m(0)
        last = nominal_scene.head_world_m(nominal_scene.n_frames - 1)
        assert first.shape == last.shape
        assert not np.allclose(first, last)

    def test_a_frame_outside_the_record_is_refused(
        self, nominal_scene: ShotScene
    ) -> None:
        with pytest.raises(ValueError, match="outside the recorded shot"):
            nominal_scene.head_world_m(nominal_scene.n_frames)


class TestTheSandSurfaceIsAHeightNotABed:
    """The load-bearing honesty property of #8706."""

    def test_the_surface_is_a_single_free_surface_height(
        self, nominal_scene: ShotScene
    ) -> None:
        surface = nominal_scene.surface
        assert isinstance(surface, SandSurface)
        assert np.isfinite(surface.height_m)

    def test_the_surface_says_it_is_not_a_resolved_grain_bed(
        self, nominal_scene: ShotScene
    ) -> None:
        """Machine-readable, so a renderer cannot quietly imply otherwise."""
        assert nominal_scene.surface.resolves_grains is False
        assert "grain" in nominal_scene.surface.describe().lower()

    def test_the_free_surface_is_recovered_from_the_trace_not_assumed(
        self, nominal_scene: ShotScene
    ) -> None:
        """The sole depth and the sole world height must agree at every sample.

        ``sole_depth = free_surface - z(sole reference)`` by definition, so a
        scene whose surface came from somewhere else would fail this.
        """
        heights = nominal_scene.sole_reference_world_m[:, 2]
        depths = nominal_scene.surface.height_m - heights
        assert np.allclose(depths, nominal_scene.sole_depth_m, atol=1e-12)

    def test_the_surface_spans_the_whole_path(self, nominal_scene: ShotScene) -> None:
        low, high = nominal_scene.surface.along_extent_m
        path_x = nominal_scene.path_world_m[:, 0]
        assert low <= float(path_x.min())
        assert high >= float(path_x.max())


class TestTheDivotIsASweptEnvelope:
    """F0 transports no sand, so the divot is a statement about the head."""

    def test_the_section_says_what_it_is(self, nominal_scene: ShotScene) -> None:
        divot = nominal_scene.divot
        assert isinstance(divot, DivotSection)
        assert divot.is_swept_envelope is True
        assert "swept" in divot.describe().lower()

    def test_the_section_has_one_floor_per_frame_and_station(
        self, nominal_scene: ShotScene
    ) -> None:
        divot = nominal_scene.divot
        assert divot.floor_m.shape == (nominal_scene.n_frames, divot.n_stations)

    def test_the_divot_only_ever_deepens(self, nominal_scene: ShotScene) -> None:
        """A swept envelope is cumulative: nothing fills back in mid-shot."""
        floor = nominal_scene.divot.floor_m
        assert np.all(np.diff(floor, axis=0) <= 1e-15)

    def test_the_untouched_floor_sits_at_the_free_surface(
        self, nominal_scene: ShotScene
    ) -> None:
        divot = nominal_scene.divot
        assert float(divot.floor_m.max()) <= divot.surface_height_m + 1e-15

    def test_the_depth_is_never_negative(self, nominal_scene: ShotScene) -> None:
        assert float(nominal_scene.divot.depth_m.min()) >= 0.0

    def test_the_head_actually_cuts_something(self, nominal_scene: ShotScene) -> None:
        assert float(nominal_scene.divot.depth_m.max()) > 0.0

    def test_the_section_area_grows_with_the_shot(
        self, nominal_scene: ShotScene
    ) -> None:
        areas = nominal_scene.divot.section_area_m2
        assert areas.shape == (nominal_scene.n_frames,)
        assert np.all(np.diff(areas) >= -1e-15)


class TestTheCameraPresetsMatchHowAShotIsDiscussed:
    """#8706 names three views; a preset is data, not a matplotlib call."""

    def test_the_three_named_views_exist(self) -> None:
        assert {preset.value for preset in CameraPreset} == {
            "down_the_line",
            "face_on",
            "sole_level",
        }

    def test_every_preset_states_its_angles_in_degrees(self) -> None:
        for preset in CameraPreset:
            assert -90.0 <= preset.elevation_deg <= 90.0
            assert -180.0 <= preset.azimuth_deg <= 180.0

    def test_every_preset_carries_a_backend_neutral_eye_direction(self) -> None:
        """A provider that is not matplotlib still has to place a camera."""
        for preset in CameraPreset:
            direction = preset.eye_direction
            assert direction.shape == (3,)
            assert np.isclose(float(np.linalg.norm(direction)), 1.0)

    def test_the_sole_level_view_sights_along_the_leading_edge(self) -> None:
        """Sole level means level: the eye sits in the free-surface plane."""
        assert CameraPreset.SOLE_LEVEL.elevation_deg == pytest.approx(0.0)
        assert abs(float(CameraPreset.SOLE_LEVEL.eye_direction[2])) < 1e-12

    def test_down_the_line_looks_along_the_target_line(self) -> None:
        """The eye sits behind the entry, so the view direction is ``+x``."""
        direction = CameraPreset.DOWN_THE_LINE.eye_direction
        assert float(direction[0]) < 0.0
        assert abs(float(direction[1])) < 1e-9

    def test_face_on_looks_across_the_target_line(self) -> None:
        direction = CameraPreset.FACE_ON.eye_direction
        assert abs(float(direction[1])) > abs(float(direction[0]))

    def test_every_preset_says_what_it_is_for(self) -> None:
        for preset in CameraPreset:
            assert preset.label
            assert preset.description

    def test_an_unknown_preset_names_the_valid_ones(self) -> None:
        with pytest.raises(ValueError, match="down_the_line"):
            CameraPreset("from_the_moon")


class TestTheSceneCarriesItsValidity:
    """ADR-0032: a picture without its verdict reads as a measurement."""

    def test_the_scene_carries_the_verdict_of_the_shot(
        self, nominal_scene: ShotScene, nominal_shot
    ) -> None:  # type: ignore[no-untyped-def]
        assert nominal_scene.verdict is nominal_shot.verdict
        assert nominal_scene.status is nominal_shot.verdict.status

    def test_the_scene_carries_its_fidelity_tier(
        self, nominal_scene: ShotScene
    ) -> None:
        assert nominal_scene.fidelity_tier is FidelityTier.F0

    def test_the_nominal_shot_is_outside_the_validated_envelope(
        self, nominal_scene: ShotScene
    ) -> None:
        """Pinned so a scene that silently reads WITHIN is caught."""
        assert nominal_scene.status is not EnvelopeStatus.WITHIN


class TestTheSceneRendersThroughTheCanonicalViewport:
    """ADR-0027: the payload is backend-neutral, so no renderer is duplicated."""

    def test_the_scene_builds_a_viewport_overlay_payload(
        self, nominal_scene: ShotScene
    ) -> None:
        payload = viewport_payload(nominal_scene)
        assert isinstance(payload, ViewportOverlayPayload)
        assert payload.frame == "world_Zup"
        assert payload.units == "SI"

    def test_the_payload_trajectory_is_the_sole_reference_path(
        self, nominal_scene: ShotScene
    ) -> None:
        payload = viewport_payload(nominal_scene)
        assert np.allclose(payload.trajectory_xyz, nominal_scene.sole_reference_world_m)

    def test_the_payload_carries_named_markers(self, nominal_scene: ShotScene) -> None:
        payload = viewport_payload(nominal_scene)
        assert payload.has_marker_overlay
        assert len(payload.marker_names) == payload.markers_xyz.shape[1]
        assert "leading_edge" in payload.marker_names

    def test_the_payload_carries_the_wrench_when_it_is_given_one(
        self, nominal_scene: ShotScene
    ) -> None:
        wrench = np.zeros((nominal_scene.n_frames, 6), dtype=float)
        payload = viewport_payload(nominal_scene, wrench=wrench)
        assert payload.has_wrench_overlay

    def test_the_payload_records_the_validity_in_its_metadata(
        self, nominal_scene: ShotScene
    ) -> None:
        """A backend that draws the payload must be able to stamp the frame."""
        payload = viewport_payload(nominal_scene)
        assert payload.meta["envelope_status"] == nominal_scene.status.value
        assert payload.meta["fidelity_tier"] == nominal_scene.fidelity_tier.value
        assert payload.meta["resolves_grains"] is False

    def test_a_wrench_of_the_wrong_length_is_refused(
        self, nominal_scene: ShotScene
    ) -> None:
        with pytest.raises(ValueError, match="wrench"):
            viewport_payload(nominal_scene, wrench=np.zeros((3, 6)))


class TestTheValueObjectsDefendThemselves:
    """``raise``, never ``assert``: ``python -O`` must not strip these."""

    def test_a_surface_with_a_non_finite_height_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            SandSurface(
                height_m=float("nan"),
                along_extent_m=(-0.1, 0.1),
                across_extent_m=(-0.05, 0.05),
            )

    def test_a_surface_with_an_inverted_extent_is_refused(self) -> None:
        with pytest.raises(ValueError, match="increase"):
            SandSurface(
                height_m=0.0,
                along_extent_m=(0.1, -0.1),
                across_extent_m=(-0.05, 0.05),
            )

    def test_a_divot_floor_above_the_surface_is_refused(self) -> None:
        with pytest.raises(ValueError, match="above the free surface"):
            DivotSection(
                station_m=np.array([0.0, 1.0]),
                floor_m=np.array([[0.5, 0.0], [0.0, 0.0]]),
                surface_height_m=0.0,
            )

    def test_a_divot_that_fills_back_in_is_refused(self) -> None:
        """A swept envelope cannot un-sweep; that would be sand transport."""
        with pytest.raises(ValueError, match="only ever deepen"):
            DivotSection(
                station_m=np.array([0.0, 1.0]),
                floor_m=np.array([[-1.0, -1.0], [-0.5, -1.0]]),
                surface_height_m=0.0,
            )

    def test_a_divot_with_unsorted_stations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            DivotSection(
                station_m=np.array([1.0, 0.0]),
                floor_m=np.zeros((2, 2)),
                surface_height_m=0.0,
            )
