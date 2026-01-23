"""Tests for Multi-Perspective Viewpoint Controls.

Guideline L1 implementation tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from shared.python.viewpoint_controls import (
    CameraPreset,
    CameraState,
    TrackingTarget,
    ViewpointController,
    compute_tracking_look_at,
    create_camera_from_preset,
    create_custom_camera,
    create_multiview_layout,
    create_standard_2x2_layout,
    create_transition_sequence,
    get_preset_camera_params,
    interpolate_camera_states,
    spherical_to_cartesian,
)


class TestSphericalToCartesian:
    """Tests for spherical coordinate conversion."""

    def test_azimuth_zero_is_positive_x(self) -> None:
        """Azimuth 0° should place camera along +X."""
        center = np.zeros(3)
        pos = spherical_to_cartesian(0, 0, 1.0, center)

        assert pos[0] == pytest.approx(1.0)
        assert pos[1] == pytest.approx(0.0)
        assert pos[2] == pytest.approx(0.0)

    def test_azimuth_90_is_positive_y(self) -> None:
        """Azimuth 90° should place camera along +Y."""
        center = np.zeros(3)
        pos = spherical_to_cartesian(90, 0, 1.0, center)

        assert pos[0] == pytest.approx(0.0, abs=1e-10)
        assert pos[1] == pytest.approx(1.0)
        assert pos[2] == pytest.approx(0.0)

    def test_elevation_90_is_positive_z(self) -> None:
        """Elevation 90° should place camera along +Z."""
        center = np.zeros(3)
        pos = spherical_to_cartesian(0, 90, 1.0, center)

        assert pos[0] == pytest.approx(0.0, abs=1e-10)
        assert pos[1] == pytest.approx(0.0, abs=1e-10)
        assert pos[2] == pytest.approx(1.0)

    def test_distance_scaling(self) -> None:
        """Distance should scale position proportionally."""
        center = np.zeros(3)
        pos1 = spherical_to_cartesian(0, 0, 1.0, center)
        pos2 = spherical_to_cartesian(0, 0, 3.0, center)

        np.testing.assert_allclose(pos2, 3.0 * pos1)

    def test_center_offset(self) -> None:
        """Position should be offset by center."""
        center = np.array([5.0, 3.0, 1.0])
        pos = spherical_to_cartesian(0, 0, 1.0, center)

        expected = center + np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(pos, expected)


class TestPresetCameraParams:
    """Tests for camera preset parameters."""

    def test_face_on_preset(self) -> None:
        """Face-on preset should look at golfer from -X direction."""
        azimuth, elevation, look_at = get_preset_camera_params(CameraPreset.FACE_ON)

        # Default face-on: camera behind ball (opposite target direction)
        assert azimuth == pytest.approx(180.0)
        assert elevation == pytest.approx(0.0)

    def test_overhead_preset(self) -> None:
        """Overhead preset should have high elevation."""
        azimuth, elevation, look_at = get_preset_camera_params(CameraPreset.OVERHEAD)

        assert elevation == pytest.approx(80.0)


class TestCreateCameraFromPreset:
    """Tests for camera creation from presets."""

    def test_creates_valid_camera(self) -> None:
        """Should create a valid camera state."""
        camera = create_camera_from_preset(CameraPreset.FACE_ON)

        assert camera is not None
        assert camera.position.shape == (3,)
        assert camera.look_at.shape == (3,)
        assert camera.up_vector.shape == (3,)
        assert camera.preset == CameraPreset.FACE_ON

    def test_all_presets_create_cameras(self) -> None:
        """All presets should create valid cameras."""
        presets = [
            CameraPreset.FACE_ON,
            CameraPreset.DOWN_TARGET_LINE,
            CameraPreset.OVERHEAD,
            CameraPreset.RIGHT_SIDE,
            CameraPreset.LEFT_SIDE,
            CameraPreset.BEHIND_BALL,
            CameraPreset.IMPACT_CLOSE,
        ]

        for preset in presets:
            camera = create_camera_from_preset(preset)
            assert camera is not None
            assert camera.preset == preset


class TestCustomCamera:
    """Tests for custom camera creation."""

    def test_creates_camera_at_position(self) -> None:
        """Custom camera should be at computed position."""
        look_at = np.array([0.0, 0.0, 1.0])
        camera = create_custom_camera(0, 0, 3.0, look_at)

        # Camera should be 3m from look_at along +X
        expected_pos = look_at + np.array([3.0, 0.0, 0.0])
        np.testing.assert_allclose(camera.position, expected_pos, atol=1e-10)
        np.testing.assert_allclose(camera.look_at, look_at)
        assert camera.preset == CameraPreset.CUSTOM


class TestCameraInterpolation:
    """Tests for camera state interpolation."""

    def test_t_zero_returns_start(self) -> None:
        """t=0 should return start camera."""
        start = CameraState(
            position=np.array([0.0, 0.0, 3.0]),
            look_at=np.array([0.0, 0.0, 0.0]),
        )
        end = CameraState(
            position=np.array([3.0, 0.0, 0.0]),
            look_at=np.array([0.0, 0.0, 0.0]),
        )

        result = interpolate_camera_states(start, end, 0.0)

        np.testing.assert_allclose(result.position, start.position)

    def test_t_one_returns_end(self) -> None:
        """t=1 should return end camera."""
        start = CameraState(
            position=np.array([0.0, 0.0, 3.0]),
            look_at=np.array([0.0, 0.0, 0.0]),
        )
        end = CameraState(
            position=np.array([3.0, 0.0, 0.0]),
            look_at=np.array([0.0, 0.0, 0.0]),
        )

        result = interpolate_camera_states(start, end, 1.0)

        np.testing.assert_allclose(result.position, end.position)

    def test_t_half_is_between(self) -> None:
        """t=0.5 should be between start and end."""
        start = CameraState(
            position=np.array([0.0, 0.0, 0.0]),
            look_at=np.array([0.0, 0.0, 0.0]),
        )
        end = CameraState(
            position=np.array([4.0, 0.0, 0.0]),
            look_at=np.array([0.0, 0.0, 0.0]),
        )

        result = interpolate_camera_states(start, end, 0.5)

        # With smoothstep, t=0.5 gives exactly 0.5
        assert 1.0 < result.position[0] < 3.0


class TestTransitionSequence:
    """Tests for camera transition sequences."""

    def test_creates_correct_number_of_frames(self) -> None:
        """Should create specified number of frames."""
        start = create_camera_from_preset(CameraPreset.FACE_ON)
        end = create_camera_from_preset(CameraPreset.DOWN_TARGET_LINE)

        sequence = create_transition_sequence(start, end, 10)

        assert len(sequence) == 10

    def test_sequence_starts_at_start(self) -> None:
        """First frame should match start camera."""
        start = create_camera_from_preset(CameraPreset.FACE_ON)
        end = create_camera_from_preset(CameraPreset.DOWN_TARGET_LINE)

        sequence = create_transition_sequence(start, end, 10)

        np.testing.assert_allclose(sequence[0].position, start.position)


class TestTrackingLookAt:
    """Tests for camera tracking."""

    def test_clubhead_tracking(self) -> None:
        """Should return clubhead position when tracking clubhead."""
        clubhead = np.array([1.0, 2.0, 3.0])

        look_at = compute_tracking_look_at(
            TrackingTarget.CLUBHEAD,
            clubhead_position=clubhead,
        )

        np.testing.assert_allclose(look_at, clubhead)

    def test_fallback_when_none(self) -> None:
        """Should return default when target not available."""
        look_at = compute_tracking_look_at(
            TrackingTarget.CLUBHEAD,
            clubhead_position=None,
        )

        # Should return some default, not fail
        assert look_at.shape == (3,)


class TestMultiviewLayout:
    """Tests for multi-view layout creation."""

    def test_standard_2x2_layout(self) -> None:
        """Should create 2x2 layout with 4 cameras."""
        layout = create_standard_2x2_layout()

        assert layout.rows == 2
        assert layout.cols == 2
        assert len(layout.camera_states) == 4

    def test_custom_layout_sizing(self) -> None:
        """Should size layout based on number of presets."""
        presets_2 = [CameraPreset.FACE_ON, CameraPreset.DOWN_TARGET_LINE]
        layout_2 = create_multiview_layout(presets_2)
        assert layout_2.rows == 1
        assert layout_2.cols == 2

        presets_4 = [
            CameraPreset.FACE_ON,
            CameraPreset.DOWN_TARGET_LINE,
            CameraPreset.OVERHEAD,
            CameraPreset.RIGHT_SIDE,
        ]
        layout_4 = create_multiview_layout(presets_4)
        assert layout_4.rows == 2
        assert layout_4.cols == 2


class TestViewpointController:
    """Tests for ViewpointController class."""

    def test_initial_camera_is_face_on(self) -> None:
        """Controller should start with face-on preset."""
        controller = ViewpointController()

        assert controller.current_camera.preset == CameraPreset.FACE_ON

    def test_set_preset_changes_camera(self) -> None:
        """Setting preset should change camera state."""
        controller = ViewpointController()

        camera = controller.set_preset(CameraPreset.OVERHEAD)

        assert camera.preset == CameraPreset.OVERHEAD

    def test_set_custom_view(self) -> None:
        """Should set custom view angles."""
        controller = ViewpointController()

        camera = controller.set_custom_view(45, 30, 5.0)

        assert camera.preset == CameraPreset.CUSTOM

    def test_transition_creates_frames(self) -> None:
        """Transition should create intermediate frames."""
        controller = ViewpointController()

        controller.set_preset(CameraPreset.OVERHEAD, transition_frames=10)

        assert controller.transition_in_progress
        assert len(controller.transition_frames) == 10

    def test_update_advances_transition(self) -> None:
        """Update should advance through transition."""
        controller = ViewpointController()

        controller.set_preset(CameraPreset.OVERHEAD, transition_frames=5)

        # First update advances index from 0 to 1
        controller.update()
        assert controller.transition_index == 1

        # After enough updates, transition completes
        for _ in range(5):
            controller.update()

        assert not controller.transition_in_progress

    def test_tracking_mode_updates_look_at(self) -> None:
        """Tracking mode should update look-at point."""
        controller = ViewpointController()
        controller.set_tracking_target(TrackingTarget.CLUBHEAD)

        new_clubhead = np.array([5.0, 0.0, 0.5])

        controller.update(clubhead_position=new_clubhead)

        # Look-at should have changed to clubhead position
        np.testing.assert_allclose(controller.current_camera.look_at, new_clubhead)
