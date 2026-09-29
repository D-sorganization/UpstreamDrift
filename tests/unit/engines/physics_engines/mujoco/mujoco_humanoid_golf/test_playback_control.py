"""Unit tests for playback_control.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.playback_control import (
    PlaybackController,
    PlaybackMode,
)


@pytest.fixture
def mock_data():
    times = np.linspace(0, 1.0, 11)  # 11 frames, dt=0.1
    states = np.zeros((11, 4))
    controls = np.ones((11, 2))
    return times, states, controls


def test_playback_controller_init(mock_data):
    """Test PlaybackController initialization."""
    times, states, controls = mock_data
    controller = PlaybackController(times, states, controls)

    assert controller.num_frames == 11
    assert controller.duration == 1.0
    assert controller.mode == PlaybackMode.STOPPED
    assert controller.speed == 1.0


def test_playback_controls(mock_data):
    """Test play, pause, stop."""
    times, states, controls = mock_data
    controller = PlaybackController(times, states, controls)

    controller.play()
    assert controller.mode == PlaybackMode.PLAYING
    assert controller.is_playing() is True

    controller.pause()
    assert controller.mode == PlaybackMode.PAUSED

    # Move forward then stop
    controller.seek_to_frame(5)
    controller.stop()
    assert controller.mode == PlaybackMode.STOPPED
    assert controller.current_frame == 0


def test_stepping(mock_data):
    """Test stepping logic."""
    times, states, controls = mock_data
    controller = PlaybackController(times, states, controls)

    controller.step_forward(3)
    assert controller.current_frame == 3

    controller.step_backward(1)
    assert controller.current_frame == 2

    # Test bounds
    controller.step_forward(20)
    assert controller.current_frame == 10

    controller.step_backward(20)
    assert controller.current_frame == 0


def test_seeking(mock_data):
    """Test seek logic."""
    times, states, controls = mock_data
    controller = PlaybackController(times, states, controls)

    controller.seek_to_time(0.5)
    assert controller.current_frame == 5

    controller.seek_to_percent(100.0)
    assert controller.current_frame == 10


def test_update_logic(mock_data):
    """Test update loop."""
    times, states, controls = mock_data
    controller = PlaybackController(times, states, controls)

    # Must be playing
    assert controller.update(0.1) is False

    controller.play()

    # Update by exactly dt=0.1. Should advance 1 frame.
    advanced = controller.update(0.1)
    assert advanced is True
    assert controller.current_frame == 1

    # Update by half dt. Shouldn't advance until accumulator reaches 1
    assert controller.update(0.05) is False
    assert controller.current_frame == 1
    assert controller.update(0.06) is True
    assert controller.current_frame == 2


def test_loop_logic(mock_data):
    """Test looping logic."""
    times, states, controls = mock_data
    controller = PlaybackController(times, states, controls)
    controller.play()
    controller.set_loop(True)

    controller.seek_to_frame(9)
    controller.update(0.2)  # Advance 2 frames
    assert controller.current_frame == 0  # Wraps around
    assert controller.mode == PlaybackMode.PLAYING


def test_stop_at_end(mock_data):
    """Test stopping at the end without loop."""
    times, states, controls = mock_data
    controller = PlaybackController(times, states, controls)
    controller.play()
    controller.set_loop(False)

    mock_finished = MagicMock()
    controller.on_playback_finished = mock_finished

    controller.seek_to_frame(9)
    controller.update(0.2)  # Advance 2 frames
    assert controller.current_frame == 10
    assert controller.mode == PlaybackMode.PAUSED
    mock_finished.assert_called_once()


@patch("cv2.cvtColor")
@patch("cv2.imwrite")
def test_export_frame(mock_imwrite, mock_cvtColor, mock_data):
    """Test exporting frame."""
    times, states, controls = mock_data
    controller = PlaybackController(times, states, controls)

    mock_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_bgr = np.ones((10, 10, 3), dtype=np.uint8)
    mock_cvtColor.return_value = mock_bgr

    def render(s, c):
        return mock_rgb

    controller.export_frame_as_image(5, "test.png", render)

    mock_cvtColor.assert_called_once()
    mock_imwrite.assert_called_once_with("test.png", mock_bgr)
