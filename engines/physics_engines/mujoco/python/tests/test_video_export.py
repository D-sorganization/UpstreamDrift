import sys
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

# Mock dependencies before import
sys.modules["cv2"] = MagicMock()
sys.modules["imageio"] = MagicMock()

from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export import (
    VideoExporter,
    VideoFormat,
    create_metrics_overlay,
    export_simulation_video,
)

# Mock constants for headless environment
WIDTH = 640
HEIGHT = 480
FPS = 30


@pytest.fixture
def mock_mujoco():
    """Create mock MuJoCo model and data."""
    model = MagicMock(spec=mujoco.MjModel)
    model.nq = 2
    model.nv = 2
    model.nu = 1

    data = MagicMock(spec=mujoco.MjData)
    data.qpos = np.zeros(2)
    data.qvel = np.zeros(2)
    data.ctrl = np.zeros(1)

    return model, data


@pytest.fixture
def mock_renderer():
    """Mock the MuJoCo Renderer."""
    with patch("mujoco.Renderer") as mock:
        renderer = mock.return_value
        # Return a dummy black frame
        renderer.render.return_value = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        yield mock


@pytest.fixture
def mock_cv2():
    """Mock cv2 module."""
    # Since we mocked it in sys.modules, we can just grab it
    mock = sys.modules["cv2"]

    # Reset mock
    mock.reset_mock()

    # Mock VideoWriter
    writer = MagicMock()
    writer.isOpened.return_value = True
    mock.VideoWriter.return_value = writer

    # Mock cvtColor (pass-through for shape check)
    mock.cvtColor.side_effect = lambda img, code: img

    # Mock constants
    mock.COLOR_RGB2BGR = 1
    mock.FONT_HERSHEY_SIMPLEX = 1

    return mock


@pytest.fixture
def mock_imageio():
    """Mock imageio module."""
    mock = sys.modules["imageio"]
    mock.reset_mock()
    return mock


class TestVideoExporter:
    def test_init(self, mock_mujoco, mock_renderer):
        model, data = mock_mujoco
        exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS)

        assert exporter.width == WIDTH
        assert exporter.height == HEIGHT
        assert exporter.fps == FPS
        assert exporter.format == VideoFormat.MP4
        mock_renderer.assert_called_once_with(model, width=WIDTH, height=HEIGHT)

    def test_start_recording_mp4(self, mock_mujoco, mock_renderer, mock_cv2):
        model, data = mock_mujoco
        exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.MP4)

        with patch(
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.CV2_AVAILABLE",
            True,
        ):
            success = exporter.start_recording("test.mp4")

        assert success
        mock_cv2.VideoWriter.assert_called_once()
        args = mock_cv2.VideoWriter.call_args[0]
        assert args[0] == "test.mp4"
        assert args[2] == FPS
        assert args[3] == (WIDTH, HEIGHT)
        assert exporter.writer is not None

    def test_start_recording_gif(self, mock_mujoco, mock_renderer, mock_imageio):
        model, data = mock_mujoco
        exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.GIF)

        with patch(
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.IMAGEIO_AVAILABLE",
            True,
        ):
            success = exporter.start_recording("test.gif")

        assert success
        assert exporter.frames == []
        assert exporter.writer is None

    def test_add_frame_video(self, mock_mujoco, mock_renderer, mock_cv2):
        model, data = mock_mujoco
        exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.MP4)

        with patch(
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.CV2_AVAILABLE",
            True,
        ):
            exporter.start_recording("test.mp4")
            exporter.add_frame()

        mock_renderer.return_value.update_scene.assert_called_with(data, camera=None)
        mock_renderer.return_value.render.assert_called_once()
        mock_cv2.cvtColor.assert_called_once()
        exporter.writer.write.assert_called_once()
        assert exporter.frame_count == 1

    def test_add_frame_gif(self, mock_mujoco, mock_renderer, mock_imageio):
        model, data = mock_mujoco
        exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.GIF)

        with patch(
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.IMAGEIO_AVAILABLE",
            True,
        ):
            exporter.start_recording("test.gif")
            exporter.add_frame()

        assert len(exporter.frames) == 1
        assert exporter.frame_count == 1

    def test_finish_recording_video(self, mock_mujoco, mock_renderer, mock_cv2):
        model, data = mock_mujoco
        exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.MP4)

        with patch(
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.CV2_AVAILABLE",
            True,
        ):
            exporter.start_recording("test.mp4")
            writer = exporter.writer  # Capture the writer before it's cleared
            exporter.finish_recording()

        writer.release.assert_called_once()
        assert exporter.writer is None

    def test_finish_recording_gif(self, mock_mujoco, mock_renderer, mock_imageio):
        model, data = mock_mujoco
        exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.GIF)

        with patch(
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.IMAGEIO_AVAILABLE",
            True,
        ):
            exporter.start_recording("test.gif")
            exporter.add_frame()
            exporter.finish_recording("test.gif")

        mock_imageio.mimsave.assert_called_once()
        assert exporter.frames == []

    def test_export_recording(self, mock_mujoco, mock_renderer, mock_cv2):
        model, data = mock_mujoco
        exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.MP4)

        initial_state = np.zeros(4)  # nq=2 + nv=2

        def control_func(t):
            return np.array([0.0])

        def progress_cb(current, total):
            pass

        with (
            patch("mujoco.mj_forward"),
            patch("mujoco.mj_step"),
            patch(
                "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.CV2_AVAILABLE",
                True,
            ),
        ):
            # We need to capture the writer after start_recording is called inside export_recording
            # But we can't easily access it from outside.
            # We can verify the mock_cv2.VideoWriter().release() was called.

            success = exporter.export_recording(
                "test.mp4",
                initial_state,
                control_func,
                duration=0.1,  # Short duration -> few frames
                progress_callback=progress_cb,
            )

        assert success
        assert exporter.frame_count > 0
        mock_cv2.VideoWriter.return_value.release.assert_called_once()

    def test_metrics_overlay(self, mock_mujoco, mock_cv2):
        model, data = mock_mujoco
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        metrics = {"Test Metric": lambda d: 42.0}

        with patch(
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.CV2_AVAILABLE",
            True,
        ):
            out_frame = create_metrics_overlay(frame, 1.0, data, metrics)

        # Check that putText was called for time and metric
        assert mock_cv2.putText.call_count >= 2
        assert out_frame is not frame  # Should return a copy or modified frame

    def test_export_simulation_video_function(
        self, mock_mujoco, mock_renderer, mock_cv2
    ):
        model, data = mock_mujoco

        N = 10
        states = np.zeros((N, 4))
        controls = np.zeros((N, 1))
        times = np.linspace(0, 1, N)

        with (
            patch("mujoco.mj_forward"),
            patch("mujoco.mj_name2id", return_value=-1),
            patch(
                "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.CV2_AVAILABLE",
                True,
            ),
        ):
            success = export_simulation_video(
                model,
                data,
                "output.mp4",
                states,
                controls,
                times,
                width=WIDTH,
                height=HEIGHT,
                fps=FPS,
                show_metrics=True,
            )

        assert success
        # Verify VideoExporter usage implicitly via mock_cv2
        mock_cv2.VideoWriter.assert_called_once()
