"""Tests for video export functionality."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

# Mock dependencies before import
sys.modules["cv2"] = MagicMock()
sys.modules["imageio"] = MagicMock()

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export import (  # noqa: E402, E501
    VideoExporter,
    VideoFormat,
    create_metrics_overlay,
    export_simulation_video,
)

# Path to the mujoco module reference inside video_export (imported as `mj`)
_VID_EXPORT_MJ = (
    "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.mj"
)

# Mock constants for headless environment
WIDTH = 640
HEIGHT = 480
FPS = 30


def _make_mock_mj(**overrides: Any) -> MagicMock:
    """Create a mock mujoco module that preserves enum types but stubs C functions."""
    mock = MagicMock()
    mock.mjtObj = mujoco.mjtObj
    mock.mjtJoint = mujoco.mjtJoint
    mock.mjtGeom = mujoco.mjtGeom
    # Return a dummy black frame from the Renderer mock
    renderer_inst = MagicMock()
    renderer_inst.render.return_value = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    mock.Renderer.return_value = renderer_inst
    for key, val in overrides.items():
        setattr(mock, key, val)
    return mock


@pytest.fixture
def mock_mujoco() -> tuple[MagicMock, MagicMock]:
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
def mock_cv2() -> Any:
    """Mock cv2 module."""
    mock = sys.modules["cv2"]
    mock.reset_mock()

    writer = MagicMock()
    writer.isOpened.return_value = True
    mock.VideoWriter.return_value = writer
    mock.cvtColor.side_effect = lambda img, code: img
    mock.COLOR_RGB2BGR = 1  # type: ignore[attr-defined]
    mock.FONT_HERSHEY_SIMPLEX = 1  # type: ignore[attr-defined]
    mock.VideoWriter_fourcc.return_value = 0x7634706D  # mp4v

    return mock


@pytest.fixture
def mock_imageio() -> Any:
    """Mock imageio module."""
    mock = sys.modules["imageio"]
    mock.reset_mock()
    return mock


class TestVideoExporter:
    """Tests for the VideoExporter class."""

    def test_init(self, mock_mujoco: tuple[MagicMock, MagicMock]) -> None:
        """Test VideoExporter initialization."""
        model, data = mock_mujoco
        mock_mj = _make_mock_mj()

        with patch(_VID_EXPORT_MJ, mock_mj):
            exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS)

        assert exporter.width == WIDTH
        assert exporter.height == HEIGHT
        assert exporter.fps == FPS
        assert exporter.format == VideoFormat.MP4
        mock_mj.Renderer.assert_called_once_with(model, width=WIDTH, height=HEIGHT)

    def test_start_recording_mp4(
        self,
        mock_mujoco: tuple[MagicMock, MagicMock],
        mock_cv2: Any,
    ) -> None:
        """Test starting MP4 recording."""
        model, data = mock_mujoco
        mock_mj = _make_mock_mj()

        with (
            patch(_VID_EXPORT_MJ, mock_mj),
            patch(
                "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.CV2_AVAILABLE",
                True,
            ),
        ):
            exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.MP4)
            success = exporter.start_recording("test.mp4")

        assert success
        mock_cv2.VideoWriter.assert_called_once()
        args = mock_cv2.VideoWriter.call_args[0]
        assert args[0] == "test.mp4"
        assert args[2] == FPS
        assert args[3] == (WIDTH, HEIGHT)
        assert exporter.writer is not None

    def test_start_recording_gif(
        self,
        mock_mujoco: tuple[MagicMock, MagicMock],
        mock_imageio: Any,
    ) -> None:
        """Test starting GIF recording."""
        model, data = mock_mujoco
        mock_mj = _make_mock_mj()

        with (
            patch(_VID_EXPORT_MJ, mock_mj),
            patch(
                "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.IMAGEIO_AVAILABLE",
                True,
            ),
        ):
            exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.GIF)
            success = exporter.start_recording("test.gif")

        assert success
        assert exporter.frames == []
        assert exporter.writer is None

    def test_add_frame_video(
        self,
        mock_mujoco: tuple[MagicMock, MagicMock],
        mock_cv2: Any,
    ) -> None:
        """Test adding a frame to video recording."""
        model, data = mock_mujoco
        mock_mj = _make_mock_mj()

        with (
            patch(_VID_EXPORT_MJ, mock_mj),
            patch(
                "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.CV2_AVAILABLE",
                True,
            ),
        ):
            exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.MP4)
            exporter.start_recording("test.mp4")
            exporter.add_frame()

        renderer = mock_mj.Renderer.return_value
        renderer.update_scene.assert_called_with(data, camera=None)
        renderer.render.assert_called_once()
        mock_cv2.cvtColor.assert_called_once()
        exporter.writer.write.assert_called_once()
        assert exporter.frame_count == 1

    def test_add_frame_gif(
        self,
        mock_mujoco: tuple[MagicMock, MagicMock],
        mock_imageio: Any,
    ) -> None:
        """Test adding a frame to GIF recording."""
        model, data = mock_mujoco
        mock_mj = _make_mock_mj()

        with (
            patch(_VID_EXPORT_MJ, mock_mj),
            patch(
                "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.IMAGEIO_AVAILABLE",
                True,
            ),
        ):
            exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.GIF)
            exporter.start_recording("test.gif")
            exporter.add_frame()

        assert len(exporter.frames) == 1
        assert exporter.frame_count == 1

    def test_finish_recording_video(
        self,
        mock_mujoco: tuple[MagicMock, MagicMock],
        mock_cv2: Any,
    ) -> None:
        """Test finishing video recording."""
        model, data = mock_mujoco
        mock_mj = _make_mock_mj()

        with (
            patch(_VID_EXPORT_MJ, mock_mj),
            patch(
                "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.CV2_AVAILABLE",
                True,
            ),
        ):
            exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.MP4)
            exporter.start_recording("test.mp4")
            writer = exporter.writer
            exporter.finish_recording()

        writer.release.assert_called_once()
        assert exporter.writer is None

    def test_finish_recording_gif(
        self,
        mock_mujoco: tuple[MagicMock, MagicMock],
        mock_imageio: Any,
    ) -> None:
        """Test finishing GIF recording."""
        model, data = mock_mujoco
        mock_mj = _make_mock_mj()

        with (
            patch(_VID_EXPORT_MJ, mock_mj),
            patch(
                "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.IMAGEIO_AVAILABLE",
                True,
            ),
        ):
            exporter = VideoExporter(model, data, WIDTH, HEIGHT, FPS, VideoFormat.GIF)
            exporter.start_recording("test.gif")
            exporter.add_frame()
            exporter.finish_recording("test.gif")

        mock_imageio.mimsave.assert_called_once()
        assert exporter.frames == []

    def test_export_recording(
        self,
        mock_mujoco: tuple[MagicMock, MagicMock],
        mock_cv2: Any,
    ) -> None:
        """Test full export recording workflow."""
        model, data = mock_mujoco
        mock_mj = _make_mock_mj()

        initial_state = np.zeros(4)  # nq=2 + nv=2

        def control_func(t: float) -> np.ndarray:
            """Return zero control for the given time."""
            return np.array([0.0])

        def progress_cb(current: int, total: int) -> None:
            """No-op progress callback."""

        with (
            patch(_VID_EXPORT_MJ, mock_mj),
            patch(
                "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.CV2_AVAILABLE",
                True,
            ),
        ):
            success = exporter = VideoExporter(
                model, data, WIDTH, HEIGHT, FPS, VideoFormat.MP4
            )
            success = exporter.export_recording(
                "test.mp4",
                initial_state,
                control_func,
                duration=0.1,
                progress_callback=progress_cb,
            )

        assert success
        assert exporter.frame_count > 0
        mock_cv2.VideoWriter.return_value.release.assert_called_once()

    def test_metrics_overlay(
        self,
        mock_mujoco: tuple[MagicMock, MagicMock],
        mock_cv2: Any,
    ) -> None:
        """Test metrics overlay rendering on frames."""
        model, data = mock_mujoco
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        metrics = {"Test Metric": lambda d: 42.0}

        with patch(
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.CV2_AVAILABLE",
            True,
        ):
            out_frame = create_metrics_overlay(frame, 1.0, data, metrics)

        assert mock_cv2.putText.call_count >= 2
        assert out_frame is not frame

    def test_export_simulation_video_function(
        self,
        mock_mujoco: tuple[MagicMock, MagicMock],
        mock_cv2: Any,
    ) -> None:
        """Test the export_simulation_video convenience function."""
        model, data = mock_mujoco
        mock_mj = _make_mock_mj()

        N = 10
        states = np.zeros((N, 4))
        controls = np.zeros((N, 1))
        times = np.linspace(0, 1, N)

        with (
            patch(_VID_EXPORT_MJ, mock_mj),
            patch(
                "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.video_export.CV2_AVAILABLE",
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
        mock_cv2.VideoWriter.assert_called_once()
