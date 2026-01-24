"""Video export module for golf swing animations.

This module provides professional video export capabilities:
- Multiple format support (MP4, AVI, GIF)
- Configurable resolution and frame rate
- Optional metric overlays
- Progress tracking
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any  # noqa: ICN003

import mujoco as mj
import numpy as np

from src.shared.python.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# Configure logging
logger = get_logger(__name__)


class VideoFormat(Enum):
    """Supported video formats."""

    MP4 = "mp4"
    AVI = "avi"
    GIF = "gif"


class VideoResolution(Enum):
    """Standard video resolutions."""

    HD_720 = (1280, 720)
    HD_1080 = (1920, 1080)
    UHD_4K = (3840, 2160)
    CUSTOM = (0, 0)  # User-defined


class VideoExporter:
    """Export MuJoCo simulations as video files."""

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        width: int = 1920,
        height: int = 1080,
        fps: int = 60,
        format: VideoFormat = VideoFormat.MP4,
    ) -> None:
        """Initialize video exporter.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            format: Output video format
        """
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.fps = fps
        self.format = format

        # Create rendering context
        self.renderer = mj.Renderer(model, width=width, height=height)

        # Video writer
        self.writer: Any = None
        self.frames: list[np.ndarray] = []  # For GIF export
        self.frame_count = 0

    def start_recording(self, output_path: str, codec: str | None = None) -> bool:
        """Start video recording.

        Args:
            output_path: Output file path
            codec: Video codec (default: auto-detect from format)

        Returns:
            True if recording started successfully
        """
        output_path_obj = Path(output_path)

        # Auto-detect codec if not specified
        if codec is None:
            if self.format == VideoFormat.MP4:
                codec = "mp4v"  # or 'avc1' for H.264
            elif self.format == VideoFormat.AVI:
                codec = "XVID"

        try:
            if self.format == VideoFormat.GIF:
                # For GIF, accumulate frames in memory
                if not IMAGEIO_AVAILABLE:
                    msg = "imageio required for GIF export"
                    raise ImportError(msg)
                self.frames = []
            else:
                # For video formats, use OpenCV
                if not CV2_AVAILABLE:
                    msg = "opencv-python required for video export"
                    raise ImportError(msg)

                fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
                self.writer = cv2.VideoWriter(
                    str(output_path_obj),
                    fourcc,
                    self.fps,
                    (self.width, self.height),
                )

                if self.writer is not None and not self.writer.isOpened():
                    return False

            self.frame_count = 0
            return True

        except Exception as e:
            logger.error(f"Failed to start video recording: {e}")
            return False

    def add_frame(
        self,
        camera_id: int | None = None,
        overlay_callback: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Add a frame to the video.

        Args:
            camera_id: Camera ID to render (None = default)
            overlay_callback: Optional function to overlay metrics on frame
        """
        # Update renderer with current data
        self.renderer.update_scene(self.data, camera=camera_id)

        # Render frame
        frame = self.renderer.render()

        # Apply overlay if provided
        if overlay_callback is not None:
            frame = overlay_callback(frame)

        # Convert RGB to BGR for OpenCV
        if self.format != VideoFormat.GIF:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Add frame
        if self.format == VideoFormat.GIF:
            self.frames.append(frame)
        elif self.writer is not None:
            self.writer.write(frame)

        self.frame_count += 1

    def finish_recording(self, output_path: str | None = None) -> None:
        """Finish video recording and save file.

        Args:
            output_path: Output path (required for GIF)
        """
        if self.format == VideoFormat.GIF:
            if output_path and self.frames:
                imageio.mimsave(
                    output_path,
                    self.frames,  # type: ignore[arg-type]
                    fps=self.fps,
                    loop=0,  # Infinite loop
                )
                self.frames = []
        elif self.writer is not None:
            self.writer.release()
            self.writer = None

    def export_recording(
        self,
        output_path: str,
        initial_state: np.ndarray,
        control_function: Callable[[float], np.ndarray],
        duration: float,
        camera_id: int | None = None,
        overlay_callback: (
            Callable[[np.ndarray, float, mj.MjData], np.ndarray] | None
        ) = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> bool:
        """Export a complete simulation as video.

        Args:
            output_path: Output file path
            initial_state: Initial qpos and qvel
            control_function: Function that returns control given time
            duration: Simulation duration in seconds
            camera_id: Camera ID for rendering
            overlay_callback: Function to overlay metrics (frame, time, data) -> frame
            progress_callback: Function called with (current_frame, total_frames)

        Returns:
            True if export successful
        """
        # Reset to initial state
        nq = self.model.nq
        self.data.qpos[:] = initial_state[:nq]
        self.data.qvel[:] = initial_state[nq:]
        mj.mj_forward(self.model, self.data)

        # Calculate total frames
        total_frames = int(duration * self.fps)
        dt = 1.0 / self.fps

        # Start recording
        if not self.start_recording(output_path):
            return False

        try:
            for frame_idx in range(total_frames):
                t = frame_idx * dt

                # Apply control
                ctrl = control_function(t)
                self.data.ctrl[:] = ctrl

                # Step simulation
                mj.mj_step(self.model, self.data)

                # Create overlay function with captured time
                frame_overlay = None
                if overlay_callback is not None:

                    def frame_overlay(
                        frame: np.ndarray, time: float = t, data: mj.MjData = self.data
                    ) -> np.ndarray:
                        """Apply overlay callback with captured time and data."""
                        return overlay_callback(frame, time, data)

                # Add frame
                self.add_frame(camera_id, frame_overlay)

                # Report progress
                if progress_callback is not None:
                    progress_callback(frame_idx + 1, total_frames)

            # Finish
            self.finish_recording(output_path)
            return True

        except Exception as e:
            logger.error(f"Failed during video export: {e}")
            self.finish_recording()
            return False

    def __del__(self) -> None:
        """Cleanup."""
        if self.writer is not None:
            self.writer.release()


def create_metrics_overlay(
    frame: np.ndarray,
    time: float,
    data: mj.MjData,
    metrics: dict[str, Any],
    font_scale: float = 1.0,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Create overlay with metrics on frame.

    Args:
        frame: Original frame (RGB or BGR)
        time: Current simulation time
        data: MuJoCo data
        metrics: Dictionary of metric names and extraction functions
        font_scale: Font size scale
        color: Text color (RGB or BGR)

    Returns:
        Frame with overlaid metrics
    """
    if not CV2_AVAILABLE:
        return frame

    frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 2))

    y_offset = 30
    line_height = int(30 * font_scale)

    # Time
    cv2.putText(
        frame,
        f"Time: {time:.2f}s",
        (10, y_offset),
        font,
        font_scale,
        color,
        thickness,
    )
    y_offset += line_height

    # Custom metrics
    for name, extractor in metrics.items():
        try:
            value = extractor(data)
            if isinstance(value, int | float | np.number):
                text = f"{name}: {value:.2f}"
            else:
                text = f"{name}: {value}"

            cv2.putText(frame, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
        except Exception:
            # Skip metric if extraction fails
            pass

    return frame


def export_simulation_video(
    model: mj.MjModel,
    data: mj.MjData,
    output_path: str,
    recorded_states: np.ndarray,
    recorded_controls: np.ndarray,
    times: np.ndarray,
    width: int = 1920,
    height: int = 1080,
    fps: int = 60,
    camera_id: int | None = None,
    show_metrics: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> bool:
    """Export a recorded simulation as video.

    Args:
        model: MuJoCo model
        data: MuJoCo data (will be modified)
        output_path: Output file path
        recorded_states: Array of states (N x (nq+nv))
        recorded_controls: Array of controls (N x nu)
        times: Array of timestamps (N,)
        width: Video width
        height: Video height
        fps: Frames per second
        camera_id: Camera for rendering
        show_metrics: Whether to overlay metrics
        progress_callback: Progress callback function

    Returns:
        True if successful
    """
    # Determine format from extension
    ext = Path(output_path).suffix.lower()
    if ext == ".mp4":
        format = VideoFormat.MP4
    elif ext == ".avi":
        format = VideoFormat.AVI
    elif ext == ".gif":
        format = VideoFormat.GIF
    else:
        msg = f"Unsupported format: {ext}"
        raise ValueError(msg)

    # Create exporter
    exporter = VideoExporter(model, data, width, height, fps, format)

    # Start recording
    if not exporter.start_recording(output_path):
        return False

    try:
        nq = model.nq
        total_frames = len(recorded_states)

        for i in range(total_frames):
            # Set state
            data.qpos[:] = recorded_states[i, :nq]
            data.qvel[:] = recorded_states[i, nq:]
            data.ctrl[:] = recorded_controls[i]

            # Forward kinematics
            mj.mj_forward(model, data)

            # Create overlay
            overlay_fn = None
            if show_metrics:
                t = times[i]

                # Define metrics to display
                metrics = {
                    "Frame": lambda d, frame_num=i: frame_num,
                }

                # Add club head speed if available
                try:
                    club_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "club_head")
                    if club_id >= 0:
                        jacp = np.zeros((3, model.nv))
                        jacr = np.zeros((3, model.nv))
                        mj.mj_jacBody(model, data, jacp, jacr, club_id)
                        vel = jacp @ data.qvel
                        speed = np.linalg.norm(vel) * 2.237  # m/s to mph
                        metrics["Club Speed"] = lambda d, s=speed: int(s)  # type: ignore[assignment]
                except Exception:
                    # Ignore club speed if club head not found
                    pass

                def overlay_fn(
                    frame: np.ndarray,
                    time: float = t,
                    sim_data: mj.MjData = data,
                    frame_metrics: dict[str, Any] = metrics,
                ) -> np.ndarray:
                    """Create metrics overlay on frame."""
                    return create_metrics_overlay(
                        frame,
                        time,
                        sim_data,
                        frame_metrics,
                        font_scale=0.8,
                    )

            # Add frame
            exporter.add_frame(camera_id, overlay_fn)

            # Progress
            if progress_callback:
                progress_callback(i + 1, total_frames)

        # Finish
        exporter.finish_recording(output_path)
        return True

    except Exception as e:
        logger.error(f"Failed to export simulation video: {e}")
        exporter.finish_recording()
        return False
