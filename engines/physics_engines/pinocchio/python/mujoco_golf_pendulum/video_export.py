"""Video export module for golf swing animations.

This module provides professional video export capabilities:
- Multiple format support (MP4, AVI, GIF)
- Configurable resolution and frame rate
- Optional metric overlays
- Progress tracking
"""

from __future__ import annotations

import logging
import typing
from enum import Enum
from pathlib import Path
from typing import Final

import mujoco as mj

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


# Configure logging
LOGGER = logging.getLogger(__name__)

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


# Conversion factor: m/s to mph
# 1 m/s = 2.23694 mph
MPS_TO_MPH: Final[float] = 2.23694


class VideoExporter:
    """Export MuJoCo simulations as video files."""

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        width: int = 1920,
        height: int = 1080,
        fps: int = 60,
        video_format: VideoFormat = VideoFormat.MP4,
    ) -> None:
        """Initialize video exporter.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            video_format: Output video format
        """
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.fps = fps
        self.format = video_format

        # Create rendering context
        self.renderer = mj.Renderer(model, width=width, height=height)

        # Video writer
        self.writer: typing.Any = None
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

        # Check dependencies before try block
        if self.format == VideoFormat.GIF:
            if not IMAGEIO_AVAILABLE:
                msg = "imageio required for GIF export"
                LOGGER.error(msg)
                raise ImportError(msg)
        elif not CV2_AVAILABLE:
            msg = "opencv-python required for video export"
            LOGGER.error(msg)
            raise ImportError(msg)

        try:
            if self.format == VideoFormat.GIF:
                # For GIF, accumulate frames in memory
                self.frames = []
            else:
                # For video formats, use OpenCV
                fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
                self.writer = cv2.VideoWriter(
                    str(output_path_obj),
                    fourcc,
                    self.fps,
                    (self.width, self.height),
                )

                if self.writer is not None and not self.writer.isOpened():
                    LOGGER.error("Failed to open video writer.")
                    return False

            self.frame_count = 0
            return True

        except Exception:
            LOGGER.exception("Error starting recording.")
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
                frame_overlay_fn = None
                if overlay_callback is not None:

                    def frame_overlay(
                        frame: np.ndarray,
                        t_val: float = t,
                        data_val: mj.MjData = self.data,
                    ) -> np.ndarray:
                        """Apply overlay callback with captured time and data."""
                        return overlay_callback(frame, t_val, data_val)

                    frame_overlay_fn = frame_overlay

                # Add frame
                self.add_frame(camera_id, frame_overlay_fn)

                # Report progress
                if progress_callback is not None:
                    progress_callback(frame_idx + 1, total_frames)

            # Finish
            self.finish_recording(output_path)
            return True

        except Exception:
            LOGGER.exception("Error during recording export.")
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
    metrics: dict,
    font_scale: float = 1.0,
    color: tuple = (255, 255, 255),
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
            text = (
                f"{name}: {value:.2f}"
                if isinstance(value, int | float)
                else f"{name}: {value}"
            )

            cv2.putText(frame, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
        except Exception:
            LOGGER.debug("Failed to extract metric: %s", name)
            # Continue to next metric

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
        video_format = VideoFormat.MP4
    elif ext == ".avi":
        video_format = VideoFormat.AVI
    elif ext == ".gif":
        video_format = VideoFormat.GIF
    else:
        msg = f"Unsupported format: {ext}"
        raise ValueError(msg)

    # Create exporter
    exporter = VideoExporter(model, data, width, height, fps, video_format)

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
                metrics = _setup_metrics_for_frame(model, data, i)

                def frame_bg_fn(
                    frame: np.ndarray,
                    t_val: float = t,
                    m_val: dict = metrics,
                    d_val: mj.MjData = data,
                ) -> np.ndarray:
                    """Create metrics overlay on frame."""
                    return create_metrics_overlay(
                        frame,
                        t_val,
                        d_val,
                        m_val,
                        font_scale=0.8,
                    )

                overlay_fn = frame_bg_fn

            # Add frame
            exporter.add_frame(camera_id, overlay_fn)

            # Progress
            if progress_callback:
                progress_callback(i + 1, total_frames)

        # Finish
        exporter.finish_recording(output_path)
        return True

    except Exception:
        LOGGER.exception("Error exporting simulation video.")
        exporter.finish_recording()
        return False


def _setup_metrics_for_frame(
    model: mj.MjModel, data: mj.MjData, i: int
) -> dict[str, typing.Any]:
    """Setup metrics dictionary for a single frame.

    Returns:
        dict: {
            "Frame": int,
            "Club Head Speed (m/s)": float,
            "Club Head Speed (mph)": float,
        }

    Club head speed is calculated as the Euclidean norm of the club\
    head's linear velocity.
    Conversion: 1 m/s = 2.23694 mph (NIST SP 811, 2008, Table 8.\
    Unit conversion factors).
    """
    # The name of the club head body in the model. This must match\
    # the model's definition.
    club_head_body_name: Final[str] = "club_head"
    try:
        club_head_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, club_head_body_name)
        if club_head_id < 0:
            # Body not found, return NaNs
            club_head_speed_mps = float("nan")
            club_head_speed_mph = float("nan")
        else:
            # mj.MjData.cvel is (nbody, 6): [linear(3), angular(3)]
            # velocities in global frame
            # mj.MjData.cvel is (nbody, 6): [linear(3), angular(3)]\
            # velocities in global frame
            club_head_cvel = data.cvel[club_head_id]  # shape (6,)
            club_head_linear_vel = club_head_cvel[:3]  # [vx, vy, vz] in m/s

            # Compute Euclidean norm (speed)
            club_head_speed_mps = float((club_head_linear_vel**2).sum() ** 0.5)
            # Use defined constant for conversion
            club_head_speed_mph = club_head_speed_mps * MPS_TO_MPH

    except Exception:
        # If any error occurs during calculation, log it and return NaNs
        LOGGER.exception("Could not compute club head speed for frame %d", i)
        club_head_speed_mps = float("nan")
        club_head_speed_mph = float("nan")

    return {
        "Frame": lambda _, idx=i: idx,
        "Club Head Speed (m/s)": lambda _, v=club_head_speed_mps: v,
        "Club Head Speed (mph)": lambda _, v=club_head_speed_mph: v,
    }
