"""Video export module for golf swing animations.

This module provides professional video export capabilities:
- Multiple format support (MP4, AVI, GIF)
- Configurable resolution and frame rate
- Optional metric overlays
- Progress tracking
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any  # noqa: ICN003

import mujoco as mj
import numpy as np

from src.shared.python.core.constants import DEFAULT_FPS, HD_HEIGHT, HD_WIDTH
from src.shared.python.logging_pkg.logging_config import get_logger

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
    HD_1080 = (HD_WIDTH, HD_HEIGHT)
    UHD_4K = (3840, 2160)
    CUSTOM = (0, 0)  # User-defined


class VideoExporter:
    """Export MuJoCo simulations as video files."""

    def __init__(  # noqa: PLR0913
        self,
        model: mj.MjModel,
        data: mj.MjData,
        width: int = HD_WIDTH,
        height: int = HD_HEIGHT,
        fps: int = DEFAULT_FPS,
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
        if model is None:
            raise ValueError("model must be provided")
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
        if output_path is None:
            raise ValueError("output_path must be provided")
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

                if codec is None:
                    raise ValueError(f"no default codec for {self.format.value}")
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

        except (RuntimeError, ValueError, OSError) as e:
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
        self.renderer.update_scene(
            self.data, camera=-1 if camera_id is None else camera_id
        )

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

    def export_recording(  # noqa: PLR0913
        self,
        output_path: str,
        initial_state: np.ndarray,
        control_function: Callable[[float], np.ndarray],
        duration: float,
        camera_id: int | None = None,
        overlay_callback: (
            Callable[[np.ndarray, float, mj.MjData], np.ndarray] | None
        ) = None,  # noqa: E501
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
        if output_path is None:
            raise ValueError("output_path must be provided")
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

        except (ValueError, TypeError, RuntimeError) as e:
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
    if frame is None:
        raise ValueError("frame must be provided")
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
        except (ValueError, TypeError, RuntimeError):
            # Skip metric if extraction fails
            pass

    return frame


def _detect_video_format(output_path: str) -> VideoFormat:
    ext = Path(output_path).suffix.lower()
    if ext == ".mp4":
        return VideoFormat.MP4
    if ext == ".avi":
        return VideoFormat.AVI
    if ext == ".gif":
        return VideoFormat.GIF
    msg = f"Unsupported format: {ext}"
    raise ValueError(msg)


def _build_frame_metrics(
    model: mj.MjModel,
    data: mj.MjData,
    frame_idx: int,
) -> dict[str, Any]:
    if model is None:
        raise ValueError("model must be provided")
    metrics: dict[str, Any] = {
        "Frame": lambda d, frame_num=frame_idx: frame_num,
    }

    try:
        club_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "club_head")
        if club_id >= 0:
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mj.mj_jacBody(model, data, jacp, jacr, club_id)
            vel = jacp @ data.qvel
            speed = (
                math.hypot(vel[0], vel[1], vel[2]) * 2.237
            )  # ⚡ Bolt: math.hypot is significantly faster than np.linalg.norm for small 3D arrays  # noqa: E501  # m/s to mph
            metrics["Club Speed"] = lambda d, s=speed: int(s)  # type: ignore[assignment]
    except (ValueError, TypeError, RuntimeError):
        pass

    return metrics


@dataclass(frozen=True)
class VideoExportOptions:
    """Rendering options for :func:`export_simulation_video`.

    Invariant (checked on construction): ``width``, ``height`` and ``fps`` are
    positive integers, ``camera_id`` is ``None`` (free camera) or ``>= -1``,
    and ``progress_callback`` is ``None`` or callable.
    """

    width: int = HD_WIDTH
    height: int = HD_HEIGHT
    fps: int = DEFAULT_FPS
    camera_id: int | None = None
    show_metrics: bool = True
    progress_callback: Callable[[int, int], None] | None = None

    def __post_init__(self) -> None:
        for name in ("width", "height", "fps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.camera_id is not None and self.camera_id < -1:
            raise ValueError(f"camera_id must be None or >= -1, got {self.camera_id}")
        if self.progress_callback is not None and not callable(self.progress_callback):
            raise TypeError("progress_callback must be callable or None")


def export_simulation_video(
    model: mj.MjModel,
    data: mj.MjData,
    output_path: str,
    recorded_states: np.ndarray,
    recorded_controls: np.ndarray,
    times: np.ndarray,
    options: VideoExportOptions | None = None,
) -> bool:
    """Export a recorded simulation as video.

    Args:
        model: MuJoCo model
        data: MuJoCo data (will be modified)
        output_path: Output file path
        recorded_states: Array of states (N x (nq+nv))
        recorded_controls: Array of controls (N x nu)
        times: Array of timestamps (N,)
        options: Size, frame rate, camera, overlay and progress settings
            (defaults to :class:`VideoExportOptions`).

    Returns:
        True if successful
    """
    if model is None:
        raise ValueError("model must be provided")
    opts = VideoExportOptions() if options is None else options
    format = _detect_video_format(output_path)
    exporter = VideoExporter(model, data, opts.width, opts.height, opts.fps, format)

    if not exporter.start_recording(output_path):
        return False

    try:
        nq = model.nq
        total_frames = len(recorded_states)

        for i in range(total_frames):
            data.qpos[:] = recorded_states[i, :nq]
            data.qvel[:] = recorded_states[i, nq:]
            data.ctrl[:] = recorded_controls[i]
            mj.mj_forward(model, data)

            overlay_fn = None
            if opts.show_metrics:
                t = times[i]
                metrics = _build_frame_metrics(model, data, i)

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

            exporter.add_frame(opts.camera_id, overlay_fn)

            if opts.progress_callback:
                opts.progress_callback(i + 1, total_frames)

        exporter.finish_recording(output_path)
        return True

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Failed to export simulation video: {e}")
        exporter.finish_recording()
        return False


def _recorded_states(
    model: mj.MjModel, recorder: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(states, controls, times)`` arrays for a recorded swing.

    Raises:
        ValueError: nothing was recorded, or a frame does not match ``model``.
    """
    frames = list(getattr(recorder, "frames", []))
    if len(frames) < 1:
        raise ValueError(
            "Nothing is recorded: press Start Recording, run the simulation, "
            "stop, then export."
        )
    states = np.zeros((len(frames), model.nq + model.nv))
    controls = np.zeros((len(frames), model.nu))
    for i, frame in enumerate(frames):
        pos = np.asarray(frame.joint_positions, dtype=float)
        vel = np.asarray(frame.joint_velocities, dtype=float)
        if pos.shape != (model.nq,) or vel.shape != (model.nv,):
            raise ValueError(
                f"Recorded frame {i} has {pos.size} joint positions and "
                f"{vel.size} velocities, but the loaded model has "
                f"nq={model.nq}, nv={model.nv}; record again with this model."
            )
        states[i] = np.concatenate([pos, vel])
    times = np.array([f.time for f in frames], dtype=float)
    return states, controls, times


def _fit_to_framebuffer(model: mj.MjModel, size: tuple[int, int]) -> tuple[int, int]:
    """Scale ``size`` down (aspect kept) to the model's offscreen framebuffer.

    MuJoCo refuses an image larger than ``<visual><global offwidth/offheight>``
    (640x480 by default), so the HD default must shrink for stock models.
    """
    width, height = size
    offscreen = model.vis.global_
    scale = min(1.0, offscreen.offwidth / width, offscreen.offheight / height)
    # Even dimensions: most H.264 encoders reject odd frame sizes.
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def export_recorded_swing_video(
    model: mj.MjModel,
    data: mj.MjData,
    recorder: Any,
    output_path: str | Path,
    *,
    size: tuple[int, int] = (HD_WIDTH, HD_HEIGHT),
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Render a ``SwingRecorder`` recording to a video file (UI export action).

    The frame rate follows the recorded time step (1-60 fps), so playback is
    real time. Postcondition: the returned file exists and is non-empty.

    Raises:
        ValueError: nothing recorded, or the recording does not fit ``model``.
        RuntimeError: the video writer could not be opened or the render failed.
    """
    if model is None or data is None:
        raise ValueError("a loaded model is required to export video")
    states, controls, times = _recorded_states(model, recorder)
    steps = np.diff(times)
    step = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 0.0
    fps = int(min(60, max(1, round(1.0 / step)))) if step > 0 else DEFAULT_FPS
    width, height = _fit_to_framebuffer(model, size)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok = export_simulation_video(
        model,
        data,
        str(target),
        states,
        controls,
        times,
        VideoExportOptions(
            width=width,
            height=height,
            fps=fps,
            show_metrics=False,
            progress_callback=progress_callback,
        ),
    )
    if not ok or not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"video export to {target} failed; see the log for details")
    return target
