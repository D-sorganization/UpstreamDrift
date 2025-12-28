"""Playback control for recorded simulations.

Provides:
- Frame-by-frame stepping
- Variable playback speed
- Timeline scrubbing
- Loop control
"""

from collections.abc import Callable
from enum import Enum

import cv2
import numpy as np


class PlaybackMode(Enum):
    """Playback modes."""

    STOPPED = 0
    PLAYING = 1
    PAUSED = 2


class PlaybackController:
    """Control playback of recorded simulation data."""

    def __init__(
        self,
        times: np.ndarray,
        states: np.ndarray,
        controls: np.ndarray,
    ) -> None:
        """Initialize playback controller.

        Args:
            times: Time array (N,)
            states: State array (N, nq+nv)
            controls: Control array (N, nu)
        """
        self.times = times
        self.states = states
        self.controls = controls

        self.num_frames = len(times)
        self.duration = times[-1] - times[0] if len(times) > 1 else 0.0

        # Playback state
        self.current_frame = 0
        self.mode = PlaybackMode.STOPPED
        self.speed = 1.0  # Playback speed multiplier
        self.loop = False

        # Callbacks
        self.on_frame_changed: Callable[[int], None] | None = None
        self.on_playback_finished: Callable[[], None] | None = None

    def play(self) -> None:
        """Start playback."""
        self.mode = PlaybackMode.PLAYING

    def pause(self) -> None:
        """Pause playback."""
        self.mode = PlaybackMode.PAUSED

    def stop(self) -> None:
        """Stop playback and reset to start."""
        self.mode = PlaybackMode.STOPPED
        self.seek_to_frame(0)

    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self.mode == PlaybackMode.PLAYING

    def step_forward(self, num_frames: int = 1) -> None:
        """Step forward by frames.

        Args:
            num_frames: Number of frames to step
        """
        new_frame = min(self.current_frame + num_frames, self.num_frames - 1)
        self.seek_to_frame(new_frame)

    def step_backward(self, num_frames: int = 1) -> None:
        """Step backward by frames.

        Args:
            num_frames: Number of frames to step
        """
        new_frame = max(self.current_frame - num_frames, 0)
        self.seek_to_frame(new_frame)

    def seek_to_frame(self, frame: int) -> None:
        """Seek to specific frame.

        Args:
            frame: Frame index (0 to num_frames-1)
        """
        frame = max(0, min(frame, self.num_frames - 1))
        if frame != self.current_frame:
            self.current_frame = frame
            if self.on_frame_changed:
                self.on_frame_changed(frame)

    def seek_to_time(self, time: float) -> None:
        """Seek to specific time.

        Args:
            time: Time in seconds
        """
        # Find closest frame
        frame = np.argmin(np.abs(self.times - time))
        self.seek_to_frame(int(frame))

    def seek_to_percent(self, percent: float) -> None:
        """Seek to percentage of total duration.

        Args:
            percent: Percentage (0.0 to 100.0)
        """
        frame = int((percent / 100.0) * (self.num_frames - 1))
        self.seek_to_frame(frame)

    def set_speed(self, speed: float) -> None:
        """Set playback speed.

        Args:
            speed: Speed multiplier (0.1 to 10.0)
                  1.0 = normal speed
                  0.5 = half speed
                  2.0 = double speed
        """
        self.speed = max(0.1, min(speed, 10.0))

    def set_loop(self, loop: bool) -> None:
        """Enable/disable looping.

        Args:
            loop: Whether to loop playback
        """
        self.loop = loop

    def update(self, dt: float) -> bool:
        """Update playback state.

        Call this at regular intervals (e.g., 60 Hz).

        Args:
            dt: Time step in seconds (real time)

        Returns:
            True if frame changed
        """
        if self.mode != PlaybackMode.PLAYING:
            return False

        if self.num_frames <= 1:
            return False

        # Calculate frame advance based on speed and dt
        # Average frame time in recording
        if self.duration > 0:
            avg_frame_time = self.duration / (self.num_frames - 1)
            # Frames to advance (accounting for speed)
            frames_to_advance = (dt * self.speed) / avg_frame_time
        else:
            # Zero duration (e.g. single frame or identical timestamps)
            # Advance 1 frame per update or handled by num_frames check above
            # If we are here, num_frames > 1 but duration is 0.
            # This implies infinite frame rate.
            frames_to_advance = 1.0  # Fallback

        # Use fractional accumulation for smooth playback
        if not hasattr(self, "_frame_accumulator"):
            self._frame_accumulator = 0.0

        self._frame_accumulator += frames_to_advance
        frames = int(self._frame_accumulator)
        self._frame_accumulator -= frames

        if frames > 0:
            new_frame = self.current_frame + frames

            if new_frame >= self.num_frames:
                if self.loop:
                    new_frame = new_frame % self.num_frames
                    self._frame_accumulator = 0.0
                else:
                    new_frame = self.num_frames - 1
                    self.pause()
                    if self.on_playback_finished:
                        self.on_playback_finished()

            self.seek_to_frame(new_frame)
            return True

        return False

    def get_current_state(self) -> tuple:
        """Get current state and control.

        Returns:
            (state, control, time) tuple
        """
        state = self.states[self.current_frame]
        control = self.controls[self.current_frame]
        time = self.times[self.current_frame]
        return (state, control, time)

    def get_current_time(self) -> float:
        """Get current playback time."""
        return float(self.times[self.current_frame])

    def get_current_frame(self) -> int:
        """Get current frame index."""
        return self.current_frame

    def get_progress_percent(self) -> float:
        """Get playback progress as percentage.

        Returns:
            Progress (0.0 to 100.0)
        """
        if self.num_frames <= 1:
            return 0.0
        return (self.current_frame / (self.num_frames - 1)) * 100.0

    def get_info(self) -> dict:
        """Get playback information.

        Returns:
            Dictionary with playback info
        """
        return {
            "current_frame": self.current_frame,
            "total_frames": self.num_frames,
            "current_time": self.get_current_time(),
            "duration": self.duration,
            "progress_percent": self.get_progress_percent(),
            "mode": self.mode.name,
            "speed": self.speed,
            "loop": self.loop,
        }

    def export_frame_as_image(
        self,
        frame: int,
        output_path: str,
        render_callback: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        """Export a specific frame as image.

        Args:
            frame: Frame index
            output_path: Output image path
            render_callback: Function that takes (state, control) and returns RGB image
        """
        frame = max(0, min(frame, self.num_frames - 1))
        state = self.states[frame]
        control = self.controls[frame]

        image = render_callback(state, control)

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, image_bgr)


class PlaybackSpeedPresets:
    """Common playback speed presets."""

    VERY_SLOW = 0.1
    SLOW = 0.25
    HALF = 0.5
    NORMAL = 1.0
    DOUBLE = 2.0
    FAST = 4.0
    VERY_FAST = 10.0

    @classmethod
    def get_all_presets(cls) -> list:
        """Get all preset speeds."""
        return [
            ("Very Slow (0.1x)", cls.VERY_SLOW),
            ("Slow (0.25x)", cls.SLOW),
            ("Half Speed (0.5x)", cls.HALF),
            ("Normal (1.0x)", cls.NORMAL),
            ("Double (2.0x)", cls.DOUBLE),
            ("Fast (4.0x)", cls.FAST),
            ("Very Fast (10.0x)", cls.VERY_FAST),
        ]
