"""Smooth playback controller with frame interpolation for 60+ FPS animation.

Extracted from golf_gui_application.py for Single Responsibility Principle.
"""

from __future__ import annotations

from copy import copy

import numpy as np
from golf_data_core import FrameData, FrameProcessor
from PyQt6.QtCore import (
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    pyqtProperty,
    pyqtSignal,
)


class SmoothPlaybackController(QObject):
    """Smooth playback controller with frame interpolation for 60+ FPS animation.

    Features:
    - VSync-synchronized rendering (60+ FPS)
    - Frame interpolation for smooth motion between keyframes
    - Variable playback speed
    - Scrubbing support
    """

    # Signals
    frameUpdated = pyqtSignal(FrameData)  # Emits interpolated frame data
    positionChanged = pyqtSignal(float)  # Emits current position (0.0 to total_frames)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Frame data
        self.frame_processor: FrameProcessor | None = None
        self._current_position: float = 0.0
        self._playback_speed: float = 1.0

        # Animation
        self.animation = QPropertyAnimation(self, b"position")
        self.animation.setEasingCurve(QEasingCurve.Type.Linear)
        self.animation.valueChanged.connect(self._on_position_changed)
        self.animation.finished.connect(self._on_animation_finished)

        # State
        self.is_playing = False
        self.loop_playback = True  # Default to looping

    def load_frame_processor(self, frame_processor: FrameProcessor) -> None:
        """Load frame processor with motion data."""
        self.frame_processor = frame_processor
        self.stop()
        self.seek(0.0)

    # ========================================================================
    # Position Property (for QPropertyAnimation)
    # ========================================================================

    @pyqtProperty(float)
    def position(self) -> float:
        """Current playback position (0.0 to total_frames - 1)."""
        return self._current_position

    @position.setter
    def position(self, value: float) -> None:
        """Set playback position with interpolation."""
        if self.frame_processor is None:
            return

        total_frames = len(self.frame_processor.time_vector)
        self._current_position = np.clip(value, 0.0, total_frames - 1)
        self.positionChanged.emit(self._current_position)

        # Interpolate frame data
        interpolated_frame = self._get_interpolated_frame(self._current_position)
        self.frameUpdated.emit(interpolated_frame)

    # ========================================================================
    # Playback Control
    # ========================================================================

    def play(self) -> None:
        """Start smooth playback."""
        if self.frame_processor is None:
            return

        if self.is_playing:
            return  # Already playing

        total_frames = len(self.frame_processor.time_vector)

        # Calculate duration based on actual data time span
        start_pos = self._current_position
        end_pos = total_frames - 1

        if start_pos >= end_pos - 0.1:  # Near end, restart from beginning
            start_pos = 0.0
            self.seek(0.0)

        # Duration in milliseconds (maintain original timing)
        frame_time_ms = 33.33  # ~30 FPS from motion capture
        duration_ms = int((end_pos - start_pos) * frame_time_ms / self._playback_speed)

        # Setup animation
        self.animation.setStartValue(start_pos)
        self.animation.setEndValue(end_pos)
        self.animation.setDuration(duration_ms)
        self.animation.start()

        self.is_playing = True

    def pause(self) -> None:
        """Pause playback."""
        if not self.is_playing:
            return

        self.animation.pause()
        self.is_playing = False

    def stop(self) -> None:
        """Stop playback and reset to beginning."""
        self.animation.stop()
        self.is_playing = False
        self.seek(0.0)

    def toggle_playback(self) -> None:
        """Toggle between play and pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def seek(self, position: float) -> None:
        """Seek to specific frame position."""
        if self.frame_processor is None:
            return

        was_playing = self.is_playing

        if was_playing:
            self.animation.stop()

        self.position = position

        if was_playing:
            self.play()

    def set_playback_speed(self, speed: float) -> None:
        """Set playback speed multiplier (0.5 = half speed, 2.0 = double speed)."""
        self._playback_speed = np.clip(speed, 0.1, 10.0)

        # If playing, restart with new speed
        if self.is_playing:
            current_pos = self._current_position
            self.pause()
            self.seek(current_pos)
            self.play()

    # ========================================================================
    # Frame Interpolation (The Magic!)
    # ========================================================================

    def _get_interpolated_frame(self, position: float) -> FrameData:
        """Get interpolated frame data at fractional position.

        For example:
        - position = 5.0 -> Frame 5 exactly
        - position = 5.7 -> 70% between frame 5 and 6

        This creates smooth motion between keyframes!
        """
        if self.frame_processor is None:
            raise ValueError("No frame processor loaded")

        total_frames = len(self.frame_processor.time_vector)

        # Clamp position
        position = np.clip(position, 0.0, total_frames - 1)

        # Get integer frame indices
        low_idx = int(np.floor(position))
        high_idx = min(low_idx + 1, total_frames - 1)

        # Calculate interpolation factor (0.0 to 1.0)
        t = position - low_idx

        # Get frames at integer indices
        frame_low = self.frame_processor.get_frame_data(low_idx)
        frame_high = self.frame_processor.get_frame_data(high_idx)

        # Interpolate all positions
        return self._lerp_frame_data(frame_low, frame_high, t)

    @staticmethod
    def _lerp_frame_data(frame_a: FrameData, frame_b: FrameData, t: float) -> FrameData:
        """Linear interpolation between two frames.

        Args:
            frame_a: Starting frame
            frame_b: Ending frame
            t: Interpolation factor (0.0 = frame_a, 1.0 = frame_b)

        Returns:
            Interpolated frame data
        """
        result = copy(frame_a)

        # List of all position attributes to interpolate
        position_attrs = [
            "left_wrist",
            "left_elbow",
            "left_shoulder",
            "right_wrist",
            "right_elbow",
            "right_shoulder",
            "hub",
            "butt",
            "clubhead",
            "midpoint",
        ]

        # Lerp each position: result = a * (1 - t) + b * t
        for attr in position_attrs:
            if not hasattr(frame_a, attr) or not hasattr(frame_b, attr):
                continue

            pos_a = getattr(frame_a, attr)
            pos_b = getattr(frame_b, attr)

            # Check for valid data
            if np.isfinite(pos_a).all() and np.isfinite(pos_b).all():
                interpolated_pos = pos_a * (1.0 - t) + pos_b * t
                setattr(result, attr, interpolated_pos)

        # Interpolate forces
        result.forces = {}
        if hasattr(frame_a, "forces") and hasattr(frame_b, "forces"):
            for key in frame_a.forces:
                if key in frame_b.forces:
                    f_a = frame_a.forces[key]
                    f_b = frame_b.forces[key]
                    if np.isfinite(f_a).all() and np.isfinite(f_b).all():
                        result.forces[key] = f_a * (1.0 - t) + f_b * t
                    else:
                        result.forces[key] = f_a

        # Interpolate torques
        result.torques = {}
        if hasattr(frame_a, "torques") and hasattr(frame_b, "torques"):
            for key in frame_a.torques:
                if key in frame_b.torques:
                    t_a = frame_a.torques[key]
                    t_b = frame_b.torques[key]
                    if np.isfinite(t_a).all() and np.isfinite(t_b).all():
                        result.torques[key] = t_a * (1.0 - t) + t_b * t
                    else:
                        result.torques[key] = t_a

        return result

    # ========================================================================
    # Internal Callbacks
    # ========================================================================

    def _on_position_changed(self, value: float) -> None:
        """Called by QPropertyAnimation on every frame update."""
        # Position property setter handles the interpolation

    def _on_animation_finished(self) -> None:
        """Called when animation completes."""
        self.is_playing = False

        if self.loop_playback:
            self.seek(0.0)
            self.play()
