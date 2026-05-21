"""Type definitions for the video analyzer tool.

Provides the core data structures for video pose analysis. Kept
dependency-free so that tests can import and use them without requiring
MediaPipe or OpenCV to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Landmark:
    """A 3-D world-space landmark produced by a pose estimator.

    Args:
        x: Horizontal position in normalized image coordinates.
        y: Vertical position in normalized image coordinates.
        z: Depth estimate (sign convention: positive is closer).
        visibility: Model confidence that the landmark is visible.
    """

    x: float
    y: float
    z: float
    visibility: float = 0.0


@dataclass
class PoseFrame:
    """A single frame of pose estimation output.

    Args:
        frame_number: 0-indexed frame counter within the video.
        timestamp: Wall-clock timestamp in seconds since video start.
        landmarks: Ordered list of landmarks following the MediaPipe
            convention (index 0 = nose, indices 11-16 = arms, etc.).
        confidence: Overall frame-level pose confidence in [0, 1].
    """

    frame_number: int
    timestamp: float
    landmarks: list[Landmark] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class PostureMetrics:
    """Summary metrics derived from a swing's pose sequence.

    Args:
        head_stability: 0–100 score; 100 means no head movement detected.
        spine_tilt_deg: Spine tilt angle in degrees at address.
        hip_turn_deg: Peak hip rotation relative to address in degrees.
        shoulder_turn_deg: Peak shoulder turn relative to address.
    """

    head_stability: float = 100.0
    spine_tilt_deg: float = 0.0
    hip_turn_deg: float = 0.0
    shoulder_turn_deg: float = 0.0
