"""Swing analyzer — pose-based golf swing assessment.

All heavy mediapipe / OpenCV imports are deferred so that the module
can be imported and its math methods tested in a headless environment
without any external dependencies installed.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .types import Landmark, PoseFrame, PostureMetrics

logger = logging.getLogger(__name__)

# MediaPipe nose landmark index (matches the 33-point BlazePose model).
_MEDIAPIPE_NOSE_IDX = 0


class SwingAnalyzer:
    """Analyse golf swing kinematics from a sequence of :class:`PoseFrame` objects.

    All pose-estimation is delegated to MediaPipe (optional runtime
    dependency).  Every *math* method is available without MediaPipe
    so that unit tests can exercise the numerical code paths independently.
    """

    def _calculate_angle(
        self,
        a: Landmark,
        b: Landmark,
        c: Landmark,
    ) -> float:
        """Return the angle at joint *b* formed by the three landmarks.

        Uses :func:`math.hypot` for numerically stable magnitude computation.

        Args:
            a: Proximal landmark.
            b: Vertex / joint landmark.
            c: Distal landmark.

        Returns:
            Angle in degrees in [0, 180].  Returns 0 if either vector is
            degenerate (zero-length), rather than raising.
        """
        ba = (a.x - b.x, a.y - b.y, a.z - b.z)
        bc = (c.x - b.x, c.y - b.y, c.z - b.z)

        mag_ba = math.hypot(*ba)
        mag_bc = math.hypot(*bc)

        if mag_ba == 0.0 or mag_bc == 0.0:
            return 0.0

        dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
        # Clamp to [-1, 1] to guard against floating-point rounding outside acos domain.
        cos_theta = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cos_theta))

    def _calculate_posture(
        self,
        frames: list[PoseFrame],
        key_frames: dict[str, int],
        stance: Any,
    ) -> PostureMetrics:
        """Derive posture metrics from a list of :class:`PoseFrame` objects.

        Args:
            frames: Ordered pose frames covering the full swing.
            key_frames: Mapping of phase label (e.g. ``"address"``) to
                frame index within *frames*.
            stance: Optional stance descriptor (unused in the current
                implementation — reserved for future side/alignment logic).

        Returns:
            :class:`PostureMetrics` with at minimum ``head_stability`` set.
        """
        if not frames:
            return PostureMetrics(head_stability=100.0)

        return PostureMetrics(
            head_stability=self._compute_head_stability(frames, key_frames),
        )

    def _compute_head_stability(
        self,
        frames: list[PoseFrame],
        key_frames: dict[str, int],
    ) -> float:
        """Compute a 0-100 head-stability score from nose landmark drift.

        100 means the nose landmark did not move at all across the supplied
        frames; the score decreases proportionally with total drift magnitude.

        Args:
            frames: Ordered pose frames.
            key_frames: Mapping used to locate the address pose.  The
                ``"address"`` key is preferred; falls back to frame 0.

        Returns:
            Float in [0, 100].
        """
        address_idx = key_frames.get("address", 0)
        address_frame = frames[address_idx]

        if not address_frame.landmarks:
            return 100.0

        nose_0 = address_frame.landmarks[_MEDIAPIPE_NOSE_IDX]
        max_drift = 0.0
        for frame in frames:
            if not frame.landmarks:
                continue
            nose = frame.landmarks[_MEDIAPIPE_NOSE_IDX]
            drift = math.hypot(nose.x - nose_0.x, nose.y - nose_0.y, nose.z - nose_0.z)
            max_drift = max(max_drift, drift)

        # Map drift into a 0–100 score; 0.1 normalised units → 0% stability.
        _DRIFT_SCALE = 0.1
        stability = max(0.0, 100.0 * (1.0 - max_drift / _DRIFT_SCALE))
        return stability
