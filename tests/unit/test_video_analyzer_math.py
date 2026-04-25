"""Tests for math.hypot usage in SwingAnalyzer._calculate_angle.

Validates numerical correctness of the angle calculation after replacing
manual math.sqrt(x**2 + y**2 + z**2) expressions with math.hypot.
"""

from __future__ import annotations

import math

import pytest

from src.tools.video_analyzer.types import Landmark

pytestmark = pytest.mark.unit


def _make_landmark(x: float, y: float, z: float) -> Landmark:
    return Landmark(x=x, y=y, z=z)


class TestCalculateAngle:
    """Verify _calculate_angle is numerically correct after math.hypot migration."""

    @pytest.fixture()
    def analyzer(self):  # type: ignore[return]
        """Create a SwingAnalyzer without mediapipe dependency."""
        from src.tools.video_analyzer.analyzer import SwingAnalyzer

        return SwingAnalyzer()

    def test_right_angle(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        """Three points forming a 90-degree angle at B."""
        a = _make_landmark(1.0, 0.0, 0.0)
        b = _make_landmark(0.0, 0.0, 0.0)
        c = _make_landmark(0.0, 1.0, 0.0)
        angle = analyzer._calculate_angle(a, b, c)
        assert abs(angle - 90.0) < 1e-6

    def test_straight_line(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        """Points collinear in opposite directions → 180 degrees."""
        a = _make_landmark(-1.0, 0.0, 0.0)
        b = _make_landmark(0.0, 0.0, 0.0)
        c = _make_landmark(1.0, 0.0, 0.0)
        angle = analyzer._calculate_angle(a, b, c)
        assert abs(angle - 180.0) < 1e-6

    def test_zero_length_vector_returns_zero(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        """If A == B (zero-length vector), the function must return 0."""
        a = _make_landmark(0.0, 0.0, 0.0)
        b = _make_landmark(0.0, 0.0, 0.0)
        c = _make_landmark(1.0, 0.0, 0.0)
        angle = analyzer._calculate_angle(a, b, c)
        assert angle == 0

    def test_3d_diagonal(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        """Angle in 3-D: vectors along (1,1,0) and (0,1,0) → 45 degrees."""
        a = _make_landmark(1.0, 1.0, 0.0)
        b = _make_landmark(0.0, 0.0, 0.0)
        c = _make_landmark(0.0, 1.0, 0.0)
        angle = analyzer._calculate_angle(a, b, c)
        expected = math.degrees(math.acos(1.0 / math.sqrt(2)))
        assert abs(angle - expected) < 1e-5

    def test_head_stability_zero_movement(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        """Smoke-test _calculate_posture path: no nose movement → 100% stability."""
        from src.tools.video_analyzer.types import PoseFrame

        nose = _make_landmark(0.5, 0.5, 0.0)
        dummy_landmarks = [nose] * 33
        address_pose = PoseFrame(
            frame_number=0,
            timestamp=0.0,
            landmarks=dummy_landmarks,
            confidence=1.0,
        )
        posture = analyzer._calculate_posture(
            [address_pose],
            key_frames={"address": 0},
            stance=None,  # type: ignore[arg-type]
        )
        # Single pose → head_stability == 100
        assert posture.head_stability == pytest.approx(100.0, abs=1.0)
