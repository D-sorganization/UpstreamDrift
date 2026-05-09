"""Tests for src.shared.python.injury.joint_stress (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.injury.joint_stress import (
    JointSide,
    JointStressAnalyzer,
    JointStressResult,
    StressType,
)


def _make_analyzer(body_weight: float = 80.0) -> JointStressAnalyzer:
    return JointStressAnalyzer(body_weight=body_weight)


_T = np.linspace(0.0, 1.0, 5)
_ANGLES: dict = {
    "hip_rotation_lead": np.array([0.0, 10.0, 20.0, 30.0, 25.0]),
    "hip_rotation_trail": np.array([0.0, 5.0, 10.0, 15.0, 12.0]),
    "hip_abduction_lead": np.array([0.0, 2.0, 5.0, 8.0, 6.0]),
    "hip_abduction_trail": np.array([0.0, 1.0, 3.0, 5.0, 4.0]),
    "shoulder_elevation_lead": np.array([30.0, 60.0, 90.0, 80.0, 70.0]),
    "shoulder_elevation_trail": np.array([20.0, 45.0, 70.0, 60.0, 50.0]),
    "shoulder_rotation_lead": np.array([0.0, 15.0, 30.0, 45.0, 40.0]),
    "shoulder_rotation_trail": np.array([0.0, 10.0, 20.0, 35.0, 30.0]),
    "elbow_flexion_lead": np.array([20.0, 40.0, 60.0, 80.0, 70.0]),
    "elbow_flexion_trail": np.array([15.0, 30.0, 50.0, 65.0, 55.0]),
    "wrist_flexion_lead": np.array([5.0, 10.0, 20.0, 25.0, 20.0]),
    "wrist_flexion_trail": np.array([3.0, 8.0, 15.0, 20.0, 15.0]),
    "wrist_ulnar_deviation_lead": np.array([2.0, 5.0, 10.0, 15.0, 12.0]),
    "wrist_ulnar_deviation_trail": np.array([1.0, 4.0, 8.0, 12.0, 10.0]),
}
_VELS: dict = {k: np.gradient(v, _T) for k, v in _ANGLES.items()}
_TORQUES: dict = {k: v * 0.1 for k, v in _VELS.items()}


class TestJointSideEnum:
    def test_lead_value(self) -> None:
        assert JointSide.LEAD.value == "lead"

    def test_trail_value(self) -> None:
        assert JointSide.TRAIL.value == "trail"

    def test_both_value(self) -> None:
        assert JointSide.BOTH.value == "both"


class TestStressTypeEnum:
    def test_compression_value(self) -> None:
        assert StressType.COMPRESSION.value == "compression"

    def test_tension_value(self) -> None:
        assert StressType.TENSION.value == "tension"


class TestJointStressAnalyzerInit:
    def test_joint_stress_instantiates(self) -> None:
        a = _make_analyzer()
        assert a is not None

    def test_body_weight_stored(self) -> None:
        a = _make_analyzer(75.0)
        assert a.body_weight == 75.0

    def test_body_weight_n_positive(self) -> None:
        a = _make_analyzer(80.0)
        assert a.body_weight_N > 0.0

    def test_default_handedness_right(self) -> None:
        a = _make_analyzer()
        assert a.handedness == "right"

    def test_left_handedness_stored(self) -> None:
        a = JointStressAnalyzer(body_weight=70.0, handedness="left")
        assert a.handedness == "left"

    def test_height_optional(self) -> None:
        a = _make_analyzer()
        assert a.height is None

    def test_height_stored_when_provided(self) -> None:
        a = JointStressAnalyzer(body_weight=80.0, height=1.8)
        assert a.height == 1.8


class TestAnalyzeAllJoints:
    def test_joint_stress_returns_dict(self) -> None:
        a = _make_analyzer()
        results = a.analyze_all_joints(_ANGLES, _VELS, _TORQUES, _T)
        assert isinstance(results, dict)

    def test_has_joint_keys(self) -> None:
        a = _make_analyzer()
        results = a.analyze_all_joints(_ANGLES, _VELS, _TORQUES, _T)
        assert len(results) > 0

    def test_values_are_joint_stress_results(self) -> None:
        a = _make_analyzer()
        results = a.analyze_all_joints(_ANGLES, _VELS, _TORQUES, _T)
        for key, val in results.items():
            assert isinstance(val, JointStressResult), f"Key {key!r} has wrong type"


class TestGetSummary:
    def test_joint_stress_returns_dict(self) -> None:
        a = _make_analyzer()
        results = a.analyze_all_joints(_ANGLES, _VELS, _TORQUES, _T)
        summary = a.get_summary(results)
        assert isinstance(summary, dict)
