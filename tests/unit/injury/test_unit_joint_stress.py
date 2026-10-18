"""Tests for src.shared.python.injury.joint_stress (Issues #1949, #1744)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from src.shared.python.injury.joint_stress import (
    JointSide,
    JointStressAnalyzer,
    JointStressResult,
    StressType,
    WristDeviation,
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


_WRIST_T = np.linspace(0.0, 1.0, 5)
_WRIST_ROT = {"wrist_rotation": np.zeros(5)}


@pytest.mark.unit
class TestWristDeviationInput:
    def test_unsigned_convention_raises(self) -> None:
        with pytest.raises(ValueError, match="sign_convention"):
            WristDeviation(np.zeros(5), JointSide.LEAD, sign_convention="unsigned")

    def test_both_side_raises(self) -> None:
        with pytest.raises(ValueError, match="side"):
            WristDeviation(
                np.zeros(5), JointSide.BOTH, sign_convention="ulnar_positive"
            )

    def test_non_1d_angles_raise(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            WristDeviation(
                np.zeros((5, 2)), JointSide.LEAD, sign_convention="ulnar_positive"
            )


@pytest.mark.unit
class TestAnalyzeWristDeviation:
    def _analyze(self, side: JointSide, deviation, angles=None):
        analyzer = _make_analyzer()
        joint_angles = dict(_WRIST_ROT) if angles is None else {**angles, **_WRIST_ROT}
        return analyzer.analyze_wrist(
            joint_angles, {}, {}, _WRIST_T, side, deviation=deviation
        )

    def test_untagged_raw_array_raises(self) -> None:
        with pytest.raises(ValueError, match="WristDeviation"):
            self._analyze(JointSide.LEAD, np.zeros(5))

    def test_side_mismatch_raises(self) -> None:
        trail = WristDeviation(
            np.radians(np.full(5, 10.0)),
            JointSide.TRAIL,
            sign_convention="ulnar_positive",
        )
        with pytest.raises(ValueError, match="side"):
            self._analyze(JointSide.LEAD, trail)

    def test_missing_deviation_is_unavailable_not_zero(self) -> None:
        result = self._analyze(JointSide.LEAD, None)
        assert math.isnan(result.risk_score)
        assert result.unavailable_reason is not None
        assert "deviation" in result.unavailable_reason
        assert result.overload_risk is False

    def test_wrist_cock_is_not_used_as_deviation(self) -> None:
        result = self._analyze(
            JointSide.LEAD, None, angles={"wrist_cock": np.radians(np.full(5, 40.0))}
        )
        assert math.isnan(result.risk_score)
        assert result.overload_risk is False

    def test_missing_rotation_is_unavailable(self) -> None:
        analyzer = _make_analyzer()
        dev = WristDeviation(
            np.radians(np.full(5, 10.0)),
            JointSide.LEAD,
            sign_convention="ulnar_positive",
        )
        result = analyzer.analyze_wrist(
            {}, {}, {}, _WRIST_T, JointSide.LEAD, deviation=dev
        )
        assert math.isnan(result.risk_score)
        assert "wrist_rotation" in (result.unavailable_reason or "")

    def test_left_ulnar_and_right_radial_same_motion_agree(self) -> None:
        lead = WristDeviation(
            np.radians(np.full(5, 40.0)),
            JointSide.LEAD,
            sign_convention="ulnar_positive",
        )
        trail = WristDeviation(
            np.radians(np.full(5, -40.0)),
            JointSide.TRAIL,
            sign_convention="radial_positive",
        )
        r_lead = self._analyze(JointSide.LEAD, lead)
        r_trail = self._analyze(JointSide.TRAIL, trail)
        assert r_lead.unavailable_reason is None
        assert r_trail.unavailable_reason is None
        assert r_lead.overload_risk is True
        assert r_trail.overload_risk is True
        assert np.isclose(r_lead.rom_utilization, r_trail.rom_utilization)

    def test_declared_sign_changes_outcome(self) -> None:
        mis_signed = WristDeviation(
            np.radians(np.full(5, -40.0)),
            JointSide.LEAD,
            sign_convention="ulnar_positive",
        )
        result = self._analyze(JointSide.LEAD, mis_signed)
        assert result.overload_risk is False

    def test_analyze_all_joints_takes_per_side_deviation(self) -> None:
        analyzer = _make_analyzer()
        t = _WRIST_T
        angles = {"elbow_flexion": np.full(5, 0.5), **_WRIST_ROT}
        lead = WristDeviation(
            np.radians(np.full(5, 10.0)),
            JointSide.LEAD,
            sign_convention="ulnar_positive",
        )
        results = analyzer.analyze_all_joints(
            angles,
            {},
            {},
            t,
            wrist_deviation={JointSide.LEAD: lead},
        )
        assert math.isfinite(results["wrist_lead"].rom_utilization)
        assert math.isnan(results["wrist_trail"].risk_score)


@pytest.mark.unit
class TestNoSilentZeroDefaults:
    """Missing required series give unavailable results, never zero-filled ones."""

    def _elbow_inputs(self, with_wrist_torque: bool) -> tuple[dict, dict]:
        angles = {"elbow_flexion": np.radians(np.full(5, 60.0))}
        torques = {"elbow_flexion": np.full(5, 10.0)}
        if with_wrist_torque:
            torques["wrist_cock"] = np.full(5, 4.0)
        return angles, torques

    def test_elbow_missing_wrist_torque_is_unavailable(self) -> None:
        a = _make_analyzer()
        angles, torques = self._elbow_inputs(with_wrist_torque=False)
        result = a.analyze_elbow(angles, {}, torques, _T, JointSide.LEAD)
        assert result.unavailable_reason is not None
        assert "wrist_cock" in result.unavailable_reason
        assert math.isnan(result.risk_score)
        assert math.isnan(result.peak_torsion)
        assert result.overload_risk is False

    def test_elbow_with_wrist_torque_is_available(self) -> None:
        a = _make_analyzer()
        angles, torques = self._elbow_inputs(with_wrist_torque=True)
        result = a.analyze_elbow(angles, {}, torques, _T, JointSide.LEAD)
        assert result.unavailable_reason is None
        assert math.isfinite(result.risk_score)

    def test_elbow_missing_flexion_is_unavailable(self) -> None:
        a = _make_analyzer()
        _, torques = self._elbow_inputs(with_wrist_torque=True)
        result = a.analyze_elbow({}, {}, torques, _T, JointSide.TRAIL)
        assert result.unavailable_reason is not None
        assert "elbow flexion" in result.unavailable_reason
        assert math.isnan(result.rom_utilization)

    def test_hip_missing_inputs_is_unavailable(self) -> None:
        a = _make_analyzer()
        result = a.analyze_hip({}, {}, {}, _T, JointSide.LEAD)
        assert result.unavailable_reason is not None
        assert math.isnan(result.risk_score)
        assert math.isnan(result.rom_utilization)
        assert result.impingement_risk is False

    def test_shoulder_missing_torque_is_unavailable(self) -> None:
        a = _make_analyzer()
        angles = {"shoulder_horizontal": np.radians(np.full(5, 20.0))}
        result = a.analyze_shoulder(angles, {}, {}, _T, JointSide.TRAIL)
        assert result.unavailable_reason is not None
        assert "shoulder torque" in result.unavailable_reason
        assert math.isnan(result.risk_score)
        assert math.isnan(result.peak_torsion)
