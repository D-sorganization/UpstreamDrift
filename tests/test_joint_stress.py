import numpy as np
import pytest

from shared.python.injury.joint_stress import (
    JointSide,
    JointStressAnalyzer,
    JointStressResult,
)


class TestJointStressAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return JointStressAnalyzer(body_weight=80.0, handedness="right")

    @pytest.fixture
    def sample_data(self):
        time = np.linspace(0, 1.0, 100)
        zeros = np.zeros_like(time)
        return {
            "time": time,
            "joint_angles": {
                "hip_rotation": np.ones_like(time) * np.radians(30),  # 30 deg
                "hip_flexion": np.ones_like(time) * np.radians(20),  # 20 deg
                "shoulder_horizontal": np.ones_like(time) * np.radians(90),
                "shoulder_vertical": zeros,
                "elbow_flexion": np.ones_like(time) * np.radians(90),
                "wrist_cock": np.ones_like(time) * np.radians(20),
                "wrist_rotation": np.ones_like(time) * np.radians(10),
            },
            "joint_velocities": {
                "shoulder_horizontal": np.ones_like(time) * 10.0,  # High velocity
                "wrist_cock": np.ones_like(time) * 5.0,
            },
            "joint_torques": {
                "shoulder_horizontal": np.ones_like(time) * 50.0,
                "elbow_flexion": np.ones_like(time) * 30.0,
                "wrist_cock": np.ones_like(time) * 10.0,
            },
        }

    def test_initialization(self, analyzer):
        assert analyzer.body_weight == 80.0
        assert analyzer.lead_side == "left"
        assert analyzer.trail_side == "right"

    def test_analyze_hip(self, analyzer, sample_data):
        result = analyzer.analyze_hip(
            sample_data["joint_angles"],
            sample_data["joint_velocities"],
            sample_data["joint_torques"],
            sample_data["time"],
            JointSide.LEAD,
        )

        assert isinstance(result, JointStressResult)
        assert result.joint_name == "hip"
        assert result.side == JointSide.LEAD
        # Metric checks
        # internal rot=30, flexion=20 -> indicator = 30 + 10 = 40
        # risk_score = max(0, 40 - 50) = 0
        assert result.risk_score == 0.0

    def test_analyze_shoulder(self, analyzer, sample_data):
        result = analyzer.analyze_shoulder(
            sample_data["joint_angles"],
            sample_data["joint_velocities"],
            sample_data["joint_torques"],
            sample_data["time"],
            JointSide.TRAIL,
        )

        assert hasattr(result, "peak_torsion")
        # RC loading = velocity(10) * torque(50) / 100 = 5.0
        # Multiplier 1.2 for trail side -> 6.0
        assert result.risk_score == pytest.approx(6.0)

    def test_analyze_elbow(self, analyzer, sample_data):
        result = analyzer.analyze_elbow(
            sample_data["joint_angles"],
            sample_data["joint_velocities"],
            sample_data["joint_torques"],
            sample_data["time"],
            JointSide.LEAD,
        )

        # Valgus est = abs(30)*0.3 + abs(10)*0.5 = 9.0 + 5.0 = 14.0
        # Limit is 35.0
        # Risk = (14 / 35) * 50 * 1.3 (side) = 26.0
        assert result.risk_score > 0
        assert not result.overload_risk

    def test_analyze_wrist(self, analyzer, sample_data):
        result = analyzer.analyze_wrist(
            sample_data["joint_angles"],
            sample_data["joint_velocities"],
            sample_data["joint_torques"],
            sample_data["time"],
            JointSide.LEAD,
        )

        # Ulnar=20 (deg) -> safe (<35)
        assert not result.overload_risk
        assert result.risk_score >= 0

    def test_analyze_all_joints(self, analyzer, sample_data):
        results = analyzer.analyze_all_joints(
            sample_data["joint_angles"],
            sample_data["joint_velocities"],
            sample_data["joint_torques"],
            sample_data["time"],
        )

        expected_keys = [
            "hip_lead",
            "hip_trail",
            "shoulder_lead",
            "shoulder_trail",
            "elbow_lead",
            "elbow_trail",
            "wrist_lead",
            "wrist_trail",
        ]
        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], JointStressResult)

    def test_get_summary(self, analyzer):
        # Create mock results
        results = {
            "hip_lead": JointStressResult(
                "hip", JointSide.LEAD, risk_score=80.0, overload_risk=True
            ),
            "shoulder_lead": JointStressResult(
                "shoulder", JointSide.LEAD, risk_score=40.0
            ),
        }

        summary = analyzer.get_summary(results)

        assert summary["highest_risk_joint"] == "hip_lead"
        assert summary["highest_risk_score"] == 80.0
        assert "hip_lead" in summary["joints_at_risk"]
        assert len(summary["recommendations"]) > 0
