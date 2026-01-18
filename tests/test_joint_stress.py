
import numpy as np
import pytest

from shared.python.injury.joint_stress import (
    JointSide,
    JointStressAnalyzer,
    JointStressResult,
)


class TestJointStressAnalyzer:
    """Test suite for JointStressAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return JointStressAnalyzer(body_weight=80.0, handedness="right")

    @pytest.fixture
    def mock_data(self):
        time = np.linspace(0, 1.0, 20)
        zeros = np.zeros_like(time)
        return {
            "time": time,
            "joint_angles": {
                "hip_rotation": zeros + 0.1,
                "hip_flexion": zeros + 0.1,
                "shoulder_horizontal": zeros + 0.1,
                "elbow_flexion": zeros + 1.0,
                "wrist_cock": zeros + 0.1,
                "wrist_rotation": zeros + 0.1,
            },
            "joint_velocities": {},
            "joint_torques": {},
        }

    def test_analyze_all_joints(self, analyzer, mock_data):
        """Test complete analysis runs without error."""
        results = analyzer.analyze_all_joints(
            mock_data["joint_angles"],
            mock_data["joint_velocities"],
            mock_data["joint_torques"],
            mock_data["time"],
        )
        
        assert "hip_lead" in results
        assert "shoulder_trail" in results
        assert isinstance(results["hip_lead"], JointStressResult)
        assert results["hip_lead"].side == JointSide.LEAD

    def test_hip_impingement_risk(self, analyzer, mock_data):
        """Test hip impingement logic."""
        # High internal rotation + flexion
        mock_data["joint_angles"]["hip_lead_rotation"] = np.radians(np.full(20, 50.0)) # 50 deg
        mock_data["joint_angles"]["hip_lead_flexion"] = np.radians(np.full(20, 110.0)) # 110 deg
        
        result = analyzer.analyze_hip(
            mock_data["joint_angles"],
            mock_data["joint_velocities"],
            mock_data["joint_torques"],
            mock_data["time"],
            JointSide.LEAD
        )
        
        # Indicator = 50 + 110*0.5 = 105 > 100
        assert result.impingement_risk is True
        assert result.risk_score > 0

    def test_elbow_valgus_risk(self, analyzer, mock_data):
        """Test elbow valgus torque risk."""
        # High torque
        mock_data["joint_torques"]["elbow_flexion"] = np.full(20, 200.0) # High torque (200 * 0.3 = 60 > 35)
        
        result = analyzer.analyze_elbow(
            mock_data["joint_angles"],
            mock_data["joint_velocities"],
            mock_data["joint_torques"],
            mock_data["time"],
            JointSide.LEAD
        )
        
        assert result.overload_risk is True
        assert result.risk_score > 0

    def test_summary_generation(self, analyzer):
        """Test summary report generation."""
        # Create dummy results
        results = {
            "hip_lead": JointStressResult("hip", JointSide.LEAD, risk_score=80.0, overload_risk=True),
            "hip_trail": JointStressResult("hip", JointSide.TRAIL, risk_score=20.0),
        }
        
        summary = analyzer.get_summary(results)
        
        assert summary["highest_risk_joint"] == "hip_lead"
        assert summary["highest_risk_score"] == 80.0
        assert "hip_lead" in summary["joints_at_risk"]
        assert len(summary["recommendations"]) > 0
