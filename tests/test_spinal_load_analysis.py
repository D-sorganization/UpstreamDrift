
import pytest
import numpy as np
from shared.python.injury.spinal_load_analysis import (
    SpinalLoadAnalyzer,
    SpinalLoadResult,
    SpinalRiskLevel
)

class TestSpinalLoadAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return SpinalLoadAnalyzer(body_weight=80.0, height=1.80)

    @pytest.fixture
    def sample_data(self):
        # Create 1 second of data
        time = np.linspace(0, 1.0, 100)
        zeros = np.zeros_like(time)
        ones = np.ones_like(time)
        
        return {
            "time": time,
            "joint_angles": {
                "lumbar_flexion": ones * 0.1,  # Radians
                "lumbar_lateral": ones * 0.05,
                "lumbar_rotation": ones * 0.2,
            },
            "joint_velocities": {
                "lumbar_flexion": zeros,
                "lumbar_lateral": zeros,
                "lumbar_rotation": ones * 5.0, # High velocity
            },
            "joint_torques": {
                "lumbar_flexion": ones * 100.0, # 100 Nm flexion torque
                "lumbar_lateral": ones * 50.0,
                "lumbar_rotation": ones * 80.0,
            },
            "pelvis_angles": np.zeros((100, 3)),
            "thorax_angles": np.column_stack([zeros, zeros, ones * 0.5]), # 0.5 rad rotation diff
        }

    def test_compute_segment_loads(self, analyzer, sample_data):
        # Test physics calculation
        segment = analyzer._compute_segment_loads(
            "L5-S1",
            sample_data["joint_angles"],
            sample_data["joint_velocities"],
            sample_data["joint_torques"],
            sample_data["time"]
        )
        
        # Verify non-zero loads
        assert np.any(segment.compression > 0)
        assert np.any(segment.ap_shear > 0)
        assert np.any(segment.lateral_shear > 0)
        assert np.any(segment.torsion > 0)
        
        # Check specific physics: Compression = Gravity + Torque/Arm
        # Muscle compression = 100 / 0.05 = 2000 N
        # Gravity > 0
        assert segment.compression[0] > 2000.0

    def test_compute_x_factor(self, analyzer, sample_data):
        metrics = analyzer._compute_x_factor(
            sample_data["pelvis_angles"],
            sample_data["thorax_angles"],
            sample_data["time"]
        )
        
        # rotation diff is 0.5 rad = approx 28.6 degrees
        assert metrics.x_factor_stretch == pytest.approx(np.degrees(0.5))
        assert metrics.separation_rate >= 0

    def test_compute_crunch_factor(self, analyzer, sample_data):
        metrics = analyzer._compute_crunch_factor(
            sample_data["joint_angles"]["lumbar_lateral"],
            sample_data["joint_angles"]["lumbar_rotation"],
            sample_data["time"]
        )
        
        # deg(0.05) * deg(0.2) / 100
        # 2.86 * 11.46 / 100 = ~0.33
        assert metrics.peak_crunch > 0
        assert metrics.crunch_factor.shape == sample_data["time"].shape

    def test_full_analysis_workflow(self, analyzer, sample_data):
        result = analyzer.analyze(
            sample_data["joint_angles"],
            sample_data["joint_velocities"],
            sample_data["joint_torques"],
            sample_data["time"],
            sample_data["pelvis_angles"],
            sample_data["thorax_angles"]
        )
        
        assert isinstance(result, SpinalLoadResult)
        assert "L5-S1" in result.segments
        assert result.x_factor is not None
        assert result.crunch_factor is not None
        
        # Risk assessment should run
        assert isinstance(result.overall_risk, SpinalRiskLevel)
        
        # Cumulative loads should be computed
        assert result.cumulative_compression_impulse > 0

    def test_risk_assessment_thresholds(self, analyzer):
        result = SpinalLoadResult(time=np.array([0]))
        
        # Case 1: Safe
        result.peak_compression_bw = 3.0
        result = analyzer._assess_risk(result)
        assert result.compression_risk == SpinalRiskLevel.SAFE
        
        # Case 2: Critical
        result.peak_compression_bw = 9.0
        result = analyzer._assess_risk(result)
        assert result.compression_risk == SpinalRiskLevel.CRITICAL

    def test_get_recommendations(self, analyzer):
        result = SpinalLoadResult(time=np.array([0]))
        result.overall_risk = SpinalRiskLevel.HIGH_RISK
        result.compression_risk = SpinalRiskLevel.HIGH_RISK
        result.peak_compression_bw = 7.0
        
        recs = analyzer.get_recommendations(result)
        assert len(recs) > 0
        assert "compression" in recs[0].lower()
