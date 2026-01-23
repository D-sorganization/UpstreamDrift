import numpy as np
import pytest

from src.shared.python.injury.spinal_load_analysis import (
    SpinalLoadAnalyzer,
    SpinalLoadResult,
    SpinalRiskLevel,
    create_example_analysis,
)


class TestSpinalLoadAnalysis:
    """Test suite for SpinalLoadAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a default analyzer."""
        return SpinalLoadAnalyzer(body_weight=80.0, height=1.80)

    @pytest.fixture
    def example_data(self):
        """Generate synthetic swing data."""
        time = np.linspace(0, 1.0, 50)
        zeros = np.zeros_like(time)
        return {
            "time": time,
            "joint_angles": {
                "lumbar_flexion": zeros + 0.1,  # Constant flexion
                "lumbar_lateral": zeros,
                "lumbar_rotation": zeros,
            },
            "joint_velocities": {
                "lumbar_flexion": zeros,
                "lumbar_lateral": zeros,
                "lumbar_rotation": zeros,
            },
            "joint_torques": {
                "lumbar_flexion": zeros + 10.0,
                "lumbar_lateral": zeros,
                "lumbar_rotation": zeros,
            },
        }

    def test_initialization(self, analyzer):
        """Test analyzer initialization parameters."""
        assert analyzer.body_weight == 80.0
        assert analyzer.height == 1.80
        # Check derived trunk length
        assert analyzer.trunk_length == pytest.approx(0.288 * 1.80)
        assert len(analyzer.lumbar_segments) == 3

    def test_analyze_structure(self, analyzer, example_data):
        """Test the analyze method returns correct structure."""
        result = analyzer.analyze(
            joint_angles=example_data["joint_angles"],
            joint_velocities=example_data["joint_velocities"],
            joint_torques=example_data["joint_torques"],
            time=example_data["time"],
        )

        assert isinstance(result, SpinalLoadResult)
        assert len(result.segments) == 3
        assert "L4-L5" in result.segments
        assert result.peak_compression_bw > 0

    def test_example_analysis(self):
        """Test the integrated example function."""
        analyzer, result = create_example_analysis()

        assert hasattr(result, "x_factor")
        assert result.x_factor is not None
        assert result.crunch_factor is not None

        # Verify risks are calculated
        assert isinstance(result.overall_risk, SpinalRiskLevel)

        # Verify recommendations
        recs = analyzer.get_recommendations(result)
        assert isinstance(recs, list)

    def test_risk_thresholds(self, analyzer, example_data):
        """Test that high loads trigger correct risk levels."""
        # Create high compression scenario
        # Force = Compression Threshold * BodyWeight * 9.81
        # Body Weight = 80kg
        # High Risk Compression > 6.0 BW
        # Let's inject a torque that causes high compression
        # Compression ~= gravity + torque/moment_arm
        # moment_arm = 0.05
        # needed_muscle_force = (6.0 * 80 * 9.81)
        # needed_torque = needed_muscle_force * 0.05

        high_torque = (6.5 * 80 * 9.81) * 0.05

        example_data["joint_torques"]["lumbar_flexion"][:] = high_torque

        result = analyzer.analyze(
            joint_angles=example_data["joint_angles"],
            joint_velocities=example_data["joint_velocities"],
            joint_torques=example_data["joint_torques"],
            time=example_data["time"],
        )

        assert result.peak_compression_bw >= 6.0
        assert result.compression_risk in [
            SpinalRiskLevel.HIGH_RISK,
            SpinalRiskLevel.CRITICAL,
        ]

    def test_input_validation_shapes(self, analyzer):
        """Test handling of mismatched array shapes (should likely raise error)."""
        time_short = np.linspace(0, 1.0, 10)

        angles = {"lumbar_flexion": np.zeros(10)}  # Matches short
        velocities = {"lumbar_flexion": np.zeros(20)}  # Mismatch

        # The current implementation might not explicitly check, so it might fail with ValueError
        # or operate on mismatched arrays depending on numpy behavior.
        # Ideally it should raise ValueError.

        with pytest.raises((ValueError, IndexError)):
            analyzer.analyze(
                joint_angles=angles,
                joint_velocities=velocities,
                joint_torques={"lumbar_flexion": np.zeros(10)},
                time=time_short,
            )
