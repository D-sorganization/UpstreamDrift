"""Tests for data_fitting module (Issue #754).

Tests the A3 model fitting pipeline:
- Inverse kinematics solving
- Parameter estimation
- Sensitivity analysis
- Pose-to-marker conversion
"""

import numpy as np
import pytest

from src.shared.python.data_fitting import (
    A3FittingPipeline,
    BodySegmentParams,
    FitResult,
    InverseKinematicsSolver,
    KinematicState,
    ParameterEstimator,
    SensitivityAnalyzer,
    SensitivityResult,
    convert_poses_to_markers,
)


class TestBodySegmentParams:
    """Tests for BodySegmentParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = BodySegmentParams(
            name="upper_arm",
            length=0.3,
            mass=2.5,
        )

        assert params.name == "upper_arm"
        assert params.length == 0.3
        assert params.mass == 2.5
        assert params.com_position == 0.5  # Default
        assert params.radius_gyration == 0.3  # Default

    def test_to_dict(self):
        """Test serialization to dictionary."""
        params = BodySegmentParams(
            name="forearm",
            length=0.25,
            mass=1.5,
            com_position=0.43,
            inertia=np.array([0.01, 0.01, 0.001]),
        )

        d = params.to_dict()

        assert d["name"] == "forearm"
        assert d["length"] == 0.25
        assert d["mass"] == 1.5
        assert d["com_position"] == 0.43
        assert d["inertia"] == [0.01, 0.01, 0.001]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "name": "hand",
            "length": 0.1,
            "mass": 0.5,
            "com_position": 0.51,
        }

        params = BodySegmentParams.from_dict(d)

        assert params.name == "hand"
        assert params.length == 0.1
        assert params.mass == 0.5
        assert params.com_position == 0.51


class TestInverseKinematicsSolver:
    """Tests for inverse kinematics solver."""

    def test_init(self):
        """Test solver initialization."""
        solver = InverseKinematicsSolver(
            segment_lengths={"upper_arm": 0.3, "forearm": 0.25},
            joint_names=["shoulder_joint", "elbow_joint"],
        )

        assert len(solver.segment_lengths) == 2
        assert len(solver.joint_names) == 2

    def test_analytical_2d_reachable(self):
        """Test 2D analytical IK for reachable target."""
        solver = InverseKinematicsSolver(
            segment_lengths={},
            joint_names=[],
        )

        L1, L2 = 0.3, 0.25
        target = np.array([0.4, 0.2])  # Within reach

        theta1, theta2 = solver.solve_analytical_2d(target, L1, L2)

        # Verify solution reaches target
        x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
        y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)

        assert abs(x - target[0]) < 1e-6
        assert abs(y - target[1]) < 1e-6

    def test_analytical_2d_unreachable(self):
        """Test 2D analytical IK for unreachable target."""
        solver = InverseKinematicsSolver(
            segment_lengths={},
            joint_names=[],
        )

        L1, L2 = 0.3, 0.25
        target = np.array([1.0, 0.0])  # Too far

        with pytest.raises(ValueError, match="unreachable"):
            solver.solve_analytical_2d(target, L1, L2)

    def test_numerical_ik(self):
        """Test numerical IK optimization."""
        solver = InverseKinematicsSolver(
            segment_lengths={"seg1": 0.3, "seg2": 0.25},
            joint_names=["joint1", "joint2"],
        )

        # Target positions for end effectors
        targets = np.array([[0.3, 0.2, 0.0], [0.4, 0.3, 0.0]])

        result = solver.solve_numerical(targets)

        assert isinstance(result, FitResult)
        assert len(result.parameters) == 2
        assert "joint1" in result.parameters
        assert "joint2" in result.parameters


class TestParameterEstimator:
    """Tests for parameter estimation."""

    def test_init_dempster(self):
        """Test initialization with Dempster model."""
        estimator = ParameterEstimator(anthropometric_model="dempster")

        assert estimator.anthropometric_model == "dempster"
        assert "upper_arm" in estimator.coefficients

    def test_init_winter(self):
        """Test initialization with Winter model."""
        estimator = ParameterEstimator(anthropometric_model="winter")

        assert estimator.anthropometric_model == "winter"

    def test_estimate_segment_length(self):
        """Test segment length estimation from markers."""
        estimator = ParameterEstimator()

        # Simulate marker positions with slight noise
        n_frames = 100
        true_length = 0.3
        noise = 0.005

        proximal = np.zeros((n_frames, 3))
        distal = np.zeros((n_frames, 3))
        distal[:, 0] = true_length + np.random.normal(0, noise, n_frames)

        mean_length, std_length = estimator.estimate_segment_length(proximal, distal)

        assert abs(mean_length - true_length) < 0.01
        assert std_length < 0.02

    def test_estimate_segment_params(self):
        """Test segment parameter estimation."""
        estimator = ParameterEstimator()

        params = estimator.estimate_segment_params(
            segment_name="upper_arm",
            segment_length=0.3,
            total_body_mass=75.0,
        )

        assert params.name == "upper_arm"
        assert params.length == 0.3
        assert params.mass > 0  # Should be fraction of body mass
        assert params.mass < 75.0  # Should be less than total mass
        assert 0 < params.com_position < 1  # Should be between 0 and 1
        assert np.all(params.inertia > 0)  # All inertias should be positive

    def test_fit_parameters_no_data(self):
        """Test fitting with no data."""
        estimator = ParameterEstimator()

        result = estimator.fit_parameters_to_kinematics(
            kinematic_data=[],
            segment_names=["upper_arm"],
            total_body_mass=75.0,
        )

        assert result.success is False

    def test_fit_parameters_with_data(self):
        """Test fitting with kinematic data."""
        estimator = ParameterEstimator()

        # Create kinematic data with marker positions
        kinematic_data = []
        for t in range(10):
            state = KinematicState(
                timestamp=t * 0.01,
                marker_positions=np.array([[0, 0, 0], [0.3, 0, 0], [0.55, 0, 0]]),
            )
            kinematic_data.append(state)

        result = estimator.fit_parameters_to_kinematics(
            kinematic_data=kinematic_data,
            segment_names=["upper_arm", "forearm"],
            total_body_mass=75.0,
        )

        assert result.success is True
        assert "upper_arm_length" in result.parameters
        assert "forearm_length" in result.parameters


class TestSensitivityAnalyzer:
    """Tests for sensitivity analysis."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = SensitivityAnalyzer(perturbation_size=0.01)

        assert analyzer.perturbation_size == 0.01

    def test_compute_sensitivity(self):
        """Test sensitivity computation."""
        analyzer = SensitivityAnalyzer()

        # Simple linear model: output = 2 * param
        def model_func(params):
            return {"output": 2 * params["param"]}

        result = analyzer.compute_sensitivity(
            model_func=model_func,
            parameter_name="param",
            nominal_value=1.0,
            output_metric="output",
        )

        assert isinstance(result, SensitivityResult)
        assert result.parameter_name == "param"
        assert abs(result.partial_derivative - 2.0) < 0.1  # Should be close to 2

    def test_sensitivity_report(self):
        """Test sensitivity report generation."""
        analyzer = SensitivityAnalyzer()

        sensitivities = [
            SensitivityResult(
                parameter_name="param1",
                nominal_value=1.0,
                sensitivity_index=0.5,
                partial_derivative=0.5,
                confidence_interval=(0.9, 1.1),
                elasticity=0.5,
            ),
            SensitivityResult(
                parameter_name="param2",
                nominal_value=2.0,
                sensitivity_index=0.2,
                partial_derivative=0.2,
                confidence_interval=(1.8, 2.2),
                elasticity=0.2,
            ),
        ]

        report = analyzer.sensitivity_report(sensitivities)

        assert report["total_parameters"] == 2
        assert report["most_sensitive"] == "param1"
        assert report["least_sensitive"] == "param2"
        assert len(report["rankings"]) == 2


class TestConvertPosesToMarkers:
    """Tests for pose-to-marker conversion."""

    def test_basic_conversion(self):
        """Test basic pose to marker conversion."""
        keypoints = np.array(
            [
                [0.5, 0.3],  # left_shoulder
                [0.7, 0.3],  # right_shoulder
            ]
        )
        keypoint_names = ["left_shoulder", "right_shoulder"]

        markers, names = convert_poses_to_markers(keypoints, keypoint_names)

        assert len(markers) == 2
        assert len(names) == 2
        assert "LSHO" in names
        assert "RSHO" in names

    def test_2d_to_3d_conversion(self):
        """Test that 2D keypoints get z=0 added."""
        keypoints = np.array([[0.5, 0.3]])
        keypoint_names = ["left_shoulder"]

        markers, names = convert_poses_to_markers(keypoints, keypoint_names)

        assert markers.shape[1] == 3
        assert markers[0, 2] == 0.0

    def test_filter_by_target(self):
        """Test filtering to specific target markers."""
        keypoints = np.array(
            [
                [0.5, 0.3, 0.0],
                [0.7, 0.3, 0.0],
            ]
        )
        keypoint_names = ["left_shoulder", "right_shoulder"]
        target = ["LSHO"]

        markers, names = convert_poses_to_markers(
            keypoints, keypoint_names, target_markers=target
        )

        assert len(names) == 1
        assert names[0] == "LSHO"


class TestA3FittingPipeline:
    """Tests for the complete A3 pipeline."""

    def test_init(self):
        """Test pipeline initialization."""
        pipeline = A3FittingPipeline()

        assert pipeline.param_estimator is not None
        assert pipeline.sensitivity_analyzer is not None
        assert len(pipeline.segment_names) > 0

    def test_fit_from_markers(self):
        """Test fitting from marker data."""
        pipeline = A3FittingPipeline()

        # Create synthetic marker data
        n_frames = 50
        n_markers = 5
        marker_positions = np.random.randn(n_frames, n_markers, 3) * 0.1
        marker_positions[:, :, 0] += np.arange(n_markers) * 0.2  # Spread out

        marker_names = ["LSHO", "LELB", "LWRI", "RSHO", "RELB"]
        timestamps = np.arange(n_frames) / 100.0

        report = pipeline.fit_from_markers(
            marker_positions=marker_positions,
            marker_names=marker_names,
            timestamps=timestamps,
            subject_mass=75.0,
            subject_id="test_subject",
        )

        assert report.subject_id == "test_subject"
        assert report.fit_result is not None
        assert report.quality_metrics["n_frames"] == n_frames
        assert report.quality_metrics["n_markers"] == n_markers
