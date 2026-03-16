"""Unit tests for data_fitting.py."""

import numpy as np
import pytest

from src.shared.python.validation_pkg.data_fitting import (
    A3FittingPipeline,
    BodySegmentParams,
    InverseKinematicsSolver,
    KinematicState,
    ParameterEstimator,
    SensitivityAnalyzer,
    SensitivityResult,
    convert_poses_to_markers,
)


def test_body_segment_params() -> None:
    """Test BodySegmentParams serialization."""
    params = BodySegmentParams(
        name="test_segment",
        length=0.5,
        mass=2.0,
        com_position=0.4,
        inertia=np.array([0.1, 0.2, 0.3]),
        radius_gyration=0.25,
    )
    
    data = params.to_dict()
    assert data["name"] == "test_segment"
    assert data["length"] == 0.5
    assert data["inertia"] == [0.1, 0.2, 0.3]
    
    new_params = BodySegmentParams.from_dict(data)
    assert new_params.name == params.name
    assert new_params.length == params.length
    assert np.allclose(new_params.inertia, params.inertia)


class TestInverseKinematicsSolver:
    """Tests for IK solver."""
    
    def setup_method(self) -> None:
        """Set up solver for testing."""
        self.segment_lengths = {"link1": 0.3, "link2": 0.4}
        self.joint_names = ["link1_joint", "link2_joint"]
        self.solver = InverseKinematicsSolver(self.segment_lengths, self.joint_names)
        
    def test_solve_analytical_2d(self) -> None:
        """Test analytical 2-link IK."""
        # Arm fully extended
        t1, t2 = self.solver.solve_analytical_2d(np.array([0.7, 0.0]), 0.3, 0.4)
        assert np.isclose(t1, 0.0, atol=1e-5)
        assert np.isclose(t2, 0.0, atol=1e-5)
        
        # Target unreachable
        with pytest.raises(ValueError, match="unreachable"):
            self.solver.solve_analytical_2d(np.array([10.0, 0.0]), 0.3, 0.4)
            
        # Target too close 
        with pytest.raises(ValueError, match="too close"):
            self.solver.solve_analytical_2d(np.array([0.01, 0.0]), 0.3, 0.4)

    def test_solve_numerical(self) -> None:
        """Test numerical IK fitting."""
        target = np.array([[0.3, 0.0, 0.0], [0.7, 0.0, 0.0]])
        result = self.solver.solve_numerical(target)
        
        assert result.success is True
        assert "link1_joint" in result.parameters
        assert "link2_joint" in result.parameters
        
        # Test no initial angles
        result2 = self.solver.solve_numerical(target, initial_angles=None)
        assert result2.success is True


class TestParameterEstimator:
    """Tests for anthropometric parameter estimation."""

    def setup_method(self) -> None:
        """Set up parameter estimator."""
        self.estimator = ParameterEstimator(anthropometric_model="dempster")

    def test_estimate_segment_length(self) -> None:
        """Test segment length estimation from markers."""
        proximal = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        distal = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        mean_len, std_len = self.estimator.estimate_segment_length(proximal, distal)
        assert np.isclose(mean_len, 1.0)
        assert np.isclose(std_len, 0.0)

    def test_estimate_segment_params(self) -> None:
        """Test estimating params for a segment."""
        # Known dempster fraction for thigh: mass 0.1
        params = self.estimator.estimate_segment_params("thigh", 0.4, 70.0)
        
        assert params.name == "thigh"
        assert params.length == 0.4
        assert np.isclose(params.mass, 70.0 * 0.1)
        assert len(params.inertia) == 3
        
        # Unknown segment
        params = self.estimator.estimate_segment_params("unknown_body", 0.5, 70.0)
        assert params.mass == 70.0 * 0.02  # Default mass fraction 0.02

    def test_fit_parameters_to_kinematics_no_markers(self) -> None:
        """Test fitting with only anthropometry (no markers)."""
        states = [KinematicState(timestamp=0.0)]
        result = self.estimator.fit_parameters_to_kinematics(
            states, ["upper_arm", "forearm"], 70.0, {"upper_arm": 0.3}
        )
        
        assert result.success is True
        assert "upper_arm_length" in result.parameters
        assert result.parameters["upper_arm_length"] == 0.3
        
        # Test empty kinematic data
        result_empty = self.estimator.fit_parameters_to_kinematics(
            [], ["upper_arm", "forearm"], 70.0
        )
        assert result_empty.success is False
        
    def test_fit_parameters_to_kinematics_with_markers(self) -> None:
        """Test fitting with markers."""
        marker_data = np.array([
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],  # Frame 1
            [[0, 1, 0], [1, 1, 0], [2, 1, 0]],  # Frame 2
        ])
        states = [
            KinematicState(timestamp=0.0, marker_positions=marker_data[0]),
            KinematicState(timestamp=0.1, marker_positions=marker_data[1]),
        ]
        
        result = self.estimator.fit_parameters_to_kinematics(
            states, ["upper_arm", "forearm"], 70.0, known_lengths={"upper_arm": 1.0}
        )
        
        assert result.success is True
        assert "upper_arm_length" in result.parameters
        assert np.isclose(result.parameters["upper_arm_length"], 1.0)


class TestSensitivityAnalyzer:
    """Tests for sensitivity analysis."""

    def test_compute_sensitivity(self) -> None:
        """Test computing sensitivity index."""
        analyzer = SensitivityAnalyzer(perturbation_size=0.1)
        
        # f(x) = x^2. Expected partial at x=2 is roughly 4.
        def dummy_model(params: dict[str, float]) -> dict[str, float]:
            return {"output": params["x"]**2}
            
        result = analyzer.compute_sensitivity(
            dummy_model, "x", 2.0, "output"
        )
        
        assert result.parameter_name == "x"
        assert result.nominal_value == 2.0
        assert np.isclose(result.partial_derivative, 4.0)
        # Elasticity = (df/dx) * (x/f) = 4 * 2 / 4 = 2.0
        assert np.isclose(result.elasticity, 2.0)
        
    def test_compute_sensitivity_exception(self) -> None:
        analyzer = SensitivityAnalyzer(perturbation_size=0.1)
        
        def bad_model(params: dict[str, float]) -> dict[str, float]:
            raise ValueError("bad")
            
        result = analyzer.compute_sensitivity(bad_model, "x", 2.0, "output")
        assert result.sensitivity_index == 0.0
        
    def test_sensitivity_report_empty(self) -> None:
        analyzer = SensitivityAnalyzer()
        assert "error" in analyzer.sensitivity_report([])

    def test_sensitivity_report(self) -> None:
        """Test report generation."""
        analyzer = SensitivityAnalyzer()
        
        s1 = SensitivityResult("x1", 1.0, 10.0, 5.0, (4, 6), 2.0)
        s2 = SensitivityResult("x2", 1.0, 2.0, 1.0, (0.5, 1.5), 0.5)
        
        report = analyzer.sensitivity_report([s1, s2])
        
        assert report["total_parameters"] == 2
        assert report["most_sensitive"] == "x1"
        assert report["least_sensitive"] == "x2"

def test_convert_poses_to_markers() -> None:
    # 2D case
    poses = np.array([[1.0, 2.0], [3.0, 4.0]])
    names = ["left_shoulder", "unknown"]
    markers, mn = convert_poses_to_markers(poses, names)
    assert len(markers) == 1
    assert mn[0] == "LSHO"
    assert markers.shape[1] == 3
    
    # 3D case with target
    poses3d = np.array([[1.0, 2.0, 3.0]])
    names3d = ["left_shoulder"]
    markers3d, mn3d = convert_poses_to_markers(poses3d, names3d, target_markers=["RSHO"])
    assert len(markers3d) == 0

class TestA3FittingPipeline:
    def setup_method(self) -> None:
        self.pipeline = A3FittingPipeline()
        
    def test_fit_from_markers(self) -> None:
        marker_data = np.array([
            [[0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [1, 1, 0]],
        ])
        timestamps = np.array([0.0, 0.1])
        names = ["pelvis", "trunk"]
        
        report = self.pipeline.fit_from_markers(
            marker_data, names, timestamps, 70.0, "subj1"
        )
        assert report.subject_id == "subj1"
        assert len(report.segment_params) > 0
        
    def test_export_report(self, tmp_path) -> None:
        marker_data = np.array([[[0, 0, 0], [1, 0, 0]]])
        timestamps = np.array([0.0])
        report = self.pipeline.fit_from_markers(
            marker_data, ["pelvis", "trunk"], timestamps, 70.0
        )
        out_file = tmp_path / "report.json"
        self.pipeline.export_report(report, out_file, format="json")
        assert out_file.exists()
        
        with pytest.raises(ValueError, match="Unsupported"):
             self.pipeline.export_report(report, out_file, format="xml")

    def test_fit_from_c3d(self, tmp_path, monkeypatch) -> None:
        import sys
        from unittest.mock import MagicMock
        
        # Mock ezc3d
        mock_ezc3d = MagicMock()
        mock_c3d_data = {
            "data": {
                "points": np.zeros((4, 2, 10))  # [4 x markers x frames]
            },
            "parameters": {
                "POINT": {
                    "LABELS": {"value": ["LSHO", "RSHO"]},
                    "RATE": {"value": [100.0]}
                }
            }
        }
        mock_ezc3d.c3d.return_value = mock_c3d_data
        sys.modules["ezc3d"] = mock_ezc3d
        
        c3d_file = tmp_path / "test.c3d"
        report = self.pipeline.fit_from_c3d(c3d_file, 70.0)
        assert report.subject_id == "test"
        assert report.quality_metrics["n_frames"] == 10
        
        # Clean up
        sys.modules.pop("ezc3d", None)
        
    def test_fit_from_c3d_no_ezc3d(self, tmp_path, monkeypatch) -> None:
        import sys
        
        # Force import error
        sys.modules["ezc3d"] = None
        
        c3d_file = tmp_path / "test.c3d"
        with pytest.raises(ImportError, match="Install ezc3d"):
            self.pipeline.fit_from_c3d(c3d_file, 70.0)
            
        # Clean up
        sys.modules.pop("ezc3d", None)
