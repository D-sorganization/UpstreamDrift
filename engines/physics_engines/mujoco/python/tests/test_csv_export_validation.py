"""Tests for CSV export validation per Assessment A Finding A-007."""

import numpy as np
import pytest
from mujoco_humanoid_golf.inverse_dynamics import (
    InverseDynamicsResult,
    export_inverse_dynamics_to_csv,
)


class TestCSVExportValidation:
    """Test comprehensive input validation for CSV export (Finding A-007)."""

    def test_export_validates_times_is_numpy_array(self, tmp_path):
        """Test that times must be a numpy array."""
        results = [
            InverseDynamicsResult(joint_torques=np.array([1.0, 2.0]), success=True)
        ]
        filepath = tmp_path / "test.csv"

        # Should fail with list instead of array
        with pytest.raises(TypeError, match="times must be numpy array"):
            export_inverse_dynamics_to_csv([0.0], results, str(filepath))

    def test_export_validates_results_is_list(self, tmp_path):
        """Test that results must be a list."""
        times = np.array([0.0])
        result = InverseDynamicsResult(joint_torques=np.array([1.0, 2.0]), success=True)
        filepath = tmp_path / "test.csv"

        # Should fail with single result instead of list
        with pytest.raises(TypeError, match="results must be list"):
            export_inverse_dynamics_to_csv(times, result, str(filepath))

    def test_export_validates_non_empty_results(self, tmp_path):
        """Test that results list cannot be empty."""
        times = np.array([])
        results: list[InverseDynamicsResult] = []
        filepath = tmp_path / "test.csv"

        with pytest.raises(ValueError, match="Cannot export empty results list"):
            export_inverse_dynamics_to_csv(times, results, str(filepath))

    def test_export_validates_length_match(self, tmp_path):
        """Test that times and results must have same length."""
        times = np.array([0.0, 1.0, 2.0])
        results = [
            InverseDynamicsResult(joint_torques=np.array([1.0, 2.0]), success=True),
            InverseDynamicsResult(joint_torques=np.array([3.0, 4.0]), success=True),
        ]
        filepath = tmp_path / "test.csv"

        with pytest.raises(
            ValueError, match="Length mismatch: times has 3 elements, results has 2"
        ):
            export_inverse_dynamics_to_csv(times, results, str(filepath))

    def test_export_validates_result_types(self, tmp_path):
        """Test that all results must be InverseDynamicsResult instances."""
        times = np.array([0.0, 1.0])
        results = [
            InverseDynamicsResult(joint_torques=np.array([1.0, 2.0]), success=True),
            {"not": "a result"},  # Wrong type
        ]
        filepath = tmp_path / "test.csv"

        with pytest.raises(TypeError, match=r"results\[1\] is dict, expected"):
            export_inverse_dynamics_to_csv(times, results, str(filepath))

    def test_export_validates_consistent_joint_counts(self, tmp_path):
        """Test that all results must have same number of joints."""
        times = np.array([0.0, 1.0])
        results = [
            InverseDynamicsResult(
                joint_torques=np.array([1.0, 2.0, 3.0]), success=True
            ),  # 3 joints
            InverseDynamicsResult(
                joint_torques=np.array([4.0, 5.0]), success=True
            ),  # 2 joints
        ]
        filepath = tmp_path / "test.csv"

        with pytest.raises(
            ValueError, match="Inconsistent joint count: results\\[0\\] has 3 joints"
        ):
            export_inverse_dynamics_to_csv(times, results, str(filepath))

    def test_export_succeeds_with_valid_input(self, tmp_path):
        """Test that export succeeds with valid input."""
        times = np.array([0.0, 1.0, 2.0])
        results = [
            InverseDynamicsResult(
                joint_torques=np.array([1.0, 2.0]),
                inertial_torques=np.array([0.5, 1.0]),
                coriolis_torques=np.array([0.2, 0.3]),
                gravity_torques=np.array([0.3, 0.7]),
                residual_norm=0.01,
                success=True,
            ),
            InverseDynamicsResult(
                joint_torques=np.array([3.0, 4.0]),
                inertial_torques=np.array([1.5, 2.0]),
                coriolis_torques=np.array([0.4, 0.5]),
                gravity_torques=np.array([1.1, 1.5]),
                residual_norm=0.02,
                success=True,
            ),
            InverseDynamicsResult(
                joint_torques=np.array([5.0, 6.0]),
                inertial_torques=np.array([2.5, 3.0]),
                coriolis_torques=np.array([0.6, 0.7]),
                gravity_torques=np.array([1.9, 2.3]),
                residual_norm=0.03,
                success=True,
            ),
        ]
        filepath = tmp_path / "test.csv"

        # Should succeed
        export_inverse_dynamics_to_csv(times, results, str(filepath))

        # Verify file was created
        assert filepath.exists()

        # Verify file content (basic check)
        content = filepath.read_text()
        assert "time,torque_0,torque_1" in content
        assert "0.0,1.0" in content
        assert "1.0,3.0" in content
        assert "2.0,5.0" in content

    def test_export_handles_none_optional_fields(self, tmp_path):
        """Test export with None values for optional fields."""
        times = np.array([0.0])
        results = [
            InverseDynamicsResult(
                joint_torques=np.array([1.0, 2.0]),
                # All optional fields left as None
                success=True,
            )
        ]
        filepath = tmp_path / "test.csv"

        # Should succeed and use 0.0 for None values
        export_inverse_dynamics_to_csv(times, results, str(filepath))

        assert filepath.exists()
        content = filepath.read_text()
        # Verify 0.0 used for missing fields
        assert ",0.0,0.0,0.0" in content  # inertial, coriolis, gravity all 0
