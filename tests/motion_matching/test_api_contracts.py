"""Unit tests for motion_matching.api_contracts (W4-UP-001).

Tests cover:
  - FitResult dataclass validation
  - ThetaContractValidator for all 4 engines
  - InitialPoseValidator for root frame + joints
  - Edge cases and error handling
  - DbC preconditions/postconditions
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from src.shared.python.motion_matching.api_contracts import (
    ENGINE_DOF_MAP,
    FitResult,
    InitialPose,
    InitialPoseValidator,
    ThetaContractValidator,
    validate_initial_pose,
    validate_theta_contract,
)

# ============================================================================
# FitResult Tests (7 tests)
# ============================================================================


class TestFitResultConstruction:
    """Tests for FitResult dataclass construction and validation."""

    def test_fit_result_valid_construction(self) -> None:
        """FitResult accepts valid inputs."""
        coeffs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        loss = 0.05
        metadata = {"engine": "drake", "time_s": 1.5}

        result = FitResult(
            coefficients=coeffs,
            final_loss=loss,
            metadata=metadata,
        )
        assert result.coefficients is coeffs
        assert result.final_loss == loss
        assert result.metadata == metadata
        assert result.trajectory is None

    def test_fit_result_with_trajectory(self) -> None:
        """FitResult stores trajectory metadata."""
        coeffs = np.array([0.1], dtype=np.float64)
        loss = 0.0
        metadata = {"engine": "mujoco", "time_s": 0.1}
        traj = {"positions": np.array([0, 1, 2])}

        result = FitResult(
            coefficients=coeffs,
            final_loss=loss,
            metadata=metadata,
            trajectory=traj,
        )
        assert result.trajectory == traj

    def test_fit_result_rejects_non_ndarray_coefficients(self) -> None:
        """FitResult raises TypeError for non-ndarray coefficients."""
        with pytest.raises(TypeError, match="coefficients must be np.ndarray"):
            FitResult(
                coefficients=[0.1, 0.2],  # list, not ndarray
                final_loss=0.05,
                metadata={"engine": "drake", "time_s": 1.0},
            )

    def test_fit_result_rejects_2d_coefficients(self) -> None:
        """FitResult rejects non-1D coefficient arrays."""
        with pytest.raises(ValueError, match="1-D"):
            FitResult(
                coefficients=np.array([[0.1, 0.2], [0.3, 0.4]]),
                final_loss=0.05,
                metadata={"engine": "drake", "time_s": 1.0},
            )

    def test_fit_result_rejects_nan_coefficients(self) -> None:
        """FitResult rejects NaN in coefficients."""
        coeffs = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN or Inf"):
            FitResult(
                coefficients=coeffs,
                final_loss=0.05,
                metadata={"engine": "drake", "time_s": 1.0},
            )

    def test_fit_result_rejects_negative_loss(self) -> None:
        """FitResult rejects negative final_loss."""
        with pytest.raises(ValueError, match=">= 0"):
            FitResult(
                coefficients=np.array([0.1], dtype=np.float32),
                final_loss=-0.01,
                metadata={"engine": "drake", "time_s": 1.0},
            )

    def test_fit_result_rejects_missing_metadata_keys(self) -> None:
        """FitResult rejects metadata missing required keys."""
        with pytest.raises(ValueError, match="required keys"):
            FitResult(
                coefficients=np.array([0.1], dtype=np.float32),
                final_loss=0.05,
                metadata={"engine": "drake"},  # missing "time_s"
            )


# ============================================================================
# ThetaContractValidator Tests (16 tests)
# ============================================================================


class TestThetaContractValidator:
    """Tests for theta (joint configuration) validation."""

    def test_validator_init_drake(self) -> None:
        """Validator initializes for Drake with correct DOF."""
        val = ThetaContractValidator("drake")
        assert val.engine == "drake"
        assert val.n_dof == 23

    def test_validator_init_mujoco(self) -> None:
        """Validator initializes for MuJoCo with correct DOF."""
        val = ThetaContractValidator("mujoco")
        assert val.engine == "mujoco"
        assert val.n_dof == 17

    def test_validator_init_opensim(self) -> None:
        """Validator initializes for OpenSim."""
        val = ThetaContractValidator("opensim")
        assert val.engine == "opensim"
        assert val.n_dof == 23

    def test_validator_init_pinocchio(self) -> None:
        """Validator initializes for Pinocchio."""
        val = ThetaContractValidator("pinocchio")
        assert val.engine == "pinocchio"
        assert val.n_dof == 23

    def test_validator_rejects_unknown_engine(self) -> None:
        """Validator rejects unknown engine name."""
        with pytest.raises(ValueError, match="engine must be one of"):
            ThetaContractValidator("bad_engine")

    def test_validator_custom_dof(self) -> None:
        """Validator accepts custom DOF count."""
        val = ThetaContractValidator("drake", n_dof=15)
        assert val.n_dof == 15

    def test_theta_valid_shape_and_values(self) -> None:
        """Validator accepts valid theta (23 DOF)."""
        val = ThetaContractValidator("drake")
        theta = np.linspace(-0.5, 0.5, 23, dtype=np.float64)
        error = val.validate(theta)
        assert error is None

    def test_theta_rejects_wrong_shape(self) -> None:
        """Validator rejects theta with wrong DOF count."""
        val = ThetaContractValidator("drake")  # expects 23
        theta = np.zeros(20, dtype=np.float64)
        error = val.validate(theta)
        assert error is not None
        assert "length mismatch" in error

    def test_theta_rejects_2d_array(self) -> None:
        """Validator rejects 2D theta array."""
        val = ThetaContractValidator("drake")
        theta = np.zeros((23, 1), dtype=np.float64)
        error = val.validate(theta)
        assert error is not None
        assert "1-D" in error

    def test_theta_rejects_nan(self) -> None:
        """Validator rejects theta with NaN."""
        val = ThetaContractValidator("drake")
        theta = np.zeros(23, dtype=np.float64)
        theta[10] = np.nan
        error = val.validate(theta)
        assert error is not None
        assert "non-finite" in error

    def test_theta_rejects_inf(self) -> None:
        """Validator rejects theta with infinity."""
        val = ThetaContractValidator("drake")
        theta = np.zeros(23, dtype=np.float64)
        theta[5] = np.inf
        error = val.validate(theta)
        assert error is not None
        assert "non-finite" in error

    def test_theta_rejects_out_of_range_high(self) -> None:
        """Validator rejects theta with values > MAX_THETA_VALUE."""
        val = ThetaContractValidator("drake")
        theta = np.zeros(23, dtype=np.float64)
        theta[0] = 2000.0  # > 1e3
        error = val.validate(theta)
        assert error is not None
        assert "out of range" in error

    def test_theta_rejects_out_of_range_low(self) -> None:
        """Validator rejects theta with values < MIN_THETA_VALUE."""
        val = ThetaContractValidator("drake")
        theta = np.zeros(23, dtype=np.float64)
        theta[0] = -2000.0  # < -1e3
        error = val.validate(theta)
        assert error is not None
        assert "out of range" in error

    def test_validate_raise_valid(self) -> None:
        """validate_raise() accepts valid theta."""
        val = ThetaContractValidator("drake")
        theta = np.zeros(23, dtype=np.float64)
        val.validate_raise(theta)  # Should not raise

    def test_validate_raise_invalid(self) -> None:
        """validate_raise() raises ValueError for invalid theta."""
        val = ThetaContractValidator("drake")
        theta = np.zeros(20, dtype=np.float64)
        with pytest.raises(ValueError, match="length mismatch"):
            val.validate_raise(theta)


# ============================================================================
# InitialPoseValidator Tests (8 tests)
# ============================================================================


class TestInitialPoseValidator:
    """Tests for initial pose (root frame + joints) validation."""

    def test_pose_validator_init_drake(self) -> None:
        """Validator initializes for Drake."""
        val = InitialPoseValidator("drake")
        assert val.engine == "drake"
        assert val.n_dof_total == 23
        assert val.n_joints == 17

    def test_pose_validator_init_mujoco(self) -> None:
        """Validator initializes for MuJoCo (fixed root)."""
        val = InitialPoseValidator("mujoco")
        assert val.n_dof_total == 17
        assert val.n_joints == 11

    def test_pose_valid_construction(self) -> None:
        """Validator accepts valid initial pose."""
        val = InitialPoseValidator("drake")
        pose = InitialPose(
            root_position=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            joint_angles=np.zeros(17, dtype=np.float64),
        )
        error = val.validate(pose)
        assert error is None

    def test_pose_rejects_invalid_position_norm(self) -> None:
        """Validator rejects position with too large norm."""
        val = InitialPoseValidator("drake")
        pose = InitialPose(
            root_position=np.array([100.0, 0.0, 0.0], dtype=np.float64),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            joint_angles=np.zeros(17, dtype=np.float64),
        )
        error = val.validate(pose)
        assert error is not None
        assert "norm" in error

    def test_pose_rejects_non_unit_quaternion(self) -> None:
        """Validator rejects non-unit quaternion."""
        val = InitialPoseValidator("drake")
        pose = InitialPose(
            root_position=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            root_quat=np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64),
            joint_angles=np.zeros(17, dtype=np.float64),
        )
        error = val.validate(pose)
        assert error is not None
        assert "not 1.0" in error

    def test_pose_rejects_nan_position(self) -> None:
        """Validator rejects NaN in position."""
        val = InitialPoseValidator("drake")
        pose = InitialPose(
            root_position=np.array([0.0, np.nan, 0.0], dtype=np.float64),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            joint_angles=np.zeros(17, dtype=np.float64),
        )
        error = val.validate(pose)
        assert error is not None
        assert "position" in error

    def test_pose_rejects_wrong_joint_count(self) -> None:
        """Validator rejects wrong number of joint angles."""
        val = InitialPoseValidator("drake")
        pose = InitialPose(
            root_position=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            joint_angles=np.zeros(10, dtype=np.float64),  # wrong count
        )
        error = val.validate(pose)
        assert error is not None
        assert "17 values" in error

    def test_validate_raise_invalid_pose(self) -> None:
        """validate_raise() raises ValueError for invalid pose."""
        val = InitialPoseValidator("drake")
        pose = InitialPose(
            root_position=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            root_quat=np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64),
            joint_angles=np.zeros(17, dtype=np.float64),
        )
        with pytest.raises(ValueError):
            val.validate_raise(pose)


# ============================================================================
# Module-level Entry Points (4 tests)
# ============================================================================


class TestModuleLevelEntryPoints:
    """Tests for validate_theta_contract and validate_initial_pose functions."""

    def test_validate_theta_contract_entry_point(self) -> None:
        """validate_theta_contract() works end-to-end."""
        theta = np.zeros(23, dtype=np.float64)
        error = validate_theta_contract(theta, "drake")
        assert error is None

    def test_validate_theta_contract_custom_dof(self) -> None:
        """validate_theta_contract() accepts custom DOF."""
        theta = np.zeros(15, dtype=np.float64)
        error = validate_theta_contract(theta, "mujoco", n_dof=15)
        assert error is None

    def test_validate_initial_pose_entry_point(self) -> None:
        """validate_initial_pose() works end-to-end."""
        pose = InitialPose(
            root_position=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            joint_angles=np.zeros(17, dtype=np.float64),
        )
        error = validate_initial_pose(pose, "drake")
        assert error is None

    def test_validate_initial_pose_custom_dof(self) -> None:
        """validate_initial_pose() accepts custom DOF."""
        pose = InitialPose(
            root_position=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            joint_angles=np.zeros(11, dtype=np.float64),
        )
        error = validate_initial_pose(pose, "mujoco", n_dof=17)
        assert error is None


# ============================================================================
# Cross-Engine Contract Tests (6 tests)
# ============================================================================


class TestCrossEngineContracts:
    """Tests for theta/pose validation across all 4 engines."""

    @pytest.mark.parametrize("engine", ["drake", "opensim", "pinocchio"])
    def test_theta_all_freeflyer_engines(self, engine: str) -> None:
        """Theta validation works for all 3 free-flyer engines."""
        n_dof = ENGINE_DOF_MAP[engine]
        val = ThetaContractValidator(engine)
        theta = np.linspace(-0.1, 0.1, n_dof, dtype=np.float64)
        error = val.validate(theta)
        assert error is None

    def test_theta_mujoco_fixed_root(self) -> None:
        """Theta validation for MuJoCo (fixed root, 17 DOF)."""
        val = ThetaContractValidator("mujoco")
        theta = np.zeros(17, dtype=np.float64)
        error = val.validate(theta)
        assert error is None

    @pytest.mark.parametrize("engine", ["drake", "opensim", "pinocchio"])
    def test_pose_all_freeflyer_engines(self, engine: str) -> None:
        """Pose validation works for all 3 free-flyer engines."""
        val = InitialPoseValidator(engine)
        pose = InitialPose(
            root_position=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            joint_angles=np.zeros(17, dtype=np.float64),
        )
        error = val.validate(pose)
        assert error is None

    def test_pose_mujoco_fixed_root(self) -> None:
        """Pose validation for MuJoCo with 11 joint angles."""
        val = InitialPoseValidator("mujoco")
        pose = InitialPose(
            root_position=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            joint_angles=np.zeros(11, dtype=np.float64),
        )
        error = val.validate(pose)
        assert error is None

    def test_contract_enforcement_levels(self) -> None:
        """Contracts work with both float32 and float64."""
        val = ThetaContractValidator("drake")
        theta_f32 = np.zeros(23, dtype=np.float32)
        theta_f64 = np.zeros(23, dtype=np.float64)
        assert val.validate(theta_f32) is None
        assert val.validate(theta_f64) is None

    def test_boundary_theta_values(self) -> None:
        """Theta validation respects boundary values."""
        val = ThetaContractValidator("drake")
        # Test near upper bound
        theta_high = np.ones(23, dtype=np.float64) * 999.9
        assert val.validate(theta_high) is None
        # Test at upper bound
        theta_limit = np.ones(23, dtype=np.float64) * 1000.0
        assert val.validate(theta_limit) is None


# ============================================================================
# Error Handling and Edge Cases (2 tests)
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_fitresult_is_frozen(self) -> None:
        """FitResult is a frozen dataclass (immutable)."""
        result = FitResult(
            coefficients=np.array([0.1], dtype=np.float32),
            final_loss=0.05,
            metadata={"engine": "drake", "time_s": 1.0},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.final_loss = 0.1  # type: ignore[misc]

    def test_initial_pose_is_namedtuple(self) -> None:
        """InitialPose is immutable (named tuple)."""
        pose = InitialPose(
            root_position=np.array([0.0, 0.0, 1.0]),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            joint_angles=np.zeros(17),
        )
        with pytest.raises(AttributeError):
            pose.root_position = np.array([1.0, 1.0, 1.0])  # type: ignore[misc]
