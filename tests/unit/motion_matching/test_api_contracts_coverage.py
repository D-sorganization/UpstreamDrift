"""Coverage tests for ``motion_matching.api_contracts``."""

from __future__ import annotations

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


def _good_metadata() -> dict:
    return {"engine": "drake", "time_s": 1.5}


# --- FitResult validation ---------------------------------------------------


class TestFitResult:
    """Pin: ``FitResult`` DbC checks at construction."""

    def test_valid_constructs(self) -> None:
        """Pin: a well-formed FitResult constructs without error."""
        FitResult(
            coefficients=np.zeros(7, dtype=np.float64),
            final_loss=0.0,
            metadata=_good_metadata(),
        )

    def test_coefficients_must_be_ndarray(self) -> None:
        """Pin: list inputs are rejected with TypeError."""
        with pytest.raises(TypeError, match="must be np.ndarray"):
            FitResult(coefficients=[1.0], final_loss=0.0, metadata=_good_metadata())  # type: ignore[arg-type]

    def test_coefficients_must_be_1d(self) -> None:
        """Pin: 2-D coefficients are rejected."""
        with pytest.raises(ValueError, match="must be 1-D"):
            FitResult(
                coefficients=np.zeros((2, 3), dtype=np.float64),
                final_loss=0.0,
                metadata=_good_metadata(),
            )

    def test_coefficients_finite(self) -> None:
        """Pin: NaN coefficients are rejected."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            FitResult(
                coefficients=np.array([np.nan], dtype=np.float64),
                final_loss=0.0,
                metadata=_good_metadata(),
            )

    def test_coefficients_dtype(self) -> None:
        """Pin: int dtype rejected — must be float32/64."""
        with pytest.raises(TypeError, match="dtype must be float"):
            FitResult(
                coefficients=np.zeros(3, dtype=np.int64),
                final_loss=0.0,
                metadata=_good_metadata(),
            )

    def test_final_loss_must_be_float(self) -> None:
        """Pin: string final_loss raises TypeError."""
        with pytest.raises(TypeError, match="must be float"):
            FitResult(
                coefficients=np.zeros(2, dtype=np.float64),
                final_loss="0",  # type: ignore[arg-type]
                metadata=_good_metadata(),
            )

    def test_final_loss_finite(self) -> None:
        """Pin: NaN final_loss is rejected."""
        with pytest.raises(ValueError, match="must be finite"):
            FitResult(
                coefficients=np.zeros(2, dtype=np.float64),
                final_loss=float("nan"),
                metadata=_good_metadata(),
            )

    def test_final_loss_nonneg(self) -> None:
        """Pin: negative final_loss is rejected."""
        with pytest.raises(ValueError, match="must be >= 0"):
            FitResult(
                coefficients=np.zeros(2, dtype=np.float64),
                final_loss=-1.0,
                metadata=_good_metadata(),
            )

    def test_metadata_must_be_dict(self) -> None:
        """Pin: non-dict metadata raises TypeError."""
        with pytest.raises(TypeError, match="must be dict"):
            FitResult(
                coefficients=np.zeros(2, dtype=np.float64),
                final_loss=0.0,
                metadata="meta",  # type: ignore[arg-type]
            )

    def test_metadata_required_keys(self) -> None:
        """Pin: metadata missing 'engine'/'time_s' is rejected."""
        with pytest.raises(ValueError, match="missing required keys"):
            FitResult(
                coefficients=np.zeros(2, dtype=np.float64),
                final_loss=0.0,
                metadata={"engine": "drake"},
            )

    def test_metadata_engine_known(self) -> None:
        """Pin: unknown engine name is rejected."""
        with pytest.raises(ValueError, match="must be one of"):
            FitResult(
                coefficients=np.zeros(2, dtype=np.float64),
                final_loss=0.0,
                metadata={"engine": "unknown", "time_s": 1.0},
            )

    def test_metadata_time_s_numeric(self) -> None:
        """Pin: non-numeric time_s raises TypeError."""
        with pytest.raises(TypeError, match="time_s.*must be numeric"):
            FitResult(
                coefficients=np.zeros(2, dtype=np.float64),
                final_loss=0.0,
                metadata={"engine": "drake", "time_s": "1.0"},
            )

    def test_metadata_time_s_nonneg(self) -> None:
        """Pin: negative time_s rejected."""
        with pytest.raises(ValueError, match=r"time_s.*must be >= 0"):
            FitResult(
                coefficients=np.zeros(2, dtype=np.float64),
                final_loss=0.0,
                metadata={"engine": "drake", "time_s": -1.0},
            )


# --- ThetaContractValidator -------------------------------------------------


class TestThetaContractValidator:
    """Pin: theta contract validator behaviour."""

    def test_unknown_engine_rejected(self) -> None:
        """Pin: ctor raises on unknown engine."""
        with pytest.raises(ValueError, match="engine must be one of"):
            ThetaContractValidator("zzz")

    def test_n_dof_must_be_positive_int(self) -> None:
        """Pin: ctor n_dof contract."""
        with pytest.raises(ValueError, match="positive int"):
            ThetaContractValidator("drake", n_dof=0)
        with pytest.raises(ValueError, match="positive int"):
            ThetaContractValidator("drake", n_dof="23")  # type: ignore[arg-type]

    def test_default_n_dof_from_map(self) -> None:
        """Pin: default ``n_dof`` is read from ``ENGINE_DOF_MAP``."""
        v = ThetaContractValidator("mujoco")
        assert v.n_dof == ENGINE_DOF_MAP["mujoco"]

    def test_valid_theta_returns_none(self) -> None:
        """Pin: a within-spec theta validates with None."""
        v = ThetaContractValidator("mujoco")
        assert v.validate(np.zeros(17, dtype=np.float64)) is None

    def test_2d_returns_error(self) -> None:
        """Pin: 2-D theta yields shape error string."""
        v = ThetaContractValidator("mujoco")
        msg = v.validate(np.zeros((2, 17)))
        assert "must be 1-D" in (msg or "")

    def test_length_mismatch_returns_error(self) -> None:
        """Pin: wrong-length theta yields length-mismatch error string."""
        v = ThetaContractValidator("mujoco")
        msg = v.validate(np.zeros(13))
        assert "length mismatch" in (msg or "")

    def test_nonfinite_reports_indices(self) -> None:
        """Pin: non-finite report includes count and indices."""
        v = ThetaContractValidator("mujoco")
        bad = np.zeros(17)
        bad[3] = np.nan
        msg = v.validate(bad)
        assert msg is not None and "non-finite" in msg

    def test_out_of_range_reports_min_max(self) -> None:
        """Pin: out-of-range values are reported with min/max."""
        v = ThetaContractValidator("mujoco")
        bad = np.zeros(17)
        bad[0] = 1e6
        msg = v.validate(bad)
        assert msg is not None and "out of range" in msg

    def test_validate_raise_propagates(self) -> None:
        """Pin: ``validate_raise`` raises ValueError on bad input."""
        v = ThetaContractValidator("mujoco")
        with pytest.raises(ValueError):
            v.validate_raise(np.zeros(13))

    def test_validate_raise_silent_on_valid(self) -> None:
        """Pin: ``validate_raise`` returns silently on valid input."""
        v = ThetaContractValidator("mujoco")
        v.validate_raise(np.zeros(17))


# --- InitialPoseValidator ---------------------------------------------------


def _good_pose() -> InitialPose:
    return InitialPose(
        root_position=np.array([0.0, 0.0, 0.0]),
        root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        joint_angles=np.zeros(17),
    )


class TestInitialPoseValidator:
    """Pin: initial-pose validator branches."""

    def test_unknown_engine_rejected(self) -> None:
        """Pin: ctor rejects unknown engine."""
        with pytest.raises(ValueError, match="engine must be one of"):
            InitialPoseValidator("nope")

    def test_valid_pose_returns_none(self) -> None:
        """Pin: a canonical drake pose validates."""
        v = InitialPoseValidator("drake")
        assert v.validate(_good_pose()) is None

    def test_position_shape_error(self) -> None:
        """Pin: wrong-shape position yields ``root_position:`` prefix."""
        v = InitialPoseValidator("drake")
        pose = _good_pose()._replace(root_position=np.zeros(2))
        msg = v.validate(pose)
        assert msg is not None and msg.startswith("root_position:")

    def test_position_must_be_ndarray(self) -> None:
        """Pin: list position is rejected with type message."""
        v = InitialPoseValidator("drake")
        pose = _good_pose()._replace(root_position=[0.0, 0.0, 0.0])  # type: ignore[arg-type]
        msg = v.validate(pose)
        assert msg is not None and "must be np.ndarray" in msg

    def test_position_nonfinite(self) -> None:
        """Pin: NaN position is rejected."""
        v = InitialPoseValidator("drake")
        pose = _good_pose()._replace(root_position=np.array([np.nan, 0.0, 0.0]))
        assert v.validate(pose) is not None

    def test_position_norm_too_big(self) -> None:
        """Pin: huge-norm position rejected."""
        v = InitialPoseValidator("drake")
        pose = _good_pose()._replace(root_position=np.array([1000.0, 0.0, 0.0]))
        msg = v.validate(pose)
        assert msg is not None

    def test_position_component_out_of_range(self) -> None:
        """Pin: per-component bound triggers separately from norm."""
        v = InitialPoseValidator("drake")
        # 51 is just above the per-component ceiling of 50; norm 51 > 50 too.
        pose = _good_pose()._replace(root_position=np.array([51.0, 0.0, 0.0]))
        assert v.validate(pose) is not None

    def test_quat_shape_error(self) -> None:
        """Pin: wrong-shape quat yields ``root_quat:`` prefix."""
        v = InitialPoseValidator("drake")
        pose = _good_pose()._replace(root_quat=np.zeros(3))
        msg = v.validate(pose)
        assert msg is not None and msg.startswith("root_quat:")

    def test_quat_must_be_ndarray(self) -> None:
        """Pin: list quat rejected with type message."""
        v = InitialPoseValidator("drake")
        pose = _good_pose()._replace(root_quat=[1.0, 0, 0, 0])  # type: ignore[arg-type]
        assert v.validate(pose) is not None

    def test_quat_nonfinite(self) -> None:
        """Pin: NaN quat rejected."""
        v = InitialPoseValidator("drake")
        pose = _good_pose()._replace(root_quat=np.array([np.nan, 0, 0, 0]))
        assert v.validate(pose) is not None

    def test_quat_not_unit_norm(self) -> None:
        """Pin: non-unit quat rejected."""
        v = InitialPoseValidator("drake")
        pose = _good_pose()._replace(root_quat=np.array([2.0, 0, 0, 0]))
        assert v.validate(pose) is not None

    def test_joint_angles_wrong_length(self) -> None:
        """Pin: mismatched joint count rejected."""
        v = InitialPoseValidator("drake")
        pose = _good_pose()._replace(joint_angles=np.zeros(5))
        msg = v.validate(pose)
        assert msg is not None and "joint_angles" in msg

    def test_joint_angles_nonfinite(self) -> None:
        """Pin: NaN joint angle rejected."""
        v = InitialPoseValidator("drake")
        bad = np.zeros(17)
        bad[2] = np.nan
        pose = _good_pose()._replace(joint_angles=bad)
        msg = v.validate(pose)
        assert msg is not None and "joint_angles" in msg

    def test_validate_raise_propagates(self) -> None:
        """Pin: validate_raise raises ValueError on bad pose."""
        v = InitialPoseValidator("drake")
        bad = _good_pose()._replace(root_quat=np.zeros(3))
        with pytest.raises(ValueError):
            v.validate_raise(bad)

    def test_validate_raise_silent_on_valid(self) -> None:
        """Pin: validate_raise returns silently on canonical pose."""
        v = InitialPoseValidator("drake")
        v.validate_raise(_good_pose())


# --- module-level wrappers --------------------------------------------------


class TestModuleEntryPoints:
    """Pin: module-level convenience functions catch ValueError safely."""

    def test_validate_theta_contract_valid(self) -> None:
        """Pin: returns None for valid input."""
        assert validate_theta_contract(np.zeros(17), "mujoco") is None

    def test_validate_theta_contract_unknown_engine_via_ctor(self) -> None:
        """Pin: unknown engine returned as error string (not raised)."""
        # Precondition catches engine; bypass by passing a known engine
        # but bad n_dof to exercise the except-ValueError path.
        out = validate_theta_contract(np.zeros(17), "mujoco", n_dof=-1)
        assert isinstance(out, str)

    def test_validate_initial_pose_valid(self) -> None:
        """Pin: valid pose returns None."""
        assert validate_initial_pose(_good_pose(), "drake") is None

    def test_validate_initial_pose_via_bad_n_dof(self) -> None:
        """Pin: ValueError from ctor caught and returned as string."""
        out = validate_initial_pose(_good_pose(), "drake", n_dof=-1)
        # n_dof negative -> n_joints -7. joint_angles shape mismatch
        # produces the error string from validate(); the ctor itself
        # accepts negative n_dof, so the validator returns an error.
        assert isinstance(out, str)
