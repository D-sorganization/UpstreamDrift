"""
Unit tests for motion pipeline contracts (CIR).

Tests cover:
- Pydantic v2 validation
- Invariants via @invariant decorator
- Round-trip serialization (JSON)
- Edge cases and error handling
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    CameraExtrinsics,
    CameraIntrinsics,
    JointDef,
    JointLimit,
    JointStateFrame,
    JointTrajectory,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    MotionMatchingRequest,
    MotionMatchingResult,
    MotionTrajectory,
    SkeletonRig,
    UpAxis,
    deserialize_model,
    load_model,
    save_model,
    serialize_model,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_intrinsics() -> CameraIntrinsics:
    """Create a sample camera intrinsics."""
    return CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=480.0)


@pytest.fixture
def sample_extrinsics() -> CameraExtrinsics:
    """Create a sample camera extrinsics."""
    return CameraExtrinsics(
        rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        translation=[0.0, 0.0, 0.0],
    )


@pytest.fixture
def sample_calibration() -> Calibration:
    """Create a sample calibration."""
    return Calibration(
        id="calib_001",
        cameras={
            "cam_0": {
                "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 480.0},
                "extrinsics": {
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation": [0, 0, 0],
                },
            }
        },
        unit_system="meters",
        source_fps=30.0,
        world_up_axis="+Y",
    )


@pytest.fixture
def sample_keypoint_frame() -> KeypointFrame:
    """Create a sample keypoint frame."""
    return KeypointFrame(
        timestamp=0.0,
        keypoints=[
            Keypoint(x=100.0, y=200.0, z=0.5, confidence=0.95, name="nose"),
            Keypoint(x=110.0, y=210.0, z=0.6, confidence=0.90, name="left_eye"),
        ],
        schema_name="BODY_25",
        frame_index=0,
    )


@pytest.fixture
def sample_skeleton() -> SkeletonRig:
    """Create a sample skeleton rig."""
    return SkeletonRig(
        id="skeleton_001",
        joints={
            "pelvis": JointDef(
                name="pelvis", parent=None, children=["spine"], tpose_offset=[0, 0, 0]
            ),
            "spine": JointDef(
                name="spine",
                parent="pelvis",
                children=["neck"],
                tpose_offset=[0, 0.1, 0],
            ),
            "neck": JointDef(
                name="neck", parent="spine", children=[], tpose_offset=[0, 0.15, 0]
            ),
        },
        root_joint="pelvis",
        up_axis="+Y",
    )


@pytest.fixture
def sample_joint_trajectory(sample_skeleton: SkeletonRig) -> JointTrajectory:
    """Create a sample joint trajectory."""
    frames = [
        JointStateFrame(
            timestamp=0.0, q=[0.0] * sample_skeleton.num_dofs, frame_index=0
        ),
        JointStateFrame(
            timestamp=0.033, q=[0.1] * sample_skeleton.num_dofs, frame_index=1
        ),
        JointStateFrame(
            timestamp=0.066, q=[0.2] * sample_skeleton.num_dofs, frame_index=2
        ),
    ]
    return JointTrajectory(id="traj_001", skeleton=sample_skeleton, frames=frames)


@pytest.fixture
def sample_motion_trajectory(
    sample_skeleton: SkeletonRig, sample_joint_trajectory: JointTrajectory
) -> MotionTrajectory:
    """Create a sample motion trajectory."""
    return MotionTrajectory(
        id="motion_001",
        skeleton=sample_skeleton,
        trajectory=sample_joint_trajectory,
        subject={"height_m": 1.75, "mass_kg": 70.0, "age": 25},
        sport="golf",
        club="driver",
    )


# =============================================================================
# Calibration Tests
# =============================================================================


class TestCameraIntrinsics:
    """Tests for CameraIntrinsics model."""

    def test_valid_intrinsics(self, sample_intrinsics: CameraIntrinsics):
        """Test valid intrinsics creation."""
        assert sample_intrinsics.fx == 800.0
        assert sample_intrinsics.fy == 800.0
        assert sample_intrinsics.cx == 640.0
        assert sample_intrinsics.cy == 480.0

    def test_invalid_focal_length_zero(self):
        """Test that zero focal length raises error."""
        with pytest.raises(ValueError, match="greater than 0"):
            CameraIntrinsics(fx=0.0, fy=800.0, cx=640.0, cy=480.0)

    def test_invalid_focal_length_negative(self):
        """Test that negative focal length raises error."""
        with pytest.raises(ValueError, match="greater than 0"):
            CameraIntrinsics(fx=-800.0, fy=800.0, cx=640.0, cy=480.0)

    def test_nan_focal_length(self):
        """Test that NaN focal length raises error."""
        with pytest.raises(ValueError, match="finite"):
            CameraIntrinsics(fx=float("nan"), fy=800.0, cx=640.0, cy=480.0)

    def test_inf_focal_length(self):
        """Test that infinite focal length raises error."""
        with pytest.raises(ValueError, match="finite"):
            CameraIntrinsics(fx=float("inf"), fy=800.0, cx=640.0, cy=480.0)

    def test_default_distortion(self):
        """Test default distortion coefficients."""
        intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=480.0)
        assert intrinsics.k1 == 0.0
        assert intrinsics.k2 == 0.0
        assert intrinsics.p1 == 0.0
        assert intrinsics.p2 == 0.0


class TestCameraExtrinsics:
    """Tests for CameraExtrinsics model."""

    def test_valid_extrinsics(self, sample_extrinsics: CameraExtrinsics):
        """Test valid extrinsics creation."""
        assert len(sample_extrinsics.rotation) == 3
        assert len(sample_extrinsics.translation) == 3

    def test_default_rotation_identity(self):
        """Test default rotation is identity."""
        extrinsics = CameraExtrinsics()
        assert extrinsics.rotation == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_default_translation_zero(self):
        """Test default translation is zero."""
        extrinsics = CameraExtrinsics()
        assert extrinsics.translation == [0.0, 0.0, 0.0]

    def test_invalid_rotation_shape(self):
        """Test that invalid rotation shape raises error."""
        with pytest.raises(ValueError, match="3x3"):
            CameraExtrinsics(rotation=[[1, 0], [0, 1]], translation=[0, 0, 0])

    def test_invalid_translation_shape(self):
        """Test that invalid translation shape raises error."""
        with pytest.raises(ValueError, match="length 3"):
            CameraExtrinsics(
                rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation=[0, 0]
            )


class TestCalibration:
    """Tests for Calibration model."""

    def test_valid_calibration(self, sample_calibration: Calibration):
        """Test valid calibration creation."""
        assert sample_calibration.id == "calib_001"
        assert sample_calibration.unit_system == "meters"
        assert sample_calibration.source_fps == 30.0
        assert sample_calibration.world_up_axis == "+Y"

    def test_invalid_fps_zero(self):
        """Test that zero FPS raises error."""
        with pytest.raises(ValueError, match="greater than 0"):
            Calibration(
                id="calib_001",
                cameras={
                    "cam_0": {
                        "intrinsics": {
                            "fx": 800.0,
                            "fy": 800.0,
                            "cx": 640.0,
                            "cy": 480.0,
                        }
                    }
                },
                source_fps=0.0,
            )

    def test_missing_intrinsics(self):
        """Test that missing intrinsics raises error."""
        with pytest.raises(ValueError, match="missing intrinsics"):
            Calibration(
                id="calib_001",
                cameras={
                    "cam_0": {
                        "extrinsics": {
                            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            "translation": [0, 0, 0],
                        }
                    }
                },
                source_fps=30.0,
            )

    def test_invalid_unit_system(self):
        """Test that invalid unit system raises error."""
        with pytest.raises(ValueError):
            Calibration(
                id="calib_001",
                cameras={
                    "cam_0": {
                        "intrinsics": {
                            "fx": 800.0,
                            "fy": 800.0,
                            "cx": 640.0,
                            "cy": 480.0,
                        }
                    }
                },
                unit_system="invalid",  # type: ignore
                source_fps=30.0,
            )

    def test_invalid_up_axis(self):
        """Test that invalid up axis raises error."""
        with pytest.raises(ValueError):
            Calibration(
                id="calib_001",
                cameras={
                    "cam_0": {
                        "intrinsics": {
                            "fx": 800.0,
                            "fy": 800.0,
                            "cx": 640.0,
                            "cy": 480.0,
                        }
                    }
                },
                source_fps=30.0,
                world_up_axis="invalid",  # type: ignore
            )


# =============================================================================
# Keypoint Tests
# =============================================================================


class TestKeypoint:
    """Tests for Keypoint model."""

    def test_valid_3d_keypoint(self):
        """Test valid 3D keypoint creation."""
        kp = Keypoint(x=100.0, y=200.0, z=0.5, confidence=0.95, name="nose")
        assert kp.x == 100.0
        assert kp.y == 200.0
        assert kp.z == 0.5
        assert kp.confidence == 0.95

    def test_valid_2d_keypoint(self):
        """Test valid 2D keypoint creation."""
        kp = Keypoint(x=100.0, y=200.0, confidence=0.95)
        assert kp.x == 100.0
        assert kp.y == 200.0
        assert kp.z is None

    def test_confidence_bounds_zero(self):
        """Test confidence at lower bound."""
        kp = Keypoint(x=100.0, y=200.0, confidence=0.0)
        assert kp.confidence == 0.0

    def test_confidence_bounds_one(self):
        """Test confidence at upper bound."""
        kp = Keypoint(x=100.0, y=200.0, confidence=1.0)
        assert kp.confidence == 1.0

    def test_confidence_negative(self):
        """Test that negative confidence raises error."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            Keypoint(x=100.0, y=200.0, confidence=-0.1)

    def test_confidence_over_one(self):
        """Test that confidence over 1 raises error."""
        with pytest.raises(ValueError, match="less than or equal to 1"):
            Keypoint(x=100.0, y=200.0, confidence=1.1)

    def test_nan_coordinate(self):
        """Test that NaN coordinate raises error."""
        with pytest.raises(ValueError, match="finite"):
            Keypoint(x=float("nan"), y=200.0)


class TestKeypointFrame:
    """Tests for KeypointFrame model."""

    def test_valid_frame(self, sample_keypoint_frame: KeypointFrame):
        """Test valid keypoint frame creation."""
        assert sample_keypoint_frame.timestamp == 0.0
        assert len(sample_keypoint_frame.keypoints) == 2
        assert sample_keypoint_frame.schema_name == "BODY_25"

    def test_empty_keypoints(self):
        """Test that empty keypoints raises error."""
        with pytest.raises(ValueError, match="at least 1"):
            KeypointFrame(timestamp=0.0, keypoints=[], schema_name="BODY_25")

    def test_negative_timestamp(self):
        """Test that negative timestamp raises error."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            KeypointFrame(
                timestamp=-1.0,
                keypoints=[Keypoint(x=100.0, y=200.0)],
                schema_name="BODY_25",
            )

    def test_depth_consistency_3d(self):
        """Test 3D keypoints pass depth consistency check."""
        frame = KeypointFrame(
            timestamp=0.0,
            keypoints=[
                Keypoint(x=100.0, y=200.0, z=0.5),
                Keypoint(x=110.0, y=210.0, z=0.6),
            ],
            schema_name="BODY_25",
        )
        assert frame.check_keypoint_depth_consistency()

    def test_depth_consistency_2d(self):
        """Test 2D keypoints pass depth consistency check."""
        frame = KeypointFrame(
            timestamp=0.0,
            keypoints=[
                Keypoint(x=100.0, y=200.0),
                Keypoint(x=110.0, y=210.0),
            ],
            schema_name="BODY_25",
        )
        assert frame.check_keypoint_depth_consistency()

    def test_depth_inconsistency_mixed(self):
        """Test mixed 2D/3D keypoints fail depth consistency check."""
        frame = KeypointFrame(
            timestamp=0.0,
            keypoints=[
                Keypoint(x=100.0, y=200.0, z=0.5),
                Keypoint(x=110.0, y=210.0),
            ],
            schema_name="BODY_25",
        )
        assert not frame.check_keypoint_depth_consistency()


class TestKeypointSequence:
    """Tests for KeypointSequence model."""

    def test_valid_sequence(self, sample_keypoint_frame: KeypointFrame):
        """Test valid keypoint sequence creation."""
        seq = KeypointSequence(id="seq_001", frames=[sample_keypoint_frame])
        assert seq.id == "seq_001"
        assert seq.num_frames == 1
        assert seq.num_keypoints == 2

    def test_monotonic_timestamps(self, sample_keypoint_frame: KeypointFrame):
        """Test monotonically increasing timestamps."""
        frame2 = KeypointFrame(
            timestamp=0.033,
            keypoints=[Keypoint(x=105.0, y=205.0)],
            schema_name="BODY_25",
        )
        seq = KeypointSequence(id="seq_001", frames=[sample_keypoint_frame, frame2])
        assert seq.duration == 0.033

    def test_non_monotonic_timestamps(self, sample_keypoint_frame: KeypointFrame):
        """Test that non-monotonic timestamps raise error.

        Uses three frames where the third comes before the second so that each frame
        is individually valid (timestamp >= 0 satisfied) but the sequence is not
        monotonically increasing.
        """
        frame2 = KeypointFrame(
            timestamp=0.1,
            keypoints=[Keypoint(x=105.0, y=205.0)],
            schema_name="BODY_25",
        )
        frame3 = KeypointFrame(
            timestamp=0.05,  # earlier than frame2 — violates monotonicity
            keypoints=[Keypoint(x=107.0, y=207.0)],
            schema_name="BODY_25",
        )
        with pytest.raises(ValueError, match="monotonically increasing"):
            KeypointSequence(
                id="seq_001", frames=[sample_keypoint_frame, frame2, frame3]
            )

    def test_consistent_schema(self, sample_keypoint_frame: KeypointFrame):
        """Test consistent schema across frames."""
        frame2 = KeypointFrame(
            timestamp=0.033,
            keypoints=[Keypoint(x=105.0, y=205.0)],
            schema_name="BODY_25",
        )
        seq = KeypointSequence(id="seq_001", frames=[sample_keypoint_frame, frame2])
        assert seq.check_consistent_schema()

    def test_inconsistent_schema(self, sample_keypoint_frame: KeypointFrame):
        """Test that inconsistent schema raises error."""
        frame2 = KeypointFrame(
            timestamp=0.033,
            keypoints=[Keypoint(x=105.0, y=205.0)],
            schema_name="COCO_17",
        )
        with pytest.raises(ValueError, match="Inconsistent schemas"):
            KeypointSequence(id="seq_001", frames=[sample_keypoint_frame, frame2])


# =============================================================================
# Skeleton Tests
# =============================================================================


class TestSkeletonRig:
    """Tests for SkeletonRig model."""

    def test_valid_skeleton(self, sample_skeleton: SkeletonRig):
        """Test valid skeleton creation."""
        assert sample_skeleton.id == "skeleton_001"
        assert sample_skeleton.root_joint == "pelvis"
        assert sample_skeleton.num_joints == 3

    def test_root_not_exists(self):
        """Test that non-existent root raises error."""
        with pytest.raises(ValueError, match="not found"):
            SkeletonRig(
                id="skeleton_001",
                joints={"pelvis": JointDef(name="pelvis")},
                root_joint="nonexistent",
            )

    def test_invalid_child(self):
        """Test that invalid child reference raises error."""
        with pytest.raises(ValueError, match="invalid child"):
            SkeletonRig(
                id="skeleton_001",
                joints={
                    "pelvis": JointDef(name="pelvis", children=["nonexistent"]),
                },
                root_joint="pelvis",
            )

    def test_invalid_parent(self):
        """Test that invalid parent reference raises error."""
        with pytest.raises(ValueError, match="invalid parent"):
            SkeletonRig(
                id="skeleton_001",
                joints={
                    "pelvis": JointDef(name="pelvis", parent="nonexistent"),
                },
                root_joint="pelvis",
            )

    def test_joint_chain(self, sample_skeleton: SkeletonRig):
        """Test joint chain retrieval."""
        chain = sample_skeleton.get_joint_chain("neck")
        assert chain == ["pelvis", "spine", "neck"]

    def test_num_dofs(self):
        """Test DOF calculation."""
        skeleton = SkeletonRig(
            id="skeleton_001",
            joints={
                "pelvis": JointDef(name="pelvis", axes=["X", "Y", "Z"]),
                "spine": JointDef(name="spine", axes=["X", "Y", "Z"]),
            },
            root_joint="pelvis",
        )
        assert skeleton.num_dofs == 6


class TestJointLimit:
    """Tests for JointLimit model."""

    def test_valid_limit(self):
        """Test valid joint limit creation."""
        limit = JointLimit(lower=-0.5, upper=0.5)
        assert limit.lower == -0.5
        assert limit.upper == 0.5

    def test_limit_order_valid(self):
        """Test valid limit order."""
        limit = JointLimit(lower=-0.5, upper=0.5)
        assert limit.check_limit_order()

    def test_limit_order_invalid(self):
        """Test that invalid limit order raises error."""
        with pytest.raises(ValueError, match="Lower limit must be"):
            JointLimit(lower=0.5, upper=-0.5)

    def test_optional_limits(self):
        """Test optional limits."""
        limit = JointLimit()
        assert limit.lower is None
        assert limit.upper is None


# =============================================================================
# Joint Trajectory Tests
# =============================================================================


class TestJointStateFrame:
    """Tests for JointStateFrame model."""

    def test_valid_frame(self):
        """Test valid joint state frame creation."""
        frame = JointStateFrame(timestamp=0.0, q=[0.0, 0.1, 0.2])
        assert frame.num_dofs == 3

    def test_frame_with_velocities(self):
        """Test frame with velocities."""
        frame = JointStateFrame(
            timestamp=0.0,
            q=[0.0, 0.1, 0.2],
            qdot=[0.01, 0.02, 0.03],
        )
        assert frame.qdot == [0.01, 0.02, 0.03]

    def test_frame_with_accelerations(self):
        """Test frame with accelerations."""
        frame = JointStateFrame(
            timestamp=0.0,
            q=[0.0, 0.1, 0.2],
            qdot=[0.01, 0.02, 0.03],
            qddot=[0.001, 0.002, 0.003],
        )
        assert frame.qddot == [0.001, 0.002, 0.003]

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        with pytest.raises(ValueError, match="matching_dimensions"):
            JointStateFrame(
                timestamp=0.0,
                q=[0.0, 0.1, 0.2],
                qdot=[0.01, 0.02],  # Different length
            )

    def test_nan_values(self):
        """Test that NaN values raise error."""
        with pytest.raises(ValueError, match="finite"):
            JointStateFrame(timestamp=0.0, q=[float("nan"), 0.1, 0.2])


class TestJointTrajectory:
    """Tests for JointTrajectory model."""

    def test_valid_trajectory(self, sample_joint_trajectory: JointTrajectory):
        """Test valid joint trajectory creation."""
        assert sample_joint_trajectory.id == "traj_001"
        assert sample_joint_trajectory.num_frames == 3
        assert sample_joint_trajectory.duration > 0

    def test_dof_consistency(self, sample_skeleton: SkeletonRig):
        """Test DOF consistency validation."""
        frames = [
            JointStateFrame(timestamp=0.0, q=[0.0] * sample_skeleton.num_dofs),
            JointStateFrame(
                timestamp=0.033, q=[0.0] * (sample_skeleton.num_dofs + 1)
            ),  # Wrong DOF
        ]
        with pytest.raises(ValueError, match="DOFs"):
            JointTrajectory(id="traj_001", skeleton=sample_skeleton, frames=frames)


# =============================================================================
# Motion Trajectory Tests
# =============================================================================


class TestMotionTrajectory:
    """Tests for MotionTrajectory model."""

    def test_valid_motion(self, sample_motion_trajectory: MotionTrajectory):
        """Test valid motion trajectory creation."""
        assert sample_motion_trajectory.id == "motion_001"
        assert sample_motion_trajectory.sport == "golf"
        assert sample_motion_trajectory.club == "driver"

    def test_skeleton_mismatch(
        self, sample_skeleton: SkeletonRig, sample_joint_trajectory: JointTrajectory
    ):
        """Test that skeleton mismatch raises error."""
        other_skeleton = SkeletonRig(
            id="other_skeleton",
            joints={"pelvis": JointDef(name="pelvis")},
            root_joint="pelvis",
        )
        with pytest.raises(ValueError, match="does not match"):
            MotionTrajectory(
                id="motion_001",
                skeleton=other_skeleton,
                trajectory=sample_joint_trajectory,
            )


# =============================================================================
# Motion Matching Tests
# =============================================================================


class TestMotionMatchingRequest:
    """Tests for MotionMatchingRequest model."""

    def test_request_with_trajectory(self, sample_motion_trajectory: MotionTrajectory):
        """Test request with trajectory target."""
        request = MotionMatchingRequest(
            id="request_001",
            target_trajectory=sample_motion_trajectory,
            skeleton=sample_motion_trajectory.skeleton,
        )
        assert request.id == "request_001"

    def test_request_without_target(self, sample_skeleton: SkeletonRig):
        """Test that missing target raises error."""
        with pytest.raises(ValueError, match="at least one target"):
            MotionMatchingRequest(id="request_001", skeleton=sample_skeleton)


class TestMotionMatchingResult:
    """Tests for MotionMatchingResult model."""

    def test_successful_result(self):
        """Test successful result creation."""
        from src.shared.python.motion_pipeline.contracts import (
            TorqueFrame,
            TorqueTrajectory,
        )

        torques = TorqueTrajectory(
            frames=[
                TorqueFrame(timestamp=0.0, tau=[0.0]),
                TorqueFrame(timestamp=0.01, tau=[0.1]),
            ],
            rig_joint_names=["j0"],
        )
        result = MotionMatchingResult(
            request_id="request_001",
            success=True,
            torques=torques,
            error_metrics={"rmse": 0.05, "max_error": 0.1},
            iterations=10,
            solve_time=0.5,
        )
        assert result.success
        assert result.iterations == 10
        assert result.torques is torques

    def test_failed_result(self):
        """Test failed result creation."""
        result = MotionMatchingResult(
            request_id="request_001",
            success=False,
            message="Solver did not converge",
        )
        assert not result.success
        assert "not converge" in result.message

    def test_negative_solve_time(self):
        """Test that negative solve time raises error.

        Provides torques to satisfy the success-payload invariant so that
        the solve_time validator is the one that triggers.
        """
        from src.shared.python.motion_pipeline.contracts import (
            TorqueFrame,
            TorqueTrajectory,
        )

        torques = TorqueTrajectory(
            frames=[
                TorqueFrame(timestamp=0.0, tau=[0.0]),
                TorqueFrame(timestamp=0.01, tau=[0.1]),
            ],
            rig_joint_names=["j0"],
        )
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            MotionMatchingResult(
                request_id="request_001",
                success=True,
                torques=torques,
                solve_time=-1.0,
            )


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for serialization/deserialization."""

    def test_serialize_calibration(self, sample_calibration: Calibration):
        """Test calibration serialization."""
        json_str = serialize_model(sample_calibration)
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["id"] == "calib_001"

    def test_roundtrip_calibration(self, sample_calibration: Calibration):
        """Test calibration round-trip."""
        json_str = serialize_model(sample_calibration)
        restored = deserialize_model(json_str, Calibration)
        assert restored.id == sample_calibration.id
        assert restored.source_fps == sample_calibration.source_fps

    def test_serialize_motion(self, sample_motion_trajectory: MotionTrajectory):
        """Test motion trajectory serialization."""
        json_str = serialize_model(sample_motion_trajectory)
        data = json.loads(json_str)
        assert data["id"] == "motion_001"
        assert data["sport"] == "golf"

    def test_save_load_file(self, sample_calibration: Calibration):
        """Test save/load to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            save_model(sample_calibration, path)
            assert path.exists()
            restored = load_model(path, Calibration)
            assert restored.id == sample_calibration.id


# =============================================================================
# Marker Tests
# =============================================================================


class TestMarker:
    """Tests for Marker model."""

    def test_valid_marker(self):
        """Test valid marker creation."""
        marker = Marker(name="LASI", x=0.1, y=0.2, z=0.3)
        assert marker.name == "LASI"
        assert marker.x == 0.1

    def test_marker_with_residual(self):
        """Test marker with residual."""
        marker = Marker(name="LASI", x=0.1, y=0.2, z=0.3, residual=0.5)
        assert marker.residual == 0.5

    def test_negative_residual(self):
        """Test that negative residual raises error."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            Marker(name="LASI", x=0.1, y=0.2, z=0.3, residual=-0.1)


class TestMarkerFrame:
    """Tests for MarkerFrame model."""

    def test_valid_frame(self):
        """Test valid marker frame creation."""
        frame = MarkerFrame(
            timestamp=0.0,
            markers={
                "LASI": Marker(name="LASI", x=0.1, y=0.2, z=0.3),
                "RASI": Marker(name="RASI", x=0.2, y=0.2, z=0.3),
            },
        )
        assert frame.num_markers == 2

    def test_marker_names(self):
        """Test marker names property."""
        frame = MarkerFrame(
            timestamp=0.0,
            markers={"LASI": Marker(name="LASI", x=0.1, y=0.2, z=0.3)},
        )
        assert "LASI" in frame.marker_names


class TestMarkerTrajectory:
    """Tests for MarkerTrajectory model."""

    def test_valid_trajectory(self):
        """Test valid marker trajectory creation."""
        frames = [
            MarkerFrame(
                timestamp=0.0,
                markers={"LASI": Marker(name="LASI", x=0.1, y=0.2, z=0.3)},
            ),
            MarkerFrame(
                timestamp=0.033,
                markers={"LASI": Marker(name="LASI", x=0.11, y=0.21, z=0.31)},
            ),
        ]
        traj = MarkerTrajectory(id="traj_001", frames=frames)
        assert traj.num_frames == 2
        assert traj.duration > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
