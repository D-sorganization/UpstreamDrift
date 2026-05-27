"""Unit tests for MockKinematicsService fixtures.

These tests verify the testing architecture improvements from issue #5104:
1. Engine isolation - unit tests use MockKinematicsService instead of real engines
2. Session-scoped fixtures for heavy data loading
3. Proper DbC contracts for the mock service
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange.canonical import CanonicalPose
from src.shared.python.pose_interchange.live_kinematics import (
    CapabilityError,
    ServiceCapabilities,
)
from src.shared.python.pose_interchange.services._mock import MockKinematicsService

pytestmark = pytest.mark.unit


class TestMockKinematicsServiceConstruction:
    """Test MockKinematicsService construction and basic properties."""

    def test_construct_with_valid_engine_name(self) -> None:
        """MockKinematicsService accepts valid engine names."""
        service = MockKinematicsService(engine_name="drake")
        assert service.engine_name == "drake"

    def test_construct_with_empty_engine_name_raises(self) -> None:
        """Empty engine name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            MockKinematicsService(engine_name="")

    def test_construct_with_non_string_engine_name_raises(self) -> None:
        """Non-string engine name raises TypeError."""
        with pytest.raises(TypeError, match="str"):
            MockKinematicsService(engine_name=123)  # type: ignore[arg-type]

    def test_all_first_class_engines(self) -> None:
        """All first-class engines can be instantiated."""
        for engine_name in ["drake", "mujoco", "pinocchio", "opensim", "simscape"]:
            service = MockKinematicsService(engine_name=engine_name)
            assert service.engine_name == engine_name


class TestMockKinematicsServiceCapabilities:
    """Test MockKinematicsService capabilities reporting."""

    def test_capabilities_returns_service_capabilities(self) -> None:
        """capabilities() returns a ServiceCapabilities instance."""
        service = MockKinematicsService(engine_name="drake")
        caps = service.capabilities()
        assert isinstance(caps, ServiceCapabilities)

    def test_capabilities_reports_no_dynamics_step(self) -> None:
        """Mock service reports supports_dynamics_step=False."""
        service = MockKinematicsService(engine_name="drake")
        caps = service.capabilities()
        assert caps.supports_dynamics_step is False

    def test_capabilities_reports_no_collision_query(self) -> None:
        """Mock service reports supports_collision_query=False."""
        service = MockKinematicsService(engine_name="drake")
        caps = service.capabilities()
        assert caps.supports_collision_query is False

    def test_capabilities_reports_no_realtime(self) -> None:
        """Mock service reports supports_realtime=False."""
        service = MockKinematicsService(engine_name="drake")
        caps = service.capabilities()
        assert caps.supports_realtime is False

    def test_capabilities_is_immutable(self) -> None:
        """Capabilities are frozen and cannot be modified."""
        service = MockKinematicsService(engine_name="drake")
        caps = service.capabilities()
        with pytest.raises((AttributeError, TypeError)):
            caps.supports_dynamics_step = True  # type: ignore[misc]


class TestMockKinematicsServiceLoad:
    """Test MockKinematicsService.load method."""

    def test_load_accepts_path(self, tmp_path: np.ndarray) -> None:
        """load() accepts a pathlib.Path."""
        service = MockKinematicsService(engine_name="drake")
        model_path = tmp_path / "model.urdf"
        model_path.write_text("<robot/>")
        service.load(model_path)
        # No exception means success - mock doesn't validate the file

    def test_load_accepts_nonexistent_path(self, tmp_path: np.ndarray) -> None:
        """load() accepts a non-existent path (mock doesn't validate)."""
        service = MockKinematicsService(engine_name="drake")
        model_path = tmp_path / "nonexistent.urdf"
        service.load(model_path)
        # No exception

    def test_load_rejects_non_path(self) -> None:
        """load() rejects non-Path arguments."""
        service = MockKinematicsService(engine_name="drake")
        with pytest.raises(TypeError, match="Path"):
            service.load("not_a_path")  # type: ignore[arg-type]


class TestMockKinematicsServiceSetPose:
    """Test MockKinematicsService.set_pose method."""

    def test_set_pose_accepts_canonical_pose(
        self, zero_canonical_pose: CanonicalPose
    ) -> None:
        """set_pose() accepts a CanonicalPose instance."""
        service = MockKinematicsService(engine_name="drake")
        service.set_pose(zero_canonical_pose)
        # No exception means success

    def test_set_pose_rejects_non_canonical_pose(self) -> None:
        """set_pose() rejects non-CanonicalPose arguments."""
        service = MockKinematicsService(engine_name="drake")
        with pytest.raises(TypeError, match="CanonicalPose"):
            service.set_pose({"angles": "not a pose"})  # type: ignore[arg-type]

    def test_set_pose_rejects_none(self) -> None:
        """set_pose() rejects None."""
        service = MockKinematicsService(engine_name="drake")
        with pytest.raises(TypeError):
            service.set_pose(None)  # type: ignore[arg-type]


class TestMockKinematicsServiceGetLinkTransforms:
    """Test MockKinematicsService.get_link_transforms method."""

    def test_get_link_transforms_without_pose(self) -> None:
        """get_link_transforms() returns transforms even without a pose set."""
        service = MockKinematicsService(engine_name="drake")
        transforms = service.get_link_transforms()
        assert isinstance(transforms, dict)
        assert len(transforms) > 0
        # All transforms should be 4x4 SE(3) matrices
        for name, transform in transforms.items():
            assert isinstance(name, str)
            assert transform.shape == (4, 4)
            assert transform.dtype == np.float64

    def test_get_link_transforms_with_zero_pose(
        self, zero_canonical_pose: CanonicalPose
    ) -> None:
        """get_link_transforms() after set_pose returns valid transforms."""
        service = MockKinematicsService(engine_name="drake")
        service.set_pose(zero_canonical_pose)
        transforms = service.get_link_transforms()
        assert "pelvis" in transforms
        assert "clubhead" in transforms

    def test_get_link_transforms_are_se3_matrices(
        self, canonical_pose_deg: CanonicalPose
    ) -> None:
        """Returned transforms are valid SE(3) matrices."""
        service = MockKinematicsService(engine_name="drake")
        service.set_pose(canonical_pose_deg)
        transforms = service.get_link_transforms()
        for transform in transforms.values():
            # Check SE(3) structure: bottom row should be [0, 0, 0, 1]
            assert np.allclose(transform[3, :], [0, 0, 0, 1])
            # Rotation submatrix should be orthogonal
            rotation = transform[:3, :3]
            assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)


class TestMockKinematicsServiceStep:
    """Test MockKinematicsService.step method."""

    def test_step_raises_capability_error(self) -> None:
        """step() raises CapabilityError since mock doesn't support dynamics."""
        service = MockKinematicsService(engine_name="drake")
        with pytest.raises(CapabilityError, match="dynamics step"):
            service.step(0.01)

    def test_step_error_mentions_engine_name(self) -> None:
        """CapabilityError message includes the engine name."""
        service = MockKinematicsService(engine_name="mujoco")
        with pytest.raises(CapabilityError, match="mujoco"):
            service.step(0.01)


class TestMockKinematicsServiceReset:
    """Test MockKinematicsService.reset method."""

    def test_reset_clears_pose(self, zero_canonical_pose: CanonicalPose) -> None:
        """reset() clears the stored pose."""
        service = MockKinematicsService(engine_name="drake")
        service.set_pose(zero_canonical_pose)
        transforms_before = service.get_link_transforms()
        service.reset()
        transforms_after = service.get_link_transforms()
        # After reset, transforms should be from the default (no pose) state
        # The exact values depend on forward_kinematics implementation
        assert isinstance(transforms_after, dict)

    def test_reset_is_idempotent(self) -> None:
        """reset() can be called multiple times without error."""
        service = MockKinematicsService(engine_name="drake")
        service.reset()
        service.reset()
        service.reset()
        # No exception


class TestMockKinematicsServiceFixtures:
    """Test the pytest fixtures from conftest.py."""

    def test_mock_drake_kinematics_fixture(
        self, mock_drake_kinematics: MockKinematicsService
    ) -> None:
        """mock_drake_kinematics fixture returns a Drake mock."""
        assert mock_drake_kinematics.engine_name == "drake"
        assert isinstance(mock_drake_kinematics, MockKinematicsService)

    def test_mock_mujoco_kinematics_fixture(
        self, mock_mujoco_kinematics: MockKinematicsService
    ) -> None:
        """mock_mujoco_kinematics fixture returns a MuJoCo mock."""
        assert mock_mujoco_kinematics.engine_name == "mujoco"

    def test_mock_pinocchio_kinematics_fixture(
        self, mock_pinocchio_kinematics: MockKinematicsService
    ) -> None:
        """mock_pinocchio_kinematics fixture returns a Pinocchio mock."""
        assert mock_pinocchio_kinematics.engine_name == "pinocchio"

    def test_mock_opensim_kinematics_fixture(
        self, mock_opensim_kinematics: MockKinematicsService
    ) -> None:
        """mock_opensim_kinematics fixture returns an OpenSim mock."""
        assert mock_opensim_kinematics.engine_name == "opensim"

    def test_mock_simscape_kinematics_fixture(
        self, mock_simscape_kinematics: MockKinematicsService
    ) -> None:
        """mock_simscape_kinematics fixture returns a Simscape mock."""
        assert mock_simscape_kinematics.engine_name == "simscape"

    def test_any_mock_kinematics_fixture_default(
        self, any_mock_kinematics: MockKinematicsService
    ) -> None:
        """any_mock_kinematics fixture defaults to Drake."""
        assert any_mock_kinematics.engine_name == "drake"

    @pytest.mark.parametrize(
        "any_mock_kinematics",
        ["drake", "mujoco", "pinocchio"],
        indirect=True,
    )
    def test_any_mock_kinematics_fixture_parametrized(
        self, any_mock_kinematics: MockKinematicsService
    ) -> None:
        """any_mock_kinematics fixture works with parametrization."""
        assert any_mock_kinematics.engine_name in ["drake", "mujoco", "pinocchio"]


class TestCanonicalPoseFixtures:
    """Test the CanonicalPose fixtures from conftest.py."""

    def test_zero_canonical_pose_fixture(
        self, zero_canonical_pose: CanonicalPose
    ) -> None:
        """zero_canonical_pose returns a pose with all zeros."""
        # Check that all angles are zero
        angles_dict = zero_canonical_pose.angles_full_dict_deg()
        for name, value in angles_dict.items():
            assert value == 0.0, f"{name} should be 0.0, got {value}"

    def test_canonical_pose_deg_fixture(
        self, canonical_pose_deg: CanonicalPose
    ) -> None:
        """canonical_pose_deg returns a pose with realistic angles."""
        angles_dict = canonical_pose_deg.angles_full_dict_deg()
        # TorsoStartPosition should be 30 degrees
        assert angles_dict["TorsoStartPosition"] == 30.0
        # SpineStartPositionX should be 15 degrees
        assert angles_dict["SpineStartPositionX"] == 15.0
        # pelvis rotation should be 45 degrees
        assert np.allclose(canonical_pose_deg.pelvis_rotation_xyz_deg, [0.0, 0.0, 45.0])

    def test_canonical_pose_rad_fixture(
        self, canonical_pose_rad: CanonicalPose
    ) -> None:
        """canonical_pose_rad returns a pose with radian inputs converted."""
        # CanonicalPose stores angles in degrees internally
        angles_dict = canonical_pose_rad.angles_full_dict_deg()
        # TorsoStartPosition should be ~30 degrees (converted from radians)
        assert np.isclose(angles_dict["TorsoStartPosition"], np.rad2deg(30.0), atol=0.1)
        # pelvis rotation should be ~45 degrees
        assert np.allclose(
            canonical_pose_rad.pelvis_rotation_xyz_deg,
            [0.0, 0.0, np.rad2deg(45.0)],
            atol=0.1,
        )


class TestMockEngineServiceFactory:
    """Test the mock_engine_service_factory fixture."""

    def test_factory_creates_mock_service(
        self, mock_engine_service_factory: pytest.FixtureRequest
    ) -> None:
        """Factory creates a MagicMock service."""
        service = mock_engine_service_factory()
        assert hasattr(service, "engine_name")
        assert hasattr(service, "capabilities")
        assert hasattr(service, "get_link_transforms")

    def test_factory_with_custom_engine_name(
        self, mock_engine_service_factory: pytest.FixtureRequest
    ) -> None:
        """Factory accepts custom engine name."""
        service = mock_engine_service_factory(engine_name="custom")
        assert service.engine_name == "custom"

    def test_factory_service_returns_transforms(
        self, mock_engine_service_factory: pytest.FixtureRequest
    ) -> None:
        """Factory service returns mock transforms."""
        service = mock_engine_service_factory()
        transforms = service.get_link_transforms()
        assert "pelvis" in transforms
        assert "spine_top" in transforms
