"""Shared fixtures for motion-matching unit tests.

This module provides session-scoped fixtures and mock services to support
the testing architecture overhaul described in issue #5104.

Key design decisions:
1. Heavy C3D data files are loaded exactly once per session using @pytest.fixture(scope="session")
2. MockKinematicsService is provided as the default for unit tests (no real engine required)
3. Engine isolation is enforced - unit tests never spin up real physics engines
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.pose_interchange.canonical import CanonicalPose
from src.shared.python.pose_interchange.services._mock import MockKinematicsService

# =============================================================================
# Session-scoped C3D data fixtures (Issue #5104 - State & Data Management)
# =============================================================================


@pytest.fixture(scope="session")
def c3d_data_dir() -> Path:
    """Return the path to the C3D data directory.

    This fixture is session-scoped to avoid repeated path resolution.
    The actual C3D file loading is done in a separate fixture that
    loads the data exactly once.
    """
    # Search for C3D data in common locations
    possible_paths = [
        Path(__file__).resolve().parents[3] / "Data" / "Mocap C3D Files",
        Path(__file__).resolve().parents[3] / "data",
        Path(__file__).resolve().parents[4] / "Data" / "Mocap C3D Files",
        Path(__file__).resolve().parents[4] / "data",
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return Path("/nonexistent")  # Fallback for CI environments


@pytest.fixture(scope="session")
def real_c3d_path(c3d_data_dir: Path) -> Path | None:
    """Return the path to a real C3D file if available, else None.

    This fixture is session-scoped so the existence check happens once.
    Tests that require real C3D data should skip if this returns None.
    """
    # Try multiple common C3D file names
    candidate_names = [
        "C3D_TA_Driver.c3d",
        "C3DExport Tour average.c3d",
        "test.c3d",
    ]
    for name in candidate_names:
        candidate = c3d_data_dir / name
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="session")
def loaded_c3d_data(real_c3d_path: Path | None) -> dict[str, Any] | None:
    """Load C3D data exactly once per session.

    This addresses the issue #5104 concern about tests reloading multi-megabyte
    C3D dataframes per test, causing severe garbage collection latency.

    Returns:
        A dictionary containing the loaded C3D data structure, or None if
        no real C3D file is available.
    """
    if real_c3d_path is None:
        return None

    try:
        from sidekick.lab.bio.c3d_reader import (
            C3DDataReader,
        )

        reader = C3DDataReader(real_c3d_path)
        # Return the raw c3d data structure for tests to consume
        return reader._load()
    except Exception:  # noqa: BLE001 - fixture skips on any C3D load failure
        return None


# =============================================================================
# MockKinematicsService fixtures (Issue #5104 - Engine Isolation)
# =============================================================================


@pytest.fixture
def mock_drake_kinematics() -> MockKinematicsService:
    """Return a MockKinematicsService impersonating Drake.

    Use this in unit tests to avoid requiring real Drake installation.
    The mock satisfies the LiveKinematicsService Protocol.
    """
    return MockKinematicsService(engine_name="drake")


@pytest.fixture
def mock_mujoco_kinematics() -> MockKinematicsService:
    """Return a MockKinematicsService impersonating MuJoCo."""
    return MockKinematicsService(engine_name="mujoco")


@pytest.fixture
def mock_pinocchio_kinematics() -> MockKinematicsService:
    """Return a MockKinematicsService impersonating Pinocchio."""
    return MockKinematicsService(engine_name="pinocchio")


@pytest.fixture
def mock_opensim_kinematics() -> MockKinematicsService:
    """Return a MockKinematicsService impersonating OpenSim."""
    return MockKinematicsService(engine_name="opensim")


@pytest.fixture
def mock_simscape_kinematics() -> MockKinematicsService:
    """Return a MockKinematicsService impersonating Simscape."""
    return MockKinematicsService(engine_name="simscape")


@pytest.fixture
def any_mock_kinematics(
    request: pytest.FixtureRequest,
    mock_drake_kinematics: MockKinematicsService,
) -> MockKinematicsService:
    """Return any mock kinematics service for parametric tests.

    This fixture accepts an optional 'engine_name' parameter:

        @pytest.mark.parametrize("any_mock_kinematics", ["drake", "mujoco"], indirect=True)
        def test_something(any_mock_kinematics): ...
    """
    engine_name = getattr(request, "param", "drake")
    return MockKinematicsService(engine_name=engine_name)


# =============================================================================
# CanonicalPose fixtures
# =============================================================================


@pytest.fixture
def zero_canonical_pose() -> CanonicalPose:
    """Return a CanonicalPose with all angles at zero degrees.

    This is useful for testing forward kinematics at the neutral pose.
    """
    import numpy as np

    return CanonicalPose(
        pelvis_translation_m=np.zeros(3, dtype=np.float64),
        pelvis_rotation_xyz_deg=np.zeros(3, dtype=np.float64),
        joint_angles_deg={},  # Empty dict means all fields default to 0.0
    )


@pytest.fixture
def canonical_pose_deg() -> CanonicalPose:
    """Return a CanonicalPose with realistic golf swing angles (in degrees).

    This represents a mid-swing pose for testing.
    """
    import numpy as np

    # Use the actual REFERENCE_GOLFER_FIELDS names
    return CanonicalPose(
        pelvis_translation_m=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        pelvis_rotation_xyz_deg=np.array([0.0, 0.0, 45.0], dtype=np.float64),
        joint_angles_deg={
            "TorsoStartPosition": 30.0,
            "SpineStartPositionX": 15.0,
            "LScapStartPositionX": 45.0,
            "RScapStartPositionX": 45.0,
            "LEStartPosition": 30.0,
            "REStartPosition": 60.0,
            "LFStartPosition": 10.0,
            "RFStartPosition": -10.0,
            "HipStartPositionX": 10.0,
        },
    )


@pytest.fixture
def canonical_pose_rad() -> CanonicalPose:
    """Return a CanonicalPose with angles in radians.

    Note: CanonicalPose internally stores angles in degrees, so this
    fixture converts from radians for testing purposes.
    """
    import numpy as np

    return CanonicalPose(
        pelvis_translation_m=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        pelvis_rotation_xyz_deg=np.array(
            [0.0, 0.0, np.rad2deg(45.0)], dtype=np.float64
        ),
        joint_angles_deg={
            "TorsoStartPosition": np.rad2deg(30.0),
            "SpineStartPositionX": np.rad2deg(15.0),
            "LScapStartPositionX": np.rad2deg(45.0),
            "RScapStartPositionX": np.rad2deg(45.0),
            "LEStartPosition": np.rad2deg(30.0),
            "REStartPosition": np.rad2deg(60.0),
            "LFStartPosition": np.rad2deg(10.0),
            "RFStartPosition": np.rad2deg(-10.0),
            "HipStartPositionX": np.rad2deg(10.0),
        },
    )


# =============================================================================
# Mock utilities for engine services
# =============================================================================


@pytest.fixture
def mock_engine_service_factory() -> Any:
    """Return a factory for creating mock engine services.

    This is useful for tests that need to create multiple mock services
    with different configurations.
    """

    def _create(engine_name: str = "mock") -> MagicMock:
        service = MagicMock()
        service.engine_name = engine_name
        service.capabilities.return_value = MagicMock(
            supports_dynamics_step=False,
            supports_collision_query=False,
            supports_realtime=False,
        )
        service.get_link_transforms.return_value = {
            "pelvis": np.eye(4, dtype=np.float64),
            "spine_top": np.eye(4, dtype=np.float64),
            "clubhead": np.eye(4, dtype=np.float64),
        }
        return service

    return _create


@pytest.fixture
def patch_kinematics_registry() -> Any:
    """Context manager for patching the KINEMATICS_SERVICE_REGISTRY.

    Use this to test the registry dispatch logic without requiring
    real engine installations.

    Example:
        with patch_kinematics_registry() as mock_registry:
            mock_registry["drake"] = lambda: MockKinematicsService("drake")
            # ... test code ...
    """
    from src.shared.python.pose_interchange.services import KINEMATICS_SERVICE_REGISTRY

    original_registry = dict(KINEMATICS_SERVICE_REGISTRY)

    def _patch() -> dict[str, Any]:
        # Create a fresh mock registry
        mock_registry = {}
        with patch.dict(KINEMATICS_SERVICE_REGISTRY, mock_registry, clear=True):
            yield mock_registry
        # Restore original registry
        KINEMATICS_SERVICE_REGISTRY.clear()
        KINEMATICS_SERVICE_REGISTRY.update(original_registry)

    return _patch


# =============================================================================
# Test data fixtures
# =============================================================================


@pytest.fixture
def sample_joint_angles() -> dict[str, float]:
    """Return a dictionary of sample joint angles for testing.

    This represents a neutral standing pose.
    """
    return {
        "pelvis_x": 0.0,
        "pelvis_y": 0.0,
        "pelvis_z": 0.0,
        "pelvis_rot_x": 0.0,
        "pelvis_rot_y": 0.0,
        "pelvis_rot_z": 0.0,
        "spine": 0.0,
        "torso": 0.0,
        "l_shoulder": 0.0,
        "r_shoulder": 0.0,
        "l_elbow": 0.0,
        "r_elbow": 0.0,
        "l_wrist": 0.0,
        "r_wrist": 0.0,
        "l_hip": 0.0,
        "r_hip": 0.0,
        "l_knee": 0.0,
        "r_knee": 0.0,
        "l_ankle": 0.0,
        "r_ankle": 0.0,
    }


@pytest.fixture
def sample_skeleton_points() -> dict[str, tuple[float, float, float]]:
    """Return sample skeleton landmark positions for testing."""
    return {
        "pelvis": (0.0, 0.0, 1.0),
        "spine_top": (0.0, 0.0, 1.3),
        "torso_top": (0.0, 0.0, 1.5),
        "l_shoulder": (-0.15, 0.0, 1.45),
        "r_shoulder": (0.15, 0.0, 1.45),
        "l_elbow": (-0.3, 0.0, 1.2),
        "r_elbow": (0.3, 0.0, 1.2),
        "l_wrist": (-0.4, 0.0, 0.9),
        "r_wrist": (0.4, 0.0, 0.9),
        "l_hand": (-0.45, 0.0, 0.85),
        "r_hand": (0.45, 0.0, 0.85),
        "butt": (0.0, 0.0, 0.9),
        "clubhead": (0.5, 0.0, 0.1),
    }


# =============================================================================
# Global Patches for Environmental Issues
# =============================================================================

# Patch pytest.importorskip to catch OSError on Windows
_original_importorskip = pytest.importorskip


def _safe_importorskip(modname, minversion=None, reason=None, **kwargs):
    try:
        return _original_importorskip(
            modname, minversion=minversion, reason=reason, **kwargs
        )
    except OSError as e:
        if "WinError" in str(e):
            pytest.skip(f"Skipping {modname} due to OSError: {e}")
        raise


pytest.importorskip = _safe_importorskip
