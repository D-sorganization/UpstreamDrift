"""Tests for OpenSim core module.

These tests verify that OpenSim failures produce clear errors, not silent fallbacks.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.engines.physics_engines.opensim.python.opensim_golf.core import (
    GolfSwingModel,
    OpenSimModelLoadError,
    OpenSimNotInstalledError,
)


@pytest.fixture
def mock_opensim_env():
    """Context manager to mock opensim environment."""
    mock_opensim = MagicMock()
    model_mock = MagicMock()
    mock_opensim.Model.return_value = model_mock

    # Mock Manager
    mock_opensim.Manager = MagicMock()

    # Configure model dimensions
    model_mock.getNumCoordinates.return_value = 2
    model_mock.getNumSpeeds.return_value = 2
    model_mock.getNumControls.return_value = 1

    # Configure Muscles
    muscles_mock = MagicMock()
    muscles_mock.getSize.return_value = 2
    model_mock.getMuscles.return_value = muscles_mock

    # Configure MarkerSet
    marker_set_mock = MagicMock()
    marker_set_mock.getSize.return_value = 1
    marker_mock = MagicMock()
    marker_mock.getName.return_value = "TestMarker"

    # Configure location return
    loc = MagicMock()
    loc.get.side_effect = lambda i: 0.0
    marker_mock.getLocationInGround.return_value = loc

    marker_set_mock.get.return_value = marker_mock
    model_mock.getMarkerSet.return_value = marker_set_mock

    # Configure State accessors
    # initSystem returns state
    state_mock = MagicMock()
    model_mock.initSystem.return_value = state_mock
    model_mock.initializeState.return_value = state_mock

    # getQ/getU returns Vector
    vec_q = MagicMock()
    vec_q.get.side_effect = lambda i: 0.1 * i
    state_mock.getQ.return_value = vec_q

    vec_u = MagicMock()
    vec_u.get.side_effect = lambda i: 0.2 * i
    state_mock.getU.return_value = vec_u

    # getControls
    vec_ctrl = MagicMock()
    vec_ctrl.get.side_effect = lambda i: 0.5
    model_mock.getControls.return_value = vec_ctrl

    with patch.dict(sys.modules, {"opensim": mock_opensim}):
        yield mock_opensim


@pytest.fixture
def mock_opensim_missing_env():
    """Context manager to mock missing opensim environment."""
    with patch.dict(sys.modules, {"opensim": None}):
        yield


@pytest.fixture
def temp_model_file(tmp_path):
    """Create a temporary .osim file for testing."""
    model_file = tmp_path / "test_model.osim"
    model_file.write_text("<OpenSimModel/>")
    return str(model_file)


class TestGolfSwingModel:
    """Test suite for GolfSwingModel without fallback behavior."""

    def test_model_path_required(self):
        """Test that model_path is required - no silent fallback."""
        with pytest.raises(ValueError, match="model_path is required"):
            GolfSwingModel(model_path=None)

    def test_model_file_not_found(self):
        """Test that missing model file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="OpenSim model file not found"):
            GolfSwingModel(model_path="/nonexistent/path/model.osim")

    def test_opensim_not_installed_error(
        self, mock_opensim_missing_env, temp_model_file
    ):
        """Test that missing OpenSim raises OpenSimNotInstalledError."""
        with pytest.raises(OpenSimNotInstalledError, match="OpenSim is not installed"):
            GolfSwingModel(model_path=temp_model_file)

    def test_opensim_model_load_error(self, mock_opensim_env, temp_model_file):
        """Test that OpenSim Model load failure raises OpenSimModelLoadError."""
        # Make the mock Model constructor raise an exception
        mock_opensim_env.Model.side_effect = RuntimeError("Model load failed")

        with pytest.raises(OpenSimModelLoadError, match="Failed to load OpenSim model"):
            GolfSwingModel(model_path=temp_model_file)

    def test_opensim_model_loads_successfully(self, mock_opensim_env, temp_model_file):
        """Test successful model loading with mocked OpenSim."""
        model = GolfSwingModel(model_path=temp_model_file)

        # Verify it loaded
        assert model.use_opensim is True
        assert model._opensim_model is not None

    def test_simulation_runs(self, mock_opensim_env, temp_model_file):
        """Test that simulation runs without error."""
        model = GolfSwingModel(model_path=temp_model_file)
        model.duration = 0.01  # Short duration for test

        result = model.run_simulation()

        assert result is not None
        assert len(result.time) > 0
        assert result.states.shape[1] == 4  # 2Q + 2U
        assert result.marker_positions["TestMarker"].shape[1] == 3

    def test_use_opensim_always_true(self, mock_opensim_env, temp_model_file):
        """Test that use_opensim is always True (no fallback mode)."""
        model = GolfSwingModel(model_path=temp_model_file)
        assert model.use_opensim is True


class TestNoFallbackBehavior:
    """Tests to verify there is NO fallback/demo behavior."""

    def test_no_demo_simulation_method(self):
        """Verify _run_demo_simulation method does not exist."""
        # This test ensures we don't accidentally re-add the demo mode
        assert not hasattr(GolfSwingModel, "_run_demo_simulation")

    def test_error_messages_are_helpful(
        self, mock_opensim_missing_env, temp_model_file
    ):
        """Test that error messages include installation guidance."""
        try:
            GolfSwingModel(model_path=temp_model_file)
            pytest.fail("Should have raised OpenSimNotInstalledError")
        except OpenSimNotInstalledError as e:
            error_msg = str(e)
            # Verify helpful content
            assert (
                "conda install" in error_msg.lower()
                or "pip install" in error_msg.lower()
            )
            assert "mujoco" in error_msg.lower() or "pinocchio" in error_msg.lower()
