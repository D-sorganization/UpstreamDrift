"""Tests for OpenSim core module.

These tests verify that OpenSim failures produce clear errors, not silent fallbacks.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from engines.physics_engines.opensim.python.opensim_golf.core import (
    GolfSwingModel,
    OpenSimModelLoadError,
    OpenSimNotInstalledError,
)


@pytest.fixture
def mock_opensim_env():
    """Context manager to mock opensim environment."""
    mock_opensim = MagicMock()
    mock_opensim.Model.return_value = MagicMock()
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

    def test_simulation_not_yet_implemented(self, mock_opensim_env, temp_model_file):
        """Test that simulation raises NotImplementedError with clear message."""
        model = GolfSwingModel(model_path=temp_model_file)

        with pytest.raises(
            NotImplementedError,
            match="OpenSim simulation integration is not yet complete",
        ):
            model.run_simulation()

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
