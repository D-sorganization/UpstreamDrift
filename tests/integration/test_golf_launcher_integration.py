"""Integration tests for GolfLauncher."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PyQt6.QtWidgets import QApplication

from launchers.golf_launcher import GolfLauncher
from shared.python.model_registry import ModelRegistry

# Improve headless stability
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def launcher_env(qapp):
    """Setup launcher environment with temp config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a fake models.yaml
        config_dir = temp_path / "config"
        config_dir.mkdir()
        models_yaml = config_dir / "models.yaml"

        # Create a dummy model file
        engines_dir = temp_path / "engines" / "test"
        engines_dir.mkdir(parents=True)
        model_xml = engines_dir / "test_model.xml"
        model_xml.write_text("<mujoco/>", encoding="utf-8")

        config_content = f"""
models:
  - id: test_integration_model
    name: Integration Test Model
    description: A model for integration testing
    type: mjcf
    path: {str(model_xml).replace(os.sep, '/')}
"""
        models_yaml.write_text(config_content, encoding="utf-8")

        # Patch ModelRegistry to use our temp file
        temp_registry = ModelRegistry(models_yaml)

        # Override ASSETS_DIR to point to our empty temp dir.
        # In the real application, GolfLauncher creates QIcon instances from
        # files under ASSETS_DIR. In headless/CI test runs, fully initializing
        # QIcon with real image assets can trigger Qt plugin loading and C++
        # type handling that is fragile without a complete desktop environment
        # (e.g. missing platform plugins or windowing system), which has
        # previously resulted in hard-to-debug C++-side errors.
        #
        # Instead of mocking QIcon itself (which would bypass the real
        # filesystem-based icon lookup that we want to exercise), we point
        # ASSETS_DIR at an empty temporary directory. This makes icon lookups
        # resolve via genuine filesystem checks but guarantees that no real
        # icon files are found, so Qt never attempts heavy-weight icon
        # initialization while the logic we care about (graceful handling of
        # missing assets) is still executed.
        with (
            patch("shared.python.model_registry.ModelRegistry") as MockRegistry,
            patch("launchers.golf_launcher.ASSETS_DIR", new=temp_path),
        ):

            MockRegistry.return_value = temp_registry

            # Create Launcher
            launcher = GolfLauncher()
            yield launcher, model_xml


def test_launcher_detects_real_model_files(launcher_env):
    """Test that launcher correctly loads and identifies valid/invalid paths."""
    launcher, model_path = launcher_env

    # 1. Verify model loaded from registry (UI cards)
    assert "Integration Test Model" in launcher.model_cards

    # 2. Select it
    launcher.select_model("Integration Test Model")
    assert launcher.selected_model == "Integration Test Model"

    # 3. Verify path resolving via configuration
    # Note: Registry lookups are by ID, but UI selection is by Name.
    # The temp config ID is "test_integration_model"
    model_config = launcher.registry.get_model("test_integration_model")
    assert model_config is not None
    assert Path(model_config.path).resolve() == model_path.resolve()


def test_launcher_handles_missing_file_on_launch(launcher_env):
    """Test launching a model where the file was deleted after load."""
    launcher, model_path = launcher_env

    launcher.select_model("Integration Test Model")

    # DELETE the file
    os.remove(model_path)
    assert not model_path.exists()

    # Attempt launch while mocking subprocess to avoid any real execution.
    # GolfLauncher is expected to check the path exists before invoking subprocess.
    with (
        patch("launchers.golf_launcher.subprocess.Popen") as mock_popen,
        patch("PyQt6.QtWidgets.QMessageBox.critical") as mock_msg,
    ):

        launcher.launch_simulation()

        # Should NOT call Popen because file is missing
        mock_popen.assert_not_called()

        # Should show error message
        mock_msg.assert_called_once()
        args = mock_msg.call_args[0]
        # Ensure a non-empty error message text is provided
        assert isinstance(args[2], str)
        assert args[2].strip() != ""
