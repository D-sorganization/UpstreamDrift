import sys
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.engine_availability import PYQT6_AVAILABLE

pytestmark = pytest.mark.skipif(
    not PYQT6_AVAILABLE, reason="PyQt6 GUI libraries not available"
)

if PYQT6_AVAILABLE:
    from src.launchers.unified_launcher import UnifiedLauncher


@pytest.fixture
def mock_app():
    with patch("launchers.unified_launcher.QApplication") as MockApp:
        mock_app_instance = MockApp.instance.return_value
        if mock_app_instance is None:
            mock_app_instance = MockApp.return_value
        MockApp.instance.return_value = mock_app_instance
        yield MockApp


@pytest.fixture
def launcher(mock_app):
    # Patch the GolfLauncher class where it is defined, so when it is imported
    # by UnifiedLauncher it uses the mock.
    with patch("launchers.golf_launcher.GolfLauncher"):
        launcher = UnifiedLauncher()
        return launcher


def test_initialization(launcher):
    """Test UnifiedLauncher initialization."""
    assert launcher.app is not None
    # launcher.launcher is lazy-loaded, so it's None until mainloop() is called
    assert launcher.launcher is None


def test_mainloop(launcher):
    """Test mainloop execution."""
    # Mock GolfLauncher at its source for lazy loading
    with patch("launchers.golf_launcher.GolfLauncher") as MockGolfLauncher:
        mock_golf_launcher = MagicMock()
        MockGolfLauncher.return_value = mock_golf_launcher
        launcher.app.exec.return_value = 0

        exit_code = launcher.mainloop()

        assert exit_code == 0
        # Verify GolfLauncher was instantiated
        MockGolfLauncher.assert_called_once()
        # Verify show was called on the golf launcher
        mock_golf_launcher.show.assert_called_once()
        launcher.app.exec.assert_called_once()


def test_show_status(launcher):
    """Test show_status output."""
    with (
        patch("shared.python.engine_manager.EngineManager") as MockEngineManager,
        patch("builtins.print") as mock_print,
    ):
        mock_manager = MockEngineManager.return_value
        mock_manager.get_available_engines.return_value = [MagicMock(value="mujoco")]

        launcher.show_status()

        # Verify that print was called with expected headers
        assert any(
            "Status Report" in call.args[0]
            for call in mock_print.call_args_list
            if call.args
        )
        assert any(
            "Available Engines" in call.args[0]
            for call in mock_print.call_args_list
            if call.args
        )
        assert any(
            "MUJOCO" in call.args[0] for call in mock_print.call_args_list if call.args
        )


def test_get_version(launcher):
    """Test version retrieval."""
    # Test fallback first since package might not be fully installed in metadata
    version = launcher.get_version()
    assert isinstance(version, str)
    assert len(version) > 0

    # Test with mock package metadata
    with patch("importlib.metadata.version") as mock_version:
        mock_version.return_value = "2.0.0"
        assert launcher.get_version() == "2.0.0"

    # Test with mock shared version when package metadata fails
    with patch("importlib.metadata.version", side_effect=ImportError):
        with patch.dict(sys.modules, {"shared.python": MagicMock(__version__="1.5.0")}):
            assert launcher.get_version() == "1.5.0"


def test_cli_launch():
    """Test CLI launch function."""
    with patch("launchers.unified_launcher.UnifiedLauncher") as MockLauncher:
        from src.launchers.unified_launcher import launch

        MockLauncher.return_value.mainloop.return_value = 0
        assert launch() == 0
