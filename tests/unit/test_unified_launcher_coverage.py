import sys
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.engine_availability import PYQT6_AVAILABLE, skip_if_unavailable

pytestmark = skip_if_unavailable("pyqt6")

if PYQT6_AVAILABLE:
    from src.launchers.unified_launcher import UnifiedLauncher


@pytest.fixture
def launcher():
    """Create a UnifiedLauncher instance."""
    return UnifiedLauncher()


def test_initialization(launcher):
    """Test UnifiedLauncher initialization succeeds."""
    # UnifiedLauncher now uses lazy initialization - no app/launcher attributes at init
    assert launcher is not None
    assert isinstance(launcher, UnifiedLauncher)


def test_mainloop(launcher):
    """Test mainloop execution delegates to golf_launcher.main()."""
    # Mock the main function from golf_launcher that mainloop calls
    with patch("src.launchers.golf_launcher.main") as mock_main:
        mock_main.return_value = 0

        launcher.mainloop()
        mock_main.assert_called_once()


def test_show_status(launcher):
    """Test show_status method."""
    # Mock the EngineManager to avoid actual engine initialization
    with patch("src.shared.python.engine_manager.EngineManager") as MockEngineManager:
        mock_manager = MockEngineManager.return_value
        mock_manager.get_available_engines.return_value = []

        # show_status should not raise, even with mock
        launcher.show_status()


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
        with patch.dict(
            sys.modules, {"src.shared.python": MagicMock(__version__="1.5.0")}
        ):
            assert launcher.get_version() == "1.5.0"


def test_cli_launch():
    """Test CLI launch function."""
    # launch() directly calls golf_launcher.main(), so patch that
    with patch("src.launchers.golf_launcher.main") as mock_main:
        from src.launchers.unified_launcher import launch

        mock_main.return_value = 0
        launch()
        mock_main.assert_called_once()
