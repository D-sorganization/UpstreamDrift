from unittest.mock import patch

import pytest

from src.shared.python.engine_core.engine_availability import (
    PYQT6_AVAILABLE,
    skip_if_unavailable,
)

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
    import sys

    # Temporarily remove the legacy module mock that test_unified_launcher.py sets
    # at session level, so _get_golf_main falls through to src.launchers.golf_launcher
    legacy_key = "launchers.golf_launcher"
    legacy_saved = sys.modules.pop(legacy_key, None)
    try:
        with patch("src.launchers.golf_launcher.main") as mock_main:
            mock_main.return_value = 0
            launcher.mainloop()
            mock_main.assert_called_once()
    finally:
        if legacy_saved is not None:
            sys.modules[legacy_key] = legacy_saved


def test_show_status(launcher):
    """Test show_status method."""
    # Mock the EngineManager to avoid actual engine initialization
    with patch("src.shared.python.engine_core.engine_manager.EngineManager") as MockEngineManager:
        mock_manager = MockEngineManager.return_value
        mock_manager.get_available_engines.return_value = []

        # show_status should not raise, even with mock
        launcher.show_status()


def test_get_version(launcher):
    """Test version retrieval.

    Design-by-Contract:
        Postcondition: get_version() always returns a non-empty string matching SemVer
        pattern or a recognised beta/rc suffix.
    """
    # Test metadata path first since package might not be fully installed in metadata
    version = launcher.get_version()
    assert isinstance(version, str)
    assert len(version) > 0

    # Test with mock package metadata (happy path)
    with patch("importlib.metadata.version") as mock_version:
        mock_version.return_value = "2.0.0"
        assert launcher.get_version() == "2.0.0"

    # Test fallback — when metadata fails, reads pyproject.toml or uses hardcoded string
    with patch("importlib.metadata.version", side_effect=ImportError):
        fallback = launcher.get_version()
        assert isinstance(fallback, str)
        assert len(fallback) > 0
        # Must be a valid semver-like string (e.g. "2.1.0", "2.1.0-beta")
        assert "." in fallback, f"Expected SemVer version, got: {fallback!r}"


def test_cli_launch():
    """Test CLI launch function."""
    # launch() directly calls golf_launcher.main(), so patch that
    with patch("src.launchers.golf_launcher.main") as mock_main:
        from src.launchers.unified_launcher import launch

        mock_main.return_value = 0
        launch()
        mock_main.assert_called_once()
