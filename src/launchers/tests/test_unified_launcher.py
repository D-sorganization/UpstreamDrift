import sys
from unittest.mock import MagicMock, patch

import pytest

# We need to mock golf_launcher import inside unified_launcher because it might trigger Qt stuff
# causing issues in headless env if not careful.
sys.modules["launchers.golf_launcher"] = MagicMock()

from src.launchers.unified_launcher import UnifiedLauncher, launch  # noqa: E402


@pytest.fixture
def mock_qapp():
    with patch("launchers.unified_launcher.QApplication") as mock_app_cls:
        mock_app_instance = MagicMock()
        mock_app_cls.instance.return_value = mock_app_instance
        yield mock_app_instance


@pytest.fixture
def mock_golf_launcher():
    # unified_launcher imports GolfLauncher from .golf_launcher locally
    # We need to patch the class where it is used.
    # Since it's imported inside __init__, we patch it there?
    # Or rely on sys.modules mock.

    mock_module = sys.modules["launchers.golf_launcher"]
    mock_cls = mock_module.GolfLauncher
    mock_instance = mock_cls.return_value
    return mock_instance


def test_init(mock_qapp, mock_golf_launcher):
    launcher = UnifiedLauncher()
    assert launcher is not None
    # Verify GolfLauncher was instantiated
    # Note: since we mocked the module before import, this should work
    sys.modules["launchers.golf_launcher"].GolfLauncher.assert_called_once()


def test_init_no_pyqt():
    with patch("launchers.unified_launcher.PYQT_AVAILABLE", False):
        with pytest.raises(ImportError, match="PyQt6 is required"):
            UnifiedLauncher()


def test_mainloop(mock_qapp, mock_golf_launcher):
    launcher = UnifiedLauncher()
    mock_qapp.exec.return_value = 0

    ret = launcher.mainloop()

    mock_golf_launcher.show.assert_called_once()
    mock_qapp.exec.assert_called_once()
    assert ret == 0


def test_launch_function(mock_qapp):
    with patch("launchers.unified_launcher.UnifiedLauncher") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.mainloop.return_value = 42

        ret = launch()
        assert ret == 42
        mock_cls.assert_called_once()


def test_show_status():
    with (
        patch("shared.python.engine_manager.EngineManager") as mock_mgr_cls,
        patch("builtins.print") as mock_print,
    ):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.get_available_engines.return_value = []

        # Patch UnifiedLauncher to avoid init logic (GUI) if called,
        # but show_status is a function that creates instance.
        # Wait, show_status() instantiates UnifiedLauncher().
        # We should mock UnifiedLauncher class to avoid GUI init.

        with patch("launchers.unified_launcher.UnifiedLauncher"):
            # We need to call the REAL show_status method of the mock?
            # Or simpler: test the method on an instance.
            # But the test is calling the module-level function `show_status`.

            # The module function `show_status` does:
            # launcher = UnifiedLauncher()
            # launcher.show_status()

            # So if we mock UnifiedLauncher class, we can mock the `show_status` method on the instance.
            # But we want to test the LOGIC of show_status.
            # So we should probably test `UnifiedLauncher.show_status` directly.
            pass

    # Better approach: Instantiate UnifiedLauncher with mocks, then call show_status method
    with (
        patch("launchers.unified_launcher.QApplication"),
        patch("launchers.golf_launcher.GolfLauncher"),
    ):
        launcher = UnifiedLauncher()

        with (
            patch("shared.python.engine_manager.EngineManager") as mock_mgr_cls,
            patch("builtins.print") as mock_print,
        ):
            mock_mgr = mock_mgr_cls.return_value
            mock_mgr.get_available_engines.return_value = [
                MagicMock(value="test_engine")
            ]

            launcher.show_status()

            # Check if print was called with engine name
            # print calls are many. We check if any args contain "TEST_ENGINE"
            found = False
            for call in mock_print.call_args_list:
                args = call[0]
                if args and "TEST_ENGINE" in str(args[0]):
                    found = True
                    break
            assert found


def test_get_version():
    with (
        patch("launchers.unified_launcher.QApplication"),
        patch("launchers.golf_launcher.GolfLauncher"),
    ):
        launcher = UnifiedLauncher()

        # Case 1: Package metadata
        with patch("importlib.metadata.version", return_value="1.2.3"):
            assert launcher.get_version() == "1.2.3"

        # Case 2: shared.__version__
        with (
            patch("importlib.metadata.version", side_effect=ImportError),
            patch("shared.python.__version__", "4.5.6", create=True),
        ):
            assert launcher.get_version() == "4.5.6"

        # Case 3: Fallback
        with (
            patch("importlib.metadata.version", side_effect=ImportError),
            patch.dict(sys.modules, {"shared.python": MagicMock()}),
        ):
            # Ensure shared.python doesn't have __version__
            del sys.modules["shared.python"].__version__
            assert launcher.get_version() == "1.0.0-beta"  # Default hardcoded
