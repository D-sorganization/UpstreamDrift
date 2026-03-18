"""Tests for unified_launcher.py."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.launchers.unified_launcher import (
    UnifiedLauncher,
    _get_golf_main,
    _is_pyqt6_available,
    launch,
    show_status,
)


@pytest.fixture
def clean_sys_modules():
    """Remove specific modules from sys.modules."""
    modules_to_remove = ["launchers.unified_launcher", "launchers.golf_launcher"]
    for mod in modules_to_remove:
        if mod in sys.modules:
            del sys.modules[mod]
    yield
    for mod in modules_to_remove:
        if mod in sys.modules:
            del sys.modules[mod]


def test_is_pyqt6_available_legacy_override(clean_sys_modules):
    # Set up legacy module mock
    legacy_mock = MagicMock()
    legacy_mock.PYQT6_AVAILABLE = False
    sys.modules["launchers.unified_launcher"] = legacy_mock

    assert _is_pyqt6_available() is False


def test_is_pyqt6_available_using_constant(clean_sys_modules):
    # Ensure no legacy module
    if "launchers.unified_launcher" in sys.modules:
        del sys.modules["launchers.unified_launcher"]

    with patch("src.launchers.unified_launcher.PYQT6_AVAILABLE", True):
        assert _is_pyqt6_available() is True


def test_get_golf_main_prefer_legacy(clean_sys_modules):
    legacy_mock = MagicMock()

    def fake_main():
        pass

    legacy_mock.main = fake_main
    sys.modules["launchers.golf_launcher"] = legacy_mock

    main_func = _get_golf_main(prefer_legacy=True)
    assert main_func is fake_main


def test_get_golf_main_absolute_import(clean_sys_modules):
    import builtins

    orig_import = builtins.__import__

    mock_module = MagicMock()

    def fake_main():
        pass

    mock_module.main = fake_main

    def mock_import(name, *args, **kwargs):
        if "golf_launcher" in name:
            raise ImportError("fake")
        return orig_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        patch(
            "src.launchers.unified_launcher.importlib.import_module",
            return_value=mock_module,
        ),
    ):
        main_func = _get_golf_main()
        assert main_func is fake_main


def test_unified_launcher_init_fails_if_no_pyqt6():
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=False),
        pytest.raises(ImportError, match="PyQt6 is required"),
    ):
        UnifiedLauncher()


def test_unified_launcher_mainloop():
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("src.launchers.unified_launcher._get_golf_main") as mock_get_main,
    ):
        mock_main = MagicMock()
        mock_get_main.return_value = mock_main

        launcher = UnifiedLauncher()
        launcher.mainloop()

        mock_get_main.assert_called_once_with(prefer_legacy=True)
        mock_main.assert_called_once()


def test_unified_launcher_show_status(caplog, capsys):
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch(
            "src.shared.python.engine_core.engine_manager.EngineManager.get_available_engines",
            return_value=["engine_a", "engine_b"],
        ),
        patch("src.launchers.unified_launcher.Path") as mock_path,
    ):
        # Setup mock paths for launchers and engines
        mock_launcher_dir = MagicMock()
        mock_launcher_file = MagicMock()
        mock_launcher_file.name = "fake_launcher.py"
        mock_launcher_dir.glob.return_value = [mock_launcher_file]

        mock_engine_dir = MagicMock()
        mock_engine_file = MagicMock()
        mock_engine_file.is_dir.return_value = True
        mock_engine_file.name = "engine1"
        mock_engine_dir.exists.return_value = True
        mock_engine_dir.iterdir.return_value = [mock_engine_file]

        # When creating Path(__file__).parent
        mock_path_instance = MagicMock()
        mock_path_instance.parent = mock_launcher_dir
        mock_path.return_value = mock_path_instance

        # Mock SUITE_ROOT
        with patch("src.shared.python.SUITE_ROOT", mock_engine_dir, create=True):
            launcher = UnifiedLauncher()
            launcher.show_status()

            captured = capsys.readouterr()
            assert (
                "ENGINE_A" in captured.out
                or "engine_a" in captured.out
                or "ENGINE_A" in captured.out.upper()
            )
            assert (
                "ENGINE_B" in captured.out
                or "engine_b" in captured.out
                or "ENGINE_B" in captured.out.upper()
            )


def test_unified_launcher_get_version_metadata():
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("importlib.metadata.version", return_value="1.2.3"),
    ):
        launcher = UnifiedLauncher()
        assert launcher.get_version() == "1.2.3"


def test_unified_launcher_get_version_fallback():
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("importlib.metadata.version", side_effect=ImportError("broken")),
    ):
        # If importlib fails, it falls back to shared.python or pyproject.toml
        # Let's mock sys.modules for shared.python to fail
        launcher = UnifiedLauncher()
        version = launcher.get_version()
        assert isinstance(version, str)
        assert len(version) > 0


def test_launch():
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True),
        patch("src.launchers.unified_launcher._get_golf_main") as mock_get_main,
    ):
        mock_main = MagicMock()
        mock_get_main.return_value = mock_main

        launch()
        mock_get_main.assert_called_once_with(prefer_legacy=False)
        mock_main.assert_called_once()


def test_launch_no_pyqt6():
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=False),
        patch("src.launchers.unified_launcher._get_golf_main") as mock_get_main,
    ):
        launch()
        mock_get_main.assert_not_called()


def test_show_status_fn():
    with patch("src.launchers.unified_launcher.UnifiedLauncher.show_status") as mock_ss:
        show_status()
        mock_ss.assert_called_once()
