import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

# Ensure PYQT6_AVAILABLE is True during tests
with patch("src.launchers.unified_launcher.PYQT6_AVAILABLE", True):
    from src.launchers.unified_launcher import UnifiedLauncher, launch  # noqa: E402


@pytest.fixture(autouse=True)
def mock_pyqt6_available() -> Generator[None, None, None]:
    """Force PyQt6 to be available for all tests."""
    with patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=True):
        yield


@pytest.fixture(autouse=True)
def mock_upstream_drift_launcher_module() -> Generator[MagicMock, None, None]:
    """Mock upstream_drift_launcher module."""
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"src.launchers.upstream_drift_launcher": mock_mod}):
        yield mock_mod


@pytest.fixture
def mock_qapp() -> Generator[MagicMock, None, None]:
    """Mock qapp."""
    with patch("src.launchers.unified_launcher.QApplication") as mock_app_cls:
        mock_app_instance = MagicMock()
        mock_app_cls.instance.return_value = mock_app_instance
        yield mock_app_instance


@pytest.fixture
def mock_upstream_drift_launcher(mock_upstream_drift_launcher_module: MagicMock) -> MagicMock:
    """Mock golf launcher."""
    mock_module = mock_upstream_drift_launcher_module
    mock_cls = mock_module.UpstreamDriftLauncher
    mock_instance = mock_cls.return_value
    return mock_instance


def test_unified_launcher_init(mock_qapp, mock_upstream_drift_launcher) -> None:
    launcher = UnifiedLauncher()
    assert launcher is not None


def test_init_no_pyqt() -> None:
    with (
        patch("src.launchers.unified_launcher._is_pyqt6_available", return_value=False),
        pytest.raises(ImportError, match="PyQt6 is required"),
    ):
        UnifiedLauncher()


def test_mainloop(mock_qapp, mock_upstream_drift_launcher, mock_upstream_drift_launcher_module) -> None:
    launcher = UnifiedLauncher()
    mock_qapp.exec.return_value = 0

    with patch(
        "src.launchers.unified_launcher._get_golf_main",
        return_value=mock_upstream_drift_launcher_module.main,
    ):
        launcher.mainloop()

    # mainloop now delegates to upstream_drift_launcher.main which calls sys.exit
    # Since we mocked it, the main function should be called.
    mock_upstream_drift_launcher_module.main.assert_called_once()


def test_launch_function() -> None:
    with patch("src.launchers.unified_launcher._get_golf_main") as mock_get_main:
        mock_main = MagicMock()
        mock_get_main.return_value = mock_main
        launch()
        mock_main.assert_called_once()


def test_show_status(capsys) -> None:
    with (
        patch("src.launchers.unified_launcher.QApplication"),
        patch("src.launchers.upstream_drift_launcher.UpstreamDriftLauncher"),
    ):
        launcher = UnifiedLauncher()

        # EngineManager is imported inside the method
        mock_mgr_instance = MagicMock()

        # In unified_launcher.py, it calls getattr(_engine, "value", str(_engine))
        engine_mock = MagicMock()
        engine_mock.value = "TEST_ENGINE"
        mock_mgr_instance.get_available_engines.return_value = [engine_mock]

        with patch.dict(
            sys.modules,
            {
                "src.shared.python.engine_core.engine_manager": MagicMock(
                    EngineManager=MagicMock(return_value=mock_mgr_instance)
                )
            },
        ):
            launcher.show_status()

    captured = capsys.readouterr()
    assert "TEST_ENGINE" in captured.out


def test_get_version() -> None:
    with (
        patch("src.launchers.unified_launcher.QApplication"),
        patch("src.launchers.upstream_drift_launcher.UpstreamDriftLauncher"),
    ):
        launcher = UnifiedLauncher()

        # Case 1: Package metadata
        with patch("importlib.metadata.version", return_value="1.2.3"):
            assert launcher.get_version() == "1.2.3"

        # Case 2: pyproject.toml
        from importlib.metadata import PackageNotFoundError

        # We mock what is actually in unified_launcher.py
        with (
            patch("importlib.metadata.version", side_effect=PackageNotFoundError),
            patch("builtins.open"),
            patch("os.path.exists", return_value=True),
        ):
            # Patch tomllib conditionally depending on whether it exists in unified_launcher
            import importlib

            unified = importlib.import_module("src.launchers.unified_launcher")
            if hasattr(unified, "tomllib"):
                with patch(
                    "src.launchers.unified_launcher.tomllib.load",
                    return_value={"project": {"version": "3.4.5"}},
                ):
                    assert launcher.get_version() == "3.4.5"
            else:
                # If tomllib is not imported there, we might need to patch tomli
                if hasattr(unified, "tomli"):
                    with patch(
                        "src.launchers.unified_launcher.tomli.load",
                        return_value={"project": {"version": "3.4.5"}},
                    ):
                        assert launcher.get_version() == "3.4.5"
                elif hasattr(unified, "toml"):
                    with patch(
                        "src.launchers.unified_launcher.toml.load",
                        return_value={"project": {"version": "3.4.5"}},
                    ):
                        assert launcher.get_version() == "3.4.5"
                else:
                    # just mock it by replacing the whole method or setting __version__ to skip TOML logic
                    with patch("shared.python.__version__", "3.4.5", create=True):
                        assert launcher.get_version() == "3.4.5"

        # Case 3: shared.__version__
        with (
            patch("importlib.metadata.version", side_effect=ImportError),
            patch("shared.python.__version__", "4.5.6", create=True),
        ):
            assert launcher.get_version() == "4.5.6"

        # Case 4: Fallback
        with (
            patch("importlib.metadata.version", side_effect=ImportError),
            patch.dict(sys.modules, {"shared.python": MagicMock()}),
        ):
            # Ensure shared.python doesn't have __version__
            del sys.modules["shared.python"].__version__
            assert launcher.get_version() == "1.0.0-beta"  # Default hardcoded
