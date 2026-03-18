"""Tests for startup.py."""

import builtins  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtGui import QFont, QPainter  # noqa: E402

from src.launchers.startup import (  # noqa: E402
    AsyncStartupWorker,
    GolfSplashScreen,
    StartupResults,
    _get_theme_colors,
)


@pytest.fixture
def mock_theme_available():
    import types

    colors = types.SimpleNamespace(
        BG_DEEP="#000000",
        TEXT_PRIMARY="1",
        TEXT_TERTIARY="2",
        PRIMARY="3",
        BG_ELEVATED="4",
        TEXT_QUATERNARY="5",
    )
    sizes = types.SimpleNamespace(XXL=24, MD=11, SM=9, XS=8)
    weights = types.SimpleNamespace(BOLD=75, NORMAL=50, MEDIUM=57)
    with (
        patch("src.launchers.startup.THEME_AVAILABLE", True),
        patch("src.launchers.startup.Colors", colors, create=True),
        patch("src.launchers.startup.Sizes", sizes, create=True),
        patch("src.launchers.startup.Weights", weights, create=True),
        patch("src.launchers.startup.get_display_font", create=True) as mock_disp,
        patch("src.launchers.startup.get_qfont", create=True) as mock_qf,
    ):
        mock_disp.return_value = QFont("Arial", 10)
        mock_qf.return_value = QFont("Arial", 10)
        yield


@pytest.fixture
def mock_theme_unavailable():
    with patch("src.launchers.startup.THEME_AVAILABLE", False):
        yield


def test_get_theme_colors_available(mock_theme_available):
    with patch("src.shared.python.theme.get_current_colors", create=True) as mock_get:
        mock_get.return_value = "mock_theme"
        assert _get_theme_colors() == "mock_theme"
        mock_get.assert_called_once()


def test_get_theme_colors_not_available():
    # Force import error
    with patch("src.launchers.startup.THEME_AVAILABLE", False):
        # Even if theme isn't available, the fallback to DARK_THEME should happen if get_current_colors raises ImportError

        real_import = builtins.__import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "src.shared.python.theme" and "get_current_colors" in fromlist:
                raise ImportError("Mocked error")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=mock_import):
            res = _get_theme_colors()
            assert res is not None  # Returns DARK_THEME


def test_startup_results_init():
    res = StartupResults()
    assert res.registry is None
    assert res.engine_manager is None
    assert res.available_engines == []
    assert res.ai_available is False
    assert res.docker_available is False
    assert res.startup_time_ms == 0


def test_startup_results_from_dict():
    data = {
        "registry": "mock_registry",
        "engine_manager": "mock_engine",
        "available_engines": ["e1", "e2"],
        "ai_available": True,
        "docker_available": True,
        "startup_time_ms": 1234,
    }
    res = StartupResults.from_dict(data)
    assert res.registry == "mock_registry"
    assert res.engine_manager == "mock_engine"
    assert res.available_engines == ["e1", "e2"]
    assert res.ai_available is True
    assert res.docker_available is True
    assert res.startup_time_ms == 1234


def test_startup_results_from_dict_empty():
    res = StartupResults.from_dict({})
    assert res.registry is None
    assert res.engine_manager is None
    assert res.available_engines == []
    assert res.ai_available is False
    assert res.docker_available is False
    assert res.startup_time_ms == 0


# ==== GolfSplashScreen Tests ====


def test_splash_screen_init_theme_unavailable(mock_theme_unavailable, qapp):
    with patch("src.launchers.startup.Path.exists", return_value=False):
        splash = GolfSplashScreen()
        assert splash.loading_message == "Initializing UpstreamDrift..."
        assert splash.progress == 0
        assert splash.logo_pixmap is None


def test_splash_screen_init_theme_available(mock_theme_available, qapp):
    # Pass a valid path so it tries to load, but it won't exist so isNull will be True
    with patch("src.launchers.startup.Path.exists", return_value=False):
        splash = GolfSplashScreen()
        assert splash.loading_message == "Initializing UpstreamDrift..."


def test_splash_screen_resolve_theme_colors(mock_theme_available):
    res = GolfSplashScreen._resolve_theme_colors()
    assert res == ("1", "2", "3", "4", "5")


def test_splash_screen_resolve_theme_colors_fallback(mock_theme_unavailable):
    res = GolfSplashScreen._resolve_theme_colors()
    assert res == ("#FFFFFF", "#A0A0A0", "#0A84FF", "#2D2D2D", "#666666")


def test_drawContents(mock_theme_available, qapp):
    splash = GolfSplashScreen()
    splash.logo_pixmap = None
    splash.progress = 50
    splash.loading_message = "msg"
    splash.width = MagicMock(return_value=500)
    splash.height = MagicMock(return_value=300)
    # create rect manually or let it return magicmock?
    mock_rect = MagicMock()
    mock_rect.adjusted.return_value = mock_rect
    splash.rect = MagicMock(return_value=mock_rect)

    painter = MagicMock(spec=QPainter)
    with (
        patch.object(
            splash, "_resolve_theme_colors", return_value=("1", "2", "3", "4", "5")
        ),
        patch("src.launchers.startup.THEME_AVAILABLE", False),
    ):
        # test with THEME_AVAILABLE=False to hit fallback fonts easily
        splash.drawContents(painter)

        painter.setRenderHint.assert_called()
        painter.drawText.assert_called()


def test_drawContents_none(qapp):
    splash = GolfSplashScreen()
    # Shouldn"t throw when painter is None
    splash.drawContents(None)


@patch("src.launchers.startup.QApplication.processEvents")
def test_splash_show_message(mock_process_events, qapp):
    splash = GolfSplashScreen()
    splash.showMessage = MagicMock()
    splash.repaint = MagicMock()

    splash.show_message("Test message", 45)

    assert splash.loading_message == "Test message"
    assert splash.progress == 45
    splash.showMessage.assert_called_once()
    splash.repaint.assert_called_once()
    mock_process_events.assert_called_once()


# ==== AsyncStartupWorker Tests ====


def test_startup_worker_init():
    root = Path("fake_root")
    worker = AsyncStartupWorker(root)
    assert worker.repos_root == root
    assert isinstance(worker.results, StartupResults)


@patch("src.launchers.startup.secure_run")
@patch("src.shared.python.engine_core.engine_manager.EngineManager", autospec=True)
@patch("src.shared.python.config.model_registry.ModelRegistry", autospec=True)
def test_startup_worker_run_success(mock_registry, mock_engine_mgr, mock_secure_run):
    root = Path("fake_root")
    worker = AsyncStartupWorker(root)

    worker.progress_signal = MagicMock()
    worker.finished_signal = MagicMock()
    worker.error_signal = MagicMock()

    with patch("src.launchers.startup.time.sleep"):
        worker.run()

    assert worker.results.registry is not None
    assert worker.results.engine_manager is not None
    assert worker.results.docker_available is True
    worker.finished_signal.emit.assert_called_once_with(worker.results)
    worker.error_signal.emit.assert_not_called()


@patch("src.launchers.startup.secure_run")
@patch("src.shared.python.engine_core.engine_manager.EngineManager")
@patch("src.shared.python.config.model_registry.ModelRegistry")
def test_startup_worker_run_engine_import_error(
    mock_registry, mock_engine_mgr, mock_secure_run
):
    root = Path("fake_root")
    worker = AsyncStartupWorker(root)
    worker.progress_signal = MagicMock()
    worker.finished_signal = MagicMock()

    mock_engine_mgr.side_effect = ImportError("Test error")

    with patch("src.launchers.startup.time.sleep"):
        worker.run()

    assert worker.results.registry is not None
    assert worker.results.engine_manager is None  # Exception swallowed and handled
    assert worker.results.docker_available is True


@patch("src.launchers.startup.secure_run")
@patch("src.shared.python.engine_core.engine_manager.EngineManager")
@patch(
    "src.shared.python.config.model_registry.ModelRegistry",
    side_effect=ImportError("Critical error"),
)
def test_startup_worker_run_critical_import_error(
    mock_registry, mock_engine_mgr, mock_secure_run
):
    root = Path("fake_root")
    worker = AsyncStartupWorker(root)
    worker.progress_signal = MagicMock()
    worker.finished_signal = MagicMock()
    worker.error_signal = MagicMock()

    worker.run()

    worker.error_signal.emit.assert_called_once()
    worker.finished_signal.emit.assert_not_called()


@patch("src.launchers.startup.secure_run")
@patch("src.shared.python.engine_core.engine_manager.EngineManager")
@patch("src.shared.python.config.model_registry.ModelRegistry")
def test_startup_worker_run_docker_error(
    mock_registry, mock_engine_mgr, mock_secure_run
):
    root = Path("fake_root")
    worker = AsyncStartupWorker(root)
    worker.progress_signal = MagicMock()
    worker.finished_signal = MagicMock()

    mock_secure_run.side_effect = RuntimeError("Docker error")

    with patch("src.launchers.startup.time.sleep"):
        worker.run()

    assert worker.results.docker_available is False
    worker.finished_signal.emit.assert_called_once()
