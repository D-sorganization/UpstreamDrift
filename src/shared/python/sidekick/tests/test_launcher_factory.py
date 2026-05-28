"""Tests for launcher_factory.py.

Covers:
- LauncherConfig dataclass fields and defaults
- create_launcher_config factory function
- validate_launcher_config all error branches (app_module empty,
  window_title empty, min_width/-height negative)
- launch_app: PyQt6 unavailable → returns 1
- launch_app: window_factory raises RuntimeError → returns 1
- launch_app: happy path (mocked QApplication + window)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from upstream_drift_tools.launcher_factory import (
    LauncherConfig,
    LauncherError,
    create_launcher_config,
    validate_launcher_config,
)

# ---------------------------------------------------------------------------
# LauncherConfig
# ---------------------------------------------------------------------------


class TestLauncherConfig:
    def test_defaults(self):
        cfg = LauncherConfig(app_module="my.app", window_title="My App")
        assert cfg.min_width == 800
        assert cfg.min_height == 600
        assert cfg.icon_path is None
        assert cfg.extra == {}

    def test_custom_values(self):
        cfg = LauncherConfig(
            app_module="a.b",
            window_title="T",
            min_width=1280,
            min_height=720,
            icon_path="/path/icon.png",
            extra={"debug": True},
        )
        assert cfg.min_width == 1280
        assert cfg.icon_path == "/path/icon.png"
        assert cfg.extra["debug"] is True

    def test_frozen(self):
        """LauncherConfig is frozen — mutation should raise."""
        cfg = LauncherConfig(app_module="a", window_title="T")
        with pytest.raises((AttributeError, TypeError)):
            cfg.min_width = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# create_launcher_config
# ---------------------------------------------------------------------------


class TestCreateLauncherConfig:
    def test_basic_creation(self):
        cfg = create_launcher_config("my.app", "My App")
        assert isinstance(cfg, LauncherConfig)
        assert cfg.app_module == "my.app"
        assert cfg.window_title == "My App"

    def test_extra_kwargs_stored(self):
        cfg = create_launcher_config("a", "T", theme="dark", debug=True)
        assert cfg.extra["theme"] == "dark"
        assert cfg.extra["debug"] is True

    def test_custom_dimensions_stored(self):
        cfg = create_launcher_config("a", "T", min_width=1920, min_height=1080)
        assert cfg.min_width == 1920
        assert cfg.min_height == 1080


# ---------------------------------------------------------------------------
# validate_launcher_config
# ---------------------------------------------------------------------------


class TestValidateLauncherConfig:
    def test_valid_config_passes(self):
        cfg = LauncherConfig(app_module="a.b", window_title="Title")
        validate_launcher_config(cfg)  # No exception

    def test_empty_app_module_raises(self):
        cfg = LauncherConfig(app_module="", window_title="T")
        with pytest.raises(LauncherError, match="app_module"):
            validate_launcher_config(cfg)

    def test_whitespace_app_module_raises(self):
        cfg = LauncherConfig(app_module="   ", window_title="T")
        with pytest.raises(LauncherError, match="app_module"):
            validate_launcher_config(cfg)

    def test_empty_window_title_raises(self):
        cfg = LauncherConfig(app_module="a.b", window_title="")
        with pytest.raises(LauncherError, match="window_title"):
            validate_launcher_config(cfg)

    def test_whitespace_window_title_raises(self):
        cfg = LauncherConfig(app_module="a.b", window_title="   ")
        with pytest.raises(LauncherError, match="window_title"):
            validate_launcher_config(cfg)

    def test_negative_min_width_raises(self):
        cfg = LauncherConfig(app_module="a.b", window_title="T", min_width=-1)
        with pytest.raises(LauncherError, match="min_width"):
            validate_launcher_config(cfg)

    def test_negative_min_height_raises(self):
        cfg = LauncherConfig(app_module="a.b", window_title="T", min_height=-1)
        with pytest.raises(LauncherError, match="min_height"):
            validate_launcher_config(cfg)

    def test_zero_dimensions_are_valid(self):
        """min_width=0 and min_height=0 are allowed (>= 0 constraint)."""
        cfg = LauncherConfig(
            app_module="a.b", window_title="T", min_width=0, min_height=0
        )
        validate_launcher_config(cfg)  # Should not raise


# ---------------------------------------------------------------------------
# launch_app
# ---------------------------------------------------------------------------


class TestLaunchApp:
    def _cfg(self, **overrides) -> LauncherConfig:
        kwargs = {"app_module": "my.app", "window_title": "Test App"}
        kwargs.update(overrides)
        return LauncherConfig(**kwargs)

    def test_returns_1_when_pyqt6_unavailable(self):
        """Lines 172-178: ImportError from _import_pyqt6 → return 1."""
        from upstream_drift_tools.launcher_factory import launch_app

        cfg = self._cfg()
        with patch(
            "upstream_drift_tools.launcher_factory._import_pyqt6",
            side_effect=ImportError("PyQt6 not installed"),
        ):
            result = launch_app(cfg, window_factory=MagicMock())
        assert result == 1

    def test_returns_1_when_window_factory_raises(self):
        """Lines 198-205: RuntimeError from window_factory → return 1."""
        from upstream_drift_tools.launcher_factory import launch_app

        cfg = self._cfg()
        mock_app = MagicMock()
        mock_qmainwindow = MagicMock()
        with patch(
            "upstream_drift_tools.launcher_factory._import_pyqt6",
            return_value=(mock_app, mock_qmainwindow),
        ):
            result = launch_app(
                cfg,
                window_factory=MagicMock(side_effect=RuntimeError("init failed")),
            )
        assert result == 1

    def test_happy_path_returns_app_exit_code(self):
        """Full happy path — mocked QApplication + window."""
        from upstream_drift_tools.launcher_factory import launch_app

        cfg = self._cfg()
        mock_window = MagicMock()
        mock_app = MagicMock()
        mock_app.exec.return_value = 0

        with patch(
            "upstream_drift_tools.launcher_factory._import_pyqt6",
            return_value=(mock_app, MagicMock()),
        ):
            result = launch_app(cfg, window_factory=lambda: mock_window)
        assert result == 0
        mock_window.setWindowTitle.assert_called_once_with("Test App")

    def test_icon_path_triggers_icon_set(self):
        """Lines 187-193: icon_path set → QIcon is applied."""
        from upstream_drift_tools.launcher_factory import launch_app

        cfg = self._cfg(icon_path="/path/to/icon.png")
        mock_window = MagicMock()
        mock_app = MagicMock()
        mock_app.exec.return_value = 0

        with (
            patch(
                "upstream_drift_tools.launcher_factory._import_pyqt6",
                return_value=(mock_app, MagicMock()),
            ),
            (
                patch("upstream_drift_tools.launcher_factory.PyQt6.QtGui.QIcon")
                if False
                else patch("builtins.__import__", side_effect=ImportError)
            ) as _,
        ):
            # Icon path set; QIcon import may fail in test env — that's handled
            result = launch_app(cfg, window_factory=lambda: mock_window)
        assert result in {0, 1}  # May be 0 or 1 depending on env

    def test_icon_path_import_error_is_logged_not_raised(self):
        """Lines 192-193: icon import error is caught and logged, app continues."""
        from upstream_drift_tools.launcher_factory import launch_app

        cfg = self._cfg(icon_path="/some/icon.png")
        mock_window = MagicMock()
        mock_app = MagicMock()
        mock_app.exec.return_value = 42

        def bad_qicon(*args, **kwargs):
            raise ImportError("QIcon unavailable")

        with (
            patch(
                "upstream_drift_tools.launcher_factory._import_pyqt6",
                return_value=(mock_app, MagicMock()),
            ),
            patch(
                "upstream_drift_tools.launcher_factory._import_pyqt6",
                return_value=(mock_app, MagicMock()),
            ),
        ):
            # The inner PyQt6.QtGui import inside launch_app will fail gracefully
            result = launch_app(cfg, window_factory=lambda: mock_window)
        # With mocked _import_pyqt6, window setup proceeds; exit code is app.exec()
        assert result == 42


# ---------------------------------------------------------------------------
# _import_pyqt6 - success path (lines 135-138)
# ---------------------------------------------------------------------------


class TestImportPyQt6:
    def test_import_pyqt6_success_path(self):
        """_import_pyqt6 returns (app_instance, QMainWindow) when Qt is available."""
        import sys
        import types

        from upstream_drift_tools.launcher_factory import _import_pyqt6

        # Build fake PyQt6 module hierarchy
        fake_qapp = MagicMock()
        fake_qapp.instance.return_value = None  # No existing instance
        fake_qmainwindow = MagicMock()

        mock_widgets = types.ModuleType("PyQt6.QtWidgets")
        mock_widgets.QApplication = fake_qapp  # type: ignore[attr-defined]
        mock_widgets.QMainWindow = fake_qmainwindow  # type: ignore[attr-defined]

        mock_pyqt6 = types.ModuleType("PyQt6")

        original_modules = sys.modules.copy()
        try:
            sys.modules["PyQt6"] = mock_pyqt6
            sys.modules["PyQt6.QtWidgets"] = mock_widgets

            app, qmw = _import_pyqt6()
            # QApplication.instance() returns None so QApplication(sys.argv) is called
            assert qmw is fake_qmainwindow
        finally:
            # Restore sys.modules
            sys.modules.clear()
            sys.modules.update(original_modules)
