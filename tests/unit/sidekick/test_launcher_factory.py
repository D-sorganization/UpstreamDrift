"""Tests for sidekick.launcher_factory (Issues #1949, #1744)."""

from __future__ import annotations

from sidekick.launcher_factory import (
    LauncherConfig,
    LauncherError,
    create_launcher_config,
)


class TestLauncherConfig:
    def test_launcher_factory_construction(self) -> None:
        cfg = LauncherConfig(app_module="my.app", window_title="Test")
        assert cfg.app_module == "my.app"

    def test_window_title(self) -> None:
        cfg = LauncherConfig(app_module="my.app", window_title="My Title")
        assert cfg.window_title == "My Title"

    def test_default_dimensions(self) -> None:
        cfg = LauncherConfig(app_module="a.b", window_title="App")
        assert cfg.min_width == 800
        assert cfg.min_height == 600

    def test_custom_dimensions(self) -> None:
        cfg = LauncherConfig(
            app_module="a.b", window_title="App", min_width=1024, min_height=768
        )
        assert cfg.min_width == 1024


class TestCreateLauncherConfig:
    def test_returns_launcher_config(self) -> None:
        cfg = create_launcher_config("my.module", "Test Window")
        assert isinstance(cfg, LauncherConfig)

    def test_module_set(self) -> None:
        cfg = create_launcher_config("app.main", "App")
        assert cfg.app_module == "app.main"

    def test_extra_kwargs(self) -> None:
        cfg = create_launcher_config("app.main", "App", debug=True)
        assert cfg.extra.get("debug") is True


class TestLauncherError:
    def test_launcher_factory_is_exception(self) -> None:
        err = LauncherError("test error")
        assert isinstance(err, Exception)

    def test_launcher_factory_message(self) -> None:
        err = LauncherError("something failed")
        assert "something failed" in str(err)
