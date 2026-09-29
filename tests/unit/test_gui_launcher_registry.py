"""Tests for gui_launcher.registry (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.gui_launcher.registry import (
    GUIRegistration,
    GUIRegistry,
    GUIType,
    LaunchConfig,
    get_registry,
)


def test_gui_launcher_registry_launcher_module_imports() -> None:
    """Guard against syntax errors in the shared GUI launcher module."""
    from src.shared.python.gui_launcher import launcher

    assert launcher.launch_pyqt6_app is not None


class TestGUIType:
    def test_has_types(self) -> None:
        types = list(GUIType)
        assert len(types) > 0


class TestGUIRegistry:
    def test_get_registry_returns_instance(self) -> None:
        reg = get_registry()
        assert isinstance(reg, GUIRegistry)

    def test_gui_launcher_registry_singleton(self) -> None:
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_list_tools_callable(self) -> None:
        reg = get_registry()
        assert callable(reg.list_tools)


class TestLaunchConfig:
    def test_gui_launcher_registry_importable(self) -> None:
        assert LaunchConfig is not None


class TestGUIRegistration:
    def test_gui_launcher_registry_importable(self) -> None:
        assert GUIRegistration is not None
