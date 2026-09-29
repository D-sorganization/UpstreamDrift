"""Contract tests for launcher manager attribute forwarding."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.launchers.launcher_manager_attrs import forward_manager_attribute


@dataclass
class _Launcher:
    shared: str = "launcher"


class _Manager:
    local_class_attr = "class"

    def __init__(self, launcher: _Launcher) -> None:
        forward_manager_attribute(self, "launcher", launcher)

    def __setattr__(self, name: str, value: object) -> None:
        forward_manager_attribute(self, name, value)


@pytest.mark.unit
def test_forward_manager_attribute_sets_initial_launcher_without_recursion() -> None:
    """DbC: assigning the launcher reference must not read it first."""
    launcher = _Launcher()

    manager = _Manager(launcher)

    assert manager.launcher is launcher


@pytest.mark.unit
def test_forward_manager_attribute_routes_launcher_owned_state() -> None:
    """DbC: names already owned by the launcher are written to the launcher."""
    launcher = _Launcher()
    manager = _Manager(launcher)

    manager.shared = "updated"

    assert launcher.shared == "updated"
    assert "shared" not in manager.__dict__


@pytest.mark.unit
def test_forward_manager_attribute_keeps_manager_local_state() -> None:
    """DbC: manager class and instance state stays on the manager."""
    launcher = _Launcher()
    manager = _Manager(launcher)
    manager.local = "instance"

    manager.local = "updated"
    manager.local_class_attr = "updated-class"

    assert manager.local == "updated"
    assert manager.local_class_attr == "updated-class"
    assert not hasattr(launcher, "local")
    assert not hasattr(launcher, "local_class_attr")
