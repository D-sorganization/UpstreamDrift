"""Shared attribute forwarding for launcher manager mixins."""

from __future__ import annotations

from typing import Any


def forward_manager_attribute(
    manager: Any,
    name: str,
    value: Any,
    *,
    launcher_attr: str = "launcher",
) -> None:
    """Route manager attribute writes to the launcher when it owns the name."""
    if (
        name == launcher_attr
        or hasattr(type(manager), name)
        or name in manager.__dict__
    ):
        object.__setattr__(manager, name, value)
        return

    launcher = object.__getattribute__(manager, launcher_attr)
    if hasattr(launcher, name):
        setattr(launcher, name, value)
    else:
        object.__setattr__(manager, name, value)
