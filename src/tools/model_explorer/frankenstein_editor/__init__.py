"""Frankenstein editor package.

``URDFModel`` is Qt-free and imported eagerly so headless code (assembly
session, part catalog) can use it. The widget classes load lazily on first
access so importing this package never requires PyQt6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .model import URDFModel

if TYPE_CHECKING:
    from .dialogs import StealComponentDialog
    from .editor import FrankensteinEditor
    from .panel import ModelPanel

_LAZY = {
    "FrankensteinEditor": ".editor",
    "ModelPanel": ".panel",
    "StealComponentDialog": ".dialogs",
}

__all__ = [
    "FrankensteinEditor",
    "URDFModel",
    "ModelPanel",
    "StealComponentDialog",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
