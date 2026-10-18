"""Qt widget implementations sub-package.

Exports the four user-facing editors implemented in this package:

* :class:`ColorPicker`
* :class:`ColormapPicker`
* :class:`DataChannelEditor`
* :class:`MarkerStylePicker`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .color_picker import ColorPicker
    from .colormap_picker import ColormapPicker
    from .data_channel_editor import DataChannelEditor
    from .marker_style_picker import MarkerStylePicker

__all__ = [
    "ColorPicker",
    "ColormapPicker",
    "DataChannelEditor",
    "MarkerStylePicker",
]

_WIDGET_MODULES: dict[str, tuple[str, str]] = {
    "ColorPicker": (".color_picker", "ColorPicker"),
    "ColormapPicker": (".colormap_picker", "ColormapPicker"),
    "DataChannelEditor": (".data_channel_editor", "DataChannelEditor"),
    "MarkerStylePicker": (".marker_style_picker", "MarkerStylePicker"),
}


def __getattr__(name: str) -> Any:
    """Lazy attribute access for Qt widgets to keep PyQt6 optional."""
    target = _WIDGET_MODULES.get(name)
    if target is not None:
        module_path, symbol = target
        import importlib

        module = importlib.import_module(module_path, package=__name__)
        val = getattr(module, symbol)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_WIDGET_MODULES.keys()))
