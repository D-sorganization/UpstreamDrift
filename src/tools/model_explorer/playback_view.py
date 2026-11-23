"""Which camera the results playback opens on, and where it looks (#15150).

The default view is a setting (``playback_view_settings.json``), not a literal.
Behind-the-golfer views reuse the export's anchor rule
(:func:`behind_camera.shift_to_anchor`), so the explorer and the MP4 export
frame a swing the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.shared.python.golf_view_presets.presets import VIEW_ORDER
from src.tools.native_viewer_export.behind_camera import Vec3, shift_to_anchor

__all__ = [
    "DEFAULT_LOOKAT_M",
    "SETTINGS_PATH",
    "LayoutSettings",
    "ViewSettings",
    "load_layout_settings",
    "load_view_settings",
    "playback_lookat",
]

SETTINGS_PATH = Path(__file__).with_name("playback_view_settings.json")
#: Camera look-at for the golfer at address (matches ``ExportSettings``).
DEFAULT_LOOKAT_M: Vec3 = (1.0, 0.0, 0.9)


def _check_view(name: object) -> str:
    if name not in VIEW_ORDER:
        raise ValueError(f"unknown view {name!r}; expected one of {VIEW_ORDER}")
    return str(name)


@dataclass(frozen=True)
class ViewSettings:
    """The playback camera settings; ``default_view`` is one of ``VIEW_ORDER``."""

    default_view: str

    def __post_init__(self) -> None:
        _check_view(self.default_view)


def load_view_settings(path: Path | None = None) -> ViewSettings:
    """Read the settings; ``ValueError`` when the default view is unknown."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return ViewSettings(raw["default_view"])


def _positive_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class LayoutSettings:
    """How the playback shares the window (all from ``playback_view_settings``)."""

    render_stretch: int
    plot_stretch: int
    hide_editor_docks: bool
    slider_min_width_px: int


def load_layout_settings(path: Path | None = None) -> LayoutSettings:
    """Read the ``layout`` block; ``ValueError`` on a non-positive stretch or a
    non-boolean ``hide_editor_docks``."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))["layout"]
    hide = raw.get("hide_editor_docks")
    if not isinstance(hide, bool):
        raise ValueError(f"hide_editor_docks must be true or false, got {hide!r}")
    return LayoutSettings(
        _positive_int(raw, "render_stretch"),
        _positive_int(raw, "plot_stretch"),
        hide,
        _positive_int(raw, "slider_min_width_px"),
    )


def playback_lookat(view: str, anchor_m: Vec3 | None) -> Vec3:
    """Look-at for ``view``: the address look-at, moved onto the anchor line
    for behind views when the run's ball-to-feet ``anchor_m`` is known."""
    return shift_to_anchor(_check_view(view), DEFAULT_LOOKAT_M, anchor_m)
