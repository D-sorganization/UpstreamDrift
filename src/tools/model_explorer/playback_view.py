"""Which camera the results playback opens on, and where it looks (#15150).

The default view is a setting (``playback_view_settings.json``), not a literal.
Behind-the-golfer views reuse the export's anchor rule
(:func:`behind_camera.shift_to_anchor`), so the explorer and the MP4 export
frame a swing the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any

from src.shared.python.golf_view_presets.presets import VIEW_ORDER
from src.tools.native_viewer_export.behind_camera import Vec3, shift_to_anchor

__all__ = [
    "DEFAULT_LOOKAT_M",
    "SETTINGS_PATH",
    "FramingSettings",
    "LayoutSettings",
    "ViewSettings",
    "behind_anchor_for",
    "load_camera_distances",
    "load_framing_settings",
    "load_layout_settings",
    "load_view_settings",
    "playback_lookat",
]

logger = logging.getLogger(__name__)
SETTINGS_PATH = Path(__file__).with_name("playback_view_settings.json")
#: Camera look-at for the golfer at address (matches ``ExportSettings``).
DEFAULT_LOOKAT_M: Vec3 = (1.0, 0.0, 0.9)


#: Smallest renderer edge MuJoCo-side code accepts (see ``GolferPlayback``).
MIN_RENDER_PX = 16


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


def _positive_float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number above 0, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{key} must be a finite number above 0, got {value!r}")
    return number


def load_camera_distances(path: Path | None = None) -> dict[str, float]:
    """Per-view camera distance (m); only listed views override the preset.

    ``ValueError`` on an unknown view or a non-finite / non-positive distance.
    """
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    block = raw.get("camera_distance_m", {})
    return {_check_view(name): _positive_float(block, name) for name in block}


@dataclass(frozen=True)
class FramingSettings:
    """How the render fills its label (all from ``playback_view_settings``)."""

    render_height_px: int
    min_aspect: float
    max_aspect: float
    aspect_step: float
    resize_debounce_ms: int

    def render_size(self, aspect: float) -> tuple[int, int]:
        """``(width, height)`` px for a label of ``aspect`` = width / height.

        The aspect is clamped to ``min_aspect..max_aspect`` and rounded to
        ``aspect_step``. ``ValueError`` when it is not a finite number above 0.
        """
        if not (math.isfinite(aspect) and aspect > 0):
            raise ValueError(f"aspect must be a finite number above 0, got {aspect!r}")
        clamped = min(max(aspect, self.min_aspect), self.max_aspect)
        stepped = round(clamped / self.aspect_step) * self.aspect_step
        stepped = min(max(stepped, self.min_aspect), self.max_aspect)
        return round(self.render_height_px * stepped), self.render_height_px


def load_framing_settings(path: Path | None = None) -> FramingSettings:
    """Read the ``framing`` block; ``ValueError`` on a bad value."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))["framing"]
    height = _positive_int(raw, "render_height_px")
    if height < MIN_RENDER_PX:
        raise ValueError(f"render_height_px must be at least {MIN_RENDER_PX}")
    low = _positive_float(raw, "min_aspect")
    high = _positive_float(raw, "max_aspect")
    if low >= high:
        raise ValueError(f"min_aspect {low} must be below max_aspect {high}")
    return FramingSettings(
        height,
        low,
        high,
        _positive_float(raw, "aspect_step"),
        _positive_int(raw, "resize_debounce_ms"),
    )


def playback_lookat(view: str, anchor_m: Vec3 | None) -> Vec3:
    """Look-at for ``view``: the address look-at, moved onto the anchor line
    for behind views when the run's ball-to-feet ``anchor_m`` is known."""
    return shift_to_anchor(_check_view(view), DEFAULT_LOOKAT_M, anchor_m)


def behind_anchor_for(pose: Any) -> Vec3 | None:
    """The run's ball-to-feet anchor (export rule), or ``None`` if unavailable.

    Slow (it poses the club model), so call it off the GUI thread. ``None``,
    with a warning, keeps the preset camera: no time base, ball or feet.
    """
    from src.tools.model_explorer.playback_ball_overlay import swing_for_source
    from src.tools.native_viewer_export.behind_camera import address_anchor

    try:
        swing = swing_for_source(pose)
    except ValueError as exc:
        logger.warning("no address swing, keeping the preset camera: %s", exc)
        return None
    return address_anchor(swing)
