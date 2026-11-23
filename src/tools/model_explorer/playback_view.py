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
from pathlib import Path
from typing import Any

from src.shared.python.golf_view_presets.presets import VIEW_ORDER
from src.tools.native_viewer_export.behind_camera import Vec3, shift_to_anchor

__all__ = [
    "DEFAULT_LOOKAT_M",
    "SETTINGS_PATH",
    "ViewSettings",
    "behind_anchor_for",
    "load_view_settings",
    "playback_lookat",
]

logger = logging.getLogger(__name__)
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
