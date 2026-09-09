"""Where every part of the Capture Rig tile sits, and why (#9846).

The tile used to put its controls in the ``PaneHost`` central widget and its
viewing panes in docks. Qt honours the central widget's size hint and squeezes
docks to their minimum, so measured offscreen at 1600x900 the controls held
3190 px while each video dock got 68 px: the video, which is the point of the
tool, was crushed, and the log's stretch of ``1`` took every spare pixel of
height.

This module inverts that. The **live view is the central widget** — the
preview canvas with the record transport under it, the thing the operator
watches — and every control is a dock: the step rail and the input tabs on
the left, the action grid across the bottom, playback and results tabbed on
the right. The **log is a drawer**: a dock tabbed behind the action grid,
closed at start and opened explicitly, so it occupies no height of its own
and can never take stretch from the video again.

Nothing here builds a widget; it only says where the tile's parts go, so the
arrangement can be read (and tested) in one place instead of inside
``gui.py``'s constructor.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

from src.shared.python.core.contracts import require
from src.shared.python.theme.layout_metrics import LayoutMetrics

from .layout import PaneHost

LOG_KEY = "log"
#: Right-area panes are stacked as tabs, playback in front.
TABIFY: tuple[tuple[str, str], ...] = (("playback", "results"), ("actions", LOG_KEY))
#: Docks that start closed and are left closed by *Reset layout*.
DRAWERS: tuple[str, ...] = (LOG_KEY,)

#: Starting widths: the controls take a column each and the video keeps the
#: rest. Without these every dock claims its full size hint and the central
#: widget is squeezed back to nothing, which is the bug this module exists to
#: fix.
CONTROL_WIDTH = LayoutMetrics.SIDEBAR_MIN_WIDTH
VIEW_WIDTH = LayoutMetrics.SIDEBAR_MIN_WIDTH + LayoutMetrics.SIDEBAR_MIN_WIDTH // 2
DOCK_WIDTHS: dict[str, int] = {
    "rail": CONTROL_WIDTH,
    "inputs": CONTROL_WIDTH,
    "playback": VIEW_WIDTH,
    "results": VIEW_WIDTH,
}

_LEFT = Qt.DockWidgetArea.LeftDockWidgetArea
_RIGHT = Qt.DockWidgetArea.RightDockWidgetArea
_BOTTOM = Qt.DockWidgetArea.BottomDockWidgetArea


@dataclass(frozen=True)
class TileParts:
    """The widgets the tile arranges: one viewing centre, five docks.

    Invariant: ``live`` is the central widget and is never a dock, so no
    saved arrangement can close or float the video away.
    """

    live: QWidget
    playback: QWidget
    results: QWidget
    rail: QWidget
    inputs: QWidget
    actions: QWidget
    log: QWidget


def pane_specs(parts: TileParts) -> dict[str, tuple[str, QWidget, Qt.DockWidgetArea]]:
    """The docks around the live view, in the order they are added.

    Postcondition: :data:`LOG_KEY` and every name in :data:`TABIFY` is a key
    of the result, so the host can shut the drawer and stack the tabs.
    """
    specs: dict[str, tuple[str, QWidget, Qt.DockWidgetArea]] = {
        "rail": ("Workflow", parts.rail, _LEFT),
        "inputs": ("Settings", parts.inputs, _LEFT),
        "playback": ("Playback", parts.playback, _RIGHT),
        "results": ("Results", parts.results, _RIGHT),
        "actions": ("Actions", parts.actions, _BOTTOM),
        LOG_KEY: ("Log", parts.log, _BOTTOM),
    }
    require(LOG_KEY in specs, "the log drawer must have a dock")
    return specs


def build_host(parts: TileParts, parent: QWidget | None = None) -> PaneHost:
    """The tile's pane host: ``parts.live`` central, everything else docked.

    Postcondition: the host's central widget is ``parts.live`` and the log
    dock is closed, so the video holds the middle of a freshly built tile.
    """
    host = PaneHost(
        parts.live,
        pane_specs(parts),
        parent,
        hidden=DRAWERS,
        tabify=TABIFY,
        widths=DOCK_WIDTHS,
    )
    require(host.centralWidget() is parts.live, "the live view must be central")
    require(not host.is_visible(LOG_KEY), "the log must start closed")
    return host
