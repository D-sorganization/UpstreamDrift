"""Responsive layout modes and adaptation for the Capture Rig GUI (#9847).

Below a width threshold (1400 px, such as 1280 px laptop panels), the controls
collapse into a compact mode: left docks tabify together into a single narrow column,
leaving the majority of the width to the central video viewing pane. Above the threshold,
the roomy arrangement gives separate space to workflow rail and settings.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt

from src.shared.python.core.contracts import require

if TYPE_CHECKING:
    from .layout import PaneHost

DEFAULT_COMPACT_THRESHOLD_PX: int = 1400
HISTORICAL_MIN_WIDTH_PX: int = 3276
MAX_TILE_WIDTH_PX: int = 900
DEFAULT_WINDOW_SIZE: tuple[int, int] = (1600, 900)
COMPACT_WINDOW_SIZE: tuple[int, int] = (1280, 800)


class LayoutMode(str, Enum):
    """Layout density mode decided strictly by available width."""

    COMPACT = "compact"
    ROOMY = "roomy"


def resolve_layout_mode(
    width: int,
    threshold: int = DEFAULT_COMPACT_THRESHOLD_PX,
) -> LayoutMode:
    """Classify available window width into a compact or roomy layout mode.

    Precondition:
        ``width >= 0`` and ``threshold > 0``.

    Postcondition:
        Returns :attr:`LayoutMode.COMPACT` when ``width < threshold``,
        and :attr:`LayoutMode.ROOMY` otherwise.
    """
    require(width >= 0, "width must be non-negative", width)
    require(threshold > 0, "threshold must be positive", threshold)
    if width < threshold:
        return LayoutMode.COMPACT
    return LayoutMode.ROOMY


def apply_responsive_mode(
    host: PaneHost,
    mode: LayoutMode,
) -> None:
    """Adapt the pane host dock arrangement to the chosen layout mode.

    In compact mode, the workflow rail and settings inputs are tabbed together
    so they occupy a single dock column on the left. In roomy mode, rail and
    inputs are split into separate docks side-by-side or stacked cleanly.

    Precondition:
        ``host`` is a valid :class:`PaneHost` instance.
    """
    require(host is not None, "host must not be None")
    if "rail" not in host.docks or "inputs" not in host.docks:
        return

    rail_dock = host.docks["rail"]
    inputs_dock = host.docks["inputs"]

    tabified = host.tabifiedDockWidgets(rail_dock)
    is_tabbed = inputs_dock in tabified

    if mode == LayoutMode.COMPACT:
        if not is_tabbed:
            host.tabifyDockWidget(rail_dock, inputs_dock)
            rail_dock.raise_()
    elif mode == LayoutMode.ROOMY:
        if is_tabbed:
            host.splitDockWidget(rail_dock, inputs_dock, Qt.Orientation.Horizontal)
            rail_dock.show()
            inputs_dock.show()

    # Apply dock width preferences so the central preview keeps the majority
    if hasattr(host, "_apply_widths"):
        host._apply_widths()
