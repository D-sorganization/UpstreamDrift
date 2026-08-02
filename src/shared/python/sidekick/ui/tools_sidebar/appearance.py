"""Shared, user-adjustable panel appearance for Sidekick runtime tabs.

The Terminal, Python REPL, and Workspace tabs all suffered the same UX
problem: no visible border (the old terminal QSS only drew one on
``:focus``), so it was unclear where each surface began or where to type.

This module is the single source of truth for those colours and borders
(DRY). A :class:`PanelAppearance` value object validates its inputs (DbC)
and :func:`panel_qss` turns it into a stylesheet scoped to one widget's
object name — every widget styles itself with the same one-liner::

    self.setStyleSheet(panel_qss(self.objectName(), appearance))

Appearance is JSON-safe so it round-trips through the existing
``SidebarTabSettingsStore`` and is editable from each tab's ⚙ gear.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

__all__ = [
    "DEFAULT_DARK_PANEL_APPEARANCE",
    "DEFAULT_LIGHT_PANEL_APPEARANCE",
    "MAX_BORDER_RADIUS",
    "MAX_BORDER_WIDTH",
    "PanelAppearance",
    "coerce_appearance",
    "is_hex_color",
    "is_panel_appearance",
    "panel_qss",
]

MAX_BORDER_WIDTH = 8
MAX_BORDER_RADIUS = 24

_HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def is_hex_color(value: Any) -> bool:
    """Return ``True`` if ``value`` is a ``#RGB`` or ``#RRGGBB`` string."""
    return isinstance(value, str) and bool(_HEX_COLOR_RE.match(value.strip()))


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


@dataclass(frozen=True)
class PanelAppearance:
    """Validated colours and border for one runtime panel.

    Attributes:
        foreground: Text colour (``#RGB``/``#RRGGBB``).
        background: Surface colour.
        border_color: Border colour — drawn always, not only on focus.
        border_width: Border thickness in px (``0``..:data:`MAX_BORDER_WIDTH`).
        border_radius: Corner radius in px (``0``..:data:`MAX_BORDER_RADIUS`).

    Raises:
        ValueError: If any colour is not a valid hex string, or if a size
            is out of range.
    """

    foreground: str
    background: str
    border_color: str
    border_width: int = 2
    border_radius: int = 6

    def __post_init__(self) -> None:
        for field_name in ("foreground", "background", "border_color"):
            value = getattr(self, field_name)
            if not is_hex_color(value):
                raise ValueError(
                    f"{field_name} must be a #RGB or #RRGGBB hex colour, got {value!r}"
                )
            object.__setattr__(self, field_name, value.strip())
        if not 0 <= self.border_width <= MAX_BORDER_WIDTH:
            raise ValueError(
                f"border_width must be 0..{MAX_BORDER_WIDTH}, got {self.border_width}"
            )
        if not 0 <= self.border_radius <= MAX_BORDER_RADIUS:
            raise ValueError(
                f"border_radius must be 0..{MAX_BORDER_RADIUS}, "
                f"got {self.border_radius}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe preference payload."""
        return {
            "foreground": self.foreground,
            "background": self.background,
            "border_color": self.border_color,
            "border_width": self.border_width,
            "border_radius": self.border_radius,
        }

    def with_overrides(self, **overrides: Any) -> PanelAppearance:
        """Return a copy with selected fields replaced (validated)."""
        return replace(self, **overrides)


# Sensible defaults. Terminal/REPL read dark; the workspace inspector reads
# light to match the table-on-white look hosts expect.
DEFAULT_DARK_PANEL_APPEARANCE = PanelAppearance(
    foreground="#e6e6e6",
    background="#1e1e2e",
    border_color="#89b4fa",
    border_width=2,
    border_radius=6,
)
DEFAULT_LIGHT_PANEL_APPEARANCE = PanelAppearance(
    foreground="#1e1e1e",
    background="#ffffff",
    border_color="#3b82f6",
    border_width=2,
    border_radius=6,
)


def is_panel_appearance(value: object) -> bool:
    """Return True for canonical or compatibility-imported appearances."""
    if isinstance(value, PanelAppearance):
        return True

    value_type = type(value)
    module_name = getattr(value_type, "__module__", "")
    return (
        value_type.__name__ == "PanelAppearance"
        and module_name.endswith(".ui.tools_sidebar.appearance")
        and all(
            hasattr(value, field_name)
            for field_name in (
                "foreground",
                "background",
                "border_color",
                "border_width",
                "border_radius",
                "to_dict",
                "with_overrides",
            )
        )
    )


def coerce_appearance(
    values: Mapping[str, Any] | None,
    base: PanelAppearance = DEFAULT_DARK_PANEL_APPEARANCE,
) -> PanelAppearance:
    """Return a :class:`PanelAppearance` from possibly-stale ``values``.

    Tolerant: each field falls back to ``base`` when missing or invalid, so
    hand-edited or out-of-date persisted state never raises.

    Args:
        values: Raw mapping (or ``None``).
        base: Appearance supplying fallbacks for missing/invalid fields.

    Returns:
        A validated :class:`PanelAppearance`.

    Raises:
        TypeError: If ``values`` is provided but is not a mapping.
    """
    if values is None:
        return base
    if not isinstance(values, Mapping):
        raise TypeError("appearance values must be a mapping or None")

    def _color(key: str, fallback: str) -> str:
        candidate = values.get(key)
        return str(candidate).strip() if is_hex_color(candidate) else fallback

    return PanelAppearance(
        foreground=_color("foreground", base.foreground),
        background=_color("background", base.background),
        border_color=_color("border_color", base.border_color),
        border_width=_clamp_int(
            values.get("border_width"), 0, MAX_BORDER_WIDTH, base.border_width
        ),
        border_radius=_clamp_int(
            values.get("border_radius"), 0, MAX_BORDER_RADIUS, base.border_radius
        ),
    )


def panel_qss(object_name: str, appearance: PanelAppearance) -> str:
    """Return a stylesheet giving ``object_name``'s panels a visible border.

    Targets the common editable/scrollable children (``QPlainTextEdit``,
    ``QLineEdit``, ``QTableView``, ``QListWidget``) so a single call styles
    a terminal, a REPL, or a workspace table identically (DRY).

    Args:
        object_name: The host widget's ``objectName()``.
        appearance: Validated colours/border to apply.

    Raises:
        ValueError: If ``object_name`` is empty.
        TypeError: If ``appearance`` is not a :class:`PanelAppearance`.
    """
    if not isinstance(object_name, str) or not object_name.strip():
        raise ValueError("object_name must be a non-empty string")
    if not is_panel_appearance(appearance):
        raise TypeError("appearance must be a PanelAppearance")

    scope = f"QWidget#{object_name}"
    panels = ", ".join(
        f"{scope} {child}"
        for child in ("QPlainTextEdit", "QLineEdit", "QTableView", "QListWidget")
    )
    return f"""
{panels} {{
    color: {appearance.foreground};
    background-color: {appearance.background};
    border: {appearance.border_width}px solid {appearance.border_color};
    border-radius: {appearance.border_radius}px;
    selection-background-color: {appearance.border_color};
    selection-color: {appearance.background};
}}

{scope} QTableView {{
    gridline-color: {appearance.border_color};
}}

{scope} QHeaderView::section {{
    color: {appearance.foreground};
    background-color: {appearance.background};
    border: 1px solid {appearance.border_color};
    padding: 3px;
}}
""".strip()
