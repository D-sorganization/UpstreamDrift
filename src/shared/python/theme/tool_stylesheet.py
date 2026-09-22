"""Shared stylesheet helpers for ``src/tools/*`` GUIs (issue #8885).

UD-owned home for the tokens and stylesheet snippets that ``src/tools``
widgets should use instead of hardcoding hex colors in ``setStyleSheet``
calls. This mirrors ``layout_metrics.py`` (issue #8972): it cannot live on
``style_constants.Styles`` because that module is a Tools-owned child copy
which UpstreamDrift must not edit directly
(``tests/unit/repo_hygiene/test_tools_child_copy_contract.py``).

Widgets should call :func:`apply_tool_theme` / :func:`primary_button_style`
rather than reaching into theme submodules directly (Law of Demeter) --
this module is the single place that knows how to resolve a live theme
colour and what the fallback palette is.

Design by Contract:
    Precondition: none -- every function here degrades gracefully when the
        live theme cannot be resolved (see issue #8972 for the underlying
        ``_get_theme_colors`` gap this module works around).
    Postcondition: every returned stylesheet string is non-empty and every
        colour token is a valid ``#rrggbb`` hex literal.
"""

from __future__ import annotations

from typing import Any

# Fallback palette, matching the existing dark-app default already used by
# ``terrain_engine/gui.py`` before this module existed. Used whenever the
# live theme cannot be resolved.
_FALLBACK_COLORS: dict[str, str] = {
    "surface_primary": "#1f2329",
    "surface_secondary": "#252a31",
    "border_default": "#3a414a",
    "text_primary": "#f0f3f6",
    "text_secondary": "#a8b0bb",
}

# Canonical "primary action" token (issue #8885): before this change the
# primary "run" button was a different colour in every tool
# (putting_green_gui #2E7D32, ball_flight_gui #1565C0,
# swing_flight_pipeline #4CAF50, training_controller #0A84FF). Blue matches
# the web app's canonical primary accent (see
# ``ui/src/utils/colorGuard.allowlist.json``), so it is the one kept.
PRIMARY_ACTION_COLOR = "#0A84FF"
PRIMARY_ACTION_HOVER_COLOR = "#2997FF"
PRIMARY_ACTION_TEXT_COLOR = "#ffffff"
PRIMARY_ACTION_DISABLED_COLOR = "#555555"
PRIMARY_ACTION_DISABLED_TEXT_COLOR = "#aaaaaa"


def _resolve_color(attr: str, fallback: str) -> str:
    """Resolve one theme colour by attribute name, falling back safely.

    Isolates every caller from *how* the live theme is fetched (Law of
    Demeter): callers never import ``src.shared.python.theme.palette``
    themselves.
    """
    try:
        from src.shared.python.theme.palette import get_current_colors
    except ImportError:
        try:
            from src.shared.python.theme.palette import DARK_THEME

            colors: Any = DARK_THEME
        except ImportError:
            return fallback
    else:
        try:
            colors = get_current_colors()
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
            return fallback
    if isinstance(colors, dict):
        return str(colors.get(attr, fallback))
    return str(getattr(colors, attr, fallback))


def get_tool_colors() -> dict[str, str]:
    """Return the resolved (live-theme-or-fallback) base colour tokens."""
    return {
        attr: _resolve_color(attr, fallback)
        for attr, fallback in _FALLBACK_COLORS.items()
    }


def apply_tool_theme(widget: Any) -> None:
    """Apply the shared dark-app tool stylesheet to *widget*.

    Postcondition: ``widget.styleSheet()`` is non-empty and uses only the
    resolved theme tokens -- no hardcoded hex literals.
    """
    colors = get_tool_colors()
    widget.setStyleSheet(f"""
        QWidget {{
            background: {colors["surface_primary"]};
            color: {colors["text_primary"]};
        }}
        QGroupBox {{
            border: 1px solid {colors["border_default"]};
            border-radius: 8px;
            margin-top: 12px;
            padding: 12px 10px 10px 10px;
            background: {colors["surface_secondary"]};
        }}
        QLabel {{
            color: {colors["text_secondary"]};
        }}
    """)


def wire_primary_action_button(button: Any, on_clicked: Any) -> None:
    """Style a tool's primary action button and connect its click handler."""
    button.setStyleSheet(primary_button_style())
    button.clicked.connect(on_clicked)


def error_field_border_style() -> str:
    """Return a red border for invalid field input (issues #8887, #8885).

    Callers under ``src/tools`` should use this instead of embedding a hex
    literal in ``setStyleSheet`` (see ``check_hardcoded_style_ratchet.py``).
    """
    color = _resolve_color("error", "#d06060")
    return f"border: 2px solid {color};"


def primary_button_style() -> str:
    """Return the shared stylesheet snippet for a tool's primary action button.

    Every "run" button across ``src/tools`` should use this instead of a
    hardcoded ``background-color`` literal, so the primary action reads as
    one consistent colour fleet-wide (issue #8885).
    """
    return f"""
        QPushButton {{
            background-color: {PRIMARY_ACTION_COLOR};
            color: {PRIMARY_ACTION_TEXT_COLOR};
            font-weight: bold;
            padding: 12px;
            border-radius: 5px;
        }}
        QPushButton:hover {{
            background-color: {PRIMARY_ACTION_HOVER_COLOR};
        }}
        QPushButton:disabled {{
            background-color: {PRIMARY_ACTION_DISABLED_COLOR};
            color: {PRIMARY_ACTION_DISABLED_TEXT_COLOR};
        }}
    """
