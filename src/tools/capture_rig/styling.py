"""Every colour and stylesheet string the Capture Rig tile uses (#9816).

No other module in the package names a colour. They ask this one, which
reads the active palette (:func:`get_current_colors`) and the fleet's
read-only :class:`Styles` constants and composes the strings; spacing comes
from :class:`LayoutMetrics`. Signal colours (recording red, warning amber,
success green) are not palette tokens, so they are taken from the ``Styles``
status constants rather than written here. :func:`connect_theme_changed`
lets a widget restyle itself when the operator switches theme, and
:func:`apply_theme` puts the generated stylesheet on a root widget (the
standalone window and the launcher's embedded widget alike).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget

from src.shared.python.core.contracts import require
from src.shared.python.theme.color_derivation import is_dark_bg
from src.shared.python.theme.layout_metrics import LayoutMetrics
from src.shared.python.theme.palette import get_current_colors
from src.shared.python.theme.style_constants import Styles
from src.shared.python.theme.theme_manager import ThemeManager

Palette = Mapping[str, str]


def reference_marker_colors() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """BGR marker and contrasting outline from the current application palette."""
    palette = get_current_colors()
    foreground = palette["accent"]
    outline = contrast_text(foreground, palette)
    front, back = QColor(foreground), QColor(outline)
    return (
        (front.blue(), front.green(), front.red()),
        (back.blue(), back.green(), back.red()),
    )


def help_document_style() -> str:
    """Keep rich-text help links readable against the active window theme."""
    return f"a {{ color: {get_current_colors()['text']}; text-decoration: underline; }}"


_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
CHIP_KINDS: tuple[str, ...] = ("neutral", "ok", "warning", "record")
STEP_KINDS: tuple[str, ...] = ("done", "current", "ready", "blocked", "skipped")
HEADER_OBJECT_NAME = "captureRigHeader"


def color_of(style: str) -> str:
    """The first ``#rrggbb`` in a ``Styles`` constant, lower-cased.

    Precondition: ``style`` names a colour (``ValueError`` otherwise).
    """
    match = _HEX.search(style)
    require(match is not None, "style must name a colour", style)
    assert match is not None
    return match.group(0).lower()


@dataclass(frozen=True)
class SignalColors:
    """State colours that must read the same in every theme."""

    record: str
    warning: str
    ok: str
    idle: str


def signal_colors(colors: Palette | None = None) -> SignalColors:
    """Recording / warning / success from ``Styles``; idle from the palette."""
    palette = colors if colors is not None else get_current_colors()
    return SignalColors(
        record=color_of(Styles.STATUS_ERROR),
        warning=color_of(Styles.STATUS_WARNING),
        ok=color_of(Styles.STATUS_SUCCESS),
        idle=palette["text_secondary"],
    )


def contrast_text(background: str, colors: Palette) -> str:
    """The palette's text or background colour, whichever reads on ``background``.

    Postcondition: the result is one of ``colors["text"]`` / ``colors["bg"]``.
    """
    same_side = is_dark_bg(background) == is_dark_bg(colors["bg"])
    return colors["text"] if same_side else colors["bg"]


# -- stylesheet strings --------------------------------------------------------------
def tile_style(colors: Palette | None = None) -> str:
    """Preview tile: a recessed panel that shows the frame's letterbox."""
    c = colors if colors is not None else get_current_colors()
    return (
        f"background: {c['bg']}; color: {c['text_secondary']}; "
        f"border: 1px solid {c['border']};"
    )


def header_style(colors: Palette | None = None) -> str:
    """The toolbar-like header frame (session line, status strip, layouts)."""
    c = colors if colors is not None else get_current_colors()
    return (
        f"QFrame#{HEADER_OBJECT_NAME} {{ background: {c['title_bg']}; "
        f"border: 1px solid {c['title_border']}; "
        f"border-radius: {LayoutMetrics.RADIUS_SM}px; }} "
        f"QFrame#{HEADER_OBJECT_NAME} QLabel {{ background: transparent; "
        f"border: none; color: {c['text']}; }}"
    )


def section_label_style(colors: Palette | None = None) -> str:
    """Muted caption above / beside a group of buttons."""
    c = colors if colors is not None else get_current_colors()
    return f"color: {c['label']}; background: transparent; border: none;"


def chip_style(kind: str, colors: Palette | None = None) -> str:
    """A small rounded status chip. Precondition: ``kind`` in :data:`CHIP_KINDS`."""
    require(kind in CHIP_KINDS, "unknown chip kind", kind)
    c = colors if colors is not None else get_current_colors()
    signals = signal_colors(c)
    foreground = {
        "neutral": c["text_secondary"],
        "ok": signals.ok,
        "warning": signals.warning,
        "record": signals.record,
    }[kind]
    weight = "bold" if kind == "record" else "normal"
    return (
        f"QLabel {{ color: {foreground}; background: {c['group_bg']}; "
        f"border: 1px solid {foreground}; "
        f"border-radius: {LayoutMetrics.RADIUS_SM}px; "
        f"padding: 1px {LayoutMetrics.SPACING_SM}px; font-weight: {weight}; }}"
    )


def step_row_style(kind: str, colors: Palette | None = None) -> str:
    """One step rail row. Precondition: ``kind`` in :data:`STEP_KINDS`.

    Postcondition: the current step is the only bold row, so the eye lands
    on where the operator is.
    """
    require(kind in STEP_KINDS, "unknown step kind", kind)
    c = colors if colors is not None else get_current_colors()
    foreground = {
        "done": signal_colors(c).ok,
        "current": c["accent"],
        "ready": c["text"],
        "blocked": c["text_secondary"],
        "skipped": c["label"],
    }[kind]
    weight = "bold" if kind == "current" else "normal"
    return (
        f"QLabel {{ color: {foreground}; background: transparent; "
        f"border: none; font-weight: {weight}; }}"
    )


def rail_action_style(primary: bool) -> str:
    """The rail's one big next action, or one of its lesser siblings."""
    return Styles.BTN_PRIMARY if primary else Styles.BTN_SECONDARY


def readout_style(recording: bool, countdown: bool) -> str:
    """The transport strip's REC readout: red while recording, amber counting down."""
    if recording:
        return Styles.STATUS_ERROR_BOLD
    return Styles.STATUS_WARNING if countdown else Styles.COLOR_RESET


def record_button_style(recording: bool) -> str:
    return Styles.BTN_STOP if recording else Styles.BTN_RUN


# -- theme plumbing ------------------------------------------------------------------
def theme_manager() -> ThemeManager:
    return ThemeManager.instance()


def apply_theme(root: QWidget) -> None:
    """Put the generated stylesheet on ``root`` and keep it updated on theme change."""
    require(root is not None, "root widget is required")
    theme_manager().apply_theme_to_window(root)


def connect_theme_changed(slot: Callable[[str], None] | Callable[[], None]) -> None:
    """Call ``slot`` whenever the operator switches theme."""
    theme_manager().themeChanged.connect(slot)


def compact_tabs_style() -> str:
    """Keep short inspector tabs visible when the preview owns most of the width."""
    return (
        "QTabBar::tab { min-width: 0px; "
        f"padding: {LayoutMetrics.SPACING_SM}px {LayoutMetrics.SPACING_MD}px; }}"
    )
