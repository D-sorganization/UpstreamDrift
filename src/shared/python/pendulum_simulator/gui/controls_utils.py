"""
Shared utilities for double and triple pendulum control widgets.

DRY: All parse helpers, stylesheet tokens, and font constants
     used by ControlsWidget, ControlsWidgetTriple, etc. live here.

Design by Contract
------------------
- parse_float raises ValueError (not returns None) on bad input
- parse_coeffs raises ValueError on non-numeric tokens
- Font size constants must all be >= MIN_FONT_PX
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .controls_widget import LabeledInput

# ---------------------------------------------------------------------------
# Stylesheet tokens shared by both control panels
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Font size constants — DRY, single source of truth (#1134)
# All font sizes in pixels; zoom system (#1147) scales these.
# ---------------------------------------------------------------------------
MIN_FONT_PX: int = 11  # absolute minimum for readability
FONT_BODY: int = 11  # body text, labels, checkboxes
FONT_GROUP: int = 11  # group box titles
FONT_EDIT: int = 11  # text inputs (monospace)
FONT_BTN: int = 11  # button labels
FONT_TITLE: int = 14  # major titles
FONT_STATUS: int = 11  # status bar text

assert all(
    v >= MIN_FONT_PX for v in (FONT_BODY, FONT_GROUP, FONT_EDIT, FONT_BTN, FONT_STATUS)
), "All font sizes must meet minimum readability threshold"

STYLE_GROUP = (
    f"QGroupBox {{ color: #c8c8e0; border: 1px solid #404060;"
    f"border-radius: 6px; margin-top: 10px; padding-top: 14px;"
    f"font-weight: bold; font-size: {FONT_GROUP}px; }}"
    f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}"
)
STYLE_EDIT = (
    f"background: #1e1e30; color: #e0e0f0; border: 1px solid #404060;"
    f"border-radius: 3px; padding: 2px 5px; font-family: monospace; font-size: {FONT_EDIT}px;"
)
STYLE_LABEL = f"color: #9090b0; font-size: {FONT_BODY}px;"
STYLE_SPIN = (
    f"background: #1e1e30; color: #e0e0f0; border: 1px solid #404060;"
    f"border-radius: 3px; padding: 1px 4px; font-size: {FONT_BODY}px;"
)
STYLE_CHECK = (
    f"QCheckBox {{ color: #b8b8d0; font-size: {FONT_BODY}px; spacing: 4px; }}"
    f"QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid #484868;"
    f"border-radius: 3px; background: #22223a; }}"
    f"QCheckBox::indicator:checked {{ background: #5060a0; border-color: #7080c0; }}"
)
STYLE_SLIDER = (
    "QSlider::groove:horizontal { background: #252540; height: 6px; border-radius: 3px; }"
    "QSlider::sub-page:horizontal { background: #5060a0; border-radius: 3px; }"
    "QSlider::handle:horizontal { background: #8090d0; width: 14px;"
    "margin: -5px 0; border-radius: 7px; border: 1px solid #6070b0; }"
    "QSlider::handle:horizontal:hover { background: #a0b0f0; }"
)
STYLE_BTN = (
    f"QPushButton {{ background: #282848; color: #c0c0e8; border: 1px solid #404068;"
    f"border-radius: 5px; padding: 6px 14px; font-size: {FONT_BTN}px; }}"
    f"QPushButton:hover {{ background: #30306a; color: #e0e0ff; }}"
    f"QPushButton:pressed {{ background: #20204a; }}"
)
STYLE_BTN_IMPORT = (
    f"QPushButton {{ background: #1e4a2a; color: #a0e8b0; border: 1px solid #285a38;"
    f"border-radius: 5px; padding: 7px 14px; font-weight: bold; font-size: {FONT_BTN}px; }}"
    f"QPushButton:hover {{ background: #286038; }}"
    f"QPushButton:pressed {{ background: #153820; }}"
)

# ── Full-window dark stylesheet (fallback when fleet ThemeManager unavailable)
# TODO(#1042): Derive from fleet ThemeManager palette when it's a hard dep.
PENDULUM_DARK_STYLE = f"""
    QMainWindow {{ background: #12121c; }}
    QStatusBar  {{ background: #12121c; color: #7878a0; font-size: {FONT_STATUS}px;
                  border-top: 1px solid #282840; }}
    QTabWidget::pane {{ border: 1px solid #303050; background: #12121c; }}
    QTabBar::tab {{ background: #1e1e30; color: #9090b0; border: 1px solid #303050;
                   padding: 7px 18px; margin-right: 2px; border-bottom: none;
                   font-size: {FONT_BTN}px; }}
    QTabBar::tab:selected {{ background: #282848; color: #d0d0f0;
                            border-bottom: 2px solid #6070c0; }}
    QTabBar::tab:hover    {{ background: #222238; color: #c0c0e8; }}
    QSplitter::handle {{ background: #282848; width: 4px; }}
    QSplitter::handle:hover {{ background: #404068; }}
    QScrollBar:vertical {{ background: #1a1a2a; width: 10px; border: none; }}
    QScrollBar::handle:vertical {{ background: #404060; min-height: 20px;
                                  border-radius: 5px; }}
    QScrollBar::handle:vertical:hover {{ background: #5060a0; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{ background: #1a1a2a; height: 10px; border: none; }}
    QScrollBar::handle:horizontal {{ background: #404060; min-width: 20px;
                                    border-radius: 5px; }}
    QScrollBar::handle:horizontal:hover {{ background: #5060a0; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QLabel {{ color: #c0c0d8; font-size: {FONT_BODY}px; }}
    QMenuBar {{ background: #16162a; color: #b0b0d0; font-size: {FONT_BODY}px; }}
    QMenuBar::item:selected {{ background: #282848; }}
    QMenu {{ background: #1e1e30; color: #c0c0d8; border: 1px solid #404060;
            font-size: {FONT_BODY}px; }}
    QMenu::item:selected {{ background: #383868; }}
"""


# ---------------------------------------------------------------------------
# Input parsing helpers
# ---------------------------------------------------------------------------


def parse_float(widget: LabeledInput, name: str) -> float:
    """Parse a single float from a LabeledInput; raise ValueError on failure.

    Parameters
    ----------
    widget:
        The LabeledInput whose .value is parsed.
    name:
        Human-readable field name used in the error message.

    Raises
    ------
    ValueError
        If the text cannot be converted to float.
    """
    try:
        return float(widget.value)
    except ValueError:
        raise ValueError(f"Cannot parse '{name}': '{widget.value}'") from None


def parse_coeffs(widget: LabeledInput, name: str) -> list[float]:
    """Parse comma-separated polynomial coefficients from a LabeledInput.

    Parameters
    ----------
    widget:
        The LabeledInput whose .value is parsed.
    name:
        Human-readable field name used in the error message.

    Returns
    -------
    list[float]
        Parsed coefficients; empty list converts to [0.0] at call site.

    Raises
    ------
    ValueError
        If any token cannot be converted to float.
    """
    try:
        parts = widget.value.split(",")
        return [float(p.strip()) for p in parts if p.strip()]
    except ValueError:
        raise ValueError(
            f"Cannot parse '{name}' coefficients: '{widget.value}'"
        ) from None


def parse_coeffs_lenient(widget: LabeledInput) -> list[float]:
    """Silently-tolerant version used for live torque preview rendering.

    Returns [0.0] on any parse failure rather than raising.
    """
    parts = widget.value.split(",")
    result: list[float] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            result.append(float(part))
        except ValueError:
            return []
    return result or [0.0]


def clamp_dt(raw: float) -> float:
    """Clamp a raw dt value to a sane simulation range [1e-5, 0.1].

    Precondition: raw is a finite float (already parsed).
    """
    assert isinstance(raw, float), "dt must be a float"
    return max(1e-5, min(0.1, raw))


def require_positive(value: float, name: str) -> float:
    """Require a strictly positive input value."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def require_non_negative(value: float, name: str) -> float:
    """Require a non-negative input value."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value
