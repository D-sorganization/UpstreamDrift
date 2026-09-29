"""Pendulum Simulator GUI theme — thin adapter over ``theme/v1``.

This module used to be a free-standing, hand-rolled dark palette. It is now a
**thin adapter** that declares the pendulum dark palette as a
:class:`src.shared.python.theme.v1.RolePalette` conforming to the fleet-wide
versioned theme contract (``theme/v1``), then derives its public colour
constants from that single source.

The exact hex values are unchanged — visual output is preserved byte-for-byte
— but the palette now speaks the canonical role-token vocabulary
(``bg``, ``group_bg``, ``accent`` …) and the diagnostics severity colours are
sourced from the contract's :func:`~src.shared.python.theme.v1.severity_palette`
rather than being duplicated here.

Closes #1197: DRY — Define every piece of knowledge in exactly one place.
Refactored under #6566: unify the three theme implementations.

Design by Contract
------------------
- All colors are hex strings starting with '#'.
- All stylesheet constants are valid Qt stylesheet syntax.
- Font sizes are in pixels (px) for consistency.
"""

from __future__ import annotations

from src.shared.python.theme.v1 import RolePalette, severity_palette

# ---------------------------------------------------------------------------
# Color palette — dark theme (declared via the versioned theme contract)
# ---------------------------------------------------------------------------
# The pendulum dark palette expressed in the canonical ``theme/v1`` role
# vocabulary. Module-level constants below delegate to this single object so
# there is exactly one place each colour is defined.
PALETTE = RolePalette(
    name="Pendulum Dark",
    tokens={
        "bg": "#12121c",  # Main panel background
        "group_bg": "#1a1a2e",  # Card / group box background
        "input_bg": "#0e0e1e",  # Deep background (text editors, code areas)
        "border": "#2a2a4a",  # Default border
        "text": "#c0c0e0",  # Primary text
        "text_secondary": "#9090c8",  # Secondary text, labels
        "label": "#808090",  # Muted text, metadata
        "focus": "#4888c8",  # Primary accent (selections, links)
        "accent": "#4888c8",  # Primary accent (selections, links)
        "title_bg": "#1a1a2e",  # Title/header background (= group bg)
        "title_border": "#2a2a4a",  # Title/header border (= default border)
        "table_header": "#1a1a2e",  # Table header background (= group bg)
        "table_alt": "#12121c",  # Alternating table row (= main bg)
        "button_hover": "#303060",  # Hover state
        "success": "#50a060",  # Success, positive values
        "warning": "#e0a060",  # Warning, attention
        "error": "#d06060",  # Error, negative values, critical
        "info": "#8060c0",  # Special, info
    },
)

# ---------------------------------------------------------------------------
# Color constants — delegate to the versioned-contract palette
# ---------------------------------------------------------------------------

# Base backgrounds (darkest → lightest)
BG_DARKEST = PALETTE["input_bg"]  # Deep background (text editors, code areas)
BG_DARK = PALETTE["bg"]  # Main panel background
BG_MEDIUM = PALETTE["group_bg"]  # Card / group box background
BG_ELEVATED = "#252540"  # Elevated surfaces (buttons, inputs)
BG_HOVER = PALETTE["button_hover"]  # Hover state
BG_ACTIVE = "#404060"  # Active/selected state

# Text colors
TEXT_PRIMARY = PALETTE["text"]  # Primary text
TEXT_SECONDARY = PALETTE["text_secondary"]  # Secondary text, labels
TEXT_MUTED = PALETTE["label"]  # Muted text, metadata
TEXT_DISABLED = "#505070"  # Disabled text

# Accent colors
ACCENT_BLUE = PALETTE["accent"]  # Primary accent (selections, links)
ACCENT_GREEN = PALETTE["success"]  # Success, positive values
ACCENT_RED = PALETTE["error"]  # Error, negative values, critical
ACCENT_AMBER = PALETTE["warning"]  # Warning, attention
ACCENT_PURPLE = PALETTE["info"]  # Special, info

# Border colors
BORDER_DEFAULT = PALETTE["border"]  # Default border
BORDER_ACCENT = "#404060"  # Accent border (inputs, focused)

# Matrix / physics display
MATRIX_BG = "#1a1a2a"  # Matrix cell background
MATRIX_BORDER = "#303050"  # Matrix cell border

# Button styles
BTN_PRIMARY_BG = "#5060a0"  # Primary button background
BTN_PRIMARY_HOVER = "#6070b0"  # Primary button hover
BTN_DANGER_BG = "#804040"  # Danger button background

# ---------------------------------------------------------------------------
# Font specifications
# ---------------------------------------------------------------------------

FONT_FAMILY = "'Segoe UI', 'Inter', 'Roboto', sans-serif"
FONT_MONO = "'Consolas', 'Courier New', monospace"
FONT_SIZE_XS = "10px"
FONT_SIZE_SM = "11px"
FONT_SIZE_MD = "12px"
FONT_SIZE_LG = "13px"
FONT_SIZE_XL = "15px"
FONT_SIZE_TITLE = "18px"

# ---------------------------------------------------------------------------
# Reusable stylesheet fragments
# ---------------------------------------------------------------------------

STYLE_GROUP_BOX = f"""
QGroupBox {{
    background: {BG_MEDIUM};
    border: 1px solid {BORDER_DEFAULT};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: {TEXT_SECONDARY};
}}
"""

STYLE_BUTTON = f"""
QPushButton {{
    background: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_ACCENT};
    border-radius: 4px;
    padding: 4px 12px;
    font-weight: bold;
}}
QPushButton:hover {{
    background: {BG_HOVER};
}}
QPushButton:pressed {{
    background: {BG_ACTIVE};
}}
QPushButton:disabled {{
    background: {BG_DARK};
    color: {TEXT_DISABLED};
}}
"""

STYLE_INPUT = f"""
QDoubleSpinBox, QSpinBox, QLineEdit {{
    background: {BG_DARKEST};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_ACCENT};
    border-radius: 3px;
    padding: 2px 4px;
}}
"""

STYLE_COMBO = f"""
QComboBox {{
    background: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_ACCENT};
    padding: 3px 8px;
}}
"""

STYLE_TAB_WIDGET = f"""
QTabWidget::pane {{
    border: 1px solid {BORDER_DEFAULT};
    background: {BG_DARK};
}}
QTabBar::tab {{
    background: {BG_MEDIUM};
    color: {TEXT_MUTED};
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    border-bottom: 2px solid {ACCENT_BLUE};
}}
QTabBar::tab:hover:!selected {{
    background: {BG_HOVER};
}}
"""

STYLE_LABEL_HEADER = f"""
QLabel {{
    color: {TEXT_PRIMARY};
    font-size: {FONT_SIZE_LG};
    font-weight: bold;
}}
"""

STYLE_LABEL_MUTED = f"""
QLabel {{
    color: {TEXT_MUTED};
    font-size: {FONT_SIZE_SM};
}}
"""

STYLE_SCROLLAREA = f"""
QScrollArea {{
    background: {BG_DARK};
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: {BG_DARK};
}}
QScrollBar:vertical {{
    background: {BG_DARK};
    width: 8px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_ACCENT};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""

STYLE_BTN_PRIMARY = f"""
QPushButton {{
    background: {BTN_PRIMARY_BG};
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 16px;
    font-weight: bold;
    font-size: {FONT_SIZE_MD};
}}
QPushButton:hover {{
    background: {BTN_PRIMARY_HOVER};
}}
"""

STYLE_BTN_EXPORT = f"""
QPushButton {{
    background: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_ACCENT};
    border-radius: 4px;
    padding: 4px 10px;
    font-size: {FONT_SIZE_SM};
}}
QPushButton:hover {{
    background: {BG_HOVER};
}}
"""

STYLE_DIAGNOSTICS_DIALOG = f"""
QDialog {{
    background: {BG_MEDIUM};
    color: {TEXT_PRIMARY};
}}
QTableWidget {{
    background: {BG_DARKEST};
    color: {TEXT_PRIMARY};
    gridline-color: {BORDER_DEFAULT};
    border: 1px solid {BORDER_DEFAULT};
    font-size: {FONT_SIZE_SM};
}}
QTableWidget::item:selected {{
    background: #2a4060;
}}
QHeaderView::section {{
    background: #1e1e3a;
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER_DEFAULT};
    padding: 4px;
    font-weight: bold;
}}
QTextEdit {{
    background: {BG_DARKEST};
    color: #a0c0e0;
    border: 1px solid {BORDER_DEFAULT};
    font-family: {FONT_MONO};
    font-size: {FONT_SIZE_SM};
}}
QComboBox {{
    background: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_ACCENT};
    padding: 3px 8px;
}}
QPushButton {{
    background: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_ACCENT};
    border-radius: 4px;
    padding: 4px 12px;
    font-weight: bold;
}}
QPushButton:hover {{
    background: {BG_HOVER};
}}
QLabel {{
    color: {TEXT_SECONDARY};
}}
"""


# ---------------------------------------------------------------------------
# Severity colors (for diagnostics)
# ---------------------------------------------------------------------------

# Sourced from the versioned theme contract so diagnostics severity colours
# stay consistent fleet-wide (values unchanged from the historical palette).
SEVERITY_COLORS: dict[str, str] = severity_palette()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Versioned-contract palette
    "PALETTE",
    # Colors
    "BG_DARKEST",
    "BG_DARK",
    "BG_MEDIUM",
    "BG_ELEVATED",
    "BG_HOVER",
    "BG_ACTIVE",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "TEXT_MUTED",
    "TEXT_DISABLED",
    "ACCENT_BLUE",
    "ACCENT_GREEN",
    "ACCENT_RED",
    "ACCENT_AMBER",
    "ACCENT_PURPLE",
    "BORDER_DEFAULT",
    "BORDER_ACCENT",
    "MATRIX_BG",
    "MATRIX_BORDER",
    "BTN_PRIMARY_BG",
    "BTN_PRIMARY_HOVER",
    "BTN_DANGER_BG",
    # Fonts
    "FONT_FAMILY",
    "FONT_MONO",
    "FONT_SIZE_XS",
    "FONT_SIZE_SM",
    "FONT_SIZE_MD",
    "FONT_SIZE_LG",
    "FONT_SIZE_XL",
    "FONT_SIZE_TITLE",
    # Stylesheets
    "STYLE_GROUP_BOX",
    "STYLE_BUTTON",
    "STYLE_INPUT",
    "STYLE_COMBO",
    "STYLE_TAB_WIDGET",
    "STYLE_LABEL_HEADER",
    "STYLE_LABEL_MUTED",
    "STYLE_SCROLLAREA",
    "STYLE_BTN_PRIMARY",
    "STYLE_BTN_EXPORT",
    "STYLE_DIAGNOSTICS_DIALOG",
    "SEVERITY_COLORS",
]
