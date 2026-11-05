"""The putting tile's dark controls-panel stylesheet (#14473)."""

from __future__ import annotations

from src.shared.python.theme.tool_stylesheet import get_tool_colors

PANEL_STYLE = """
QWidget#puttControls { background-color: #1b2530; }
QLabel, QCheckBox { color: #d7e2ec; }
QLabel#puttTitle { color: #ffffff; font-size: 16px; font-weight: bold; }
QGroupBox {
    color: #9fd8a8; border: 1px solid #2f4254; border-radius: 6px;
    margin-top: 10px; padding: 8px; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #0f1822; color: #eaf2f8;
    border: 1px solid #2f4254; border-radius: 4px; padding: 3px;
}
QPushButton {
    background-color: #243343; color: #eaf2f8;
    border: 1px solid #34536b; border-radius: 5px; padding: 6px;
}
QPushButton:hover { background-color: #2c3f52; }
"""


def panel_style() -> str:
    """The panel stylesheet; disabled inputs dim to the shared theme tokens."""
    colors = get_tool_colors()
    disabled = (
        "QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled {"
        f" background-color: {colors['surface_secondary']};"
        f" color: {colors['text_secondary']};"
        f" border-color: {colors['border_default']}; }}"
    )
    return PANEL_STYLE + disabled + "\n"
