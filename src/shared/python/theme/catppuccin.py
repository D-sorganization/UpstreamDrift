"""Catppuccin Mocha theme - shared color palette and stylesheet.

This module centralises the Catppuccin Mocha color palette and the
standard application-wide QSS stylesheet.
"""

from __future__ import annotations

__all__ = [
    "CATPPUCCIN_MOCHA",
    "get_stylesheet",
]

CATPPUCCIN_MOCHA: dict[str, str] = {
    "rosewater": "#f5e0dc",
    "flamingo": "#f2cdcd",
    "pink": "#f5c2e7",
    "mauve": "#cba6f7",
    "red": "#f38ba8",
    "maroon": "#eba0ac",
    "peach": "#fab387",
    "yellow": "#f9e2af",
    "green": "#a6e3a1",
    "teal": "#94e2d5",
    "sky": "#89dceb",
    "sapphire": "#74c7ec",
    "blue": "#89b4fa",
    "lavender": "#b4befe",
    "text": "#cdd6f4",
    "subtext1": "#bac2de",
    "subtext0": "#a6adc8",
    "overlay2": "#9399b2",
    "overlay1": "#7f849c",
    "overlay0": "#6c7086",
    "surface2": "#585b70",
    "surface1": "#45475a",
    "surface0": "#313244",
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
}


def get_stylesheet(palette: dict[str, str] | None = None) -> str:
    """Build a complete Qt stylesheet from a colour palette.

    Args:
        palette: Colour palette dictionary. Defaults to CATPPUCCIN_MOCHA.

    Returns:
        A Qt stylesheet string ready for ""setStyleSheet()"".
    """
    p = palette or CATPPUCCIN_MOCHA
    return f"""
QMainWindow {{
    background-color: {p["base"]};
}}

QWidget {{
    background-color: {p["base"]};
    color: {p["text"]};
    font-family: "Segoe UI", "Arial", sans-serif;
}}

QScrollArea {{
    border: none;
    background-color: {p["base"]};
}}

QTabWidget::pane {{
    border: 1px solid {p["surface1"]};
    background-color: {p["mantle"]};
    border-radius: 4px;
}}

QTabBar::tab {{
    background-color: {p["surface0"]};
    color: {p["subtext1"]};
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background-color: {p["surface1"]};
    color: {p["blue"]};
}}

QTabBar::tab:hover {{
    background-color: {p["surface1"]};
}}

QGroupBox {{
    background-color: {p["surface0"]};
    border: 1px solid {p["surface1"]};
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {p["mauve"]};
}}

QLabel {{
    color: {p["text"]};
    background-color: transparent;
}}

QLabel[class="result-label"] {{
    color: {p["green"]};
    font-weight: bold;
}}

QLabel[class="unit-label"] {{
    color: {p["subtext0"]};
}}

QLabel[class="header-label"] {{
    color: {p["blue"]};
    font-size: 14px;
    font-weight: bold;
}}

QListWidget {{
    background-color: {p["surface0"]};
    color: {p["text"]};
    border: 1px solid {p["surface2"]};
    border-radius: 4px;
    padding: 4px;
}}

QListWidget::item {{
    padding: 4px;
}}

QListWidget::item:selected {{
    background-color: {p["surface2"]};
    color: {p["blue"]};
}}

QTableWidget {{
    background-color: {p["surface0"]};
    color: {p["text"]};
    border: 1px solid {p["surface2"]};
    border-radius: 4px;
    gridline-color: {p["surface1"]};
}}

QTableWidget::item {{
    padding: 4px;
}}

QHeaderView::section {{
    background-color: {p["surface1"]};
    color: {p["text"]};
    padding: 6px;
    border: none;
}}

QLineEdit, QDoubleSpinBox, QSpinBox {{
    background-color: {p["surface0"]};
    color: {p["text"]};
    border: 1px solid {p["surface2"]};
    border-radius: 4px;
    padding: 6px 10px;
    selection-background-color: {p["surface2"]};
}}

QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {p["blue"]};
}}

QComboBox {{
    background-color: {p["surface0"]};
    color: {p["text"]};
    border: 1px solid {p["surface2"]};
    border-radius: 4px;
    padding: 6px 10px;
    min-width: 120px;
}}

QComboBox:hover {{
    border: 1px solid {p["blue"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {p["text"]};
    margin-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {p["surface0"]};
    color: {p["text"]};
    selection-background-color: {p["surface2"]};
    border: 1px solid {p["surface1"]};
}}

QTextEdit {{
    background-color: {p["surface0"]};
    color: {p["text"]};
    border: 1px solid {p["surface2"]};
    border-radius: 4px;
    padding: 8px;
    font-family: "Consolas", "Courier New", monospace;
}}

QSlider::groove:horizontal {{
    background-color: {p["surface1"]};
    border-radius: 3px;
    height: 6px;
}}

QSlider::handle:horizontal {{
    background-color: {p["blue"]};
    border-radius: 7px;
    margin: -4px 0;
    width: 14px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {p["sapphire"]};
}}

QPushButton {{
    background-color: {p["blue"]};
    color: {p["crust"]};
    border: none;
    border-radius: 4px;
    padding: 10px 24px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: {p["sapphire"]};
}}

QPushButton:pressed {{
    background-color: {p["lavender"]};
}}

QPushButton:disabled {{
    background-color: {p["surface2"]};
    color: {p["overlay0"]};
}}

QPushButton#loadBtn {{
    background-color: {p["green"]};
}}

QPushButton#loadBtn:hover {{
    background-color: {p["teal"]};
}}

QFrame[class="separator"] {{
    background-color: {p["surface1"]};
}}
"""
