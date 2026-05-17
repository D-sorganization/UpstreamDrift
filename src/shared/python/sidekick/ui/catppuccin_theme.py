"""Catppuccin Mocha theme — shared color palette and stylesheet.

This module centralises the Catppuccin Mocha color palette and the
standard application‐wide QSS stylesheet that was previously duplicated
across the Scrubber Calculator and WGS Reactor main windows.
"""

from __future__ import annotations

# Catppuccin Mocha palette
# https://github.com/catppuccin/catppuccin
COLORS: dict[str, str] = {
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
    "text": "#cdd6f4",
    "subtext0": "#a6adc8",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "blue": "#89b4fa",
    "green": "#a6e3a1",
    "yellow": "#f9e2af",
    "red": "#f38ba8",
    "mauve": "#cba6f7",
    "teal": "#94e2d5",
}


def get_stylesheet() -> str:
    """Get the Catppuccin Mocha stylesheet for PyQt6 applications."""
    return f"""
        QMainWindow, QWidget {{
            background-color: {COLORS["base"]};
            color: {COLORS["text"]};
        }}
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {COLORS["surface1"]};
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 10px;
            background-color: {COLORS["mantle"]};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: {COLORS["blue"]};
        }}
        QLabel {{
            color: {COLORS["text"]};
        }}
        QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {{
            background-color: {COLORS["surface0"]};
            border: 1px solid {COLORS["surface1"]};
            border-radius: 4px;
            padding: 5px;
            color: {COLORS["text"]};
        }}
        QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
            border: 1px solid {COLORS["blue"]};
        }}
        QPushButton {{
            background-color: {COLORS["blue"]};
            color: {COLORS["crust"]};
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {COLORS["mauve"]};
        }}
        QPushButton:pressed {{
            background-color: {COLORS["surface2"]};
        }}
        QTableWidget {{
            background-color: {COLORS["mantle"]};
            border: 1px solid {COLORS["surface1"]};
            border-radius: 4px;
            gridline-color: {COLORS["surface1"]};
        }}
        QTableWidget::item {{
            padding: 5px;
            color: {COLORS["text"]};
        }}
        QHeaderView::section {{
            background-color: {COLORS["surface0"]};
            color: {COLORS["text"]};
            padding: 5px;
            border: 1px solid {COLORS["surface1"]};
            font-weight: bold;
        }}
        QTabWidget::pane {{
            border: 1px solid {COLORS["surface1"]};
            border-radius: 4px;
            background-color: {COLORS["base"]};
        }}
        QTabBar::tab {{
            background-color: {COLORS["surface0"]};
            color: {COLORS["text"]};
            padding: 8px 16px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{
            background-color: {COLORS["blue"]};
            color: {COLORS["crust"]};
        }}
        QTextEdit {{
            background-color: {COLORS["mantle"]};
            border: 1px solid {COLORS["surface1"]};
            border-radius: 4px;
            color: {COLORS["text"]};
            padding: 5px;
        }}
        QScrollBar:vertical {{
            background-color: {COLORS["mantle"]};
            width: 12px;
        }}
        QScrollBar::handle:vertical {{
            background-color: {COLORS["surface2"]};
            border-radius: 6px;
        }}
    """
