"""Qt stylesheet for the Training Controller dashboard.

Kept in its own module so :mod:`gui` stays within the 1200-line file
budget and so the styling can evolve independently of the widget logic.

The palette mirrors the launcher's dark theme accent (``#0A84FF``) and
status colours (``#30D158`` success, ``#FF375F`` error) used elsewhere
in the fleet; no new brand colours are introduced here.
"""

from __future__ import annotations

__all__ = ["DARK_STYLE"]


# Sleek dark QSS theme. Colours intentionally match the shared dark
# palette accent / status hues rather than introducing new ones.
DARK_STYLE = """
QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 13px;
}
QTableWidget {
    background-color: #252526;
    alternate-background-color: #1e1e1e;
    gridline-color: #2d2d2d;
    border: 1px solid #3e3e42;
    border-radius: 4px;
}
QTableWidget::item:selected {
    background-color: #0A84FF;
    color: #ffffff;
}
QHeaderView::section {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #3e3e42;
    padding: 6px;
    font-weight: bold;
}
QListWidget {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 4px;
}
QListWidget::item:hover {
    background-color: #2d2d2d;
}
QListWidget::item:selected {
    background-color: #0A84FF;
    color: #ffffff;
}
QTabWidget::pane {
    border: 1px solid #3e3e42;
    background-color: #1e1e1e;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #2d2d2d;
    color: #888888;
    padding: 8px 16px;
    margin-right: 4px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #3e3e42;
    border-bottom: none;
}
QTabBar::tab:hover {
    background-color: #333333;
    color: #d4d4d4;
}
QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: #ffffff;
    border-bottom: 2px solid #0A84FF;
}
QPushButton {
    background-color: #333333;
    border: 1px solid #444444;
    border-radius: 4px;
    color: #ffffff;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #444444;
    border-color: #555555;
}
QPushButton:pressed {
    background-color: #0A84FF;
    border-color: #0A84FF;
}
QPushButton:disabled {
    background-color: #222222;
    border-color: #333333;
    color: #666666;
}
QPushButton#submit-btn {
    background-color: #0A84FF;
    border-color: #0A84FF;
    font-weight: bold;
}
QPushButton#submit-btn:hover {
    background-color: #2997FF;
}
QLineEdit, QComboBox {
    background-color: #2d2d2d;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 6px;
    color: #ffffff;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #0A84FF;
}
QDialog {
    background-color: #1e1e1e;
}
"""
