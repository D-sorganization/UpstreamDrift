"""Style constants for the Golf Swing Visualizer GUI.

Extracted from golf_gui_application.py for Single Responsibility Principle.
"""

from __future__ import annotations


def get_window_tab_style() -> str:
    """Return QSS for main window and tab widgets."""
    return """
        QMainWindow {
            background-color: #ffffff;
            color: #333333;
        }

        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: #ffffff;
        }

        QTabBar::tab {
            background-color: #f0f0f0;
            color: #333333;
            padding: 8px 16px;
            margin-right: 2px;
            border: 1px solid #cccccc;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }

        QTabBar::tab:selected {
            background-color: #ffffff;
            border-bottom: 1px solid #ffffff;
        }

        QTabBar::tab:hover {
            background-color: #e8e8e8;
        }
    """


def get_groupbox_button_style() -> str:
    """Return QSS for group boxes and buttons."""
    return """
        QGroupBox {
            font-weight: bold;
            border: 1px solid #cccccc;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
            background-color: #fafafa;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px 0 4px;
            color: #333333;
        }

        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #106ebe;
        }

        QPushButton:pressed {
            background-color: #005a9e;
        }

        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
    """


def get_slider_checkbox_style() -> str:
    """Return QSS for sliders and checkboxes."""
    return """
        QSlider::groove:horizontal {
            border: 1px solid #cccccc;
            height: 6px;
            background-color: #f0f0f0;
            border-radius: 3px;
        }

        QSlider::handle:horizontal {
            background-color: #0078d4;
            border: 1px solid #0078d4;
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }

        QSlider::handle:horizontal:hover {
            background-color: #106ebe;
        }

        QCheckBox {
            color: #333333;
        }

        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #cccccc;
            border-radius: 2px;
            background-color: #ffffff;
        }

        QCheckBox::indicator:checked {
            background-color: #0078d4;
            border-color: #0078d4;
        }
    """


def get_combobox_label_statusbar_style() -> str:
    """Return QSS for combo boxes, labels, and status bar."""
    return """
        QComboBox {
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 4px 8px;
            background-color: #ffffff;
            color: #333333;
        }

        QComboBox::drop-down {
            border: none;
            width: 20px;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #333333;
        }

        QLabel {
            color: #333333;
        }

        QStatusBar {
            background-color: #f0f0f0;
            color: #333333;
            border-top: 1px solid #cccccc;
        }
    """


def get_full_modern_style() -> str:
    """Return the complete modern white theme stylesheet."""
    return (
        get_window_tab_style()
        + get_groupbox_button_style()
        + get_slider_checkbox_style()
        + get_combobox_label_statusbar_style()
    )
