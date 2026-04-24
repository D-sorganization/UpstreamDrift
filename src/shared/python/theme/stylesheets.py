# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Qt stylesheet generation for the fleet-wide theme system.

This module generates complete QSS stylesheets for PyQt6 applications
based on theme color definitions.

Each ``_qss_*`` helper returns the QSS fragment for a widget category,
keeping each section independently readable and modifiable.
"""

from __future__ import annotations

# ── Section generators ───────────────────────────────────────────


def _qss_base(t: dict[str, str]) -> str:
    """Main window / widget defaults."""
    return f"""
        QMainWindow, QWidget {{
            background-color: {t["bg"]};
            color: {t["text"]};
        }}
    """


def _qss_group_box(t: dict[str, str]) -> str:
    return f"""
        QGroupBox {{
            font-weight: bold;
            border: 2px solid {t["border"]};
            border-radius: 5px;
            margin: 5px;
            padding-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }}
    """


def _qss_scroll_area(t: dict[str, str]) -> str:
    return f"""
        QScrollArea {{
            border: none;
            background-color: {t["bg"]};
        }}
    """


def _qss_inputs(t: dict[str, str]) -> str:
    return f"""
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            padding: 5px 8px;
            font-size: 12px;
            border: 1px solid {t["border"]};
            border-radius: 3px;
            background-color: {t["input_bg"]};
            color: {t["text"]};
        }}
        QLineEdit:focus, QComboBox:focus,
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {t["focus"]};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
            background-color: {t["input_bg"]};
        }}
        QComboBox::down-arrow {{
            width: 10px;
            height: 10px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {t["input_bg"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            selection-background-color: {t["accent"]};
            selection-color: white;
        }}
    """


def _qss_labels(t: dict[str, str]) -> str:
    return f"""
        QLabel {{
            color: {t["text_secondary"]};
            font-size: 11px;
            background: transparent;
        }}
    """


def _qss_buttons(t: dict[str, str]) -> str:
    return f"""
        QPushButton {{
            padding: 6px 14px;
            font-size: 11px;
            border: 1px solid {t["border"]};
            border-radius: 4px;
            background-color: {t["group_bg"]};
            color: {t["text"]};
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {t["accent"]};
            color: white;
            border-color: {t["accent"]};
        }}
        QPushButton:pressed {{
            background-color: {t["button_hover"]};
            border-color: {t["button_hover"]};
        }}
        QPushButton:disabled {{
            background-color: {t["border"]};
            color: {t["label"]};
            border-color: {t["border"]};
        }}
        QPushButton:checked {{
            background-color: {t["accent"]};
            color: white;
            border-color: {t["accent"]};
        }}
    """


def _qss_menus(t: dict[str, str]) -> str:
    return f"""
        QMenuBar {{
            background-color: {t["bg"]};
            border-bottom: 1px solid {t["border"]};
            padding: 4px;
        }}
        QMenuBar::item {{
            padding: 6px 12px;
            background: transparent;
            color: {t["text"]};
        }}
        QMenuBar::item:selected {{
            background: rgba(128, 189, 255, 0.2);
            color: {t["text"]};
            border-radius: 3px;
        }}
        QMenu {{
            background-color: {t["input_bg"]};
            border: 1px solid {t["border"]};
            padding: 4px;
        }}
        QMenu::item {{
            padding: 8px 24px 8px 12px;
            color: {t["text"]};
            border-radius: 2px;
        }}
        QMenu::item:selected {{
            background-color: rgba(128, 189, 255, 0.3);
            color: {t["text"]};
        }}
        QMenu::separator {{
            height: 1px;
            background: {t["border"]};
            margin: 4px 8px;
        }}
    """


def _qss_tabs(t: dict[str, str]) -> str:
    return f"""
        QTabWidget::pane {{
            border: 1px solid {t["border"]};
            background-color: {t["bg"]};
            border-radius: 2px;
        }}
        QTabBar::tab {{
            background-color: {t["group_bg"]};
            color: {t["text_secondary"]};
            border: 1px solid {t["border"]};
            border-bottom: none;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 2px;
            border-top-right-radius: 2px;
            font-size: 11px;
            font-weight: 500;
        }}
        QTabBar::tab:selected {{
            background-color: {t["bg"]};
            color: {t["accent"]};
            border-bottom: 2px solid {t["accent"]};
            font-weight: 600;
        }}
        QTabBar::tab:hover {{
            background-color: {t["title_bg"]};
            color: {t["text"]};
        }}
    """


def _qss_tables_and_trees(t: dict[str, str]) -> str:
    return f"""
        QTableWidget, QTableView {{
            background-color: {t["input_bg"]};
            alternate-background-color: {t["table_alt"]};
            gridline-color: {t["border"]};
            border: 1px solid {t["border"]};
            selection-background-color: {t["accent"]};
            selection-color: white;
        }}
        QHeaderView::section {{
            background-color: {t["table_header"]};
            color: {t["text_secondary"]};
            padding: 6px;
            border: 1px solid {t["border"]};
            font-weight: 600;
            font-size: 11px;
        }}
        QTreeWidget, QTreeView {{
            background-color: {t["input_bg"]};
            alternate-background-color: {t["table_alt"]};
            border: 1px solid {t["border"]};
            selection-background-color: {t["accent"]};
            selection-color: white;
        }}
        QTreeWidget::item, QTreeView::item {{
            padding: 4px;
        }}
        QTreeWidget::item:selected, QTreeView::item:selected {{
            background-color: {t["accent"]};
            color: white;
        }}
        QListWidget, QListView {{
            background-color: {t["input_bg"]};
            alternate-background-color: {t["table_alt"]};
            border: 1px solid {t["border"]};
            selection-background-color: {t["accent"]};
            selection-color: white;
        }}
    """


def _qss_text_edits(t: dict[str, str]) -> str:
    return f"""
        QTextEdit, QPlainTextEdit {{
            background-color: {t["input_bg"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 3px;
            padding: 6px;
            font-size: 11px;
        }}
    """


def _qss_progress_status(t: dict[str, str]) -> str:
    return f"""
        QProgressBar {{
            border: 1px solid {t["border"]};
            border-radius: 3px;
            text-align: center;
            background-color: {t["group_bg"]};
            color: {t["text"]};
        }}
        QProgressBar::chunk {{
            background-color: {t["accent"]};
            border-radius: 2px;
        }}
        QStatusBar {{
            background-color: {t["bg"]};
            color: {t["text_secondary"]};
            border-top: 1px solid {t["border"]};
        }}
    """


def _qss_sliders_and_checks(t: dict[str, str]) -> str:
    return f"""
        QSlider::groove:horizontal {{
            border: 1px solid {t["border"]};
            height: 4px;
            background: {t["group_bg"]};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {t["accent"]};
            border: 1px solid {t["accent"]};
            width: 14px;
            margin: -6px 0;
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {t["button_hover"]};
        }}
        QSlider::groove:vertical {{
            border: 1px solid {t["border"]};
            width: 4px;
            background: {t["group_bg"]};
            border-radius: 2px;
        }}
        QSlider::handle:vertical {{
            background: {t["accent"]};
            border: 1px solid {t["accent"]};
            height: 14px;
            margin: 0 -6px;
            border-radius: 7px;
        }}
        QCheckBox, QRadioButton {{
            color: {t["text_secondary"]};
            spacing: 8px;
        }}
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {t["border"]};
            background-color: {t["input_bg"]};
        }}
        QCheckBox::indicator {{
            border-radius: 3px;
        }}
        QRadioButton::indicator {{
            border-radius: 8px;
        }}
        QCheckBox::indicator:checked,
        QRadioButton::indicator:checked {{
            background-color: {t["accent"]};
            border-color: {t["accent"]};
        }}
    """


def _qss_tooltips_toolbar(t: dict[str, str]) -> str:
    return f"""
        QToolTip {{
            background-color: {t["title_bg"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            padding: 4px;
            border-radius: 3px;
        }}
        QToolBar {{
            background-color: {t["bg"]};
            border-bottom: 1px solid {t["border"]};
            spacing: 4px;
            padding: 4px;
        }}
        QToolBar::separator {{
            background-color: {t["border"]};
            width: 1px;
            margin: 4px 2px;
        }}
        QToolButton {{
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 4px;
        }}
        QToolButton:hover {{
            background-color: {t["group_bg"]};
            border-color: {t["border"]};
        }}
        QToolButton:pressed {{
            background-color: {t["accent"]};
        }}
    """


def _qss_dock_scrollbar(t: dict[str, str]) -> str:
    return f"""
        QDockWidget {{
            color: {t["text"]};
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }}
        QDockWidget::title {{
            background-color: {t["title_bg"]};
            border: 1px solid {t["title_border"]};
            padding: 6px;
            text-align: left;
        }}
        QDockWidget::close-button,
        QDockWidget::float-button {{
            border: none;
            background: transparent;
            padding: 2px;
        }}
        QScrollBar:vertical {{
            background-color: {t["bg"]};
            width: 12px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background-color: {t["border"]};
            border-radius: 6px;
            min-height: 20px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {t["accent"]};
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar:horizontal {{
            background-color: {t["bg"]};
            height: 12px;
            margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {t["border"]};
            border-radius: 6px;
            min-width: 20px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: {t["accent"]};
        }}
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
    """


def _qss_containers(t: dict[str, str]) -> str:
    """Splitter, Dialog, MessageBox, FileDialog, Calendar, Frame."""
    return f"""
        QSplitter::handle {{
            background-color: {t["border"]};
        }}
        QSplitter::handle:horizontal {{
            width: 2px;
        }}
        QSplitter::handle:vertical {{
            height: 2px;
        }}
        QSplitter::handle:hover {{
            background-color: {t["accent"]};
        }}
        QDialog {{
            background-color: {t["bg"]};
            color: {t["text"]};
        }}
        QDialogButtonBox {{
            background-color: transparent;
        }}
        QMessageBox {{
            background-color: {t["bg"]};
        }}
        QMessageBox QLabel {{
            color: {t["text"]};
        }}
        QFileDialog {{
            background-color: {t["bg"]};
        }}
        QCalendarWidget {{
            background-color: {t["bg"]};
        }}
        QCalendarWidget QToolButton {{
            color: {t["text"]};
            background-color: {t["group_bg"]};
        }}
        QCalendarWidget QMenu {{
            background-color: {t["input_bg"]};
        }}
        QCalendarWidget QSpinBox {{
            background-color: {t["input_bg"]};
        }}
        QFrame {{
            border: none;
        }}
        QFrame[frameShape="4"], QFrame[frameShape="5"] {{
            border: 1px solid {t["border"]};
        }}
    """


def _qss_tool_cards(t: dict[str, str]) -> str:
    """Launcher ToolCard widgets."""
    return f"""
        ToolCard {{
            background-color: {t["input_bg"]};
            border: 1px solid {t["border"]};
            border-radius: 8px;
        }}
        ToolCard:hover {{
            border: 1px solid {t["accent"]};
            background-color: {t["group_bg"]};
        }}
        #toolCardTitle {{
            font-size: 14px;
            font-weight: bold;
            color: {t["text"]};
            background: transparent;
        }}
        #toolCardDescription {{
            color: {t["text_secondary"]};
            font-size: 12px;
            background: transparent;
        }}
        #toolCardPath {{
            color: {t["label"]};
            font-family: monospace;
            font-size: 10px;
            background: transparent;
        }}
        #launchButton {{
            background-color: {t["accent"]};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px;
            font-weight: bold;
        }}
        #launchButton:hover {{
            background-color: {t["button_hover"]};
        }}
        #launchButton:pressed {{
            background-color: {t["focus"]};
        }}
        #helpButton {{
            background-color: {t["group_bg"]};
            color: {t["accent"]};
            border: 1px solid {t["border"]};
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
        }}
        #helpButton:hover {{
            background-color: {t["title_bg"]};
            border-color: {t["accent"]};
        }}
        #helpButton:pressed {{
            background-color: {t["bg"]};
        }}
    """


# ── Public API ───────────────────────────────────────────────────

# Ordered list of section generators — append here when adding
# new widget categories.
_STYLESHEET_SECTIONS = [
    _qss_base,
    _qss_group_box,
    _qss_scroll_area,
    _qss_inputs,
    _qss_labels,
    _qss_buttons,
    _qss_menus,
    _qss_tabs,
    _qss_tables_and_trees,
    _qss_text_edits,
    _qss_progress_status,
    _qss_sliders_and_checks,
    _qss_tooltips_toolbar,
    _qss_dock_scrollbar,
    _qss_containers,
    _qss_tool_cards,
]


def generate_stylesheet(theme: dict[str, str]) -> str:
    """Generate a complete QSS stylesheet for a theme.

    Args:
        theme: Theme dictionary with color values

    Returns:
        Complete QSS stylesheet string
    """
    return "\n".join(section(theme) for section in _STYLESHEET_SECTIONS)


def generate_minimal_stylesheet(theme: dict[str, str]) -> str:
    """Generate a minimal stylesheet for embedding in parent applications.

    This is useful for applications that want to apply theme colors
    without overriding all widget styles.

    Args:
        theme: Theme dictionary with color values

    Returns:
        Minimal QSS stylesheet string
    """
    return f"""
        QWidget {{
            background-color: {theme["bg"]};
            color: {theme["text"]};
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {theme["input_bg"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
        }}
        QPushButton {{
            background-color: {theme["group_bg"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
        }}
        QPushButton:hover {{
            background-color: {theme["accent"]};
            color: white;
        }}
    """


__all__ = [
    "generate_minimal_stylesheet",
    "generate_stylesheet",
]
