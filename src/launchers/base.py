#!/usr/bin/env python3
"""
Base Launcher Module

Provides abstract base classes and common utilities for all launcher UIs
in the Golf Modeling Suite. This eliminates DRY violations across
multiple launcher implementations.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtGui import QFont, QScreen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass

from src.shared.python.theme.style_constants import Styles

logger = logging.getLogger(__name__)

# Repository root - single source of truth
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()


class LaunchItem:
    """Represents a launchable item (model, tool, or application)."""

    def __init__(
        self,
        name: str,
        description: str,
        path: str | None = None,
        item_type: str = "tool",
        icon: str | None = None,
        action: callable | None = None,
    ) -> None:
        """Initialize a launch item.

        Args:
            name: Display name of the item
            description: Short description
            path: Relative path to file (from REPO_ROOT)
            item_type: Type category ("model", "tool", "app")
            icon: Optional icon path
            action: Optional custom action callable (overrides path launching)
        """
        self.name = name
        self.description = description
        self.path = path
        self.item_type = item_type
        self.icon = icon
        self.action = action

    def get_full_path(self) -> Path | None:
        """Get the full resolved path."""
        if self.path:
            return REPO_ROOT / self.path
        return None


class BaseLauncher(QMainWindow):
    """Base class for all launcher windows.

    Provides common functionality:
    - Window initialization and centering
    - Consistent styling/theming
    - File launching utilities
    - Grid-based card layout
    - Error handling dialogs

    Note: Cannot use ABC mixin due to PyQt6 metaclass conflict
    (QMainWindow's sip.wrappertype is incompatible with ABCMeta).
    Subclasses must override get_items() -- failure to do so raises
    NotImplementedError at runtime.
    """

    # Default window properties - override in subclasses
    WINDOW_TITLE: str = "Launcher"
    WINDOW_WIDTH: int = 800
    WINDOW_HEIGHT: int = 600
    GRID_COLUMNS: int = 2

    # Style constants
    CARD_STYLE = """
        QFrame {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
        }
        QFrame:hover {
            background-color: #e9ecef;
            border-color: #adb5bd;
        }
    """

    LAUNCH_BUTTON_STYLE = """
        QPushButton {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #0056b3; }
        QPushButton:pressed { background-color: #004085; }
        QPushButton:disabled { background-color: #6c757d; }
    """

    HEADER_FONT = QFont("Segoe UI", 24, QFont.Weight.Bold)
    CARD_TITLE_FONT = QFont("Segoe UI", 12, QFont.Weight.Bold)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the base launcher."""
        super().__init__(parent)
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self._items: list[LaunchItem] = []

    def center_window(self) -> None:
        """Center the window on the primary screen."""
        screen: QScreen | None = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            window_geometry = self.frameGeometry()
            center_point = screen_geometry.center()
            window_geometry.moveCenter(center_point)
            self.move(window_geometry.topLeft())

    def show_error(self, title: str, message: str) -> None:
        """Show an error dialog."""
        QMessageBox.critical(self, title, message)

    def show_warning(self, title: str, message: str) -> None:
        """Show a warning dialog."""
        QMessageBox.warning(self, title, message)

    def show_info(self, title: str, message: str) -> None:
        """Show an info dialog."""
        QMessageBox.information(self, title, message)

    def launch_file(self, file_path: str | Path) -> bool:
        """Launch a file using the system default application.

        Args:
            file_path: Absolute or relative path to the file

        Returns:
            True if launch was successful, False otherwise
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Handle relative paths
        if not file_path.is_absolute():
            file_path = REPO_ROOT / file_path

        if not file_path.exists():
            self.show_error("File Not Found", f"File not found:\n{file_path}")
            return False

        try:
            if sys.platform == "win32":
                os.startfile(file_path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(file_path)], check=True)
            else:
                subprocess.run(["xdg-open", str(file_path)], check=True)
            logger.info(f"Launched file: {file_path}")
            return True
        except (OSError, ValueError) as e:
            self.show_error("Launch Failed", f"Could not launch file:\n{e}")
            logger.exception(f"Failed to launch file: {file_path}")
            return False

    def create_card_widget(self, item: LaunchItem) -> QFrame:
        """Create a styled card widget for a launch item.

        Args:
            item: The LaunchItem to create a card for

        Returns:
            QFrame containing the card UI
        """
        card = QFrame()
        card.setStyleSheet(self.CARD_STYLE)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(8)

        # Title
        title_label = QLabel(item.name)
        title_label.setFont(self.CARD_TITLE_FONT)
        title_label.setStyleSheet(Styles.LABEL_TITLE_TRANSPARENT)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(item.description)
        desc_label.setStyleSheet(
            "color: #666; font-size: 11px; border: none; background: transparent;"
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Spacer
        layout.addStretch()

        # Launch button
        launch_btn = QPushButton("Launch")
        launch_btn.setStyleSheet(self.LAUNCH_BUTTON_STYLE)
        launch_btn.clicked.connect(lambda: self._on_item_launch(item))
        layout.addWidget(launch_btn)

        return card

    def _on_item_launch(self, item: LaunchItem) -> None:
        """Handle launch button click for an item."""
        if item.action:
            # Custom action
            try:
                item.action()
            except (RuntimeError, ValueError, OSError) as e:
                self.show_error("Action Failed", f"Failed to execute action:\n{e}")
        elif item.path:
            # File launch
            self.launch_file(item.path)
        else:
            self.show_warning("No Action", f"No action defined for {item.name}")

    def build_grid_layout(
        self, items: list[LaunchItem], columns: int | None = None
    ) -> QGridLayout:
        """Build a grid layout of card widgets.

        Args:
            items: List of LaunchItems to display
            columns: Number of columns (defaults to self.GRID_COLUMNS)

        Returns:
            QGridLayout containing all cards
        """
        if columns is None:
            columns = self.GRID_COLUMNS

        grid = QGridLayout()
        grid.setSpacing(15)

        for idx, item in enumerate(items):
            card = self.create_card_widget(item)
            row = idx // columns
            col = idx % columns
            grid.addWidget(card, row, col)

        return grid

    def create_header(self, title: str, subtitle: str | None = None) -> QVBoxLayout:
        """Create a standard header section.

        Args:
            title: Main title text
            subtitle: Optional subtitle/description

        Returns:
            QVBoxLayout containing header widgets
        """
        layout = QVBoxLayout()
        layout.setSpacing(5)

        title_label = QLabel(title)
        title_label.setFont(self.HEADER_FONT)
        layout.addWidget(title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setStyleSheet(Styles.TEXT_SUBTITLE_LAUNCHER)
            layout.addWidget(subtitle_label)

        return layout

    def create_separator(self) -> QFrame:
        """Create a horizontal separator line."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    @abstractmethod
    def get_items(self) -> list[LaunchItem]:
        """Get the list of items to display.

        Subclasses must override this to provide their launch items.

        Note:
            ABC cannot be used as a mixin here due to PyQt6 metaclass conflict
            (QMainWindow's sip.wrappertype is incompatible with ABCMeta).
            The @abstractmethod decorator serves as documentation; enforcement
            is at runtime via the method body.

        Returns:
            List of LaunchItem objects
        """
        ...

    def init_ui(self) -> None:
        """Initialize the standard UI layout.

        Can be overridden for custom layouts, but provides a sensible default.
        """
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header = self.create_header(self.WINDOW_TITLE, "Select an item to launch.")
        main_layout.addLayout(header)

        # Separator
        main_layout.addWidget(self.create_separator())

        # Items grid
        self._items = self.get_items()
        grid = self.build_grid_layout(self._items)
        main_layout.addLayout(grid)

        # Stretch at bottom
        main_layout.addStretch()


def run_launcher(launcher_class: type[BaseLauncher], **kwargs: Any) -> int:
    """Standard launcher entry point.

    Args:
        launcher_class: The launcher class to instantiate
        **kwargs: Additional arguments to pass to the launcher constructor

    Returns:
        Exit code from QApplication
    """
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = launcher_class(**kwargs)
    window.init_ui()
    window.center_window()
    window.show()

    return app.exec()
