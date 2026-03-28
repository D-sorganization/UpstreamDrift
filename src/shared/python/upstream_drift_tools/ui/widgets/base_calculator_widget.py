"""Base class for all calculator widgets and windows in the fleet."""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QWidget

from ..mixins.base_calculator_mixin import BaseCalculatorMixin

logger = logging.getLogger(__name__)


class BaseCalculatorWidget(QWidget, BaseCalculatorMixin):
    """Base QWidget for calculator modules that can be embedded."""

    def __init__(
        self,
        calculator_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        QWidget.__init__(self, parent)
        BaseCalculatorMixin.__init__(self, calculator_name)

    def show_error(self, title: str, message: str) -> None:
        """Display an error message box."""
        QMessageBox.critical(self, title, message)

    def show_info(self, title: str, message: str) -> None:
        """Display an information message box."""
        QMessageBox.information(self, title, message)


class BaseCalculatorWindow(QMainWindow, BaseCalculatorMixin):
    """Base QMainWindow for standalone calculator applications."""

    def __init__(
        self,
        calculator_name: str,
        window_title: str | None = None,
        min_size: tuple[int, int] = (1000, 700),
        parent: QWidget | None = None,
    ) -> None:
        if not (calculator_name is not None):
            raise ValueError("calculator_name must be provided")
        QMainWindow.__init__(self, parent)
        BaseCalculatorMixin.__init__(self, calculator_name)

        self.setWindowTitle(window_title or calculator_name)
        self.setMinimumSize(*min_size)

        # Main layout will be set by subclasses in their _setup_ui method
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

    def show_error(self, title: str, message: str) -> None:
        """Display an error message box."""
        QMessageBox.critical(self, title, message)

    def show_info(self, title: str, message: str) -> None:
        """Display an information message box."""
        QMessageBox.information(self, title, message)

    def closeEvent(self, event: Any) -> None:
        """Handle window close event with state saving."""
        # Use handle_close_event from CalculatorStateMixin
        self.handle_close_event(event)
