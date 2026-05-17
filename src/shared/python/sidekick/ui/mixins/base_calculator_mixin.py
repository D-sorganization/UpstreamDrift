"""Base Calculator Widget Mixin
===========================

Common UI logic for all calculator widgets in the fleet.
"""

from __future__ import annotations

import logging
from typing import Any

from .calculator_state_mixin import CalculatorStateMixin

logger = logging.getLogger(__name__)


class BaseCalculatorMixin(CalculatorStateMixin):
    """
    Mixin providing common functionality for calculator widgets.
    Expected to be mixed into a QWidget or QMainWindow.
    """

    def __init__(self, calculator_name: str | None = None) -> None:
        """Initialize the mixin logic."""
        CalculatorStateMixin.__init__(self, calculator_name)
        self.calculator_name = calculator_name or self.__class__.__name__
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize state management attributes if not already present
        if not hasattr(self, "_splitters"):
            self._splitters: list[Any] = []
        if not hasattr(self, "_copyable_widgets"):
            self._copyable_widgets: list[Any] = []
        if not hasattr(self, "_state"):
            self._state: dict[str, Any] = {}

    def log_info(self, message: str) -> None:
        """Log an info message."""
        self._logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self._logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self._logger.error(message)
