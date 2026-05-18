"""Base class for all calculation engines in the fleet."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseCalculationEngine(ABC):
    """Abstract base class for all calculation engines.

    This class defines the standard interface for generic engineering engines
    that can be shared across multiple UI shell (PyQt, Tkinter, React/API).
    """

    @abstractmethod
    def calculate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform a calculation and return a JSON-serializable dictionary.

        This method must be implemented by subclasses to define specific
        calculation logic.
        """


__all__ = ["BaseCalculationEngine"]
