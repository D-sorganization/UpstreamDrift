"""
Plugin architecture for model generation integration.

Plugins provide integration with external systems:
- Golf Modeling Suite (existing URDF tools)
- ROS (future)
- Physics engines (Drake, MuJoCo, Pinocchio)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelGenerationPlugin(ABC):
    """Base class for model generation plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        ...

    @abstractmethod
    def initialize(self, context: dict[str, Any]) -> None:
        """Initialize the plugin with context."""
        ...

    def shutdown(self) -> None:
        """Clean up plugin resources."""


__all__ = ["ModelGenerationPlugin"]
