"""Visualization and overlay mixin for Pinocchio GUI."""

from __future__ import annotations

from ..pinocchio_visualization_mixin import (
    COM_COLOR,
    COM_SPHERE_RADIUS,
    PinocchioVisualizationMixin,
)

__all__ = [
    "COM_COLOR",
    "COM_SPHERE_RADIUS",
    "VisualizationMixin",
]


class VisualizationMixin(PinocchioVisualizationMixin):
    """Visualization and overlay mixin subclassing shared PinocchioVisualizationMixin."""
