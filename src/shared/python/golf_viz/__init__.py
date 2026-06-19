"""Shared, headless rendering primitives for golf simulation GUIs.

This package centralises the geometry and colour maths used by the putting
green, driving range, and ball-flight viewers so each GUI stays a thin Qt /
OpenGL adapter (DRY). Everything here is pure numpy with no Qt dependency,
which keeps it exhaustively unit-testable in any headless environment.

Public API:
    Colours
        :data:`ROLL_MODE_RGBA`, :func:`sample_gradient`, :func:`terrain_colors`,
        :func:`speed_colors`, :func:`roll_mode_colors`
    Geometry
        :func:`rect_vertices`, :func:`circle_fan_vertices`, :func:`disc_mesh`,
        :func:`grid_surface_mesh`, :func:`flagstick_lines`
"""

from __future__ import annotations

from src.shared.python.golf_viz._colors import (
    RGBA,
    ROLL_MODE_RGBA,
    roll_mode_colors,
    sample_gradient,
    speed_colors,
    terrain_colors,
)
from src.shared.python.golf_viz._geometry import (
    circle_fan_vertices,
    disc_mesh,
    flagstick_lines,
    grid_surface_mesh,
    rect_vertices,
)

__all__ = [
    "RGBA",
    "ROLL_MODE_RGBA",
    "circle_fan_vertices",
    "disc_mesh",
    "flagstick_lines",
    "grid_surface_mesh",
    "rect_vertices",
    "roll_mode_colors",
    "sample_gradient",
    "speed_colors",
    "terrain_colors",
]
