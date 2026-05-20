"""Unit tests for contact dynamics module.

Tests cover:
    - ContactState creation and validation
    - FrictionCone operations
    - ContactManager functionality
    - Grasp analysis
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.robotics.contact.friction_cone import (
    FrictionCone,
    compute_friction_cone_constraint,
    linearize_friction_cone,
    project_to_friction_cone,
)
from src.robotics.contact.grasp_analysis import (
    check_force_closure,
    compute_grasp_matrix,
    compute_grasp_quality,
)
from src.robotics.core.types import ContactState


class TestContactManagerIntegration:
    """Integration-level tests for ContactManager."""

    def test_support_polygon_computation(self) -> None:
        """Test support polygon from foot contacts."""
        from src.robotics.contact.contact_manager import (
            _convex_hull_2d,
            _point_in_polygon,
        )

        # Square contact pattern
        points = np.array(
            [
                [0.1, 0.1],
                [0.1, -0.1],
                [-0.1, 0.1],
                [-0.1, -0.1],
            ]
        )

        hull = _convex_hull_2d(points)

        # Should have 4 vertices
        assert len(hull) == 4, "Assertion failed: len(hull) == 4"

        # Center should be inside
        assert (
            _point_in_polygon(np.array([0, 0]), hull) is True
        ), "Assertion failed: _point_in_polygon(np.array([0, 0]), hull) is True"

        # Point outside should return False
        assert (
            _point_in_polygon(np.array([0.5, 0.5]), hull) is False
        ), "Assertion failed: _point_in_polygon(np.array([0.5, 0.5]), hull) is False"

    def test_point_in_triangle(self) -> None:
        """Test point in polygon for triangle."""
        from src.robotics.contact.contact_manager import _point_in_polygon

        triangle = np.array(
            [
                [0, 0],
                [1, 0],
                [0.5, 1],
            ]
        )

        # Inside
        assert (
            _point_in_polygon(np.array([0.5, 0.3]), triangle) is True
        ), "Assertion failed: _point_in_polygon(np.array([0.5, 0.3]), triangle) is True"

        # Outside
        assert (
            _point_in_polygon(np.array([1.5, 0.5]), triangle) is False
        ), "Assertion failed: _point_in_polygon(np.array([1.5, 0.5]), triangle) is False"

        # On edge (may be inside or outside depending on implementation)
        # Just verify it doesn't crash
        _point_in_polygon(np.array([0.5, 0]), triangle)
