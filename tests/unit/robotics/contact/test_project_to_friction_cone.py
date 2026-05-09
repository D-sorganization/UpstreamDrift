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


class TestProjectToFrictionCone:
    """Tests for friction cone projection."""

    def test_project_inside_force_unchanged(self) -> None:
        """Test that force inside cone is unchanged."""
        cone = FrictionCone(mu=0.5, normal=np.array([0, 0, 1]))
        force = np.array([10.0, 0.0, 100.0])

        projected = project_to_friction_cone(force, cone)
        assert_allclose(projected, force)

    def test_project_outside_force(self) -> None:
        """Test projection of force outside cone."""
        cone = FrictionCone(mu=0.5, normal=np.array([0, 0, 1]))
        force = np.array([100.0, 0.0, 100.0])  # Tangential exceeds limit

        projected = project_to_friction_cone(force, cone)

        # Projected force should be inside cone
        assert cone.contains(projected), "Assertion failed: cone.contains(projected)"

        # Normal component preserved
        assert_allclose(np.dot(projected, cone.normal), 100.0)

        # Tangential at limit
        f_t = projected - 100.0 * cone.normal
        assert_allclose(np.linalg.norm(f_t), 50.0, atol=1e-10)

    def test_project_pulling_force(self) -> None:
        """Test projection of pulling (negative normal) force."""
        cone = FrictionCone(mu=0.5, normal=np.array([0, 0, 1]))
        force = np.array([0.0, 0.0, -100.0])  # Pulling

        projected = project_to_friction_cone(force, cone)

        # Should project to zero (no tensile contact)
        assert_allclose(projected, [0, 0, 0])
