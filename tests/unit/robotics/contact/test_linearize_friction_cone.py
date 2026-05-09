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


class TestLinearizeFrictionCone:
    """Tests for friction cone linearization."""

    def test_linearization_dimensions(self) -> None:
        """Test linearization returns correct dimensions."""
        A, b = linearize_friction_cone(
            mu=0.5,
            normal=np.array([0, 0, 1]),
            num_faces=8,
        )

        assert A.shape == (8, 3), "Assertion failed: A.shape == (8, 3)"
        assert b.shape == (8,), "Assertion failed: b.shape == (8,)"

    def test_linearization_preserves_cone(self) -> None:
        """Test that linearization approximates the cone."""
        mu = 0.5
        normal = np.array([0.0, 0.0, 1.0])
        A, b = linearize_friction_cone(mu, normal, num_faces=16)

        # Test several points that should be inside
        inside_forces = [
            np.array([0, 0, 100]),  # Pure normal
            np.array([10, 0, 100]),  # Small tangential
            np.array([0, 10, 100]),  # Small tangential other direction
        ]

        for f in inside_forces:
            # Should satisfy A @ f <= b (approximately, due to linearization)
            violations = A @ f - b
            assert np.all(violations <= 1e-6), (
                f"Force {f} should be inside linearized cone"
            )

    def test_compute_friction_cone_constraint(self) -> None:
        """Test compute_friction_cone_constraint returns complete info."""
        result = compute_friction_cone_constraint(
            contact_normal=np.array([0, 0, 1]),
            contact_position=np.array([0, 0, 0]),
            friction_coeff=0.5,
            num_faces=8,
        )

        assert "A" in result, "Assertion failed: A in result"
        assert "b" in result, "Assertion failed: b in result"
        assert "normal" in result, "Assertion failed: normal in result"
        assert "generators" in result, "Assertion failed: generators in result"

        # A includes friction + normal force constraint
        assert (
            result["A"].shape[0] == 9
        )  # 8 friction + 1 normal, "Assertion failed: result[A].shape[0] == 9  # 8 friction + 1 normal"
