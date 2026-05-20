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


class TestFrictionCone:
    """Tests for FrictionCone class."""

    def test_create_valid_cone(self) -> None:
        """Test creating a valid friction cone."""
        cone = FrictionCone(
            mu=0.5,
            normal=np.array([0.0, 0.0, 1.0]),
            num_sides=8,
        )

        assert cone.mu == 0.5, "Assertion failed: cone.mu == 0.5"
        assert_allclose(cone.normal, [0, 0, 1])
        assert cone.num_sides == 8, "Assertion failed: cone.num_sides == 8"

    def test_negative_mu_raises(self) -> None:
        """Test that negative friction coefficient raises."""
        with pytest.raises(ValueError, match="Friction coefficient"):
            FrictionCone(mu=-0.5, normal=np.array([0, 0, 1]))

    def test_zero_normal_raises(self) -> None:
        """Test that zero normal vector raises."""
        with pytest.raises(ValueError, match="Normal vector cannot be zero"):
            FrictionCone(mu=0.5, normal=np.array([0, 0, 0]))

    def test_contains_force_inside_cone(self) -> None:
        """Test contains returns True for force inside cone."""
        cone = FrictionCone(mu=0.5, normal=np.array([0, 0, 1]))

        # Pure normal force
        assert (
            cone.contains(np.array([0, 0, 100])) is True
        ), "Assertion failed: cone.contains(np.array([0, 0, 100])) is True"

        # Force within friction limit
        assert (
            cone.contains(np.array([10, 0, 100])) is True
        )  # 10 < 0.5 * 100, "Assertion failed: cone.contains(np.array([10, 0, 100])) is True  # 10 < 0.5 * 100"

    def test_contains_force_outside_cone(self) -> None:
        """Test contains returns False for force outside cone."""
        cone = FrictionCone(mu=0.5, normal=np.array([0, 0, 1]))

        # Tangential force exceeds limit
        assert (
            cone.contains(np.array([60, 0, 100])) is False
        )  # 60 > 0.5 * 100, "Assertion failed: cone.contains(np.array([60, 0, 100])) is False  # 60 > 0.5 * 100"

        # Pulling force
        assert (
            cone.contains(np.array([0, 0, -100])) is False
        ), "Assertion failed: cone.contains(np.array([0, 0, -100])) is False"

    def test_get_generators_shape(self) -> None:
        """Test get_generators returns correct shape."""
        cone = FrictionCone(mu=0.5, normal=np.array([0, 0, 1]), num_sides=8)
        generators = cone.get_generators()

        assert generators.shape == (
            3,
            8,
        ), "Assertion failed: generators.shape == (3, 8)"

    def test_generators_on_cone_surface(self) -> None:
        """Test that generators lie on friction cone surface."""
        cone = FrictionCone(mu=0.5, normal=np.array([0, 0, 1]), num_sides=8)
        generators = cone.get_generators()

        for i in range(generators.shape[1]):
            g = generators[:, i]
            g_norm = g / np.linalg.norm(g)

            # Generator should be on cone surface
            # Check: tangential / normal ratio ≈ mu
            f_n = np.dot(g_norm, cone.normal)
            f_t = g_norm - f_n * cone.normal
            ratio = np.linalg.norm(f_t) / f_n if f_n > 0 else float("inf")

            assert_allclose(ratio, cone.mu, atol=1e-10)
