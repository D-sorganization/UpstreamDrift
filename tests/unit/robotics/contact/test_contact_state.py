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


class TestContactState:
    """Tests for ContactState dataclass."""

    def test_create_valid_contact(self) -> None:
        """Test creating a valid contact state."""
        contact = ContactState(
            contact_id=0,
            body_a="foot",
            body_b="ground",
            position=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            penetration=0.001,
            normal_force=100.0,
            friction_coefficient=0.5,
        )

        assert contact.contact_id == 0, "Assertion failed: contact.contact_id == 0"
        assert contact.body_a == "foot", "Assertion failed: contact.body_a == foot"
        assert contact.body_b == "ground", "Assertion failed: contact.body_b == ground"
        assert_allclose(contact.position, [0, 0, 0])
        assert_allclose(contact.normal, [0, 0, 1])
        assert (
            contact.penetration == 0.001
        ), "Assertion failed: contact.penetration == 0.001"
        assert (
            contact.normal_force == 100.0
        ), "Assertion failed: contact.normal_force == 100.0"
        assert (
            contact.friction_coefficient == 0.5
        ), "Assertion failed: contact.friction_coefficient == 0.5"
        assert contact.is_active is True, "Assertion failed: contact.is_active is True"

    def test_normal_is_normalized(self) -> None:
        """Test that normal vector is automatically normalized."""
        contact = ContactState(
            contact_id=0,
            body_a="a",
            body_b="b",
            position=np.zeros(3),
            normal=np.array([0.0, 0.0, 2.0]),  # Not unit length
        )

        assert_allclose(np.linalg.norm(contact.normal), 1.0)
        assert_allclose(contact.normal, [0, 0, 1])

    def test_invalid_position_shape_raises(self) -> None:
        """Test that invalid position shape raises ValueError."""
        with pytest.raises(ValueError, match="position must be"):
            ContactState(
                contact_id=0,
                body_a="a",
                body_b="b",
                position=np.array([0.0, 0.0]),  # Wrong shape
                normal=np.array([0.0, 0.0, 1.0]),
            )

    def test_negative_penetration_raises(self) -> None:
        """Test that negative penetration raises ValueError."""
        with pytest.raises(ValueError, match="penetration must be >= 0"):
            ContactState(
                contact_id=0,
                body_a="a",
                body_b="b",
                position=np.zeros(3),
                normal=np.array([0.0, 0.0, 1.0]),
                penetration=-0.001,
            )

    def test_negative_normal_force_raises(self) -> None:
        """Test that negative normal force raises ValueError."""
        with pytest.raises(ValueError, match="normal_force must be >= 0"):
            ContactState(
                contact_id=0,
                body_a="a",
                body_b="b",
                position=np.zeros(3),
                normal=np.array([0.0, 0.0, 1.0]),
                normal_force=-10.0,
            )

    def test_get_wrench(self) -> None:
        """Test get_wrench method."""
        friction = np.array([10.0, 5.0, 0.0])
        contact = ContactState(
            contact_id=0,
            body_a="a",
            body_b="b",
            position=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            normal_force=100.0,
            friction_force=friction,
        )

        wrench = contact.get_wrench()
        expected_force = np.array([10.0, 5.0, 100.0])
        assert_allclose(wrench[:3], expected_force)
        assert_allclose(wrench[3:], [0, 0, 0])  # No torque at contact point

    def test_is_sliding(self) -> None:
        """Test is_sliding method."""
        # Contact not at friction limit
        contact_not_sliding = ContactState(
            contact_id=0,
            body_a="a",
            body_b="b",
            position=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            normal_force=100.0,
            friction_force=np.array([10.0, 0.0, 0.0]),
            friction_coefficient=0.5,  # Limit is 50 N
        )
        assert (
            contact_not_sliding.is_sliding() is False
        ), "Assertion failed: contact_not_sliding.is_sliding() is False"

        # Contact at friction limit
        contact_sliding = ContactState(
            contact_id=0,
            body_a="a",
            body_b="b",
            position=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            normal_force=100.0,
            friction_force=np.array([50.0, 0.0, 0.0]),
            friction_coefficient=0.5,
        )
        assert (
            contact_sliding.is_sliding() is True
        ), "Assertion failed: contact_sliding.is_sliding() is True"

    def test_with_force_creates_new_contact(self) -> None:
        """Test with_force creates new ContactState."""
        original = ContactState(
            contact_id=0,
            body_a="a",
            body_b="b",
            position=np.array([1.0, 2.0, 3.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            normal_force=100.0,
        )

        new_contact = original.with_force(
            normal_force=200.0,
            friction_force=np.array([10.0, 0.0, 0.0]),
        )

        # Original unchanged
        assert (
            original.normal_force == 100.0
        ), "Assertion failed: original.normal_force == 100.0"
        assert_allclose(original.friction_force, [0, 0, 0])

        # New contact has updated forces
        assert (
            new_contact.normal_force == 200.0
        ), "Assertion failed: new_contact.normal_force == 200.0"
        assert_allclose(new_contact.friction_force, [10, 0, 0])

        # Other fields preserved
        assert (
            new_contact.contact_id == 0
        ), "Assertion failed: new_contact.contact_id == 0"
        assert_allclose(new_contact.position, [1, 2, 3])
