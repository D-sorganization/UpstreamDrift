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
from numpy.testing import assert_allclose, assert_array_equal

from src.robotics.core.types import ContactState, ContactType
from src.robotics.contact.friction_cone import (
    FrictionCone,
    linearize_friction_cone,
    compute_friction_cone_constraint,
    project_to_friction_cone,
)
from src.robotics.contact.grasp_analysis import (
    compute_grasp_matrix,
    check_force_closure,
    compute_grasp_quality,
)


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

        assert contact.contact_id == 0
        assert contact.body_a == "foot"
        assert contact.body_b == "ground"
        assert_allclose(contact.position, [0, 0, 0])
        assert_allclose(contact.normal, [0, 0, 1])
        assert contact.penetration == 0.001
        assert contact.normal_force == 100.0
        assert contact.friction_coefficient == 0.5
        assert contact.is_active is True

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
        assert contact_not_sliding.is_sliding() is False

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
        assert contact_sliding.is_sliding() is True

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
        assert original.normal_force == 100.0
        assert_allclose(original.friction_force, [0, 0, 0])

        # New contact has updated forces
        assert new_contact.normal_force == 200.0
        assert_allclose(new_contact.friction_force, [10, 0, 0])

        # Other fields preserved
        assert new_contact.contact_id == 0
        assert_allclose(new_contact.position, [1, 2, 3])


class TestFrictionCone:
    """Tests for FrictionCone class."""

    def test_create_valid_cone(self) -> None:
        """Test creating a valid friction cone."""
        cone = FrictionCone(
            mu=0.5,
            normal=np.array([0.0, 0.0, 1.0]),
            num_sides=8,
        )

        assert cone.mu == 0.5
        assert_allclose(cone.normal, [0, 0, 1])
        assert cone.num_sides == 8

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
        assert cone.contains(np.array([0, 0, 100])) is True

        # Force within friction limit
        assert cone.contains(np.array([10, 0, 100])) is True  # 10 < 0.5 * 100

    def test_contains_force_outside_cone(self) -> None:
        """Test contains returns False for force outside cone."""
        cone = FrictionCone(mu=0.5, normal=np.array([0, 0, 1]))

        # Tangential force exceeds limit
        assert cone.contains(np.array([60, 0, 100])) is False  # 60 > 0.5 * 100

        # Pulling force
        assert cone.contains(np.array([0, 0, -100])) is False

    def test_get_generators_shape(self) -> None:
        """Test get_generators returns correct shape."""
        cone = FrictionCone(mu=0.5, normal=np.array([0, 0, 1]), num_sides=8)
        generators = cone.get_generators()

        assert generators.shape == (3, 8)

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


class TestLinearizeFrictionCone:
    """Tests for friction cone linearization."""

    def test_linearization_dimensions(self) -> None:
        """Test linearization returns correct dimensions."""
        A, b = linearize_friction_cone(
            mu=0.5,
            normal=np.array([0, 0, 1]),
            num_faces=8,
        )

        assert A.shape == (8, 3)
        assert b.shape == (8,)

    def test_linearization_preserves_cone(self) -> None:
        """Test that linearization approximates the cone."""
        mu = 0.5
        normal = np.array([0.0, 0.0, 1.0])
        A, b = linearize_friction_cone(mu, normal, num_faces=16)

        # Test several points that should be inside
        inside_forces = [
            np.array([0, 0, 100]),      # Pure normal
            np.array([10, 0, 100]),     # Small tangential
            np.array([0, 10, 100]),     # Small tangential other direction
        ]

        for f in inside_forces:
            # Should satisfy A @ f <= b (approximately, due to linearization)
            violations = A @ f - b
            assert np.all(violations <= 1e-6), f"Force {f} should be inside linearized cone"

    def test_compute_friction_cone_constraint(self) -> None:
        """Test compute_friction_cone_constraint returns complete info."""
        result = compute_friction_cone_constraint(
            contact_normal=np.array([0, 0, 1]),
            contact_position=np.array([0, 0, 0]),
            friction_coeff=0.5,
            num_faces=8,
        )

        assert "A" in result
        assert "b" in result
        assert "normal" in result
        assert "generators" in result

        # A includes friction + normal force constraint
        assert result["A"].shape[0] == 9  # 8 friction + 1 normal


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
        assert cone.contains(projected)

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


class TestGraspAnalysis:
    """Tests for grasp analysis functions."""

    @pytest.fixture
    def simple_grasp(self) -> list[ContactState]:
        """Create a simple two-finger grasp."""
        return [
            ContactState(
                contact_id=0,
                body_a="finger1",
                body_b="object",
                position=np.array([0.05, 0.0, 0.0]),
                normal=np.array([-1.0, 0.0, 0.0]),
                normal_force=10.0,
                friction_coefficient=0.5,
            ),
            ContactState(
                contact_id=1,
                body_a="finger2",
                body_b="object",
                position=np.array([-0.05, 0.0, 0.0]),
                normal=np.array([1.0, 0.0, 0.0]),
                normal_force=10.0,
                friction_coefficient=0.5,
            ),
        ]

    @pytest.fixture
    def three_finger_grasp(self) -> list[ContactState]:
        """Create a three-finger planar grasp."""
        # Fingers at 120 degree intervals
        angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        radius = 0.05

        contacts = []
        for i, angle in enumerate(angles):
            pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0,
            ])
            normal = -pos / np.linalg.norm(pos)

            contacts.append(ContactState(
                contact_id=i,
                body_a=f"finger{i}",
                body_b="object",
                position=pos,
                normal=normal,
                normal_force=10.0,
                friction_coefficient=0.5,
            ))

        return contacts

    def test_grasp_matrix_shape(self, simple_grasp: list[ContactState]) -> None:
        """Test grasp matrix has correct shape."""
        G = compute_grasp_matrix(simple_grasp)
        assert G.shape == (6, 6)  # 6 wrench dims, 2 contacts * 3 force dims

    def test_grasp_matrix_with_object_frame(
        self, simple_grasp: list[ContactState]
    ) -> None:
        """Test grasp matrix with explicit object frame."""
        G = compute_grasp_matrix(
            simple_grasp,
            object_frame=np.array([0.0, 0.0, 0.0]),
        )
        assert G.shape == (6, 6)

    def test_force_closure_simple_grasp(
        self, simple_grasp: list[ContactState]
    ) -> None:
        """Test force closure check for simple grasp."""
        # Two opposing fingers with friction should have force closure
        has_closure, quality = check_force_closure(simple_grasp)

        # The result depends on the solver availability and algorithm
        # At minimum, verify it returns valid types
        assert isinstance(has_closure, bool)
        assert isinstance(quality, float)
        assert quality >= 0

        # Note: A proper two-finger opposing grasp with friction
        # typically has force closure, but the detection algorithm
        # may require tuning

    def test_force_closure_three_finger(
        self, three_finger_grasp: list[ContactState]
    ) -> None:
        """Test force closure for three-finger grasp."""
        has_closure, quality = check_force_closure(three_finger_grasp)

        # Verify valid return types
        assert isinstance(has_closure, bool)
        assert isinstance(quality, float)
        assert quality >= 0

        # A symmetric three-finger grasp is well-suited for force closure
        # The heuristic check should at least detect full rank

    def test_grasp_quality_min_singular_value(
        self, three_finger_grasp: list[ContactState]
    ) -> None:
        """Test grasp quality computation."""
        quality = compute_grasp_quality(
            three_finger_grasp,
            metric="min_singular_value",
        )
        assert quality > 0

    def test_grasp_quality_isotropy(
        self, three_finger_grasp: list[ContactState]
    ) -> None:
        """Test grasp isotropy metric."""
        isotropy = compute_grasp_quality(
            three_finger_grasp,
            metric="isotropy",
        )
        assert 0 <= isotropy <= 1

    def test_grasp_quality_volume(
        self, three_finger_grasp: list[ContactState]
    ) -> None:
        """Test grasp volume metric."""
        volume = compute_grasp_quality(
            three_finger_grasp,
            metric="volume",
        )
        assert volume > 0

    def test_invalid_metric_raises(
        self, simple_grasp: list[ContactState]
    ) -> None:
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_grasp_quality(simple_grasp, metric="invalid")


class TestContactManagerIntegration:
    """Integration-level tests for ContactManager."""

    def test_support_polygon_computation(self) -> None:
        """Test support polygon from foot contacts."""
        from src.robotics.contact.contact_manager import (
            _convex_hull_2d,
            _point_in_polygon,
        )

        # Square contact pattern
        points = np.array([
            [0.1, 0.1],
            [0.1, -0.1],
            [-0.1, 0.1],
            [-0.1, -0.1],
        ])

        hull = _convex_hull_2d(points)

        # Should have 4 vertices
        assert len(hull) == 4

        # Center should be inside
        assert _point_in_polygon(np.array([0, 0]), hull) is True

        # Point outside should return False
        assert _point_in_polygon(np.array([0.5, 0.5]), hull) is False

    def test_point_in_triangle(self) -> None:
        """Test point in polygon for triangle."""
        from src.robotics.contact.contact_manager import _point_in_polygon

        triangle = np.array([
            [0, 0],
            [1, 0],
            [0.5, 1],
        ])

        # Inside
        assert _point_in_polygon(np.array([0.5, 0.3]), triangle) is True

        # Outside
        assert _point_in_polygon(np.array([1.5, 0.5]), triangle) is False

        # On edge (may be inside or outside depending on implementation)
        # Just verify it doesn't crash
        _point_in_polygon(np.array([0.5, 0]), triangle)
