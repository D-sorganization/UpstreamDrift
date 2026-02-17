"""DbC tests for friction cone and contact dynamics (robotics).

Validates that:
- FrictionCone rejects negative friction, zero normal, too few sides
- FrictionCone.contains() correctly classifies forces
- Generators have correct shape and count
- Linearized friction cone produces valid A,b constraint matrices
- Projection keeps forces inside cone
- compute_friction_cone_constraint returns valid structure
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ["DBC_LEVEL"] = "enforce"


class TestFrictionConePreconditions(unittest.TestCase):
    """FrictionCone __post_init__ validation."""

    def test_negative_mu_raises(self) -> None:
        from src.robotics.contact.friction_cone import FrictionCone

        with self.assertRaises(ValueError):
            FrictionCone(mu=-0.5, normal=np.array([0.0, 0.0, 1.0]))

    def test_zero_normal_raises(self) -> None:
        from src.robotics.contact.friction_cone import FrictionCone

        with self.assertRaises(ValueError):
            FrictionCone(mu=0.5, normal=np.zeros(3))

    def test_too_few_sides_raises(self) -> None:
        from src.robotics.contact.friction_cone import FrictionCone

        with self.assertRaises(ValueError):
            FrictionCone(mu=0.5, normal=np.array([0.0, 0.0, 1.0]), num_sides=2)

    def test_valid_construction(self) -> None:
        from src.robotics.contact.friction_cone import FrictionCone

        cone = FrictionCone(mu=0.6, normal=np.array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(cone.mu, 0.6)

    def test_normal_auto_normalized(self) -> None:
        from src.robotics.contact.friction_cone import FrictionCone

        cone = FrictionCone(mu=0.5, normal=np.array([0.0, 0.0, 5.0]))
        np.testing.assert_array_almost_equal(cone.normal, [0.0, 0.0, 1.0])


class TestFrictionConeContains(unittest.TestCase):
    """FrictionCone.contains() postconditions."""

    def _cone(self, mu: float = 0.5):  # type: ignore[no-untyped-def]
        from src.robotics.contact.friction_cone import FrictionCone

        return FrictionCone(mu=mu, normal=np.array([0.0, 0.0, 1.0]))

    def test_pure_normal_force_inside(self) -> None:
        """Force purely along normal is always inside."""
        cone = self._cone()
        self.assertTrue(cone.contains(np.array([0.0, 0.0, 10.0])))

    def test_zero_force_inside(self) -> None:
        """Zero force is on the boundary, hence inside."""
        cone = self._cone()
        self.assertTrue(cone.contains(np.zeros(3)))

    def test_pulling_force_outside(self) -> None:
        """Force opposite to normal (tensile) is outside."""
        cone = self._cone()
        self.assertFalse(cone.contains(np.array([0.0, 0.0, -10.0])))

    def test_tangential_within_limit_inside(self) -> None:
        """Force with tangential component within mu*f_n is inside."""
        cone = self._cone(mu=0.5)
        # f_n=10, f_t=4 < 0.5*10=5 => inside
        self.assertTrue(cone.contains(np.array([4.0, 0.0, 10.0])))

    def test_tangential_exceeds_limit_outside(self) -> None:
        """Force with tangential component exceeding mu*f_n is outside."""
        cone = self._cone(mu=0.5)
        # f_n=10, f_t=6 > 0.5*10=5 => outside
        self.assertFalse(cone.contains(np.array([6.0, 0.0, 10.0])))

    def test_tangential_at_limit_inside(self) -> None:
        """Force exactly at friction limit is inside (within tolerance)."""
        cone = self._cone(mu=0.5)
        # f_n=10, f_t=5 == 0.5*10=5 => on boundary => inside
        self.assertTrue(cone.contains(np.array([5.0, 0.0, 10.0])))

    def test_zero_mu_only_normal_inside(self) -> None:
        """With mu=0, only forces along the normal are inside."""
        cone = self._cone(mu=0.0)
        self.assertTrue(cone.contains(np.array([0.0, 0.0, 10.0])))
        self.assertFalse(cone.contains(np.array([0.001, 0.0, 10.0])))


class TestFrictionConeGenerators(unittest.TestCase):
    """Generator postconditions."""

    def test_generator_count(self) -> None:
        from src.robotics.contact.friction_cone import FrictionCone

        cone = FrictionCone(mu=0.5, normal=np.array([0.0, 0.0, 1.0]), num_sides=8)
        generators = cone.get_generators()
        self.assertEqual(generators.shape, (3, 8))

    def test_generators_shape_custom_sides(self) -> None:
        from src.robotics.contact.friction_cone import FrictionCone

        cone = FrictionCone(mu=0.5, normal=np.array([0.0, 0.0, 1.0]), num_sides=12)
        generators = cone.get_generators()
        self.assertEqual(generators.shape, (3, 12))

    def test_generators_along_cone_surface(self) -> None:
        """Each generator should lie on the cone surface."""
        from src.robotics.contact.friction_cone import FrictionCone

        cone = FrictionCone(mu=0.5, normal=np.array([0.0, 0.0, 1.0]))
        generators = cone.get_generators()
        for i in range(generators.shape[1]):
            g = generators[:, i]
            f_n = np.dot(g, cone.normal)
            f_t = g - f_n * cone.normal
            f_t_mag = np.linalg.norm(f_t)
            # On the surface: f_t_mag ≈ mu * f_n
            self.assertAlmostEqual(f_t_mag, cone.mu * f_n, places=10)

    def test_generators_finite(self) -> None:
        from src.robotics.contact.friction_cone import FrictionCone

        cone = FrictionCone(mu=0.5, normal=np.array([0.0, 0.0, 1.0]))
        generators = cone.get_generators()
        self.assertTrue(np.all(np.isfinite(generators)))


class TestLinearizeFrictionConePreconditions(unittest.TestCase):
    """linearize_friction_cone preconditions."""

    def test_negative_mu_raises(self) -> None:
        from src.robotics.contact.friction_cone import linearize_friction_cone
        from src.shared.python.core.contracts import ContractViolationError

        with self.assertRaises((ContractViolationError, ValueError)):
            linearize_friction_cone(
                mu=-0.5, normal=np.array([0.0, 0.0, 1.0])
            )


class TestLinearizeFrictionConePostconditions(unittest.TestCase):
    """linearize_friction_cone output structure."""

    def test_output_shapes(self) -> None:
        from src.robotics.contact.friction_cone import linearize_friction_cone

        A, b = linearize_friction_cone(
            0.5, np.array([0.0, 0.0, 1.0]), 8
        )
        self.assertEqual(A.shape, (8, 3))
        self.assertEqual(b.shape, (8,))

    def test_b_is_zero(self) -> None:
        """The b vector should be all zeros for homogeneous friction cone."""
        from src.robotics.contact.friction_cone import linearize_friction_cone

        _, b = linearize_friction_cone(
            0.5, np.array([0.0, 0.0, 1.0]), 8
        )
        np.testing.assert_array_almost_equal(b, 0.0)

    def test_valid_force_satisfies_constraints(self) -> None:
        """A force inside the cone should satisfy A @ f <= b."""
        from src.robotics.contact.friction_cone import linearize_friction_cone

        A, b = linearize_friction_cone(
            0.5, np.array([0.0, 0.0, 1.0]), 16
        )
        # Pure normal force should satisfy
        f = np.array([0.0, 0.0, 10.0])
        result = A @ f
        self.assertTrue(np.all(result <= b + 1e-6),
                        f"Pure normal force violates constraints: {result}")


class TestProjectToFrictionCone(unittest.TestCase):
    """project_to_friction_cone postconditions."""

    def _cone(self, mu: float = 0.5):  # type: ignore[no-untyped-def]
        from src.robotics.contact.friction_cone import FrictionCone

        return FrictionCone(mu=mu, normal=np.array([0.0, 0.0, 1.0]))

    def test_inside_force_unchanged(self) -> None:
        from src.robotics.contact.friction_cone import project_to_friction_cone

        cone = self._cone()
        f = np.array([0.0, 0.0, 10.0])
        projected = project_to_friction_cone(f, cone)
        np.testing.assert_array_almost_equal(projected, f)

    def test_projected_force_inside_cone(self) -> None:
        from src.robotics.contact.friction_cone import project_to_friction_cone

        cone = self._cone()
        f = np.array([10.0, 0.0, 5.0])  # Outside
        projected = project_to_friction_cone(f, cone)
        self.assertTrue(cone.contains(projected, tolerance=1e-5))

    def test_pulling_force_projects_to_zero_or_surface(self) -> None:
        from src.robotics.contact.friction_cone import project_to_friction_cone

        cone = self._cone()
        f = np.array([0.0, 0.0, -10.0])
        projected = project_to_friction_cone(f, cone)
        # Either zero or on cone surface
        self.assertTrue(
            cone.contains(projected, tolerance=1e-5) or
            np.linalg.norm(projected) < 1e-10
        )

    def test_projected_force_finite(self) -> None:
        from src.robotics.contact.friction_cone import project_to_friction_cone

        cone = self._cone()
        for _ in range(20):
            f = np.random.randn(3) * 10
            projected = project_to_friction_cone(f, cone)
            self.assertTrue(np.all(np.isfinite(projected)))


class TestComputeFrictionConeConstraintPreconditions(unittest.TestCase):
    """compute_friction_cone_constraint preconditions."""

    def test_negative_friction_raises(self) -> None:
        from src.robotics.contact.friction_cone import (
            compute_friction_cone_constraint,
        )
        from src.shared.python.core.contracts import ContractViolationError

        with self.assertRaises((ContractViolationError, ValueError)):
            compute_friction_cone_constraint(
                contact_normal=np.array([0.0, 0.0, 1.0]),
                contact_position=np.zeros(3),
                friction_coeff=-0.3,
            )


class TestComputeFrictionConeConstraintPostconditions(unittest.TestCase):
    """compute_friction_cone_constraint output structure."""

    def test_valid_output_structure(self) -> None:
        from src.robotics.contact.friction_cone import (
            compute_friction_cone_constraint,
        )

        result = compute_friction_cone_constraint(
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            0.5,
            8,
        )
        self.assertIn("A", result)
        self.assertIn("b", result)
        self.assertIn("normal", result)
        self.assertIn("generators", result)
        # A should be (num_faces+1, 3) including normal force constraint
        self.assertEqual(result["A"].shape, (9, 3))
        self.assertEqual(result["b"].shape, (9,))
        self.assertEqual(result["generators"].shape, (3, 8))

    def test_output_matrices_finite(self) -> None:
        from src.robotics.contact.friction_cone import (
            compute_friction_cone_constraint,
        )

        result = compute_friction_cone_constraint(
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            0.5,
            8,
        )
        self.assertTrue(np.all(np.isfinite(result["A"])))
        self.assertTrue(np.all(np.isfinite(result["b"])))
        self.assertTrue(np.all(np.isfinite(result["generators"])))


if __name__ == "__main__":
    unittest.main()
