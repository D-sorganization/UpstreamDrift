"""
TDD tests for capsule inertia parallel axis theorem fix.

Bug: The parallel axis term for hemisphere offset in capsule_inertia()
used ``0.5 * m_sphere * hemisphere_offset**2`` when it should be
``m_sphere * hemisphere_offset**2``.

Two hemispheres, each with mass m_sphere/2, are each offset by
hemisphere_offset from the capsule center.  The total parallel axis
contribution is:

    2 * (m_sphere / 2) * hemisphere_offset**2
    = m_sphere * hemisphere_offset**2

The factor of 0.5 was incorrect.

These tests cover:
- Known-value regression for a concrete capsule (mass=10, r=0.1, l=0.3)
- Consistency between the three capsule_inertia implementations
- Edge cases: zero-length capsule (sphere), very long capsule
- All three axis orientations (x, y, z)
- Physical properties: positive definiteness, triangle inequality
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# Helpers to compute the *correct* capsule inertia from first principles so
# that we have an independent oracle.
# ---------------------------------------------------------------------------


def _correct_capsule_inertia(mass: float, radius: float, length: float) -> tuple[float, float]:
    """Return (i_axial, i_perp) for a capsule aligned along z.

    Uses the *correct* parallel axis theorem (factor 1, not 0.5).
    """
    v_cyl = math.pi * radius**2 * length
    v_sphere = (4.0 / 3.0) * math.pi * radius**3
    v_total = v_cyl + v_sphere

    m_cyl = mass * v_cyl / v_total
    m_sphere = mass * v_sphere / v_total

    i_cyl_axial = 0.5 * m_cyl * radius**2
    i_cyl_perp = (m_cyl / 12.0) * (3.0 * radius**2 + length**2)

    i_sphere_center = (2.0 / 5.0) * m_sphere * radius**2

    hemisphere_offset = length / 2.0 + (3.0 / 8.0) * radius

    # CORRECT: factor is 1.0, not 0.5
    i_sphere_perp = i_sphere_center + m_sphere * hemisphere_offset**2

    i_axial = i_cyl_axial + i_sphere_center
    i_perp = i_cyl_perp + i_sphere_perp

    return i_axial, i_perp


def _buggy_capsule_inertia(mass: float, radius: float, length: float) -> tuple[float, float]:
    """Return the BUGGY (i_axial, i_perp) -- used to show the fix matters."""
    v_cyl = math.pi * radius**2 * length
    v_sphere = (4.0 / 3.0) * math.pi * radius**3
    v_total = v_cyl + v_sphere

    m_cyl = mass * v_cyl / v_total
    m_sphere = mass * v_sphere / v_total

    i_cyl_axial = 0.5 * m_cyl * radius**2
    i_cyl_perp = (m_cyl / 12.0) * (3.0 * radius**2 + length**2)

    i_sphere_center = (2.0 / 5.0) * m_sphere * radius**2

    hemisphere_offset = length / 2.0 + (3.0 / 8.0) * radius

    # BUGGY: factor is 0.5
    i_sphere_perp = i_sphere_center + 0.5 * m_sphere * hemisphere_offset**2

    i_axial = i_cyl_axial + i_sphere_center
    i_perp = i_cyl_perp + i_sphere_perp

    return i_axial, i_perp


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

MASS = 10.0
RADIUS = 0.1
LENGTH = 0.3
ATOL = 1e-12  # Tight tolerance -- all implementations use the same formula


class TestCapsuleInertiaKnownValues:
    """Regression tests against a hand-computed oracle."""

    def test_perpendicular_inertia_matches_oracle(self) -> None:
        """primitives.capsule_inertia perpendicular moment matches oracle."""
        from model_generation.inertia.primitives import capsule_inertia

        result = capsule_inertia(MASS, RADIUS, LENGTH, axis="z")
        _, expected_perp = _correct_capsule_inertia(MASS, RADIUS, LENGTH)

        assert result["ixx"] == pytest.approx(expected_perp, abs=ATOL)
        assert result["iyy"] == pytest.approx(expected_perp, abs=ATOL)

    def test_axial_inertia_matches_oracle(self) -> None:
        """primitives.capsule_inertia axial moment matches oracle."""
        from model_generation.inertia.primitives import capsule_inertia

        result = capsule_inertia(MASS, RADIUS, LENGTH, axis="z")
        expected_axial, _ = _correct_capsule_inertia(MASS, RADIUS, LENGTH)

        assert result["izz"] == pytest.approx(expected_axial, abs=ATOL)

    def test_types_capsule_matches_oracle(self) -> None:
        """Inertia.from_capsule perpendicular moment matches oracle."""
        from model_generation.core.types import Inertia

        result = Inertia.from_capsule(MASS, RADIUS, LENGTH, axis="z")
        _, expected_perp = _correct_capsule_inertia(MASS, RADIUS, LENGTH)

        assert result.ixx == pytest.approx(expected_perp, abs=ATOL)
        assert result.iyy == pytest.approx(expected_perp, abs=ATOL)

    def test_types_axial_matches_oracle(self) -> None:
        """Inertia.from_capsule axial moment matches oracle."""
        from model_generation.core.types import Inertia

        result = Inertia.from_capsule(MASS, RADIUS, LENGTH, axis="z")
        expected_axial, _ = _correct_capsule_inertia(MASS, RADIUS, LENGTH)

        assert result.izz == pytest.approx(expected_axial, abs=ATOL)


class TestCapsuleInertiaFixIsLarger:
    """The corrected perpendicular inertia must be strictly larger than buggy."""

    def test_primitives_perp_greater_than_buggy(self) -> None:
        from model_generation.inertia.primitives import capsule_inertia

        result = capsule_inertia(MASS, RADIUS, LENGTH, axis="z")
        _, buggy_perp = _buggy_capsule_inertia(MASS, RADIUS, LENGTH)
        # Fixed value must exceed buggy value
        assert result["ixx"] > buggy_perp

    def test_types_perp_greater_than_buggy(self) -> None:
        from model_generation.core.types import Inertia

        result = Inertia.from_capsule(MASS, RADIUS, LENGTH, axis="z")
        _, buggy_perp = _buggy_capsule_inertia(MASS, RADIUS, LENGTH)
        # Fixed value must exceed buggy value
        assert result.ixx > buggy_perp


class TestConsistencyBetweenImplementations:
    """All three capsule inertia implementations must agree."""

    def test_primitives_vs_types(self) -> None:
        """primitives.capsule_inertia and Inertia.from_capsule agree."""
        from model_generation.core.types import Inertia
        from model_generation.inertia.primitives import capsule_inertia

        prim = capsule_inertia(MASS, RADIUS, LENGTH, axis="z")
        typed = Inertia.from_capsule(MASS, RADIUS, LENGTH, axis="z")

        assert prim["ixx"] == pytest.approx(typed.ixx, abs=ATOL)
        assert prim["iyy"] == pytest.approx(typed.iyy, abs=ATOL)
        assert prim["izz"] == pytest.approx(typed.izz, abs=ATOL)

    def test_primitives_vs_humanoid_builder(self) -> None:
        """primitives.capsule_inertia and humanoid builder agree."""
        from model_generation.inertia.primitives import capsule_inertia

        try:
            from humanoid_character_builder.mesh.primitive_inertia import (
                PrimitiveInertiaCalculator,
            )
        except ImportError:
            pytest.skip("humanoid_character_builder not importable")

        prim = capsule_inertia(MASS, RADIUS, LENGTH, axis="z")
        hcb = PrimitiveInertiaCalculator.compute_capsule(MASS, RADIUS, LENGTH, axis="z")

        assert prim["ixx"] == pytest.approx(hcb.ixx, abs=ATOL)
        assert prim["iyy"] == pytest.approx(hcb.iyy, abs=ATOL)
        assert prim["izz"] == pytest.approx(hcb.izz, abs=ATOL)

    def test_types_vs_humanoid_builder(self) -> None:
        """Inertia.from_capsule and humanoid builder agree."""
        from model_generation.core.types import Inertia

        try:
            from humanoid_character_builder.mesh.primitive_inertia import (
                PrimitiveInertiaCalculator,
            )
        except ImportError:
            pytest.skip("humanoid_character_builder not importable")

        typed = Inertia.from_capsule(MASS, RADIUS, LENGTH, axis="z")
        hcb = PrimitiveInertiaCalculator.compute_capsule(MASS, RADIUS, LENGTH, axis="z")

        assert typed.ixx == pytest.approx(hcb.ixx, abs=ATOL)
        assert typed.iyy == pytest.approx(hcb.iyy, abs=ATOL)
        assert typed.izz == pytest.approx(hcb.izz, abs=ATOL)


class TestAxisOrientations:
    """Capsule inertia must assign axial/perp correctly for each axis."""

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_primitives_axis(self, axis: str) -> None:
        from model_generation.inertia.primitives import capsule_inertia

        result = capsule_inertia(MASS, RADIUS, LENGTH, axis=axis)
        expected_axial, expected_perp = _correct_capsule_inertia(MASS, RADIUS, LENGTH)

        if axis == "x":
            assert result["ixx"] == pytest.approx(expected_axial, abs=ATOL)
            assert result["iyy"] == pytest.approx(expected_perp, abs=ATOL)
            assert result["izz"] == pytest.approx(expected_perp, abs=ATOL)
        elif axis == "y":
            assert result["ixx"] == pytest.approx(expected_perp, abs=ATOL)
            assert result["iyy"] == pytest.approx(expected_axial, abs=ATOL)
            assert result["izz"] == pytest.approx(expected_perp, abs=ATOL)
        else:  # z
            assert result["ixx"] == pytest.approx(expected_perp, abs=ATOL)
            assert result["iyy"] == pytest.approx(expected_perp, abs=ATOL)
            assert result["izz"] == pytest.approx(expected_axial, abs=ATOL)

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_types_axis(self, axis: str) -> None:
        from model_generation.core.types import Inertia

        result = Inertia.from_capsule(MASS, RADIUS, LENGTH, axis=axis)
        expected_axial, expected_perp = _correct_capsule_inertia(MASS, RADIUS, LENGTH)

        if axis == "x":
            assert result.ixx == pytest.approx(expected_axial, abs=ATOL)
            assert result.iyy == pytest.approx(expected_perp, abs=ATOL)
            assert result.izz == pytest.approx(expected_perp, abs=ATOL)
        elif axis == "y":
            assert result.ixx == pytest.approx(expected_perp, abs=ATOL)
            assert result.iyy == pytest.approx(expected_axial, abs=ATOL)
            assert result.izz == pytest.approx(expected_perp, abs=ATOL)
        else:  # z
            assert result.ixx == pytest.approx(expected_perp, abs=ATOL)
            assert result.iyy == pytest.approx(expected_perp, abs=ATOL)
            assert result.izz == pytest.approx(expected_axial, abs=ATOL)

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_off_diagonal_zero(self, axis: str) -> None:
        """Off-diagonal terms must be zero for a capsule on a principal axis."""
        from model_generation.inertia.primitives import capsule_inertia

        result = capsule_inertia(MASS, RADIUS, LENGTH, axis=axis)
        assert result["ixy"] == pytest.approx(0.0, abs=ATOL)
        assert result["ixz"] == pytest.approx(0.0, abs=ATOL)
        assert result["iyz"] == pytest.approx(0.0, abs=ATOL)


class TestEdgeCases:
    """Edge cases for capsule geometry."""

    def test_zero_length_reduces_to_sphere(self) -> None:
        """A capsule with length=0 should give sphere inertia."""
        from model_generation.inertia.primitives import capsule_inertia, sphere_inertia

        capsule_result = capsule_inertia(MASS, RADIUS, 0.0)
        sphere_result = sphere_inertia(MASS, RADIUS)

        # With length=0 the cylinder volume is zero, so all mass is in the
        # sphere.  The only difference is the hemisphere offset (3r/8),
        # which makes the perpendicular inertia slightly larger than a
        # sphere because the two hemisphere COMs are displaced.
        # Axial inertia should still equal a sphere.
        assert capsule_result["izz"] == pytest.approx(sphere_result["izz"], abs=ATOL)

        # Perpendicular should be >= sphere (due to parallel axis offset)
        assert capsule_result["ixx"] >= sphere_result["ixx"] - ATOL

    def test_very_long_capsule_dominated_by_cylinder(self) -> None:
        """For a very long capsule, cylinder inertia dominates."""
        from model_generation.inertia.primitives import (
            capsule_inertia,
            cylinder_inertia,
        )

        long_length = 100.0
        cap = capsule_inertia(MASS, RADIUS, long_length, axis="z")
        cyl = cylinder_inertia(MASS, RADIUS, long_length, axis="z")

        # Perpendicular should be close (within a few %) to pure cylinder
        rel_diff = abs(cap["ixx"] - cyl["ixx"]) / cyl["ixx"]
        assert rel_diff < 0.05

    def test_perpendicular_exceeds_axial(self) -> None:
        """For a capsule with nonzero length, perpendicular > axial."""
        from model_generation.inertia.primitives import capsule_inertia

        result = capsule_inertia(MASS, RADIUS, LENGTH, axis="z")
        assert result["ixx"] > result["izz"]


class TestPhysicalProperties:
    """Verify physical validity of the inertia tensor."""

    def test_triangle_inequality(self) -> None:
        """The principal moments must satisfy the triangle inequality."""
        from model_generation.core.types import Inertia

        result = Inertia.from_capsule(MASS, RADIUS, LENGTH, axis="z")
        assert result.satisfies_triangle_inequality()

    def test_positive_definite(self) -> None:
        """The inertia matrix must be positive definite."""
        from model_generation.core.types import Inertia

        result = Inertia.from_capsule(MASS, RADIUS, LENGTH, axis="z")
        assert result.is_positive_definite()

    def test_all_moments_positive(self) -> None:
        """Every diagonal moment must be strictly positive."""
        from model_generation.inertia.primitives import capsule_inertia

        for axis in ("x", "y", "z"):
            result = capsule_inertia(MASS, RADIUS, LENGTH, axis=axis)
            assert result["ixx"] > 0
            assert result["iyy"] > 0
            assert result["izz"] > 0
