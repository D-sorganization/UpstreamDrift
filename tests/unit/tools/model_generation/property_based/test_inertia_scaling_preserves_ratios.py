"""
Hypothesis property-based tests for URDF model generation.

These tests verify invariants that must hold for *all* valid inputs,
not just hand-picked examples.  Each property is documented with:
  - **What** invariant is tested
  - **Why** it matters for downstream consumers

References:
  - GitHub issue #1694 (Hypothesis property-based tests)
"""

from __future__ import annotations

import defusedxml.ElementTree as DefusedET  # noqa: S314  # Security: defusedxml prevents XML attacks
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from model_generation.builders.manual_builder import ManualBuilder
from model_generation.core.types import (
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointType,
    Link,
    Origin,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies for valid model parameters
# ---------------------------------------------------------------------------

# Physical mass: positive, finite, realistic range
mass_strategy = st.floats(min_value=0.01, max_value=500.0, allow_nan=False)

# Physical dimension: positive, finite, realistic
dimension_strategy = st.floats(min_value=0.005, max_value=5.0, allow_nan=False)

# Inertia diagonal: positive, finite
inertia_diag_strategy = st.floats(min_value=1e-6, max_value=100.0, allow_nan=False)

# Off-diagonal inertia: small relative to diagonal to stay positive-definite
inertia_offdiag_strategy = st.floats(min_value=-0.001, max_value=0.001, allow_nan=False)

# Link name: non-empty alphanumeric
link_name_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"
    ),
    min_size=1,
    max_size=30,
).filter(lambda s: s[0].isalpha())

# Mirror axis
mirror_axis_strategy = st.sampled_from(["x", "y", "z"])

# Scale factor for inertia scaling
scale_factor_strategy = st.floats(min_value=0.01, max_value=100.0, allow_nan=False)


@st.composite
def valid_inertia(draw: st.DrawFn) -> Inertia:
    """Generate a physically valid Inertia (positive-definite, mass > 0).

    Uses primitive factory methods so the triangle inequality is
    always satisfied by construction.
    """
    mass = draw(mass_strategy)
    shape = draw(st.sampled_from(["box", "cylinder", "sphere"]))
    if shape == "box":
        return Inertia.from_box(
            mass,
            draw(dimension_strategy),
            draw(dimension_strategy),
            draw(dimension_strategy),
        )
    if shape == "cylinder":
        return Inertia.from_cylinder(
            mass, draw(dimension_strategy), draw(dimension_strategy)
        )
    return Inertia.from_sphere(mass, draw(dimension_strategy))


@st.composite
def valid_link(draw: st.DrawFn) -> Link:
    """Generate a valid Link with physically valid inertia and a geometry."""
    name = draw(link_name_strategy)
    inertia = draw(valid_inertia())
    # Choose a random primitive geometry
    geom_type = draw(
        st.sampled_from(
            [
                GeometryType.BOX,
                GeometryType.CYLINDER,
                GeometryType.SPHERE,
            ]
        )
    )
    if geom_type == GeometryType.BOX:
        dims = (
            draw(dimension_strategy),
            draw(dimension_strategy),
            draw(dimension_strategy),
        )
    elif geom_type == GeometryType.CYLINDER:
        dims = (draw(dimension_strategy), draw(dimension_strategy))
    else:
        dims = (draw(dimension_strategy),)

    geometry = Geometry(geometry_type=geom_type, dimensions=dims)
    return Link(
        name=name,
        inertia=inertia,
        visual_geometry=geometry,
        collision_geometry=geometry,
    )


@st.composite
def valid_body_params(draw: st.DrawFn) -> dict:
    """Generate valid body parameters (mass, box dimensions)."""
    mass = draw(mass_strategy)
    sx = draw(dimension_strategy)
    sy = draw(dimension_strategy)
    sz = draw(dimension_strategy)
    return {"mass": mass, "size_x": sx, "size_y": sy, "size_z": sz}


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


class TestInertiaScalingPreservesRatios:
    """Property: scaling inertia to a new mass preserves moment ratios."""

    @given(inertia=valid_inertia(), new_mass=mass_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_scale_to_mass_preserves_ratios(
        self,
        inertia: Inertia,
        new_mass: float,
    ) -> None:
        """
        Invariant: Inertia.scale_to_mass(m') produces moments such that
        I'_xx / I'_yy == I_xx / I_yy (ratios are preserved).
        """
        scaled = inertia.scale_to_mass(new_mass)

        # Mass is correct
        assert abs(scaled.mass - new_mass) < 1e-10

        # Scale factor
        expected_factor = new_mass / inertia.mass

        # Each component scales by the same factor
        assert abs(scaled.ixx - inertia.ixx * expected_factor) < 1e-8
        assert abs(scaled.iyy - inertia.iyy * expected_factor) < 1e-8
        assert abs(scaled.izz - inertia.izz * expected_factor) < 1e-8
        assert abs(scaled.ixy - inertia.ixy * expected_factor) < 1e-8
        assert abs(scaled.ixz - inertia.ixz * expected_factor) < 1e-8
        assert abs(scaled.iyz - inertia.iyz * expected_factor) < 1e-8

        # Ratio preserved: I_xx / I_yy == I'_xx / I'_yy (when both > 0)
        if inertia.iyy > 1e-9 and scaled.iyy > 1e-9:
            original_ratio = inertia.ixx / inertia.iyy
            scaled_ratio = scaled.ixx / scaled.iyy
            assert abs(original_ratio - scaled_ratio) < 1e-6

    @given(inertia=valid_inertia(), factor=scale_factor_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_scale_twice_equals_direct_scale(
        self,
        inertia: Inertia,
        factor: float,
    ) -> None:
        """
        Invariant: scaling by factor k then by 1/k returns the original
        (within floating-point tolerance).
        """
        intermediate = inertia.scale_to_mass(inertia.mass * factor)
        restored = intermediate.scale_to_mass(inertia.mass)

        assert abs(restored.mass - inertia.mass) < 1e-8
        assert abs(restored.ixx - inertia.ixx) < 1e-6
        assert abs(restored.iyy - inertia.iyy) < 1e-6
        assert abs(restored.izz - inertia.izz) < 1e-6
