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


class TestInertiaFromPrimitivesPositiveDefinite:
    """Property: factory methods always produce positive-definite tensors."""

    @given(
        mass=mass_strategy,
        sx=dimension_strategy,
        sy=dimension_strategy,
        sz=dimension_strategy,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_box_inertia_positive_definite(
        self,
        mass: float,
        sx: float,
        sy: float,
        sz: float,
    ) -> None:
        """Invariant: Inertia.from_box always produces positive-definite tensor."""
        inertia = Inertia.from_box(mass, sx, sy, sz)
        assert inertia.mass == mass
        assert inertia.ixx > 0
        assert inertia.iyy > 0
        assert inertia.izz > 0
        assert inertia.is_positive_definite()

    @given(mass=mass_strategy, radius=dimension_strategy, length=dimension_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_cylinder_inertia_positive_definite(
        self,
        mass: float,
        radius: float,
        length: float,
    ) -> None:
        """Invariant: Inertia.from_cylinder always produces positive-definite tensor."""
        inertia = Inertia.from_cylinder(mass, radius, length)
        assert inertia.is_positive_definite()

    @given(mass=mass_strategy, radius=dimension_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_sphere_inertia_positive_definite(
        self,
        mass: float,
        radius: float,
    ) -> None:
        """Invariant: Inertia.from_sphere always produces positive-definite tensor."""
        inertia = Inertia.from_sphere(mass, radius)
        assert inertia.is_positive_definite()
        # Sphere inertia must be isotropic
        assert abs(inertia.ixx - inertia.iyy) < 1e-12
        assert abs(inertia.iyy - inertia.izz) < 1e-12

    @given(mass=mass_strategy, radius=dimension_strategy, length=dimension_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_capsule_inertia_positive_definite(
        self,
        mass: float,
        radius: float,
        length: float,
    ) -> None:
        """Invariant: Inertia.from_capsule always produces positive-definite tensor."""
        inertia = Inertia.from_capsule(mass, radius, length)
        assert inertia.is_positive_definite()
