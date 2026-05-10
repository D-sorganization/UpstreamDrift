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


class TestMirrorInvolution:
    """Property: mirror(axis) applied twice == identity."""

    @given(axis=mirror_axis_strategy)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_mirror_twice_is_identity_single_link(self, axis: str) -> None:
        """
        Invariant: mirroring a single-link model about any axis twice
        restores the original visual/collision origins and inertia COM.
        """
        original_xyz = (0.1, 0.2, 0.3)
        original_com = (0.05, -0.1, 0.15)
        link = Link(
            name="body",
            inertia=Inertia(
                ixx=1.0,
                iyy=2.0,
                izz=3.0,
                ixy=0.01,
                ixz=0.02,
                iyz=0.03,
                mass=5.0,
                center_of_mass=original_com,
            ),
            visual_geometry=Geometry.box(0.3, 0.4, 0.5),
            visual_origin=Origin(xyz=original_xyz),
            collision_geometry=Geometry.box(0.3, 0.4, 0.5),
            collision_origin=Origin(xyz=original_xyz),
        )

        builder = ManualBuilder("mirror_test", validate_on_add=False)
        builder.add_link(link)

        # Mirror twice
        builder.mirror(axis)
        builder.mirror(axis)

        result_link = builder.links[0]

        # Visual origin restored
        for i in range(3):
            assert abs(result_link.visual_origin.xyz[i] - original_xyz[i]) < 1e-10, (
                f"visual_origin[{i}] mismatch after double mirror({axis}): "
                f"{result_link.visual_origin.xyz[i]} != {original_xyz[i]}"
            )

        # Collision origin restored
        for i in range(3):
            assert abs(result_link.collision_origin.xyz[i] - original_xyz[i]) < 1e-10

        # COM restored
        for i in range(3):
            assert abs(result_link.inertia.center_of_mass[i] - original_com[i]) < 1e-10

    @given(axis=mirror_axis_strategy)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_mirror_twice_is_identity_two_link_chain(self, axis: str) -> None:
        """
        Invariant: mirroring a two-link chain twice restores joint origins
        and joint axes to their original values.
        """
        joint_origin = (0.0, 0.5, -0.3)
        joint_axis = (1.0, 0.0, 0.0)

        builder = ManualBuilder("chain_mirror", validate_on_add=False)
        builder.add_link(
            Link(
                name="base",
                inertia=Inertia.from_box(10, 0.5, 0.5, 0.5),
                visual_geometry=Geometry.box(0.5, 0.5, 0.5),
            )
        )
        builder.add_link(
            Link(
                name="arm",
                inertia=Inertia.from_cylinder(2, 0.05, 0.4),
                visual_geometry=Geometry.cylinder(0.05, 0.4),
            )
        )
        builder.add_joint(
            Joint(
                name="base_to_arm",
                joint_type=JointType.REVOLUTE,
                parent="base",
                child="arm",
                origin=Origin(xyz=joint_origin),
                axis=joint_axis,
            )
        )

        builder.mirror(axis)
        builder.mirror(axis)

        result_joint = builder.joints[0]

        for i in range(3):
            assert (
                abs(result_joint.origin.xyz[i] - joint_origin[i]) < 1e-10
            ), f"joint origin[{i}] mismatch after double mirror({axis})"
            assert (
                abs(result_joint.axis[i] - joint_axis[i]) < 1e-10
            ), f"joint axis[{i}] mismatch after double mirror({axis})"

    @given(axis=mirror_axis_strategy)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_mirror_toggles_handedness(self, axis: str) -> None:
        """
        Invariant: mirroring toggles handedness, so double mirror
        restores original handedness.
        """

        builder = ManualBuilder("hand_test")
        original_handedness = builder.handedness

        builder.mirror(axis)
        assert builder.handedness != original_handedness

        builder.mirror(axis)
        assert builder.handedness == original_handedness
