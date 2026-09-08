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


class TestLinkJointHierarchyConsistency:
    """Property: arbitrary valid link/joint combos produce consistent hierarchy."""

    @given(
        root_link=valid_link(),
        child_link=valid_link(),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_two_link_chain_hierarchy_is_consistent(
        self,
        root_link: Link,
        child_link: Link,
    ) -> None:
        """
        Invariant: connecting two links with a joint always produces
        URDF where the parent/child references match the link names.
        """
        # Ensure distinct names
        if root_link.name == child_link.name:
            child_link = Link(
                name=child_link.name + "_child",
                inertia=child_link.inertia,
                visual_geometry=child_link.visual_geometry,
                collision_geometry=child_link.collision_geometry,
            )

        joint = Joint(
            name=f"{root_link.name}_to_{child_link.name}",
            joint_type=JointType.FIXED,
            parent=root_link.name,
            child=child_link.name,
        )

        builder = ManualBuilder("hierarchy_test", validate_on_add=False)
        builder.add_link(root_link)
        builder.add_link(child_link)
        builder.add_joint(joint)
        result = builder.build()

        assert result.success, f"Build failed: {result.error_message}"
        assert result.urdf_xml is not None

        root = DefusedET.fromstring(result.urdf_xml)

        # Verify link names present
        link_names_in_xml = {le.get("name") for le in root.findall(".//link")}
        assert root_link.name in link_names_in_xml
        assert child_link.name in link_names_in_xml

        # Verify joint parent/child references
        joint_elems = root.findall(".//joint")
        assert len(joint_elems) >= 1
        j_elem = joint_elems[0]
        parent_elem = j_elem.find("parent")
        child_elem = j_elem.find("child")
        assert parent_elem is not None
        assert child_elem is not None
        assert parent_elem.get("link") == root_link.name
        assert child_elem.get("link") == child_link.name

    @given(n_children=st.integers(min_value=1, max_value=5))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_star_topology_all_children_connected(self, n_children: int) -> None:
        """
        Invariant: a root with N children produces N joints and N+1 links.
        Each child is reachable from the root via exactly one joint.
        """
        root = Link(name="root", inertia=Inertia.from_box(10, 0.5, 0.5, 0.5))

        builder = ManualBuilder("star_robot", validate_on_add=False)
        builder.add_link(root)

        for i in range(n_children):
            child = Link(
                name=f"child_{i}",
                inertia=Inertia.from_sphere(1.0, 0.1),
            )
            joint = Joint(
                name=f"root_to_child_{i}",
                joint_type=JointType.REVOLUTE,
                parent="root",
                child=f"child_{i}",
                axis=(0, 0, 1),
            )
            builder.add_link(child)
            builder.add_joint(joint)

        result = builder.build()
        assert result.success
        assert len(result.links) == n_children + 1
        assert len(result.joints) == n_children

        # Verify every child is reachable
        child_names = result.get_children("root")
        assert len(child_names) == n_children
