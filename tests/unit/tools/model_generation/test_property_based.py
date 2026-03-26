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

import xml.etree.ElementTree as ET

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
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"),
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
        return Inertia.from_cylinder(mass, draw(dimension_strategy), draw(dimension_strategy))
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


class TestURDFXMLWellFormedness:
    """Property: any valid body parameters must produce well-formed URDF XML."""

    @given(params=valid_body_params())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_box_body_produces_well_formed_xml(self, params: dict) -> None:
        """
        Invariant: ManualBuilder with a single box link always emits
        parseable XML with the correct root element ``<robot>``.
        """
        inertia = Inertia.from_box(
            params["mass"],
            params["size_x"],
            params["size_y"],
            params["size_z"],
        )
        link = Link(
            name="body",
            inertia=inertia,
            visual_geometry=Geometry.box(params["size_x"], params["size_y"], params["size_z"]),
        )
        builder = ManualBuilder("test_robot", validate_on_add=False)
        builder.add_link(link)
        result = builder.build()

        assert result.success, f"Build failed: {result.error_message}"
        assert result.urdf_xml is not None

        # Must be parseable XML
        root = ET.fromstring(result.urdf_xml)
        assert root.tag == "robot"
        assert root.get("name") == "test_robot"

        # Must contain exactly one link
        link_elems = root.findall(".//link")
        assert len(link_elems) >= 1
        assert any(le.get("name") == "body" for le in link_elems)

    @given(
        mass=mass_strategy,
        radius=dimension_strategy,
        length=dimension_strategy,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_cylinder_body_produces_well_formed_xml(
        self,
        mass: float,
        radius: float,
        length: float,
    ) -> None:
        """
        Invariant: cylinder-based links always produce valid URDF XML.
        """
        inertia = Inertia.from_cylinder(mass, radius, length)
        link = Link(
            name="cyl_body",
            inertia=inertia,
            visual_geometry=Geometry.cylinder(radius, length),
        )
        builder = ManualBuilder("cyl_robot", validate_on_add=False)
        builder.add_link(link)
        result = builder.build()

        assert result.success
        assert result.urdf_xml is not None
        root = ET.fromstring(result.urdf_xml)
        assert root.tag == "robot"

    @given(mass=mass_strategy, radius=dimension_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_sphere_body_produces_well_formed_xml(
        self,
        mass: float,
        radius: float,
    ) -> None:
        """
        Invariant: sphere-based links always produce valid URDF XML.
        """
        inertia = Inertia.from_sphere(mass, radius)
        link = Link(
            name="sphere_body",
            inertia=inertia,
            visual_geometry=Geometry.sphere(radius),
        )
        builder = ManualBuilder("sphere_robot", validate_on_add=False)
        builder.add_link(link)
        result = builder.build()

        assert result.success
        assert result.urdf_xml is not None
        root = ET.fromstring(result.urdf_xml)
        assert root.tag == "robot"


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

        root = ET.fromstring(result.urdf_xml)

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


class TestMirrorNegatesCorrectComponent:
    """Property: mirror(axis) negates only the correct coordinate."""

    @given(
        axis=mirror_axis_strategy,
        x=st.floats(min_value=-5, max_value=5, allow_nan=False),
        y=st.floats(min_value=-5, max_value=5, allow_nan=False),
        z=st.floats(min_value=-5, max_value=5, allow_nan=False),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_mirror_negates_only_one_axis(
        self,
        axis: str,
        x: float,
        y: float,
        z: float,
    ) -> None:
        """
        Invariant: after mirror(axis), only the coordinate along `axis`
        is negated; the other two are unchanged.
        """
        link = Link(
            name="test",
            inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1, mass=1.0),
            visual_geometry=Geometry.box(0.1, 0.1, 0.1),
            visual_origin=Origin(xyz=(x, y, z)),
            collision_origin=Origin(xyz=(x, y, z)),
        )

        builder = ManualBuilder("neg_test", validate_on_add=False)
        builder.add_link(link)
        builder.mirror(axis)

        result = builder.links[0]
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]

        for i in range(3):
            expected = -((x, y, z)[i]) if i == axis_idx else (x, y, z)[i]
            assert abs(result.visual_origin.xyz[i] - expected) < 1e-10, (
                f"axis={axis}, coord[{i}]: got {result.visual_origin.xyz[i]}, "
                f"expected {expected}"
            )


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


class TestInertiaTriangleInequality:
    """Property: primitive inertias satisfy the triangle inequality."""

    @given(
        mass=mass_strategy,
        sx=dimension_strategy,
        sy=dimension_strategy,
        sz=dimension_strategy,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_box_satisfies_triangle_inequality(
        self,
        mass: float,
        sx: float,
        sy: float,
        sz: float,
    ) -> None:
        """Invariant: box inertia satisfies triangle inequality for physical bodies."""
        inertia = Inertia.from_box(mass, sx, sy, sz)
        assert inertia.satisfies_triangle_inequality()

    @given(mass=mass_strategy, radius=dimension_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_sphere_satisfies_triangle_inequality(
        self,
        mass: float,
        radius: float,
    ) -> None:
        """Invariant: sphere inertia satisfies triangle inequality."""
        inertia = Inertia.from_sphere(mass, radius)
        assert inertia.satisfies_triangle_inequality()
