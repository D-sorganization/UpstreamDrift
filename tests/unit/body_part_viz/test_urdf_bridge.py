"""Tests for :mod:`body_part_viz.urdf_bridge` (issue #4765)."""

from __future__ import annotations

import xml.etree.ElementTree as ET  # noqa: S405 — generating, not parsing untrusted input
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.body_part_viz.shapes import (
    CapsuleShape,
    CompositeShape,
    CylinderShape,
    EllipsoidShape,
    LineShape,
    MeshShape,
)
from src.shared.python.body_part_viz.urdf_bridge import (
    DEFAULT_PACKAGE,
    shape_to_urdf_visual,
    urdf_to_shape,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolver_for_mesh(stl_path: Path) -> Callable[[str], Path]:
    def resolver(uri: str) -> Path:
        # Bridge always emits "package://body_part_viz/<filename>"; we ignore
        # the URI and return the canonical on-disk file the test wrote.
        del uri
        return stl_path

    return resolver


def _no_mesh_resolver(uri: str) -> Path:
    raise AssertionError(f"asset_resolver should not be called; got {uri!r}")


def _make_mesh_shape(tmp_path: Path) -> MeshShape:
    """Build a tiny on-disk STL (a non-degenerate tetrahedron) and load it."""
    stl_path = tmp_path / "tiny.stl"
    # Tetrahedron with vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1).
    stl_path.write_text(
        "solid tiny\n"
        " facet normal 0 0 -1\n"
        "  outer loop\n"
        "   vertex 0 0 0\n"
        "   vertex 0 1 0\n"
        "   vertex 1 0 0\n"
        "  endloop\n"
        " endfacet\n"
        " facet normal 0 -1 0\n"
        "  outer loop\n"
        "   vertex 0 0 0\n"
        "   vertex 1 0 0\n"
        "   vertex 0 0 1\n"
        "  endloop\n"
        " endfacet\n"
        " facet normal -1 0 0\n"
        "  outer loop\n"
        "   vertex 0 0 0\n"
        "   vertex 0 0 1\n"
        "   vertex 0 1 0\n"
        "  endloop\n"
        " endfacet\n"
        " facet normal 1 1 1\n"
        "  outer loop\n"
        "   vertex 1 0 0\n"
        "   vertex 0 1 0\n"
        "   vertex 0 0 1\n"
        "  endloop\n"
        " endfacet\n"
        "endsolid tiny\n",
        encoding="utf-8",
    )
    return MeshShape.load(stl_path)


# ---------------------------------------------------------------------------
# Forward translation
# ---------------------------------------------------------------------------


def test_line_shape_raises() -> None:
    with pytest.raises(ValueError, match="URDF cannot render line visuals"):
        shape_to_urdf_visual(LineShape(length=1.0))


def test_cylinder_visual_payload() -> None:
    cyl = CylinderShape(length=2.0, radius=0.05)
    visual = shape_to_urdf_visual(cyl)
    assert isinstance(visual, ET.Element)
    assert visual.tag == "visual"
    assert visual.find("origin") is None  # zero origin omitted
    geom = visual.find("geometry")
    assert geom is not None
    cyl_elem = geom.find("cylinder")
    assert cyl_elem is not None
    assert float(cyl_elem.attrib["length"]) == pytest.approx(2.0, abs=1e-12)
    assert float(cyl_elem.attrib["radius"]) == pytest.approx(0.05, abs=1e-12)


def test_ellipsoid_emits_mesh_with_encoded_axes() -> None:
    ell = EllipsoidShape(a=0.1, b=0.2, c=0.3)
    visual = shape_to_urdf_visual(ell)
    assert isinstance(visual, ET.Element)
    mesh = visual.find("geometry/mesh")
    assert mesh is not None
    filename = mesh.attrib["filename"]
    assert filename.startswith(f"package://{DEFAULT_PACKAGE}/__bpv_ellipsoid__")
    assert filename.endswith(".obj")


def test_capsule_emits_mesh_with_encoded_dims() -> None:
    cap = CapsuleShape(length=0.4, radius=0.03)
    visual = shape_to_urdf_visual(cap)
    mesh = visual.find("geometry/mesh")
    assert mesh is not None
    filename = mesh.attrib["filename"]
    assert filename.startswith(f"package://{DEFAULT_PACKAGE}/__bpv_capsule__")


def test_mesh_shape_uses_package_uri(tmp_path: Path) -> None:
    mesh_shape = _make_mesh_shape(tmp_path)
    visual = shape_to_urdf_visual(mesh_shape)
    mesh_elem = visual.find("geometry/mesh")
    assert mesh_elem is not None
    assert mesh_elem.attrib["filename"] == (
        f"package://{DEFAULT_PACKAGE}/{mesh_shape.source_path.name}"
    )


def test_mesh_shape_without_source_path_raises() -> None:
    mesh = MeshShape(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        rest_dimensions=(1.0, 1.0, 0.001),
    )
    with pytest.raises(ValueError, match="source_path"):
        shape_to_urdf_visual(mesh)


def test_composite_returns_list_with_correct_child_count() -> None:
    children = [
        (CylinderShape(length=1.0, radius=0.02), np.eye(4)),
        (CylinderShape(length=0.5, radius=0.03), np.eye(4)),
        (CylinderShape(length=0.3, radius=0.04), np.eye(4)),
    ]
    composite = CompositeShape(children=children)
    visuals = shape_to_urdf_visual(composite)
    assert isinstance(visuals, list)
    assert len(visuals) == 3
    for visual in visuals:
        assert visual.tag == "visual"
        assert visual.find("geometry/cylinder") is not None


def test_composite_propagates_child_local_transform() -> None:
    transform = np.eye(4)
    transform[0, 3] = 0.1
    transform[1, 3] = -0.2
    transform[2, 3] = 0.3
    children = [(CylinderShape(length=1.0, radius=0.02), transform)]
    composite = CompositeShape(children=children)
    visuals = shape_to_urdf_visual(composite)
    assert isinstance(visuals, list) and len(visuals) == 1
    origin = visuals[0].find("origin")
    assert origin is not None
    xyz = [float(v) for v in origin.attrib["xyz"].split()]
    assert xyz == pytest.approx([0.1, -0.2, 0.3], abs=1e-9)


def test_origin_emitted_when_non_zero() -> None:
    cyl = CylinderShape(length=1.0, radius=0.02)
    visual = shape_to_urdf_visual(
        cyl,
        rest_origin_xyz=(0.0, 0.0, 0.5),
        rest_origin_rpy=(0.0, 0.0, 0.0),
    )
    assert isinstance(visual, ET.Element)
    origin = visual.find("origin")
    assert origin is not None
    xyz = [float(v) for v in origin.attrib["xyz"].split()]
    assert xyz == pytest.approx([0.0, 0.0, 0.5], abs=1e-12)


def test_invalid_shape_type_raises() -> None:
    class _Bogus:
        shape_id = "bogus"
        rest_dimensions = (1.0,)

        def vertices_at_rest(self):
            return np.zeros((1, 3))

        def faces(self):
            return np.zeros((0, 3), dtype=np.int64)

        def transform(self, fitted):
            return self.vertices_at_rest()

    with pytest.raises(TypeError, match="Unsupported shape type"):
        shape_to_urdf_visual(_Bogus())


def test_none_shape_raises() -> None:
    with pytest.raises(TypeError, match="shape must not be None"):
        shape_to_urdf_visual(None)  # type: ignore[arg-type]


def test_invalid_origin_tuple_raises() -> None:
    cyl = CylinderShape(length=1.0, radius=0.02)
    with pytest.raises(TypeError, match="rest_origin_xyz"):
        shape_to_urdf_visual(cyl, rest_origin_xyz=(0.0, 0.0))  # type: ignore[arg-type]


def test_empty_package_name_raises() -> None:
    cyl = CylinderShape(length=1.0, radius=0.02)
    with pytest.raises(ValueError, match="package_name"):
        shape_to_urdf_visual(cyl, package_name="")


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_cylinder() -> None:
    original = CylinderShape(length=1.234, radius=0.0567)
    visual = shape_to_urdf_visual(original)
    recovered = urdf_to_shape(visual, _no_mesh_resolver)
    assert isinstance(recovered, CylinderShape)
    assert recovered.rest_dimensions == pytest.approx(
        original.rest_dimensions, abs=1e-9
    )


def test_round_trip_ellipsoid() -> None:
    original = EllipsoidShape(a=0.123, b=0.234, c=0.345)
    visual = shape_to_urdf_visual(original)
    recovered = urdf_to_shape(visual, _no_mesh_resolver)
    assert isinstance(recovered, EllipsoidShape)
    assert recovered.rest_dimensions == pytest.approx(
        original.rest_dimensions, abs=1e-9
    )


def test_round_trip_capsule() -> None:
    original = CapsuleShape(length=0.42, radius=0.07)
    visual = shape_to_urdf_visual(original)
    recovered = urdf_to_shape(visual, _no_mesh_resolver)
    assert isinstance(recovered, CapsuleShape)
    assert recovered.rest_dimensions == pytest.approx(
        original.rest_dimensions, abs=1e-9
    )


def test_round_trip_mesh(tmp_path: Path) -> None:
    original = _make_mesh_shape(tmp_path)
    visual = shape_to_urdf_visual(original)
    recovered = urdf_to_shape(visual, _resolver_for_mesh(original.source_path))
    assert isinstance(recovered, MeshShape)
    assert recovered.source_path is not None
    assert recovered.source_path.name == original.source_path.name


def test_round_trip_composite() -> None:
    children = [
        (CylinderShape(length=1.0, radius=0.02), np.eye(4)),
        (EllipsoidShape(a=0.1, b=0.1, c=0.2), np.eye(4)),
    ]
    composite = CompositeShape(children=children)
    visuals = shape_to_urdf_visual(composite)
    assert isinstance(visuals, list)
    recovered = [urdf_to_shape(v, _no_mesh_resolver) for v in visuals]
    assert isinstance(recovered[0], CylinderShape)
    assert isinstance(recovered[1], EllipsoidShape)
    assert recovered[0].rest_dimensions == pytest.approx((1.0, 0.02), abs=1e-9)
    assert recovered[1].rest_dimensions == pytest.approx((0.1, 0.1, 0.2), abs=1e-9)


# ---------------------------------------------------------------------------
# Inverse-only error paths
# ---------------------------------------------------------------------------


def test_urdf_to_shape_rejects_none() -> None:
    with pytest.raises(ValueError, match="must not be None"):
        urdf_to_shape(None, _no_mesh_resolver)  # type: ignore[arg-type]


def test_urdf_to_shape_rejects_non_visual_tag() -> None:
    with pytest.raises(ValueError, match="must be 'visual'"):
        urdf_to_shape(ET.Element("link"), _no_mesh_resolver)


def test_urdf_to_shape_requires_geometry() -> None:
    with pytest.raises(ValueError, match="<geometry>"):
        urdf_to_shape(ET.Element("visual"), _no_mesh_resolver)


def test_urdf_to_shape_rejects_multiple_geometries() -> None:
    visual = ET.Element("visual")
    geom = ET.SubElement(visual, "geometry")
    ET.SubElement(geom, "cylinder", length="1", radius="0.1")
    ET.SubElement(geom, "sphere", radius="0.5")
    with pytest.raises(ValueError, match="exactly one child"):
        urdf_to_shape(visual, _no_mesh_resolver)


def test_urdf_to_shape_rejects_unknown_geometry() -> None:
    visual = ET.Element("visual")
    geom = ET.SubElement(visual, "geometry")
    ET.SubElement(geom, "torus", inner_radius="0.1", outer_radius="0.5")
    with pytest.raises(ValueError, match="Unsupported URDF geometry tag"):
        urdf_to_shape(visual, _no_mesh_resolver)


def test_urdf_to_shape_rejects_malformed_ellipsoid_filename() -> None:
    visual = ET.Element("visual")
    geom = ET.SubElement(visual, "geometry")
    ET.SubElement(
        geom,
        "mesh",
        filename="package://body_part_viz/__bpv_ellipsoid__1.0_2.0.obj",
        scale="1 1 1",
    )
    with pytest.raises(ValueError, match="Malformed ellipsoid"):
        urdf_to_shape(visual, _no_mesh_resolver)


def test_urdf_to_shape_rejects_malformed_capsule_filename() -> None:
    visual = ET.Element("visual")
    geom = ET.SubElement(visual, "geometry")
    ET.SubElement(
        geom,
        "mesh",
        filename="package://body_part_viz/__bpv_capsule__1.0.obj",
        scale="1 1 1",
    )
    with pytest.raises(ValueError, match="Malformed capsule"):
        urdf_to_shape(visual, _no_mesh_resolver)


# ---------------------------------------------------------------------------
# Internal: rotation matrix decomposition
# ---------------------------------------------------------------------------


def test_composite_zero_rotation_omits_origin_when_at_origin() -> None:
    children = [(CylinderShape(length=1.0, radius=0.02), np.eye(4))]
    composite = CompositeShape(children=children)
    visuals = shape_to_urdf_visual(composite)
    assert isinstance(visuals, list) and len(visuals) == 1
    assert visuals[0].find("origin") is None


def test_composite_rotation_propagates_to_rpy() -> None:
    transform = np.eye(4)
    # 90-degree rotation about z: roll=0, pitch=0, yaw=pi/2.
    cos_a, sin_a = np.cos(np.pi / 2), np.sin(np.pi / 2)
    transform[:3, :3] = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]]
    )
    children = [(CylinderShape(length=1.0, radius=0.02), transform)]
    composite = CompositeShape(children=children)
    visuals = shape_to_urdf_visual(composite)
    assert isinstance(visuals, list) and len(visuals) == 1
    origin = visuals[0].find("origin")
    assert origin is not None
    rpy = [float(v) for v in origin.attrib["rpy"].split()]
    assert rpy[2] == pytest.approx(np.pi / 2, abs=1e-9)
    assert rpy[0] == pytest.approx(0.0, abs=1e-9)
    assert rpy[1] == pytest.approx(0.0, abs=1e-9)
