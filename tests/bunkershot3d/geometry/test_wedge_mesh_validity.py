"""Mesh validity tests (issue #8609).

Manifoldness, closure (Euler characteristic), consistent outward normals
and positive volume are *preconditions* for any solver consuming a
clubhead mesh, so they are computed and asserted, never assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.geometry.mesh import (
    MeshValidationError,
    TriangleMesh,
    check_mesh_validity,
    require_watertight,
)
from bunkershot3d.geometry.solids import (
    box_mesh,
    cylinder_mesh,
    icosphere_mesh,
    tetrahedron_mesh,
)

pytestmark = pytest.mark.unit


def _all_solids() -> dict[str, TriangleMesh]:
    return {
        "box": box_mesh(0.03, 0.02, 0.01),
        "tetrahedron": tetrahedron_mesh(
            np.array(
                [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.04]]
            )
        ),
        "icosphere": icosphere_mesh(0.02, subdivisions=2),
        "cylinder": cylinder_mesh(0.015, 0.04, n_segments=24),
    }


class TestTriangleMeshValueObject:
    def test_structure_of_arrays(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        assert mesh.vertices.ndim == 2
        assert mesh.vertices.shape[1] == 3
        assert mesh.faces.shape[1] == 3
        assert mesh.vertices.dtype == np.float64
        assert np.issubdtype(mesh.faces.dtype, np.integer)

    def test_arrays_are_read_only(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            mesh.vertices[0, 0] = 5.0

    def test_counts(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        assert mesh.n_vertices == 8
        assert mesh.n_faces == 12

    def test_rejects_bad_shapes(self) -> None:
        with pytest.raises(ValueError):
            TriangleMesh(np.zeros((4, 2)), np.zeros((1, 3), dtype=np.int64))
        with pytest.raises(ValueError):
            TriangleMesh(np.zeros((4, 3)), np.zeros((1, 4), dtype=np.int64))

    def test_rejects_out_of_range_indices(self) -> None:
        with pytest.raises(ValueError):
            TriangleMesh(np.zeros((4, 3)), np.array([[0, 1, 9]], dtype=np.int64))

    def test_rejects_non_finite_vertices(self) -> None:
        vertices = np.zeros((4, 3))
        vertices[2, 1] = np.nan
        with pytest.raises(ValueError):
            TriangleMesh(vertices, np.array([[0, 1, 2]], dtype=np.int64))

    def test_face_normals_are_unit_length(self) -> None:
        mesh = box_mesh(0.03, 0.02, 0.01)
        norms = np.linalg.norm(mesh.face_normals(), axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=0, atol=1e-12)

    def test_surface_area_of_a_box(self) -> None:
        mesh = box_mesh(2.0, 3.0, 4.0)
        assert mesh.surface_area() == pytest.approx(2 * (6 + 8 + 12))


class TestValidSolids:
    @pytest.mark.parametrize("name", ["box", "tetrahedron", "icosphere", "cylinder"])
    def test_solids_are_watertight(self, name: str) -> None:
        report = check_mesh_validity(_all_solids()[name])
        assert report.is_edge_manifold
        assert report.is_closed
        assert report.is_consistently_oriented
        assert report.is_outward_oriented
        assert report.n_boundary_edges == 0
        assert report.n_nonmanifold_edges == 0
        assert report.n_degenerate_faces == 0
        assert report.signed_volume_m3 > 0.0
        assert report.is_watertight_solid

    @pytest.mark.parametrize("name", ["box", "tetrahedron", "icosphere", "cylinder"])
    def test_euler_characteristic_is_two(self, name: str) -> None:
        report = check_mesh_validity(_all_solids()[name])
        assert report.euler_characteristic == 2
        assert report.genus == 0

    def test_require_watertight_passes_silently(self) -> None:
        require_watertight(box_mesh(1.0, 1.0, 1.0), context="unit test")


class TestInvalidMeshesAreDetected:
    def test_open_mesh_is_rejected(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        holed = TriangleMesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)[1:])
        report = check_mesh_validity(holed)
        assert not report.is_closed
        assert report.n_boundary_edges == 3
        assert not report.is_watertight_solid
        with pytest.raises(MeshValidationError):
            require_watertight(holed, context="unit test")

    def test_inverted_orientation_is_rejected(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        flipped = TriangleMesh(
            np.asarray(mesh.vertices), np.asarray(mesh.faces)[:, ::-1]
        )
        report = check_mesh_validity(flipped)
        assert report.is_closed
        assert report.is_edge_manifold
        assert not report.is_outward_oriented
        assert report.signed_volume_m3 < 0.0
        with pytest.raises(MeshValidationError):
            require_watertight(flipped, context="unit test")

    def test_inconsistent_winding_is_rejected(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        faces = np.array(mesh.faces, copy=True)
        faces[0] = faces[0, ::-1]
        report = check_mesh_validity(TriangleMesh(np.asarray(mesh.vertices), faces))
        assert not report.is_consistently_oriented
        assert not report.is_watertight_solid

    def test_non_manifold_edge_is_detected(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        vertices = np.vstack([np.asarray(mesh.vertices), [[2.0, 2.0, 2.0]]])
        faces = np.vstack([np.asarray(mesh.faces), [[0, 1, 8]]])
        report = check_mesh_validity(TriangleMesh(vertices, faces))
        assert not report.is_edge_manifold
        assert report.n_nonmanifold_edges > 0

    def test_degenerate_face_is_detected(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        faces = np.vstack([np.asarray(mesh.faces), [[0, 1, 1]]])
        report = check_mesh_validity(TriangleMesh(np.asarray(mesh.vertices), faces))
        assert report.n_degenerate_faces == 1
        assert not report.is_watertight_solid

    def test_unreferenced_vertex_is_detected(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        vertices = np.vstack([np.asarray(mesh.vertices), [[9.0, 9.0, 9.0]]])
        report = check_mesh_validity(TriangleMesh(vertices, np.asarray(mesh.faces)))
        assert report.n_unreferenced_vertices == 1
        assert not report.is_watertight_solid

    def test_error_message_names_the_context_and_the_defect(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        flipped = TriangleMesh(
            np.asarray(mesh.vertices), np.asarray(mesh.faces)[:, ::-1]
        )
        with pytest.raises(MeshValidationError, match="solver precondition"):
            require_watertight(flipped, context="solver precondition")


class TestTransforms:
    def test_translation_moves_every_vertex(self) -> None:
        mesh = box_mesh(1.0, 2.0, 3.0)
        shift = np.array([0.5, -1.0, 2.0])
        moved = mesh.transformed(translation=shift)
        np.testing.assert_allclose(moved.vertices, mesh.vertices + shift)
        assert check_mesh_validity(moved).is_watertight_solid

    def test_rotation_preserves_validity(self) -> None:
        mesh = box_mesh(1.0, 2.0, 3.0)
        angle = 0.7
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        turned = mesh.transformed(rotation=rotation)
        assert check_mesh_validity(turned).is_watertight_solid

    def test_rejects_a_non_orthogonal_rotation(self) -> None:
        mesh = box_mesh(1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            mesh.transformed(rotation=np.full((3, 3), 2.0))


class TestStlRoundTrip:
    def test_export_writes_every_facet(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        mesh = box_mesh(0.03, 0.02, 0.01)
        path = tmp_path / "box.stl"
        mesh.to_stl(path, name="box")
        text = path.read_text(encoding="utf-8")
        assert text.count("facet normal") == mesh.n_faces
        assert text.startswith("solid box")
