"""Triangle-mesh value object and validity checks (issue #8609).

The mesh is stored **structure-of-arrays** - one ``(n, 3)`` vertex array
and one ``(m, 3)`` index array - not as per-facet objects.  That is both
the Law-of-Demeter answer (``mesh.vertices`` is one dot, and there is no
``Facet`` class to reach through) and 10-100x faster in NumPy.

Watertightness is a **precondition** for anything that integrates over
the surface, so :func:`check_mesh_validity` computes it and
:func:`require_watertight` raises.  Nothing here uses ``assert``:
``python -O`` strips assertions and these checks are load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..exceptions import BunkerShot3DValueError

__all__ = [
    "MeshValidationError",
    "MeshValidity",
    "TriangleMesh",
    "check_mesh_validity",
    "require_watertight",
]

_DEGENERATE_AREA_M2 = 1e-20


class MeshValidationError(BunkerShot3DValueError):
    """Raised when a mesh fails a validity precondition."""


@dataclass(frozen=True, eq=False, init=False)
class TriangleMesh:
    """An immutable indexed triangle mesh in SI units (metres).

    Args:
        vertices: ``(n, 3)`` float coordinates.
        faces: ``(m, 3)`` integer vertex indices, wound counter-clockwise
            seen from outside the solid.

    Raises:
        ValueError: If the arrays are malformed, non-finite, or index out
            of range.
    """

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]

    def __init__(self, vertices: ArrayLike, faces: ArrayLike) -> None:
        vertex_array = np.array(vertices, dtype=np.float64, copy=True)
        face_array = np.array(faces, dtype=np.int64, copy=True)
        if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (n, 3), got {vertex_array.shape}"
            )
        if face_array.ndim != 2 or face_array.shape[1] != 3:
            raise ValueError(f"faces must have shape (m, 3), got {face_array.shape}")
        if not np.all(np.isfinite(vertex_array)):
            raise ValueError("vertices contain non-finite values (NaN or Inf)")
        if face_array.size and (
            face_array.min() < 0 or face_array.max() >= vertex_array.shape[0]
        ):
            raise ValueError(
                "face indices out of range for "
                f"{vertex_array.shape[0]} vertices: "
                f"[{face_array.min()}, {face_array.max()}]"
            )
        vertex_array.flags.writeable = False
        face_array.flags.writeable = False
        object.__setattr__(self, "vertices", vertex_array)
        object.__setattr__(self, "faces", face_array)

    @property
    def n_vertices(self) -> int:
        """Number of vertices."""
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        """Number of triangles."""
        return int(self.faces.shape[0])

    def triangle_corners(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """The three corner arrays of every triangle, each ``(m, 3)``."""
        corners = self.vertices[self.faces]
        return corners[:, 0, :], corners[:, 1, :], corners[:, 2, :]

    def face_area_vectors(self) -> NDArray[np.float64]:
        """Half the edge cross product per face: area * outward normal."""
        first, second, third = self.triangle_corners()
        vectors = 0.5 * np.cross(second - first, third - first)
        return np.asarray(vectors, dtype=np.float64)

    def face_areas(self) -> NDArray[np.float64]:
        """Triangle areas in m^2."""
        return np.linalg.norm(self.face_area_vectors(), axis=1)

    def face_normals(self) -> NDArray[np.float64]:
        """Unit outward normals; degenerate faces yield a zero row."""
        vectors = self.face_area_vectors()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe = np.where(norms > 0.0, norms, 1.0)
        return np.where(norms > 0.0, vectors / safe, 0.0)

    def surface_area(self) -> float:
        """Total surface area in m^2."""
        return float(self.face_areas().sum())

    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Axis-aligned bounding box as ``(minimum, maximum)``."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def transformed(
        self,
        *,
        rotation: ArrayLike | None = None,
        translation: ArrayLike | None = None,
        check_orthogonal: bool = True,
    ) -> TriangleMesh:
        """Return a rigidly moved copy: ``v -> R v + t``.

        Args:
            rotation: ``(3, 3)`` matrix. Checked for orthogonality with a
                positive determinant unless ``check_orthogonal`` is off,
                which permits uniform scaling for dimensional tests.
            translation: ``(3,)`` offset in metres.
            check_orthogonal: Enforce that ``rotation`` is a proper
                rotation.

        Raises:
            ValueError: If the transform is malformed.
        """
        moved = self.vertices
        if rotation is not None:
            matrix = np.asarray(rotation, dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError(f"rotation must be (3, 3), got {matrix.shape}")
            if not np.all(np.isfinite(matrix)):
                raise ValueError("rotation contains non-finite values")
            if check_orthogonal:
                if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10):
                    raise ValueError("rotation is not orthogonal")
                if float(np.linalg.det(matrix)) <= 0.0:
                    raise ValueError("rotation must have a positive determinant")
            moved = moved @ matrix.T
        if translation is not None:
            offset = np.asarray(translation, dtype=np.float64)
            if offset.shape != (3,):
                raise ValueError(f"translation must be (3,), got {offset.shape}")
            moved = moved + offset
        return TriangleMesh(moved, self.faces)

    def to_stl(self, path: Path | str, *, name: str = "mesh") -> None:
        """Write an ASCII STL with per-face normals from the winding."""
        first, second, third = self.triangle_corners()
        normals = self.face_normals()
        lines = [f"solid {name}"]
        for index in range(self.n_faces):
            nx, ny, nz = normals[index]
            lines.append(f"  facet normal {nx:.9e} {ny:.9e} {nz:.9e}")
            lines.append("    outer loop")
            for corner in (first[index], second[index], third[index]):
                lines.append(
                    f"      vertex {corner[0]:.9e} {corner[1]:.9e} {corner[2]:.9e}"
                )
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {name}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class MeshValidity:
    """The outcome of :func:`check_mesh_validity`."""

    n_vertices: int
    n_faces: int
    n_edges: int
    n_boundary_edges: int
    n_nonmanifold_edges: int
    n_degenerate_faces: int
    n_unreferenced_vertices: int
    n_reversed_edge_pairs: int
    signed_volume_m3: float

    @property
    def is_closed(self) -> bool:
        """Every edge is shared by exactly two triangles."""
        return self.n_boundary_edges == 0 and self.n_nonmanifold_edges == 0

    @property
    def is_edge_manifold(self) -> bool:
        """No edge is shared by more than two triangles."""
        return self.n_nonmanifold_edges == 0

    @property
    def is_consistently_oriented(self) -> bool:
        """Each shared edge is traversed in opposite directions."""
        return self.is_closed and self.n_reversed_edge_pairs == self.n_edges

    @property
    def is_outward_oriented(self) -> bool:
        """Face winding encloses a positive volume."""
        return self.signed_volume_m3 > 0.0

    @property
    def euler_characteristic(self) -> int:
        """``V - E + F`` over the referenced vertices."""
        used_vertices = self.n_vertices - self.n_unreferenced_vertices
        return used_vertices - self.n_edges + self.n_faces

    @property
    def genus(self) -> int:
        """Surface genus from the Euler characteristic (closed surfaces)."""
        return (2 - self.euler_characteristic) // 2

    @property
    def is_watertight_solid(self) -> bool:
        """Closed, manifold, consistently wound outward, non-degenerate."""
        return (
            self.is_closed
            and self.is_edge_manifold
            and self.is_consistently_oriented
            and self.is_outward_oriented
            and self.n_degenerate_faces == 0
            and self.n_unreferenced_vertices == 0
        )

    def failures(self) -> tuple[str, ...]:
        """Human-readable reasons the mesh is not a valid solid."""
        reasons: list[str] = []
        if self.n_boundary_edges:
            reasons.append(f"{self.n_boundary_edges} boundary edge(s): not closed")
        if self.n_nonmanifold_edges:
            reasons.append(f"{self.n_nonmanifold_edges} non-manifold edge(s)")
        if self.is_closed and not self.is_consistently_oriented:
            reasons.append("inconsistent face winding")
        if not self.is_outward_oriented:
            reasons.append(
                f"signed volume {self.signed_volume_m3:.6g} m^3 is not positive: "
                "normals point inward"
            )
        if self.n_degenerate_faces:
            reasons.append(f"{self.n_degenerate_faces} zero-area face(s)")
        if self.n_unreferenced_vertices:
            reasons.append(f"{self.n_unreferenced_vertices} unreferenced vertex/ices")
        return tuple(reasons)


def signed_volume_m3(mesh: TriangleMesh) -> float:
    """Signed volume by the divergence theorem over the triangle fan.

    Positive when the winding is counter-clockwise seen from outside.
    """
    first, second, third = mesh.triangle_corners()
    return float(np.einsum("ij,ij->i", first, np.cross(second, third)).sum() / 6.0)


def check_mesh_validity(mesh: TriangleMesh) -> MeshValidity:
    """Compute every solid-mesh precondition in one vectorised pass.

    Args:
        mesh: Mesh to inspect.

    Returns:
        A :class:`MeshValidity` report; nothing is raised, so callers can
        inspect a broken mesh.
    """
    faces = mesh.faces
    directed = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    undirected = np.sort(directed, axis=1)
    unique_edges, counts = np.unique(undirected, axis=0, return_counts=True)

    # A closed, consistently wound surface traverses every edge once in
    # each direction, so each directed edge must find its reverse.
    stride = np.int64(mesh.n_vertices + 1)
    forward_keys = directed[:, 0] * stride + directed[:, 1]
    reverse_keys = directed[:, 1] * stride + directed[:, 0]
    reversed_pairs = int(np.count_nonzero(np.isin(forward_keys, reverse_keys)) // 2)

    referenced = np.zeros(mesh.n_vertices, dtype=bool)
    referenced[faces.reshape(-1)] = True

    return MeshValidity(
        n_vertices=mesh.n_vertices,
        n_faces=mesh.n_faces,
        n_edges=int(unique_edges.shape[0]),
        n_boundary_edges=int(np.count_nonzero(counts == 1)),
        n_nonmanifold_edges=int(np.count_nonzero(counts > 2)),
        n_degenerate_faces=int(
            np.count_nonzero(mesh.face_areas() < _DEGENERATE_AREA_M2)
        ),
        n_unreferenced_vertices=int(np.count_nonzero(~referenced)),
        n_reversed_edge_pairs=reversed_pairs,
        signed_volume_m3=signed_volume_m3(mesh),
    )


def require_watertight(mesh: TriangleMesh, *, context: str) -> MeshValidity:
    """Precondition: ``mesh`` is a closed, outward-wound manifold solid.

    Args:
        mesh: Mesh to validate.
        context: What the mesh is about to be used for; quoted in the
            error so the failure is traceable.

    Returns:
        The validity report, so callers can reuse it.

    Raises:
        MeshValidationError: If any precondition fails.
    """
    report = check_mesh_validity(mesh)
    if not report.is_watertight_solid:
        raise MeshValidationError(
            f"{context}: mesh is not a watertight solid - "
            + "; ".join(report.failures())
        )
    return report
