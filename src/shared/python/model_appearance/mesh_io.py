"""Binary STL writer for visual meshes (metres in, metres out)."""

from __future__ import annotations

import numpy as np

from src.shared.python.model_appearance.geometry import Mesh

_RECORD = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
HEADER_BYTES = 80


def stl_bytes(mesh: Mesh, header: str = "UpstreamDrift visual mesh; units=m") -> bytes:
    """Serialize ``mesh`` as a deterministic binary STL.

    Preconditions: finite vertices, faces index into them. Facet normals are
    recomputed from the winding. Postcondition: ``84 + 50 * n_faces`` bytes.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not np.isfinite(vertices).all():
        raise ValueError("mesh vertices must be finite")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
        raise ValueError("mesh faces must be a non-empty (n, 3) array")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("mesh faces index outside the vertex array")
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1, keepdims=True)
    records = np.zeros(len(faces), dtype=_RECORD)
    records["n"] = np.divide(cross, norm, out=np.zeros_like(cross), where=norm > 0)
    records["v"] = tri
    head = header.encode("ascii", "replace")[:HEADER_BYTES].ljust(HEADER_BYTES, b" ")
    return head + np.uint32(len(faces)).tobytes() + records.tobytes()


def read_stl_bytes(raw: bytes) -> Mesh:
    """Parse a binary STL into an unwelded :class:`Mesh` (3 vertices per face)."""
    if len(raw) < 84:
        raise ValueError("STL is too short")
    count = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    if len(raw) != 84 + count * _RECORD.itemsize:
        raise ValueError("not a binary STL of the declared triangle count")
    tri = np.frombuffer(raw, dtype=_RECORD, count=count, offset=84)["v"]
    return Mesh(
        np.asarray(tri, float).reshape(-1, 3),
        np.arange(3 * count, dtype=np.int64).reshape(-1, 3),
    )
