"""Mesh file I/O for :mod:`body_part_viz.shapes`.

Loaders accept STL (ascii + binary), OBJ (vertex-only; mtl ignored),
PLY (binary + ascii), and GLB (first mesh node only). The ``.gltf``
extension is rejected with a helpful message.

The trimesh dependency is encapsulated here: callers receive plain
NumPy arrays plus an OBB-extents tuple, never a live trimesh object.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
try:
    import trimesh
    from trimesh.bounds import oriented_bounds
except ImportError:
    trimesh = None
    oriented_bounds = None


__all__ = ["LoadedMesh", "load_mesh"]


_SUPPORTED_EXTS = frozenset({".stl", ".obj", ".ply", ".glb"})


class LoadedMesh:
    """Plain container for a loaded mesh.

    Attributes
    ----------
    vertices:
        ``(V, 3)`` float64 array.
    faces:
        ``(F, 3)`` int64 triangle indices.
    obb_extents:
        Oriented-bounding-box extents (length-3 tuple of floats).
    obb_centroid:
        ``(3,)`` float64 OBB centroid (in the mesh's input frame).
    """

    __slots__ = ("vertices", "faces", "obb_extents", "obb_centroid")

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        obb_extents: tuple[float, float, float],
        obb_centroid: np.ndarray,
    ) -> None:
        self.vertices = vertices
        self.faces = faces
        self.obb_extents = obb_extents
        self.obb_centroid = obb_centroid


def _extract_first_mesh(loaded: object):
    if trimesh is None:
        raise ImportError('trimesh is required for mesh IO.')
    """Return the first triangle mesh from a trimesh load result.

    GLB files load as a Scene; we accept the first mesh node only.
    """
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("file contains no triangle mesh")
        return meshes[0]
    raise ValueError(f"unsupported mesh container: {type(loaded).__name__}")


def load_mesh(path: Path | str) -> LoadedMesh:
    """Load a mesh from disk and return canonical NumPy arrays.

    Parameters
    ----------
    path:
        Filesystem path to an ``.stl``, ``.obj``, ``.ply``, or ``.glb`` file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        On ``.gltf`` (use ``.glb``), unsupported extension, or malformed
        contents.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".gltf":
        raise ValueError(
            "GLTF files are not supported; export as .glb (single binary file)."
        )
    if suffix not in _SUPPORTED_EXTS:
        raise ValueError(
            f"unsupported mesh extension {suffix!r}; "
            f"expected one of {sorted(_SUPPORTED_EXTS)}"
        )
    if not p.exists():
        raise FileNotFoundError(f"mesh file not found: {p}")

    if trimesh is None:
        raise ImportError('trimesh is required for mesh IO.')
    try:
        loaded = trimesh.load(str(p), force=None, process=False)
    except Exception as exc:  # noqa: BLE001 — trimesh raises diverse types
        raise ValueError(f"failed to read {p}: {exc}") from exc

    mesh = _extract_first_mesh(loaded)

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"mesh in {p} has no vertices")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"mesh in {p} has no faces")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    _, extents = oriented_bounds(mesh)
    extents_tuple = (
        float(extents[0]),
        float(extents[1]),
        float(extents[2]),
    )
    centroid = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5

    return LoadedMesh(
        vertices=vertices,
        faces=faces,
        obb_extents=extents_tuple,
        obb_centroid=centroid.astype(np.float64),
    )
