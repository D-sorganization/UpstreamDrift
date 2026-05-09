"""Triangle-mesh body-part shape, loaded from STL/OBJ/PLY/GLB.

:class:`MeshShape` implements the
:class:`body_part_viz.contracts.BodyPartShape` Protocol. The mesh is
re-centred on its OBB centroid before storage, decimated to a per-shape
vertex budget, and then exposed as canonical NumPy arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .._types import FittedShape
from ._mesh_decimation import DecimationStrategy, decimate
from ._mesh_io import load_mesh

__all__ = ["MeshShape"]


class MeshShape:
    """Triangle-mesh body-part shape loaded from a file.

    The mesh is stored re-centred on its oriented-bounding-box centroid;
    ``rest_dimensions`` reports the OBB extents (NOT axis-aligned bbox).
    """

    shape_id: str
    rest_dimensions: tuple[float, float, float]

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        rest_dimensions: tuple[float, float, float],
        source_path: Path | None = None,
        *,
        shape_id: str | None = None,
    ) -> None:
        if not isinstance(vertices, np.ndarray):
            raise TypeError(
                f"vertices must be numpy.ndarray; got {type(vertices).__name__}"
            )
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must have shape (V, 3); got {vertices.shape}")
        if not isinstance(faces, np.ndarray):
            raise TypeError(f"faces must be numpy.ndarray; got {type(faces).__name__}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (F, 3); got {faces.shape}")
        if len(vertices) == 0:
            raise ValueError("MeshShape requires at least one vertex")
        if len(faces) == 0:
            raise ValueError("MeshShape requires at least one face")
        if not isinstance(rest_dimensions, tuple) or len(rest_dimensions) != 3:
            raise ValueError(
                f"rest_dimensions must be a length-3 tuple; got {rest_dimensions!r}"
            )
        if any((not np.isfinite(d)) or d <= 0.0 for d in rest_dimensions):
            raise ValueError(
                f"rest_dimensions entries must be finite and > 0; "
                f"got {rest_dimensions!r}"
            )

        self._vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self._faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.rest_dimensions = (
            float(rest_dimensions[0]),
            float(rest_dimensions[1]),
            float(rest_dimensions[2]),
        )
        self.source_path = Path(source_path) if source_path is not None else None
        if shape_id is not None:
            if not isinstance(shape_id, str) or not shape_id:
                raise ValueError("shape_id must be a non-empty string")
            self.shape_id = shape_id
        elif self.source_path is not None:
            self.shape_id = f"mesh:{self.source_path.stem}"
        else:
            self.shape_id = "mesh:anonymous"

    # ---- BodyPartShape Protocol ----------------------------------------

    def vertices_at_rest(self) -> np.ndarray:
        return self._vertices

    def faces(self) -> np.ndarray:
        return self._faces

    def transform(self, fitted: FittedShape) -> np.ndarray:
        """Apply ``fitted`` per-frame transform to rest vertices.

        Returns ``(T, V, 3)`` for ``T`` frames in ``fitted`` (or ``(V, 3)``
        when ``T == 1``, to match a common single-frame caller pattern).
        """
        if fitted.shape_id != self.shape_id:
            raise ValueError(
                f"FittedShape.shape_id={fitted.shape_id!r} does not match "
                f"this shape's shape_id={self.shape_id!r}"
            )
        n_frames = fitted.centroid.shape[0]
        if n_frames == 0:
            return np.zeros((0, len(self._vertices), 3), dtype=np.float64)

        # vertices_scaled[t, v, :] = scale[t] * rest_v
        scaled = self._vertices[None, :, :] * fitted.scale[:, None, :]
        # rotated[t, v, :] = R[t] @ scaled[t, v, :]
        rotated = np.einsum("tij,tvj->tvi", fitted.rotation_matrix, scaled)
        out = rotated + fitted.centroid[:, None, :]
        if n_frames == 1:
            return out[0]
        return out

    # ---- Loader --------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        max_vertices: int = 5000,
        decimation_strategy: Literal["quadric", "uniform"] = "quadric",
    ) -> MeshShape:
        """Load a mesh from disk, decimate if needed, return a MeshShape.

        Parameters
        ----------
        path:
            Path to an STL/OBJ/PLY/GLB file.
        max_vertices:
            Strict per-shape vertex budget (must be >= 4).
        decimation_strategy:
            ``"quadric"`` (default) or ``"uniform"``.

        Raises
        ------
        FileNotFoundError, ValueError
            See :func:`._mesh_io.load_mesh`.
        """
        if max_vertices < 4:
            raise ValueError(f"max_vertices must be >= 4; got {max_vertices}")

        loaded = load_mesh(path)
        verts = loaded.vertices
        faces = loaded.faces

        if len(verts) > max_vertices:
            verts, faces = decimate(
                verts,
                faces,
                max_vertices=max_vertices,
                strategy=_check_strategy(decimation_strategy),
            )

        # Re-centre on OBB centroid (computed in input frame).
        verts = verts - loaded.obb_centroid[None, :]

        return cls(
            vertices=verts,
            faces=faces,
            rest_dimensions=loaded.obb_extents,
            source_path=Path(path),
        )


def _check_strategy(value: str) -> DecimationStrategy:
    if value not in ("quadric", "uniform"):
        raise ValueError(
            f"decimation_strategy must be 'quadric' or 'uniform'; got {value!r}"
        )
    return value  # type: ignore[return-value]
