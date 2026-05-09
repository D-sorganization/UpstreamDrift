"""Custom-mesh marker primitive.

Accepts either a pre-built :class:`CustomMeshSpec` or a filesystem path
to an STL / OBJ / PLY / GLB asset (delegated to
:func:`body_part_viz.shapes._mesh_io.load_mesh` — DRY).

The supplied geometry is *normalised* so its bounding sphere has
radius 1: the centroid (mid-AABB) is translated to the origin and the
mesh is divided by the maximum distance from the origin. The
``mesh()`` method then scales by ``size_px / 2``.

The :attr:`MarkerStyle.shape` of the style passed to :meth:`mesh` must
be :attr:`MarkerShape.CUSTOM_MESH` *and* its ``custom_mesh`` field must
be the same :class:`CustomMeshSpec` this marker was constructed with —
this enforces the contract documented on :class:`MarkerStyle`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...body_part_viz.shapes._mesh_io import load_mesh
from ..markers import CustomMeshSpec, MarkerShape, MarkerStyle

__all__ = ["CustomMeshMarker"]


def _normalise_to_unit_radius(
    vertices: np.ndarray,
) -> np.ndarray:
    """Translate to AABB centroid and rescale so the bounding sphere is 1."""
    centroid = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    centred = vertices - centroid
    radii = np.linalg.norm(centred, axis=1)
    max_radius = float(radii.max())
    if max_radius <= 0.0:
        raise ValueError("custom mesh has zero extent; cannot normalise to unit radius")
    return (centred / max_radius).astype(np.float64)


class CustomMeshMarker:
    """Custom-mesh marker shape renderer.

    Construct from either:

    * a :class:`CustomMeshSpec` directly, or
    * a :class:`pathlib.Path` (or string) pointing to an STL / OBJ /
      PLY / GLB asset — see :func:`body_part_viz.shapes._mesh_io.load_mesh`.
    """

    shape_id: str = MarkerShape.CUSTOM_MESH.value

    def __init__(
        self,
        source: CustomMeshSpec | Path | str,
        *,
        name: str | None = None,
    ) -> None:
        if isinstance(source, CustomMeshSpec):
            spec = source
        elif isinstance(source, (Path, str)):
            loaded = load_mesh(source)
            mesh_name = name or Path(source).stem
            spec = CustomMeshSpec(
                name=mesh_name,
                vertices=loaded.vertices,
                faces=loaded.faces,
            )
        else:
            raise TypeError(
                "source must be CustomMeshSpec, pathlib.Path, or str; "
                f"got {type(source).__name__}"
            )

        self._spec = spec
        self._unit_vertices = _normalise_to_unit_radius(spec.vertices)
        self._faces = spec.faces.astype(np.int64, copy=True)

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def spec(self) -> CustomMeshSpec:
        """Return the (possibly file-loaded) spec held by this marker."""
        return self._spec

    def mesh(self, style: MarkerStyle) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        if style.shape is not MarkerShape.CUSTOM_MESH:
            raise ValueError(
                "CustomMeshMarker requires style.shape == CUSTOM_MESH; "
                f"got {style.shape}"
            )
        radius = float(style.size_px) / 2.0
        return self._unit_vertices * radius, self._faces.copy()
