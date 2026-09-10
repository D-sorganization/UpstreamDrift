"""Scene references rendered through the suite's existing native mesh protocol."""

from math import isfinite
from typing import Literal

import numpy as np

from src.motion_capture.reference.registration import canonical_z_up_to_adr0041_world
from src.shared.python.visualization.fsp_renderer import Viewport

from .geometry import ReferenceGeometry, ReferencePlane, ReferencePoint

POINT_GLYPH_RADIUS_M = 0.015
_PLANE_FACES = np.array(((0, 1, 2), (0, 2, 3)), dtype=int)
_POINT_VERTICES = np.array(
    ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)), dtype=float
)
_POINT_FACES = np.array(
    (
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    ),
    dtype=int,
)


class NativeGeometryRenderer:
    """Own reference mesh handles without changing a viewport's other objects."""

    def __init__(
        self,
        viewport: Viewport,
        *,
        scene_id: str,
        frame: Literal["world_Zup", "adr0041_y_up"],
    ) -> None:
        if not isinstance(viewport, Viewport):
            raise TypeError("Viewport must support add_mesh and remove_mesh")
        if not scene_id or frame not in {"world_Zup", "adr0041_y_up"}:
            raise ValueError(
                "Native reference geometry requires an explicit scene and frame"
            )
        self.viewport, self.scene_id, self.frame = viewport, scene_id, frame
        self._handles: list[object] = []

    def _mesh(self, item: ReferencePlane | ReferencePoint) -> object:
        if isinstance(item, ReferencePlane):
            vertices, faces = item.vertices(), _PLANE_FACES
        else:
            vertices = _POINT_VERTICES * POINT_GLYPH_RADIUS_M + item.position_m
            faces = _POINT_FACES
        if self.frame == "world_Zup":
            # Inverse of the same proper frame conversion used by reference playback.
            vertices = vertices @ canonical_z_up_to_adr0041_world(np.eye(3)).T
        colour = tuple(
            int(item.colour[index : index + 2], 16) / 255 for index in (1, 3, 5)
        )
        return self.viewport.add_mesh(
            vertices.copy(), faces.copy(), color=colour, alpha=item.opacity
        )

    def render(self, geometry: ReferenceGeometry, scene_time: float) -> None:
        """Replace visible references; a failed submission preserves prior meshes."""
        if geometry.scene_id != self.scene_id:
            raise ValueError("Native reference geometry belongs to another scene")
        if not isfinite(scene_time):
            raise ValueError("Scene time must be finite")
        items: tuple[ReferencePlane | ReferencePoint, ...] = (
            *geometry.planes,
            *geometry.points,
        )
        added = []
        try:
            for item in items:
                if item.at(scene_time) and item.opacity > 0:
                    added.append(self._mesh(item))
        except (RuntimeError, ValueError, TypeError, OSError):
            for handle in added:
                self.viewport.remove_mesh(handle)
            raise
        self.clear()
        self._handles = added

    def clear(self) -> None:
        """Release only meshes owned by this renderer; safe when already empty."""
        for handle in self._handles:
            self.viewport.remove_mesh(handle)
        self._handles = []
