"""PyVista-based viewer backend implementation.

Extracted from viewer_backends.py to keep that module under the 1200 LOC budget.
Import via :func:`viewer_backends.create_viewer` rather than directly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .data_models import Quaternion, Vector3
from .mesh_loader import LoadedMesh
from ._viewer_base import ViewerBackend, ViewerConfig

logger = logging.getLogger(__name__)


class PyVistaBackend(ViewerBackend):
    """PyVista-based viewer backend.

    Uses PyVista (VTK) for high-performance desktop visualization.
    Suitable for offline rendering and high-quality screenshots.

    Example:
        >>> backend = PyVistaBackend()
        >>> with backend:
        ...     backend.add_mesh(mesh, name="golfer")
        ...     backend.render()
    """

    def __init__(self, config: ViewerConfig | None = None) -> None:
        """Initialize PyVista backend."""
        super().__init__(config)
        self._plotter: Any = None
        self._object_counter = 0

    @property
    def plotter(self) -> Any:
        """Get the PyVista plotter, asserting it's initialized."""
        if not (self._plotter is not None):
            raise ValueError("PyVista plotter not initialized")
        return self._plotter

    def initialize(self) -> None:
        """Initialize PyVista plotter."""
        if self._is_initialized:
            return

        try:
            import pyvista as pv

            self._plotter = pv.Plotter(
                off_screen=True,
                window_size=(self.config.width, self.config.height),
            )
            self._plotter.background_color = self.config.background_color

            self._plotter.camera.position = self._camera.position.to_numpy()
            self._plotter.camera.focal_point = self._camera.target.to_numpy()
            self._plotter.camera.up = self._camera.up.to_numpy()
            self._plotter.camera.view_angle = self._camera.fov

            self._is_initialized = True
            logger.info("PyVista backend initialized")

        except ImportError as e:
            logger.error("Failed to import pyvista: %s", e)
            raise RuntimeError(
                "PyVista not available. Install with: pip install pyvista"
            ) from e

    def shutdown(self) -> None:
        """Shutdown PyVista plotter."""
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
        self._is_initialized = False
        self._objects.clear()
        logger.info("PyVista backend shutdown")

    def add_mesh(
        self,
        mesh: LoadedMesh,
        name: str | None = None,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: float = 1.0,
    ) -> str:
        """Add mesh to PyVista scene."""
        if not (mesh is not None):
            raise ValueError("mesh must be provided")
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        import pyvista as pv

        if name is None:
            name = f"mesh_{self._object_counter}"
            self._object_counter += 1

        positions, faces = mesh.to_arrays()

        pv_faces = []
        for face in faces:
            pv_faces.append(len(face))
            pv_faces.extend(face)

        poly_data = pv.PolyData(positions, np.array(pv_faces))
        actor = self._plotter.add_mesh(poly_data, name=name)
        self._apply_transform(actor, position, rotation, scale)

        self._objects[name] = {
            "mesh": mesh,
            "actor": actor,
            "position": position or Vector3.zero(),
            "rotation": rotation or Quaternion.identity(),
            "scale": scale,
        }

        return name

    def update_transform(
        self,
        name: str,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: float | None = None,
    ) -> None:
        """Update object transform in PyVista."""
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        if name not in self._objects:
            logger.warning("Object not found: %s", name)
            return

        obj = self._objects[name]
        actor = obj["actor"]

        if position is not None:
            obj["position"] = position
        if rotation is not None:
            obj["rotation"] = rotation
        if scale is not None:
            obj["scale"] = scale

        self._apply_transform(
            actor,
            obj["position"],
            obj["rotation"],
            obj["scale"],
        )

    def _apply_transform(
        self,
        actor: Any,
        position: Vector3 | None,
        rotation: Quaternion | None,
        scale: float,
    ) -> None:
        """Apply transform to PyVista actor."""
        if not (scale is not None):
            raise ValueError("scale must be provided")
        T = np.eye(4)
        T[:3, :3] *= scale

        if rotation is not None:
            q = rotation
            rot = np.array(
                [
                    [
                        1 - 2 * q.y * q.y - 2 * q.z * q.z,
                        2 * q.x * q.y - 2 * q.z * q.w,
                        2 * q.x * q.z + 2 * q.y * q.w,
                    ],
                    [
                        2 * q.x * q.y + 2 * q.z * q.w,
                        1 - 2 * q.x * q.x - 2 * q.z * q.z,
                        2 * q.y * q.z - 2 * q.x * q.w,
                    ],
                    [
                        2 * q.x * q.z - 2 * q.y * q.w,
                        2 * q.y * q.z + 2 * q.x * q.w,
                        1 - 2 * q.x * q.x - 2 * q.y * q.y,
                    ],
                ]
            )
            T[:3, :3] = rot @ T[:3, :3]

        if position is not None:
            T[:3, 3] = position.to_numpy()

        actor.user_matrix = T

    def remove_object(self, name: str) -> bool:
        """Remove object from PyVista scene."""
        if not (name is not None):
            raise ValueError("name must be provided")
        if not self._is_initialized:
            return False

        if name in self._objects:
            self._plotter.remove_actor(name)
            del self._objects[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all objects from PyVista scene."""
        if not self._is_initialized:
            return

        self._plotter.clear()
        self._objects.clear()

    def render(self) -> np.ndarray | None:
        """Render current PyVista frame.

        Returns:
            Rendered image as numpy array (RGBA).
        """
        if not self._is_initialized:
            return None

        self._plotter.camera.position = self._camera.position.to_numpy()
        self._plotter.camera.focal_point = self._camera.target.to_numpy()
        self._plotter.camera.up = self._camera.up.to_numpy()

        self._plotter.render()
        image = self._plotter.screenshot(return_img=True)
        return image
