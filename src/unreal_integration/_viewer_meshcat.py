from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.unreal_integration.data_models import (
    Quaternion,
    Vector3,
)
from src.unreal_integration.mesh_loader import LoadedMesh

from ._viewer_base import ViewerBackend, ViewerConfig

logger = logging.getLogger(__name__)


class MeshcatBackend(ViewerBackend):
    """Meshcat-based viewer backend.

    Uses Meshcat for web-based Three.js visualization.
    Supports the existing project's Meshcat infrastructure.

    Example:
        >>> backend = MeshcatBackend()
        >>> with backend:
        ...     backend.add_mesh(mesh, name="golfer")
        ...     backend.render()
    """

    def __init__(self, config: ViewerConfig | None = None) -> None:
        """Initialize Meshcat backend.

        Args:
            config: Viewer configuration.
        """
        super().__init__(config)
        self._vis: Any = None
        self._object_counter = 0

    @property
    def _visualizer(self) -> Any:
        """Get the meshcat visualizer, asserting it's initialized."""
        if not (self._vis is not None):
            raise ValueError("Meshcat visualizer not initialized")
        return self._vis

    def initialize(self) -> None:
        """Initialize Meshcat visualizer."""
        if self._is_initialized:
            return

        try:
            import meshcat
            import meshcat.geometry as g
            import meshcat.transformations as tf

            self._vis = meshcat.Visualizer()

            # Store modules for later use
            self._geometry = g
            self._transformations = tf

            # Meshcat doesn't have direct background color setting,
            # but we store it for reference
            self._background_color = self.config.background_color

            self._is_initialized = True
            logger.info(f"Meshcat backend initialized at {self._visualizer.url()}")

        except ImportError as e:
            logger.error(f"Failed to import meshcat: {e}")
            raise RuntimeError(
                "Meshcat not available. Install with: pip install meshcat"
            ) from e

    def shutdown(self) -> None:
        """Shutdown Meshcat visualizer."""
        if self._vis is not None:
            self._vis.close()
            self._vis = None
        self._is_initialized = False
        self._objects.clear()
        logger.info("Meshcat backend shutdown")

    def add_mesh(
        self,
        mesh: LoadedMesh,
        name: str | None = None,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: float = 1.0,
    ) -> str:
        """Add mesh to Meshcat scene."""
        if mesh is None:
            raise ValueError("mesh must be provided")
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        # Generate name if not provided
        if name is None:
            name = f"mesh_{self._object_counter}"
            self._object_counter += 1

        # Convert mesh to Meshcat format
        positions, faces = mesh.to_arrays()

        # Create Meshcat geometry
        # Note: Meshcat uses TriangularMeshGeometry
        geom = self._geometry.TriangularMeshGeometry(
            vertices=positions.T,  # Meshcat expects 3xN
            faces=faces.T,  # 3xF for triangles
        )

        # Create material
        material = self._geometry.MeshLambertMaterial(
            color=0x808080,  # Default gray
            reflectivity=0.5,
        )

        # Add to scene
        self._visualizer[name].set_object(geom, material)

        # Apply transform
        self._apply_transform(name, position, rotation, scale)

        # Store reference
        self._objects[name] = {
            "mesh": mesh,
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
        """Update object transform in Meshcat."""
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        if name not in self._objects:
            logger.warning(f"Object not found: {name}")
            return

        obj = self._objects[name]
        if position is not None:
            obj["position"] = position
        if rotation is not None:
            obj["rotation"] = rotation
        if scale is not None:
            obj["scale"] = scale

        self._apply_transform(
            name,
            obj["position"],
            obj["rotation"],
            obj["scale"],
        )

    def _apply_transform(
        self,
        name: str,
        position: Vector3 | None,
        rotation: Quaternion | None,
        scale: float,
    ) -> None:
        """Apply transform to Meshcat object."""
        # Build transformation matrix
        if name is None:
            raise ValueError("name must be provided")
        T = np.eye(4)

        # Apply scale
        T[:3, :3] *= scale

        # Apply rotation (quaternion to matrix)
        if rotation is not None:
            q = rotation
            # Rotation matrix from quaternion
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

        # Apply translation
        if position is not None:
            T[:3, 3] = position.to_numpy()

        self._visualizer[name].set_transform(T)

    def remove_object(self, name: str) -> bool:
        """Remove object from Meshcat scene."""
        if name is None:
            raise ValueError("name must be provided")
        if not self._is_initialized:
            return False

        if name in self._objects:
            self._visualizer[name].delete()
            del self._objects[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all objects from Meshcat scene."""
        if not self._is_initialized:
            return

        for name in list(self._objects.keys()):
            self._visualizer[name].delete()
        self._objects.clear()

    def render(self) -> np.ndarray | None:
        """Render current Meshcat frame.

        Meshcat renders in browser, so this returns None.
        """
        # Meshcat renders automatically in browser
        return None

    @property
    def url(self) -> str | None:
        """Get Meshcat viewer URL."""
        if self._vis is not None:
            return str(self._visualizer.url())
        return None
