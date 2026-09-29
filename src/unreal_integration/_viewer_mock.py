from __future__ import annotations

import logging

import numpy as np

from src.unreal_integration.data_models import (
    Quaternion,
    Vector3,
)
from src.unreal_integration.mesh_loader import LoadedMesh

from ._viewer_base import ViewerBackend, ViewerConfig

logger = logging.getLogger(__name__)


class MockBackend(ViewerBackend):
    """Mock viewer backend for testing.

    Provides a fully functional backend that doesn't require
    any external dependencies.
    """

    def __init__(self, config: ViewerConfig | None = None) -> None:
        """Initialize mock backend."""
        super().__init__(config)
        self._render_calls = 0

    def initialize(self) -> None:
        """Initialize mock backend."""
        self._is_initialized = True
        logger.debug("Mock backend initialized")

    def shutdown(self) -> None:
        """Shutdown mock backend."""
        self._is_initialized = False
        self._objects.clear()
        logger.debug("Mock backend shutdown")

    def add_mesh(
        self,
        mesh: LoadedMesh,
        name: str | None = None,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: float = 1.0,
    ) -> str:
        """Add mesh to mock scene."""
        if mesh is None:
            raise ValueError("mesh must be provided")
        if name is None:
            name = f"mock_mesh_{len(self._objects)}"

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
        """Update mock object transform."""
        if name in self._objects:
            if position is not None:
                self._objects[name]["position"] = position
            if rotation is not None:
                self._objects[name]["rotation"] = rotation
            if scale is not None:
                self._objects[name]["scale"] = scale

    def remove_object(self, name: str) -> bool:
        """Remove mock object."""
        if name is None:
            raise ValueError("name must be provided")
        if name in self._objects:
            del self._objects[name]
            return True
        return False

    def clear(self) -> None:
        """Clear mock scene."""
        self._objects.clear()

    def render(self) -> np.ndarray | None:
        """Render mock frame (returns black image)."""
        self._render_calls += 1
        # Return a simple test image
        return np.zeros((self.config.height, self.config.width, 4), dtype=np.uint8)

    @property
    def render_count(self) -> int:
        """Get number of render calls (for testing)."""
        return self._render_calls
