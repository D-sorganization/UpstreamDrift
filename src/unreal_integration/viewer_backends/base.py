from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.unreal_integration.data_models import Quaternion, Vector3
from src.unreal_integration.mesh_loader import LoadedMesh

from .config import CameraState, LightState, ViewerConfig


class ViewerBackend(ABC):
    """Abstract base class for viewer backends.

    All visualization backends must implement this interface
    to ensure consistent behavior across different rendering
    technologies.

    Design by Contract:
        Preconditions:
            - initialize() must be called before other methods
            - add_mesh requires valid mesh data

        Postconditions:
            - After render(), display should be updated
            - After clear(), scene should be empty

        Invariants:
            - is_initialized reflects actual state
    """

    def __init__(self, config: ViewerConfig | None = None) -> None:
        """Initialize backend.

        Args:
            config: Viewer configuration.
        """
        self.config = config or ViewerConfig()
        self._is_initialized = False
        self._objects: dict[str, Any] = {}
        self._camera = CameraState()
        self._lights: list[LightState] = [LightState()]

    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._is_initialized

    @property
    def object_count(self) -> int:
        """Get number of objects in scene."""
        return len(self._objects)

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend.

        Must be called before using other methods.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown and cleanup backend resources."""

    @abstractmethod
    def add_mesh(
        self,
        mesh: LoadedMesh,
        name: str | None = None,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: float = 1.0,
    ) -> str:
        """Add mesh to scene.

        Args:
            mesh: Loaded mesh data.
            name: Optional name for the mesh (auto-generated if None).
            position: Initial position.
            rotation: Initial rotation.
            scale: Initial scale.

        Returns:
            Name/ID of the added mesh.
        """

    @abstractmethod
    def update_transform(
        self,
        name: str,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: float | None = None,
    ) -> None:
        """Update object transform.

        Args:
            name: Object name.
            position: New position (optional).
            rotation: New rotation (optional).
            scale: New scale (optional).
        """

    @abstractmethod
    def remove_object(self, name: str) -> bool:
        """Remove object from scene.

        Args:
            name: Object name to remove.

        Returns:
            True if object was removed, False if not found.
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all objects from scene."""

    @abstractmethod
    def render(self) -> np.ndarray | None:
        """Render current frame.

        Returns:
            Rendered image as numpy array (RGBA), or None if not applicable.
        """

    def set_camera(
        self,
        position: Vector3 | None = None,
        target: Vector3 | None = None,
        fov: float | None = None,
    ) -> None:
        """Set camera parameters.

        Args:
            position: Camera position.
            target: Look-at target.
            fov: Field of view.
        """
        if position is not None:
            self._camera.position = position
        if target is not None:
            self._camera.target = target
        if fov is not None:
            self._camera.fov = fov

    def add_light(self, light: LightState) -> None:
        """Add light to scene.

        Args:
            light: Light configuration.
        """
        self._lights.append(light)

    def clear_lights(self) -> None:
        """Remove all lights."""
        self._lights.clear()

    def get_object_names(self) -> list[str]:
        """Get list of all object names in scene.

        Returns:
            List of object names.
        """
        return list(self._objects.keys())

    def __enter__(self) -> "ViewerBackend":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.shutdown()
