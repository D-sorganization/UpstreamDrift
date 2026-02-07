"""Unified viewer backend abstraction for visualization.

This module provides a common interface for different visualization
backends, allowing seamless switching between Meshcat, PyVista,
and future game engine integrations.

Design by Contract:
    - All backends implement the same ViewerBackend protocol
    - Backends handle their own initialization and cleanup
    - State is managed consistently across backends

Backends:
    - MeshcatBackend: Web-based Three.js visualization
    - PyVistaBackend: Desktop VTK-based visualization (future)
    - UnrealBridgeBackend: Unreal Engine connection (future)

Usage:
    from src.unreal_integration.viewer_backends import (
        ViewerBackend,
        MeshcatBackend,
        create_viewer,
    )

    viewer = create_viewer("meshcat")
    viewer.add_mesh(mesh_data)
    viewer.render()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from src.unreal_integration.data_models import Quaternion, Vector3
from src.unreal_integration.mesh_loader import LoadedMesh

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available viewer backend types."""

    MESHCAT = auto()
    PYVISTA = auto()
    UNREAL_BRIDGE = auto()
    MOCK = auto()  # For testing


@dataclass
class ViewerConfig:
    """Configuration for viewer backend.

    Attributes:
        backend_type: Type of backend to use.
        width: Viewport width.
        height: Viewport height.
        background_color: Background color (RGB).
        enable_shadows: Whether to enable shadows.
        enable_antialiasing: Whether to enable antialiasing.
        fov: Field of view in degrees.
        near_clip: Near clipping plane.
        far_clip: Far clipping plane.
        server_host: Host for web-based backends.
        server_port: Port for web-based backends.
    """

    backend_type: BackendType = BackendType.MESHCAT
    width: int = 1280
    height: int = 720
    background_color: tuple[float, float, float] = (0.1, 0.1, 0.1)
    enable_shadows: bool = True
    enable_antialiasing: bool = True
    fov: float = 45.0
    near_clip: float = 0.01
    far_clip: float = 1000.0
    server_host: str = "localhost"
    server_port: int = 7000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend_type": self.backend_type.name.lower(),
            "width": self.width,
            "height": self.height,
            "background_color": list(self.background_color),
            "enable_shadows": self.enable_shadows,
            "enable_antialiasing": self.enable_antialiasing,
            "fov": self.fov,
            "near_clip": self.near_clip,
            "far_clip": self.far_clip,
            "server_host": self.server_host,
            "server_port": self.server_port,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ViewerConfig:
        """Create from dictionary."""
        return cls(
            backend_type=BackendType[d.get("backend_type", "meshcat").upper()],
            width=d.get("width", 1280),
            height=d.get("height", 720),
            background_color=tuple(d.get("background_color", [0.1, 0.1, 0.1])),
            enable_shadows=d.get("enable_shadows", True),
            enable_antialiasing=d.get("enable_antialiasing", True),
            fov=d.get("fov", 45.0),
            near_clip=d.get("near_clip", 0.01),
            far_clip=d.get("far_clip", 1000.0),
            server_host=d.get("server_host", "localhost"),
            server_port=d.get("server_port", 7000),
        )


@dataclass
class CameraState:
    """Camera state for viewer.

    Attributes:
        position: Camera position.
        target: Look-at target.
        up: Up vector.
        fov: Field of view.
    """

    position: Vector3 = field(default_factory=lambda: Vector3(x=3.0, y=3.0, z=2.0))
    target: Vector3 = field(default_factory=Vector3.zero)
    up: Vector3 = field(default_factory=lambda: Vector3(x=0.0, y=0.0, z=1.0))
    fov: float = 45.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "position": self.position.to_dict(),
            "target": self.target.to_dict(),
            "up": self.up.to_dict(),
            "fov": self.fov,
        }


@dataclass
class LightState:
    """Light configuration for viewer.

    Attributes:
        light_type: Type of light ("directional", "point", "ambient").
        position: Light position (for point lights).
        direction: Light direction (for directional lights).
        color: Light color (RGB).
        intensity: Light intensity.
        cast_shadows: Whether light casts shadows.
    """

    light_type: str = "directional"
    position: Vector3 = field(default_factory=lambda: Vector3(x=5.0, y=5.0, z=5.0))
    direction: Vector3 = field(default_factory=lambda: Vector3(x=-1.0, y=-1.0, z=-1.0))
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    cast_shadows: bool = True


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

    def __init__(self, config: ViewerConfig | None = None):
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

    def __enter__(self) -> ViewerBackend:
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.shutdown()


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

    def __init__(self, config: ViewerConfig | None = None):
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
        assert self._vis is not None, "Meshcat visualizer not initialized"
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


class PyVistaBackend(ViewerBackend):
    """PyVista-based viewer backend.

    Uses PyVista (VTK) for desktop visualization.
    """

    def __init__(self, config: ViewerConfig | None = None):
        """Initialize PyVista backend."""
        super().__init__(config)
        self._plotter: Any = None
        self._actors: dict[str, Any] = {}

    def initialize(self) -> None:
        """Initialize PyVista plotter."""
        if self._is_initialized:
            return

        try:
            import pyvista as pv

            self._plotter = pv.Plotter(
                window_size=(self.config.width, self.config.height),
                off_screen=False,  # Can be configurable
            )
            self._plotter.set_background(self.config.background_color)

            if self.config.enable_antialiasing:
                self._plotter.enable_anti_aliasing()

            if self.config.enable_shadows:
                self._plotter.enable_shadows()

            self._is_initialized = True
            logger.info("PyVista backend initialized")

        except ImportError as e:
            logger.error(f"Failed to import pyvista: {e}")
            raise RuntimeError(
                "PyVista not available. Install with: pip install pyvista"
            ) from e

    def shutdown(self) -> None:
        """Shutdown PyVista plotter."""
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
        self._is_initialized = False
        self._actors.clear()
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
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        if name is None:
            name = f"mesh_{len(self._objects)}"

        # Convert to PyVista mesh
        import pyvista as pv

        vertices, faces = mesh.to_arrays()
        # PyVista faces format: [n_nodes, node1, node2, ..., n_nodes, node1, ...]
        # Assuming triangles
        n_faces = faces.shape[1]
        pv_faces = np.column_stack((np.full(n_faces, 3), faces.T)).flatten()

        pv_mesh = pv.PolyData(vertices.T, pv_faces)

        # Add to plotter
        actor = self._plotter.add_mesh(pv_mesh, name=name)

        self._actors[name] = actor
        self._objects[name] = {
            "mesh": mesh,
            "position": position or Vector3.zero(),
            "rotation": rotation or Quaternion.identity(),
            "scale": scale,
        }

        self.update_transform(name, position, rotation, scale)

        return name

    def update_transform(
        self,
        name: str,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: float | None = None,
    ) -> None:
        """Update object transform in PyVista."""
        if not self._is_initialized or name not in self._actors:
            return

        actor = self._actors[name]
        obj = self._objects[name]

        if position is not None:
            obj["position"] = position
        if rotation is not None:
            obj["rotation"] = rotation
        if scale is not None:
            obj["scale"] = scale

        # Apply transform
        # PyVista actors have user_matrix property
        T = np.eye(4)

        # Scale
        T[:3, :3] *= obj["scale"]

        # Rotation
        q = obj["rotation"]
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

        # Translation
        T[:3, 3] = obj["position"].to_numpy()

        actor.user_matrix = T

    def remove_object(self, name: str) -> bool:
        """Remove object from PyVista scene."""
        if not self._is_initialized:
            return False

        if name in self._actors:
            self._plotter.remove_actor(self._actors[name])
            del self._actors[name]
            del self._objects[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all objects from PyVista scene."""
        if not self._is_initialized:
            return

        self._plotter.clear()
        self._actors.clear()
        self._objects.clear()

    def render(self) -> np.ndarray | None:
        """Render current frame."""
        if not self._is_initialized:
            return None

        # If off_screen is False, this updates the window
        # If off_screen is True, we can capture the image
        self._plotter.render()

        # Return image if possible (simplified)
        try:
            return self._plotter.screenshot(return_img=True)
        except Exception:
            return None


class MockBackend(ViewerBackend):
    """Mock viewer backend for testing.

    Provides a fully functional backend that doesn't require
    any external dependencies.
    """

    def __init__(self, config: ViewerConfig | None = None):
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


def create_viewer(
    backend_type: str | BackendType = "meshcat",
    config: ViewerConfig | None = None,
) -> ViewerBackend:
    """Factory function to create viewer backend.

    Args:
        backend_type: Type of backend ("meshcat", "pyvista", "mock").
        config: Viewer configuration.

    Returns:
        Appropriate ViewerBackend instance.

    Raises:
        ValueError: If backend type is not supported.
    """
    if isinstance(backend_type, str):
        try:
            backend_type = BackendType[backend_type.upper()]
        except KeyError as e:
            raise ValueError(f"Unknown backend type: {backend_type}") from e

    if config is None:
        config = ViewerConfig(backend_type=backend_type)

    if backend_type == BackendType.MESHCAT:
        return MeshcatBackend(config)
    elif backend_type == BackendType.MOCK:
        return MockBackend(config)
    elif backend_type == BackendType.PYVISTA:
        return PyVistaBackend(config)
    elif backend_type == BackendType.UNREAL_BRIDGE:
        raise NotImplementedError("Unreal Bridge backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
