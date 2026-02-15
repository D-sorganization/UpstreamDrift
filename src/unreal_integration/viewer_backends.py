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

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Empty, Queue
from typing import Any

import numpy as np

from src.unreal_integration.data_models import (
    BallState,
    ClubState,
    JointState,
    Quaternion,
    UnrealDataFrame,
    Vector3,
)
from src.unreal_integration.mesh_loader import LoadedMesh
from src.unreal_integration.streaming import StreamingConfig, UnrealStreamingServer

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
        """Initialize PyVista backend.

        Args:
            config: Viewer configuration.
        """
        super().__init__(config)
        self._plotter: Any = None
        self._object_counter = 0

    @property
    def plotter(self) -> Any:
        """Get the PyVista plotter, asserting it's initialized."""
        assert self._plotter is not None, "PyVista plotter not initialized"
        return self._plotter

    def initialize(self) -> None:
        """Initialize PyVista plotter."""
        if self._is_initialized:
            return

        try:
            import pyvista as pv

            # Initialize plotter with config
            self._plotter = pv.Plotter(
                off_screen=True,  # Default to off-screen for safety
                window_size=(self.config.width, self.config.height),
            )
            self._plotter.background_color = self.config.background_color

            # Configure camera
            self._plotter.camera.position = self._camera.position.to_numpy()
            self._plotter.camera.focal_point = self._camera.target.to_numpy()
            self._plotter.camera.up = self._camera.up.to_numpy()
            self._plotter.camera.view_angle = self._camera.fov

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

        import pyvista as pv

        # Generate name if not provided
        if name is None:
            name = f"mesh_{self._object_counter}"
            self._object_counter += 1

        # Convert mesh to PyVista format
        positions, faces = mesh.to_arrays()

        # PyVista expects faces as [n_points, p1, p2, ..., n_points, ...]
        # We assume all faces are triangles for now (MeshLoader handles this usually)
        # Flatten and prepend vertex count
        pv_faces = []
        for face in faces:
            pv_faces.append(len(face))
            pv_faces.extend(face)

        # Create PolyData
        poly_data = pv.PolyData(positions, np.array(pv_faces))

        # Add to plotter
        actor = self._plotter.add_mesh(poly_data, name=name)

        # Apply initial transform
        self._apply_transform(actor, position, rotation, scale)

        # Store reference
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
            logger.warning(f"Object not found: {name}")
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

        # Update actor matrix
        actor.user_matrix = T

    def remove_object(self, name: str) -> bool:
        """Remove object from PyVista scene."""
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

        # Update camera before render
        self._plotter.camera.position = self._camera.position.to_numpy()
        self._plotter.camera.focal_point = self._camera.target.to_numpy()
        self._plotter.camera.up = self._camera.up.to_numpy()

        # Render
        # If off_screen is True, this returns None or updates internal buffer
        # We can use screenshot() to get the array
        self._plotter.render()

        # Capture image
        # return_img=True returns numpy array
        image = self._plotter.screenshot(return_img=True)
        return image


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
        return UnrealBridgeBackend(config)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


class UnrealBridgeBackend(ViewerBackend):
    """Unreal Engine bridge viewer backend.

    Streams simulation data to Unreal Engine via WebSocket.
    Runs the streaming server in a background thread to maintain
    responsiveness of the main simulation loop.

    Example:
        >>> backend = UnrealBridgeBackend()
        >>> with backend:
        ...     backend.add_mesh(mesh, name="club")
        ...     backend.render()
    """

    def __init__(self, config: ViewerConfig | None = None) -> None:
        """Initialize Unreal Bridge backend.

        Args:
            config: Viewer configuration.
        """
        super().__init__(config)
        self._server: UnrealStreamingServer | None = None
        self._server_thread: threading.Thread | None = None
        self._frame_queue: Queue[UnrealDataFrame] = Queue()
        self._stop_event = threading.Event()
        self._frame_counter = 0
        self._object_counter = 0
        self._start_time = 0.0

    def initialize(self) -> None:
        """Initialize streaming server in background thread."""
        if self._is_initialized:
            return

        self._start_time = time.time()

        # Configure streaming
        streaming_config = StreamingConfig(
            host=self.config.server_host,
            port=self.config.server_port,
        )
        self._server = UnrealStreamingServer(config=streaming_config)

        # Start background thread
        self._stop_event.clear()
        self._server_thread = threading.Thread(
            target=self._run_server_loop, daemon=True, name="UnrealBridgeThread"
        )
        self._server_thread.start()

        # Wait a bit for server to start (not strictly necessary but safer)
        time.sleep(0.1)

        self._is_initialized = True
        logger.info(
            f"Unreal Bridge backend initialized on port {self.config.server_port}"
        )

    def _run_server_loop(self) -> None:
        """Run the asyncio event loop for the server."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run() -> None:
            if self._server is None:
                return

            # Start server
            await self._server.start()

            # Process frames
            while not self._stop_event.is_set():
                try:
                    # Non-blocking check for new frames
                    # We check frequently but sleep briefly to avoid busy loop
                    try:
                        frame = self._frame_queue.get_nowait()
                        await self._server.broadcast(frame)
                        self._frame_queue.task_done()
                    except Empty:
                        await asyncio.sleep(0.001)

                except Exception as e:
                    logger.error(f"Error in streaming loop: {e}")
                    await asyncio.sleep(1.0)  # Backoff on error

            # Stop server
            await self._server.stop()

        try:
            loop.run_until_complete(run())
        finally:
            loop.close()

    def shutdown(self) -> None:
        """Shutdown streaming server."""
        self._stop_event.set()
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            if self._server_thread.is_alive():
                logger.warning("Unreal Bridge thread did not stop cleanly")
            self._server_thread = None

        self._server = None
        self._is_initialized = False
        self._objects.clear()
        logger.info("Unreal Bridge backend shutdown")

    def add_mesh(
        self,
        mesh: LoadedMesh,
        name: str | None = None,
        position: Vector3 | None = None,
        rotation: Quaternion | None = None,
        scale: float = 1.0,
    ) -> str:
        """Add mesh to tracked objects."""
        if not self._is_initialized:
            raise RuntimeError("Backend not initialized")

        if name is None:
            name = f"mesh_{self._object_counter}"
            self._object_counter += 1

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
        """Update object transform."""
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

    def remove_object(self, name: str) -> bool:
        """Remove object."""
        if not self._is_initialized:
            return False

        if name in self._objects:
            del self._objects[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all objects."""
        self._objects.clear()

    def render(self) -> np.ndarray | None:
        """Queue current frame for streaming."""
        if not self._is_initialized:
            return None

        # Construct UnrealDataFrame
        timestamp = time.time() - self._start_time

        joints: dict[str, JointState] = {}
        club: ClubState | None = None
        ball: BallState | None = None

        for name, obj in self._objects.items():
            pos = obj["position"]
            rot = obj["rotation"]

            # Map known names to specific fields
            if name == "club":
                club = ClubState(
                    head_position=pos,
                    # We don't track velocity/acceleration in ViewerBackend currently
                    head_velocity=Vector3.zero(),
                )
            elif name == "ball":
                ball = BallState(
                    position=pos,
                    velocity=Vector3.zero(),
                )
            else:
                # Map everything else to joints
                joints[name] = JointState(
                    name=name,
                    position=pos,
                    rotation=rot,
                )

        frame = UnrealDataFrame(
            timestamp=timestamp,
            frame_number=self._frame_counter,
            joints=joints,
            club=club,
            ball=ball,
        )
        self._frame_counter += 1

        # Queue for sending
        self._frame_queue.put(frame)

        # No image return for streaming backend
        return None
