from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from src.unreal_integration.data_models import Vector3


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
    def from_dict(cls, d: dict[str, Any]) -> "ViewerConfig":
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
