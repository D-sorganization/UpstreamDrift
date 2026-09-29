from .base import ViewerBackend
from .config import BackendType, ViewerConfig
from .meshcat_backend import MeshcatBackend
from .mock_backend import MockBackend
from .pyvista_backend import PyVistaBackend
from .unreal_backend import UnrealBridgeBackend


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
    if backend_type == BackendType.MOCK:
        return MockBackend(config)
    if backend_type == BackendType.PYVISTA:
        return PyVistaBackend(config)
    if backend_type == BackendType.UNREAL_BRIDGE:
        return UnrealBridgeBackend(config)
    raise ValueError(f"Unknown backend type: {backend_type}")
