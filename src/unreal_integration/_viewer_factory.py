from __future__ import annotations

from ._viewer_base import BackendType, ViewerBackend, ViewerConfig
from ._viewer_meshcat import MeshcatBackend
from ._viewer_mock import MockBackend
from ._viewer_unreal_bridge import UnrealBridgeBackend


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
        from .viewer_backends_pyvista import PyVistaBackend

        return PyVistaBackend(config)
    if backend_type == BackendType.UNREAL_BRIDGE:
        return UnrealBridgeBackend(config)
    raise ValueError(f"Unknown backend type: {backend_type}")
