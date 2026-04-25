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

from ._viewer_base import (
    BackendType,
    CameraState,
    LightState,
    ViewerBackend,
    ViewerConfig,
)
from ._viewer_factory import create_viewer
from ._viewer_meshcat import MeshcatBackend
from ._viewer_mock import MockBackend
from ._viewer_unreal_bridge import UnrealBridgeBackend

__all__ = [
    "BackendType",
    "CameraState",
    "LightState",
    "MeshcatBackend",
    "MockBackend",
    "UnrealBridgeBackend",
    "ViewerBackend",
    "ViewerConfig",
    "create_viewer",
]
