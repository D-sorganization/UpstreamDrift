"""Unit tests for visualization, VR interaction, and viewer backends.

TDD tests for the remaining Unreal Engine integration components.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.unreal_integration.data_models import (
    ForceVector,
    Quaternion,
    SwingMetrics,
    TrajectoryPoint,
    Vector3,
)
from src.unreal_integration.mesh_loader import (
    LoadedMesh,
    MeshFace,
    MeshVertex,
)
from src.unreal_integration.viewer_backends import (
    BackendType,
    CameraState,
    MockBackend,
    ViewerConfig,
    create_viewer,
)
from src.unreal_integration.visualization import (
    ForceVectorRenderer,
    HUDDataProvider,
    RenderData,
    TrajectoryRenderer,
    VisualizationConfig,
    VisualizationType,
)
from src.unreal_integration.vr_interaction import (
    VRControllerHand,
    VRControllerState,
    VRHeadsetState,
    VRInteractionManager,
    VRLocomotionMode,
)

# ============================================================================
# Visualization Tests
# ============================================================================


# ============================================================================
# VR Interaction Tests
# ============================================================================


# ============================================================================
# Viewer Backend Tests
# ============================================================================


class TestMockBackend:
    """Tests for MockBackend."""

    def test_create_mock_backend(self) -> None:
        """Test mock backend creation."""
        backend = MockBackend()
        assert backend is not None
        assert not backend.is_initialized

    def test_initialize_shutdown(self) -> None:
        """Test initialization and shutdown."""
        backend = MockBackend()
        backend.initialize()
        assert backend.is_initialized
        backend.shutdown()
        assert not backend.is_initialized

    def test_context_manager(self) -> None:
        """Test context manager usage."""
        with MockBackend() as backend:
            assert backend.is_initialized
        assert not backend.is_initialized

    def test_add_mesh(self) -> None:
        """Test adding mesh to mock backend."""
        backend = MockBackend()
        backend.initialize()

        mesh = LoadedMesh(
            name="test",
            vertices=[MeshVertex(position=np.array([0.0, 0.0, 0.0]))],
            faces=[MeshFace(indices=np.array([0, 0, 0]))],
        )
        name = backend.add_mesh(mesh)
        assert name is not None
        assert backend.object_count == 1

    def test_remove_mesh(self) -> None:
        """Test removing mesh from mock backend."""
        backend = MockBackend()
        backend.initialize()

        mesh = LoadedMesh(
            name="test",
            vertices=[MeshVertex(position=np.array([0.0, 0.0, 0.0]))],
            faces=[MeshFace(indices=np.array([0, 0, 0]))],
        )
        name = backend.add_mesh(mesh)
        assert backend.remove_object(name)
        assert backend.object_count == 0

    def test_visualization_vr_backends_clear(self) -> None:
        """Test clearing mock backend."""
        backend = MockBackend()
        backend.initialize()

        mesh = LoadedMesh(
            name="test",
            vertices=[MeshVertex(position=np.array([0.0, 0.0, 0.0]))],
            faces=[MeshFace(indices=np.array([0, 0, 0]))],
        )
        backend.add_mesh(mesh)
        backend.add_mesh(mesh, name="mesh2")
        assert backend.object_count == 2

        backend.clear()
        assert backend.object_count == 0

    def test_render(self) -> None:
        """Test mock backend render."""
        backend = MockBackend()
        backend.initialize()

        image = backend.render()
        assert image is not None
        assert image.shape[0] == backend.config.height
        assert image.shape[1] == backend.config.width
        assert backend.render_count == 1

    def test_update_transform(self) -> None:
        """Test updating object transform."""
        backend = MockBackend()
        backend.initialize()

        mesh = LoadedMesh(
            name="test",
            vertices=[MeshVertex(position=np.array([0.0, 0.0, 0.0]))],
            faces=[MeshFace(indices=np.array([0, 0, 0]))],
        )
        name = backend.add_mesh(mesh)
        backend.update_transform(name, position=Vector3(x=1.0, y=2.0, z=3.0))
        # Should not raise
