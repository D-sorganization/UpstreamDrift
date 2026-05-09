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


class TestForceVectorRenderer:
    """Tests for ForceVectorRenderer."""

    def test_create_renderer(self) -> None:
        """Test renderer creation."""
        renderer = ForceVectorRenderer()
        assert renderer is not None

    def test_render_single_force(self) -> None:
        """Test rendering a single force vector."""
        renderer = ForceVectorRenderer()
        force = ForceVector(
            origin=Vector3.zero(),
            direction=Vector3(x=0.0, y=0.0, z=1.0),
            magnitude=10.0,
            force_type="force",
        )
        results = renderer.render([force])
        assert len(results) == 1
        assert results[0].visualization_type == VisualizationType.FORCE_ARROW

    def test_render_torque(self) -> None:
        """Test rendering a torque vector."""
        renderer = ForceVectorRenderer()
        torque = ForceVector(
            origin=Vector3.zero(),
            direction=Vector3(x=0.0, y=0.0, z=1.0),
            magnitude=5.0,
            force_type="torque",
        )
        results = renderer.render([torque])
        assert len(results) == 1
        assert results[0].visualization_type == VisualizationType.TORQUE_RING

    def test_render_multiple_forces(self) -> None:
        """Test rendering multiple forces."""
        renderer = ForceVectorRenderer()
        forces = [
            ForceVector(
                origin=Vector3(x=float(i), y=0.0, z=0.0),
                direction=Vector3(x=0.0, y=1.0, z=0.0),
                magnitude=10.0,
            )
            for i in range(5)
        ]
        results = renderer.render(forces)
        assert len(results) == 5

    def test_render_with_custom_color(self) -> None:
        """Test rendering with custom color."""
        renderer = ForceVectorRenderer()
        force = ForceVector(
            origin=Vector3.zero(),
            direction=Vector3(x=1.0, y=0.0, z=0.0),
            magnitude=10.0,
            color=(1.0, 0.0, 0.0, 1.0),  # Red
        )
        results = renderer.render([force])
        assert results[0].colors is not None
        assert results[0].colors[0, 0] == 1.0  # Red channel

    def test_render_data_metadata(self) -> None:
        """Test render data contains expected metadata."""
        renderer = ForceVectorRenderer()
        force = ForceVector(
            origin=Vector3.zero(),
            direction=Vector3(x=1.0, y=0.0, z=0.0),
            magnitude=15.5,
            force_type="ground_reaction",
            joint_name="ankle_L",
        )
        results = renderer.render([force])
        assert results[0].metadata["magnitude"] == 15.5
        assert results[0].metadata["force_type"] == "ground_reaction"
        assert results[0].metadata["joint_name"] == "ankle_L"


# ============================================================================
# VR Interaction Tests
# ============================================================================


# ============================================================================
# Viewer Backend Tests
# ============================================================================


# Should not raise
