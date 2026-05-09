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


class TestTrajectoryRenderer:
    """Tests for TrajectoryRenderer."""

    def test_create_renderer(self) -> None:
        """Test renderer creation."""
        renderer = TrajectoryRenderer()
        assert renderer is not None

    def test_render_empty_trajectory(self) -> None:
        """Test rendering empty trajectory."""
        renderer = TrajectoryRenderer()
        result = renderer.render([])
        assert result.vertices.size == 0

    def test_render_trajectory_line(self) -> None:
        """Test rendering trajectory as line."""
        renderer = TrajectoryRenderer()
        points = [
            TrajectoryPoint(
                time=float(i) * 0.1, position=Vector3(x=float(i), y=0.0, z=0.0)
            )
            for i in range(10)
        ]
        result = renderer.render(points)
        assert result.visualization_type == VisualizationType.TRAJECTORY_LINE
        assert len(result.vertices) == 10

    def test_render_trajectory_ribbon(self) -> None:
        """Test rendering trajectory as ribbon."""
        renderer = TrajectoryRenderer()
        points = [
            TrajectoryPoint(
                time=float(i) * 0.1, position=Vector3(x=float(i), y=0.0, z=0.0)
            )
            for i in range(10)
        ]
        result = renderer.render(points, as_ribbon=True)
        assert result.visualization_type == VisualizationType.TRAJECTORY_RIBBON

    def test_render_with_velocity_colors(self) -> None:
        """Test trajectory with velocity-based colors."""
        renderer = TrajectoryRenderer()
        points = [
            TrajectoryPoint(
                time=float(i) * 0.1,
                position=Vector3(x=float(i), y=0.0, z=0.0),
                velocity=Vector3(x=float(i) * 5, y=0.0, z=0.0),  # Increasing velocity
            )
            for i in range(10)
        ]
        result = renderer.render(points)
        assert result.colors is not None
        # Colors should vary with velocity

    def test_render_ball_flight(self) -> None:
        """Test ball flight trajectory with landing marker."""
        renderer = TrajectoryRenderer()
        points = [
            TrajectoryPoint(
                time=float(i) * 0.1,
                position=Vector3(x=float(i) * 10, y=0.0, z=float(i) * (10 - i)),
            )
            for i in range(11)
        ]
        results = renderer.render_ball_flight(points, landing_marker=True)
        assert len(results) >= 1  # At least trajectory
        assert any(r.metadata.get("marker_type") == "landing" for r in results)


# ============================================================================
# VR Interaction Tests
# ============================================================================


# ============================================================================
# Viewer Backend Tests
# ============================================================================


# Should not raise
