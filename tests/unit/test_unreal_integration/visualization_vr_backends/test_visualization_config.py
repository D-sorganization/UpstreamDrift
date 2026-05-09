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


class TestVisualizationConfig:
    """Tests for VisualizationConfig."""

    def test_visualization_vr_backends_default_config(self) -> None:
        """Test default configuration values."""
        config = VisualizationConfig.default()
        assert config.force_scale > 0
        assert config.trajectory_width > 0
        assert len(config.force_color_map) > 0

    def test_vr_config(self) -> None:
        """Test VR-optimized configuration."""
        config = VisualizationConfig.for_vr()
        # VR should have larger scales for visibility
        assert config.force_scale >= VisualizationConfig.default().force_scale
        assert config.show_labels is False


# ============================================================================
# VR Interaction Tests
# ============================================================================


# ============================================================================
# Viewer Backend Tests
# ============================================================================


# Should not raise
