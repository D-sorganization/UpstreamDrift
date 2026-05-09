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


class TestViewerConfig:
    """Tests for ViewerConfig."""

    def test_visualization_vr_backends_default_config(self) -> None:
        """Test default configuration."""
        config = ViewerConfig()
        assert config.width == 1280
        assert config.height == 720
        assert config.backend_type == BackendType.MESHCAT

    def test_to_dict_from_dict(self) -> None:
        """Test serialization round-trip."""
        config = ViewerConfig(width=1920, height=1080)
        d = config.to_dict()
        restored = ViewerConfig.from_dict(d)
        assert restored.width == config.width
        assert restored.height == config.height

        # Should not raise
