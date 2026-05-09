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


class TestVRHeadsetState:
    """Tests for VRHeadsetState."""

    def test_create_headset_state(self) -> None:
        """Test headset state creation."""
        state = VRHeadsetState(
            position=Vector3(x=0.0, y=0.0, z=1.7),  # Eye height
            rotation=Quaternion.identity(),
        )
        assert state.position.z == 1.7

    def test_forward_vector(self) -> None:
        """Test forward direction calculation."""
        state = VRHeadsetState(
            position=Vector3.zero(),
            rotation=Quaternion.identity(),
        )
        forward = state.forward
        # Identity rotation should give forward in -Z
        assert isinstance(forward, Vector3)


# ============================================================================
# Viewer Backend Tests
# ============================================================================


# Should not raise
