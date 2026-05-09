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


class TestVRControllerState:
    """Tests for VRControllerState."""

    def test_create_controller_state(self) -> None:
        """Test controller state creation."""
        state = VRControllerState(
            hand=VRControllerHand.LEFT,
            position=Vector3(x=0.0, y=1.0, z=0.0),
            rotation=Quaternion.identity(),
        )
        assert state.hand == VRControllerHand.LEFT
        assert state.position.y == 1.0

    def test_trigger_pressed(self) -> None:
        """Test trigger pressed property."""
        state = VRControllerState(
            hand=VRControllerHand.RIGHT,
            position=Vector3.zero(),
            rotation=Quaternion.identity(),
            trigger=0.8,
        )
        assert state.is_trigger_pressed

    def test_grip_pressed(self) -> None:
        """Test grip pressed property."""
        state = VRControllerState(
            hand=VRControllerHand.LEFT,
            position=Vector3.zero(),
            rotation=Quaternion.identity(),
            grip=0.6,
        )
        assert state.is_grip_pressed

    def test_to_dict_from_dict(self) -> None:
        """Test serialization round-trip."""
        state = VRControllerState(
            hand=VRControllerHand.RIGHT,
            position=Vector3(x=1.0, y=2.0, z=3.0),
            rotation=Quaternion.identity(),
            trigger=0.5,
            grip=0.3,
        )
        d = state.to_dict()
        restored = VRControllerState.from_dict(d)
        assert restored.hand == state.hand
        assert restored.trigger == state.trigger


# ============================================================================
# Viewer Backend Tests
# ============================================================================


# Should not raise
