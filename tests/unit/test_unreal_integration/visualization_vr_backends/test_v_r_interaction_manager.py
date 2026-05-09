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


class TestVRInteractionManager:
    """Tests for VRInteractionManager."""

    def test_create_manager(self) -> None:
        """Test manager creation."""
        manager = VRInteractionManager()
        assert manager is not None
        assert manager.locomotion_mode == VRLocomotionMode.TELEPORT

    def test_update_headset(self) -> None:
        """Test headset update."""
        manager = VRInteractionManager()
        headset = VRHeadsetState(
            position=Vector3(x=0.0, y=0.0, z=1.7),
            rotation=Quaternion.identity(),
        )
        manager.update_headset(headset, timestamp=0.0)
        assert manager.headset is not None

    def test_update_controller(self) -> None:
        """Test controller update."""
        manager = VRInteractionManager()
        controller = VRControllerState(
            hand=VRControllerHand.LEFT,
            position=Vector3.zero(),
            rotation=Quaternion.identity(),
        )
        manager.update_controller(controller, timestamp=0.0)
        assert manager.left_controller is not None

    def test_trigger_event_callback(self) -> None:
        """Test trigger press event callback."""
        manager = VRInteractionManager()
        events = []

        def on_trigger(event) -> None:
            events.append(event)

        manager.on_trigger_press(on_trigger)

        # First update - no trigger
        controller1 = VRControllerState(
            hand=VRControllerHand.RIGHT,
            position=Vector3.zero(),
            rotation=Quaternion.identity(),
            trigger=0.0,
        )
        manager.update_controller(controller1, timestamp=0.0)

        # Second update - trigger pressed
        controller2 = VRControllerState(
            hand=VRControllerHand.RIGHT,
            position=Vector3.zero(),
            rotation=Quaternion.identity(),
            trigger=0.8,
        )
        manager.update_controller(controller2, timestamp=0.1)

        assert len(events) == 1
        assert events[0].event_type == "trigger_press"

    def test_set_locomotion_mode(self) -> None:
        """Test locomotion mode change."""
        manager = VRInteractionManager()
        events = []
        manager.on("locomotion_mode_changed", lambda e: events.append(e))

        manager.set_locomotion_mode(VRLocomotionMode.SMOOTH)
        assert manager.locomotion_mode == VRLocomotionMode.SMOOTH
        assert len(events) == 1

    def test_visualization_vr_backends_get_state(self) -> None:
        """Test getting complete VR state."""
        manager = VRInteractionManager()
        state = manager.get_state()
        assert "locomotion_mode" in state
        assert "interaction_mode" in state


# ============================================================================
# Viewer Backend Tests
# ============================================================================


# Should not raise
