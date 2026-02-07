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

    def test_default_config(self):
        """Test default configuration values."""
        config = VisualizationConfig.default()
        assert config.force_scale > 0
        assert config.trajectory_width > 0
        assert len(config.force_color_map) > 0

    def test_vr_config(self):
        """Test VR-optimized configuration."""
        config = VisualizationConfig.for_vr()
        # VR should have larger scales for visibility
        assert config.force_scale >= VisualizationConfig.default().force_scale
        assert config.show_labels is False


class TestForceVectorRenderer:
    """Tests for ForceVectorRenderer."""

    def test_create_renderer(self):
        """Test renderer creation."""
        renderer = ForceVectorRenderer()
        assert renderer is not None

    def test_render_single_force(self):
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

    def test_render_torque(self):
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

    def test_render_multiple_forces(self):
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

    def test_render_with_custom_color(self):
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

    def test_render_data_metadata(self):
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


class TestTrajectoryRenderer:
    """Tests for TrajectoryRenderer."""

    def test_create_renderer(self):
        """Test renderer creation."""
        renderer = TrajectoryRenderer()
        assert renderer is not None

    def test_render_empty_trajectory(self):
        """Test rendering empty trajectory."""
        renderer = TrajectoryRenderer()
        result = renderer.render([])
        assert result.vertices.size == 0

    def test_render_trajectory_line(self):
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

    def test_render_trajectory_ribbon(self):
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

    def test_render_with_velocity_colors(self):
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

    def test_render_ball_flight(self):
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


class TestHUDDataProvider:
    """Tests for HUDDataProvider."""

    def test_create_provider_metric(self):
        """Test provider with metric units."""
        provider = HUDDataProvider(units="metric")
        assert provider.units == "metric"

    def test_create_provider_imperial(self):
        """Test provider with imperial units."""
        provider = HUDDataProvider(units="imperial")
        assert provider.units == "imperial"

    def test_get_hud_data_with_metrics(self):
        """Test HUD data with swing metrics."""
        provider = HUDDataProvider()
        metrics = SwingMetrics(
            club_head_speed=45.0,
            x_factor=52.0,
            smash_factor=1.48,
        )
        hud = provider.get_hud_data(metrics=metrics, timestamp=0.5, frame_number=30)

        assert hud["timestamp"] == 0.5
        assert hud["frame"] == 30
        assert "panels" in hud
        assert "club_head_speed" in hud["panels"]

    def test_format_value(self):
        """Test value formatting."""
        provider = HUDDataProvider()
        panel = {"value": 45.234, "unit": "m/s", "format": "{:.1f}"}
        formatted = provider.format_value(panel)
        assert formatted == "45.2 m/s"

    def test_get_compact_hud(self):
        """Test compact HUD output."""
        provider = HUDDataProvider()
        metrics = SwingMetrics(club_head_speed=45.0, x_factor=52.0)
        compact = provider.get_compact_hud(metrics)
        assert "Club Head Speed" in compact

    def test_unit_conversion_imperial(self):
        """Test unit conversion to imperial."""
        provider = HUDDataProvider(units="imperial")
        metrics = SwingMetrics(club_head_speed=44.7)  # ~100 mph
        hud = provider.get_hud_data(metrics=metrics)
        # Should be converted to mph
        speed_panel = hud["panels"]["club_head_speed"]
        assert speed_panel["unit"] == "mph"


# ============================================================================
# VR Interaction Tests
# ============================================================================


class TestVRControllerState:
    """Tests for VRControllerState."""

    def test_create_controller_state(self):
        """Test controller state creation."""
        state = VRControllerState(
            hand=VRControllerHand.LEFT,
            position=Vector3(x=0.0, y=1.0, z=0.0),
            rotation=Quaternion.identity(),
        )
        assert state.hand == VRControllerHand.LEFT
        assert state.position.y == 1.0

    def test_trigger_pressed(self):
        """Test trigger pressed property."""
        state = VRControllerState(
            hand=VRControllerHand.RIGHT,
            position=Vector3.zero(),
            rotation=Quaternion.identity(),
            trigger=0.8,
        )
        assert state.is_trigger_pressed

    def test_grip_pressed(self):
        """Test grip pressed property."""
        state = VRControllerState(
            hand=VRControllerHand.LEFT,
            position=Vector3.zero(),
            rotation=Quaternion.identity(),
            grip=0.6,
        )
        assert state.is_grip_pressed

    def test_to_dict_from_dict(self):
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


class TestVRHeadsetState:
    """Tests for VRHeadsetState."""

    def test_create_headset_state(self):
        """Test headset state creation."""
        state = VRHeadsetState(
            position=Vector3(x=0.0, y=0.0, z=1.7),  # Eye height
            rotation=Quaternion.identity(),
        )
        assert state.position.z == 1.7

    def test_forward_vector(self):
        """Test forward direction calculation."""
        state = VRHeadsetState(
            position=Vector3.zero(),
            rotation=Quaternion.identity(),
        )
        forward = state.forward
        # Identity rotation should give forward in -Z
        assert isinstance(forward, Vector3)


class TestVRInteractionManager:
    """Tests for VRInteractionManager."""

    def test_create_manager(self):
        """Test manager creation."""
        manager = VRInteractionManager()
        assert manager is not None
        assert manager.locomotion_mode == VRLocomotionMode.TELEPORT

    def test_update_headset(self):
        """Test headset update."""
        manager = VRInteractionManager()
        headset = VRHeadsetState(
            position=Vector3(x=0.0, y=0.0, z=1.7),
            rotation=Quaternion.identity(),
        )
        manager.update_headset(headset, timestamp=0.0)
        assert manager.headset is not None

    def test_update_controller(self):
        """Test controller update."""
        manager = VRInteractionManager()
        controller = VRControllerState(
            hand=VRControllerHand.LEFT,
            position=Vector3.zero(),
            rotation=Quaternion.identity(),
        )
        manager.update_controller(controller, timestamp=0.0)
        assert manager.left_controller is not None

    def test_trigger_event_callback(self):
        """Test trigger press event callback."""
        manager = VRInteractionManager()
        events = []

        def on_trigger(event):
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

    def test_set_locomotion_mode(self):
        """Test locomotion mode change."""
        manager = VRInteractionManager()
        events = []
        manager.on("locomotion_mode_changed", lambda e: events.append(e))

        manager.set_locomotion_mode(VRLocomotionMode.SMOOTH)
        assert manager.locomotion_mode == VRLocomotionMode.SMOOTH
        assert len(events) == 1

    def test_get_state(self):
        """Test getting complete VR state."""
        manager = VRInteractionManager()
        state = manager.get_state()
        assert "locomotion_mode" in state
        assert "interaction_mode" in state


# ============================================================================
# Viewer Backend Tests
# ============================================================================


class TestViewerConfig:
    """Tests for ViewerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ViewerConfig()
        assert config.width == 1280
        assert config.height == 720
        assert config.backend_type == BackendType.MESHCAT

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        config = ViewerConfig(width=1920, height=1080)
        d = config.to_dict()
        restored = ViewerConfig.from_dict(d)
        assert restored.width == config.width
        assert restored.height == config.height


class TestCameraState:
    """Tests for CameraState."""

    def test_default_camera(self):
        """Test default camera state."""
        camera = CameraState()
        assert camera.fov == 45.0
        assert (
            camera.position.x != 0 or camera.position.y != 0 or camera.position.z != 0
        )


class TestMockBackend:
    """Tests for MockBackend."""

    def test_create_mock_backend(self):
        """Test mock backend creation."""
        backend = MockBackend()
        assert backend is not None
        assert not backend.is_initialized

    def test_initialize_shutdown(self):
        """Test initialization and shutdown."""
        backend = MockBackend()
        backend.initialize()
        assert backend.is_initialized
        backend.shutdown()
        assert not backend.is_initialized

    def test_context_manager(self):
        """Test context manager usage."""
        with MockBackend() as backend:
            assert backend.is_initialized
        assert not backend.is_initialized

    def test_add_mesh(self):
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

    def test_remove_mesh(self):
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

    def test_clear(self):
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

    def test_render(self):
        """Test mock backend render."""
        backend = MockBackend()
        backend.initialize()

        image = backend.render()
        assert image is not None
        assert image.shape[0] == backend.config.height
        assert image.shape[1] == backend.config.width
        assert backend.render_count == 1

    def test_update_transform(self):
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


class TestCreateViewer:
    """Tests for create_viewer factory function."""

    def test_create_mock_viewer(self):
        """Test creating mock viewer."""
        viewer = create_viewer("mock")
        assert isinstance(viewer, MockBackend)

    def test_create_viewer_with_config(self):
        """Test creating viewer with custom config."""
        config = ViewerConfig(width=800, height=600)
        viewer = create_viewer("mock", config=config)
        assert viewer.config.width == 800
        assert viewer.config.height == 600

    def test_create_unsupported_viewer(self):
        """Test creating unsupported viewer raises error."""
        with pytest.raises(ValueError):
            create_viewer("nonexistent_backend")


class TestRenderData:
    """Tests for RenderData."""

    def test_create_render_data(self):
        """Test render data creation."""
        data = RenderData(
            visualization_type=VisualizationType.FORCE_ARROW,
            vertices=np.array([[0, 0, 0], [1, 0, 0]]),
        )
        assert data.visualization_type == VisualizationType.FORCE_ARROW
        assert len(data.vertices) == 2

    def test_render_data_to_dict(self):
        """Test render data serialization."""
        data = RenderData(
            visualization_type=VisualizationType.TRAJECTORY_LINE,
            vertices=np.array([[0, 0, 0], [1, 1, 1]]),
            colors=np.array([[1, 0, 0, 1], [0, 1, 0, 1]]),
            metadata={"point_count": 2},
        )
        d = data.to_dict()
        assert d["type"] == "trajectory_line"
        assert len(d["vertices"]) == 2
        assert d["metadata"]["point_count"] == 2
