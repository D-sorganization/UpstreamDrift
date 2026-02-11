"""Unit tests for Drake Visualizer."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock pydrake before importing
mock_pydrake = MagicMock()
sys.modules["pydrake"] = mock_pydrake
sys.modules["pydrake.all"] = mock_pydrake


# Fix mocking for class methods and types
# RotationMatrix needs to be a class with MakeYRotation classmethod
class MockRotationMatrix:
    @staticmethod
    def MakeXRotation(angle):
        return MagicMock()

    @staticmethod
    def MakeYRotation(angle):
        return MagicMock()

    @staticmethod
    def MakeZRotation(angle):
        return MagicMock()

    def __init__(self, *args):
        pass


mock_pydrake.RotationMatrix = MockRotationMatrix


# RigidTransform needs to handle numpy array inputs without inspecting them for
# coroutines (which causes ValueError on .T).
# So we make it a simple class or function that returns a mock.
class MockRigidTransform:
    def __init__(self, *args, **kwargs):
        pass


mock_pydrake.RigidTransform = MockRigidTransform
mock_pydrake.Cylinder = MagicMock()
mock_pydrake.Sphere = MagicMock()
mock_pydrake.Rgba = MagicMock()


from src.engines.physics_engines.drake.python.src.drake_visualizer import (  # noqa: E402
    DrakeVisualizer,
)


class TestDrakeVisualizer:
    """Test suite for DrakeVisualizer."""

    @pytest.fixture
    def mock_meshcat(self):
        """Mock Meshcat."""
        return MagicMock()

    @pytest.fixture
    def mock_plant(self):
        """Mock MultibodyPlant."""
        return MagicMock()

    @pytest.fixture
    def visualizer(self, mock_meshcat, mock_plant):
        """Create visualizer instance."""
        return DrakeVisualizer(mock_meshcat, mock_plant)

    def test_initialization(self, visualizer, mock_meshcat, mock_plant):
        """Test initialization."""
        assert visualizer.meshcat == mock_meshcat
        assert visualizer.plant == mock_plant
        assert visualizer.prefix == "visual_overlays"
        assert len(visualizer.visible_frames) == 0
        assert len(visualizer.visible_coms) == 0
        assert len(visualizer.visible_ellipsoids) == 0

    def test_toggle_frame_visible(self, visualizer, mock_meshcat):
        """Test enabling frame visualization."""
        body_name = "test_body"

        visualizer.toggle_frame(body_name, True)

        assert body_name in visualizer.visible_frames

        # Should have set objects for x, y, z axes
        # base_path = f"visual_overlays/frames/{body_name}"
        # Cylinder calls return new mocks each time, so we check call count
        assert mock_meshcat.SetObject.call_count == 3

        # Verify Transforms were set
        assert mock_meshcat.SetTransform.call_count == 3

    def test_toggle_frame_hidden(self, visualizer, mock_meshcat):
        """Test disabling frame visualization."""
        body_name = "test_body"
        visualizer.visible_frames.add(body_name)

        visualizer.toggle_frame(body_name, False)

        assert body_name not in visualizer.visible_frames
        mock_meshcat.Delete.assert_called_with(f"visual_overlays/frames/{body_name}")

    def test_update_frame_transforms(self, visualizer, mock_meshcat, mock_plant):
        """Test updating frame transforms."""
        body_name = "test_body"
        visualizer.visible_frames.add(body_name)
        context = MagicMock()
        plant_context = MagicMock()
        mock_plant.GetMyContextFromRoot.return_value = plant_context

        body = MagicMock()
        mock_plant.GetBodyByName.return_value = body

        X_WB = MagicMock()  # World-Body transform
        mock_plant.EvalBodyPoseInWorld.return_value = X_WB

        visualizer.update_frame_transforms(context)

        mock_plant.GetBodyByName.assert_called_with(body_name)
        mock_plant.EvalBodyPoseInWorld.assert_called_with(plant_context, body)
        mock_meshcat.SetTransform.assert_called_with(
            f"visual_overlays/frames/{body_name}", X_WB
        )

    def test_toggle_com_visible(self, visualizer, mock_meshcat):
        """Test enabling COM visualization."""
        body_name = "test_body"

        visualizer.toggle_com(body_name, True)

        assert body_name in visualizer.visible_coms
        mock_meshcat.SetObject.assert_called_once()
        args, _ = mock_meshcat.SetObject.call_args
        assert args[0] == f"visual_overlays/coms/{body_name}"

    def test_toggle_com_hidden(self, visualizer, mock_meshcat):
        """Test disabling COM visualization."""
        body_name = "test_body"
        visualizer.visible_coms.add(body_name)

        visualizer.toggle_com(body_name, False)

        assert body_name not in visualizer.visible_coms
        mock_meshcat.Delete.assert_called_with(f"visual_overlays/coms/{body_name}")

    def test_update_com_transforms(self, visualizer, mock_meshcat, mock_plant):
        """Test updating COM transforms."""
        body_name = "test_body"
        visualizer.visible_coms.add(body_name)
        context = MagicMock()
        plant_context = MagicMock()
        mock_plant.GetMyContextFromRoot.return_value = plant_context

        body = MagicMock()
        mock_plant.GetBodyByName.return_value = body

        # Setup Transforms
        # X_WB: World -> Body
        X_WB = MagicMock()
        mock_plant.EvalBodyPoseInWorld.return_value = X_WB

        # M_B: Spatial Inertia
        M_B = MagicMock()
        body.CalcSpatialInertiaInBodyFrame.return_value = M_B

        # p_BCom: COM in body frame
        p_BCom = np.array([0.1, 0.2, 0.3])
        M_B.get_com.return_value = p_BCom

        # Result of X_WB @ p_BCom
        p_WCom = np.array([1.1, 1.2, 1.3])
        X_WB.__matmul__.return_value = p_WCom

        visualizer.update_com_transforms(context)

        # Check Transform was created (we can't easily inspect args if
        # MockRigidTransform eats them, but no crash is good)
        mock_meshcat.SetTransform.assert_called()

    def test_draw_ellipsoid(self, visualizer, mock_meshcat):
        """Test drawing ellipsoid."""
        name = "test_ellipsoid"
        rotation_matrix = np.eye(3)
        radii = np.array([1.0, 0.5, 0.2])
        position = np.array([10.0, 20.0, 30.0])
        color = (1.0, 0.0, 0.0, 0.5)

        visualizer.draw_ellipsoid(name, rotation_matrix, radii, position, color)

        assert name in visualizer.visible_ellipsoids
        mock_meshcat.SetObject.assert_called()
        mock_meshcat.SetTransform.assert_called()

        args, _ = mock_meshcat.SetTransform.call_args
        assert args[0] == f"visual_overlays/ellipsoids/{name}"
        T_arg = args[1]
        assert isinstance(T_arg, np.ndarray)
        assert T_arg.shape == (4, 4)
        np.testing.assert_allclose(T_arg[:3, 3], position)

    def test_clear_ellipsoids(self, visualizer, mock_meshcat):
        """Test clearing ellipsoids."""
        visualizer.visible_ellipsoids.add("e1")
        visualizer.clear_ellipsoids()

        assert len(visualizer.visible_ellipsoids) == 0
        mock_meshcat.Delete.assert_called_with("visual_overlays/ellipsoids")

    def test_clear_all(self, visualizer, mock_meshcat):
        """Test clearing all overlays."""
        visualizer.visible_frames.add("f1")
        visualizer.visible_coms.add("c1")
        visualizer.visible_ellipsoids.add("e1")

        visualizer.clear_all()

        assert len(visualizer.visible_frames) == 0
        assert len(visualizer.visible_coms) == 0
        assert len(visualizer.visible_ellipsoids) == 0
        mock_meshcat.Delete.assert_called_with("visual_overlays")
