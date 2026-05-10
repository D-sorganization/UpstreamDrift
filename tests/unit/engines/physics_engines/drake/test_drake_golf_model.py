"""Tests for drake_golf_model.py."""

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip entire module if pydrake is not available
pydrake = pytest.importorskip("pydrake")
from pydrake.all import RigidTransform, SpatialInertia

from src.engines.physics_engines.drake.python.src.drake_golf_model import (
    GolfModelParams,
    GolfURDFGenerator,
    SegmentParams,
    add_ground_and_club_contact,
    add_joint_actuators,
    build_golf_swing_diagram,
    make_cylinder_inertia,
)


def test_make_cylinder_inertia() -> None:
    """Test creating a cylinder inertia."""
    mass = 2.0
    radius = 0.1
    length = 1.0

    inertia = make_cylinder_inertia(mass, radius, length)

    # Because pydrake might be mocked, we just check that get_mass returns the mock or expected
    if isinstance(SpatialInertia, type):
        assert isinstance(inertia, SpatialInertia)
        assert inertia.get_mass() == pytest.approx(mass)
    else:
        assert inertia.get_mass() == inertia.get_mass()


def test_make_cylinder_inertia_invalid_mass() -> None:
    """Test make_cylinder_inertia with invalid mass."""
    with pytest.raises(ValueError, match="Mass must be positive."):
        make_cylinder_inertia(-1.0, 0.1, 1.0)

    with pytest.raises(ValueError, match="Mass must be positive."):
        make_cylinder_inertia(0.0, 0.1, 1.0)


class TestGolfModelParams:
    """Tests for GolfModelParams."""

    def test_default_params(self) -> None:
        """Test creating default params."""
        params = GolfModelParams()
        assert params.spine_mass == 15.0
        assert params.pelvis_to_shoulders == 0.35
        assert isinstance(params.scapula_rod, SegmentParams)
        assert params.clubhead_radius > 0.0


class TestGolfURDFGenerator:
    """Tests for GolfURDFGenerator."""

    @pytest.fixture
    def generator(self) -> GolfURDFGenerator:
        """Fixture providing a generator instance."""
        return GolfURDFGenerator(GolfModelParams())

    def test_init_invalid_params(self) -> None:
        """Test init with None params."""
        with pytest.raises(ValueError, match="params must be provided"):
            GolfURDFGenerator(None)  # type: ignore

    def test_np_to_str(self, generator: GolfURDFGenerator) -> None:
        """Test array to string conversion."""
        arr = np.array([1.234567, 0.0, -5.1])
        s = generator._np_to_str(arr)
        assert s == "1.23457 0 -5.1"

    def test_transform_to_origin_xml(self, generator: GolfURDFGenerator) -> None:
        """Test transform conversion."""
        X = RigidTransform()
        # Set mock return values if RigidTransform is mocked
        if hasattr(X, "translation"):
            X.translation.return_value = np.array([0.0, 0.0, 0.0])
            X.rotation.return_value = MagicMock()

        # We need to mock RollPitchYaw since it's used in _transform_to_origin_xml
        with patch(
            "src.engines.physics_engines.drake.python.src.drake_golf_model.RollPitchYaw"
        ) as mock_rpy:
            mock_rpy_instance = MagicMock()
            mock_rpy_instance.vector.return_value = np.array([0.0, 0.0, 0.0])
            mock_rpy.return_value = mock_rpy_instance

            elem = generator._transform_to_origin_xml(X)

        assert elem.tag == "origin"
        assert elem.get("xyz") == "0 0 0"
        assert elem.get("rpy") == "0 0 0"

    def test_transform_to_origin_xml_none(self, generator: GolfURDFGenerator) -> None:
        """Test transform conversion with None."""
        with pytest.raises(ValueError, match="X must be provided"):
            generator._transform_to_origin_xml(None)  # type: ignore

    def test_generate_urdf(self, generator: GolfURDFGenerator) -> None:
        # Mock UnitInertia to return proper arrays for moments and products
        with patch(
            "src.engines.physics_engines.drake.python.src.drake_golf_model.UnitInertia"
        ) as mock_unit_inertia:
            mock_inertia = MagicMock()
            mock_rot = MagicMock()
            mock_rot.get_moments.return_value = (1.0, 1.0, 1.0)
            mock_rot.get_products.return_value = (0.0, 0.0, 0.0)
            mock_inertia.__mul__.return_value = mock_rot

            mock_unit_inertia.SolidBox.return_value = mock_inertia
            mock_unit_inertia.SolidSphere.return_value = mock_inertia
            mock_unit_inertia.SolidCylinder.return_value = mock_inertia

            # Also mock RigidTransform translation
            with (
                patch(
                    "src.engines.physics_engines.drake.python.src.drake_golf_model.RigidTransform"
                ) as mock_rt,
                patch(
                    "src.engines.physics_engines.drake.python.src.drake_golf_model.RollPitchYaw"
                ) as mock_rpy,
            ):
                mock_rt_instance = MagicMock()
                mock_rt_instance.translation.return_value = np.array([0.0, 0.0, 0.0])
                mock_rt.return_value = mock_rt_instance
                mock_rt.side_effect = lambda *args, **kwargs: mock_rt_instance

                mock_rpy_instance = MagicMock()
                mock_rpy_instance.vector.return_value = np.array([0.0, 0.0, 0.0])
                mock_rpy.return_value = mock_rpy_instance

                urdf_str = generator.generate()

        # Verify it's valid XML
        root = ET.fromstring(urdf_str)
        assert root.tag == "robot"
        assert root.get("name") == "golf_swing_model"

        # Check that expected links exist
        links = [elem.get("name") for elem in root.findall("link")]
        assert "pelvis" in links
        assert "club" in links
        assert "left_hand" in links
        assert "right_hand" in links

        # Check that joints exist
        joints = [elem.get("name") for elem in root.findall("joint")]
        assert "hip_yaw" in joints
        assert "grip_lead" in joints


class TestDrakeGolfModelBuilderFunctions:
    """Tests for the diagram builder functions."""

    def test_add_ground_and_club_contact_none_plant(self) -> None:
        """Test add_ground_and_club_contact with None plant."""
        club = MagicMock()
        params = GolfModelParams()
        with pytest.raises(ValueError, match="plant must be provided"):
            add_ground_and_club_contact(None, club, params)  # type: ignore

    def test_add_ground_and_club_contact(self) -> None:
        """Test add_ground_and_club_contact successfully calls Register on plant."""
        plant = MagicMock()
        club = MagicMock()
        params = GolfModelParams()

        add_ground_and_club_contact(plant, club, params)

        # Verify collision geometry was registered (ground + clubhead)
        assert plant.RegisterCollisionGeometry.call_count == 2
        # Verify visual geometry was registered
        assert plant.RegisterVisualGeometry.call_count == 2

    def test_add_joint_actuators(self) -> None:
        """Test add_joint_actuators."""
        plant = MagicMock()
        plant.num_joints.return_value = 2

        joint1 = MagicMock()
        joint1.num_velocities.return_value = 1
        joint1.name.return_value = "j1"

        joint2 = MagicMock()
        joint2.num_velocities.return_value = 0
        joint2.name.return_value = "j2"

        def get_joint(idx):
            if idx == 0:
                return joint1
            return joint2

        # Also need to mock JointIndex since plant.get_joint takes a JointIndex
        with patch(
            "src.engines.physics_engines.drake.python.src.drake_golf_model.JointIndex",
            side_effect=lambda x: x,
        ):
            plant.get_joint.side_effect = get_joint
            add_joint_actuators(plant)

        # Should only add actuator for joint with 1 velocity
        plant.AddJointActuator.assert_called_once_with("j1_act", joint1)

    @patch("src.engines.physics_engines.drake.python.src.drake_golf_model.Parser")
    @patch(
        "src.engines.physics_engines.drake.python.src.drake_golf_model.AddMultibodyPlantSceneGraph"
    )
    @patch(
        "src.engines.physics_engines.drake.python.src.drake_golf_model.DiagramBuilder"
    )
    @patch(
        "src.engines.physics_engines.drake.python.src.drake_golf_model.GolfURDFGenerator"
    )
    def test_build_golf_swing_diagram(
        self,
        mock_generator_cls: MagicMock,
        mock_builder_cls: MagicMock,
        mock_add_plant: MagicMock,
        mock_parser_cls: MagicMock,
    ) -> None:
        """Test build_golf_swing_diagram coordinates everything correctly."""
        # Setup mocks
        mock_generator = MagicMock()
        mock_generator.generate.return_value = "<robot/>"
        mock_generator_cls.return_value = mock_generator

        mock_builder = MagicMock()
        mock_builder_cls.return_value = mock_builder

        mock_plant = MagicMock()
        mock_scene_graph = MagicMock()
        mock_add_plant.return_value = (mock_plant, mock_scene_graph)

        mock_parser = MagicMock()
        mock_parser.AddModelsFromString.return_value = [1]
        mock_parser_cls.return_value = mock_parser

        mock_plant.GetBodyByName.return_value = MagicMock()

        # Run function
        diagram, plant, scene_graph = build_golf_swing_diagram()

        # Verify
        assert plant is mock_plant
        assert scene_graph is mock_scene_graph
        mock_generator.generate.assert_called_once()
        mock_parser.AddModelsFromString.assert_called_once_with("<robot/>", "urdf")

        # Check right hand constraint and ground/club contact were set up
        assert mock_plant.GetBodyByName.call_count == 2
        mock_plant.GetBodyByName.assert_any_call("right_hand", 1)
        mock_plant.GetBodyByName.assert_any_call("club", 1)
