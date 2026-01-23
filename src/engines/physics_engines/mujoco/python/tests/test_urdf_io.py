import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.urdf_io import (
    URDFExporter,
    URDFImporter,
    export_model_to_urdf,
    import_urdf_to_mujoco,
)


@pytest.fixture
def mock_mujoco_model():
    """Create a mock MuJoCo model."""
    model = MagicMock(spec=mujoco.MjModel)

    # Setup standard model structure
    model.nbody = 3  # World, Link1, Link2
    model.njnt = 2
    model.ngeom = 2
    model.nmat = 1

    # Body properties
    model.body_parentid = np.array(
        [0, 0, 1]
    )  # Link1 child of World, Link2 child of Link1
    model.body_jntadr = np.array([-1, 0, 1])  # Link1 has joint 0, Link2 has joint 1
    model.body_mass = np.array([0, 1.0, 1.0])
    model.body_inertia = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
    model.body_ipos = np.array([[0, 0, 0], [0, 0, 0], [0.5, 0, 0]])

    # Joint properties
    model.jnt_type = np.array(
        [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]
    )
    model.jnt_pos = np.array([[0, 0, 0], [1, 0, 0]])
    model.jnt_axis = np.array([[0, 0, 1], [1, 0, 0]])
    model.jnt_limited = np.array([True, False])
    model.jnt_range = np.array([[-np.pi, np.pi], [0, 0]])

    # Geom properties
    model.geom_bodyid = np.array([1, 2])
    model.geom_type = np.array(
        [mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE]
    )
    model.geom_size = np.array([[0.1, 0.1, 0.1], [0.05, 0, 0]])
    model.geom_pos = np.array([[0, 0, 0], [0, 0, 0]])
    model.geom_quat = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])
    model.geom_matid = np.array([0, -1])
    model.geom_dataid = np.array([-1, -1])

    # Material properties
    model.mat_rgba = np.array([[1, 0, 0, 1]])

    return model


@pytest.fixture
def sample_urdf_xml():
    return """
    <robot name="test_robot">
        <link name="base_link">
            <inertial>
                <mass value="1.0"/>
                <origin xyz="0 0 0"/>
                <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
            </inertial>
            <visual>
                <geometry>
                    <box size="1 1 1"/>
                </geometry>
            </visual>
        </link>
        <link name="child_link">
             <inertial>
                <mass value="0.5"/>
            </inertial>
            <collision>
                <geometry>
                    <sphere radius="0.5"/>
                </geometry>
            </collision>
        </link>
        <joint name="joint1" type="revolute">
            <parent link="base_link"/>
            <child link="child_link"/>
            <origin xyz="1 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-1" upper="1"/>
        </joint>
    </robot>
    """


def test_urdf_exporter_init(mock_mujoco_model):
    with patch("mujoco.MjData", autospec=True) as mock_data_cls:
        exporter = URDFExporter(mock_mujoco_model)
        assert exporter.model == mock_mujoco_model
        assert exporter.data == mock_data_cls.return_value


def test_export_to_urdf(mock_mujoco_model):
    with patch("mujoco.MjData", autospec=True):
        exporter = URDFExporter(mock_mujoco_model)

        # Mock mj_id2name to return names
        def id2name(m, obj_type, obj_id):
            if obj_type == mujoco.mjtObj.mjOBJ_BODY:
                return ["world", "link1", "link2"][obj_id]
            if obj_type == mujoco.mjtObj.mjOBJ_JOINT:
                return f"joint_{obj_id}"
            return "obj"

        with (
            patch("mujoco.mj_id2name", side_effect=id2name),
            patch("pathlib.Path.write_text"),
        ):
            urdf_str = exporter.export_to_urdf("output.urdf", "test_robot")

            # Debug output if assertion fails
            # print(urdf_str)  # noqa: T201

            assert 'robot name="test_robot"' in urdf_str
            assert 'link name="link1"' in urdf_str
            assert 'link name="link2"' in urdf_str

            # Check for joint_1 instead of joint_0 because link1 is root (body_id 1)
            # and joint_0 is attached to link1.
            # Wait, body_jntadr=[ -1, 0, 1]. body 1 (link1) has joint 0.
            # In _build_children:
            #   Iterate children of root (link1).
            #   Link1 is found as root in _find_root_body because it's first child
            #   of world.
            #   So root link is created for Link1.
            #   Then _build_children for Link1.
            #   Child is Link2 (parent=1).
            #   Creates joint between Link1 and Link2.
            #   Joint is defined by child body (Link2). Link2 has joint 1.
            #   So we should see joint_1.

            # The joint attached to Link1 (joint 0) is effectively the "root joint"
            # connecting to world.
            # URDF roots don't have joints.
            # But wait, MJCF uses joints inside bodies to define DOF.
            # If Link1 is floating base, it has free joint.
            # Here Link1 has HINGE joint (joint 0).
            # If it's attached to world (parent 0), it should be connected to world
            # with a fixed joint or the robot base link should be world?
            # Typically URDF robot "base_link" is the first link.
            # If MJCF "world" is 0, and Link1 is 1 attached to 0.
            # Exporter finds root body = Link1.
            # Creates Link1.
            # It does NOT create a joint for Link1 because it's the root of the
            # URDF tree.
            # So joint 0 is LOST in this export unless we have a dummy world link.
            # But normally URDF roots are floating or fixed to world implicitly.

            # Joint 1 connects Link1 -> Link2.
            assert 'joint name="joint_1"' in urdf_str
            assert 'type="prismatic"' in urdf_str


def test_export_model_to_urdf_function(mock_mujoco_model):
    # Test convenience function
    with (
        patch("mujoco.mj_id2name", return_value="test_name"),
        patch("pathlib.Path.write_text"),
        patch("mujoco.MjData", autospec=True),
    ):
        urdf_str = export_model_to_urdf(mock_mujoco_model, "out.urdf")
        assert len(urdf_str) > 0


def test_urdf_importer_import(sample_urdf_xml):
    importer = URDFImporter()

    with (
        patch("defusedxml.ElementTree.parse") as mock_parse,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_tree = MagicMock()
        mock_tree.getroot.return_value = ET.fromstring(sample_urdf_xml)
        mock_parse.return_value = mock_tree

        mjcf_str = importer.import_from_urdf("input.urdf")

        assert '<mujoco model="test_robot">' in mjcf_str
        assert '<body name="base_link"' in mjcf_str
        assert '<body name="child_link"' in mjcf_str
        assert 'type="hinge"' in mjcf_str
        assert 'pos="1 0 0"' in mjcf_str  # From joint origin


def test_import_urdf_to_mujoco_function(sample_urdf_xml):
    with (
        patch("defusedxml.ElementTree.parse") as mock_parse,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_tree = MagicMock()
        mock_tree.getroot.return_value = ET.fromstring(sample_urdf_xml)
        mock_parse.return_value = mock_tree

        mjcf_str = import_urdf_to_mujoco("input.urdf")
        assert len(mjcf_str) > 0


def test_importer_file_not_found():
    importer = URDFImporter()
    with pytest.raises(FileNotFoundError):
        importer.import_from_urdf("nonexistent.urdf")


def test_exporter_no_root_body(mock_mujoco_model):
    # Setup model so find_root_body returns None
    mock_mujoco_model.nbody = 1  # Only world

    with patch("mujoco.MjData", autospec=True):
        exporter = URDFExporter(mock_mujoco_model)
        with (
            patch("mujoco.mj_id2name", return_value="world"),
            patch("pathlib.Path.write_text"),
        ):
            urdf_str = exporter.export_to_urdf("out.urdf")
            # Should just be empty robot tag basically
            assert "<robot" in urdf_str
            assert "<link" not in urdf_str
