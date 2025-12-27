"""
Unit tests for URDF I/O module.
"""

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.urdf_io import (
    URDFExporter,
    URDFImporter,
    export_model_to_urdf,
    import_urdf_to_mujoco,
)


@pytest.fixture
def sample_urdf(tmp_path):
    """Create a sample URDF file."""
    urdf_content = """<?xml version="1.0" ?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
  </link>
  <link name="child_link">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <origin rpy="0 0 0" xyz="0 0 1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="10"/>
  </joint>
</robot>
"""
    urdf_path = tmp_path / "test_robot.urdf"
    urdf_path.write_text(urdf_content)
    return urdf_path


@pytest.fixture
def mock_mujoco_model():
    """Create a mock MuJoCo model."""
    # Since we can't easily construct a valid MjModel without XML parsing,
    # we'll mock the attributes needed by URDFExporter.
    model = MagicMock(spec=mujoco.MjModel)
    model.nbody = 2
    model.ngeom = 2

    # Body properties
    model.body_mass = np.array([0, 1.0, 0.5])  # ID 0 is world
    model.body_pos = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    model.body_ipos = np.zeros((3, 3))
    model.body_inertia = np.ones((3, 3))
    model.body_parentid = np.array([0, 0, 1])
    model.body_jntadr = np.array(
        [-1, -1, 0]
    )  # Body 0 has no joint, 1 has no joint, 2 has joint 0

    # Joint properties
    model.jnt_type = np.array([mujoco.mjtJoint.mjJNT_HINGE])
    model.jnt_pos = np.array([[0, 0, 1]])
    model.jnt_axis = np.array([[0, 0, 1]])
    model.jnt_range = np.array([[-1.57, 1.57]])
    model.jnt_limited = np.array([True])

    # Geom properties
    model.geom_type = np.array([mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_BOX])
    model.geom_size = np.array([[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]])
    model.geom_pos = np.zeros((2, 3))
    model.geom_quat = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])
    model.geom_bodyid = np.array([1, 2])
    model.geom_matid = np.array([-1, -1])

    return model


class TestURDFImporter:
    """Test suite for URDFImporter."""

    def test_import_from_urdf(self, sample_urdf):
        """Test importing URDF to MJCF XML."""
        importer = URDFImporter()
        mjcf_xml = importer.import_from_urdf(sample_urdf)

        # Verify basic structure
        # defusedxml parse returns an ElementTree, but import_from_urdf calls tostring on custom root
        # so it returns a string.
        assert isinstance(mjcf_xml, str)
        assert "<mujoco" in mjcf_xml

        root = ET.fromstring(mjcf_xml)
        assert root.tag == "mujoco"
        assert root.get("model") == "test_robot"

        # Check bodies
        bodies = root.findall(".//body")
        # Should have base_link and child_link
        body_names = [b.get("name") for b in bodies]
        assert "base_link" in body_names
        assert "child_link" in body_names

        # Check joints
        joints = root.findall(".//joint")
        assert len(joints) == 1
        assert joints[0].get("name") == "joint1"
        assert joints[0].get("type") == "hinge"


class TestURDFExporter:
    """Test suite for URDFExporter."""

    @patch("mujoco.mj_id2name")
    @patch("mujoco.MjData")
    def test_export_to_urdf(
        self, mock_mjdata, mock_id2name, mock_mujoco_model, tmp_path
    ):
        """Test exporting MJCF to URDF."""

        # Mock id2name to return reasonable names based on ID
        def id2name_side_effect(model, obj_type, obj_id):
            if obj_type == mujoco.mjtObj.mjOBJ_BODY:
                # 0 is world, 1 is base, 2 is child
                if obj_id == 0:
                    return "world"
                if obj_id == 1:
                    return "base_link"
                if obj_id == 2:
                    return "child_link"
            elif obj_type == mujoco.mjtObj.mjOBJ_JOINT:
                return f"joint_{obj_id}"
            return None

        mock_id2name.side_effect = id2name_side_effect

        exporter = URDFExporter(mock_mujoco_model)

        output_path = tmp_path / "exported.urdf"
        urdf_str = exporter.export_to_urdf(output_path, model_name="test_export")

        assert output_path.exists()
        assert 'robot name="test_export"' in urdf_str
        assert 'link name="base_link"' in urdf_str
        assert 'link name="child_link"' in urdf_str

        # Since body 2 (child_link) is child of 1 (base_link), checking for joint logic
        # In _build_children: parent is base_link (id 1).
        # Iterates children. Finds 2 (parentid[2]==1).
        # Calls _create_joint(1, 2).
        # child_jntadr[2] is 0. Joint 0 exists.
        # Joint 0 name is joint_0.

        assert 'joint name="joint_0"' in urdf_str


def test_convenience_functions(sample_urdf, mock_mujoco_model, tmp_path):
    """Test convenience functions."""
    # Import
    mjcf = import_urdf_to_mujoco(sample_urdf)
    assert "<mujoco" in mjcf

    # Export
    # We need to mock URDFExporter inside the function or pass a mock model that works
    # Using patch to avoid complexity of real exporter running on mock model
    with patch(
        "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.urdf_io.URDFExporter"
    ) as MockExporter:
        instance = MockExporter.return_value
        instance.export_to_urdf.return_value = "<robot></robot>"

        res = export_model_to_urdf(mock_mujoco_model, tmp_path / "out.urdf")
        assert res == "<robot></robot>"
