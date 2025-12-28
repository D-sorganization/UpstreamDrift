"""
Unit tests for URDF I/O module.
"""

import importlib
import sys
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.urdf_io import (
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
    model.nbody = 3
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
        assert len(joints) == 2  # May include additional joints created by importer
        joint_names = [j.get("name") for j in joints]
        assert "joint1" in joint_names


class TestURDFExporter:
    """Test suite for URDFExporter."""

    @patch(
        "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.urdf_io.mujoco.MjData"
    )
    def test_export_to_urdf(self, mock_mjdata_class, tmp_path):
        """Test exporting MJCF to URDF."""
        # Setup comprehensive mujoco mock in sys.modules to bypass C-extension issues
        mock_mujoco = MagicMock()
        mock_mujoco.MjModel = MagicMock
        mock_mujoco.mjtObj.mjOBJ_BODY = 1
        mock_mujoco.mjtObj.mjOBJ_JOINT = 2
        mock_mujoco.mjtObj.mjOBJ_MODEL = 3
        mock_mujoco.mjtJoint.mjJNT_HINGE = 0  # Define this for the mock model

        # Mock id2name logic - need to handle the actual MjModel mock being passed
        def id2name_side_effect(model, obj_type, obj_id):
            # Convert obj_type to int if it's a mock object
            if hasattr(obj_type, "value"):
                obj_type = obj_type.value
            elif not isinstance(obj_type, int):
                obj_type = int(obj_type)

            if obj_type == 1:  # mjOBJ_BODY
                if obj_id == 0:
                    return "world"
                if obj_id == 1:
                    return "base_link"
                if obj_id == 2:
                    return "child_link"
            elif obj_type == 2:  # mjOBJ_JOINT
                return f"joint_{obj_id}"
            return None

        mock_mujoco.mj_id2name.side_effect = id2name_side_effect

        with patch.dict(sys.modules, {"mujoco": mock_mujoco}):
            # Re-import to pick up the mock
            importlib.reload(
                sys.modules[
                    "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.urdf_io"
                ]
            )
            from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.urdf_io import (
                URDFExporter,
                mujoco,
            )

            mock_mujoco_model = MagicMock()
            mock_mujoco_model.nbody = 3
            mock_mujoco_model.body_jntadr = [-1, -1, 0]  # Child has joint at index 0
            mock_mujoco_model.jnt_type = [
                mujoco.mjtJoint.mjJNT_HINGE
            ]  # Joint 0 is hinge
            mock_mujoco_model.body_parentid = [0, 0, 1]
            mock_mujoco_model.body_mass = [0, 1.0, 1.0]
            mock_mujoco_model.body_inertia = np.zeros((3, 3))
            mock_mujoco_model.body_ipos = np.zeros((3, 3))
            mock_mujoco_model.ngeom = 0
            mock_mujoco_model.jnt_pos = np.zeros((1, 3))
            mock_mujoco_model.jnt_axis = np.array([[0, 0, 1]])
            mock_mujoco_model.jnt_limited = [False]
            mock_mujoco_model.jnt_range = np.zeros((1, 2))

            # Constants that might have been lost in reload/mock
            # Ensure URDFExporter uses the mocked enums/constants

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
