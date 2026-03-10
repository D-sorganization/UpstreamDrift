"""Tests for URDF import and export functionality."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import defusedxml.ElementTree as ET
import mujoco
import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.urdf_io import (
    URDFExporter,
    URDFImporter,
    export_model_to_urdf,
    import_urdf_to_mujoco,
)

# Path to the mujoco module reference inside urdf_io
_URDF_IO_MUJOCO = (
    "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.urdf_io.mujoco"
)


def _make_mock_mujoco(**overrides: Any) -> MagicMock:
    """Create a mock mujoco module that preserves enum types but stubs C functions."""
    mock = MagicMock()
    # Preserve real enum types so comparisons work
    mock.mjtObj = mujoco.mjtObj
    mock.mjtJoint = mujoco.mjtJoint
    mock.mjtGeom = mujoco.mjtGeom
    mock.mjtTrn = getattr(mujoco, "mjtTrn", MagicMock())
    for key, val in overrides.items():
        setattr(mock, key, val)
    return mock


@pytest.fixture
def mock_mujoco_model() -> MagicMock:
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
def sample_urdf_xml() -> str:
    """Return a sample URDF XML string for testing."""
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


def test_urdf_exporter_init(mock_mujoco_model: MagicMock) -> None:
    """Test URDFExporter initialization."""
    mock_mj = _make_mock_mujoco()
    mock_mj.MjData.return_value = MagicMock()

    with patch(_URDF_IO_MUJOCO, mock_mj):
        exporter = URDFExporter(mock_mujoco_model)
        assert exporter.model == mock_mujoco_model
        assert exporter.data == mock_mj.MjData.return_value


def test_export_to_urdf(mock_mujoco_model: MagicMock) -> None:
    """Test exporting a MuJoCo model to URDF format."""

    def id2name(m: Any, obj_type: int, obj_id: int) -> str:
        """Map MuJoCo object type and id to a name string."""
        if obj_type == mujoco.mjtObj.mjOBJ_BODY:
            return ["world", "link1", "link2"][obj_id]
        if obj_type == mujoco.mjtObj.mjOBJ_JOINT:
            return f"joint_{obj_id}"
        if obj_type == mujoco.mjtObj.mjOBJ_GEOM:
            return f"geom_{obj_id}"
        return "obj"

    mock_mj = _make_mock_mujoco(
        mj_id2name=MagicMock(side_effect=id2name),
    )
    mock_mj.MjData.return_value = MagicMock()

    with patch(_URDF_IO_MUJOCO, mock_mj), patch("pathlib.Path.write_text"):
        exporter = URDFExporter(mock_mujoco_model)
        urdf_str = exporter.export_to_urdf("output.urdf", "test_robot")

        assert 'robot name="test_robot"' in urdf_str
        assert 'link name="link1"' in urdf_str
        assert 'link name="link2"' in urdf_str
        assert 'joint name="joint_1"' in urdf_str
        assert 'type="prismatic"' in urdf_str


def test_export_model_to_urdf_function(mock_mujoco_model: MagicMock) -> None:
    """Test the export_model_to_urdf convenience function."""
    mock_mj = _make_mock_mujoco(
        mj_id2name=MagicMock(return_value="test_name"),
    )
    mock_mj.MjData.return_value = MagicMock()

    with patch(_URDF_IO_MUJOCO, mock_mj), patch("pathlib.Path.write_text"):
        urdf_str = export_model_to_urdf(mock_mujoco_model, "out.urdf")
        assert len(urdf_str) > 0


def test_urdf_importer_import(sample_urdf_xml: str) -> None:
    """Test importing a URDF file to MJCF format."""
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


def test_import_urdf_to_mujoco_function(sample_urdf_xml: str) -> None:
    """Test the import_urdf_to_mujoco convenience function."""
    with (
        patch("defusedxml.ElementTree.parse") as mock_parse,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_tree = MagicMock()
        mock_tree.getroot.return_value = ET.fromstring(sample_urdf_xml)
        mock_parse.return_value = mock_tree

        mjcf_str = import_urdf_to_mujoco("input.urdf")
        assert len(mjcf_str) > 0


def test_importer_file_not_found() -> None:
    """Test that importing a non-existent URDF raises FileNotFoundError."""
    importer = URDFImporter()
    with pytest.raises(FileNotFoundError):
        importer.import_from_urdf("nonexistent.urdf")


def test_exporter_no_root_body(mock_mujoco_model: MagicMock) -> None:
    """Test exporting a model with only the world body and no root."""
    mock_mujoco_model.nbody = 1  # Only world

    mock_mj = _make_mock_mujoco(
        mj_id2name=MagicMock(return_value="world"),
    )
    mock_mj.MjData.return_value = MagicMock()

    with patch(_URDF_IO_MUJOCO, mock_mj), patch("pathlib.Path.write_text"):
        exporter = URDFExporter(mock_mujoco_model)
        urdf_str = exporter.export_to_urdf("out.urdf")
        # Should just be empty robot tag basically
        assert "<robot" in urdf_str
        assert "<link" not in urdf_str
