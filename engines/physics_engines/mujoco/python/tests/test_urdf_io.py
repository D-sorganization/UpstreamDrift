"""Tests for URDF import/export functionality."""

import math
import tempfile
import xml.etree.ElementTree as ET  # noqa: N817, RUF100
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mujoco_humanoid_golf.models import DOUBLE_PENDULUM_XML
from mujoco_humanoid_golf.urdf_io import (
    URDFExporter,
    URDFImporter,
    export_model_to_urdf,
    import_urdf_to_mujoco,
)


class TestURDFExporter:
    """Tests for URDF export functionality."""

    def test_export_double_pendulum(self) -> None:
        """Test exporting double pendulum model to URDF."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        exporter = URDFExporter(model)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            urdf_path = Path(f.name)

        try:
            urdf_xml = exporter.export_to_urdf(urdf_path)

            # Check that URDF was created
            assert urdf_path.exists()
            assert len(urdf_xml) > 0
            assert "robot" in urdf_xml.lower()
            assert "link" in urdf_xml.lower()

        finally:
            if urdf_path.exists():
                urdf_path.unlink()

    def test_export_with_visual_and_collision(self) -> None:
        """Test exporting with visual and collision geometries."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        exporter = URDFExporter(model)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            urdf_path = Path(f.name)

        try:
            urdf_xml = exporter.export_to_urdf(
                urdf_path,
                include_visual=True,
                include_collision=True,
            )

            assert "visual" in urdf_xml.lower()
            assert "collision" in urdf_xml.lower()

        finally:
            if urdf_path.exists():
                urdf_path.unlink()

    def test_export_convenience_function(self) -> None:
        """Test convenience function for export."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            urdf_path = Path(f.name)

        try:
            urdf_xml = export_model_to_urdf(model, urdf_path)

            assert urdf_path.exists()
            assert len(urdf_xml) > 0

        finally:
            if urdf_path.exists():
                urdf_path.unlink()

    def test_export_finds_root_body(self) -> None:
        """Test that exporter correctly finds root body."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        exporter = URDFExporter(model)

        root_body_id = exporter._find_root_body()
        assert root_body_id is not None
        assert root_body_id > 0  # Not worldbody


class TestURDFImporter:
    """Tests for URDF import functionality."""

    def test_import_simple_urdf(self) -> None:
        """Test importing a simple URDF file."""
        # Create a minimal URDF
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
  </link>
</robot>
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            urdf_path = Path(f.name)
            f.write(urdf_content)

        try:
            importer = URDFImporter()
            mujoco_xml = importer.import_from_urdf(urdf_path)

            assert len(mujoco_xml) > 0
            assert "mujoco" in mujoco_xml.lower()
            assert "body" in mujoco_xml.lower()

            # Try to load in MuJoCo
            model = mujoco.MjModel.from_xml_string(mujoco_xml)
            assert model.nbody > 1  # At least base_link + worldbody

        finally:
            if urdf_path.exists():
                urdf_path.unlink()

    def test_import_with_joint(self) -> None:
        """Test importing URDF with a joint."""
        # Use math.pi for joint limits to avoid magic numbers
        pi_lower = f"-{math.pi:.2f}"
        pi_upper = f"{math.pi:.2f}"
        urdf_content = f"""<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="{pi_lower}" upper="{pi_upper}"/>
  </joint>
</robot>
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            urdf_path = Path(f.name)
            f.write(urdf_content)

        try:
            importer = URDFImporter()
            mujoco_xml = importer.import_from_urdf(urdf_path)

            # Check that joint was created
            assert "joint" in mujoco_xml.lower()

            # Try to load in MuJoCo
            model = mujoco.MjModel.from_xml_string(mujoco_xml)
            assert model.njnt > 0

        finally:
            if urdf_path.exists():
                urdf_path.unlink()

    def test_import_convenience_function(self) -> None:
        """Test convenience function for import."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            urdf_path = Path(f.name)
            f.write(urdf_content)

        try:
            mujoco_xml = import_urdf_to_mujoco(urdf_path)

            assert len(mujoco_xml) > 0
            assert "mujoco" in mujoco_xml.lower()

        finally:
            if urdf_path.exists():
                urdf_path.unlink()

    def test_import_nonexistent_file(self) -> None:
        """Test that importing nonexistent file raises error."""
        importer = URDFImporter()

        with pytest.raises(FileNotFoundError):
            importer.import_from_urdf("nonexistent.urdf")


class TestURDFRoundTrip:
    """Tests for round-trip conversion (MuJoCo -> URDF -> MuJoCo)."""

    def test_round_trip_double_pendulum(self) -> None:
        """Test round-trip conversion of double pendulum."""
        # Export to URDF
        model1 = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            urdf_path = Path(f.name)

        try:
            export_model_to_urdf(model1, urdf_path)

            # Import back
            mujoco_xml = import_urdf_to_mujoco(urdf_path)

            # Load in MuJoCo
            model2 = mujoco.MjModel.from_xml_string(mujoco_xml)

            # Basic checks
            assert model2.nbody > 0
            assert model2.ngeom > 0

        finally:
            if urdf_path.exists():
                urdf_path.unlink()


class TestURDFGeometryConversion:
    """Tests for geometry conversion between formats."""

    def test_quat_to_rpy_conversion(self) -> None:
        """Test quaternion to RPY conversion."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        exporter = URDFExporter(model)

        # Test identity quaternion
        quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        rpy = exporter._quat_to_rpy(quat)

        assert len(rpy) == 3
        assert np.allclose(rpy, [0, 0, 0], atol=1e-6)

    def test_geometry_element_creation(self) -> None:
        """Test creating geometry elements from MuJoCo geoms."""
        model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
        exporter = URDFExporter(model)

        # Find a geom
        for geom_id in range(model.ngeom):
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                parent = ET.Element("geometry")
                geom_elem = exporter._create_geometry_element(geom_id, parent)

                assert geom_elem is not None
                assert geom_elem.tag == "box"
                break
