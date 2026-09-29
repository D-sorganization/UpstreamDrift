"""Tests for Pinocchio utility modules.

Tests for MatDatasetParser and URDFExporter classes.
Issue #1741: Populate Pinocchio test directories.
"""

from pathlib import Path

import pytest
from src.engines.physics_engines.pinocchio.python.dtack.utils.mat_dataset_parser import (
    MatDatasetParser,
)


class TestMatDatasetParser:
    """Tests for the proprietary .gpcap parser stub."""

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        missing_file = tmp_path / "nonexistent.gpcap"
        with pytest.raises(FileNotFoundError, match="File not found"):
            MatDatasetParser.load(missing_file)

    def test_load_stub_raises_runtime_error(self, tmp_path: Path) -> None:
        """Test that the stub raises RuntimeError with guidance message."""
        gpcap_file = tmp_path / "test.gpcap"
        gpcap_file.write_bytes(b"\x00\x01\x02")

        with pytest.raises(RuntimeError, match="not yet implemented"):
            MatDatasetParser.load(gpcap_file)

    def test_load_accepts_string_path(self, tmp_path: Path) -> None:
        """Test that string paths are accepted."""
        gpcap_file = tmp_path / "test.gpcap"
        gpcap_file.write_bytes(b"\x00")

        with pytest.raises(RuntimeError):
            MatDatasetParser.load(str(gpcap_file))


class TestURDFExporter:
    """Tests for the URDF exporter from YAML specifications."""

    @pytest.fixture
    def minimal_yaml_spec(self, tmp_path: Path) -> Path:
        """Create a minimal YAML spec for testing."""
        yaml_content = """
root:
  name: base_link
  mass: 10.0
  inertia:
    ixx: 0.1
    ixy: 0.0
    ixz: 0.0
    iyy: 0.1
    iyz: 0.0
    izz: 0.1
  geometry:
    type: box
    size: [0.2, 0.2, 0.2]
    visual_rgba: [0.5, 0.5, 0.5, 1.0]

segments:
  - name: upper_arm
    parent: base_link
    mass: 3.0
    inertia:
      ixx: 0.05
      ixy: 0.0
      ixz: 0.0
      iyy: 0.05
      iyz: 0.0
      izz: 0.05
    joint:
      type: revolute
      axis: [0, 0, 1]
      limits: [-1.57, 1.57]
      damping: 0.5
    origin:
      xyz: [0, 0, 0.3]
      rpy: [0, 0, 0]
    geometry:
      type: cylinder
      size: [0.05, 0.2]
      visual_rgba: [0.8, 0.3, 0.3, 1.0]
"""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")
        return yaml_file

    @pytest.fixture
    def gimbal_yaml_spec(self, tmp_path: Path) -> Path:
        """Create a YAML spec with gimbal joint for testing."""
        yaml_content = """
root:
  name: torso
  mass: 20.0
  inertia:
    ixx: 0.5
    ixy: 0.0
    ixz: 0.0
    iyy: 0.5
    iyz: 0.0
    izz: 0.5
  geometry:
    type: box
    size: [0.3, 0.2, 0.4]
    visual_rgba: [0.4, 0.4, 0.8, 1.0]

segments:
  - name: shoulder
    parent: torso
    mass: 2.0
    inertia:
      ixx: 0.02
      ixy: 0.0
      ixz: 0.0
      iyy: 0.02
      iyz: 0.0
      izz: 0.02
    joint:
      type: gimbal
    origin:
      xyz: [0, 0.15, 0.3]
    geometry:
      type: sphere
      size: 0.05
      visual_rgba: [0.3, 0.8, 0.3, 1.0]
"""
        yaml_file = tmp_path / "gimbal_model.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")
        return yaml_file

    def test_exporter_init(self, minimal_yaml_spec: Path) -> None:
        """Test exporter initialization loads YAML."""
        from src.engines.physics_engines.pinocchio.python.dtack.utils.urdf_exporter import (
            URDFExporter,
        )

        exporter = URDFExporter(minimal_yaml_spec)
        assert exporter.spec is not None
        assert "root" in exporter.spec

    def test_export_produces_valid_urdf(
        self, minimal_yaml_spec: Path, tmp_path: Path
    ) -> None:
        """Test export produces a valid URDF file."""
        from src.engines.physics_engines.pinocchio.python.dtack.utils.urdf_exporter import (
            URDFExporter,
        )

        exporter = URDFExporter(minimal_yaml_spec)
        output = tmp_path / "output.urdf"
        exporter.export(output)

        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert '<?xml version="1.0"?>' in content
        assert '<robot name="golfer">' in content
        assert "</robot>" in content
        assert '<link name="base_link">' in content
        assert '<link name="upper_arm">' in content
        assert "revolute" in content

    def test_export_includes_joint_properties(
        self, minimal_yaml_spec: Path, tmp_path: Path
    ) -> None:
        """Test export includes joint limits, damping, and axis."""
        from src.engines.physics_engines.pinocchio.python.dtack.utils.urdf_exporter import (
            URDFExporter,
        )

        exporter = URDFExporter(minimal_yaml_spec)
        output = tmp_path / "output.urdf"
        exporter.export(output)

        content = output.read_text(encoding="utf-8")
        assert 'axis xyz="0 0 1"' in content
        assert "damping" in content
        assert "limit" in content

    def test_export_gimbal_joint(self, gimbal_yaml_spec: Path, tmp_path: Path) -> None:
        """Test export of gimbal joint creates 3 revolute joints."""
        from src.engines.physics_engines.pinocchio.python.dtack.utils.urdf_exporter import (
            URDFExporter,
        )

        exporter = URDFExporter(gimbal_yaml_spec)
        output = tmp_path / "gimbal.urdf"
        exporter.export(output)

        content = output.read_text(encoding="utf-8")
        # Gimbal should create intermediate links
        assert "gimbal_z" in content
        assert "gimbal_y" in content
        # Should have 3 revolute joints for gimbal
        assert content.count('type="revolute"') == 3

    def test_parse_origin_none(self, minimal_yaml_spec: Path) -> None:
        """Test _parse_origin returns defaults for None."""
        from src.engines.physics_engines.pinocchio.python.dtack.utils.urdf_exporter import (
            URDFExporter,
        )

        exporter = URDFExporter(minimal_yaml_spec)
        xyz, rpy = exporter._parse_origin(None)
        assert xyz == [0.0, 0.0, 0.0]
        assert rpy == [0.0, 0.0, 0.0]

    def test_parse_origin_with_values(self, minimal_yaml_spec: Path) -> None:
        """Test _parse_origin extracts values correctly."""
        from src.engines.physics_engines.pinocchio.python.dtack.utils.urdf_exporter import (
            URDFExporter,
        )

        exporter = URDFExporter(minimal_yaml_spec)
        origin = {"xyz": [1.0, 2.0, 3.0], "rpy": [0.1, 0.2, 0.3]}
        xyz, rpy = exporter._parse_origin(origin)
        assert xyz == [1.0, 2.0, 3.0]
        assert rpy == [0.1, 0.2, 0.3]

    def test_massless_inertial(self, minimal_yaml_spec: Path) -> None:
        """Test massless inertial generation for intermediate links."""
        from src.engines.physics_engines.pinocchio.python.dtack.utils.urdf_exporter import (
            URDFExporter,
        )

        exporter = URDFExporter(minimal_yaml_spec)
        lines = exporter._generate_massless_inertial()
        text = "\n".join(lines)
        assert "0.001" in text
        assert "0.0001" in text

    def test_constants_defined(self) -> None:
        """Test that safety constants are properly defined."""
        from src.engines.physics_engines.pinocchio.python.dtack.utils.urdf_exporter import (
            JOINT_LIMIT_COUNT,
            MAX_EFFORT_NM,
            MAX_VELOCITY_RAD_S,
            MIN_GIMBAL_DOFS,
            MIN_UNIVERSAL_DOFS,
        )

        assert MAX_EFFORT_NM == 1000.0
        assert MAX_VELOCITY_RAD_S == 10.0
        assert MIN_UNIVERSAL_DOFS == 2
        assert MIN_GIMBAL_DOFS == 3
        assert JOINT_LIMIT_COUNT == 2
