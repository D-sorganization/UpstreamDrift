"""Tests for first-party OpenSim .osim loading in the model explorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_generation.core.types import GeometryType, JointType
from src.tools.model_explorer.osim_loader import OsimLoader

pytestmark = [pytest.mark.unit]


OSIM_3 = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="30500">
  <Model name="legacy_slider">
    <BodySet>
      <objects>
        <Body name="block">
          <mass>2.5</mass>
          <mass_center>0.1 0.2 0.3</mass_center>
          <inertia>1 2 3 0.1 0.2 0.3</inertia>
          <VisibleObject>
            <geometry_files>block.vtp</geometry_files>
          </VisibleObject>
        </Body>
      </objects>
    </BodySet>
    <JointSet>
      <objects>
        <SliderJoint name="slide">
          <parent_body>ground</parent_body>
          <location_in_parent>1 2 3</location_in_parent>
          <orientation_in_parent>0.1 0.2 0.3</orientation_in_parent>
          <CoordinateSet>
            <objects>
              <Coordinate name="tx">
                <motion_type>translational</motion_type>
                <range>-0.25 0.75</range>
              </Coordinate>
            </objects>
          </CoordinateSet>
          <body>block</body>
        </SliderJoint>
      </objects>
    </JointSet>
  </Model>
</OpenSimDocument>
"""


OSIM_4 = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40500">
  <Model name="modern_arm">
    <Ground name="ground" />
    <BodySet name="bodyset">
      <objects>
        <Body name="upper">
          <components>
            <PhysicalOffsetFrame name="upper_geom_frame">
              <attached_geometry>
                <Mesh name="upper_mesh">
                  <scale_factors>1 2 3</scale_factors>
                  <mesh_file>upper.stl</mesh_file>
                </Mesh>
              </attached_geometry>
              <socket_parent>/bodyset/upper</socket_parent>
              <translation>0 0 0</translation>
              <orientation>0 0 0</orientation>
            </PhysicalOffsetFrame>
          </components>
          <mass>3</mass>
          <mass_center>0 0.1 0</mass_center>
          <inertia>0.2 0.3 0.4 0 0 0</inertia>
        </Body>
        <Body name="hand">
          <mass>1</mass>
          <mass_center>0 0 0</mass_center>
          <inertia>0.1 0.1 0.1 0 0 0</inertia>
        </Body>
      </objects>
    </BodySet>
    <JointSet name="jointset">
      <objects>
        <PinJoint name="shoulder">
          <socket_parent_frame>ground_offset</socket_parent_frame>
          <socket_child_frame>upper_offset</socket_child_frame>
          <coordinates>
            <Coordinate name="q">
              <range>-1 1</range>
            </Coordinate>
          </coordinates>
          <frames>
            <PhysicalOffsetFrame name="ground_offset">
              <socket_parent>/ground</socket_parent>
              <translation>0 1 0</translation>
              <orientation>0 0 0</orientation>
            </PhysicalOffsetFrame>
            <PhysicalOffsetFrame name="upper_offset">
              <socket_parent>/bodyset/upper</socket_parent>
              <translation>0 0.5 0</translation>
              <orientation>0 0 0</orientation>
            </PhysicalOffsetFrame>
          </frames>
        </PinJoint>
        <BallJoint name="wrist">
          <socket_parent_frame>/bodyset/upper</socket_parent_frame>
          <socket_child_frame>hand_offset</socket_child_frame>
          <frames>
            <PhysicalOffsetFrame name="hand_offset">
              <socket_parent>/bodyset/hand</socket_parent>
              <translation>0 -0.2 0</translation>
              <orientation>0 0 0</orientation>
            </PhysicalOffsetFrame>
          </frames>
        </BallJoint>
      </objects>
    </JointSet>
    <ForceSet>
      <objects>
        <Thelen2003Muscle name="biceps" />
      </objects>
    </ForceSet>
    <ConstraintSet>
      <objects>
        <CoordinateCouplerConstraint name="coupler" />
      </objects>
    </ConstraintSet>
    <MarkerSet>
      <objects>
        <Marker name="marker" />
      </objects>
    </MarkerSet>
  </Model>
</OpenSimDocument>
"""


def _write(tmp_path: Path, text: str, name: str = "model.osim") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_opensim_3_parent_body_layout(tmp_path: Path) -> None:
    model = OsimLoader().load(_write(tmp_path, OSIM_3))

    assert model.name == "legacy_slider"
    assert [link.name for link in model.links] == ["ground", "block"]
    block = model.get_link("block")
    assert block is not None
    assert block.inertia.mass == pytest.approx(2.5)
    assert block.inertia.center_of_mass == pytest.approx((0.1, 0.2, 0.3))
    assert block.visual_geometry is not None
    assert block.visual_geometry.geometry_type is GeometryType.MESH
    assert block.visual_geometry.mesh_filename == "block.vtp"

    joint = model.get_joint("slide")
    assert joint is not None
    assert joint.joint_type is JointType.PRISMATIC
    assert joint.parent == "ground"
    assert joint.child == "block"
    assert joint.origin.xyz == pytest.approx((1.0, 2.0, 3.0))
    assert joint.origin.rpy == pytest.approx((0.1, 0.2, 0.3))
    assert joint.limits is not None
    assert joint.limits.lower == pytest.approx(-0.25)
    assert joint.limits.upper == pytest.approx(0.75)


def test_loads_opensim_4_socket_frame_layout_and_canonical_model(
    tmp_path: Path,
) -> None:
    loader = OsimLoader()
    parsed = loader.load(_write(tmp_path, OSIM_4))

    assert {link.name for link in parsed.links} == {"ground", "upper", "hand"}
    assert {joint.name: joint.joint_type for joint in parsed.joints} == {
        "shoulder": JointType.REVOLUTE,
        "wrist": JointType.GIMBAL,
    }
    assert parsed.get_joint("shoulder").origin.xyz == pytest.approx((0.0, 1.0, 0.0))
    assert parsed.get_joint("wrist").parent == "upper"
    assert parsed.get_joint("wrist").child == "hand"
    assert any("ForceSet" in warning for warning in parsed.warnings)
    assert any("ConstraintSet" in warning for warning in parsed.warnings)
    assert any("MarkerSet" in warning for warning in parsed.warnings)

    canonical = loader.load_canonical(_write(tmp_path, OSIM_4))
    assert canonical.name == "modern_arm"
    assert canonical.metadata["source_format"] == "opensim-osim"
    assert canonical.validate(strict=False).is_valid


def test_custom_joint_is_best_effort_with_warning(tmp_path: Path) -> None:
    custom = OSIM_4.replace("PinJoint", "CustomJoint").replace(
        'name="shoulder"', 'name="custom_shoulder"', 1
    )
    model = OsimLoader().load(_write(tmp_path, custom))

    joint = model.get_joint("custom_shoulder")
    assert joint is not None
    assert joint.joint_type is JointType.REVOLUTE
    assert any("CustomJoint 'custom_shoulder'" in warning for warning in model.warnings)


def test_invalid_root_fails_loudly(tmp_path: Path) -> None:
    path = _write(tmp_path, "<robot name='not_opensim' />")
    with pytest.raises(ValueError, match="OpenSimDocument"):
        OsimLoader().load(path)


def test_loads_real_sibling_opensim_model_when_present() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sibling = repo_root.parent / "OpenSim_Models"
    if not sibling.is_dir():
        pytest.skip("OpenSim_Models sibling repository is not present")
    candidates = sorted(sibling.rglob("*.osim"))
    if not candidates:
        pytest.skip("OpenSim_Models sibling repository has no .osim files")

    model = OsimLoader().load(candidates[0])

    assert model.links
    assert model.source_path == candidates[0]
