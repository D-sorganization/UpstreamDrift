"""Tests for Drake/SDFormat model loading in the model explorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.model_explorer.sdf_loader import SdfLoader

pytestmark = [pytest.mark.unit]


_SDF_16 = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="two_link_arm">
    <link name="base">
      <inertial>
        <pose>0.1 0.2 0.3 0 0 0</pose>
        <mass>2.5</mass>
        <inertia>
          <ixx>0.4</ixx><iyy>0.5</iyy><izz>0.6</izz>
          <ixy>0.01</ixy><ixz>0.02</ixz><iyz>0.03</iyz>
        </inertia>
      </inertial>
      <visual name="base_visual">
        <pose>1 2 3 0.1 0.2 0.3</pose>
        <geometry><box><size>0.2 0.3 0.4</size></box></geometry>
        <material><diffuse>0.1 0.2 0.3 1</diffuse></material>
      </visual>
      <collision name="base_collision">
        <geometry><sphere><radius>0.7</radius></sphere></geometry>
      </collision>
    </link>
    <link name="tool">
      <inertial>
        <mass>1.0</mass>
        <inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia>
      </inertial>
      <visual name="tool_visual">
        <geometry><mesh><uri>meshes/tool.dae</uri><scale>2 3 4</scale></mesh></geometry>
      </visual>
    </link>
    <joint name="base_to_tool" type="revolute">
      <parent>base</parent>
      <child>tool</child>
      <pose>0 0 0.5 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-1.5</lower><upper>1.5</upper><effort>20</effort><velocity>3</velocity></limit>
        <dynamics><damping>0.2</damping><friction>0.05</friction></dynamics>
      </axis>
    </joint>
  </model>
</sdf>
"""


_SDF_18_RELATIVE_POSES = """<?xml version="1.0"?>
<sdf version="1.8">
  <model name="relative_pose_model">
    <link name="base">
      <pose>1 0 0 0 0 0</pose>
      <inertial><mass>1</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia></inertial>
    </link>
    <link name="arm">
      <pose relative_to="base">0.25 0.5 0.75 0.1 0.2 0.3</pose>
      <inertial><mass>1</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia></inertial>
    </link>
    <joint name="base_to_arm" type="fixed">
      <parent>base</parent>
      <child>arm</child>
      <pose relative_to="base">0.25 0.5 0.75 0.1 0.2 0.3</pose>
    </joint>
  </model>
</sdf>
"""


_SDF_COMPOSITE = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="composite">
    <link name="base"><inertial><mass>1</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia></inertial></link>
    <link name="middle"><inertial><mass>1</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia></inertial></link>
    <link name="tool"><inertial><mass>1</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia></inertial></link>
    <joint name="ball_mount" type="ball"><parent>base</parent><child>middle</child></joint>
    <joint name="universal_mount" type="universal"><parent>middle</parent><child>tool</child></joint>
  </model>
</sdf>
"""


def test_loads_sdf_16_links_joints_inertials_and_geometry(tmp_path: Path) -> None:
    sdf_path = tmp_path / "arm.sdf"
    sdf_path.write_text(_SDF_16, encoding="utf-8")

    model = SdfLoader().load(sdf_path)

    assert model.name == "two_link_arm"
    assert model.metadata["source_format"] == "sdf"
    assert model.metadata["sdf_version"] == "1.6"
    links = {link.name: link for link in model.links}
    assert set(links) == {"base", "tool"}
    assert links["base"].inertia.mass == 2.5
    assert links["base"].inertia.center_of_mass == (0.1, 0.2, 0.3)
    assert links["base"].visual_geometry is not None
    assert links["base"].visual_geometry.geometry_type == "box"
    assert links["base"].visual_geometry.dimensions == (0.2, 0.3, 0.4)
    assert links["base"].collision_geometry is not None
    assert links["base"].collision_geometry.geometry_type == "sphere"
    assert links["tool"].visual_geometry is not None
    assert links["tool"].visual_geometry.mesh_filename == "meshes/tool.dae"
    assert links["tool"].visual_geometry.mesh_scale == (2.0, 3.0, 4.0)

    joints = {joint.name: joint for joint in model.joints}
    assert set(joints) == {"base_to_tool"}
    joint = joints["base_to_tool"]
    assert joint.joint_type == "revolute"
    assert joint.parent == "base"
    assert joint.child == "tool"
    assert joint.origin.xyz == (0.0, 0.0, 0.5)
    assert joint.axis == (0.0, 1.0, 0.0)
    assert joint.limits is not None
    assert joint.limits.lower == -1.5
    assert joint.limits.upper == 1.5
    assert joint.dynamics.damping == 0.2
    assert joint.dynamics.friction == 0.05
    model.require_valid()


def test_sdf_18_relative_to_poses_resolve_against_named_frames(tmp_path: Path) -> None:
    sdf_path = tmp_path / "relative.sdf"
    sdf_path.write_text(_SDF_18_RELATIVE_POSES, encoding="utf-8")

    model = SdfLoader().load(sdf_path)

    arm = next(link for link in model.links if link.name == "arm")
    joint = next(joint for joint in model.joints if joint.name == "base_to_arm")
    assert arm.visual_origin.xyz == (1.25, 0.5, 0.75)
    assert joint.origin.xyz == (1.25, 0.5, 0.75)
    assert joint.origin.rpy == (0.1, 0.2, 0.3)


def test_expands_ball_and_universal_sdf_joints_to_urdf_compatible_chains(
    tmp_path: Path,
) -> None:
    sdf_path = tmp_path / "composite.sdf"
    sdf_path.write_text(_SDF_COMPOSITE, encoding="utf-8")

    model = SdfLoader().load(sdf_path)

    link_names = {link.name for link in model.links}
    joint_names = {joint.name for joint in model.joints}
    assert {"ball_mount_intermediate_1", "ball_mount_intermediate_2"} <= link_names
    assert "universal_mount_intermediate" in link_names
    assert {"ball_mount_dof1", "ball_mount_dof2", "ball_mount_dof3"} <= joint_names
    assert {"universal_mount_dof1", "universal_mount_dof2"} <= joint_names
    model.require_valid()


def test_warns_and_loads_first_model_when_sdf_contains_multiple_models(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sdf_path = tmp_path / "multi.sdf"
    sdf_path.write_text(
        """<sdf version="1.6">
        <model name="first"><link name="base"/></model>
        <model name="second"><link name="ignored"/></model>
        </sdf>""",
        encoding="utf-8",
    )

    model = SdfLoader().load(sdf_path)

    assert model.name == "first"
    assert "multiple <model> elements" in caplog.text


def test_missing_link_references_fail_closed(tmp_path: Path) -> None:
    sdf_path = tmp_path / "bad.sdf"
    sdf_path.write_text(
        """<sdf version="1.6"><model name="bad">
        <link name="base"/>
        <joint name="missing_child" type="fixed"><parent>base</parent><child>tool</child></joint>
        </model></sdf>""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown child link 'tool'"):
        SdfLoader().load(sdf_path)
