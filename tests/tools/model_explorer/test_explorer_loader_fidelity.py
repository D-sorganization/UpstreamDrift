"""Loader fidelity the explorer relies on to show the canonical models (#12878)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.model_generation.library.unified_loader import (
    UnifiedModelLoader,
)
from src.tools.model_explorer.osim_loader import OsimLoader

pytestmark = [pytest.mark.unit]

# A body carrying three hinges (shoulder-style 3-DOF), as the full-body MJCF does.
MULTI_JOINT_MJCF = """
<mujoco model="three_dof">
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <inertial pos="0 0 0" mass="5" diaginertia="0.1 0.1 0.1"/>
      <body name="shoulder" pos="0 0.2 0.3">
        <joint name="sh_x" type="hinge" axis="1 0 0" range="-1 1"/>
        <joint name="sh_y" type="hinge" axis="0 1 0" range="-1 1"/>
        <joint name="sh_z" type="hinge" axis="0 0 1" range="-1 1"/>
        <inertial pos="0 0 0" mass="2" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

OSIM_WITH_CONTACT = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40000">
  <Model name="contact_model">
    <BodySet><objects>
      <Body name="foot"><mass>1</mass><mass_center>0 0 0</mass_center>
        <inertia_xx>1</inertia_xx><inertia_yy>1</inertia_yy><inertia_zz>1</inertia_zz>
      </Body>
    </objects></BodySet>
    <JointSet><objects>
      <WeldJoint name="weld"><parent_body>ground</parent_body>
        <location_in_parent>0 0 0</location_in_parent>
        <orientation_in_parent>0 0 0</orientation_in_parent>
        <location>0 0 0</location><orientation>0 0 0</orientation>
      </WeldJoint>
    </objects></JointSet>
    <ForceSet><objects>
      <HuntCrossleyForce name="contact_heel">
        <HuntCrossleyForce::ContactParametersSet name="contact_parameters">
          <objects/>
        </HuntCrossleyForce::ContactParametersSet>
      </HuntCrossleyForce>
    </objects></ForceSet>
  </Model>
</OpenSimDocument>
"""


def test_mjcf_body_with_three_hinges_keeps_all_three_joints(tmp_path: Path) -> None:
    path = tmp_path / "three_dof.xml"
    path.write_text(MULTI_JOINT_MJCF, encoding="utf-8")

    result = UnifiedModelLoader(prefs_dir=tmp_path / "prefs").load_file(path)

    assert result.success and result.model is not None
    names = {joint.name for joint in result.model.joints}
    assert {"sh_x", "sh_y", "sh_z"} <= names
    assert {j.child for j in result.model.joints if j.name == "sh_z"} == {"shoulder"}
    link_names = {link.name for link in result.model.links}
    assert {joint.parent for joint in result.model.joints} <= link_names
    assert {joint.child for joint in result.model.joints} <= link_names


def test_mjcf_single_hinge_body_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "one.xml"
    path.write_text(
        '<mujoco model="one"><worldbody><body name="a"><body name="b">'
        '<joint name="j" type="hinge"/></body></body></worldbody></mujoco>',
        encoding="utf-8",
    )

    model = UnifiedModelLoader(prefs_dir=tmp_path / "prefs").load_file(path).model

    assert model is not None
    assert [link.name for link in model.links] == ["a", "b"]
    assert [joint.name for joint in model.joints] == ["j"]


def test_osim_namespaced_opensim_tag_names_do_not_break_loading(
    tmp_path: Path,
) -> None:
    """OpenSim itself writes ``Class::Nested`` tag names, which are not XML names."""
    path = tmp_path / "contact.osim"
    path.write_text(OSIM_WITH_CONTACT, encoding="utf-8")

    model = OsimLoader().load(path)

    assert "foot" in {link.name for link in model.links}
