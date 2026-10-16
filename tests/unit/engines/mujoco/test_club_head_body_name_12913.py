"""Regression for UD#12913: club speed was 0.0 for ``clubhead``-named bodies."""

from __future__ import annotations

import logging

import mujoco
import pytest
from mujoco_humanoid_golf.biomechanics import BiomechanicalAnalyzer
from mujoco_humanoid_golf.club_body_names import (
    CLUB_HEAD_BODY_ALIASES,
    find_club_head_body_id,
)

_XML = """
<mujoco><worldbody>
  <body name="arm" pos="0 0 1">
    <joint name="j" type="hinge" axis="0 1 0"/>
    <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.5"/>
    <body name="{name}" pos="0 0 -0.5">
      <geom type="sphere" size="0.03" mass="0.2"/>
    </body>
  </body>
</worldbody></mujoco>
"""


def _model(name: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_XML.format(name=name))


@pytest.mark.unit
@pytest.mark.parametrize(
    "name", ["clubhead", "club_head", "ClubHead", "club_head_link"]
)
def test_resolver_finds_all_naming_variants(name: str) -> None:
    model = _model(name)
    bid = find_club_head_body_id(model)
    assert bid is not None
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) == name


@pytest.mark.unit
@pytest.mark.parametrize("name", list(CLUB_HEAD_BODY_ALIASES))
def test_club_speed_nonzero_for_moving_club(name: str) -> None:
    model = _model(name)
    data = mujoco.MjData(model)
    data.qvel[0] = 4.0
    mujoco.mj_forward(model, data)
    _, _, speed = BiomechanicalAnalyzer(model, data).get_club_head_data()
    assert speed == pytest.approx(2.0, rel=1e-6)


@pytest.mark.unit
def test_missing_club_head_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    model = _model("elbow")
    with caplog.at_level(logging.WARNING):
        assert find_club_head_body_id(model, owner="Test") is None
    assert "no club-head body found" in caplog.text
