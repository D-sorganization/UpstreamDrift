"""Address pose of an address-only pipeline report as MuJoCo qpos (SHOW-C7, #14563)."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from src.shared.python.motion_matching.address_pose import address_qpos

pytestmark = pytest.mark.unit

XML = """
<mujoco><worldbody><body name="b"><joint name="TranslationInputX" type="slide" axis="1 0 0"/><geom size="0.05"/>
<body name="c"><joint name="HipInputX" type="hinge" axis="1 0 0"/>
<geom size="0.05"/></body></body></worldbody></mujoco>
"""
REPORT = {
    "address_coordinates_deg": {"HipInputX": 30.0},
    "address_translations_m": {"TranslationInputX": 1.25},
}


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(XML)


def test_slides_stay_metres_and_hinges_become_radians() -> None:
    qpos = address_qpos(_model(), REPORT)
    assert qpos[0] == pytest.approx(1.25)
    assert qpos[1] == pytest.approx(math.radians(30.0))


def test_missing_coordinate_is_a_contract_violation() -> None:
    report = {
        "address_coordinates_deg": {},
        "address_translations_m": {"TranslationInputX": 0.0},
    }
    with pytest.raises(ValueError, match="HipInputX"):
        address_qpos(_model(), report)


def test_unnamed_joints_keep_the_model_default() -> None:
    free = '<body name="d"><freejoint/><geom size="0.01"/></body>'
    xml = XML.replace("</worldbody>", free + "</worldbody>")
    model = mujoco.MjModel.from_xml_string(xml)
    qpos = address_qpos(model, REPORT)
    np.testing.assert_allclose(qpos[2:], model.qpos0[2:])
    assert len(qpos) == 9
