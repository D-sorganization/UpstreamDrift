"""Address pose of an address-only pipeline report as MuJoCo qpos (SHOW-C7, #14563)."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.engines.physics_engines.mujoco.python.address_pose import address_qpos

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]

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
    with pytest.raises(ContractViolationError, match="HipInputX"):
        address_qpos(_model(), report)


def test_unnamed_joints_keep_the_model_default() -> None:
    free = '<body name="d"><freejoint/><geom size="0.01"/></body>'
    xml = XML.replace("</worldbody>", free + "</worldbody>")
    model = mujoco.MjModel.from_xml_string(xml)
    qpos = address_qpos(model, REPORT)
    np.testing.assert_allclose(qpos[2:], model.qpos0[2:])
    assert len(qpos) == 9


def test_render_figure_writes_a_png_for_a_hand_built_pose(tmp_path) -> None:
    """Smoke test of the comparison figure on a three-point skeleton."""
    import importlib.util
    import sys

    path = (
        ROOT
        / "docs/development/full_body_models/evidence/showpiece_render/c7"
        / "render_address_stills.py"
    )
    spec = importlib.util.spec_from_file_location("render_address_stills", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    pose = module.Pose(
        points=np.array([[0, 0, 0], [0, 0, 1.0], [0, 0.2, 1.0], [0, 0.3, 0.8]]),
        links=np.array([(1, 2), (2, 3)]),
        club=np.array([[0.0, 0.3, 0.8], [0.9, 0.3, 0.0]]),
        hands={"lead": (np.array([0, 0.3, 0.8]), np.eye(3))},
    )
    out = module.render_figure({"only": pose}, tmp_path / "out.png")
    assert out.stat().st_size > 1000
