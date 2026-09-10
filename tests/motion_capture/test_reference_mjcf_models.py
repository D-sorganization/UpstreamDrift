"""Compiled MuJoCo FK agrees with the shared fitter at multiple joint anchors."""

from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reference.mjcf_models import load_mjcf_model
from src.motion_capture.reconstruct.model import ArticulatedModel

pytestmark = pytest.mark.unit


def test_compiled_hinges_match_native_forward(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    path = tmp_path / "chain.xml"
    path.write_text("""<mujoco><compiler angle="radian"/><worldbody>
    <body name="arm" pos="0 0 1" euler="0.1 0.2 0.3">
    <joint name="a" type="hinge" axis="0 1 0" pos=".1 0 0"/>
    <joint name="b" type="hinge" axis="1 0 0" pos="0 .2 0"/>
    <geom type="sphere" size=".1"/>
    <body name="tip" pos=".5 0 0"><geom type="sphere" size=".1"/></body>
    </body></worldbody></mujoco>""")
    native = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(native)
    converted = load_mjcf_model(
        path, name="chain", landmark_map={"arm": "left_elbow", "tip": "left_wrist"}
    )
    model = ArticulatedModel(converted.spec)
    q = np.zeros((1, model.n_dof))
    q[0, 6:] = [0.4, -0.3]
    data.qpos[:] = [0.4, -0.3]
    mujoco.mj_forward(native, data)
    actual = model.landmarks(q)[0]
    np.testing.assert_allclose(actual, data.xpos[1:], atol=1e-10)
