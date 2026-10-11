"""Grip-modelling tab reuses the shared weld efc helper (GCV-8, #11714).

The tab reports the hand-on-club weld wrench through ``grip_efc`` instead of
computing constraint forces itself.  A scene without the grip welds (the
contact-only hand models) is reported as unavailable with a reason, never as
zero.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.grip_efc import grip_analysis_from_efc
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (
    grip_modelling_tab as tab_module,
)

pytestmark = pytest.mark.unit

MASS = 0.5
G = 9.81
_WELDED_XML = f"""
<mujoco><option gravity="0 0 -{G}"/>
<worldbody>
 <body name="hand_r" mocap="true" pos="0.1 0 1"><site name="hr" size="0.01"/></body>
 <body name="hand_l" mocap="true" pos="-0.1 0 1"><site name="hl" size="0.01"/></body>
 <body name="club" pos="0 0 1"><freejoint/>
  <geom type="box" size="0.2 0.02 0.02" mass="{MASS}"/>
  <site name="cr" pos="0.1 0 0" size="0.01"/>
  <site name="cl" pos="-0.1 0 0" size="0.01"/></body>
</worldbody>
<equality>
 <weld name="grip_weld_r" site1="hr" site2="cr"/>
 <weld name="grip_weld_l" site1="hl" site2="cl"/>
</equality>
</mujoco>"""
_NO_WELD_XML = """
<mujoco><worldbody>
 <body name="hand_palm" pos="0 0 1"><freejoint/>
  <geom type="sphere" size="0.05"/></body>
</worldbody></mujoco>"""


def _settled(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(3000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def test_summary_reports_the_shared_helper_net_force() -> None:
    model, data = _settled(_WELDED_XML)
    summary = tab_module.grip_weld_wrench_summary(model, data)
    net = grip_analysis_from_efc(model, data).net_force_n
    assert net is not None
    assert summary.available
    assert summary.net_force_n == pytest.approx(float(np.linalg.norm(net)))
    assert summary.net_force_n == pytest.approx(MASS * G, abs=1e-3)
    assert "efc_force" in summary.text


def test_summary_without_grip_welds_is_unavailable_not_zero() -> None:
    model, data = _settled(_NO_WELD_XML)
    summary = tab_module.grip_weld_wrench_summary(model, data)
    assert not summary.available
    assert summary.net_force_n is None
    assert summary.text.startswith("unavailable")
    assert "grip_weld" in summary.text  # states which equality is missing
    assert " 0.0 N" not in summary.text


def test_summary_requires_model_and_data() -> None:
    with pytest.raises(ValueError, match="model"):
        tab_module.grip_weld_wrench_summary(None, None)  # type: ignore[arg-type]
