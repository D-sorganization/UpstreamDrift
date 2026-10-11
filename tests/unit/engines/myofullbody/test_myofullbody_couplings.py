"""Unit tests for joint-equality substitution (issue #11644)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.myofullbody.couplings import JointCoupling

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.unit

XML = """
<mujoco><worldbody><body name="a"><joint name="j1" type="hinge" axis="0 1 0"/>
<geom size=".05"/><body name="b" pos="0 0 -.2"><joint name="j2" type="hinge" axis="0 1 0"/>
<geom size=".05"/><body name="c" pos="0 0 -.2"><joint name="j3" type="slide" axis="1 0 0"/>
<geom size=".05"/></body></body></body></worldbody>
<equality>
<joint joint1="j2" joint2="j1" polycoef="0.1 2 0.5 0 0"/>
<joint joint1="j3" joint2="j2" polycoef="0 1 0 0 0"/>
</equality></mujoco>
"""


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(XML)


def test_expand_follows_polynomial_and_chains(model) -> None:
    coupling = JointCoupling.from_model(model)
    assert coupling.dependent == (1, 2) and coupling.n_q == 3
    q = coupling.expand(np.array([0.3, 9.0, 9.0]))
    j2 = 0.1 + 2 * 0.3 + 0.5 * 0.3**2
    np.testing.assert_allclose(q, [0.3, j2, j2])
    assert coupling.free_mask().tolist() == [True, False, False]


def test_jacobian_matches_finite_difference(model) -> None:
    coupling = JointCoupling.from_model(model)
    q = coupling.expand(np.array([0.4, 0.0, 0.0]))
    jac = coupling.jacobian(q)
    eps = 1e-6
    plus = coupling.expand(np.array([0.4 + eps, 0.0, 0.0]))
    minus = coupling.expand(np.array([0.4 - eps, 0.0, 0.0]))
    np.testing.assert_allclose(jac[:, 0], (plus - minus) / (2 * eps), atol=1e-8)
    assert np.all(jac[:, 1:] == np.array([[0, 0], [0, 0], [0, 0]]))


def test_matches_mujoco_constraint_residual(model) -> None:
    coupling = JointCoupling.from_model(model)
    data = mujoco.MjData(model)
    data.qpos[:] = coupling.expand(np.array([0.5, 0.0, 0.0]))
    mujoco.mj_forward(model, data)
    assert np.abs(data.efc_pos[: model.neq]).max() < 1e-12


def test_rejects_wrong_size_and_cycles() -> None:
    cyc = XML.replace(
        '<joint joint1="j3" joint2="j2" polycoef="0 1 0 0 0"/>',
        '<joint joint1="j3" joint2="j2" polycoef="0 1 0 0 0"/>'
        '<joint joint1="j1" joint2="j3" polycoef="0 1 0 0 0"/>',
    )
    with pytest.raises(ValueError, match="circular"):
        JointCoupling.from_model(mujoco.MjModel.from_xml_string(cyc))
    coupling = JointCoupling.from_model(mujoco.MjModel.from_xml_string(XML))
    with pytest.raises(ValueError):
        coupling.expand(np.zeros(5))
