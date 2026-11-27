"""RNEA built on ``jcalc`` vs MuJoCo inverse dynamics at q != 0 (issue #15233).

``jcalc`` feeds rnea/crba/aba in the shared, MuJoCo and Drake stacks, so a wrong
revolute convention only shows up at q != 0.  MuJoCo ``mj_inverse`` is an
independent implementation: it never calls ``jcalc`` or the Featherstone
transforms.  The chains below pair a revolute joint with a prismatic joint,
use offset centres of mass, full (non-diagonal) inertia tensors and non-zero
velocities and accelerations, so a transposed or sign-flipped ``E`` fails.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.rnea import (  # noqa: E402
    rnea,
)
from src.shared.python.core import constants  # noqa: E402
from src.shared.python.spatial_algebra.inertia import mcI  # noqa: E402
from src.shared.python.spatial_algebra.transforms import xlt  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

_LINK = 0.8
_BODY1 = {
    "mass": 2.0,
    "com": (0.3, -0.1, 0.2),
    "inertia": (0.30, 0.25, 0.20, 0.03, -0.02, 0.01),
}
_BODY2 = {
    "mass": 1.5,
    "com": (0.1, 0.2, -0.15),
    "inertia": (0.12, 0.10, 0.08, -0.01, 0.02, 0.015),
}
_AXIS = {"Rx": "1 0 0", "Ry": "0 1 0", "Rz": "0 0 1"}


def _inertial(body: dict) -> str:
    x, y, z = body["com"]
    ixx, iyy, izz, ixy, ixz, iyz = body["inertia"]
    return (
        f'<inertial pos="{x} {y} {z}" mass="{body["mass"]}" '
        f'fullinertia="{ixx} {iyy} {izz} {ixy} {ixz} {iyz}"/>'
    )


def _mujoco_model(revolute: str) -> mujoco.MjModel:
    xml = f"""
    <mujoco>
      <option gravity="0 0 -{float(constants.GRAVITY_M_S2)}"/>
      <worldbody>
        <body name="link1">
          <joint name="rev" type="hinge" axis="{_AXIS[revolute]}"/>
          {_inertial(_BODY1)}
          <body name="link2" pos="{_LINK} 0 0">
            <joint name="slide" type="slide" axis="1 0 0"/>
            {_inertial(_BODY2)}
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _inertia_matrix(body: dict) -> np.ndarray:
    ixx, iyy, izz, ixy, ixz, iyz = body["inertia"]
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def _featherstone_model(revolute: str) -> dict:
    return {
        "NB": 2,
        "parent": np.array([-1, 0]),
        "jtype": [revolute, "Px"],
        "Xtree": [xlt(np.zeros(3)), xlt(np.array([_LINK, 0.0, 0.0]))],
        "I": [
            mcI(_BODY1["mass"], np.array(_BODY1["com"]), _inertia_matrix(_BODY1)),
            mcI(_BODY2["mass"], np.array(_BODY2["com"]), _inertia_matrix(_BODY2)),
        ],
    }


def _mujoco_tau(model, q, qd, qdd) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.qvel[:] = qd
    data.qacc[:] = qdd
    mujoco.mj_inverse(model, data)
    return np.array(data.qfrc_inverse)


# q != 0 on both joints, non-zero qd and qdd; includes the issue's Rz+Px chain.
_STATES = [
    (0.6, 0.25, 0.7, -0.4, 1.1, 0.5),
    (-1.1, -0.15, -0.9, 0.8, -0.6, 1.3),
    (2.3, 0.4, 1.5, 0.2, 0.3, -0.8),
]


@pytest.mark.parametrize("revolute", ["Rx", "Ry", "Rz"])
@pytest.mark.parametrize("state", _STATES)
def test_rnea_matches_mujoco_inverse_dynamics_at_nonzero_q(
    revolute: str, state: tuple[float, ...]
) -> None:
    q, qd, qdd = np.array(state[0:2]), np.array(state[2:4]), np.array(state[4:6])
    # State is (q1, q2, qd1, qd2, qdd1, qdd2).
    tau_ref = _mujoco_tau(_mujoco_model(revolute), q, qd, qdd)
    tau = rnea(_featherstone_model(revolute), q, qd, qdd)
    np.testing.assert_allclose(tau, tau_ref, rtol=1e-6, atol=1e-6)


def test_rnea_reference_is_sensitive_to_the_rotation_convention() -> None:
    """Guard the guard: a transposed E (active rotation) must NOT match MuJoCo."""
    from src.shared.python.spatial_algebra import joints

    q, qd, qdd = np.array([0.6, 0.25]), np.array([0.7, -0.4]), np.array([1.1, 0.5])
    tau_ref = _mujoco_tau(_mujoco_model("Ry"), q, qd, qdd)
    original = joints._featherstone_e_rot
    try:
        joints._featherstone_e_rot = lambda jtype, angle: original(jtype, -angle)
        tau_wrong = rnea(_featherstone_model("Ry"), q, qd, qdd)
    finally:
        joints._featherstone_e_rot = original
    assert not np.allclose(tau_wrong, tau_ref, rtol=1e-6, atol=1e-6)
