"""Pinocchio ground-reaction breakdown on the force frame (GCV-2, #11708).

Pinocchio has no contact model of its own: the ground contact is the shared
Hunt-Crossley law's ``ContactSample`` records passed in by the caller.  Without
samples the ground reaction is reported unavailable with a reason, never zero.
"""

from __future__ import annotations

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
if type(pin).__module__ == "unittest.mock" or not all(
    hasattr(pin, n) for n in ("rnea", "buildModelFromXML", "centerOfMass", "Model")
):  # tests/unit/conftest.py mocks pinocchio when it is not installed
    pytest.skip(
        "real pinocchio runtime required (found mock/stub)", allow_module_level=True
    )

from src.engines.physics_engines.pinocchio.python.pinocchio_force_torque import (  # noqa: E402
    PinocchioForceTorqueSource,
)
from src.shared.python.motion_matching.contact_law import (  # noqa: E402
    ContactParameters,
    ContactSample,
    GroundPlane,
    sphere_ground_contact,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

G = 9.80665
PELVIS_KG = 70.0
FOOT_KG = 1.0
FOOT_Y = 0.15
WEIGHT_N = (PELVIS_KG + 2 * FOOT_KG) * G
WEIGHT_RTOL = 0.02
_I = '<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>'
URDF = f"""<robot name="stance">
<link name="pelvis"><inertial><mass value="{PELVIS_KG}"/>{_I}</inertial></link>
<link name="calcn_l"><inertial><mass value="{FOOT_KG}"/>{_I}</inertial></link>
<link name="calcn_r"><inertial><mass value="{FOOT_KG}"/>{_I}</inertial></link>
<joint name="wl" type="fixed"><parent link="pelvis"/><child link="calcn_l"/>
<origin xyz="0 {FOOT_Y} -0.9"/></joint>
<joint name="wr" type="fixed"><parent link="pelvis"/><child link="calcn_r"/>
<origin xyz="0 -{FOOT_Y} -0.9"/></joint>
</robot>"""


def _stance_samples() -> dict[str, ContactSample]:
    """Equilibrium samples of the shared law: penetration carries half the weight."""
    params = ContactParameters(
        stiffness_n_m=5.0e4,
        dissipation_s_m=1.0,
        static_friction=0.9,
        dynamic_friction=0.8,
        viscous_friction=0.0,
        transition_velocity_m_s=0.05,
    )
    radius = 0.03
    depth = (WEIGHT_N / 2) / params.stiffness_n_m
    plane = GroundPlane((0.0, 0.0, 1.0), 0.0)
    out = {}
    for name, y in (("calcn_l", FOOT_Y), ("calcn_r", -FOOT_Y)):
        centre = np.array([0.0, y, radius - depth])
        out[name] = sphere_ground_contact(centre, np.zeros(3), radius, plane, params)
    return out


@pytest.fixture(scope="module")
def source_and_q():
    model = pin.buildModelFromXML(URDF, pin.JointModelFreeFlyer())
    q = pin.neutral(model)
    q[2] = 0.9
    return PinocchioForceTorqueSource(model), q


def _frame(source, q, samples):
    z = np.zeros(source.model.nv)
    return source.sample(q, z, z, z, samples)


def test_static_stance_net_grf_equals_body_weight(source_and_q) -> None:
    source, q = source_and_q
    frame = _frame(source, q, _stance_samples())
    w = {x.label: x for x in frame.wrenches}
    net = np.asarray(w["contact:grf_net"].force_n)
    assert abs(net[2] - WEIGHT_N) / WEIGHT_N < WEIGHT_RTOL
    left, right = (np.asarray(w[f"contact:grf_{s}"].force_n) for s in ("left", "right"))
    np.testing.assert_allclose(left + right, net, atol=1e-9)
    assert w["contact:grf_left"].point_m[1] > 0.0 > w["contact:grf_right"].point_m[1]
    for label in ("contact:free_moment_net", "contact:moment_com_net"):
        assert label in w


def test_com_moment_uses_pinocchio_centre_of_mass(source_and_q) -> None:
    source, q = source_and_q
    frame = _frame(source, q, _stance_samples())
    w = {x.label: x for x in frame.wrenches}
    com = pin.centerOfMass(source.model, source.model.createData(), q)
    np.testing.assert_allclose(w["contact:moment_com_net"].point_m, com, atol=1e-9)
    m_com = np.asarray(w["contact:moment_com_net"].torque_nm)
    assert np.linalg.norm(m_com[:2]) < 0.01 * WEIGHT_N * 0.03


def test_without_samples_ground_reaction_is_unavailable_with_a_reason(
    source_and_q,
) -> None:
    source, q = source_and_q
    frame = _frame(source, q, None)
    assert not any(w.label.startswith("contact:grf") for w in frame.wrenches)
    reason = frame.metadata["ground_reaction_unavailable"]
    assert "no contact model" in reason


def test_unloaded_samples_give_no_arrows(source_and_q) -> None:
    source, q = source_and_q
    zero = np.zeros(3)
    lifted = {
        n: ContactSample(0.0, 0.0, np.array([0.0, 0.0, 0.1]), zero, zero)
        for n in ("calcn_l", "calcn_r")
    }
    frame = _frame(source, q, lifted)
    assert not any(w.label.startswith("contact:grf") for w in frame.wrenches)
