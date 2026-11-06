"""Shaft state for the optimiser loop (SHOW-C4, #14560).

Uses the R6 grip-clamped chain shaft (real MuJoCo model, no mocks).  The
tip deflection is the displacement of the head caused by the bent shaft:
``p(q) - p(q with the chain coordinates zeroed)``, the same definition as the
R11 evidence (``flex_shaft/r11/chain_deflection.py``), here from MuJoCo.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.impact_parameters import TargetFrame
from src.shared.python.impact_parameters.shaft_state import (
    ShaftImpactState,
    decompose_deflection,
    face_loft_deltas,
    shaft_state_series,
)
from src.shared.python.motion_matching.club_models import DRIVER
from src.shared.python.motion_matching.club_shaft_chain import (
    build_shaft_chain,
    clamped_shaft_document,
)

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.unit

FRAME = TargetFrame()  # Z-up, target along -Y
N_LINKS = 8


@pytest.fixture(scope="module")
def clamped():
    from src.engines.physics_engines.mujoco.python.shaft_chain_mjcf import (
        TIP_SITE,
        export_clamped_shaft_mjcf,
    )

    chain = build_shaft_chain(DRIVER, N_LINKS)
    xml = export_clamped_shaft_mjcf(clamped_shaft_document(chain))
    model = mujoco.MjModel.from_xml_string(xml)
    dofs = np.array([model.jnt_dofadr[j] for j in range(model.njnt)])
    return model, dofs, TIP_SITE


def _times(n: int) -> np.ndarray:
    return np.arange(n) * 0.001


def test_straight_shaft_has_zero_deflection_and_energy(clamped):
    model, dofs, tip = clamped
    q = np.zeros((5, model.nq))
    state = shaft_state_series(model, _times(5), q, np.zeros_like(q), dofs, tip, FRAME)
    assert isinstance(state, ShaftImpactState)
    np.testing.assert_allclose(state.tip_deflection_m, 0.0, atol=1e-12)
    np.testing.assert_allclose(state.strain_energy_j, 0.0, atol=1e-12)


def test_single_joint_bend_matches_chord_geometry(clamped):
    """A bend theta at one joint moves the tip by 2 r sin(theta/2)."""
    model, dofs, tip = clamped
    j = 3
    theta = 0.02
    q = np.zeros((2, model.nq))
    q[:, dofs[j]] = theta
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = float(np.linalg.norm(data.site(tip).xpos - data.xanchor[j]))
    state = shaft_state_series(model, _times(2), q, np.zeros_like(q), dofs, tip, FRAME)
    got = float(np.linalg.norm(state.tip_deflection_m[1]))
    assert got == pytest.approx(2 * r * math.sin(theta / 2), rel=2e-2)
    assert state.strain_energy_j[1] > 0.0


def test_strain_energy_is_half_k_q_squared(clamped):
    model, dofs, tip = clamped
    q = np.zeros((1, model.nq))
    q[0, dofs[2]] = 0.03
    state = shaft_state_series(model, _times(1), q, np.zeros_like(q), dofs, tip, FRAME)
    expected = 0.5 * model.jnt_stiffness[2] * 0.03**2
    assert state.strain_energy_j[0] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    ("deflection", "velocity", "lead", "droop"),
    [
        # forward along travel (-Y is the target line) = positive lead/kick
        ((0.0, -0.01, 0.0), (0.0, -40.0, 0.0), 0.01, 0.0),
        ((0.0, 0.01, 0.0), (0.0, -40.0, 0.0), -0.01, 0.0),  # lag
        ((0.0, 0.0, -0.02), (0.0, -40.0, 0.0), 0.0, 0.02),  # down = droop
        ((0.0, 0.0, 0.02), (0.0, -40.0, 0.0), 0.0, -0.02),  # up = negative droop
    ],
)
def test_lead_and_droop_sign_conventions(deflection, velocity, lead, droop):
    got_lead, got_droop = decompose_deflection(
        np.array(deflection), np.array(velocity), FRAME
    )
    assert got_lead == pytest.approx(lead, abs=1e-12)
    assert got_droop == pytest.approx(droop, abs=1e-12)


def test_state_exposes_rate_and_impact_report(clamped):
    """Shaft state must carry what C3/C5 need to penalise resonance exploits."""
    model, dofs, tip = clamped
    n = 40
    t = _times(n)
    q = np.zeros((n, model.nq))
    q[:, dofs[1]] = 0.02 * np.sin(2 * math.pi * 5.0 * t)  # 5 Hz bend
    v = np.gradient(q, t, axis=0)
    state = shaft_state_series(model, t, q, v, dofs, tip, FRAME)
    assert state.tip_deflection_rate_mps.shape == (n,)
    assert state.peak_deflection_m == pytest.approx(
        float(np.max(np.linalg.norm(state.tip_deflection_m, axis=1)))
    )
    report = state.at_impact(n - 1)
    assert set(report) >= {"lead_m", "droop_m", "tip_deflection_m", "strain_energy_j"}


def test_rejects_mismatched_shapes(clamped):
    model, dofs, tip = clamped
    q = np.zeros((3, model.nq))
    with pytest.raises(ValueError, match="same shape"):
        shaft_state_series(
            model, _times(3), q, np.zeros((2, model.nq)), dofs, tip, FRAME
        )


@pytest.mark.parametrize(
    ("bent", "face", "loft"),
    [
        (
            (-0.0872, -0.9962, 0.0),
            5.0,
            0.0,
        ),  # normal toward -X (RH golfer's right): open
        ((0.0872, -0.9962, 0.0), -5.0, 0.0),  # closed
        ((0.0, -0.9962, 0.0872), 0.0, 5.0),  # tilted up: more loft
    ],
)
def test_face_loft_deltas_sign_conventions(bent, face, loft):
    got = face_loft_deltas(np.array(bent), np.array([0.0, -1.0, 0.0]), FRAME)
    assert got == pytest.approx((face, loft), abs=0.01)


def test_bend_rotates_the_face_by_the_joint_angle(clamped):
    """A single bend rotates the tip by that angle: |(face, loft)| = theta."""
    model, dofs, tip = clamped
    j, theta = 2, 0.02  # joint 2 bends about the world X axis
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    axis = np.asarray(data.xaxis[j])
    n_world = np.cross(axis, FRAME.z_t)  # horizontal, perpendicular to the axis
    n_site = data.site(tip).xmat.reshape(3, 3).T @ (n_world / np.linalg.norm(n_world))
    q = np.zeros((2, model.nq))
    q[:, dofs[j]] = theta
    state = shaft_state_series(
        model, _times(2), q, np.zeros_like(q), dofs, tip, FRAME, face_normal_site=n_site
    )
    assert state.face_delta_deg is not None and state.loft_delta_deg is not None
    got = math.hypot(state.face_delta_deg[1], state.loft_delta_deg[1])
    assert got == pytest.approx(math.degrees(theta), rel=0.02)
    assert (
        shaft_state_series(
            model, _times(2), q, np.zeros_like(q), dofs, tip, FRAME
        ).face_delta_deg
        is None
    )


# --- toe-down component (#14600) -------------------------------------------
# Shaft 50 deg off vertical, tip moving along +x: the (shaft, velocity) plane's
# normal is oriented toward gravity, and toe-down is the deflection along it.
_SHAFT = np.array([0.0, math.sin(math.radians(50)), -math.cos(math.radians(50))])
_VELOCITY = np.array([30.0, 0.0, 0.0])
_DOWN_NORMAL = -np.cross(_VELOCITY, _SHAFT) / np.linalg.norm(
    np.cross(_VELOCITY, _SHAFT)
)


def test_toe_down_is_positive_for_deflection_along_the_downward_plane_normal():
    from src.shared.python.impact_parameters.shaft_state import toe_down_component

    value = toe_down_component(0.01 * _DOWN_NORMAL, _VELOCITY, _SHAFT, FRAME)
    assert value == pytest.approx(0.01)
    assert toe_down_component(-0.01 * _DOWN_NORMAL, _VELOCITY, _SHAFT, FRAME) == (
        pytest.approx(-0.01)
    )


@pytest.mark.parametrize("direction", [_VELOCITY / 30.0, _SHAFT])
def test_toe_down_ignores_lead_lag_and_shaft_axis_deflection(direction):
    from src.shared.python.impact_parameters.shaft_state import toe_down_component

    assert toe_down_component(0.05 * direction, _VELOCITY, _SHAFT, FRAME) == (
        pytest.approx(0.0, abs=1e-12)
    )


def test_toe_down_is_zero_when_velocity_is_parallel_to_the_shaft():
    from src.shared.python.impact_parameters.shaft_state import toe_down_component

    assert toe_down_component(_SHAFT * 0.01, 5.0 * _SHAFT, _SHAFT, FRAME) == 0.0


# --- droop decomposition (#14600 review F2) --------------------------------
def test_droop_decomposition_reconstructs_droop_from_lead_and_toe_down_terms():
    from src.shared.python.impact_parameters.shaft_state import droop_decomposition

    deflection = np.array([0.04, -0.02, 0.01])
    velocity = np.array([25.0, 10.0, -5.0])
    shaft = _SHAFT
    out = droop_decomposition(deflection, velocity, shaft, FRAME)
    _, droop = decompose_deflection(deflection, velocity, FRAME)
    assert out["droop_m"] == pytest.approx(droop)
    assert out["residual_m"] == pytest.approx(0.0, abs=1e-12)
    terms = out["term_lead_m"] + out["term_in_plane_perp_m"] + out["term_normal_m"]
    assert terms == pytest.approx(droop)
    # d1 along the tip velocity is exactly lead_m
    lead, _ = decompose_deflection(deflection, velocity, FRAME)
    assert out["lead_m"] == pytest.approx(lead)
    assert out["v_dot_z"] == pytest.approx((velocity / np.linalg.norm(velocity))[2])
    assert out["s_dot_z"] == pytest.approx(shaft[2])


def test_pure_lead_deflection_enters_droop_only_through_v_dot_z():
    from src.shared.python.impact_parameters.shaft_state import droop_decomposition

    velocity = np.array([30.0, 0.0, -10.0])
    unit_v = velocity / np.linalg.norm(velocity)
    out = droop_decomposition(0.02 * unit_v, velocity, _SHAFT, FRAME)
    assert out["droop_m"] == pytest.approx(-0.02 * unit_v[2])
    assert out["term_in_plane_perp_m"] == pytest.approx(0.0, abs=1e-12)
    assert out["term_normal_m"] == pytest.approx(0.0, abs=1e-12)


# Chain angles (x0,z0,x1,z1,x2,z2,x3,z3) read from the C4 driver replay at its
# impact index (#14600): the real bend that the foreshortening check is about.
_C4_IMPACT_ANGLES = (
    0.04567363, -0.09707265, 0.01195988, -0.01465296,
    0.01253004, -0.01832279, 0.00671477, -0.00257079,
)  # fmt: skip


def _c4_chain_model(site_quat: str):
    from src.engines.physics_engines.mujoco.python.shaft_chain_mjcf import (
        TIP_SITE,
        export_clamped_shaft_mjcf,
    )

    xml = export_clamped_shaft_mjcf(
        clamped_shaft_document(build_shaft_chain(DRIVER, 4))
    )
    old = 'name="shaft_tip" size=".001" pos="0 0.111375 0" quat="1 0 0 0"'
    assert old in xml
    xml = xml.replace(old, old.replace('quat="1 0 0 0"', f'quat="{site_quat}"'))
    model = mujoco.MjModel.from_xml_string(xml)
    return model, np.array([model.jnt_dofadr[j] for j in range(model.njnt)]), TIP_SITE


@pytest.mark.parametrize("site_quat", ["1 0 0 0", "0.7071068 0.7071068 0 0"])
def test_shaft_axis_is_the_chain_not_the_tip_site_frame(site_quat):
    """The Clubhead site of the full-body model is a face frame whose y is not
    the shaft; the axial foreshortening of the real C4 bend is ~-7 mm either way
    (the replay used to report -31 mm from the site's y axis)."""
    model, dofs, tip = _c4_chain_model(site_quat)
    q = np.array([_C4_IMPACT_ANGLES])
    state = shaft_state_series(model, _times(1), q, np.zeros_like(q), dofs, tip, FRAME)
    np.testing.assert_allclose(state.shaft_axis[0], [0.0, 1.0, 0.0], atol=1e-9)
    axial = float(state.tip_deflection_m[0] @ state.shaft_axis[0])
    assert axial == pytest.approx(-6.77e-3, abs=0.1e-3)
