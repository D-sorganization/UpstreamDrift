"""Bracketed contact root kernel: convergence and fail-closed checks."""

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_root import (
    ContactSample,
    locate_contact_root,
)


def sample(t: float, *, inside: bool = True, approach: float = 2.0) -> ContactSample:
    return ContactSample(
        np.array([t]), np.array([2.0]), 0.7 - 2.0 * t, inside, approach
    )


@pytest.mark.parametrize("start,end", [(0.0, 0.5), (0.3, 0.4), (0.34, 0.36)])
def test_bracketed_root_converges_to_actual_state(start: float, end: float) -> None:
    root = locate_contact_root(
        sample, start, end, time_tolerance_s=1e-12, distance_tolerance_m=1e-10
    )
    assert root is not None
    assert root.time_s == pytest.approx(0.35, abs=1e-11)
    np.testing.assert_allclose(root.sample.q, [0.35], atol=1e-11)
    np.testing.assert_allclose(root.sample.qd, [2.0])


@pytest.mark.parametrize("inside,approach", [(False, 2.0), (True, 0.0), (True, -1.0)])
def test_rejects_off_face_and_nonapproaching_roots(
    inside: bool, approach: float
) -> None:
    assert (
        locate_contact_root(
            lambda t: sample(t, inside=inside, approach=approach), 0.0, 0.5
        )
        is None
    )


def test_no_bracket_is_unresolved() -> None:
    assert locate_contact_root(sample, 0.0, 0.2) is None


def test_discontinuous_gap_cannot_masquerade_as_converged_contact() -> None:
    def jump(t: float) -> ContactSample:
        return ContactSample(
            np.array([t]), np.array([2.0]), 1.0 if t < 0.35 else -1.0, True, 2.0
        )

    assert locate_contact_root(jump, 0.0, 0.5) is None


def test_nonfinite_state_is_rejected() -> None:
    def broken(t: float) -> ContactSample:
        return ContactSample(
            np.array([np.nan]), np.array([2.0]), 0.7 - 2.0 * t, True, 2.0
        )

    with pytest.raises(ValueError, match="finite"):
        locate_contact_root(broken, 0.0, 0.5)


def test_mujoco_finite_face_bracket_refines_integrated_state() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body><joint type="slide" axis="1 0 0"/>
        <geom type="box" size="0.01 0.05 0.05" mass="1"/>
      </body></worldbody></mujoco>
    """)
    ball = np.array([0.7, 0.0, 0.0])
    radius = 0.021

    def evaluate(t: float) -> ContactSample:
        data = mujoco.MjData(model)
        data.qvel[0] = 2.0
        if t > 0.0:
            model.opt.timestep = t
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        centre = data.geom_xpos[0]
        face_x = centre[0] + 0.01
        inside = abs(ball[1] - centre[1]) < 0.05 and abs(ball[2] - centre[2]) < 0.05
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        point = np.array([face_x, ball[1], ball[2]])
        mujoco.mj_jac(model, data, jacp, jacr, point, 1)
        approach = float((jacp @ data.qvel)[0])
        return ContactSample(
            data.qpos.copy(),
            data.qvel.copy(),
            ball[0] - face_x - radius,
            bool(inside),
            approach,
        )

    root = locate_contact_root(evaluate, 0.3, 0.4, time_tolerance_s=1e-12)
    assert root is not None
    assert root.time_s == pytest.approx((0.7 - 0.01 - radius) / 2.0, abs=1e-11)
    assert root.sample.q[0] == pytest.approx(2.0 * root.time_s)
