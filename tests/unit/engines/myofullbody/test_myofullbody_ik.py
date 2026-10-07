"""Unit tests for the orientation IK and anatomical frames (issue #11644)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from src.shared.python.myofullbody import anatomy
from src.shared.python.myofullbody.couplings import JointCoupling
from src.shared.python.myofullbody.ik import (
    body_rotation,
    rotation_error,
    solve_orientation,
)

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.unit

XML = """
<mujoco><worldbody><body name="a"><joint name="x" type="hinge" axis="1 0 0"/>
<joint name="y" type="hinge" axis="0 1 0"/><joint name="z" type="hinge" axis="0 0 1"/>
<geom size=".05"/><body name="b" pos="0 0 -.3"><joint name="e" type="hinge" axis="0 1 0"/>
<geom size=".05"/></body></body></worldbody></mujoco>
"""


@pytest.fixture
def rig():
    model = mujoco.MjModel.from_xml_string(XML)
    return model, mujoco.MjData(model), JointCoupling.from_model(model)


def test_recovers_a_three_joint_orientation(rig) -> None:
    model, data, coupling = rig
    truth = np.array([0.4, -0.6, 0.9, 0.0])
    target = body_rotation(model, data, coupling, truth, 1)
    fit = solve_orientation(
        model,
        data,
        coupling,
        np.zeros(4),
        1,
        (0, 1, 2),
        target,
        None,
        seeds=(np.array([0.3, 0.3, 0.3]),),
    )
    assert fit.error_rad < 1e-8
    np.testing.assert_allclose(
        body_rotation(model, data, coupling, np.r_[fit.values, 0.0], 1),
        target,
        atol=1e-8,
    )


def test_bounds_clamp_and_report_the_residual(rig) -> None:
    model, data, coupling = rig
    target = body_rotation(model, data, coupling, np.array([0.0, 0.0, 1.0, 0.0]), 1)
    lo, hi = np.array([-0.3]), np.array([0.3])
    fit = solve_orientation(
        model, data, coupling, np.zeros(4), 1, (2,), target, (lo, hi)
    )
    assert fit.at_bound.tolist() == [True]
    assert fit.values[0] == pytest.approx(0.3)
    assert fit.error_rad == pytest.approx(0.7, abs=1e-6)


def test_input_vector_is_not_modified(rig) -> None:
    model, data, coupling = rig
    q = np.array([0.1, 0.2, 0.3, 0.0])
    before = q.copy()
    target = body_rotation(model, data, coupling, np.array([0.5, 0, 0, 0]), 1)
    solve_orientation(model, data, coupling, q, 1, (0,), target, None)
    np.testing.assert_array_equal(q, before)


def test_contracts(rig) -> None:
    model, data, coupling = rig
    with pytest.raises(ValueError):
        solve_orientation(model, data, coupling, np.zeros(4), 1, (0,), np.eye(2), None)
    with pytest.raises(ValueError):
        solve_orientation(model, data, coupling, np.zeros(4), 1, (), np.eye(3), None)


def test_rotation_error_is_geodesic() -> None:
    rot = Rotation.from_euler("z", 0.25).as_matrix()
    assert np.linalg.norm(rotation_error(np.eye(3), rot)) == pytest.approx(0.25)


def test_frame_from_axes_is_orthonormal_and_right_handed() -> None:
    frame = anatomy.frame_from_axes(
        np.array([0.0, 0.0, 2.0]), np.array([1.0, 0.0, 0.7])
    )
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0)
    np.testing.assert_allclose(frame[:, 1], [0, 0, 1])
    np.testing.assert_allclose(frame[:, 0], [1, 0, 0])


def test_frame_from_axes_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        anatomy.frame_from_axes(np.zeros(3), np.array([1.0, 0, 0]))
    with pytest.raises(ValueError, match="parallel"):
        anatomy.frame_from_axes(np.array([0.0, 0, 1]), np.array([0.0, 0, 3]))


def test_segment_order_puts_parents_first() -> None:
    order = anatomy.SEGMENTS
    assert order[0] == "pelvis" and order[1] == "thorax"
    for side in anatomy.SIDES:
        assert order.index(f"humerus_{side}") < order.index(f"forearm_{side}")
        assert order.index(f"forearm_{side}") < order.index(f"hand_{side}")
        assert order.index(f"femur_{side}") < order.index(f"tibia_{side}")
    assert set(anatomy.spec_frame_defs()) == set(anatomy.myo_frame_defs()) == set(order)
