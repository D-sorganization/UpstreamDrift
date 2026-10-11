"""Torque-actuated neck tests (issue #11689)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.myofullbody import neck, redundancy

pytestmark = pytest.mark.unit
ORDER = ("SpineInputX", "NeckInputX", "NeckInputY", "NeckInputZ", "hip_flexion_r")


def _basis(nm: int = 3, nc: int = 3) -> redundancy.FrameBasis:
    rng = np.random.default_rng(1)
    return redundancy.FrameBasis(
        active=rng.uniform(10, 20, nm),
        passive=rng.uniform(0, 1, nm),
        moment=rng.normal(0, 0.05, (nm, nc)),
        phi=np.zeros((4, 5)),
    )


def test_capacity_table_is_documented_and_positive() -> None:
    assert set(neck.CAPACITY_NM) == {"NeckInputX", "NeckInputY", "NeckInputZ"}
    assert all(v > 0 for v in neck.CAPACITY_NM.values())
    assert "Vasavada" in neck.SOURCE


def test_capacities_follow_anthro_neck_axes() -> None:
    """Rx(X) Ry(Y) Rz(Z), head forward +x: X lateral, Y flexion (#11729)."""
    assert neck.CAPACITY_NM["NeckInputY"] == pytest.approx(30.0)  # flexion
    assert neck.CAPACITY_NM["NeckInputX"] == pytest.approx(36.0)  # lateral bending
    assert neck.CAPACITY_NM["NeckInputZ"] == pytest.approx(15.0)  # axial rotation


def test_actuator_names_come_in_signed_pairs() -> None:
    cols = [1, 2, 3]
    names = neck.actuator_names(ORDER, cols)
    assert names == [
        "neck_torque_NeckInputX_pos", "neck_torque_NeckInputX_neg",
        "neck_torque_NeckInputY_pos", "neck_torque_NeckInputY_neg",
        "neck_torque_NeckInputZ_pos", "neck_torque_NeckInputZ_neg",
    ]  # fmt: skip


def test_augment_adds_signed_unit_moment_arm_actuators_at_capacity() -> None:
    cols = [0, 1, 2]  # SpineInputX, NeckInputX, NeckInputY
    base = _basis(3, 3)
    out = neck.augment(base, ORDER, cols)
    assert out.active.shape[0] == 3 + 4
    np.testing.assert_allclose(out.active[:3], base.active)
    np.testing.assert_allclose(out.active[3:], [36.0, 36.0, 30.0, 30.0])
    np.testing.assert_allclose(out.passive[3:], 0.0)
    np.testing.assert_allclose(out.moment[3], [0, 1, 0])
    np.testing.assert_allclose(out.moment[4], [0, -1, 0])
    np.testing.assert_allclose(out.moment[5], [0, 0, 1])
    assert out.phi is base.phi


def test_augment_without_neck_columns_is_the_identity() -> None:
    base = _basis(3, 2)
    assert neck.augment(base, ORDER, [0, 4]) is base


def test_demand_inside_capacity_leaves_no_reserve_and_outside_leaves_the_excess() -> (
    None
):
    cols = [1]  # NeckInputX, lateral bending: 36 N m
    base = redundancy.FrameBasis(
        np.zeros(0), np.zeros(0), np.zeros((0, 1)), np.zeros((1, 5))
    )
    out = neck.augment(base, ORDER, cols)
    inside = redundancy.solve_frame(
        out.active, out.passive, out.moment, np.array([20.0])
    )
    assert abs(inside.reserve[0]) < 1e-3
    over = redundancy.solve_frame(
        out.active, out.passive, out.moment, np.array([-45.0])
    )
    assert over.reserve[0] == pytest.approx(-9.0, abs=0.2)


def test_split_separates_muscle_and_neck_activation() -> None:
    act = np.arange(7.0)
    muscles, torque = neck.split(act, 3)
    assert list(muscles) == [0, 1, 2] and list(torque) == [3, 4, 5, 6]
    with pytest.raises(ValueError):
        neck.split(act, 9)


def test_torque_returns_the_signed_neck_generalised_force() -> None:
    order = ("NeckInputX", "Other")
    base = redundancy.FrameBasis(
        np.array([1.0]), np.array([0.0]), np.array([[0.5, 0.0]]), np.eye(2)
    )
    aug = neck.augment(base, order, [0, 1])
    act = np.array([0.2, 0.5, 0.1])  # muscle, neck pos, neck neg
    tau = neck.torque(aug, act, 1)
    expected = (0.5 - 0.1) * neck.CAPACITY_NM["NeckInputX"]
    assert tau.tolist() == [expected, 0.0]


def test_torque_of_an_unaugmented_basis_is_zero() -> None:
    base = redundancy.FrameBasis(
        np.array([1.0]), np.array([0.0]), np.array([[0.5]]), np.eye(1)
    )
    assert neck.torque(base, np.array([0.3]), 1).tolist() == [0.0]
