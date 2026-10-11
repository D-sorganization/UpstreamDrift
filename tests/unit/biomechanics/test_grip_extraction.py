"""Pure-math tests for grip extraction helpers (GCV-8, #11714)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.grip_extraction import (
    hand_from_arrays,
    holding_hand_wrench,
    net_only_analysis,
    unavailable_analysis,
)
from src.shared.python.biomechanics.grip_wrench import (
    analyze_grip,
    to_overlay_wrenches,
)

pytestmark = pytest.mark.unit

G = (0.0, 0.0, -9.81)
M = 1.5
COM = np.array([0.0, 0.0, 1.0])
I_W = np.diag([0.01, 0.2, 0.2])
RR = np.array([0.1, 0.0, 1.0])
RL = np.array([-0.1, 0.0, 1.0])


def _kw(**over):
    base = {
        "mass_kg": M,
        "gravity_m_s2": G,
        "com_m": COM,
        "com_acceleration_m_s2": (0, 0, 0),
        "inertia_world_kg_m2": I_W,
        "angular_velocity_rad_s": (0, 0, 0),
        "angular_acceleration_rad_s2": (0, 0, 0),
        "closing_force_n": (0, 0, 0),
        "closing_torque_nm": (0, 0, 0),
        "closing_point_m": RR,
        "holding_point_m": RL,
    }
    base.update(over)
    return base


def test_static_hold_carries_weight_when_closing_hand_is_slack() -> None:
    f, t = holding_hand_wrench(**_kw(holding_point_m=COM))
    np.testing.assert_allclose(f, (0, 0, M * 9.81), atol=1e-12)
    np.testing.assert_allclose(t, 0.0, atol=1e-12)


def test_static_balance_with_both_hands_loaded() -> None:
    f_close = np.array([1.0, -2.0, 3.0])
    f, t = holding_hand_wrench(**_kw(closing_force_n=f_close))
    total = np.array(f) + f_close + M * np.array(G)
    np.testing.assert_allclose(total, 0.0, atol=1e-12)
    moment = np.array(t) + np.cross(RR - COM, f_close) + np.cross(RL - COM, np.array(f))
    np.testing.assert_allclose(moment, 0.0, atol=1e-12)


def test_accelerating_club_follows_newton_euler() -> None:
    a = np.array([2.0, 0.0, 1.0])
    alpha = np.array([0.0, 5.0, 0.0])
    omega = np.array([0.0, 0.0, 3.0])
    f, t = holding_hand_wrench(
        **_kw(
            com_acceleration_m_s2=a,
            angular_acceleration_rad_s2=alpha,
            angular_velocity_rad_s=omega,
        )
    )
    np.testing.assert_allclose(f, M * (a - np.array(G)), atol=1e-12)
    rate = I_W @ alpha + np.cross(omega, I_W @ omega)
    moment = np.array(t) + np.cross(RL - COM, np.array(f))
    np.testing.assert_allclose(moment, rate, atol=1e-12)


@pytest.mark.parametrize(
    "bad", [{"mass_kg": 0.0}, {"mass_kg": float("nan")}, {"com_m": (0, 0)}]
)
def test_holding_wrench_rejects_bad_input(bad) -> None:
    with pytest.raises(ValueError):
        holding_hand_wrench(**_kw(**bad))


def test_net_only_analysis_has_no_per_hand_values_and_emits_net_frames() -> None:
    g = net_only_analysis(
        point_m=(0, 0, 1),
        force_on_club_n=(0, 0, 14.7),
        torque_on_club_nm=(0, 0, 0),
        split_method="allocation",
        reason="allocation yields one net wrench",
    )
    assert g.left is None and g.right is None
    assert g.split_method == "allocation"
    labels = {w.label for w in to_overlay_wrenches(g, source="t")}
    assert labels == {"grip:net_midpoint", "grip:couple_midpoint"}


def test_unavailable_analysis_is_none_not_zero() -> None:
    g = unavailable_analysis("placeholder grip model")
    assert g.net_force_n is None and g.split_method == "unavailable"
    assert to_overlay_wrenches(g, source="t") == []
    with pytest.raises(ValueError):
        unavailable_analysis("")


def test_hand_from_arrays_round_trips_through_analysis() -> None:
    left = hand_from_arrays("L", RL, (0, 0, 7), (0, 0, 0))
    right = hand_from_arrays("R", RR, (0, 0, 7.7), None)
    g = analyze_grip(left, right, split_method="efc_force")
    assert g.net_force_n == pytest.approx((0, 0, 14.7))
