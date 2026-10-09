"""Analytic tests for the shared grip wrench core (GCV-7, #11713).

Sign convention: wrench exerted by the hand ON THE CLUB, world frame, SI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.biomechanics.grip_wrench import (
    SPLIT_METHODS,
    GripAnalysis,
    GripSeries,
    HandWrench,
    about_axis,
    analyze_grip,
    to_contact_reaction_wrench,
    to_overlay_wrenches,
)
from src.shared.python.force_overlay.contracts import WrenchKind
from src.shared.python.force_overlay.conversions import move_wrench_point

pytestmark = pytest.mark.unit

RL = (0.0, 0.0, 1.0)
RR = (0.0, 0.0, 0.8)


def _hand(side, point, force, torque=(0.0, 0.0, 0.0)):
    return HandWrench(
        side=side, point_m=point, force_on_club_n=force, torque_on_club_nm=torque
    )


def _analyze(left, right, **kw):
    kw.setdefault("split_method", "logged")
    return analyze_grip(left, right, **kw)


def test_pure_couple_is_reference_point_independent():
    d = (0.0, 0.0, 0.2)  # L - R offset along the grip axis
    f = (10.0, 0.0, 0.0)
    g = _analyze(_hand("L", RL, f), _hand("R", RR, tuple(-x for x in f)))
    np.testing.assert_allclose(g.net_force_n, 0.0, atol=1e-12)
    # M_M = (rL-rM) x F_L + (rR-rM) x F_R = d x F  (half-d each side)
    np.testing.assert_allclose(g.couple_at_midpoint_nm, np.cross(d, f), atol=1e-12)
    moved = g.net_wrench_at((3.0, -2.0, 5.0))
    np.testing.assert_allclose(moved.torque_nm, np.cross(d, f), atol=1e-12)


def test_pure_squeeze_has_no_net_force_or_couple():
    f = (0.0, 0.0, 25.0)  # along the line joining the hands (z)
    g = _analyze(_hand("L", RL, (0, 0, -25.0)), _hand("R", RR, f))
    np.testing.assert_allclose(g.net_force_n, 0.0, atol=1e-12)
    np.testing.assert_allclose(g.couple_at_midpoint_nm, 0.0, atol=1e-12)


def test_symmetric_same_direction_forces_have_zero_couple():
    f = (5.0, 3.0, 1.0)
    g = _analyze(_hand("L", RL, f), _hand("R", RR, f))
    np.testing.assert_allclose(g.net_force_n, (10.0, 6.0, 2.0))
    np.testing.assert_allclose(g.couple_at_midpoint_nm, 0.0, atol=1e-12)


def test_asymmetric_same_direction_forces_known_couple():
    # L=30 N, R=10 N along x; arms +0.1 / -0.1 m along z
    g = _analyze(_hand("L", RL, (30.0, 0, 0)), _hand("R", RR, (10.0, 0, 0)))
    # (0,0,.1)x(30,0,0) = (0, 3, 0); (0,0,-.1)x(10,0,0) = (0,-1,0)
    np.testing.assert_allclose(g.couple_at_midpoint_nm, (0.0, 2.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(g.mof_left_nm, (0.0, 3.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(g.mof_right_nm, (0.0, -1.0, 0.0), atol=1e-12)


def test_split_contact_moment_and_free_torque_sum_to_couple():
    g = _analyze(
        _hand("L", RL, (30.0, 0, 0), (0.0, 0.0, 1.5)),
        _hand("R", RR, (10.0, 0, 0), (0.5, 0.0, 2.0)),
    )
    np.testing.assert_allclose(g.contact_force_moment_nm, (0.0, 2.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(g.applied_free_torque_nm, (0.5, 0.0, 3.5))
    np.testing.assert_allclose(
        g.couple_at_midpoint_nm,
        np.add(g.contact_force_moment_nm, g.applied_free_torque_nm),
    )
    np.testing.assert_allclose(
        g.couple_at_midpoint_nm,
        np.add(np.add(g.mof_left_nm, g.mof_right_nm), g.applied_free_torque_nm),
    )


def test_midpoint_is_mean_of_grip_points():
    g = _analyze(_hand("L", (0, 0, 1.0), (1, 0, 0)), _hand("R", (2, 0, 0.0), (0, 1, 0)))
    assert g.midpoint_m == (1.0, 0.0, 0.5)


def test_transport_invariance_equals_direct_sum_about_point():
    lw = _hand("L", (0.1, 0.2, 1.0), (4.0, -2.0, 7.0), (0.3, 0.1, -0.2))
    rw = _hand("R", (-0.05, 0.1, 0.8), (-1.0, 5.0, 2.0), (0.0, 0.4, 0.9))
    g = _analyze(lw, rw)
    p = np.array([0.7, -1.3, 0.4])
    direct = sum(
        np.cross(np.array(h.point_m) - p, h.force_on_club_n)
        + np.array(h.torque_on_club_nm)
        for h in (lw, rw)
    )
    moved = g.net_wrench_at(p)
    np.testing.assert_allclose(moved.torque_nm, direct, atol=1e-12)
    np.testing.assert_allclose(moved.force_n, g.net_force_n)


def test_static_club_balances_weight_and_moment_about_com():
    m, gvec = 0.45, np.array([0.0, 0.0, -9.81])
    com = np.array([0.0, 0.0, 0.2])
    weight = m * gvec
    # Both hands share the support; free torques chosen to cancel moments
    fl = fr = -weight / 2.0
    g = _analyze(_hand("L", RL, tuple(fl)), _hand("R", RR, tuple(fr)))
    np.testing.assert_allclose(g.net_force_n, -weight, atol=1e-12)
    about_com = g.net_wrench_at(com).torque_nm
    # gravity moment about CoM is zero, so hand moment about CoM must be zero
    # when support is vertical through the CoM axis
    np.testing.assert_allclose(about_com, 0.0, atol=1e-12)


def test_club_local_components_use_transpose_rotation():
    th = math.pi / 2
    rot = np.array(
        [[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]]
    )
    g = _analyze(
        _hand("L", RL, (30.0, 0, 0)),
        _hand("R", RR, (10.0, 0, 0)),
        club_rotation=rot,
    )
    np.testing.assert_allclose(
        g.couple_local_nm, rot.T @ np.array(g.couple_at_midpoint_nm), atol=1e-12
    )
    np.testing.assert_allclose(g.net_force_local_n, rot.T @ np.array(g.net_force_n))


def test_local_components_none_without_rotation():
    g = _analyze(_hand("L", RL, (1, 0, 0)), _hand("R", RR, (1, 0, 0)))
    assert g.couple_local_nm is None and g.net_force_local_n is None


def test_about_axis_projects_and_normalises():
    assert about_axis((1.0, 2.0, 3.0), (0, 0, 2.0)) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        about_axis((1.0, 2.0, 3.0), (0, 0, 0))


@pytest.mark.parametrize("missing", ["L", "R"])
def test_missing_hand_gives_none_with_reason_never_zero(missing):
    hand_l = _hand("L", RL, (1, 2, 3)) if missing != "L" else None
    hand_r = _hand("R", RR, (1, 2, 3)) if missing != "R" else None
    g = _analyze(hand_l, hand_r)
    assert g.net_force_n is None
    assert g.couple_at_midpoint_nm is None
    assert g.midpoint_m is None
    assert g.contact_force_moment_nm is None
    assert g.mof_left_nm is None and g.mof_right_nm is None
    assert missing in g.unavailable_reason or "hand" in g.unavailable_reason
    assert (g.left is None) == (missing == "L")
    assert (g.right is None) == (missing == "R")


def test_both_missing_is_unavailable():
    g = _analyze(None, None)
    assert g.net_force_n is None and g.unavailable_reason


def test_missing_free_torque_leaves_couple_unavailable_but_mof_known():
    lw = HandWrench("L", RL, (30.0, 0, 0), None)
    rw = _hand("R", RR, (10.0, 0, 0))
    g = _analyze(lw, rw)
    assert g.net_force_n == (40.0, 0.0, 0.0)
    assert g.mof_left_nm is not None
    assert g.applied_free_torque_nm is None
    assert g.couple_at_midpoint_nm is None
    assert "torque" in g.unavailable_reason


def test_split_method_validated_and_recorded():
    assert set(SPLIT_METHODS) == {
        "constraint_multiplier",
        "efc_force",
        "allocation",
        "logged",
        "bushing",
        "contact",
        "unavailable",
    }
    g = _analyze(None, None, split_method="efc_force", metadata={"solver": "x"})
    assert g.split_method == "efc_force" and g.metadata["solver"] == "x"
    with pytest.raises(ValueError):
        _analyze(None, None, split_method="bogus")


def test_hand_wrench_validation():
    with pytest.raises(ValueError):
        HandWrench("X", RL, (0, 0, 0), None)
    with pytest.raises(ValueError):
        HandWrench("L", (0, 0), (0, 0, 0), None)
    with pytest.raises(ValueError):
        HandWrench("L", RL, (math.nan, 0, 0), None)
    with pytest.raises(ValueError):
        _analyze(_hand("R", RR, (0, 0, 0)), None)  # wrong slot
    with pytest.raises(ValueError):
        _analyze(None, None, club_rotation=np.eye(2))
    with pytest.raises(ValueError):
        _analyze(None, None, club_rotation=np.diag([1.0, 1.0, 2.0]))


def test_overlay_wrenches_labels_and_halves():
    g = _analyze(
        _hand("L", RL, (30.0, 0, 0), (0, 0, 1.0)),
        _hand("R", RR, (10.0, 0, 0), (0, 0, 1.0)),
    )
    ws = {w.label: w for w in to_overlay_wrenches(g, source="test:unit")}
    assert set(ws) == {
        "grip:hand_left",
        "grip:hand_right",
        "grip:net_midpoint",
        "grip:couple_midpoint",
        "grip:mof_left",
        "grip:mof_right",
    }
    assert all(w.kind is WrenchKind.GRIP and w.body == "club" for w in ws.values())
    assert ws["grip:net_midpoint"].torque_nm is None
    assert ws["grip:net_midpoint"].point_m == g.midpoint_m
    assert ws["grip:couple_midpoint"].force_n is None
    assert ws["grip:couple_midpoint"].torque_nm == g.couple_at_midpoint_nm
    assert ws["grip:mof_left"].force_n is None
    assert ws["grip:hand_left"].point_m == RL


def test_overlay_wrenches_skip_unavailable_quantities():
    g = _analyze(_hand("L", RL, (1, 0, 0)), None)
    labels = [w.label for w in to_overlay_wrenches(g, source="test:unit")]
    assert labels == ["grip:hand_left"]
    assert to_overlay_wrenches(_analyze(None, None), source="t:u") == []


def test_contact_reaction_wrench_matches_transport_helper():
    lw = _hand("L", RL, (30.0, 0, 0), (0, 0, 1.0))
    rw = _hand("R", RR, (10.0, 0, 0), (0, 0, 1.0))
    g = _analyze(lw, rw)
    sw = to_contact_reaction_wrench(g)
    assert sw.application_frame == "world" and sw.point_m == g.midpoint_m
    assert sw.force_n == g.net_force_n and sw.torque_nm == g.couple_at_midpoint_nm
    assert to_contact_reaction_wrench(_analyze(None, None)) is None


def test_uses_shared_move_wrench_point_semantics():
    lw = _hand("L", RL, (30.0, 0, 0), (0, 0, 1.0))
    g = _analyze(lw, _hand("R", RR, (0, 0, 0)))
    w = [x for x in to_overlay_wrenches(g, source="t:u") if x.label == "grip:hand_left"]
    moved = move_wrench_point(w[0], g.midpoint_m)
    np.testing.assert_allclose(
        np.array(moved.torque_nm) - np.array((0, 0, 1.0)), g.mof_left_nm, atol=1e-12
    )


def test_series_nan_for_unavailable_and_dataframe():
    good = _analyze(_hand("L", RL, (30.0, 0, 0)), _hand("R", RR, (10.0, 0, 0)))
    bad = _analyze(_hand("L", RL, (1, 0, 0)), None)
    s = GripSeries.from_analyses([0.0, 0.1], [good, bad])
    assert s.net_force_n.shape == (2, 3)
    assert np.isfinite(s.net_force_n[0]).all() and np.isnan(s.net_force_n[1]).all()
    assert s.split_method == ("logged", "logged")
    df = s.to_dataframe()
    assert len(df) == 2
    assert {"time_s", "net_force_x_n", "couple_z_nm", "split_method"} <= set(df.columns)
    assert df["unavailable_reason"].iloc[0] == ""
    assert df["unavailable_reason"].iloc[1] != ""
    with pytest.raises(ValueError):
        GripSeries.from_analyses([0.0], [good, bad])


def test_grip_analysis_is_frozen():
    g = _analyze(None, None)
    assert isinstance(g, GripAnalysis)
    with pytest.raises(AttributeError):
        g.split_method = "efc_force"  # type: ignore[misc]


def test_net_wrench_at_raises_when_unavailable():
    with pytest.raises(ValueError, match="unavailable"):
        _analyze(None, None).net_wrench_at((0, 0, 0))


def test_allocate_min_norm_reproduces_net_wrench():
    from src.shared.python.biomechanics.grip_wrench import allocate_min_norm

    lw = _hand("L", RL, (3.0, 4.0, -9.0))
    rw = _hand("R", RR, (-2.0, 1.0, 12.0))
    g = _analyze(lw, rw)
    f_l, f_r = allocate_min_norm(g)
    assert np.allclose(np.add(f_l, f_r), g.net_force_n)
    h = (np.array(RR) - np.array(RL)) / 2.0
    moment = np.cross(-h, f_l) + np.cross(h, f_r)
    couple_perp = np.array(g.couple_at_midpoint_nm)
    couple_perp = couple_perp - h * (h @ couple_perp) / (h @ h)
    assert np.allclose(moment, couple_perp)
    with pytest.raises(ValueError):
        allocate_min_norm(_analyze(None, None))
