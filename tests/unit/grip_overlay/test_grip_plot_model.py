"""Plot data series for the grip force and couple plots (GCV-10, #11716)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.shared.python.biomechanics.grip_extraction import (
    net_only_analysis,
    unavailable_analysis,
)
from src.shared.python.biomechanics.grip_plot_model import (
    build_grip_plot_series,
    plot_series_to_json,
)
from src.shared.python.biomechanics.grip_wrench import (
    GripSeries,
    HandWrench,
    analyze_grip,
)

pytestmark = pytest.mark.unit

RL = (0.0, 0.0, 1.0)
RR = (0.0, 0.0, 0.8)


def _pair(f_l, f_r, tau=(0.0, 0.0, 0.0), method="efc_force"):
    left = HandWrench("L", RL, f_l, tau)
    right = HandWrench("R", RR, f_r, tau)
    return analyze_grip(left, right, split_method=method)


def test_equal_and_opposite_hand_forces_give_zero_net_and_pure_couple():
    g = _pair((30.0, 0, 0), (-30.0, 0, 0))
    assert g.net_force_n == pytest.approx((0, 0, 0), abs=1e-12)
    # (r_L - r_M) x F_L + (r_R - r_M) x F_R with r_L - r_M = +0.1 z
    # = 0.1 z x 30 x + (-0.1 z) x (-30 x) = 6 y
    assert g.couple_at_midpoint_nm == pytest.approx((0, 6.0, 0), abs=1e-12)
    assert g.applied_free_torque_nm == pytest.approx((0, 0, 0), abs=1e-12)


def test_series_carries_per_hand_forces_and_local_couple():
    rot = np.eye(3)
    analyses = [
        analyze_grip(
            HandWrench("L", RL, (1, 2, 3), (0, 0, 0)),
            HandWrench("R", RR, (4, 5, 6), (0, 0, 0)),
            split_method="efc_force",
            club_rotation=rot,
        )
    ]
    s = GripSeries.from_analyses([0.0], analyses)
    np.testing.assert_allclose(s.left_force_n, [[1, 2, 3]])
    np.testing.assert_allclose(s.right_force_n, [[4, 5, 6]])
    np.testing.assert_allclose(s.couple_local_nm, s.couple_nm)


def test_plot_series_traces_and_units():
    t = [0.0, 0.01, 0.02]
    analyses = [_pair((10.0 * k, 0, 0), (-10.0 * k, 0, 0)) for k in range(3)]
    p = build_grip_plot_series(t, analyses, events={"impact": 0.02})
    assert p.available is True
    assert p.split_method == "efc_force"
    assert p.events == {"impact": 0.02}
    names = set(p.traces)
    assert {
        "left_force_n",
        "right_force_n",
        "net_force_n",
        "couple_nm",
        "couple_local_nm",
        "contact_force_moment_nm",
        "applied_free_torque_nm",
    } <= names
    left = p.traces["left_force_n"]
    assert left["x"] == [0.0, 10.0, 20.0]
    assert left["magnitude"] == pytest.approx([0.0, 10.0, 20.0])
    assert p.traces["net_force_n"]["magnitude"] == pytest.approx([0, 0, 0])
    assert p.traces["couple_nm"]["magnitude"] == pytest.approx([0.0, 2.0, 4.0])


def test_unavailable_samples_are_none_never_zero():
    net_only = net_only_analysis(
        point_m=(0, 0, 0.9),
        force_on_club_n=(5.0, 0, 0),
        torque_on_club_nm=(0, 1.0, 0),
        split_method="allocation",
        reason="allocation yields one net 6-D wrench",
    )
    p = build_grip_plot_series(
        [0.0, 0.1], [_pair((1, 0, 0), (1, 0, 0)), net_only]
    )
    assert p.traces["left_force_n"]["x"][1] is None
    assert p.traces["left_force_n"]["magnitude"][1] is None
    assert p.traces["net_force_n"]["x"][1] == 5.0
    assert p.split_method_by_sample == ("efc_force", "allocation")
    assert p.unavailable_reasons[1]
    # the JSON form has no NaN and parses strictly
    text = plot_series_to_json(p)
    assert "NaN" not in text
    json.loads(text, parse_constant=_reject)


def _reject(token):  # pragma: no cover - fails the test if hit
    raise AssertionError(f"non-finite JSON constant {token}")


def test_fully_unavailable_series_reports_reason():
    p = build_grip_plot_series(
        [0.0, 0.1], [unavailable_analysis("no grip welds")] * 2
    )
    assert p.available is False
    assert "no grip welds" in p.reason
    assert p.split_method == "unavailable"
    assert all(v is None for v in p.traces["net_force_n"]["x"])


def test_mixed_split_methods_are_named():
    p = build_grip_plot_series(
        [0.0, 0.1],
        [_pair((1, 0, 0), (1, 0, 0), method="efc_force"),
         _pair((1, 0, 0), (1, 0, 0), method="bushing")],
    )
    assert p.split_method == "mixed"


def test_preconditions():
    with pytest.raises(ValueError):
        build_grip_plot_series([0.0], [])
    with pytest.raises(ValueError):
        build_grip_plot_series([], [])
    with pytest.raises(ValueError):
        build_grip_plot_series([0.0], [_pair((1, 0, 0), (1, 0, 0))], events={"x": math.nan})
