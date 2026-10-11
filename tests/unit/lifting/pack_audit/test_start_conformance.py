"""Pure-logic tests for the LIFT-3 start-pose conformance checks (#11743)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.lifting.pack_audit import start_conformance as sc
from src.shared.python.lifting.pack_audit.model import CoordinateInfo

pytestmark = pytest.mark.unit


def _coord(name, lo, hi, default, offset=0.0):
    return CoordinateInfo(name, lo, hi, default, "hinge", offset)


# ---- bar start position ---------------------------------------------------


@pytest.mark.parametrize("lift", ["deadlift", "snatch", "clean_and_jerk"])
def test_pull_bar_on_floor_passes(lift):
    res = sc.check_bar_start(lift, bar_above_sole_m=0.225, positions=None)
    assert res.status == "pass" and res.value == pytest.approx(0.0)


@pytest.mark.parametrize("height", [0.70, 0.80, 0.10])
def test_pull_bar_off_floor_fails(height):
    res = sc.check_bar_start("deadlift", height, positions=None)
    assert res.status == "fail"
    assert res.value == pytest.approx(abs(height - 0.225))


def test_pull_bar_without_sole_is_unavailable_not_zero():
    res = sc.check_bar_start("deadlift", None, positions=None)
    assert res.status == "unavailable" and res.value is None and res.reason


def _upper_body(bar, shoulder_z=1.6):
    return {
        "barbell_shaft": np.array(bar, float),
        "upper_arm_l": np.array([0.0, 0.17, shoulder_z]),
        "upper_arm_r": np.array([0.0, -0.17, shoulder_z]),
    }


def test_squat_bar_on_shoulders_passes():
    pos = _upper_body([-0.05, 0.0, 1.62])
    assert sc.check_bar_start("squat", 1.6, pos).status == "pass"


def test_squat_bar_on_floor_fails():
    pos = _upper_body([0.0, 0.0, 0.225])
    assert sc.check_bar_start("squat", 0.225, pos).status == "fail"


def test_bench_bar_over_chest_passes():
    pos = _upper_body([0.0, 0.0, 2.0], shoulder_z=1.4)
    assert sc.check_bar_start("bench_press", 0.9, pos).status == "pass"


def test_bench_bar_off_midline_fails():
    pos = _upper_body([0.0, -0.23, 2.0], shoulder_z=1.4)
    assert sc.check_bar_start("bench_press", 0.9, pos).status == "fail"


def test_bench_bar_below_shoulders_fails():
    pos = _upper_body([0.0, 0.0, 1.3], shoulder_z=1.4)
    assert sc.check_bar_start("bench_press", 0.9, pos).status == "fail"


def test_bar_start_rejects_unknown_lift_and_missing_bodies():
    with pytest.raises(ValueError, match="unknown lift"):
        sc.check_bar_start("curl", 0.2, None)
    with pytest.raises(ValueError, match="upper_arm"):
        sc.check_bar_start("squat", 1.0, {"barbell_shaft": np.zeros(3)})


# ---- feet -----------------------------------------------------------------


def test_feet_flat_and_on_ground_pass():
    res = sc.check_feet_flat(0.07, 0.07, sole_z=0.0, ground_z=0.0)
    assert res.status == "pass"


def test_feet_unlevel_fails():
    res = sc.check_feet_flat(0.07, 0.09, sole_z=0.0, ground_z=0.0)
    assert res.status == "fail" and res.value == pytest.approx(0.02)


def test_feet_floating_above_ground_fails():
    res = sc.check_feet_flat(0.57, 0.57, sole_z=0.49, ground_z=0.0)
    assert res.status == "fail"


def test_feet_without_ground_reference_is_unavailable():
    res = sc.check_feet_flat(0.07, 0.07, sole_z=-0.9, ground_z=None)
    assert res.status == "unavailable" and "ground" in res.reason


def test_feet_without_sole_is_unavailable():
    assert sc.check_feet_flat(0.07, 0.07, None, 0.0).status == "unavailable"


def test_feet_reject_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        sc.check_feet_flat(math.nan, 0.0, 0.0, 0.0)


# ---- closure --------------------------------------------------------------


def test_closure_within_5mm_passes():
    res = sc.check_hand_bar_closure({"l": 0.004, "r": 0.001})
    assert res.status == "pass" and res.value == pytest.approx(0.004)


def test_closure_over_5mm_fails():
    res = sc.check_hand_bar_closure({"l": 0.049, "r": 0.049})
    assert res.status == "fail" and res.limit == pytest.approx(0.005)


def test_closure_missing_or_nan_is_unavailable():
    for bad in (None, {}, {"l": math.nan, "r": 0.0}):
        res = sc.check_hand_bar_closure(bad, reason="no closure")
        assert res.status == "unavailable" and res.value is None


def test_closure_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        sc.check_hand_bar_closure({"l": -0.1})


# ---- joint limits ---------------------------------------------------------


def test_joint_limits_pass():
    coords = [_coord("elbow_l", 0.0, 2.5, 1.0)]
    assert sc.check_joint_limits(coords, {}).status == "pass"


def test_joint_default_outside_fails():
    coords = [_coord("shoulder_l_flex", -1.0, 1.0, 1.5)]
    res = sc.check_joint_limits(coords, {})
    assert res.status == "fail" and res.value == 1.0
    assert "shoulder_l_flex" in res.detail[0]


def test_joint_phase_target_uses_zero_offset():
    coords = [_coord("k", 0.5, 1.5, 1.0, offset=1.0)]
    ok = sc.check_joint_limits(coords, {"start": {"k": 0.4}})
    bad = sc.check_joint_limits(coords, {"start": {"k": 0.6}})
    assert ok.status == "pass" and bad.status == "fail"
    assert "start" in bad.detail[0]


def test_joint_unlimited_coordinates_ignored():
    coords = [_coord("root", None, None, 0.0)]
    assert sc.check_joint_limits(coords, {"p": {"root": 9.0}}).status == "pass"


def test_joint_unknown_target_coordinate_raises():
    with pytest.raises(ValueError, match="unknown coordinate"):
        sc.check_joint_limits([_coord("a", 0, 1, 0)], {"p": {"zzz": 0.0}})


# ---- static GRF -----------------------------------------------------------


def test_grf_within_2_percent_passes():
    res = sc.check_static_grf(1960.0, 200.0 * 9.80665)
    assert res.status == "pass"


def test_grf_off_fails():
    res = sc.check_static_grf(237.0, 1961.33)
    assert res.status == "fail" and res.value > 0.8


def test_grf_none_is_unavailable_with_reason():
    res = sc.check_static_grf(None, 1961.33, reason="no contact API")
    assert res.status == "unavailable" and res.value is None
    assert res.reason == "no contact API"


def test_grf_rejects_bad_expected():
    with pytest.raises(ValueError, match="expected_n"):
        sc.check_static_grf(100.0, 0.0)


def test_expected_weight():
    masses = {"pelvis": 10.0, "barbell_shaft": 20.0, "bench": 50.0}
    assert sc.expected_support_n(masses) == pytest.approx(30.0 * 9.80665)
    with pytest.raises(ValueError, match="mass"):
        sc.expected_support_n({"bench": 5.0})


# ---- record ---------------------------------------------------------------


def test_record_aggregates():
    ok = sc.CheckResult("a", "pass", 0.0, 1.0, "m")
    bad = sc.CheckResult("b", "fail", 2.0, 1.0, "m")
    na = sc.CheckResult("c", "unavailable", None, 1.0, "m", reason="x")
    rec = sc.StartConformance("mujoco", "deadlift", (ok, bad, na))
    assert not rec.passed
    assert [c.name for c in rec.failing] == ["b"]
    assert rec.check("c") is na
    assert rec.as_dict()["checks"][2]["value"] is None
    with pytest.raises(KeyError):
        rec.check("zzz")
    assert sc.StartConformance("mujoco", "squat", (ok, na)).passed


def test_check_result_rejects_bad_status_and_value_with_unavailable():
    with pytest.raises(ValueError, match="status"):
        sc.CheckResult("a", "maybe", 0.0, 1.0, "m")
    with pytest.raises(ValueError, match="unavailable"):
        sc.CheckResult("a", "unavailable", 0.0, 1.0, "m", reason="x")
    with pytest.raises(ValueError, match="reason"):
        sc.CheckResult("a", "unavailable", None, 1.0, "m")


# ---- engine glue with stub adapters ----------------------------------------


class _Stub:
    def __init__(self, hold=None, raises=False):
        self._hold, self._raises = hold, raises

    def bar_hold_wrench(self):
        if self._raises:
            raise RuntimeError("boom")
        return self._hold


def test_closure_prefers_adapter_closure():
    got, why = sc._closure_of(_Stub(), {"l": 0.001, "r": 0.002})
    assert got == {"l": 0.001, "r": 0.002} and why == ""
    got, _ = sc._closure_of(_Stub(), {"weld_translation_norm": 1e-9, "n_qerr": 6.0})
    assert got == {"weld": 1e-9}


def test_closure_falls_back_to_bar_hold_residual():
    hold = {"available": True, "hand_grip_residual_m": {"L": 0.0, "R": 0.003}}
    got, _ = sc._closure_of(_Stub(hold), None)
    assert got == {"L": 0.0, "R": 0.003}


def test_closure_unavailable_carries_reason():
    got, why = sc._closure_of(_Stub({"available": False, "reason": "nope"}), {})
    assert got is None and "nope" in why
    got, why = sc._closure_of(_Stub(raises=True), None)
    assert got is None and "boom" in why


def test_grf_check_reports_contact_count_context():
    res = sc._grf_check({"value_n": 0.0, "n_ground_contacts": 0}, 1961.33)
    assert res.status == "fail" and "n_ground_contacts=0" in res.detail
    res = sc._grf_check({"value_n": None, "reason": "why"}, 1961.33)
    assert res.status == "unavailable" and res.reason == "why"
