"""Pure two-hand grip-margin metric over pad-contact wrenches (#13885 PR 2)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import GripInterface
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.grip_margin import (
    UNLOADED,
    GripMarginReport,
    grasp_wrenches,
    grip_margin_report,
    min_loaded_margin,
    pad_margins,
    slip_events,
)
from src.shared.python.grip_contact.pad_contact import (
    PadWrench,
    build_pad_model,
    pad_wrench,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
FLOOR = 1.0


def _wrench(normal, tangential, force=(0.0, 0.0, 0.0), moment=(0.0, 0.0, 0.0)):
    return PadWrench(
        np.asarray(force, float),
        np.asarray(moment, float),
        np.asarray(normal, float),
        np.asarray(tangential, float),
    )


def test_static_hold_at_the_critical_friction_has_zero_margin() -> None:
    normal, tangential = [10.0, 20.0, 40.0], [5.0, 10.0, 20.0]
    mu_crit = max(t / n for t, n in zip(tangential, normal, strict=True))
    margins = pad_margins(_wrench(normal, tangential), mu_crit, FLOOR)
    assert margins == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert pad_margins(_wrench(normal, tangential), 2 * mu_crit, FLOOR) == (
        pytest.approx((0.5, 0.5, 0.5))
    )


def test_pure_internal_squeeze_has_zero_net_wrench_and_positive_margin() -> None:
    f, m = np.array([0.0, 300.0, -40.0]), np.array([1.5, 0.0, 2.0])
    lead = _wrench([300.0, 300.0], [10.0, 20.0], f, m)
    trail = _wrench([300.0, 300.0], [10.0, 20.0], -f, -m)
    net, internal = grasp_wrenches(lead, trail)
    assert np.array_equal(net, np.zeros(6))
    assert np.array_equal(internal, np.concatenate([f, m]))
    assert min_loaded_margin(pad_margins(lead, 0.9, FLOOR)) > 0.0


def test_pads_below_the_floor_are_unloaded_and_excluded_from_the_minimum() -> None:
    margins = pad_margins(_wrench([0.2, 0.0, 50.0], [5.0, 0.0, 10.0]), 0.5, FLOOR)
    assert margins[:2] == (UNLOADED, UNLOADED)
    assert min_loaded_margin(margins) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="no loaded pad"):
        min_loaded_margin((UNLOADED,))


def test_slip_events_come_from_tangential_pad_speed_above_threshold() -> None:
    rng = np.random.default_rng(13885)
    speed = rng.uniform(0.0, 0.0009, size=(200, 4))  # every pad sticks
    speed[20:25, 1] = 0.05  # 5 frames on one pad
    speed[100:103, 0] = speed[101:104, 3] = 0.02  # overlapping pads: one event
    count, duration_s = slip_events(speed, dt_s=0.001, threshold_m_s=0.001)
    assert count == 2
    assert duration_s == pytest.approx(0.009)
    assert slip_events(speed[:15], 0.001, 0.001) == (0, 0.0)


def _series(n: int = 5):
    lead = [_wrench([100.0, 100.0], [10.0 * (k + 1)] * 2) for k in range(n)]
    trail = [_wrench([100.0, 0.1], [5.0, 3.0]) for _ in range(n)]
    return {"lead": lead, "trail": trail}


def test_report_takes_the_minimum_margin_at_peak_club_acceleration() -> None:
    accel = np.array([[1.0, 0, 0], [0, 9.0, 0], [3.0, 0, 0], [0, 0, 2.0], [1, 1, 1]])
    speed = np.zeros((5, 4))
    speed[3:, 2] = 0.01
    report = grip_margin_report(
        _series(), accel, speed, 0.002, 0.5, FLOOR, 0.001, mu_bias_bound=1.07
    )
    assert isinstance(report, GripMarginReport)
    assert report.peak_frame == 1
    assert report.pad_margin["trail"][1] == UNLOADED
    assert report.min_margin == pytest.approx(0.6)  # lead pads, 20 N of 50 N
    assert (report.slip_event_count, report.slip_duration_s) == (1, 0.004)
    assert report.net_wrench.shape == report.internal_wrench.shape == (6,)
    assert report.mu_bias_bound == 1.07


@pytest.mark.parametrize(
    ("friction", "floor", "match"),
    [(0.0, 1.0, "friction"), (math.nan, 1.0, "friction"), (0.5, 0.0, "floor")]
    + [(0.5, -1.0, "floor"), (0.5, math.inf, "floor")],
)
def test_contract_rejects_bad_friction_and_floor(friction, floor, match) -> None:
    with pytest.raises(ValueError, match=match):
        pad_margins(_wrench([10.0], [1.0]), friction, floor)


def test_contract_rejects_non_finite_inputs_and_reports() -> None:
    with pytest.raises(ValueError, match="finite"):
        pad_margins(_wrench([math.nan], [1.0]), 0.5, FLOOR)
    with pytest.raises(ValueError, match="dt_s"):
        slip_events(np.zeros((3, 1)), 0.0, 0.001)
    with pytest.raises(ValueError, match="finite"):
        GripMarginReport(
            0.5, FLOOR, {}, math.nan, 0, 0, 0.0, np.zeros(6), np.zeros(6), None
        )
    for bound in (0.0, -1.07):
        with pytest.raises(ValueError, match="mu_bias_bound"):
            GripMarginReport(
                0.5, FLOOR, {}, 0.5, 0, 0, 0.0, np.zeros(6), np.zeros(6), bound
            )
    with pytest.raises(ValueError, match="frames"):
        grip_margin_report(_series(), np.ones((4, 3)), np.zeros((5, 1)), 0.01, 0.5)


def test_consumes_pad_contacts_cone_for_a_resting_hand() -> None:
    interface = GripInterface.from_spec(json.loads(SPEC.read_text()))
    model = build_pad_model(interface, 1000.0)
    hand = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    club = BushingState(np.eye(3), np.zeros(3), np.array([3e-4, 0, 0]), np.zeros(3))
    wrench = pad_wrench(model, "R", hand, club)
    margins = pad_margins(wrench, model.law.static_friction, FLOOR)
    loaded = [m for m in margins if m != UNLOADED]
    assert loaded and all(-1e-9 <= m <= 1.0 for m in loaded)
