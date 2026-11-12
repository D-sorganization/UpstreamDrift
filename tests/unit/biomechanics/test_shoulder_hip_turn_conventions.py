"""Tests for GM-1c (#14424): shoulder and hip turn conventions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.biomechanics.metric_definitions.schema import (
    GolfMetricDefinitionEntry,
    NormalisationKind,
)
from src.shared.python.biomechanics.shoulder_hip_turn_conventions import (
    CHU_2010_EVENT_ANCHORS,
    TURN_CONVENTIONS_REGISTRY_VERSION,
    TURN_METRIC_DEFINITIONS,
    SegmentProxy,
    TurnReferenceKind,
    evaluate_turn_from_points,
    evaluate_turn_from_rotations,
    evaluate_turn_series,
    turn_metric_definition,
    turn_publish_payload,
    unwrap_turn_deg,
    wrap_angle_deg,
)
from src.shared.python.swing_comparison.events import SwingEvents

pytestmark = pytest.mark.unit

EXPECTED_METRIC_IDS: tuple[str, ...] = tuple(
    f"{s}_turn_{r}_{p}"
    for s in ("shoulder", "hip")
    for r in ("address_relative", "target_line")
    for p in ("joint_centres", "landmarks")
)


def _make_events(n: int = 10) -> SwingEvents:
    return SwingEvents(0, 0.0, 4, 0.04, 7, 0.07, n - 1, (n - 1) * 0.01)


def test_turn_metrics_registered_and_conform_to_schema() -> None:
    assert len(TURN_METRIC_DEFINITIONS) == 8
    for mid in EXPECTED_METRIC_IDS:
        e = turn_metric_definition(mid)
        assert isinstance(e, GolfMetricDefinitionEntry) and e.metric_id == mid
        assert e.event_anchors == CHU_2010_EVENT_ANCHORS
        assert e.definition_version == TURN_CONVENTIONS_REGISTRY_VERSION
        assert any("02640414.2010.507249" in c or "chu" in c.lower() for c in e.cites)
        want_norm = (
            NormalisationKind.address_relative
            if "address_relative" in mid
            else NormalisationKind.none
        )
        assert e.normalisation == want_norm
        want_prx = (
            SegmentProxy.joint_centre_line.value
            if "joint_centres" in mid
            else SegmentProxy.thorax_pelvis_landmarks.value
        )
        assert e.landmark_proxy == want_prx


def test_wrap_and_unwrap_branch_cut_with_nans() -> None:
    for orig in (np.linspace(150.0, 205.0, 56), np.linspace(-150.0, -205.0, 56)):
        wrapped = wrap_angle_deg(orig)
        assert np.all((wrapped >= -180.0) & (wrapped < 180.0))
        np.testing.assert_allclose(unwrap_turn_deg(wrapped), orig, atol=1e-9)

    orig_nan = np.linspace(150.0, 205.0, 56)
    orig_nan[[10, 25]] = np.nan
    unwrapped_nan = unwrap_turn_deg(wrap_angle_deg(orig_nan))
    assert np.isnan(unwrapped_nan[10]) and np.isnan(unwrapped_nan[25])
    val = np.isfinite(orig_nan)
    np.testing.assert_allclose(unwrapped_nan[val], orig_nan[val], atol=1e-9)


def test_known_answer_synthetic_rotations_and_points() -> None:
    up = np.array([0.0, 0.0, 1.0])
    target = np.array([0.0, 1.0, 0.0])
    ev = _make_events(10)
    ang = np.linspace(-10.0, 45.0, 10)
    ang[0], ang[ev.top_idx], ang[ev.impact_idx] = -10.0, 45.0, -30.0
    rot = Rotation.from_rotvec(np.deg2rad(ang)[:, None] * up)
    d = rot.apply(target)
    left = np.array([0.0, 0.0, 1.0]) + 0.15 * d
    right = np.array([0.0, 0.0, 1.0]) - 0.15 * d

    r_tl = evaluate_turn_from_rotations(
        "hip_turn_target_line_joint_centres", rot, events=ev, up=up, target=target
    )
    r_pts = evaluate_turn_from_points(
        "hip_turn_target_line_landmarks", left, right, events=ev, up=up, target=target
    )
    assert r_tl.reference_kind == TurnReferenceKind.target_line_lab_frame
    np.testing.assert_allclose(r_pts.turn_deg, r_tl.turn_deg, atol=1e-9)
    assert (
        r_tl.turn_deg[ev.address_idx],
        r_tl.turn_deg[ev.top_idx],
        r_tl.turn_deg[ev.impact_idx],
    ) == pytest.approx((-10.0, 45.0, -30.0))

    r_ar = evaluate_turn_from_rotations(
        "hip_turn_address_relative_joint_centres", rot, events=ev, up=up, target=target
    )
    assert r_ar.reference_kind == TurnReferenceKind.address_relative
    assert (
        r_ar.turn_deg[ev.address_idx],
        r_ar.turn_deg[ev.top_idx],
        r_ar.turn_deg[ev.impact_idx],
    ) == pytest.approx((0.0, 55.0, -20.0))

    ang0 = ang.copy()
    ang0[ev.address_idx] = 0.0
    r0 = Rotation.from_rotvec(np.deg2rad(ang0)[:, None] * up)
    np.testing.assert_allclose(
        evaluate_turn_from_rotations(
            "hip_turn_target_line_joint_centres", r0, events=ev, up=up, target=target
        ).turn_deg,
        evaluate_turn_from_rotations(
            "hip_turn_address_relative_joint_centres",
            r0,
            events=ev,
            up=up,
            target=target,
        ).turn_deg,
        atol=1e-9,
    )


def test_event_reporting_and_publish_payload() -> None:
    up = np.array([0.0, 0.0, 1.0])
    target = np.array([0.0, 1.0, 0.0])
    ev = _make_events(10)
    rot = Rotation.from_rotvec(np.linspace(0.0, 0.5, 10)[:, None] * up)
    res = evaluate_turn_series(
        "shoulder_turn_address_relative_landmarks", rot, events=ev, up=up, target=target
    )
    assert res.event_values_deg == {
        "address": pytest.approx(res.turn_deg[ev.address_idx]),
        "top": pytest.approx(res.turn_deg[ev.top_idx]),
        "impact": pytest.approx(res.turn_deg[ev.impact_idx]),
        "finish": pytest.approx(res.turn_deg[ev.finish_idx]),
    }
    payload = turn_publish_payload(res)
    assert payload["metric_id"] == "shoulder_turn_address_relative_landmarks"
    assert payload["events_deg"] == res.event_values_deg
    assert len(payload["values_deg"]) == 10
    res_none = evaluate_turn_series(
        "shoulder_turn_address_relative_landmarks",
        rot,
        events=None,
        up=up,
        target=target,
    )
    assert (
        res_none.event_values_deg is None
        and turn_publish_payload(res_none)["events_deg"] is None
    )


def test_dbc_preconditions() -> None:
    rot = Rotation.identity(10)
    with pytest.raises(ValueError, match="unknown"):
        turn_metric_definition("unknown_metric")
    with pytest.raises(ValueError, match="mismatched"):
        evaluate_turn_from_points(
            "hip_turn_target_line_landmarks", np.zeros((10, 3)), np.zeros((8, 3))
        )
    with pytest.raises(ValueError, match="parallel"):
        evaluate_turn_from_rotations(
            "hip_turn_target_line_joint_centres", rot, up=[0, 0, 1], target=[0, 0, 1]
        )
    with pytest.raises(ValueError, match="out of range"):
        evaluate_turn_from_rotations(
            "hip_turn_target_line_joint_centres",
            rot,
            events=SwingEvents(0, 0.0, 4, 0.04, 7, 0.07, 99, 0.99),
        )
