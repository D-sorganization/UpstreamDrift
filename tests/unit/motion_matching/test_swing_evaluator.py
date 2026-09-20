"""Unit tests for SwingEvaluator (MS-31 / MS-104 / #10415).

Verifies:
1. Segment-by-segment marker tracking breakdown.
2. Phase-by-phase swing metrics (address, backswing, downswing, impact, follow-through).
3. Contact sphere ground penetration auditing.
4. Weld closure and marker coverage reporting.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.swing_evaluator import (
    SwingEvaluationReport,
    SwingEvaluator,
    SwingPhase,
)


@pytest.mark.unit
def test_swing_evaluator_segment_and_phase_breakdown() -> None:
    """Verify segment assignment and phase aggregation."""
    labels = (
        "Marker_2:2:1",
        "Marker_2:2:2",
        "LToeIn",
        "RToeIn",
        "LWristTop",
        "WaistLeft",
        "HeadTop",
    )
    n_markers = len(labels)
    n_nodes = 100
    dt = 1.0 / 360.0
    time_s = np.asarray(np.arange(n_nodes) * dt, dtype=np.float64)

    pred_markers = np.zeros((n_nodes, n_markers, 3))
    targ_markers = np.zeros((n_nodes, n_markers, 3))
    valid = np.ones((n_nodes, n_markers), dtype=bool)

    # Inject known errors: club has 10 mm error (0.010 m)
    pred_markers[:, 0, 0] = 0.010
    pred_markers[:, 1, 0] = 0.010
    # Foot has 5 mm error (0.005 m)
    pred_markers[:, 2, 0] = 0.005
    pred_markers[:, 3, 0] = 0.005

    evaluator = SwingEvaluator(labels=labels)
    report = evaluator.evaluate(
        time_s=time_s,
        pred_markers=pred_markers,
        target_markers=targ_markers,
        valid=valid,
        sphere_bottom_z=np.zeros((n_nodes, 6)) + 0.001,  # 1 mm above ground
        ground_height_m=0.0,
        closure_errors_m=np.ones(n_nodes) * 0.002,  # 2 mm closure error
    )

    assert isinstance(report, SwingEvaluationReport)
    # Club segment RMSE should be ~10 mm
    assert np.isclose(report.segments["club"].rmse_mm, 10.0, atol=0.1)
    # Feet segment RMSE should be ~5 mm
    assert np.isclose(report.segments["feet"].rmse_mm, 5.0, atol=0.1)
    # Ground penetration: bottom_z is +1 mm, so penetration should be 0.0 mm
    assert np.isclose(report.ground_penetration.max_penetration_mm, 0.0)
    assert np.isclose(report.closure.max_closure_mm, 2.0, atol=0.1)
    assert np.isclose(report.coverage_fraction, 1.0)


@pytest.mark.unit
def test_ground_penetration_audit() -> None:
    """Verify ground penetration detection across contact spheres."""
    labels = ("LToeIn", "RToeIn")
    n_nodes = 10
    time_s = np.asarray(np.arange(n_nodes) * 0.01, dtype=np.float64)
    pred = np.zeros((n_nodes, 2, 3))
    targ = np.zeros((n_nodes, 2, 3))
    valid = np.ones((n_nodes, 2), dtype=bool)

    # 6 spheres; sphere 0 penetrates to -0.015 m at frame 5
    sphere_bottom_z = np.zeros((n_nodes, 6))
    sphere_bottom_z[5, 0] = -0.015  # 15 mm underground

    evaluator = SwingEvaluator(labels=labels)
    report = evaluator.evaluate(
        time_s=time_s,
        pred_markers=pred,
        target_markers=targ,
        valid=valid,
        sphere_bottom_z=sphere_bottom_z,
        ground_height_m=0.0,
        closure_errors_m=np.zeros(n_nodes),
    )

    assert np.isclose(report.ground_penetration.max_penetration_mm, 15.0, atol=0.1)
    assert report.ground_penetration.worst_frame == 5


@pytest.mark.unit
def test_swing_evaluator_empty_population_fails_closed() -> None:
    """Empty-population markers must yield NaN RMSE rather than 0.0 (preventing false zero-success)."""
    labels = ("Marker_1", "Marker_2")
    n_nodes = 20
    time_s = np.linspace(0.0, 0.19, n_nodes)
    pred = np.zeros((n_nodes, 2, 3))
    targ = np.zeros((n_nodes, 2, 3))
    # All markers invalid (empty population)
    valid = np.zeros((n_nodes, 2), dtype=bool)

    evaluator = SwingEvaluator(labels=labels)
    report = evaluator.evaluate(
        time_s=time_s,
        pred_markers=pred,
        target_markers=targ,
        valid=valid,
        sphere_bottom_z=np.zeros((n_nodes, 6)) + 0.01,
        ground_height_m=0.0,
        closure_errors_m=np.zeros(n_nodes),
    )

    assert np.isnan(report.overall_rmse_mm), (
        "Empty population overall_rmse_mm must be NaN"
    )
    assert report.coverage_fraction == 0.0
    for seg in report.segments.values():
        assert np.isnan(seg.rmse_mm), f"Empty segment {seg.name} RMSE must be NaN"


@pytest.mark.unit
def test_swing_evaluator_closure_audit_separates_translation_and_rotation() -> None:
    """Closure audit must distinctly measure translation (mm) and rotation (rad/deg)."""
    labels = ("Marker_1",)
    n_nodes = 10
    time_s = np.linspace(0.0, 0.09, n_nodes)
    pred = np.zeros((n_nodes, 1, 3))
    targ = np.zeros((n_nodes, 1, 3))
    valid = np.ones((n_nodes, 1), dtype=bool)

    closure_trans_m = np.ones(n_nodes) * 0.003  # 3 mm
    closure_rot_rad = np.ones(n_nodes) * 0.02  # 0.02 rad

    evaluator = SwingEvaluator(labels=labels)
    report = evaluator.evaluate(
        time_s=time_s,
        pred_markers=pred,
        target_markers=targ,
        valid=valid,
        sphere_bottom_z=np.zeros((n_nodes, 6)) + 0.01,
        ground_height_m=0.0,
        closure_errors_m=closure_trans_m,
        closure_rotations_rad=closure_rot_rad,
    )

    assert np.isclose(report.closure.max_closure_translation_mm, 3.0, atol=1e-3)
    assert np.isclose(report.closure.max_closure_rotation_rad, 0.02, atol=1e-4)
    # Backward compatibility properties
    assert np.isclose(report.closure.max_closure_mm, 3.0, atol=1e-3)


@pytest.mark.unit
def test_swing_evaluator_does_not_invent_impact_without_event() -> None:
    """Evaluator must not fabricate impact phase as a fixed fraction of capture when events are missing."""
    labels = ("Marker_1",)
    n_nodes = 50
    time_s = np.linspace(0.0, 1.0, n_nodes)
    pred = np.zeros((n_nodes, 1, 3))
    targ = np.zeros((n_nodes, 1, 3))
    valid = np.ones((n_nodes, 1), dtype=bool)

    evaluator = SwingEvaluator(labels=labels)
    # Without t_events
    report_no_event = evaluator.evaluate(
        time_s=time_s,
        pred_markers=pred,
        target_markers=targ,
        valid=valid,
        sphere_bottom_z=np.zeros((n_nodes, 6)) + 0.01,
        ground_height_m=0.0,
        closure_errors_m=np.zeros(n_nodes),
        t_events=None,
    )
    assert "impact" not in report_no_event.phases, (
        "Should not invent impact phase without declared event"
    )

    # With declared t_events
    report_with_event = evaluator.evaluate(
        time_s=time_s,
        pred_markers=pred,
        target_markers=targ,
        valid=valid,
        sphere_bottom_z=np.zeros((n_nodes, 6)) + 0.01,
        ground_height_m=0.0,
        closure_errors_m=np.zeros(n_nodes),
        t_events={"impact": 0.65, "top_of_backswing": 0.40},
    )
    assert "impact" in report_with_event.phases
    impact_metric = report_with_event.phases["impact"]
    # Impact window should be centered on declared 0.65 s, not arbitrary 0.72-0.78
    assert impact_metric.start_time_s <= 0.65 <= impact_metric.end_time_s
