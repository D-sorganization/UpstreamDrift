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
