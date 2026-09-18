"""Negative acceptance regression fixtures (PF-01 / #10431).

Verifies:
1. Friction violation: tangential to normal force ratio > mu (0.8) fails acceptance.
2. Missing root histories: dynamics candidates missing delta_tau_root fail closed.
3. Nonzero root assistance: phantom root forces (> 0.1 N) fail acceptance.
4. Empty marker population: does not return 0.0 mm success (must be NaN).
5. Closure translation vs rotation: separated audit for position (m) and rotation (rad).
6. Phase event truthfulness: supplied t_events dictate phase windows, not 72% fraction.
7. Truncated horizon: G1 duration (0.85 s) evaluated under G3 is rejected.
8. 44 vs 41 coordinate mismatch: dimension mismatch fails closed.
9. CandidatePackage contract: serialization, controls, root history, and legacy compatibility.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    GateStatus,
    Horizon,
    evaluate,
)
from src.shared.python.motion_matching.candidate_package import (
    CandidatePackage,
    check_coordinate_mapping,
)
from src.shared.python.motion_matching.swing_evaluator import (
    ClosureAudit,
    SwingEvaluator,
    SwingPhase,
)

pytestmark = [pytest.mark.unit]


def test_rejection_on_friction_cone_violation() -> None:
    """Friction cone violation (> 0.8) must fail physical acceptance."""
    receipt: dict[str, Any] = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.030,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.040,
            "club_marker_rmse_m": 0.040,
            "pelvis_yaw_rmse_rad": 0.03,
        },
        "contact_audit": {
            "max_normal_force_n": 1500.0,
            "max_penetration_m": 0.002,
            "max_friction_ratio": 2.5,  # Violated! Exceeds mu=0.8
            "friction_violation_count": 45,
        },
        "closure": {
            "max_closure_residual_m": 0.002,
        },
        "dynamics": {
            "max_root_force_n": 0.0,
        },
        "duration_s": 1.814,
    }
    verdict = evaluate(receipt, horizon=Horizon.G3)
    assert verdict.is_physically_accepted is False
    failing_gates = {g.name: g for g in verdict.gates if g.status == GateStatus.FAILED}
    assert "friction_cone" in failing_gates
    assert failing_gates["friction_cone"].measured == pytest.approx(2.5)


def test_missing_root_history_fails_closed() -> None:
    """Missing root assistance history (delta_tau_root) in dynamics fails closed."""
    receipt: dict[str, Any] = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.030,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.040,
            "club_marker_rmse_m": 0.040,
            "pelvis_yaw_rmse_rad": 0.03,
        },
        "contact_audit": {
            "max_normal_force_n": 1500.0,
            "max_penetration_m": 0.002,
            "max_friction_ratio": 0.6,
        },
        "closure": {
            "max_closure_residual_m": 0.002,
        },
        "dynamics": {
            # Missing max_root_force_n or delta_tau_root
        },
        "duration_s": 1.814,
    }
    verdict = evaluate(receipt, horizon=Horizon.G3)
    assert verdict.is_physically_accepted is False
    missing_or_failed = {
        g.name: g
        for g in verdict.gates
        if g.status in (GateStatus.MISSING, GateStatus.FAILED)
    }
    assert "root_assistance" in missing_or_failed


def test_nonzero_root_assistance_fails_acceptance() -> None:
    """Nonzero phantom root force (e.g. 50 N) fails acceptance."""
    receipt: dict[str, Any] = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.030,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.040,
            "club_marker_rmse_m": 0.040,
            "pelvis_yaw_rmse_rad": 0.03,
        },
        "contact_audit": {
            "max_normal_force_n": 1500.0,
            "max_penetration_m": 0.002,
            "max_friction_ratio": 0.6,
        },
        "closure": {
            "max_closure_residual_m": 0.002,
        },
        "dynamics": {
            "max_root_force_n": 48.5,  # Phantom root assistance
        },
        "duration_s": 1.814,
    }
    verdict = evaluate(receipt, horizon=Horizon.G3)
    assert verdict.is_physically_accepted is False
    failing_gates = {g.name: g for g in verdict.gates if g.status == GateStatus.FAILED}
    assert "root_assistance" in failing_gates


def test_empty_marker_population_no_zero_success() -> None:
    """Empty valid marker population must yield NaN, not 0.0 mm success."""
    labels = ("Marker_1", "Marker_2", "LToeIn", "RToeIn")
    n_nodes = 20
    evaluator = SwingEvaluator(labels=labels)

    time_s = np.linspace(0.0, 0.2, n_nodes)
    pred_markers = np.ones((n_nodes, 4, 3)) * 0.1
    targ_markers = np.zeros((n_nodes, 4, 3))
    # All markers marked invalid (e.g. missing capture / occluded)
    valid = np.zeros((n_nodes, 4), dtype=bool)

    report = evaluator.evaluate(
        time_s=time_s,
        pred_markers=pred_markers,
        target_markers=targ_markers,
        valid=valid,
        sphere_bottom_z=np.ones((n_nodes, 6)) * 0.01,
        ground_height_m=0.0,
        closure_errors_m=np.zeros(n_nodes),
    )

    assert math.isnan(report.overall_rmse_mm)
    assert report.coverage_fraction == 0.0
    for seg_metric in report.segments.values():
        assert math.isnan(seg_metric.rmse_mm)
        assert seg_metric.coverage_fraction == 0.0


def test_closure_translation_vs_rotation_separation() -> None:
    """Weld closure must audit translation (m) and rotation (rad) separately."""
    closure = ClosureAudit(
        max_closure_translation_mm=3.2,
        mean_closure_translation_mm=1.1,
        max_closure_rotation_rad=0.12,  # > 0.05 rad threshold
        mean_closure_rotation_rad=0.04,
        worst_frame=15,
    )
    # Backward compatibility properties
    assert closure.max_closure_mm == pytest.approx(3.2)
    assert closure.mean_closure_mm == pytest.approx(1.1)

    receipt: dict[str, Any] = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.030,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.040,
            "club_marker_rmse_m": 0.040,
            "pelvis_yaw_rmse_rad": 0.03,
        },
        "contact_audit": {
            "max_normal_force_n": 1500.0,
            "max_penetration_m": 0.002,
            "max_friction_ratio": 0.6,
        },
        "closure": {
            "max_closure_translation_m": 0.0032,  # passes <= 0.005 m
            "max_closure_rotation_rad": 0.12,  # FAILS > 0.05 rad
        },
        "dynamics": {
            "max_root_force_n": 0.0,
        },
        "duration_s": 1.814,
    }
    verdict = evaluate(receipt, horizon=Horizon.G3)
    assert verdict.is_physically_accepted is False
    failing = {g.name: g for g in verdict.gates if g.status == GateStatus.FAILED}
    assert "closure_rotation_rad" in failing


def test_phase_event_usage() -> None:
    """Explicit t_events timestamps must define phase windows."""
    labels = ("Marker_1", "LToeIn")
    n_nodes = 100
    dt = 0.01
    time_s = np.arange(n_nodes) * dt  # 0.0 to 0.99 s

    t_events = {
        "address": 0.20,
        "top_of_backswing": 0.50,
        "impact": 0.70,
        "finish": 0.95,
    }

    evaluator = SwingEvaluator(labels=labels)
    pred_markers = np.zeros((n_nodes, 2, 3))
    targ_markers = np.zeros((n_nodes, 2, 3))
    valid = np.ones((n_nodes, 2), dtype=bool)

    report = evaluator.evaluate(
        time_s=time_s,
        pred_markers=pred_markers,
        target_markers=targ_markers,
        valid=valid,
        sphere_bottom_z=np.ones((n_nodes, 6)) * 0.01,
        ground_height_m=0.0,
        closure_errors_m=np.zeros(n_nodes),
        t_events=t_events,
    )

    phases = report.phases
    assert SwingPhase.ADDRESS.value in phases
    assert phases[SwingPhase.ADDRESS.value].end_time_s == pytest.approx(0.20)
    assert SwingPhase.BACKSWING.value in phases
    assert phases[SwingPhase.BACKSWING.value].end_time_s == pytest.approx(0.50)
    assert SwingPhase.DOWNSWING.value in phases
    assert phases[SwingPhase.DOWNSWING.value].end_time_s == pytest.approx(0.70)


def test_truncated_horizon_rejected() -> None:
    """A truncated candidate (0.85 s G1) evaluated under G3 must be rejected."""
    receipt: dict[str, Any] = {
        "shared_metrics": {
            "whole_marker_rmse_m": 0.020,
            "early_marker_rmse_m": 0.010,
            "terminal_marker_rmse_m": 0.030,
            "club_marker_rmse_m": 0.030,
            "pelvis_yaw_rmse_rad": 0.03,
        },
        "contact_audit": {
            "max_normal_force_n": 1500.0,
            "max_penetration_m": 0.002,
            "max_friction_ratio": 0.6,
        },
        "closure": {
            "max_closure_residual_m": 0.002,
        },
        "dynamics": {
            "max_root_force_n": 0.0,
        },
        "duration_s": 0.85,  # G1 duration, not G3 (full swing ~1.81 s)
    }
    verdict = evaluate(receipt, horizon=Horizon.G3)
    assert verdict.is_physically_accepted is False
    failing = {g.name: g for g in verdict.gates if g.status == GateStatus.FAILED}
    assert "horizon_duration_s" in failing


def test_44_to_41_coordinate_mismatch_fails() -> None:
    """Coordinate order mismatch between 44 and 41 coordinates raises ValueError."""
    coords_44 = (
        "TranslationInputX",
        "TranslationInputY",
        "TranslationInputZ",
        "HipInputX",
        "HipInputY",
        "HipInputZ",
        "SpineInputX",
        "SpineInputY",
        "TorsoInput",
        "LEInput",
        "LFInput",
        "LScapInputX",
        "LScapInputY",
        "LSInputX",
        "LSInputY",
        "LSInputZ",
        "LWInputX",
        "LWInputY",
        "REInput",
        "RFInput",
        "RScapInputX",
        "RScapInputY",
        "RSInputX",
        "RSInputY",
        "RSInputZ",
        "RWInputX",
        "RWInputY",
        "NeckInputX",
        "NeckInputY",
        "NeckInputZ",
        "hip_flexion_r",
        "hip_adduction_r",
        "hip_rotation_r",
        "knee_angle_r",
        "ankle_angle_r",
        "subtalar_angle_r",
        "mtp_angle_r",
        "hip_flexion_l",
        "hip_adduction_l",
        "hip_rotation_l",
        "knee_angle_l",
        "ankle_angle_l",
        "subtalar_angle_l",
        "mtp_angle_l",
    )
    coords_41 = coords_44[:41]  # missing last 3 coordinates

    with pytest.raises(ValueError, match="Coordinate dimension mismatch"):
        check_coordinate_mapping(coords_44, coords_41)


def test_candidate_package_roundtrip(tmp_path: Path) -> None:
    """CandidatePackage serializes and restores complete controls, root histories and metadata."""
    n_nodes = 10
    n_q = 44
    n_act = 38
    n_contacts = 6
    n_markers = 5

    pkg = CandidatePackage(
        time_s=np.linspace(0.0, 0.1, n_nodes),
        q=np.ones((n_nodes, n_q)),
        v=np.zeros((n_nodes, n_q)),
        a=np.zeros((n_nodes, n_q)),
        u=np.ones((n_nodes, n_act)) * 10.0,
        u_min_trail=np.ones((n_nodes, n_act)) * 2.0,
        u_hard_zero_trail=np.zeros((n_nodes, n_act)),
        delta_tau_root=np.zeros((n_nodes, 6)),
        ground_forces=np.ones((n_nodes, n_contacts * 3)) * 50.0,
        grip_wrenches=np.zeros((n_nodes, 6)),
        contact_modes=np.ones((n_nodes, n_contacts), dtype=bool),
        solver_status=np.ones(n_nodes, dtype=bool),
        equilibrium_residuals=np.zeros(n_nodes),
        coordinate_order=tuple(f"coord_{i}" for i in range(n_q)),
        actuated_indices=np.arange(6, n_q),
        handedness="right_handed",
        marker_labels=tuple(f"marker_{i}" for i in range(n_markers)),
        predicted_markers_m=np.zeros((n_nodes, n_markers, 3)),
        target_markers_m=np.zeros((n_nodes, n_markers, 3)),
        marker_valid=np.ones((n_nodes, n_markers), dtype=bool),
        interpolation_method="pchip",
        model_sha256="a" * 64,
        capture_sha256="b" * 64,
        schema_version="v2.0",
    )

    save_path = tmp_path / "candidate_test.npz"
    pkg.save(save_path)
    loaded = CandidatePackage.load(save_path)

    assert loaded.schema_version == "v2.0"
    assert loaded.time_s.shape == (n_nodes,)
    assert loaded.delta_tau_root.shape == (n_nodes, 6)
    assert np.allclose(loaded.u_min_trail, pkg.u_min_trail)
    assert np.allclose(loaded.delta_tau_root, 0.0)
    assert loaded.handedness == "right_handed"
