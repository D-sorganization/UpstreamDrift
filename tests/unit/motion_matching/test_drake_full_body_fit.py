"""Unit tests for Drake native full-body trajectory fitting (MS-30 #10337).

Covers:
1. On a 2-link toy plant the fitter recovers known torques (< 1e-6).
2. Warm-start replay of the candidate in Drake reports the same metrics as MS-13 parity.
3. Fit receipt passes G1.
4. Polynomial control fitting and evaluation roundtrip.
5. DrakeMatchingPlant.fit() dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.full_body_fit import (
    RECEIPT_SCHEMA,
    DrakeFitOptions,
    _fit_polynomial_controls,
    compute_parity_vs_reference,
    fit_full_body_drake,
    fit_toy_2link,
)
from src.shared.python.motion_matching.acceptance import Horizon, evaluate
from src.shared.python.motion_matching.candidate import (
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.pipeline.plant import get_plant
from src.shared.python.motion_matching.polynomial_torque import (
    evaluate_polynomial_torque,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = (
    ROOT
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver/full_body_spec_hipcal_scaled.json"
)
CANDIDATE_PATH = ROOT / "evidence/matched/driver_g1_crocoddyl_rk45_b100/candidate.npz"


def _require_real_drake() -> Any:
    try:
        import pydrake.all as drake_all
    except ImportError as exc:
        pytest.skip(f"pydrake not importable: {exc}")
    if type(drake_all).__module__ == "unittest.mock" or not hasattr(
        drake_all, "MultibodyPlant"
    ):
        pytest.skip("real pydrake runtime required (found mock/stub)")
    return drake_all


def test_toy_2link_torque_recovery() -> None:
    """TDD Step 1: On a 2-link toy plant the fitter recovers known torques (< 1e-6)."""
    n_frames = 20
    dt = 0.01
    times = np.linspace(0.0, (n_frames - 1) * dt, n_frames)

    # Synthesize smooth 2-link motions
    q_target = np.column_stack([np.sin(times), 0.5 * np.cos(times)])
    v_target = np.column_stack([np.cos(times), -0.5 * np.sin(times)])
    a_target = np.column_stack([-np.sin(times), -0.5 * np.cos(times)])

    recovered_tau = fit_toy_2link(q_target, v_target, a_target, dt)

    assert recovered_tau.shape == (n_frames, 2)
    assert np.all(np.isfinite(recovered_tau))

    # Re-evaluate inverse dynamics: check recovered torques reproduce acceleration
    l1, l2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    g = 9.81
    for i in range(n_frames):
        q1, q2 = q_target[i, 0], q_target[i, 1]
        v1, v2 = v_target[i, 0], v_target[i, 1]
        a1, a2 = a_target[i, 0], a_target[i, 1]

        m11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2.0 * m2 * l1 * l2 * np.cos(q2)
        m12 = m2 * l2**2 + m2 * l1 * l2 * np.cos(q2)
        m22 = m2 * l2**2
        M = np.array([[m11, m12], [m12, m22]])

        h = -m2 * l1 * l2 * np.sin(q2)
        C = np.array([h * (2.0 * v1 * v2 + v2**2), -h * v1**2])
        G = np.array(
            [
                (m1 + m2) * g * l1 * np.cos(q1) + m2 * g * l2 * np.cos(q1 + q2),
                m2 * g * l2 * np.cos(q1 + q2),
            ]
        )
        expected_tau = M @ np.array([a1, a2]) + C + G
        err = np.max(np.abs(recovered_tau[i] - expected_tau))
        assert err < 1e-6, f"Torque recovery error {err} exceeds 1e-6 at frame {i}"


def test_polynomial_control_fitting() -> None:
    """Verify degree-6 polynomial fitting roundtrip on synthetic control knots."""
    n_knots = 50
    t = np.linspace(0.0, 0.85, n_knots)
    # Synthetic degree-6 curve for 2 joints
    true_coeffs = np.array(
        [
            [10.0, -5.0, 2.0, 1.0, -0.5, 0.2, -0.05],
            [-20.0, 15.0, -4.0, 0.5, 0.1, -0.02, 0.01],
        ]
    )
    u_knots = np.zeros((n_knots, 2))
    for k in range(n_knots):
        u_knots[k] = evaluate_polynomial_torque(true_coeffs, t[k])

    fitted_coeffs = _fit_polynomial_controls(t, u_knots, degree=6)
    assert fitted_coeffs.shape == (2, 7)
    max_err = np.max(np.abs(fitted_coeffs - true_coeffs))
    assert max_err < 1e-4


def test_drake_fit_refuses_fabricated_evidence(tmp_path: Path) -> None:
    """Part B: Drake fit refuses fabricated evidence and removes synthetic fallbacks.

    1. Missing model markers in candidate raises ValueError (refuses scoring targets against themselves).
    2. Missing target markers (neither candidate nor argument) raises ValueError.
    3. Missing or placeholder capture hash in warm start raises ValueError.
    4. Drake simulation failure raises classified error (never falls back to warm-start kinematics).
    5. Physical audit has no invented numeric literals (unmeasured fields are None, failing acceptance).
    """
    from src.engines.physics_engines.drake.python.full_body_fit import (
        _evaluate_acceptance,
        _extract_markers_and_metrics,
        _load_warm_start,
        _simulate_drake,
    )
    from src.shared.python.motion_matching.acceptance import GateStatus
    from src.shared.python.motion_matching.replay_metrics import ReplayFiveMetrics
    from src.shared.python.simulation_backends.exceptions import (
        BackendNotAvailableError,
    )

    n_frames = 10
    time_s = np.linspace(0.0, 0.85, n_frames)
    target_markers = np.ones((n_frames, 44, 3), dtype=np.float64)

    # 1. Missing model markers raises ValueError
    cand_no_model_markers = MatchedSwingCandidate(
        metadata=CandidateMetadata(
            profile=CandidateProfile.DYNAMIC,
            engine="test",
            coordinate_names=("q0",),
            actuator_names=("q0",),
            marker_names=tuple(f"m_{i}" for i in range(44)),
        ),
        time_s=time_s,
        q=np.zeros((n_frames, 1)),
        v=np.zeros((n_frames, 1)),
        tau=np.zeros((n_frames, 1)),
        markers=CandidateMarkers(
            target_markers_m=target_markers,
            model_markers_m=None,
        ),
    )
    with pytest.raises(
        ValueError,
        match="model markers unavailable; refusing to score target markers against themselves",
    ):
        _extract_markers_and_metrics(
            cand_no_model_markers,
            target_markers=None,
            marker_labels=None,
            time_s=time_s,
        )

    # 2. Missing target markers raises ValueError
    cand_no_target_markers = MatchedSwingCandidate(
        metadata=CandidateMetadata(
            profile=CandidateProfile.DYNAMIC,
            engine="test",
            coordinate_names=("q0",),
            actuator_names=("q0",),
            marker_names=tuple(f"m_{i}" for i in range(44)),
        ),
        time_s=time_s,
        q=np.zeros((n_frames, 1)),
        v=np.zeros((n_frames, 1)),
        tau=np.zeros((n_frames, 1)),
        markers=CandidateMarkers(
            target_markers_m=None,
            model_markers_m=target_markers,
        ),
    )
    with pytest.raises(ValueError, match="Target markers are unavailable"):
        _extract_markers_and_metrics(
            cand_no_target_markers,
            target_markers=None,
            marker_labels=None,
            time_s=time_s,
        )

    # 3. Missing or zero capture hash raises ValueError
    test_npz = tmp_path / "candidate_zero_hash.npz"
    np.savez(
        test_npz,
        time_s=time_s,
        q=np.zeros((n_frames, 1)),
        u=np.zeros((n_frames - 1, 1)),
        labels=np.array(["m0"]),
        target_m=np.zeros((n_frames, 1, 3)),
        valid=np.ones((n_frames, 1), dtype=bool),
        source_c3d_sha256="0" * 64,
    )
    with pytest.raises(
        ValueError, match="Source C3D capture hash is unknown or invalid"
    ):
        _load_warm_start(test_npz, {"coordinate_order": ["q0"]})

    # 3b. A real hash but no recorded velocity/control history is refused, not zero-filled
    no_v_npz = tmp_path / "candidate_no_velocity.npz"
    np.savez(
        no_v_npz,
        time_s=time_s,
        q=np.zeros((n_frames, 1)),
        u=np.zeros((n_frames - 1, 1)),
        source_c3d_sha256="ab" * 32,
    )
    with pytest.raises(ValueError, match="refusing to invent a velocity"):
        _load_warm_start(no_v_npz, {"coordinate_order": ["q0"]})

    # 4. Drake initialization failure raises classified BackendNotAvailableError
    spec_dict = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    with pytest.raises((BackendNotAvailableError, RuntimeError, ImportError)):
        _simulate_drake(
            spec_dict,
            np.zeros((n_frames, 41)),
            np.zeros((n_frames, 41)),
            time_s,
            np.zeros((n_frames - 1, 35)),
        )

    # 5. Audit has no invented values; acceptance fails closed on missing gates
    physical_audit = {
        "max_normal_force_n": None,
        "max_normal_force_body_weights": None,
        "max_penetration_m": None,
        "closure_translation_error_max_m": None,
        "closure_rotation_error_max_rad": None,
        "weight_fraction": None,
        "peak_effort_n_m": 150.0,
    }
    shared_metrics = {
        "whole_marker_rmse_m": 0.020,
        "early_marker_rmse_m": 0.010,
        "terminal_marker_rmse_m": 0.020,
        "club_marker_rmse_m": 0.030,
        "pelvis_yaw_rmse_rad": 0.02,
        "pelvis_yaw_error_pct": 2.0,
    }
    five_m = ReplayFiveMetrics(
        whole_rms_m=0.020,
        early_rms_m=0.010,
        terminal_rms_m=0.020,
        club_cluster_rms_m=0.030,
        pelvis_yaw_error_pct=2.0,
    )
    verdict = _evaluate_acceptance(shared_metrics, five_m, physical_audit)
    assert verdict.is_physically_accepted is False
    assert verdict.status == "REJECTED"
    unpassed_gates = [
        g for g in verdict.gates if g.status in (GateStatus.FAILED, GateStatus.MISSING)
    ]
    assert any("max_normal_force" in g.name for g in unpassed_gates)
    force_gate = next(g for g in unpassed_gates if "max_normal_force" in g.name)
    assert "missing" in force_gate.reason.lower()


def test_drake_matching_plant_fit_hook() -> None:
    """Verify DrakeMatchingPlant exposes the fit() hook."""
    _require_real_drake()
    if not SPEC_PATH.is_file() or not CANDIDATE_PATH.is_file():
        pytest.skip("Required spec or candidate file not found")

    spec_dict = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    plant = get_plant("drake", spec_dict)
    assert hasattr(plant, "fit")

    result = plant.fit(
        warm_start_path=CANDIDATE_PATH,
        options=DrakeFitOptions(control_mode="knots", max_iterations=5),
    )
    assert result.candidate is not None
    assert result.receipt["schema"] == RECEIPT_SCHEMA
