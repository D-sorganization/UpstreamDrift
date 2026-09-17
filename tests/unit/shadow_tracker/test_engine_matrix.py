"""Unit tests for Shadow Tracker engine conformance matrix and release qualification (ST-12, #10135)."""

from __future__ import annotations

import pytest

from shared.python.shadow_tracker.contracts import (
    CandidateResult,
    FrameObservation,
    ReplayAudit,
)
from shared.python.shadow_tracker.engine_matrix import (
    EngineCapabilityMatrix,
    EngineQualificationResult,
    EngineReceipt,
    IndependentReplayAudit,
    PerformanceProfile,
    ReleaseQualificationReport,
    audit_engine_conformance,
    audit_full_release_gates,
    generate_release_evidence_inventory,
    get_shadow_tracker_scientific_registry,
    profile_shadow_tracker_performance,
    validate_cross_engine_contact_claims,
    verify_independent_replay,
)
from shared.python.shadow_tracker.evaluation import GateProfile

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------------------------------


def _valid_mujoco_receipt(
    *,
    model_sha256: str = "a" * 64,
    state_convention: str = "canonical_v2_quaternion",
    is_physically_accepted: bool = True,
    measured_closure_translation_m: float = 0.002,
    measured_closure_rotation_rad: float = 0.01,
) -> EngineReceipt:
    return EngineReceipt(
        engine_name="mujoco",
        engine_version="3.2.0",
        model_name="full_body_golfer_v1",
        model_sha256=model_sha256,
        state_convention=state_convention,
        contact_model_type="soft_complementarity",
        matlab_release=None,
        closure_translation_tolerance_m=0.005,
        closure_rotation_tolerance_rad=0.05,
        measured_closure_translation_m=measured_closure_translation_m,
        measured_closure_rotation_rad=measured_closure_rotation_rad,
        is_physically_accepted=is_physically_accepted,
    )


def _valid_simscape_receipt(
    *,
    matlab_release: str = "R2025b",
    model_sha256: str = "b" * 64,
) -> EngineReceipt:
    return EngineReceipt(
        engine_name="simscape",
        engine_version="2025.2",
        model_name="simscape_biomechanics_v2",
        model_sha256=model_sha256,
        state_convention="canonical_v2_quaternion",
        contact_model_type="penalty_viscoelastic",
        matlab_release=matlab_release,
        closure_translation_tolerance_m=0.005,
        closure_rotation_tolerance_rad=0.05,
        measured_closure_translation_m=0.001,
        measured_closure_rotation_rad=0.008,
        is_physically_accepted=True,
    )


def _dummy_replay_audit() -> ReplayAudit:
    return ReplayAudit(
        schema_version="shadow-tracker/replay-audit/1.0.0",
        candidate_id="cand-st12",
        reset_count=1,
        integrator_name="rk4",
        integrator_version="1.0",
        coverage_start_s=0.0,
        coverage_end_s=0.5,
        max_grip_translation_error_m=0.002,
        max_grip_rotation_error_rad=0.01,
        is_physically_accepted=True,
    )


def _dummy_candidate(
    *,
    trajectory: tuple[tuple[float, ...], ...] = ((0.0, 0.0, 0.0), (0.01, 0.0, 0.0)),
    diagnostics: dict[str, float] | None = None,
) -> CandidateResult:
    diag = diagnostics or {"mean_iou": 0.96, "club_error": 0.005, "joint_rmse_m": 0.02}
    return CandidateResult(
        schema_version="shadow-tracker/candidate-result/1.0.0",
        candidate_id="cand-st12",
        request_id="req-st12",
        initial_state=(0.0, 0.0, 0.0),
        trajectory=trajectory,
        diagnostics=diag,
        uncertainty_method="calibrated_posterior",
        replay_audit=_dummy_replay_audit(),
        is_accepted=True,
    )


def _dummy_observations(
    phases: tuple[str, ...] = (
        "address",
        "takeaway",
        "transition",
        "downswing",
        "impact",
        "follow_through",
    ),
) -> tuple[FrameObservation, ...]:
    obs_list: list[FrameObservation] = []
    for idx, phase in enumerate(phases):
        obs_list.append(
            FrameObservation(
                schema_version="shadow-tracker/frame-observation/1.0.0",
                shot_id="shot-001",
                camera_id="cam-001",
                frame_id=f"frame-{idx:03d}",
                pts_ticks=idx * 33,
                timebase_numerator=1,
                timebase_denominator=1000,
                physical_time_s=idx * 0.033,
                physical_time_reason=f"exact_sync_{phase}",
                body_mask_ref=f"mask-body-{idx:03d}",
                club_mask_ref=f"mask-club-{idx:03d}",
                valid_mask_ref=f"mask-valid-{idx:03d}",
                confidence_provenance="ground_truth",
                timing_mode="exact_container_pts",
                is_timing_exact=True,
                clock_evidence="hardware_sync",
                decoder_name="opencv",
            )
        )
    return tuple(obs_list)


# ---------------------------------------------------------------------------
# Test Cases: Engine Conformance & Receipts
# ---------------------------------------------------------------------------


def test_supported_engine_declaration_requires_receipt() -> None:
    """Supported engine declaration fails qualification if receipt is missing."""
    res = audit_engine_conformance(
        engine_name="mujoco",
        receipt=None,
        is_advertised=True,
    )
    assert not res.is_qualified
    assert res.status == "qualification_failed"
    assert "missing_engine_receipt" in res.failure_reasons


def test_unadvertised_engine_without_receipt_is_unsupported() -> None:
    """Unadvertised engine without receipt is documented as unsupported, not a blocking failure."""
    res = audit_engine_conformance(
        engine_name="drake",
        receipt=None,
        is_advertised=False,
    )
    assert not res.is_qualified
    assert res.status == "unsupported"
    assert "unadvertised_engine" in res.failure_reasons


def test_engine_conformance_fails_on_model_hash_mismatch() -> None:
    """Engine conformance fails if asset/model SHA-256 does not match required hash."""
    receipt = _valid_mujoco_receipt(model_sha256="wrong_hash" + "0" * 54)
    res = audit_engine_conformance(
        engine_name="mujoco",
        receipt=receipt,
        required_model_hash="expected_hash" + "1" * 51,
        is_advertised=True,
    )
    assert not res.is_qualified
    assert res.status == "qualification_failed"
    assert any("model_hash_mismatch" in r for r in res.failure_reasons)


def test_engine_conformance_fails_on_incompatible_state_convention() -> None:
    """Engine using scalar RPY coordinates cannot claim canonical quaternion qualification."""
    receipt = _valid_mujoco_receipt(state_convention="scalar_rpy")
    res = audit_engine_conformance(
        engine_name="mujoco",
        receipt=receipt,
        required_state_convention="canonical_v2_quaternion",
        is_advertised=True,
    )
    assert not res.is_qualified
    assert res.status == "qualification_failed"
    assert any("incompatible_state_convention" in r for r in res.failure_reasons)


def test_simscape_requires_explicit_r2025b_matlab_release() -> None:
    """Simscape advertised engine fails qualification unless explicit R2025b is proven."""
    # Wrong release fails
    wrong_receipt = _valid_simscape_receipt(matlab_release="R2024b")
    res_wrong = audit_engine_conformance(
        engine_name="simscape",
        receipt=wrong_receipt,
        is_advertised=True,
    )
    assert not res_wrong.is_qualified
    assert res_wrong.status == "qualification_failed"
    assert any(
        "simscape_requires_matlab_R2025b" in r for r in res_wrong.failure_reasons
    )

    # Valid R2025b passes
    valid_receipt = _valid_simscape_receipt(matlab_release="R2025b")
    res_valid = audit_engine_conformance(
        engine_name="simscape",
        receipt=valid_receipt,
        is_advertised=True,
    )
    assert res_valid.is_qualified
    assert res_valid.status == "advertised_and_qualified"


def test_cross_engine_contact_claim_rejection() -> None:
    """Reject claims of identical contact outcomes across materially different contact laws."""
    receipt_mujoco = _valid_mujoco_receipt()
    receipt_simscape = _valid_simscape_receipt()

    with pytest.raises(
        ValueError,
        match="Cannot claim identical contact results across differing contact models",
    ):
        validate_cross_engine_contact_claims(receipt_mujoco, receipt_simscape)


# ---------------------------------------------------------------------------
# Test Cases: Independent Replay & Performance Profiling
# ---------------------------------------------------------------------------


def test_independent_replay_verification_pass() -> None:
    """Independent replay within tight numerical tolerance passes audit."""
    cand = _dummy_candidate(trajectory=((0.0, 0.0, 0.0), (0.01, 0.0, 0.0)))
    replay_traj = ((0.0, 0.0, 0.0), (0.0100001, 0.0, 0.0))
    audit = verify_independent_replay(
        candidate=cand,
        replay_trajectory=replay_traj,
        engine_name="mujoco",
        tolerance=1e-5,
    )
    assert audit.is_replay_converged
    assert audit.max_state_discrepancy <= 1e-5


def test_independent_replay_verification_detects_divergence() -> None:
    """Independent replay with divergent states fails convergence check."""
    cand = _dummy_candidate(trajectory=((0.0, 0.0, 0.0), (0.01, 0.0, 0.0)))
    replay_traj = ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0))  # 4 cm divergence
    audit = verify_independent_replay(
        candidate=cand,
        replay_trajectory=replay_traj,
        engine_name="mujoco",
        tolerance=1e-5,
    )
    assert not audit.is_replay_converged
    assert audit.max_state_discrepancy > 1e-5


def test_performance_profiling_and_budget() -> None:
    """Performance profiling computes FPS and flags budget exceedance."""
    prof = profile_shadow_tracker_performance(
        duration_seconds=10.0,
        frames_processed=300,
        peak_memory_mb=256.0,
        phase_latencies_ms={"address": 10.0, "impact": 15.0},
        max_budget_seconds=60.0,
    )
    assert prof.fps == 30.0
    assert not prof.is_budget_exceeded

    # Budget exceeded
    prof_slow = profile_shadow_tracker_performance(
        duration_seconds=120.0,
        frames_processed=100,
        peak_memory_mb=512.0,
        phase_latencies_ms={"impact": 500.0},
        max_budget_seconds=60.0,
    )
    assert prof_slow.is_budget_exceeded


# ---------------------------------------------------------------------------
# Test Cases: Full G0–G7 Release Profile & Swing Phase Coverage
# ---------------------------------------------------------------------------


def test_full_release_gates_pass_on_complete_evidence() -> None:
    """Release qualification succeeds when G0-G7 pass and all swing phases are covered."""
    cand = _dummy_candidate()
    obs = _dummy_observations()
    profile = GateProfile.default_development_profile()

    mujoco_res = audit_engine_conformance(
        engine_name="mujoco",
        receipt=_valid_mujoco_receipt(),
        is_advertised=True,
    )
    drake_res = audit_engine_conformance(
        engine_name="drake",
        receipt=None,
        is_advertised=False,
    )
    matrix = EngineCapabilityMatrix(
        profiles={"mujoco": mujoco_res, "drake": drake_res},
        advertised_engines=("mujoco",),
        unsupported_engines=("drake",),
    )

    report = audit_full_release_gates(
        candidate=cand,
        observations=obs,
        engine_matrix=matrix,
        profile=profile,
    )

    assert report.is_release_qualified
    assert len(report.gate_statuses) == 8  # G0 to G7
    assert all(g.passed for g in report.gate_statuses)
    assert len(report.blocking_reasons) == 0


def test_release_qualification_fails_if_swing_phase_missing() -> None:
    """Release qualification fails if any key swing phase is missing from observations."""
    cand = _dummy_candidate()
    # Missing impact and follow_through
    obs = _dummy_observations(phases=("address", "takeaway", "transition"))
    profile = GateProfile.default_development_profile()

    mujoco_res = audit_engine_conformance(
        engine_name="mujoco",
        receipt=_valid_mujoco_receipt(),
        is_advertised=True,
    )
    matrix = EngineCapabilityMatrix(
        profiles={"mujoco": mujoco_res},
        advertised_engines=("mujoco",),
    )

    report = audit_full_release_gates(
        candidate=cand,
        observations=obs,
        engine_matrix=matrix,
        profile=profile,
    )

    assert not report.is_release_qualified
    assert any("missing_swing_phases" in b for b in report.blocking_reasons)


def test_unadvertised_engine_does_not_block_release() -> None:
    """Unadvertised engines documented as unsupported do not block first-engine release."""
    cand = _dummy_candidate()
    obs = _dummy_observations()
    profile = GateProfile.default_development_profile()

    mujoco_res = audit_engine_conformance(
        engine_name="mujoco",
        receipt=_valid_mujoco_receipt(),
        is_advertised=True,
    )
    drake_res = audit_engine_conformance(
        engine_name="drake",
        receipt=None,
        is_advertised=False,
    )
    simscape_res = audit_engine_conformance(
        engine_name="simscape",
        receipt=None,
        is_advertised=False,
    )
    matrix = EngineCapabilityMatrix(
        profiles={"mujoco": mujoco_res, "drake": drake_res, "simscape": simscape_res},
        advertised_engines=("mujoco",),
        unsupported_engines=("drake", "simscape"),
    )

    report = audit_full_release_gates(
        candidate=cand,
        observations=obs,
        engine_matrix=matrix,
        profile=profile,
    )

    assert report.is_release_qualified
    assert "drake" in report.engine_matrix.unsupported_engines
    assert "simscape" in report.engine_matrix.unsupported_engines


# ---------------------------------------------------------------------------
# Test Cases: Scientific Registry & Evidence Inventory
# ---------------------------------------------------------------------------


def test_scientific_registry_entries() -> None:
    """Scientific registry exposes non-empty, valid calculation entries."""
    registry = get_shadow_tracker_scientific_registry()
    assert len(registry) >= 4
    ids = {entry.registry_id for entry in registry}
    assert "ST-CALC-PROJECTION" in ids
    assert "ST-CALC-FORWARD-DYNAMICS" in ids
    assert "ST-CALC-CONTROL-FITTING" in ids
    assert "ST-CALC-ENGINE-CONFORMANCE" in ids


def test_release_evidence_inventory_generation() -> None:
    """Deterministic release evidence inventory is generated with verifiable SHA-256."""
    cand = _dummy_candidate()
    obs = _dummy_observations()
    profile = GateProfile.default_development_profile()

    mujoco_res = audit_engine_conformance(
        engine_name="mujoco",
        receipt=_valid_mujoco_receipt(),
        is_advertised=True,
    )
    matrix = EngineCapabilityMatrix(
        profiles={"mujoco": mujoco_res},
        advertised_engines=("mujoco",),
    )

    report = audit_full_release_gates(
        candidate=cand,
        observations=obs,
        engine_matrix=matrix,
        profile=profile,
    )

    inventory = generate_release_evidence_inventory(report)
    assert inventory["schema_version"] == "shadow-tracker/release-evidence/1.0.0"
    assert inventory["is_release_qualified"] is True
    assert "evidence_digest_sha256" in inventory
    assert len(inventory["evidence_digest_sha256"]) == 64
