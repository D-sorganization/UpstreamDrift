"""MS-111 Pinocchio G2/G3 continuation and independent replay contracts (#10385).

Software-contract layer only. Native ControlTower desk qualification remains
blocked until MS-107 produces an accepted same-integrator G1 and MS-100
receipts pass; these tests never invent native success.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.acceptance import Horizon
from src.shared.python.motion_matching.pinocchio_g2_g3 import (
    SCHEMA_VERSION,
    ArmaturePlantDeclaration,
    ClubKind,
    ContinuationSchedule,
    FailedContinuationEvidence,
    IntegratorConfig,
    OpenLoopFeedRequest,
    QualificationClaim,
    RobustnessCheckKind,
    build_continuation_schedule,
    evaluate_integrator_parity,
    export_independent_replay_package,
    import_independent_replay_package,
    ms111_contract_status,
    propagate_armature_to_replay,
    record_failed_continuation_stage,
    required_robustness_checks,
    validate_open_loop_control_feed,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "development"
    / "matched_swing_program"
    / "evidence"
    / "ms111"
    / "continuation_contract_status.json"
)


def test_schema_version_is_stable() -> None:
    assert SCHEMA_VERSION == "pinocchio-g2-g3-continuation/1.0.0"


def test_driver_g2_schedule_extends_past_g1_through_impact() -> None:
    schedule = build_continuation_schedule(ClubKind.DRIVER, Horizon.G2)
    assert isinstance(schedule, ContinuationSchedule)
    assert schedule.club == ClubKind.DRIVER
    assert schedule.target_horizon == Horizon.G2
    assert schedule.g1_end_s == pytest.approx(0.85)
    assert schedule.target_end_s == pytest.approx(1.20)
    assert schedule.stage_ends_s[0] > 0.85
    assert schedule.stage_ends_s[-1] == pytest.approx(1.20)
    assert schedule.declared_armature_kg_m2 == pytest.approx(5e-3)
    assert schedule.node_integrator == "rk45"
    assert schedule.replay_integrator == "rk45"
    assert schedule.rk45_rtol == pytest.approx(1e-6)
    assert schedule.claims_native_success is False


def test_iron_g3_schedule_uses_iron_capture_duration() -> None:
    schedule = build_continuation_schedule(ClubKind.IRON, Horizon.G3)
    assert schedule.target_end_s == pytest.approx(1.827, abs=1e-3)
    assert schedule.stage_ends_s[-1] == pytest.approx(schedule.target_end_s)
    assert all(t > 0.85 for t in schedule.stage_ends_s)
    assert schedule.document_id == "anthro_iron"
    assert schedule.claims_native_success is False


def test_driver_g3_schedule_uses_driver_capture_duration() -> None:
    schedule = build_continuation_schedule(ClubKind.DRIVER, Horizon.G3)
    assert schedule.target_end_s == pytest.approx(1.814, abs=1e-3)
    assert schedule.document_id == "anthro_driver"


def test_g1_target_is_rejected_for_ms111_continuation() -> None:
    with pytest.raises(ValueError, match="G2 or G3"):
        build_continuation_schedule(ClubKind.DRIVER, Horizon.G1)


def test_integrator_parity_rejects_mismatched_solve_and_replay() -> None:
    solve = IntegratorConfig(name="implicit_euler", rtol=1e-4, fixed_step=True)
    replay = IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False)
    verdict = evaluate_integrator_parity(solve, replay)
    assert verdict.accepted is False
    assert "integrator" in verdict.reason.lower()
    assert verdict.claims_native_success is False


def test_integrator_parity_accepts_matched_rk45() -> None:
    cfg = IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False)
    verdict = evaluate_integrator_parity(cfg, cfg)
    assert verdict.accepted is True
    assert verdict.reason == ""


def test_integrator_parity_rejects_rtol_mismatch() -> None:
    solve = IntegratorConfig(name="rk45", rtol=1e-4, fixed_step=False)
    replay = IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False)
    verdict = evaluate_integrator_parity(solve, replay)
    assert verdict.accepted is False
    assert "rtol" in verdict.reason.lower()


def test_armature_propagates_to_equivalent_model_replay() -> None:
    declaration = propagate_armature_to_replay(
        armature_kg_m2=5e-3,
        actuated_dof_count=38,
        model_sha256="abc123",
        source_engine="pinocchio",
        target_engine="mujoco",
    )
    assert isinstance(declaration, ArmaturePlantDeclaration)
    assert declaration.armature_kg_m2 == pytest.approx(5e-3)
    assert declaration.actuated_dof_count == 38
    assert declaration.source_engine == "pinocchio"
    assert declaration.target_engine == "mujoco"
    assert declaration.model_sha256 == "abc123"
    assert declaration.equivalent_model_required is True
    payload = declaration.as_dict()
    assert payload["armature_kg_m2"] == pytest.approx(5e-3)


def test_negative_armature_fails_closed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        propagate_armature_to_replay(
            armature_kg_m2=-1e-3,
            actuated_dof_count=1,
            model_sha256="x",
            source_engine="pinocchio",
            target_engine="drake",
        )


def test_failed_continuation_evidence_is_preserved() -> None:
    evidence = record_failed_continuation_stage(
        club=ClubKind.DRIVER,
        stage_end_s=1.20,
        reason="open-loop drift exceeded G2 whole gate",
        solver_cost=145.3,
        rollout_whole_rmse_m=0.340,
        replay_whole_rmse_m=0.340,
        integrator=IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False),
    )
    assert isinstance(evidence, FailedContinuationEvidence)
    assert evidence.preserved is True
    assert evidence.accepted is False
    assert evidence.claims_native_success is False
    assert evidence.stage_end_s == pytest.approx(1.20)
    dumped = evidence.as_dict()
    assert dumped["accepted"] is False
    assert "drift" in dumped["reason"]


def test_open_loop_feed_rejects_per_frame_pose_prescription() -> None:
    n, nv, nu = 5, 4, 3
    request = OpenLoopFeedRequest(
        q0=np.zeros(nv),
        v0=np.zeros(nv),
        controls_u=np.zeros((n - 1, nu)),
        prescribed_poses_q=np.zeros((n, nv)),
        timestamps_s=np.linspace(0.0, 0.1, n),
    )
    verdict = validate_open_loop_control_feed(request)
    assert verdict.accepted is False
    assert "pose" in verdict.reason.lower()


def test_open_loop_feed_accepts_q0_v0_and_controls_only() -> None:
    n, nv, nu = 5, 4, 3
    request = OpenLoopFeedRequest(
        q0=np.zeros(nv),
        v0=np.zeros(nv),
        controls_u=np.zeros((n - 1, nu)),
        prescribed_poses_q=None,
        timestamps_s=np.linspace(0.0, 0.1, n),
    )
    verdict = validate_open_loop_control_feed(request)
    assert verdict.accepted is True
    assert verdict.used_measured_state_reset is False


def test_robustness_checks_cover_driver_and_iron() -> None:
    for club in (ClubKind.DRIVER, ClubKind.IRON):
        checks = required_robustness_checks(club)
        kinds = {c.kind for c in checks}
        assert RobustnessCheckKind.TIMESTEP_REFINEMENT in kinds
        assert RobustnessCheckKind.CONTACT_TRANSITION in kinds
        assert RobustnessCheckKind.INITIAL_STATE_PERTURBATION in kinds
        assert all(c.club == club for c in checks)
        assert all(c.required_for_g2_g3 is True for c in checks)
        assert all(c.claims_native_success is False for c in checks)


def test_independent_replay_package_round_trip(tmp_path: Path) -> None:
    schedule = build_continuation_schedule(ClubKind.DRIVER, Horizon.G2)
    armature = propagate_armature_to_replay(
        armature_kg_m2=schedule.declared_armature_kg_m2,
        actuated_dof_count=38,
        model_sha256="modeldeadbeef",
        source_engine="pinocchio",
        target_engine="pinocchio",
    )
    package_dir = export_independent_replay_package(
        tmp_path / "pkg",
        schedule=schedule,
        armature=armature,
        candidate_sha256="canddeadbeef",
        controls_sha256="ctrldeadbeef",
        q0_sha256="q0deadbeef",
        v0_sha256="v0deadbeef",
    )
    loaded = import_independent_replay_package(package_dir)
    assert loaded.schedule.target_horizon == Horizon.G2
    assert loaded.armature.armature_kg_m2 == pytest.approx(5e-3)
    assert loaded.candidate_sha256 == "canddeadbeef"
    assert loaded.claims_native_success is False
    assert loaded.qualification == QualificationClaim.CONTRACT_READY
    # save/reopen identity
    reopened = import_independent_replay_package(package_dir)
    assert reopened.package_sha256 == loaded.package_sha256


def test_corrupt_replay_package_fails_closed(tmp_path: Path) -> None:
    package_dir = tmp_path / "bad"
    package_dir.mkdir()
    (package_dir / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        import_independent_replay_package(package_dir)


def test_ms111_status_is_honest_about_native_qualification() -> None:
    status = ms111_contract_status()
    assert status["schema_version"] == SCHEMA_VERSION
    assert status["issue"] == 10385
    assert status["milestone"] == "MS-111"
    assert status["qualification"] == QualificationClaim.AWAITING_NATIVE_EVIDENCE.value
    assert status["claims_native_success"] is False
    assert status["depends_on"] == ["MS-107", "MS-100", "MS-102"]
    assert "driver" in status["clubs"]
    assert "iron" in status["clubs"]
    assert EVIDENCE.is_file()
    on_disk = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert on_disk["claims_native_success"] is False
    assert on_disk["qualification"] == status["qualification"]


def test_fitter_helper_exposes_ms111_schedule() -> None:
    from src.engines.physics_engines.pinocchio.python.full_body_fit import (
        SolverSettings,
        assert_ms111_integrator_parity,
        ms111_continuation_s,
    )

    stages = ms111_continuation_s("driver", "G2")
    assert stages[-1] == pytest.approx(1.20)
    assert_ms111_integrator_parity(
        SolverSettings(node_integrator="rk45", replay_integrator="rk45", rk45_rtol=1e-6)
    )
    with pytest.raises(Exception, match="integrator|rtol"):
        assert_ms111_integrator_parity(
            SolverSettings(
                node_integrator="implicit_euler",
                replay_integrator="rk45",
                rk45_rtol=1e-6,
            )
        )
