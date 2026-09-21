"""Unit tests for PF-10: Matching strategy contracts and stage-separated qualification (#10440).

Verifies:
1. Stage-separated qualification tracking across all 6 stages:
   - model_available, kinematic_fit, force_feasible, replay_accepted, runtime_budget_met, muscle_qualified.
2. Invariant: Never marks a candidate 'accepted' merely because it animates or kinematic fit passed.
3. MatchingStrategyContract schema, presets, controller specs, and serialization round-trip.
4. CandidateStrategyPackage serialization/deserialization to .npz without pickle.
5. Name-permuted coordinate remapping with fail-closed missing coordinate detection.
6. StrategyComparisonService: torque profiles, tracking/closure errors, stage matrix.
7. Engine capability evaluation: missing SDK/engine invalidates supported state.
8. Complete driver & 7-iron contract verification across all six engines:
   - mujoco, pinocchio, drake, opensim, simscape, myosuite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.matching_strategy import (
    ALL_ENGINES,
    STRATEGY_SCHEMA_VERSION,
    CandidateStrategyPackage,
    ContactReactionHistory,
    ControllerSpecification,
    MatchingStrategyContract,
    QualificationStage,
    StageQualificationMatrix,
    StageReport,
    StageState,
    StrategyComparisonService,
    StrategyPreset,
    create_sample_strategy_package,
)

pytestmark = pytest.mark.unit


def _create_test_candidate(
    n_frames: int = 5,
    coord_names: tuple[str, ...] = ("joint_0", "joint_1", "joint_2"),
    engine: str = "mujoco",
) -> MatchedSwingCandidate:
    nq = len(coord_names)
    time_s = np.linspace(0.0, 0.2, n_frames)
    q = np.ones((n_frames, nq), dtype=np.float64) * 0.1
    v = np.ones((n_frames, nq), dtype=np.float64) * 0.2
    tau = np.ones((n_frames, nq), dtype=np.float64) * 5.0
    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.DYNAMIC,
        engine=engine,
        model_name=f"test_{engine}",
        model_sha256="0" * 64,
        coordinate_names=coord_names,
        velocity_names=coord_names,
        actuator_names=coord_names,
    )
    return MatchedSwingCandidate(
        metadata=meta,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
    )


def test_qualification_stages_ordered_and_independent() -> None:
    """StageQualificationMatrix tracks each stage independently."""
    matrix = StageQualificationMatrix()
    assert len(matrix.stages) == 6
    assert not matrix.is_stage_passed(QualificationStage.MODEL_AVAILABLE)
    assert matrix.overall_verdict() == "not_evaluated"

    # Pass model available
    matrix.record_stage(
        QualificationStage.MODEL_AVAILABLE,
        StageState.PASSED,
        reason="Model file and SDK verified",
    )
    assert matrix.is_stage_passed(QualificationStage.MODEL_AVAILABLE)
    assert matrix.is_qualified_through(QualificationStage.MODEL_AVAILABLE)
    assert matrix.overall_verdict() == "provisional"

    # Invariant: Kinematic fit alone must NEVER yield 'accepted'
    matrix.record_stage(
        QualificationStage.KINEMATIC_FIT,
        StageState.PASSED,
        metrics={"marker_rmse_mm": 18.2},
    )
    assert matrix.is_qualified_through(QualificationStage.KINEMATIC_FIT)
    assert matrix.overall_verdict() == "provisional"

    # Record failure at force feasibility
    matrix.record_stage(
        QualificationStage.FORCE_FEASIBLE,
        StageState.FAILED,
        reason="Equilibrium residual 0.12 exceeded 1e-3",
        metrics={"max_residual": 0.12},
    )
    assert not matrix.is_stage_passed(QualificationStage.FORCE_FEASIBLE)
    assert not matrix.is_qualified_through(QualificationStage.FORCE_FEASIBLE)
    assert matrix.failing_stage() == QualificationStage.FORCE_FEASIBLE
    assert matrix.overall_verdict() == "rejected"


def test_qualification_verdict_only_accepted_after_replay() -> None:
    """Only candidates passing replay_accepted and earlier stages are accepted."""
    matrix = StageQualificationMatrix()
    for stg in (
        QualificationStage.MODEL_AVAILABLE,
        QualificationStage.KINEMATIC_FIT,
        QualificationStage.FORCE_FEASIBLE,
        QualificationStage.REPLAY_ACCEPTED,
    ):
        matrix.record_stage(stg, StageState.PASSED)

    assert matrix.is_qualified_through(QualificationStage.REPLAY_ACCEPTED)
    assert matrix.overall_verdict() == "accepted"


def test_strategy_contract_serialization_roundtrip() -> None:
    """MatchingStrategyContract serializes and deserializes cleanly without data loss."""
    matrix = StageQualificationMatrix()
    matrix.record_stage(
        QualificationStage.MODEL_AVAILABLE,
        StageState.PASSED,
        reason="Local model exists",
    )
    matrix.record_stage(
        QualificationStage.KINEMATIC_FIT,
        StageState.PASSED,
        metrics={"marker_rms_mm": 24.5},
    )

    ctrl = ControllerSpecification(
        controller_type="feedforward_torque",
        control_channels=("joint_0", "joint_1"),
        gains={"kp": 100.0, "kd": 10.0},
    )

    contract = MatchingStrategyContract(
        strategy_id="strat-pinocchio-driver-001",
        engine="pinocchio",
        model_id="full_body_spec_v1",
        model_sha256="abc" * 21 + "a",
        capture_id="driver",
        strategy_preset=StrategyPreset.CROCODDYL_DDP,
        frame_convention="z_up_y_forward",
        coordinate_mapping={"joint_0": 0, "joint_1": 1},
        actuator_mapping={"joint_0": 0, "joint_1": 1},
        qualification=matrix,
        controller=ctrl,
        runtime_budget_ms=5000.0,
        measured_runtime_ms=3210.5,
        solver_hyperparameters={"max_iter": 100, "tol": 1e-4},
    )

    data = contract.to_dict()
    restored = MatchingStrategyContract.from_dict(data)

    assert restored.schema_version == STRATEGY_SCHEMA_VERSION
    assert restored.strategy_id == contract.strategy_id
    assert restored.engine == "pinocchio"
    assert restored.strategy_preset == StrategyPreset.CROCODDYL_DDP
    assert restored.qualification.is_stage_passed(QualificationStage.KINEMATIC_FIT)
    assert restored.controller is not None
    assert restored.controller.controller_type == "feedforward_torque"
    assert restored.measured_runtime_ms == 3210.5


def test_candidate_strategy_package_save_and_load(tmp_path: Path) -> None:
    """CandidateStrategyPackage saves to .npz without pickle and reloads losslessly."""
    cand = _create_test_candidate(n_frames=6, coord_names=("q0", "q1", "q2"))
    a = np.ones((6, 3), dtype=np.float64) * 0.05
    reactions = ContactReactionHistory(
        contact_names=("heel_l", "toe_l"),
        reactions=np.ones((6, 6), dtype=np.float64) * 120.0,
        closure_wrench=np.zeros((6, 6), dtype=np.float64),
        root_wrench_residual=np.zeros((6, 6), dtype=np.float64),
    )
    contract = MatchingStrategyContract(
        strategy_id="strat-drake-01",
        engine="drake",
        model_id="drake_golf",
        model_sha256="1" * 64,
        capture_id="driver",
        strategy_preset=StrategyPreset.DRAKE_QP,
        coordinate_mapping={"q0": 0, "q1": 1, "q2": 2},
    )

    package = CandidateStrategyPackage(
        candidate=cand,
        strategy=contract,
        a=a,
        reactions=reactions,
    )

    target_file = tmp_path / "package.npz"
    package.save_package(target_file)
    assert target_file.is_file()

    # Load package back
    loaded = CandidateStrategyPackage.load_package(target_file)
    assert loaded.strategy.strategy_id == "strat-drake-01"
    assert loaded.strategy.engine == "drake"
    np.testing.assert_allclose(loaded.candidate.q, cand.q)
    assert loaded.a is not None
    np.testing.assert_allclose(loaded.a, a)
    assert loaded.reactions is not None
    assert loaded.reactions.contact_names == ("heel_l", "toe_l")
    assert loaded.reactions.reactions is not None
    assert reactions.reactions is not None
    np.testing.assert_allclose(loaded.reactions.reactions, reactions.reactions)


def test_name_permuted_coordinate_remapping() -> None:
    """CandidateStrategyPackage remaps columns cleanly when coordinate order is permuted."""
    original_names = ("pelvis_tz", "lumbar_pitch", "shoulder_r")
    cand = _create_test_candidate(n_frames=4, coord_names=original_names)
    a = np.array([[1.0, 2.0, 3.0]] * 4)

    package = CandidateStrategyPackage(
        candidate=cand,
        strategy=MatchingStrategyContract(
            strategy_id="strat-perm",
            engine="mujoco",
            model_id="model",
            model_sha256="2" * 64,
            capture_id="driver",
            strategy_preset=StrategyPreset.MINIMUM_EFFORT,
            coordinate_mapping={name: idx for idx, name in enumerate(original_names)},
        ),
        a=a,
    )

    # Permute order
    permuted_order = ("shoulder_r", "pelvis_tz", "lumbar_pitch")
    remapped = package.remap_coordinates(permuted_order)

    assert remapped.candidate.metadata.coordinate_names == permuted_order
    # Original a was [1, 2, 3]; permuted indices [2, 0, 1] -> [3, 1, 2]
    expected_a = np.array([[3.0, 1.0, 2.0]] * 4)
    assert remapped.a is not None
    np.testing.assert_allclose(remapped.a, expected_a)

    # Missing coordinate must fail closed
    with pytest.raises(KeyError, match="Missing coordinate 'nonexistent'"):
        package.remap_coordinates(("shoulder_r", "nonexistent"))


def test_strategy_comparison_service() -> None:
    """StrategyComparisonService compares multiple packages across engines & metrics."""
    p1 = create_sample_strategy_package(
        engine="mujoco",
        capture="driver",
        preset=StrategyPreset.MINIMUM_EFFORT,
        stage_up_to=QualificationStage.REPLAY_ACCEPTED,
    )
    p2 = create_sample_strategy_package(
        engine="pinocchio",
        capture="driver",
        preset=StrategyPreset.CROCODDYL_DDP,
        stage_up_to=QualificationStage.FORCE_FEASIBLE,
    )

    service = StrategyComparisonService()
    comparison = service.compare_strategies([p1, p2])

    assert "comparison_table" in comparison
    table = comparison["comparison_table"]
    assert len(table) == 2
    assert table[0]["engine"] == "mujoco"
    assert table[0]["verdict"] == "accepted"
    assert table[1]["engine"] == "pinocchio"
    assert table[1]["verdict"] == "provisional"

    # Torque profiles extraction
    torque_data = service.extract_torque_profiles([p1, p2])
    assert "mujoco" in torque_data
    assert "peak_effort_nm" in torque_data["mujoco"]

    # Tracking and closure errors extraction
    err_data = service.extract_tracking_and_closure_errors([p1, p2])
    assert "mujoco" in err_data
    assert "whole_marker_rmse_mm" in err_data["mujoco"]


def test_missing_sdk_invalidates_supported_state() -> None:
    """evaluate_engine_capabilities invalidates supported state if engine runtime is absent."""
    service = StrategyComparisonService()
    # Test evaluation with simulated missing SDK
    report = service.evaluate_engine_capabilities(
        engine="pinocchio",
        override_available=False,
    )
    assert not report["supported"]
    assert report["qualification_stage_status"] == StageState.FAILED.value
    assert "Engine SDK unavailable" in report["reason"]


def test_all_six_engines_and_dual_club_coverage() -> None:
    """Verify contracts and packages generate and validate across all six engines and dual clubs."""
    assert len(ALL_ENGINES) == 6
    expected_engines = {
        "mujoco",
        "pinocchio",
        "drake",
        "opensim",
        "simscape",
        "myosuite",
    }
    assert set(ALL_ENGINES) == expected_engines

    service = StrategyComparisonService()
    packages: list[CandidateStrategyPackage] = []

    for eng in ALL_ENGINES:
        for club in ("driver", "7-iron"):
            pkg = create_sample_strategy_package(
                engine=eng,
                capture=club,
                preset=StrategyPreset.MINIMUM_EFFORT,
                stage_up_to=QualificationStage.KINEMATIC_FIT,
            )
            assert pkg.strategy.engine == eng
            assert pkg.strategy.capture_id == club
            assert pkg.candidate.n_frames >= 2
            packages.append(pkg)

    # 6 engines * 2 clubs = 12 packages
    assert len(packages) == 12
    comparison = service.compare_strategies(packages)
    assert len(comparison["comparison_table"]) == 12
