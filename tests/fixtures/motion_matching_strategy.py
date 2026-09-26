"""Test fixture builders for motion matching strategy packages (#10960 P1-10).

This module contains test-only candidate strategy package generators, isolated
from production exports to ensure unmeasured/fabricated stage states are never
emitted from runtime motion matching APIs.
"""

from __future__ import annotations

import hashlib

import numpy as np

from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.matching_strategy import (
    CandidateStrategyPackage,
    ContactReactionHistory,
    MatchingStrategyContract,
    QualificationStage,
    StageQualificationMatrix,
    StageState,
    StrategyPreset,
)

__all__ = ["create_sample_strategy_package"]


def create_sample_strategy_package(
    engine: str,
    capture: str = "driver",
    preset: StrategyPreset | str = StrategyPreset.MINIMUM_EFFORT,
    stage_up_to: QualificationStage | None = None,
    stage_state: StageState = StageState.PASSED,
    n_frames: int = 10,
    nv: int = 41,
) -> CandidateStrategyPackage:
    """Create a sample strategy package for test fixtures.

    Preconditions:
        engine is a non-empty string.
        n_frames >= 2.
        nv >= 7.
    """
    if not engine:
        raise ValueError("engine must be a non-empty string")
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2, got {n_frames}")
    if nv < 7:
        raise ValueError(f"nv must be >= 7, got {nv}")

    coord_names = tuple(f"coord_{i}" for i in range(nv))
    time_s = np.linspace(0.0, 0.3, n_frames)
    q = np.zeros((n_frames, nv), dtype=np.float64)
    q[:, 2] = 0.85
    v = np.zeros((n_frames, nv), dtype=np.float64)
    a = np.zeros((n_frames, nv), dtype=np.float64)
    tau = np.ones((n_frames, nv - 6), dtype=np.float64) * 15.0

    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.DYNAMIC,
        engine=engine,
        model_name=f"anthro_{capture}_{engine}",
        model_sha256=hashlib.sha256(f"model_{engine}".encode()).hexdigest(),
        coordinate_names=coord_names,
        velocity_names=coord_names,
        actuator_names=tuple(f"act_{i}" for i in range(nv - 6)),
    )
    cand = MatchedSwingCandidate(
        metadata=meta,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
    )

    q_matrix = StageQualificationMatrix()
    if stage_up_to is not None:
        ordered = list(QualificationStage)
        limit_idx = ordered.index(stage_up_to)
        for i in range(limit_idx + 1):
            q_matrix.record_stage(
                ordered[i],
                stage_state,
                metrics={"marker_rmse_mm": 15.0, "max_residual": 5e-5},
                reason="Simulated benchmark verification",
            )

    strat = MatchingStrategyContract(
        strategy_id=f"sample-{engine}-{capture}",
        engine=engine,
        model_id=f"anthro_{capture}",
        model_sha256=meta.model_sha256,
        capture_id=capture,
        strategy_preset=preset,
        coordinate_mapping={name: i for i, name in enumerate(coord_names)},
        actuator_mapping={f"act_{i}": i for i in range(nv - 6)},
        qualification=q_matrix,
        runtime_budget_ms=2000.0,
        measured_runtime_ms=1450.0,
    )

    reactions = ContactReactionHistory(
        contact_names=("heel_r", "forefoot_r", "heel_l", "forefoot_l"),
        reactions=np.ones((n_frames, 12), dtype=np.float64) * 80.0,
        closure_wrench=np.zeros((n_frames, 6), dtype=np.float64),
        root_wrench_residual=np.zeros((n_frames, 6), dtype=np.float64),
    )

    return CandidateStrategyPackage(
        candidate=cand,
        strategy=strat,
        a=a,
        reactions=reactions,
    )
