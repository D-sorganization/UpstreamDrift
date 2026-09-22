"""CO-04 club-only double/triple pendulum matching (#10608)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.observation import (
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.pendulum_match import (
    MATCH_SCHEMA,
    MODEL_ID_DOUBLE,
    MODEL_ID_TRIPLE,
    PendulumMatchOutcome,
    build_pendulum_match_matrix,
    evidence_payload,
    match_club_only_pendulum,
)
from src.shared.python.motion_matching.club_only.profiles import get_club_only_profile
from src.shared.python.motion_matching.club_only.seeds import (
    CandidateSeed,
    geometry_content_hash,
    profile_content_hash,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "club_only_matching"
    / "evidence"
    / "club_pendulum_match.json"
)


def _geometry_hash(obs) -> str:
    return geometry_content_hash(
        club_type=obs.club_type,
        catalog_length_m=obs.catalog_length_m,
        tool_to_model_residual_m=0.0,
    )


def _seed_for(
    *,
    trial_id: str,
    model_id: str,
    q: np.ndarray,
    obs,
) -> CandidateSeed:
    profile = get_club_only_profile(model_id)
    return CandidateSeed(
        seed_id=f"retrieval:test-{trial_id}",
        trial_id=trial_id,
        model_id=model_id,
        source="retrieval",
        q=q,
        body_configuration_hash="abc123",
        observed_residual_m=0.02,
        prior_score=0.5,
        feasibility_reasons=("synthetic_feasible",),
        timestamps_s=np.asarray(obs.native_time_s, dtype=np.float64),
        geometry_hash=_geometry_hash(obs),
        profile_hash=profile_content_hash(profile),
    )


def test_rejects_unknown_model_under_optimize() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    with pytest.raises(ValueError, match="unsupported model"):
        match_club_only_pendulum(
            observation=obs,
            model_id="not_a_pendulum",
            max_nfev=3,
        )


def test_first_frame_scored_before_integration() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    outcome = match_club_only_pendulum(
        observation=obs,
        model_id=MODEL_ID_DOUBLE,
        max_nfev=8,
    )
    assert outcome.t0_evaluated_before_step is True
    assert math.isfinite(outcome.first_frame_grip_error_m)
    assert math.isfinite(outcome.first_frame_face_error_m)


def test_reports_in_plane_and_3d_residuals_separately() -> None:
    obs = build_calibrated_observation_fixture("TW_ProV1")
    outcome = match_club_only_pendulum(
        observation=obs,
        model_id=MODEL_ID_DOUBLE,
        max_nfev=8,
    )
    assert math.isfinite(outcome.in_plane_face_rmse_m)
    assert math.isfinite(outcome.spatial_3d_face_rmse_m)
    assert outcome.in_plane_face_rmse_m != outcome.spatial_3d_face_rmse_m or (
        outcome.out_of_plane_rmse_m == 0.0
    )
    assert "in_plane_face_rmse_m" in outcome.as_dict()
    assert "spatial_3d_face_rmse_m" in outcome.as_dict()


def test_double_and_triple_have_distinct_ids_and_hub_accounting() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    double = match_club_only_pendulum(
        observation=obs, model_id=MODEL_ID_DOUBLE, max_nfev=5
    )
    triple = match_club_only_pendulum(
        observation=obs, model_id=MODEL_ID_TRIPLE, max_nfev=5
    )
    assert double.model_id == MODEL_ID_DOUBLE
    assert triple.model_id == MODEL_ID_TRIPLE
    assert double.is_moving_hub is False
    assert triple.is_moving_hub is True
    assert math.isfinite(triple.external_work_joules)


def test_warm_start_used_only_when_seed_mapping_valid() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    valid = _seed_for(
        trial_id="TW_wiffle",
        model_id=MODEL_ID_DOUBLE,
        q=np.array([-1.0, 0.4], dtype=np.float64),
        obs=obs,
    )
    invalid = _seed_for(
        trial_id="TW_wiffle",
        model_id=MODEL_ID_DOUBLE,
        q=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        obs=obs,
    )
    with_valid = match_club_only_pendulum(
        observation=obs,
        model_id=MODEL_ID_DOUBLE,
        seed=valid,
        max_nfev=5,
    )
    with_invalid = match_club_only_pendulum(
        observation=obs,
        model_id=MODEL_ID_DOUBLE,
        seed=invalid,
        max_nfev=5,
    )
    assert with_valid.warm_start_applied is True
    assert with_invalid.warm_start_applied is False
    assert with_invalid.start_mode == "cold"


def test_retains_best_of_cold_and_retrieval() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    seed = _seed_for(
        trial_id="TW_wiffle",
        model_id=MODEL_ID_DOUBLE,
        q=np.array([-0.8, 0.5], dtype=np.float64),
        obs=obs,
    )
    outcome = match_club_only_pendulum(
        observation=obs,
        model_id=MODEL_ID_DOUBLE,
        seed=seed,
        max_nfev=6,
        compare_cold_and_retrieval=True,
    )
    assert outcome.start_mode in {"cold", "retrieval"}
    assert outcome.cold_in_plane_face_rmse_m is not None
    assert outcome.retrieval_in_plane_face_rmse_m is not None
    best = min(
        outcome.cold_in_plane_face_rmse_m,
        outcome.retrieval_in_plane_face_rmse_m,
    )
    assert abs(outcome.in_plane_face_rmse_m - best) < 1e-9


def test_replay_package_is_native_ready() -> None:
    obs = build_calibrated_observation_fixture("GW_wiffle")
    outcome = match_club_only_pendulum(
        observation=obs, model_id=MODEL_ID_DOUBLE, max_nfev=5
    )
    pkg = outcome.replay_package
    assert pkg["model_id"] == MODEL_ID_DOUBLE
    assert pkg["trial_id"] == "GW_wiffle"
    assert "q0" in pkg and "v0" in pkg
    assert "torque_controls" in pkg
    assert "times_s" in pkg
    assert pkg["t0_evaluated_before_step"] is True
    assert len(pkg["q0"]) == 2


def test_eight_cell_matrix_covers_two_models_four_trials() -> None:
    matrix = build_pendulum_match_matrix(max_nfev=4)
    assert matrix.schema == MATCH_SCHEMA
    assert len(matrix.outcomes) == 8
    model_ids = {o.model_id for o in matrix.outcomes}
    trial_ids = {o.trial_id for o in matrix.outcomes}
    assert model_ids == {MODEL_ID_DOUBLE, MODEL_ID_TRIPLE}
    assert trial_ids == set(CANONICAL_TRIAL_SHEETS)
    for outcome in matrix.outcomes:
        assert isinstance(outcome, PendulumMatchOutcome)
        assert math.isfinite(outcome.native_coverage_fraction)
        assert 0.0 <= outcome.native_coverage_fraction <= 1.0


def test_evidence_payload_lists_blockers_when_gates_unmet() -> None:
    matrix = build_pendulum_match_matrix(max_nfev=3)
    payload = evidence_payload(matrix)
    assert payload["schema"] == MATCH_SCHEMA
    assert payload["governing_issue"] == 10608
    assert len(payload["outcomes"]) == 8
    assert "qualification_blockers" in payload
    assert isinstance(payload["qualification_blockers"], list)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    loaded = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert loaded["cell_count"] == 8
