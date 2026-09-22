"""CO-04 match club-only motion with double and triple pendulums (#10608)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.pendulum.python.motion_matching.club_match_matrix import (
    MATCH_SCHEMA,
    PENDULUM_MATCH_MODELS,
    build_pendulum_match_matrix,
    evidence_payload,
)
from src.engines.physics_engines.pendulum.python.motion_matching.club_pendulum_match import (
    match_club_pendulum,
)
from src.shared.python.motion_matching.club_only.hub_accounting import (
    HubMode,
    hub_variant_id,
    account_external_hub_work,
)
from src.shared.python.motion_matching.club_only.match_errors import (
    ClubMatchErrorReport,
    separate_plane_and_3d_errors,
)
from src.shared.python.motion_matching.club_only.observation import (
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.pendulum_match import (
    PendulumMatchRequest,
    PendulumMatchResult,
    map_seed_to_pendulum_q0,
    reject_reconstruction_as_club_evidence,
)
from src.shared.python.motion_matching.club_only.profiles import get_club_only_profile
from src.shared.python.motion_matching.club_only.replay_package import (
    build_replay_package,
)
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


def _seed(
    *,
    seed_id: str,
    trial_id: str,
    model_id: str,
    q: np.ndarray,
    geometry_hash: str,
    profile_hash: str,
) -> CandidateSeed:
    return CandidateSeed(
        seed_id=seed_id,
        trial_id=trial_id,
        model_id=model_id,
        source="retrieval",
        q=q,
        body_configuration_hash="body-hash",
        observed_residual_m=0.02,
        prior_score=0.7,
        feasibility_reasons=("synthetic_feasible",),
        timestamps_s=np.array([0.0, 1.0 / 240.0]),
        geometry_hash=geometry_hash,
        profile_hash=profile_hash,
    )


def test_hub_variants_have_distinct_ids_and_work_accounting() -> None:
    fixed = hub_variant_id("driven_double_pendulum", HubMode.FIXED_PIVOT)
    moving = hub_variant_id("driven_triple_pendulum", HubMode.PRESCRIBED_MOVING_HUB)
    assert fixed != moving
    assert "fixed_pivot" in fixed
    assert "prescribed_moving_hub" in moving

    times = np.array([0.0, 0.01, 0.02])
    fixed_work = account_external_hub_work(
        times=times,
        hub_positions=np.zeros((3, 2)),
        hub_reaction_forces=np.zeros((3, 2)),
        hub_mode=HubMode.FIXED_PIVOT,
    )
    assert fixed_work.is_moving_hub is False
    assert fixed_work.total_work_joules == 0.0

    moving_pos = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]])
    forces = np.array([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
    moving_work = account_external_hub_work(
        times=times,
        hub_positions=moving_pos,
        hub_reaction_forces=forces,
        hub_mode=HubMode.PRESCRIBED_MOVING_HUB,
    )
    assert moving_work.is_moving_hub is True
    assert moving_work.total_work_joules != 0.0


def test_in_plane_and_3d_errors_reported_separately() -> None:
    pred = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])
    meas = np.zeros((2, 3))
    plane_basis = np.eye(3)
    report = separate_plane_and_3d_errors(
        predicted_xyz_m=pred,
        measured_xyz_m=meas,
        plane_origin=np.zeros(3),
        plane_basis=plane_basis,
        observed_mask=np.array([True, True]),
    )
    assert isinstance(report, ClubMatchErrorReport)
    assert report.in_plane_rmse_m == pytest.approx(0.0, abs=1e-12)
    assert report.original_3d_rmse_m == pytest.approx(
        math.sqrt((0.1**2 + 0.2**2) / 2.0), rel=1e-9
    )
    assert report.in_plane_rmse_m != report.original_3d_rmse_m


def test_rejects_reconstruction_scores_as_club_evidence() -> None:
    with pytest.raises(ValueError, match="reconstruction"):
        reject_reconstruction_as_club_evidence("reconstruction_double_pendulum")
    with pytest.raises(ValueError, match="reconstruction"):
        reject_reconstruction_as_club_evidence("reconstruction_triple_pendulum")
    reject_reconstruction_as_club_evidence("driven_double_pendulum")


def test_seed_mapping_requires_valid_dof_and_hashes() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("driven_double_pendulum")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    good = _seed(
        seed_id="s1",
        trial_id="TW_wiffle",
        model_id="driven_double_pendulum",
        q=np.array([-1.0, 0.5]),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    mapped = map_seed_to_pendulum_q0(
        good,
        model_id="driven_double_pendulum",
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    assert mapped is not None
    assert mapped.shape == (2,)

    wrong_dim = _seed(
        seed_id="s2",
        trial_id="TW_wiffle",
        model_id="driven_double_pendulum",
        q=np.array([-1.0, 0.5, 0.1]),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    assert (
        map_seed_to_pendulum_q0(
            wrong_dim,
            model_id="driven_double_pendulum",
            geometry_hash=g_hash,
            profile_hash=p_hash,
        )
        is None
    )

    stale = _seed(
        seed_id="s3",
        trial_id="TW_wiffle",
        model_id="driven_double_pendulum",
        q=np.array([-1.0, 0.5]),
        geometry_hash="0" * 64,
        profile_hash=p_hash,
    )
    assert (
        map_seed_to_pendulum_q0(
            stale,
            model_id="driven_double_pendulum",
            geometry_hash=g_hash,
            profile_hash=p_hash,
        )
        is None
    )


def test_first_frame_scored_before_integration() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    result = match_club_pendulum(
        PendulumMatchRequest(
            observation=obs,
            model_id="driven_double_pendulum",
            hub_mode=HubMode.FIXED_PIVOT,
            max_nfev=3,
            seeds=(),
        )
    )
    assert isinstance(result, PendulumMatchResult)
    assert result.t0_evaluated_before_step is True
    assert math.isfinite(result.first_frame_rmse_m)
    assert result.first_frame_rmse_m >= 0.0
    assert result.coverage_fraction > 0.0


def test_cold_vs_retrieval_retains_best_feasible() -> None:
    obs = build_calibrated_observation_fixture("GW_wiffle")
    profile = get_club_only_profile("driven_double_pendulum")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    seed = _seed(
        seed_id="retrieval:good",
        trial_id="GW_wiffle",
        model_id="driven_double_pendulum",
        q=np.array([-math.pi / 2, math.pi / 4]),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    result = match_club_pendulum(
        PendulumMatchRequest(
            observation=obs,
            model_id="driven_double_pendulum",
            hub_mode=HubMode.FIXED_PIVOT,
            max_nfev=4,
            seeds=(seed,),
            geometry_hash=g_hash,
            profile_hash=p_hash,
        )
    )
    assert result.cold_start_rmse_m is not None
    assert result.retrieval_start_rmse_m is not None
    assert result.selected_start in {"cold", "retrieval"}
    selected_rmse = (
        result.cold_start_rmse_m
        if result.selected_start == "cold"
        else result.retrieval_start_rmse_m
    )
    assert selected_rmse == min(result.cold_start_rmse_m, result.retrieval_start_rmse_m)


def test_triple_moving_hub_distinct_from_fixed_double() -> None:
    obs = build_calibrated_observation_fixture("TW_ProV1")
    double = match_club_pendulum(
        PendulumMatchRequest(
            observation=obs,
            model_id="driven_double_pendulum",
            hub_mode=HubMode.FIXED_PIVOT,
            max_nfev=2,
            seeds=(),
        )
    )
    triple = match_club_pendulum(
        PendulumMatchRequest(
            observation=obs,
            model_id="driven_triple_pendulum",
            hub_mode=HubMode.PRESCRIBED_MOVING_HUB,
            max_nfev=2,
            seeds=(),
        )
    )
    assert double.hub_variant_id != triple.hub_variant_id
    assert double.model_id == "driven_double_pendulum"
    assert triple.model_id == "driven_triple_pendulum"
    assert triple.external_work_joules is not None
    assert math.isfinite(triple.external_work_joules)


def test_replay_package_leaves_native_gates_explicit() -> None:
    obs = build_calibrated_observation_fixture("GW_ProV11")
    result = match_club_pendulum(
        PendulumMatchRequest(
            observation=obs,
            model_id="driven_double_pendulum",
            hub_mode=HubMode.FIXED_PIVOT,
            max_nfev=2,
            seeds=(),
        )
    )
    package = build_replay_package(result)
    assert package.schema.startswith("club-pendulum-replay/")
    assert package.native_g1_pass is False
    assert len(package.qualification_blockers) >= 1
    assert "theta" in package.replay_inputs
    assert "q0" in package.replay_inputs
    assert package.claims_native_qualification is False


def test_eight_model_trial_matrix_and_evidence() -> None:
    matrix = build_pendulum_match_matrix(max_nfev=2)
    assert matrix.schema == MATCH_SCHEMA
    assert set(PENDULUM_MATCH_MODELS) == {
        "driven_double_pendulum",
        "driven_triple_pendulum",
    }
    assert len(matrix.outcomes) == 8
    pairs = {(o.model_id, o.trial_id) for o in matrix.outcomes}
    expected = {(m, t) for m in PENDULUM_MATCH_MODELS for t in CANONICAL_TRIAL_SHEETS}
    assert pairs == expected
    for outcome in matrix.outcomes:
        assert math.isfinite(outcome.in_plane_rmse_m)
        assert math.isfinite(outcome.original_3d_rmse_m)
        assert outcome.t0_evaluated_before_step is True
        assert outcome.native_g1_pass is False
        assert outcome.qualification_blockers
        assert outcome.replay_inputs
        assert "reconstruction" not in outcome.model_id

    payload = evidence_payload(matrix)
    assert payload["schema"] == MATCH_SCHEMA
    assert payload["governing_issue"] == 10608
    assert len(payload["outcomes"]) == 8
    assert EVIDENCE.is_file()
    on_disk = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert on_disk["schema"] == MATCH_SCHEMA
    assert len(on_disk["outcomes"]) == 8


def test_dbc_rejects_nonfinite_observation_inputs() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    bad_mid = np.array(obs.mid_hands_xyz, copy=True)
    bad_mid[0, 0] = np.nan
    # Bypass ClubObservation __post_init__ so PendulumMatchRequest DbC is exercised.
    object.__setattr__(obs, "mid_hands_xyz", bad_mid)
    with pytest.raises(ValueError, match="finite"):
        match_club_pendulum(
            PendulumMatchRequest(
                observation=obs,
                model_id="driven_double_pendulum",
                hub_mode=HubMode.FIXED_PIVOT,
                max_nfev=1,
                seeds=(),
            )
        )
