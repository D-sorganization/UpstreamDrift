"""CO-07 optimize fast matching and expose candidate diversity (#10611)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.constrained_ik import (
    generate_posture_branches,
)
from src.shared.python.motion_matching.club_only.fast_matching import (
    FAST_MATCH_SCHEMA,
    CacheKeyMismatchError,
    EmptyNeuralProposalProvider,
    FastMatchOptions,
    ImmutableMatchCache,
    MatchBudget,
    MatchCancelledError,
    MatchCheckpoint,
    MatchPreset,
    StartStrategy,
    budget_for_preset,
    build_fast_match_evidence,
    cache_key_from_parts,
    checkpoint_identity,
    prune_and_select_pareto,
    run_fast_club_match,
    score_branch,
    target_content_hash,
)
from src.shared.python.motion_matching.club_only.observation import (
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.profiles import get_club_only_profile
from src.shared.python.motion_matching.club_only.seeds import (
    CandidateSeed,
    geometry_content_hash,
    profile_content_hash,
)
from src.shared.python.motion_matching.club_only.topology_mapping import (
    get_topology_mapping,
    map_reduced_seed_to_body,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "club_only_matching"
    / "evidence"
    / "club_fast_matching.json"
)


def _geometry_hash(obs) -> str:
    return geometry_content_hash(
        club_type=obs.club_type,
        catalog_length_m=obs.catalog_length_m,
        tool_to_model_residual_m=0.0,
    )


def _seed(
    *,
    trial_id: str,
    model_id: str,
    q: np.ndarray,
    timestamps_s: np.ndarray,
    geometry_hash: str,
    profile_hash: str,
    residual: float = 0.02,
) -> CandidateSeed:
    return CandidateSeed(
        seed_id=f"seed-{trial_id}",
        trial_id=trial_id,
        model_id=model_id,
        source="retrieval",
        q=q,
        body_configuration_hash="body-hash",
        observed_residual_m=residual,
        prior_score=0.75,
        feasibility_reasons=("synthetic_feasible",),
        timestamps_s=timestamps_s,
        geometry_hash=geometry_hash,
        profile_hash=profile_hash,
    )


def test_cache_stale_or_missing_mismatch() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("driven_double_pendulum")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    t_hash = target_content_hash(obs)
    key = cache_key_from_parts(
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        geometry_hash=g_hash,
        profile_hash=p_hash,
        target_content_hash=t_hash,
    )
    cache = ImmutableMatchCache()
    cache.store(key, {"warm_q": [0.1, 0.2]})
    assert cache.load(key) == {"warm_q": [0.1, 0.2]}
    stale_key = cache_key_from_parts(
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        geometry_hash="deadbeef" * 8,
        profile_hash=p_hash,
        target_content_hash=t_hash,
    )
    assert cache.load(stale_key) is None
    with pytest.raises(CacheKeyMismatchError, match="target_content_hash"):
        cache.assert_compatible(
            key,
            cache_key_from_parts(
                trial_id=obs.trial_id,
                model_id=profile.model_id,
                geometry_hash=g_hash,
                profile_hash=p_hash,
                target_content_hash="other-target",
            ),
        )


def test_bounded_cancellation() -> None:
    obs = build_calibrated_observation_fixture("TW_ProV1")
    profile = get_club_only_profile("driven_double_pendulum")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    seed = _seed(
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        q=np.array([0.05, -0.02, 0.01, 0.0]),
        timestamps_s=obs.native_time_s.copy(),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    budget = MatchBudget(max_time_s=60.0, max_evaluations=2, max_pareto=4)
    result = run_fast_club_match(
        observation=obs,
        profile=profile,
        seed=seed,
        geometry_hash=g_hash,
        profile_hash=p_hash,
        options=FastMatchOptions(
            preset=MatchPreset.FAST_PREVIEW,
            budget=budget,
            n_branches=6,
        ),
    )
    assert result.cancelled is True
    assert result.evaluations_used <= 2
    assert result.limitations


def test_checkpoint_identity() -> None:
    obs = build_calibrated_observation_fixture("GW_wiffle")
    profile = get_club_only_profile("driven_double_pendulum")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    t_hash = target_content_hash(obs)
    key = cache_key_from_parts(
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        geometry_hash=g_hash,
        profile_hash=p_hash,
        target_content_hash=t_hash,
    )
    ckpt_a = MatchCheckpoint(
        cache_key=key,
        preset=MatchPreset.VERIFIED_FIT,
        evaluations_used=3,
        branch_index=1,
        pareto_ids=("b0", "b1"),
        diversity_seed=10611,
    )
    ckpt_b = MatchCheckpoint(
        cache_key=key,
        preset=MatchPreset.VERIFIED_FIT,
        evaluations_used=3,
        branch_index=1,
        pareto_ids=("b0", "b1"),
        diversity_seed=10611,
    )
    ckpt_c = MatchCheckpoint(
        cache_key=key,
        preset=MatchPreset.VERIFIED_FIT,
        evaluations_used=4,
        branch_index=1,
        pareto_ids=("b0", "b1"),
        diversity_seed=10611,
    )
    assert checkpoint_identity(ckpt_a) == checkpoint_identity(ckpt_b)
    assert checkpoint_identity(ckpt_a) != checkpoint_identity(ckpt_c)


def test_pruning_rejects_attractive_infeasible_candidates() -> None:
    profile = get_club_only_profile("constrained_upper_body_golfer")
    max_closure = profile.max_closure_residual_m()
    branches = (
        score_branch(
            branch_id="good",
            start_strategy=StartStrategy.RETRIEVAL,
            q=np.array([0.02, -0.01, 0.005]),
            seed_residual_m=0.02,
            profile=profile,
        ),
        score_branch(
            branch_id="attractive-infeasible",
            start_strategy=StartStrategy.COLD,
            q=np.array([3.0, 4.0, 0.0]),
            seed_residual_m=0.001,
            profile=profile,
        ),
    )
    assert branches[1].observation_fit_m < branches[0].observation_fit_m
    assert branches[1].closure_m > max_closure
    pareto, rejected = prune_and_select_pareto(
        branches,
        profile=profile,
        max_pareto=4,
    )
    assert all(r.branch_id != "attractive-infeasible" for r in pareto)
    assert any(r.branch_id == "attractive-infeasible" for r in rejected)


def test_deterministic_branch_diversity() -> None:
    base = np.array([0.05, -0.02, 0.01, 0.0])
    a = generate_posture_branches(base, n_branches=4, amplitude_rad=0.04)
    b = generate_posture_branches(base, n_branches=4, amplitude_rad=0.04)
    assert len(a) == len(b) == 4
    for qa, qb in zip(a, b, strict=True):
        assert np.allclose(qa, qb)


def test_neural_proposals_optional_without_weights() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("driven_double_pendulum")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    seed = _seed(
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        q=np.array([0.04, -0.01, 0.02, 0.0]),
        timestamps_s=obs.native_time_s.copy(),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    provider = EmptyNeuralProposalProvider()
    assert provider.propose(observation=obs, profile=profile) == ()
    result = run_fast_club_match(
        observation=obs,
        profile=profile,
        seed=seed,
        geometry_hash=g_hash,
        profile_hash=p_hash,
        options=FastMatchOptions(neural_provider=provider),
    )
    assert result.neural_proposals_used == 0
    assert result.pareto


def test_profiling_includes_verification_time() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("driven_double_pendulum")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    seed = _seed(
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        q=np.array([0.03, -0.02, 0.01, 0.0]),
        timestamps_s=obs.native_time_s.copy(),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    result = run_fast_club_match(
        observation=obs,
        profile=profile,
        seed=seed,
        geometry_hash=g_hash,
        profile_hash=p_hash,
        options=FastMatchOptions(preset=MatchPreset.VERIFIED_FIT),
    )
    timings = result.profiling
    assert timings.verification_s >= 0.0
    assert timings.load_s >= 0.0
    assert timings.calibration_s >= 0.0
    assert timings.ik_s >= 0.0
    assert timings.dynamics_s >= 0.0
    assert timings.replay_s >= 0.0
    assert timings.total_s >= timings.verification_s


def test_reduced_to_full_start_strategy_uses_topology_map() -> None:
    obs = build_calibrated_observation_fixture("TW_ProV1")
    profile = get_club_only_profile("constrained_upper_body_golfer")
    mapping = get_topology_mapping(
        source_model_id="driven_triple_pendulum",
        target_model_id=profile.model_id,
    )
    q_reduced = np.array([0.1, -0.05, 0.02])
    q_full = map_reduced_seed_to_body(
        q_reduced=q_reduced,
        mapping=mapping,
        target_nq=mapping.target_nq,
    )
    assert q_full.shape == (mapping.target_nq,)


def test_fast_match_evidence_fixture_roundtrip() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("driven_double_pendulum")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    seed = _seed(
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        q=np.array([0.05, -0.02, 0.01, 0.0]),
        timestamps_s=obs.native_time_s.copy(),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    result = run_fast_club_match(
        observation=obs,
        profile=profile,
        seed=seed,
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    payload = build_fast_match_evidence(result)
    assert payload["schema"] == FAST_MATCH_SCHEMA
    assert payload["governing_issue"] == 10611
    assert payload["claims_native_qualification"] is False
    assert payload["native_g1_pass"] is False
    assert payload["pareto"]
    assert payload["quality_vs_time"]
    assert payload["failed_attempts"] >= 0
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    assert EVIDENCE.exists()


def test_preset_budgets_are_ordered() -> None:
    fast = budget_for_preset(MatchPreset.FAST_PREVIEW)
    verified = budget_for_preset(MatchPreset.VERIFIED_FIT)
    assert fast.max_evaluations < verified.max_evaluations
    assert fast.max_time_s < verified.max_time_s


def test_cancel_check_raises_match_cancelled() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("driven_double_pendulum")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    seed = _seed(
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        q=np.array([0.05, -0.02, 0.01, 0.0]),
        timestamps_s=obs.native_time_s.copy(),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    with pytest.raises(MatchCancelledError):
        run_fast_club_match(
            observation=obs,
            profile=profile,
            seed=seed,
            geometry_hash=g_hash,
            profile_hash=p_hash,
            options=FastMatchOptions(cancel_check=lambda: True),
        )


def test_resume_from_checkpoint_skips_prior_evaluations() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("driven_double_pendulum")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    seed = _seed(
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        q=np.array([0.05, -0.02, 0.01, 0.0]),
        timestamps_s=obs.native_time_s.copy(),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    budget = MatchBudget(max_time_s=60.0, max_evaluations=8, max_pareto=4)
    partial = run_fast_club_match(
        observation=obs,
        profile=profile,
        seed=seed,
        geometry_hash=g_hash,
        profile_hash=p_hash,
        options=FastMatchOptions(
            budget=budget,
            n_branches=8,
            checkpoint=MatchCheckpoint(
                cache_key=cache_key_from_parts(
                    trial_id=obs.trial_id,
                    model_id=profile.model_id,
                    geometry_hash=g_hash,
                    profile_hash=p_hash,
                    target_content_hash=target_content_hash(obs),
                ),
                preset=MatchPreset.FAST_PREVIEW,
                evaluations_used=4,
                branch_index=4,
                pareto_ids=(),
                diversity_seed=10611,
            ),
        ),
    )
    assert partial.evaluations_used <= 8
    assert partial.checkpoint is not None
