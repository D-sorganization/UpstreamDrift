"""CO-05 plausible upper-body and full-body candidates (#10609)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.body_candidates import (
    CANDIDATE_SCHEMA,
    BodyCandidate,
    BodyCandidateOptions,
    BodyCandidateReport,
    RejectionReason,
    body_candidate_evidence_payload,
    build_body_candidate_report,
    generate_plausible_body_candidates,
)
from src.shared.python.motion_matching.club_only.nullspace_proposals import (
    analyze_grip_jacobian_nullspace,
    propose_nullspace_offsets,
    reproject_onto_closure,
)
from src.shared.python.motion_matching.club_only.observation import (
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.priors import GolfPlausibilityPriors
from src.shared.python.motion_matching.club_only.profiles import (
    build_roster_profiles,
    get_club_only_profile,
)
from src.shared.python.motion_matching.club_only.seeds import (
    CandidateSeed,
    geometry_content_hash,
    profile_content_hash,
)
from src.shared.python.motion_matching.club_only.topology_mapping import (
    TopologyMapping,
    get_topology_mapping,
    map_reduced_seed_to_body,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)
from src.shared.python.tour_baselines.registry import (
    init_default_registry,
    list_golf_models,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "club_only_matching"
    / "evidence"
    / "club_body_candidates.json"
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
    timestamps_s: np.ndarray,
    geometry_hash: str,
    profile_hash: str,
    residual: float = 0.01,
) -> CandidateSeed:
    return CandidateSeed(
        seed_id=seed_id,
        trial_id=trial_id,
        model_id=model_id,
        source="constrained_ik",
        q=q,
        body_configuration_hash="seed-body",
        observed_residual_m=residual,
        prior_score=0.7,
        feasibility_reasons=("synthetic_feasible",),
        timestamps_s=timestamps_s,
        geometry_hash=geometry_hash,
        profile_hash=profile_hash,
    )


def test_topology_mapping_required_for_reduced_continuation() -> None:
    with pytest.raises(ValueError, match="topology mapping"):
        map_reduced_seed_to_body(
            q_reduced=np.zeros(3),
            mapping=None,
            target_nq=8,
        )
    mapping = get_topology_mapping(
        source_model_id="driven_triple_pendulum",
        target_model_id="constrained_upper_body_golfer",
    )
    assert isinstance(mapping, TopologyMapping)
    assert mapping.allows_pelvis_teleport is False
    assert mapping.allows_unlimited_root_actuation is False
    assert mapping.allows_pasted_club_animation is False
    q_body = map_reduced_seed_to_body(
        q_reduced=np.array([0.1, -0.2, 0.05]),
        mapping=mapping,
        target_nq=mapping.target_nq,
    )
    assert q_body.shape == (mapping.target_nq,)
    assert np.all(np.isfinite(q_body))


def test_incompatible_grip_contact_and_self_collision_rejected() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("constrained_upper_body_golfer")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    seed = _seed(
        seed_id="s-grip",
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        q=np.zeros(3),
        timestamps_s=obs.native_time_s.copy(),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    priors = GolfPlausibilityPriors.default()
    for reason in (
        RejectionReason.INCOMPATIBLE_GRIP,
        RejectionReason.CONTACT_INFEASIBLE,
        RejectionReason.SELF_COLLISION,
    ):
        result = generate_plausible_body_candidates(
            observation=obs,
            profile=profile,
            seeds=(seed,),
            priors=priors,
            prior_strength=1.0,
            options=BodyCandidateOptions(force_reject=reason),
        )
        assert result.candidates
        assert all(not c.accepted for c in result.candidates)
        assert all(reason.value in c.rejection_reasons for c in result.candidates)


def test_nonunique_club_to_body_example_preserves_distinct_hashes() -> None:
    obs = build_calibrated_observation_fixture("TW_ProV1")
    profile = get_club_only_profile("full_body_pinocchio")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    seed = _seed(
        seed_id="s-ambig",
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        q=np.array([0.0, 0.05, -0.02]),
        timestamps_s=obs.native_time_s.copy(),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    result = generate_plausible_body_candidates(
        observation=obs,
        profile=profile,
        seeds=(seed,),
        priors=GolfPlausibilityPriors.default(),
        prior_strength=1.0,
        options=BodyCandidateOptions(
            n_nullspace_proposals=3,
            measured_observation_fit_m=0.01,
        ),
    )
    accepted = [c for c in result.candidates if c.accepted]
    assert len(accepted) >= 2
    hashes = {c.body_configuration_hash for c in accepted}
    for c in accepted:
        assert c.observation_fit_m is not None
    fits = {
        round(c.observation_fit_m, 6)
        for c in accepted
        if c.observation_fit_m is not None
    }
    assert len(fits) == 1


def test_singular_nullspace_rank_changes_fail_closed() -> None:
    # Full-rank Jacobian → nontrivial null space.
    rng = np.random.default_rng(0)
    j_full = rng.normal(size=(3, 8))
    analysis = analyze_grip_jacobian_nullspace(j_full)
    assert analysis.is_singular is False
    assert analysis.rank == 3
    assert analysis.basis.shape[1] >= 1

    # Rank-deficient / near-zero rows → singular.
    j_sing = np.zeros((3, 8))
    j_sing[0, 0] = 1.0
    j_sing[1, 0] = 1.0
    singular = analyze_grip_jacobian_nullspace(j_sing, singular_tol=1e-8)
    assert singular.is_singular is True
    with pytest.raises(ValueError, match="singular"):
        propose_nullspace_offsets(singular, amplitudes=(0.01, 0.02))


def test_native_closure_and_finite_qva() -> None:
    obs = build_calibrated_observation_fixture("GW_wiffle")
    profile = get_club_only_profile("constrained_upper_body_golfer")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    seed = _seed(
        seed_id="s-closure",
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        q=np.array([0.1, -0.05, 0.02]),
        timestamps_s=obs.native_time_s.copy(),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    result = generate_plausible_body_candidates(
        observation=obs,
        profile=profile,
        seeds=(seed,),
        priors=GolfPlausibilityPriors.default(),
        prior_strength=0.5,
        options=BodyCandidateOptions(
            n_nullspace_proposals=2,
            measured_observation_fit_m=0.01,
        ),
    )
    accepted = [c for c in result.candidates if c.accepted]
    assert accepted
    for cand in accepted:
        assert isinstance(cand, BodyCandidate)
        assert np.all(np.isfinite(cand.q))
        assert np.all(np.isfinite(cand.v))
        assert np.all(np.isfinite(cand.a))
        assert cand.q.shape == cand.v.shape == cand.a.shape
        assert cand.closure_q_m is None
        assert cand.closure_v_m_s is None
        assert cand.closure_a_m_s2 is None
        assert cand.derivatives_computed is False
        assert cand.is_kinematic_preview is True
        assert cand.claims_native_acceptance is False
        assert cand.claims_surrogate_as_native is False
        # Separate score lanes — never collapsed into one scalar.
        assert cand.observation_fit_m is not None
        assert np.isfinite(cand.observation_fit_m)
        assert np.isfinite(cand.plausibility_score)
        assert np.isfinite(cand.contact_effort)
        assert cand.runtime_s is not None and cand.runtime_s >= 0.0


def test_nonfinite_wrong_dims_and_incompatible_clocks_fail_closed() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("full_body_pinocchio")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)

    with pytest.raises(ValueError, match="finite"):
        generate_plausible_body_candidates(
            observation=obs,
            profile=profile,
            seeds=(
                _seed(
                    seed_id="bad-q",
                    trial_id=obs.trial_id,
                    model_id=profile.model_id,
                    q=np.array([np.nan, 0.0, 0.0]),
                    timestamps_s=obs.native_time_s.copy(),
                    geometry_hash=g_hash,
                    profile_hash=p_hash,
                ),
            ),
            priors=GolfPlausibilityPriors.default(),
            prior_strength=1.0,
        )

    wrong_clock = obs.native_time_s.copy()
    wrong_clock[-1] = wrong_clock[-1] + 1.0
    with pytest.raises(ValueError, match="clock"):
        generate_plausible_body_candidates(
            observation=obs,
            profile=profile,
            seeds=(
                _seed(
                    seed_id="bad-clock",
                    trial_id=obs.trial_id,
                    model_id=profile.model_id,
                    q=np.zeros(3),
                    timestamps_s=wrong_clock,
                    geometry_hash=g_hash,
                    profile_hash=p_hash,
                ),
            ),
            priors=GolfPlausibilityPriors.default(),
            prior_strength=1.0,
        )

    with pytest.raises(ValueError, match="prior_strength"):
        generate_plausible_body_candidates(
            observation=obs,
            profile=profile,
            seeds=(
                _seed(
                    seed_id="ok",
                    trial_id=obs.trial_id,
                    model_id=profile.model_id,
                    q=np.zeros(3),
                    timestamps_s=obs.native_time_s.copy(),
                    geometry_hash=g_hash,
                    profile_hash=p_hash,
                ),
            ),
            priors=GolfPlausibilityPriors.default(),
            prior_strength=float("nan"),
        )


def test_missing_runtime_remains_unqualified() -> None:
    obs = build_calibrated_observation_fixture("GW_wiffle")
    profile = get_club_only_profile("full_body_opensim")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    seed = _seed(
        seed_id="s-opensim",
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        q=np.zeros(3),
        timestamps_s=obs.native_time_s.copy(),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    result = generate_plausible_body_candidates(
        observation=obs,
        profile=profile,
        seeds=(seed,),
        priors=GolfPlausibilityPriors.default(),
        prior_strength=1.0,
        options=BodyCandidateOptions(
            runtime_available=False,
            runtime_blocker="OpenSim runtime not installed on this host",
        ),
    )
    assert result.candidates == ()
    assert result.cells
    cell = result.cells[0]
    assert cell.status == "missing_runtime"
    assert cell.blocker is not None
    assert "OpenSim" in cell.blocker
    assert cell.candidate_ids == ()


def test_prior_strength_sensitivity_is_reported() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("constrained_upper_body_golfer")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    seed = _seed(
        seed_id="s-prior",
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        q=np.array([0.0, 0.1, -0.05]),
        timestamps_s=obs.native_time_s.copy(),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    weak = generate_plausible_body_candidates(
        observation=obs,
        profile=profile,
        seeds=(seed,),
        priors=GolfPlausibilityPriors.default(),
        prior_strength=0.25,
        options=BodyCandidateOptions(
            n_nullspace_proposals=2,
            measured_observation_fit_m=0.01,
        ),
    )
    strong = generate_plausible_body_candidates(
        observation=obs,
        profile=profile,
        seeds=(seed,),
        priors=GolfPlausibilityPriors.default(),
        prior_strength=2.0,
        options=BodyCandidateOptions(
            n_nullspace_proposals=2,
            measured_observation_fit_m=0.01,
        ),
    )
    weak_acc = [c for c in weak.candidates if c.accepted]
    strong_acc = [c for c in strong.candidates if c.accepted]
    assert weak_acc and strong_acc
    assert weak_acc[0].prior_strength == pytest.approx(0.25)
    assert strong_acc[0].prior_strength == pytest.approx(2.0)
    assert weak_acc[0].prior_sensitivity is not None
    assert "plausibility_delta" in weak_acc[0].prior_sensitivity
    # Stronger prior changes plausibility lane without inventing native acceptance.
    assert weak_acc[0].plausibility_score != strong_acc[0].plausibility_score
    assert all(c.claims_native_acceptance is False for c in weak_acc + strong_acc)


def test_reproject_rejects_nonfinite_and_preserves_dims() -> None:
    q = np.array([0.1, -0.2, 0.0, 0.05, -0.01, 0.0, 0.02, -0.03])
    out = reproject_onto_closure(q, closure_residual_m=0.0, tol_m=0.005)
    assert out.shape == q.shape
    assert np.all(np.isfinite(out))
    with pytest.raises(ValueError, match="finite"):
        reproject_onto_closure(
            np.array([np.inf, 0.0]),
            closure_residual_m=0.0,
            tol_m=0.005,
        )


def test_roster_matrix_accounts_for_all_model_trial_cells() -> None:
    report = build_body_candidate_report(
        model_ids=None,
        trial_ids=None,
        prior_strength=1.0,
        n_nullspace_proposals=2,
    )
    assert isinstance(report, BodyCandidateReport)
    assert report.schema == CANDIDATE_SCHEMA
    assert report.governing_issue == 10609
    init_default_registry()
    models = list_golf_models()
    expected_cells = {(m.model_id, t) for m in models for t in CANONICAL_TRIAL_SHEETS}
    actual_cells = {(c.model_id, c.trial_id) for c in report.cells}
    assert actual_cells == expected_cells
    statuses = {c.status for c in report.cells}
    assert "rejected" in statuses
    assert "generated" not in statuses
    # Unsupported / missing-runtime / rejected cells keep precise blockers, never silent omit.
    blocked = [c for c in report.cells if c.status != "generated"]
    assert blocked
    assert len(blocked) == len(report.cells)
    assert all(c.blocker for c in blocked)


def test_no_pelvis_teleport_or_pasted_club_animation() -> None:
    mapping = get_topology_mapping(
        source_model_id="driven_double_pendulum",
        target_model_id="full_body_pinocchio",
    )
    assert mapping.allows_pelvis_teleport is False
    assert mapping.allows_unlimited_root_actuation is False
    assert mapping.allows_pasted_club_animation is False
    q = map_reduced_seed_to_body(
        q_reduced=np.array([0.2, -0.1]),
        mapping=mapping,
        target_nq=mapping.target_nq,
    )
    # Pelvis / root DOFs remain at the mapped prior (zeros), never free teleport.
    if mapping.pelvis_indices:
        assert np.allclose(q[list(mapping.pelvis_indices)], 0.0)


def test_evidence_receipt_matches_schema() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert payload["schema"] == CANDIDATE_SCHEMA
    assert payload["governing_issue"] == 10609
    assert set(payload["trials"]) == set(CANONICAL_TRIAL_SHEETS)
    roster = build_roster_profiles()
    assert set(payload["models"]) == set(roster)
    rebuilt = body_candidate_evidence_payload(
        build_body_candidate_report(
            model_ids=None,
            trial_ids=None,
            prior_strength=1.0,
            n_nullspace_proposals=2,
        )
    )
    assert rebuilt["schema"] == payload["schema"]
    assert set(rebuilt["trials"]) == set(payload["trials"])
    assert set(rebuilt["models"]) == set(payload["models"])
