"""Honesty and fail-closed tests for body candidates (#10960 P1-5).

Guarantees:
- Synthetic seeds have source 'synthetic', keep 'synthetic_co05_seed' reason,
  and every candidate produced from them is rejected (accepted=False) with
  reason 'synthetic_seed' or 'observation_fit_not_computed'.
- Body candidate observation_fit_m is float | None; candidates without measured
  fit are rejected (observation_fit_not_computed).
- Measured fit above profile gate is rejected (observation_fit_above_gate);
  measured fit at/below gate with non-synthetic seed is accepted.
- Closure fields (closure_q_m, closure_v_m_s, closure_a_m_s2) are None and
  serialize as JSON null.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.body_candidates import (
    BodyCandidate,
    BodyCandidateOptions,
    RejectionReason,
    _synthetic_seed_for,
    generate_plausible_body_candidates,
)
from src.shared.python.motion_matching.club_only.observation import (
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.priors import GolfPlausibilityPriors
from src.shared.python.motion_matching.club_only.profiles import (
    get_club_only_profile,
)
from src.shared.python.motion_matching.club_only.seeds import (
    CandidateSeed,
    geometry_content_hash,
    profile_content_hash,
)

pytestmark = pytest.mark.unit


def _non_synthetic_seed(obs, profile, residual: float = 0.01) -> CandidateSeed:
    g_hash = geometry_content_hash(
        club_type=obs.club_type,
        catalog_length_m=obs.catalog_length_m,
        tool_to_model_residual_m=0.0,
    )
    p_hash = profile_content_hash(profile)
    return CandidateSeed(
        seed_id=f"test-seed:{obs.trial_id}:{profile.model_id}",
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        source="constrained_ik",
        q=np.zeros(3, dtype=np.float64),
        body_configuration_hash="seed-body-hash",
        observed_residual_m=residual,
        prior_score=0.7,
        feasibility_reasons=("ik_feasible",),
        timestamps_s=np.asarray(obs.native_time_s, dtype=np.float64).copy(),
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )


def test_synthetic_seed_source_and_fail_closed_rejection() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("constrained_upper_body_golfer")
    seed = _synthetic_seed_for(observation=obs, profile=profile)

    assert seed.source == "synthetic"
    assert "synthetic_co05_seed" in seed.feasibility_reasons

    # Generate candidates from synthetic seed without caller-supplied fit
    result = generate_plausible_body_candidates(
        observation=obs,
        profile=profile,
        seeds=(seed,),
        priors=GolfPlausibilityPriors.default(),
        prior_strength=1.0,
        options=BodyCandidateOptions(n_nullspace_proposals=2),
    )
    assert result.candidates
    for cand in result.candidates:
        assert isinstance(cand, BodyCandidate)
        assert cand.accepted is False
        assert cand.observation_fit_m is None
        assert cand.rejection_reasons[0] in {
            "synthetic_seed",
            "observation_fit_not_computed",
        }
        assert any(
            r in {"synthetic_seed", "observation_fit_not_computed"}
            for r in cand.rejection_reasons
        )

    # Even if caller passes a measured fit, a synthetic seed must NEVER be accepted
    result_with_fit = generate_plausible_body_candidates(
        observation=obs,
        profile=profile,
        seeds=(seed,),
        priors=GolfPlausibilityPriors.default(),
        prior_strength=1.0,
        options=BodyCandidateOptions(
            n_nullspace_proposals=2,
            measured_observation_fit_m=0.01,
        ),
    )
    for cand in result_with_fit.candidates:
        assert cand.accepted is False
        assert cand.rejection_reasons[0] == "synthetic_seed"
        assert "synthetic_seed" in cand.rejection_reasons


def test_supplied_measured_fit_above_gate_and_at_below_gate() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("constrained_upper_body_golfer")
    seed = _non_synthetic_seed(obs, profile)
    gate = float(profile.observation.max_grip_position_rmse_m)

    # 1. No measured fit supplied -> not computed -> rejected
    result_none = generate_plausible_body_candidates(
        observation=obs,
        profile=profile,
        seeds=(seed,),
        priors=GolfPlausibilityPriors.default(),
        prior_strength=1.0,
        options=BodyCandidateOptions(n_nullspace_proposals=2),
    )
    for cand in result_none.candidates:
        assert cand.accepted is False
        assert cand.observation_fit_m is None
        assert cand.rejection_reasons[0] == "observation_fit_not_computed"
        assert "observation_fit_not_computed" in cand.rejection_reasons

    # 2. Measured fit strictly ABOVE gate -> rejected with observation_fit_above_gate
    above_gate = gate + 0.01
    result_above = generate_plausible_body_candidates(
        observation=obs,
        profile=profile,
        seeds=(seed,),
        priors=GolfPlausibilityPriors.default(),
        prior_strength=1.0,
        options=BodyCandidateOptions(
            n_nullspace_proposals=2,
            measured_observation_fit_m=above_gate,
        ),
    )
    for cand in result_above.candidates:
        assert cand.accepted is False
        assert cand.observation_fit_m == pytest.approx(above_gate)
        assert cand.rejection_reasons[0] == "observation_fit_above_gate"
        assert "observation_fit_above_gate" in cand.rejection_reasons

    # 3. Measured fit AT gate -> accepted
    result_at = generate_plausible_body_candidates(
        observation=obs,
        profile=profile,
        seeds=(seed,),
        priors=GolfPlausibilityPriors.default(),
        prior_strength=1.0,
        options=BodyCandidateOptions(
            n_nullspace_proposals=2,
            measured_observation_fit_m=gate,
        ),
    )
    for cand in result_at.candidates:
        assert cand.accepted is True
        assert cand.observation_fit_m == pytest.approx(gate)
        assert cand.rejection_reasons == ()
        assert cand.rejection_reasons == ()

    # 4. Measured fit strictly BELOW gate -> accepted
    below_gate = gate - 0.01
    result_below = generate_plausible_body_candidates(
        observation=obs,
        profile=profile,
        seeds=(seed,),
        priors=GolfPlausibilityPriors.default(),
        prior_strength=1.0,
        options=BodyCandidateOptions(
            n_nullspace_proposals=2,
            measured_observation_fit_m=below_gate,
        ),
    )
    for cand in result_below.candidates:
        assert cand.accepted is True
        assert cand.observation_fit_m == pytest.approx(below_gate)
        assert cand.rejection_reasons == ()
        assert cand.rejection_reasons == ()


def test_closure_fields_none_and_as_dict_serializes_as_null() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("constrained_upper_body_golfer")
    seed = _non_synthetic_seed(obs, profile)

    result = generate_plausible_body_candidates(
        observation=obs,
        profile=profile,
        seeds=(seed,),
        priors=GolfPlausibilityPriors.default(),
        prior_strength=1.0,
        options=BodyCandidateOptions(
            n_nullspace_proposals=2,
            measured_observation_fit_m=0.01,
        ),
    )
    assert result.candidates
    for cand in result.candidates:
        assert cand.closure_q_m is None
        assert cand.closure_v_m_s is None
        assert cand.closure_a_m_s2 is None
        assert cand.derivatives_computed is False

        data = cand.as_dict()
        assert data["closure_q_m"] is None
        assert data["closure_v_m_s"] is None
        assert data["closure_a_m_s2"] is None
        assert data["derivatives_computed"] is False

        serialized = json.dumps(data)
        loaded = json.loads(serialized)
        assert loaded["closure_q_m"] is None
        assert loaded["closure_v_m_s"] is None
        assert loaded["closure_a_m_s2"] is None
        assert loaded["derivatives_computed"] is False
        assert '"closure_q_m": null' in serialized
        assert '"closure_v_m_s": null' in serialized
        assert '"closure_a_m_s2": null' in serialized
