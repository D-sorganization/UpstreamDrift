"""Unit tests verifying fast-matching fit and closure fail closed (#10960 P1-3).

Acceptance criteria:
(a) Shifting observation by 5 cm changes observation_fit_m, OR the branch fails closed.
(b) No branch is feasible when fit/closure were not computed.
(c) The 0.002*||q|| and 1e-3*||q|| formulas no longer appear (assert behaviour).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.fast_matching import (
    BranchScore,
    FastMatchOptions,
    MatchPreset,
    StartStrategy,
    run_fast_club_match,
    score_branch,
)
from src.shared.python.motion_matching.club_only.observation import (
    ClubObservation,
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.profiles import get_club_only_profile
from src.shared.python.motion_matching.club_only.seeds import (
    CandidateSeed,
    geometry_content_hash,
    profile_content_hash,
)

pytestmark = pytest.mark.unit


def _build_test_seed(
    *,
    obs: ClubObservation,
    model_id: str,
    q: np.ndarray,
    geometry_hash: str,
    profile_hash: str,
) -> CandidateSeed:
    time_arr = np.asarray(obs.native_time_s, dtype=np.float64)
    timestamps = time_arr.copy()
    return CandidateSeed(
        seed_id=f"seed-{obs.trial_id}",
        trial_id=obs.trial_id,
        model_id=model_id,
        source="retrieval",
        q=q,
        body_configuration_hash="body-hash-10960",
        observed_residual_m=0.02,
        prior_score=0.8,
        feasibility_reasons=("synthetic_seed",),
        timestamps_s=timestamps,
        geometry_hash=geometry_hash,
        profile_hash=profile_hash,
    )


@pytest.mark.unit
def test_observation_shift_or_fail_closed() -> None:
    """Acceptance criterion (a): shift by 5 cm changes fit OR fails closed."""
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("driven_double_pendulum")
    g_hash = geometry_content_hash(
        club_type=obs.club_type,
        catalog_length_m=obs.catalog_length_m,
        tool_to_model_residual_m=0.0,
    )
    p_hash = profile_content_hash(profile)
    q0 = np.array([0.05, -0.02, 0.01, 0.0])
    seed = _build_test_seed(
        obs=obs,
        model_id=profile.model_id,
        q=q0,
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )

    result_unshifted = run_fast_club_match(
        observation=obs,
        profile=profile,
        seed=seed,
        geometry_hash=g_hash,
        profile_hash=p_hash,
        options=FastMatchOptions(preset=MatchPreset.FAST_PREVIEW, n_branches=2),
    )

    # Shift observation by 5 cm in world z
    shifted_mid = np.asarray(obs.mid_hands_xyz, dtype=np.float64) + np.array(
        [0.0, 0.0, 0.05]
    )
    import dataclasses

    shifted_obs = dataclasses.replace(obs, mid_hands_xyz=shifted_mid)

    result_shifted = run_fast_club_match(
        observation=shifted_obs,
        profile=profile,
        seed=seed,
        geometry_hash=g_hash,
        profile_hash=p_hash,
        options=FastMatchOptions(preset=MatchPreset.FAST_PREVIEW, n_branches=2),
    )

    # Criterion (a): Either fit changes with shifted observation, OR branches fail closed.
    all_branches = list(result_unshifted.rejected) + list(result_unshifted.pareto)
    assert len(all_branches) > 0
    for branch in all_branches:
        if branch.observation_fit_m is not None:
            # If computed, shifting observation by 5 cm must have changed fit
            shifted_branch = next(
                b
                for b in (list(result_shifted.rejected) + list(result_shifted.pareto))
                if b.branch_id == branch.branch_id
            )
            assert shifted_branch.observation_fit_m is not None
            assert shifted_branch.observation_fit_m != branch.observation_fit_m
        else:
            # Otherwise fail closed: observation_fit_m is None and feasible is False
            assert branch.feasible is False
            assert "observation_fit_not_computed" in branch.rejection_reasons


@pytest.mark.unit
def test_no_branch_feasible_when_fit_closure_not_computed() -> None:
    """Acceptance criterion (b): no branch is feasible when uncomputed."""
    profile = get_club_only_profile("driven_double_pendulum")
    q = np.array([0.02, -0.01, 0.005, 0.0])

    # Uncomputed fit and closure must fail closed
    score_uncomputed = score_branch(
        branch_id="test_uncomputed",
        start_strategy=StartStrategy.RETRIEVAL,
        q=q,
        seed_residual_m=0.01,
        profile=profile,
        observation_fit_m=None,
        closure_m=None,
    )
    assert score_uncomputed.feasible is False
    assert score_uncomputed.observation_fit_m is None
    assert score_uncomputed.closure_m is None
    assert "observation_fit_not_computed" in score_uncomputed.rejection_reasons
    assert "closure_not_computed" in score_uncomputed.rejection_reasons

    # BranchScore postcondition invariant: cannot be feasible with None fit/closure
    with pytest.raises(
        ValueError, match="cannot be feasible when fit or closure is None"
    ):
        BranchScore(
            branch_id="invalid",
            start_strategy=StartStrategy.COLD,
            q=q,
            observation_fit_m=None,
            effort=0.0,
            closure_m=None,
            feasible=True,
            rejection_reasons=(),
            configuration_hash="hash",
        )


@pytest.mark.unit
def test_q_norm_formulas_eliminated() -> None:
    """Acceptance criterion (c): 0.002*||q|| and 1e-3*||q|| formulas do not appear."""
    profile = get_club_only_profile("driven_double_pendulum")
    test_qs = [
        np.array([3.0, 4.0, 0.0]),  # ||q|| = 5.0
        np.zeros(4),  # ||q|| = 0.0
        np.array([0.1, -0.2, 0.3, 0.4]),
    ]
    seed_residual = 0.02

    for q in test_qs:
        norm_q = float(np.linalg.norm(q))
        fabricated_fit = seed_residual + 0.002 * norm_q
        fabricated_closure = 1.0e-4 + 1.0e-3 * norm_q

        scored = score_branch(
            branch_id="formula_check",
            start_strategy=StartStrategy.RETRIEVAL,
            q=q,
            seed_residual_m=seed_residual,
            profile=profile,
        )

        # Behaviour assertion: values must not match the fabricated formulas
        assert scored.observation_fit_m != fabricated_fit
        assert scored.closure_m != fabricated_closure
        assert scored.observation_fit_m is None
        assert scored.closure_m is None
        assert scored.feasible is False
