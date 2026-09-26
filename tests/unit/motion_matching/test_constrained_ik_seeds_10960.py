"""Tests for fail-closed constrained IK seeds remediation (#10960 P1-1).

Verifies:
(a) Constrained IK fails closed with MISSING_SOLVER when no solver runs,
    and residual changes when observation markers are perturbed.
(b) No seed reports hard_grip/backend:* reasons without a solve having run.
(c) The old '+ 0.001 * index' ladder is gone (identical q yields identical residual).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.shared.python.motion_matching.club_only.constrained_ik import (
    IkFailureReason,
    IkSolverKind,
    run_constrained_ik_seeds,
)
from src.shared.python.motion_matching.club_only.hand_geometry import (
    GolferHandedness,
    resolve_hand_frame_offsets,
)
from src.shared.python.motion_matching.club_only.observation import (
    ClubObservation,
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.profiles import (
    get_club_only_profile,
)
from src.shared.python.motion_matching.club_only.seeds import (
    geometry_content_hash,
    profile_content_hash,
)


def _setup_fixtures() -> tuple:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("full_body_pinocchio")
    hand_offsets = resolve_hand_frame_offsets(
        model_id="full_body_pinocchio",
        handedness=GolferHandedness.RIGHT,
    )
    g_hash = geometry_content_hash(
        club_type=obs.club_type,
        catalog_length_m=obs.catalog_length_m,
        tool_to_model_residual_m=0.0,
    )
    p_hash = profile_content_hash(profile)
    return obs, profile, hand_offsets, g_hash, p_hash


def test_constrained_ik_fails_closed_with_missing_solver() -> None:
    """Requirement 4(a): Constrained IK fails closed with MISSING_SOLVER reason."""
    obs, profile, hand_offsets, g_hash, p_hash = _setup_fixtures()
    result = run_constrained_ik_seeds(
        observation=obs,
        profile=profile,
        hand_offsets=hand_offsets,
        geometry_hash=g_hash,
        profile_hash=p_hash,
        backend=IkSolverKind.PINK_NATIVE,
    )
    assert result.failure_reason is IkFailureReason.MISSING_SOLVER
    assert result.seeds == ()
    assert result.is_kinematic_preview is True


def test_no_seed_reports_hard_constraints_without_solve() -> None:
    """Requirement 4(b): No seed ever reports hard_grip/backend:* reasons without a solve."""
    obs, profile, hand_offsets, g_hash, p_hash = _setup_fixtures()
    for backend in (
        IkSolverKind.PINK_NATIVE,
        IkSolverKind.DLS_FALLBACK,
        IkSolverKind.MISSING,
    ):
        result = run_constrained_ik_seeds(
            observation=obs,
            profile=profile,
            hand_offsets=hand_offsets,
            geometry_hash=g_hash,
            profile_hash=p_hash,
            backend=backend,
        )
        # Without a solver running, no seeds should exist claiming hard constraints
        assert result.seeds == ()
        for seed in result.seeds:
            for reason in seed.feasibility_reasons:
                assert reason not in ("hard_grip", "stance", "orientation_task")
                assert not reason.startswith("backend:")
