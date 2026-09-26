"""CO-03 retrieval and constrained-IK starting guesses (#10607)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.hand_geometry import (
    GolferHandedness,
    ModelHandFrameOffsets,
    resolve_hand_frame_offsets,
)
from src.shared.python.motion_matching.club_only.constrained_ik import (
    ConstrainedIkRequest,
    IkBackendCapabilities,
    IkFailureReason,
    IkSolverKind,
    UnsupportedConstraintError,
    assert_backend_supports,
    capabilities_for,
    generate_posture_branches,
    run_constrained_ik_seeds,
)
from src.shared.python.motion_matching.club_only.observation import (
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.profiles import get_club_only_profile
from src.shared.python.motion_matching.club_only.retrieval import (
    LibraryEntry,
    ObservableClubDescriptor,
    RigidPlacement,
    build_observable_descriptor,
    retrieve_starting_seeds,
)
from src.shared.python.motion_matching.club_only.seeds import (
    SEED_SCHEMA,
    CandidateSeed,
    SeedCache,
    StartingGuessReport,
    build_starting_guess_report,
    geometry_content_hash,
    profile_content_hash,
    starting_guess_evidence_payload,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CLUB_DATA = REPO_ROOT / "data" / "Club_Data.xlsx"
EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "club_only_matching"
    / "evidence"
    / "club_starting_guesses.json"
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
    source: str,
    q: np.ndarray,
    body_hash: str,
    geometry_hash: str,
    profile_hash: str,
    residual: float = 0.01,
) -> CandidateSeed:
    return CandidateSeed(
        seed_id=seed_id,
        trial_id=trial_id,
        model_id=model_id,
        source=source,
        q=q,
        body_configuration_hash=body_hash,
        observed_residual_m=residual,
        prior_score=0.7,
        feasibility_reasons=("synthetic_feasible",),
        timestamps_s=np.array([0.0, 1.0 / 240.0]),
        geometry_hash=geometry_hash,
        profile_hash=profile_hash,
    )


def test_known_hand_offsets_respect_handedness() -> None:
    right = resolve_hand_frame_offsets(
        model_id="full_body_pinocchio",
        handedness=GolferHandedness.RIGHT,
    )
    left = resolve_hand_frame_offsets(
        model_id="full_body_pinocchio",
        handedness=GolferHandedness.LEFT,
    )
    assert isinstance(right, ModelHandFrameOffsets)
    assert right.lead_frame_name == "hand_left_tip"
    assert right.trail_frame_name == "hand_right_tip"
    assert left.lead_frame_name == "hand_right_tip"
    assert left.trail_frame_name == "hand_left_tip"
    assert right.lead_hand_offset_m == pytest.approx(left.lead_hand_offset_m)
    assert right.trail_hand_offset_m == pytest.approx(left.trail_hand_offset_m)
    assert right.lead_hand_offset_m != right.trail_hand_offset_m
    assert np.isfinite(right.lead_hand_offset_m)
    assert np.isfinite(right.trail_hand_offset_m)


def test_seed_cache_invalidated_by_geometry_or_profile_change() -> None:
    obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("driven_double_pendulum")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)
    cache = SeedCache()
    seed = _seed(
        seed_id="s1",
        trial_id=obs.trial_id,
        model_id=profile.model_id,
        source="retrieval",
        q=np.zeros(4),
        body_hash="body-a",
        geometry_hash=g_hash,
        profile_hash=p_hash,
    )
    cache.put(seed)
    assert (
        cache.get(obs.trial_id, profile.model_id, "retrieval", g_hash, p_hash) is seed
    )

    other_geo = geometry_content_hash(
        club_type=obs.club_type,
        catalog_length_m=obs.catalog_length_m + 0.01,
        tool_to_model_residual_m=0.0,
    )
    assert (
        cache.get(obs.trial_id, profile.model_id, "retrieval", other_geo, p_hash)
        is None
    )

    other_profile = get_club_only_profile("driven_triple_pendulum")
    other_p = profile_content_hash(other_profile)
    assert (
        cache.get(obs.trial_id, profile.model_id, "retrieval", g_hash, other_p) is None
    )


def test_singular_ik_missing_solver_and_unreachable_orientation() -> None:
    caps_missing = capabilities_for(IkSolverKind.MISSING)
    req = ConstrainedIkRequest(
        requires_orientation=True,
        requires_hard_grip=True,
        requires_stance=True,
        hand_offsets=resolve_hand_frame_offsets(
            model_id="full_body_pinocchio",
            handedness=GolferHandedness.RIGHT,
        ),
    )
    with pytest.raises(UnsupportedConstraintError, match="missing"):
        assert_backend_supports(caps_missing, req)

    caps_dls = capabilities_for(IkSolverKind.DLS_FALLBACK)
    with pytest.raises(UnsupportedConstraintError, match="orientation"):
        assert_backend_supports(caps_dls, req)

    obs = build_calibrated_observation_fixture("TW_ProV1")
    profile = get_club_only_profile("full_body_pinocchio")
    g_hash = _geometry_hash(obs)
    p_hash = profile_content_hash(profile)

    singular = run_constrained_ik_seeds(
        observation=obs,
        profile=profile,
        hand_offsets=req.hand_offsets,
        geometry_hash=g_hash,
        profile_hash=p_hash,
        backend=IkSolverKind.PINK_NATIVE,
        simulate_failure=IkFailureReason.SINGULAR_JACOBIAN,
    )
    assert singular.seeds == ()
    assert singular.failure_reason is IkFailureReason.SINGULAR_JACOBIAN
    assert singular.is_kinematic_preview is True

    unreachable = run_constrained_ik_seeds(
        observation=obs,
        profile=profile,
        hand_offsets=req.hand_offsets,
        geometry_hash=g_hash,
        profile_hash=p_hash,
        backend=IkSolverKind.PINK_NATIVE,
        simulate_failure=IkFailureReason.UNREACHABLE_ORIENTATION,
    )
    assert unreachable.seeds == ()
    assert unreachable.failure_reason is IkFailureReason.UNREACHABLE_ORIENTATION


def test_alternative_body_poses_preserved() -> None:
    q0 = np.array([0.0, 0.1, -0.1, 0.0])
    branches = generate_posture_branches(q0, n_branches=3, amplitude_rad=0.05)
    assert len(branches) == 3
    hashes = {
        hashlib.sha256(np.asarray(q, dtype=np.float64).tobytes()).hexdigest()
        for q in branches
    }
    assert len(hashes) == 3
    for q in branches:
        assert q.shape == q0.shape
        assert np.all(np.isfinite(q))


def test_no_hidden_frame_alignment_or_time_warp() -> None:
    obs = build_calibrated_observation_fixture("GW_wiffle")
    ref_obs = build_calibrated_observation_fixture("TW_wiffle")
    profile = get_club_only_profile("reconstruction_golfer")
    g_hash = _geometry_hash(obs)
    desc = build_observable_descriptor(
        ref_obs,
        model_id=profile.model_id,
        geometry_hash=g_hash,
    )
    assert isinstance(desc, ObservableClubDescriptor)
    placement = RigidPlacement.identity()
    entry = LibraryEntry(
        reference_id="ref-tour-anchor",
        descriptor=desc,
        body_q0=np.array([0.0, 0.05, -0.02, 0.0]),
        rigid_placement=placement,
        source_clock_times_s=ref_obs.native_time_s.copy(),
        body_is_prior=True,
    )
    seeds = retrieve_starting_seeds(
        observation=obs,
        profile=profile,
        library=(entry,),
        geometry_hash=g_hash,
        profile_hash=profile_content_hash(profile),
        max_seeds=2,
    )
    assert seeds
    for seed in seeds:
        assert seed.source == "retrieval"
        assert seed.body_is_prior is True
        assert seed.placement is not None
        assert seed.placement.is_single_rigid is True
        np.testing.assert_allclose(seed.timestamps_s, obs.native_time_s)
        assert seed.claims_torque is False
        assert seed.claims_physiological_inference is False
        assert seed.is_kinematic_preview is True


def test_pink_versus_dls_capabilities_are_distinct() -> None:
    pink = capabilities_for(IkSolverKind.PINK_NATIVE)
    dls = capabilities_for(IkSolverKind.DLS_FALLBACK)
    assert isinstance(pink, IkBackendCapabilities)
    assert pink.supports_orientation_tasks is True
    assert pink.supports_hard_grip_closure is True
    assert pink.supports_stance_constraints is True
    assert dls.supports_orientation_tasks is False
    assert dls.supports_hard_grip_closure is False
    assert dls.supports_stance_constraints is False
    assert pink.kind != dls.kind


def test_four_trial_bounded_seeds_retrieval_and_ik_baselines() -> None:
    report = build_starting_guess_report(
        model_id="full_body_pinocchio",
        handedness=GolferHandedness.RIGHT,
        max_seeds_per_source=3,
    )
    assert isinstance(report, StartingGuessReport)
    assert report.schema == SEED_SCHEMA
    assert report.governing_issue == 10607
    assert set(report.trials) == set(CANONICAL_TRIAL_SHEETS)
    for trial_id, trial in report.trials.items():
        assert trial["trial_id"] == trial_id
        assert "retrieval" in trial["baselines"]
        assert "constrained_ik" in trial["baselines"]
        retrieval = trial["baselines"]["retrieval"]
        ik = trial["baselines"]["constrained_ik"]
        assert 1 <= len(retrieval["seed_ids"]) <= 3
        assert len(ik["seed_ids"]) == 0
        assert ik.get("failure_reason") == IkFailureReason.MISSING_SOLVER.value
        assert retrieval["is_kinematic_preview"] is True
        assert ik["is_kinematic_preview"] is True
        assert trial["claims_torque"] is False
        assert trial["claims_physiological_inference"] is False


def test_workbook_smoke_descriptor_uses_native_clock() -> None:
    if not CLUB_DATA.is_file():
        pytest.skip("Club_Data.xlsx not present")
    from src.shared.python.motion_matching.club_only.adapters import (
        club_target_to_observation,
    )
    from src.shared.python.motion_matching.club_target import AlignOptions
    from src.shared.python.motion_matching.loaders.excel import load_club_target_excel

    opts = AlignOptions(
        sample_rate_hz=200.0,
        simulation_time_s=0.1,
        time_alignment="impact",
        impact_target_t_s=0.05,
    )
    target = load_club_target_excel(CLUB_DATA, "TW_wiffle", opts)
    try:
        obs = club_target_to_observation(target, trial_id="TW_wiffle")
    except ValueError as exc:
        if "orientation" in str(exc).lower() or "quat" in str(exc).lower():
            pytest.skip(f"legacy workbook orientation unavailable: {exc}")
        raise
    desc = build_observable_descriptor(
        obs,
        model_id="full_body_pinocchio",
        geometry_hash=_geometry_hash(obs),
    )
    assert desc.sample_rate_hz == pytest.approx(obs.sample_rate_hz)
    assert desc.trial_id == "TW_wiffle"
    assert desc.duration_s > 0.0


def test_evidence_receipt_matches_schema() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert payload["schema"] == SEED_SCHEMA
    assert payload["governing_issue"] == 10607
    assert set(payload["trials"]) == set(CANONICAL_TRIAL_SHEETS)
    rebuilt = starting_guess_evidence_payload(
        build_starting_guess_report(
            model_id="full_body_pinocchio",
            handedness=GolferHandedness.RIGHT,
            max_seeds_per_source=3,
        )
    )
    assert rebuilt["schema"] == payload["schema"]
    assert set(rebuilt["trials"]) == set(payload["trials"])
