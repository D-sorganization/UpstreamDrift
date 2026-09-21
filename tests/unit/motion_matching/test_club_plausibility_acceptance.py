"""CO-02 golf plausibility priors, ambiguity and club-only acceptance (#10606)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.acceptance import AcceptanceGates, Horizon
from src.shared.python.motion_matching.club_only.ambiguity import (
    AmbiguityStatus,
    CandidateScore,
    assess_ambiguity,
)
from src.shared.python.motion_matching.club_only.acceptance import (
    ClubOnlyResidualReport,
    ClubOnlyStatuses,
    evaluate_club_only_acceptance,
    normalized_position_error,
    normalized_orientation_error,
)
from src.shared.python.motion_matching.club_only.priors import (
    GolfPlausibilityPriors,
    PriorAssumption,
)
from src.shared.python.motion_matching.club_only.profiles import (
    PROFILE_SCHEMA,
    ObjectiveRule,
    build_roster_profiles,
    get_club_only_profile,
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
    / "club_plausibility_acceptance.json"
)


def _candidate(
    *,
    candidate_id: str,
    measured_residual: float,
    prior_score: float,
    body_hash: str,
    closure_m: float | None = 0.001,
    contact_feasible: bool = True,
    synthetic_force_claim: bool = False,
) -> CandidateScore:
    return CandidateScore(
        candidate_id=candidate_id,
        measured_residual_m=measured_residual,
        prior_score=prior_score,
        body_configuration_hash=body_hash,
        closure_residual_m=closure_m,
        contact_feasible=contact_feasible,
        claims_force_measurement=synthetic_force_claim,
    )


def test_roster_profiles_cover_every_registered_model() -> None:
    init_default_registry()
    models = list_golf_models()
    roster = build_roster_profiles()
    assert len(models) >= 15
    assert set(roster) == {m.model_id for m in models}
    for model in models:
        profile = roster[model.model_id]
        assert profile.model_id == model.model_id
        assert profile.topology == model.topology.value
        assert profile.schema == PROFILE_SCHEMA
        assert profile.objective_rule in (
            ObjectiveRule.LEXICOGRAPHIC_MEASURED_THEN_PRIOR,
            ObjectiveRule.PARETO_MEASURED_PRIOR_RUNTIME,
        )


def test_identical_club_path_distinct_bodies_stays_ambiguous() -> None:
    profile = get_club_only_profile("driven_double_pendulum")
    candidates = (
        _candidate(
            candidate_id="a",
            measured_residual=0.01,
            prior_score=0.9,
            body_hash="body-alpha",
        ),
        _candidate(
            candidate_id="b",
            measured_residual=0.01,
            prior_score=0.85,
            body_hash="body-beta",
        ),
    )
    verdict = assess_ambiguity(candidates, profile)
    assert verdict.status is AmbiguityStatus.AMBIGUOUS
    assert verdict.required_diverse_candidates >= 2
    assert verdict.retained_candidate_ids == frozenset({"a", "b"})


def test_prior_score_cannot_erase_failed_measured_residual() -> None:
    profile = get_club_only_profile("driven_triple_pendulum")
    residual = ClubOnlyResidualReport(
        grip_position_rmse_m=0.25,
        face_position_rmse_m=0.25,
        grip_orientation_rmse_rad=None,
        face_orientation_rmse_rad=None,
        native_coverage_fraction=0.95,
        speed_error_m_s=1.0,
        phase_error_s=0.01,
        unweighted_physical={"closure_residual_m": 0.001},
    )
    candidates = (
        _candidate(
            candidate_id="prior-hero",
            measured_residual=0.25,
            prior_score=0.99,
            body_hash="body-x",
        ),
    )
    verdict = evaluate_club_only_acceptance(
        residual,
        candidates,
        profile,
        body_markers_present=False,
        force_labels_synthetic=True,
    )
    assert verdict.measured_accepted is False
    assert verdict.overall_accepted is False
    assert any(
        g.name == "measured_club_residual" and not g.passed for g in verdict.gates
    )


def test_missing_body_data_never_passes_body_marker_gate() -> None:
    profile = get_club_only_profile("full_body_mujoco")
    residual = ClubOnlyResidualReport(
        grip_position_rmse_m=0.02,
        face_position_rmse_m=0.02,
        grip_orientation_rmse_rad=0.05,
        face_orientation_rmse_rad=0.05,
        native_coverage_fraction=0.95,
        speed_error_m_s=0.5,
        phase_error_s=0.005,
        unweighted_physical={},
    )
    candidates = (
        _candidate(
            candidate_id="solo",
            measured_residual=0.02,
            prior_score=0.5,
            body_hash="unknown",
        ),
    )
    verdict = evaluate_club_only_acceptance(
        residual,
        candidates,
        profile,
        body_markers_present=False,
        force_labels_synthetic=False,
    )
    body_gates = [g for g in verdict.gates if "body_marker" in g.name]
    assert body_gates
    assert all(not g.passed for g in body_gates)
    assert verdict.statuses.scientific != "qualified"


def test_no_force_measurement_claim_from_synthetic_labels() -> None:
    profile = get_club_only_profile("constrained_upper_body_golfer")
    candidates = (
        _candidate(
            candidate_id="synth-force",
            measured_residual=0.02,
            prior_score=0.7,
            body_hash="body-y",
            synthetic_force_claim=True,
        ),
    )
    residual = ClubOnlyResidualReport(
        grip_position_rmse_m=0.02,
        face_position_rmse_m=0.02,
        grip_orientation_rmse_rad=0.04,
        face_orientation_rmse_rad=None,
        native_coverage_fraction=0.9,
        speed_error_m_s=None,
        phase_error_s=None,
        unweighted_physical={},
    )
    verdict = evaluate_club_only_acceptance(
        residual,
        candidates,
        profile,
        body_markers_present=False,
        force_labels_synthetic=True,
    )
    assert verdict.overall_accepted is False
    assert any("force" in g.name and not g.passed for g in verdict.gates)


def test_infeasible_contact_or_closure_fails() -> None:
    profile = get_club_only_profile("driven_double_pendulum")
    candidates = (
        _candidate(
            candidate_id="bad-contact",
            measured_residual=0.02,
            prior_score=0.8,
            body_hash="body-z",
            closure_m=0.05,
            contact_feasible=False,
        ),
    )
    residual = ClubOnlyResidualReport(
        grip_position_rmse_m=0.02,
        face_position_rmse_m=0.02,
        grip_orientation_rmse_rad=None,
        face_orientation_rmse_rad=None,
        native_coverage_fraction=0.9,
        speed_error_m_s=None,
        phase_error_s=None,
        unweighted_physical={"closure_residual_m": 0.05},
    )
    verdict = evaluate_club_only_acceptance(
        residual,
        candidates,
        profile,
        body_markers_present=False,
        force_labels_synthetic=False,
    )
    assert verdict.overall_accepted is False
    assert any(
        g.name in {"closure_residual_m", "contact_feasibility"} and not g.passed
        for g in verdict.gates
    )


def test_reduced_model_unsupported_orientation_is_limitation_not_gate() -> None:
    profile = get_club_only_profile("driven_double_pendulum")
    assert "face_orientation" in profile.observation.unsupported_components
    assert profile.observation.max_face_orientation_rmse_rad is None
    assert any("orientation" in note.lower() for note in profile.limitations)
    residual = ClubOnlyResidualReport(
        grip_position_rmse_m=0.03,
        face_position_rmse_m=0.04,
        grip_orientation_rmse_rad=0.2,
        face_orientation_rmse_rad=0.4,
        native_coverage_fraction=0.95,
        speed_error_m_s=2.0,
        phase_error_s=0.02,
        unweighted_physical={"closure_residual_m": 0.001},
    )
    candidates = (
        _candidate(
            candidate_id="planar",
            measured_residual=0.04,
            prior_score=0.6,
            body_hash="planar-a",
        ),
        _candidate(
            candidate_id="planar-b",
            measured_residual=0.04,
            prior_score=0.55,
            body_hash="planar-b",
        ),
    )
    verdict = evaluate_club_only_acceptance(
        residual,
        candidates,
        profile,
        body_markers_present=False,
        force_labels_synthetic=False,
    )
    orient_gates = [g for g in verdict.gates if "orientation" in g.name]
    assert not orient_gates
    assert verdict.limitations


def test_statuses_remain_separated() -> None:
    profile = get_club_only_profile("full_body_pinocchio")
    residual = ClubOnlyResidualReport(
        grip_position_rmse_m=0.02,
        face_position_rmse_m=0.02,
        grip_orientation_rmse_rad=0.03,
        face_orientation_rmse_rad=0.03,
        native_coverage_fraction=0.95,
        speed_error_m_s=0.4,
        phase_error_s=0.004,
        unweighted_physical={"closure_residual_m": 0.002},
    )
    candidates = (
        _candidate(
            candidate_id="k1",
            measured_residual=0.02,
            prior_score=0.7,
            body_hash="b1",
        ),
        _candidate(
            candidate_id="k2",
            measured_residual=0.021,
            prior_score=0.65,
            body_hash="b2",
        ),
    )
    verdict = evaluate_club_only_acceptance(
        residual,
        candidates,
        profile,
        body_markers_present=True,
        force_labels_synthetic=False,
        torque_replay_validated=False,
        kinematic_preview_ok=True,
    )
    assert isinstance(verdict.statuses, ClubOnlyStatuses)
    assert verdict.statuses.kinematic_preview == "passed"
    assert verdict.statuses.torque_replay == "unevaluated"
    assert verdict.statuses.scientific in {"unverified", "disqualified", "qualified"}
    assert verdict.statuses.product == "exploratory"


def test_normalized_errors_use_uncertainty_scale() -> None:
    pos = normalized_position_error(
        np.array([[0.0, 0.0, 0.0], [0.002, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        sigma_m=0.001,
    )
    assert pos == pytest.approx(np.sqrt(0.5 * (0.0 + 4.0)))
    half = 0.05 / 2.0
    ori = normalized_orientation_error(
        np.array([[1.0, 0.0, 0.0, 0.0], [np.cos(half), 0.0, 0.0, np.sin(half)]]),
        np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        sigma_rad=0.05,
    )
    assert ori == pytest.approx(np.sqrt(0.5 * (0.0 + 1.0)), rel=1e-3)


def test_priors_are_named_assumptions_not_measured_truth() -> None:
    priors = GolfPlausibilityPriors.default()
    assert priors.assumptions
    for item in priors.assumptions:
        assert isinstance(item, PriorAssumption)
        assert item.name
        assert item.source
        assert item.is_measured_truth is False
    score = priors.score_posture(
        joint_rates_rad_s=np.array([1.0, 2.0]),
        joint_accels_rad_s2=np.array([5.0, 6.0]),
        grip_closure_m=0.002,
        effort=0.3,
    )
    assert 0.0 <= score <= 1.0


def test_g3_full_body_gates_remain_unchanged() -> None:
    gates = AcceptanceGates()
    assert gates.g3_whole_driver_rmse_m == pytest.approx(0.060)
    assert gates.g3_whole_iron_rmse_m == pytest.approx(0.095)
    assert gates.g3_club_rmse_m == pytest.approx(0.100)
    profile = get_club_only_profile("full_body_mujoco")
    assert profile.preserves_full_body_g3 is True
    assert Horizon.G3.value == "G3"


def test_evidence_receipt_matches_roster() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert payload["schema"] == PROFILE_SCHEMA
    assert payload["governing_issue"] == 10606
    roster = build_roster_profiles()
    assert set(payload["models"]) == set(roster)
    for model_id, entry in payload["models"].items():
        assert entry["topology"] == roster[model_id].topology
        assert "supported_observables" in entry
        assert "limitations" in entry


def test_dbc_rejects_nonfinite_and_survives_python_dash_o() -> None:
    with pytest.raises(ValueError, match="finite"):
        ClubOnlyResidualReport(
            grip_position_rmse_m=float("nan"),
            face_position_rmse_m=0.01,
            grip_orientation_rmse_rad=None,
            face_orientation_rmse_rad=None,
            native_coverage_fraction=0.9,
            speed_error_m_s=None,
            phase_error_s=None,
            unweighted_physical={},
        )
    with pytest.raises(ValueError, match="coverage"):
        ClubOnlyResidualReport(
            grip_position_rmse_m=0.01,
            face_position_rmse_m=0.01,
            grip_orientation_rmse_rad=None,
            face_orientation_rmse_rad=None,
            native_coverage_fraction=1.5,
            speed_error_m_s=None,
            phase_error_s=None,
            unweighted_physical={},
        )
    with pytest.raises(ValueError, match="sigma"):
        normalized_position_error(
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            sigma_m=-0.1,
        )
