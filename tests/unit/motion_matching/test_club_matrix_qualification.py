"""CO-08 qualify the club-only matrix and plausibility tradeoffs (#10612)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.matrix_qualification import (
    MATRIX_SCHEMA,
    ExportedCandidatePackage,
    MatrixQualificationReport,
    OrientationClaim,
    PhaseCoverage,
    WithheldBodyComparison,
    assert_honest_native_status,
    assert_independent_evaluation,
    assert_no_hidden_body_labels,
    assert_no_target_resets,
    assert_orientation_semantics,
    assert_phase_coverage,
    build_matrix_qualification_report,
    compare_common_observables,
    evaluate_withheld_body_experiment,
    matrix_qualification_evidence_payload,
    physical_overrides_visual,
    validate_package_integrity,
)
from src.shared.python.motion_matching.club_only.observation import (
    ComponentStatus,
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.profiles import (
    build_roster_profiles,
    get_club_only_profile,
)
from src.shared.python.motion_matching.club_only.seeds import (
    geometry_content_hash,
    profile_content_hash,
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
    / "club_matrix_qualification.json"
)


def _geometry_hash(obs) -> str:
    return geometry_content_hash(
        club_type=obs.club_type,
        catalog_length_m=obs.catalog_length_m,
        tool_to_model_residual_m=0.0,
    )


def _times(n: int = 21, dt: float = 0.01) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * dt


def _package(
    *,
    trial_id: str = "TW_wiffle",
    model_id: str = "driven_double_pendulum",
    geometry_hash: str | None = None,
    profile_hash: str | None = None,
    package_hash: str | None = None,
    grip_position_rmse_m: float = 0.02,
    face_position_rmse_m: float = 0.025,
    original_3d_rmse_m: float = 0.03,
    in_plane_rmse_m: float = 0.02,
    out_of_plane_rmse_m: float = 0.015,
    grip_orientation_rmse_rad: float | None = None,
    face_orientation_rmse_rad: float | None = None,
    native_coverage_fraction: float = 1.0,
    phase_error_s: float | None = 0.01,
    closure_residual_m: float = 0.001,
    contact_feasible: bool = True,
    used_measured_state_reset: bool = False,
    body_labels_hidden: bool = False,
    body_marker_status: str = "withheld",
    orientation_claims: tuple[OrientationClaim, ...] | None = None,
    phases: PhaseCoverage | None = None,
    fitting_prior_trial_ids: tuple[str, ...] = (),
    native_g1_pass: bool = False,
    claims_native_qualification: bool = False,
    qualification_blockers: tuple[str, ...] = (
        "native_g1_qualification_requires_desk_native_receipt",
        "software_contract_matrix_is_not_native_evidence",
    ),
    visual_attractiveness: float = 0.9,
    physical_failed: bool = False,
    timestamps_s: np.ndarray | None = None,
) -> ExportedCandidatePackage:
    obs = build_calibrated_observation_fixture(trial_id)
    profile = get_club_only_profile(model_id)
    g_hash = geometry_hash if geometry_hash is not None else _geometry_hash(obs)
    p_hash = profile_hash if profile_hash is not None else profile_content_hash(profile)
    times = timestamps_s if timestamps_s is not None else obs.native_time_s.copy()
    if phases is None:
        phases = PhaseCoverage(
            address_present=True,
            top_present=True,
            impact_present=True,
            finish_present=True,
            phase_labels=("A", "T", "I", "F"),
        )
    if orientation_claims is None:
        orientation_claims = (
            OrientationClaim(
                component="mid_hands_orientation",
                declared_status=ComponentStatus.MEASURED,
                scored_as_measured=True,
            ),
            OrientationClaim(
                component="face_orientation",
                declared_status=ComponentStatus.DERIVED,
                scored_as_measured=False,
            ),
        )
    pkg = ExportedCandidatePackage(
        candidate_id=f"cand:{trial_id}:{model_id}",
        trial_id=trial_id,
        model_id=model_id,
        geometry_hash=g_hash,
        profile_hash=p_hash,
        timestamps_s=times,
        q0=np.zeros(4, dtype=np.float64),
        v0=np.zeros(4, dtype=np.float64),
        grip_position_rmse_m=grip_position_rmse_m,
        face_position_rmse_m=face_position_rmse_m,
        original_3d_rmse_m=original_3d_rmse_m,
        in_plane_rmse_m=in_plane_rmse_m,
        out_of_plane_rmse_m=out_of_plane_rmse_m,
        grip_orientation_rmse_rad=grip_orientation_rmse_rad,
        face_orientation_rmse_rad=face_orientation_rmse_rad,
        native_coverage_fraction=native_coverage_fraction,
        phase_error_s=phase_error_s,
        closure_residual_m=closure_residual_m,
        contact_feasible=contact_feasible,
        used_measured_state_reset=used_measured_state_reset,
        body_labels_hidden=body_labels_hidden,
        body_marker_status=body_marker_status,
        orientation_claims=orientation_claims,
        phases=phases,
        fitting_prior_trial_ids=fitting_prior_trial_ids,
        unsupported_components=frozenset(profile.observation.unsupported_components),
        supported_observables=frozenset(profile.observation.supported_observables),
        native_g1_pass=native_g1_pass,
        claims_native_qualification=claims_native_qualification,
        qualification_blockers=qualification_blockers,
        visual_attractiveness=visual_attractiveness,
        physical_failed=physical_failed,
        package_content_hash="",
    )
    # Recompute canonical hash after construction helpers fill it, or override.
    if package_hash is not None:
        object.__setattr__(pkg, "package_content_hash", package_hash)
    return pkg


def test_tampered_package_or_profile_fails() -> None:
    pkg = _package()
    profile = get_club_only_profile(pkg.model_id)
    validate_package_integrity(pkg, profile=profile, geometry_hash=pkg.geometry_hash)

    tampered = _package(package_hash="deadbeef" * 8)
    with pytest.raises(ValueError, match="tampered|package_content_hash|integrity"):
        validate_package_integrity(
            tampered, profile=profile, geometry_hash=tampered.geometry_hash
        )

    wrong_profile = get_club_only_profile("driven_triple_pendulum")
    with pytest.raises(ValueError, match="profile"):
        validate_package_integrity(
            pkg, profile=wrong_profile, geometry_hash=pkg.geometry_hash
        )


def test_hidden_body_labels_fail() -> None:
    with pytest.raises(ValueError, match="hidden body"):
        assert_no_hidden_body_labels(_package(body_labels_hidden=True))
    assert_no_hidden_body_labels(_package(body_labels_hidden=False))


def test_duplicate_trial_leakage_fails() -> None:
    pkg = _package(
        trial_id="TW_wiffle",
        fitting_prior_trial_ids=("TW_ProV1", "TW_wiffle"),
    )
    with pytest.raises(ValueError, match="trial leakage|independent evaluation"):
        assert_independent_evaluation(pkg)
    clean = _package(
        trial_id="TW_wiffle",
        fitting_prior_trial_ids=("TW_ProV1", "GW_wiffle"),
    )
    assert_independent_evaluation(clean)


def test_missing_phase_fails() -> None:
    incomplete = PhaseCoverage(
        address_present=True,
        top_present=True,
        impact_present=False,
        finish_present=True,
        phase_labels=("A", "T", "F"),
    )
    with pytest.raises(ValueError, match="phase|impact"):
        assert_phase_coverage(_package(phases=incomplete))
    assert_phase_coverage(_package())


def test_inferred_rotation_as_measured_fails() -> None:
    lied = (
        OrientationClaim(
            component="face_orientation",
            declared_status=ComponentStatus.DERIVED,
            scored_as_measured=True,
        ),
    )
    with pytest.raises(ValueError, match="inferred|derived.*measured"):
        assert_orientation_semantics(_package(orientation_claims=lied))
    assert_orientation_semantics(_package())


def test_target_resets_fail() -> None:
    with pytest.raises(ValueError, match="target reset|measured.state"):
        assert_no_target_resets(_package(used_measured_state_reset=True))
    assert_no_target_resets(_package(used_measured_state_reset=False))


def test_mismatched_geometry_fails() -> None:
    pkg = _package()
    profile = get_club_only_profile(pkg.model_id)
    with pytest.raises(ValueError, match="geometry"):
        validate_package_integrity(pkg, profile=profile, geometry_hash="cafebabe" * 8)


def test_false_native_status_fails() -> None:
    with pytest.raises(ValueError, match="native"):
        assert_honest_native_status(
            _package(
                native_g1_pass=True,
                claims_native_qualification=True,
                qualification_blockers=(),
            )
        )
    with pytest.raises(ValueError, match="native"):
        assert_honest_native_status(
            _package(
                native_g1_pass=True,
                claims_native_qualification=False,
                qualification_blockers=(
                    "native_g1_qualification_requires_desk_native_receipt",
                ),
            )
        )
    assert_honest_native_status(_package())


def test_withheld_body_disagreement_does_not_invalidate_club_only() -> None:
    pkg = _package(body_marker_status="withheld")
    comparison = WithheldBodyComparison(
        body_markers_used_in_fit=False,
        optional_body_rmse_m=0.25,
        club_only_feasible=True,
        claims_true_body_identified=False,
    )
    verdict = evaluate_withheld_body_experiment(pkg, comparison)
    assert verdict.club_only_accepted is True
    assert verdict.body_disagreement_invalidates is False
    assert verdict.identifies_true_body is False

    with pytest.raises(ValueError, match="true body|identify"):
        evaluate_withheld_body_experiment(
            pkg,
            WithheldBodyComparison(
                body_markers_used_in_fit=False,
                optional_body_rmse_m=0.25,
                club_only_feasible=True,
                claims_true_body_identified=True,
            ),
        )

    with pytest.raises(ValueError, match="body markers"):
        evaluate_withheld_body_experiment(
            pkg,
            WithheldBodyComparison(
                body_markers_used_in_fit=True,
                optional_body_rmse_m=0.01,
                club_only_feasible=True,
                claims_true_body_identified=False,
            ),
        )


def test_visual_cannot_override_physical_failure() -> None:
    failed = _package(visual_attractiveness=0.99, physical_failed=True)
    with pytest.raises(ValueError, match="visual|physical"):
        physical_overrides_visual(failed)
    ok = _package(visual_attractiveness=0.2, physical_failed=False)
    assert physical_overrides_visual(ok) is True


def test_common_observables_compared_across_complexities() -> None:
    init_default_registry()
    packages = (
        _package(model_id="driven_double_pendulum"),
        _package(model_id="driven_triple_pendulum"),
        _package(model_id="constrained_upper_body_golfer"),
    )
    common = compare_common_observables(packages)
    assert "grip_position" in common
    assert "face_position" in common
    assert "native_coverage" in common
    # Face orientation is unsupported on planar models — not a common score lane.
    assert "face_orientation" not in common


def test_real_matrix_receipt_validation() -> None:
    init_default_registry()
    report = build_matrix_qualification_report()
    assert isinstance(report, MatrixQualificationReport)
    assert report.schema == MATRIX_SCHEMA
    assert report.governing_issue == 10612
    assert report.claims_native_qualification is False
    assert report.native_g1_pass is False
    assert report.qualification_blockers

    roster = {m.model_id for m in list_golf_models()}
    cell_keys = {(c.model_id, c.trial_id) for c in report.cells}
    for model_id in roster:
        for trial_id in CANONICAL_TRIAL_SHEETS:
            assert (model_id, trial_id) in cell_keys

    # Every cell publishes frozen CO-02 gate outcomes or an explicit blocker.
    for cell in report.cells:
        assert cell.status in {
            "scored",
            "rejected",
            "unsupported",
            "missing_runtime",
            "unqualified",
        }
        if cell.status == "scored":
            assert cell.original_3d_rmse_m is not None
            assert cell.unsupported_components is not None
            assert cell.gate_failures is not None
            assert cell.common_observables
        else:
            assert cell.blocker

    payload = matrix_qualification_evidence_payload(report)
    assert payload["schema"] == MATRIX_SCHEMA
    assert payload["governing_issue"] == 10612
    assert payload["native_g1_pass"] is False
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert len(digest) == 64


def test_evidence_fixture_matches_schema_when_present() -> None:
    if not EVIDENCE.is_file():
        pytest.skip("evidence written after GREEN implementation")
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert data["schema"] == MATRIX_SCHEMA
    assert data["governing_issue"] == 10612
    assert data["native_g1_pass"] is False
    assert data["claims_native_qualification"] is False
    assert data["qualification_blockers"]
    assert "cells" in data or "models" in data
    assert data.get("comparison_rule") == "common_observables_not_weighted_objectives"
