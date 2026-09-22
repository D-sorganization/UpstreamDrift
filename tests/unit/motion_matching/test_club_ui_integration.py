"""CO-09 integrate club-only matching into existing UI and results (#10613)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.club_only.fast_matching import (
    MatchCancelledError,
    MatchPreset,
)
from src.shared.python.motion_matching.club_only.observation import (
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.ui_integration import (
    UI_SCHEMA,
    ClubOnlySourceKind,
    ClubOnlyUiSession,
    ObservationRole,
    VerificationDisplayStatus,
    assert_unqualified_cannot_appear_verified,
    build_club_only_result_view,
    build_observed_inferred_legend,
    clone_club_only_session,
    create_club_only_session,
    import_club_only_workbook_catalog,
    keyboard_action_map,
    list_motion_matching_source_kinds,
    publish_club_only_ledger_row,
    run_club_only_ui_match,
    ui_integration_evidence_payload,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
    IDENTITY_SCHEMA,
)
from src.shared.python.motion_matching.ledger_schema import LedgerRow
from src.shared.python.workspace.results_browser import ResultFilter, ResultsBrowser

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "club_only_matching"
    / "evidence"
    / "club_ui_integration.json"
)


def test_source_kinds_include_club_only_excel() -> None:
    kinds = list_motion_matching_source_kinds()
    assert ClubOnlySourceKind.TOUR_AVERAGE in kinds
    assert ClubOnlySourceKind.CLUB_ONLY_EXCEL in kinds


def test_import_lists_four_unique_trials_with_conflicts_and_coverage() -> None:
    catalog = import_club_only_workbook_catalog(REPO_ROOT)
    assert catalog.schema_version == IDENTITY_SCHEMA
    assert len(catalog.unique_trials) == 4
    trial_ids = {trial.trial_id for trial in catalog.unique_trials}
    assert trial_ids == set(CANONICAL_TRIAL_SHEETS)
    assert any(trial.alias_sheets for trial in catalog.unique_trials)
    for trial in catalog.unique_trials:
        assert trial.sample_count > 0
        assert trial.clock_hz == pytest.approx(240.0)
        assert "butt_position" in trial.observation_coverage
        assert trial.unit_authority_note
        assert "inferred" not in trial.observation_coverage.values()
    assert catalog.conflicts
    assert any(
        "unit" in c.lower() or "declared" in c.lower() for c in catalog.conflicts
    )
    assert any("ball" in c.lower() for c in catalog.conflicts)


def test_unknown_units_and_ball_label_surface_as_named_conflicts() -> None:
    catalog = import_club_only_workbook_catalog(REPO_ROOT)
    joined = " | ".join(catalog.conflicts).lower()
    assert "cm" in joined or "centimetre" in joined or "unit" in joined
    assert "ball" in joined


def test_session_explains_inferred_body_and_exposes_budget_priors() -> None:
    session = create_club_only_session(
        trial_id="TW_wiffle",
        model_id="double_pendulum",
        preset=MatchPreset.FAST_PREVIEW,
        prior_choices={"pose_prior": "address_plausible"},
        geometry_choices={"handedness": "right"},
        user_edits={"notes": "keep this"},
    )
    assert session.source_kind is ClubOnlySourceKind.CLUB_ONLY_EXCEL
    assert "inferred" in session.body_motion_disclaimer.lower()
    assert "not measured" in session.body_motion_disclaimer.lower()
    assert session.computation_budget.max_evaluations > 0
    assert session.prior_choices["pose_prior"] == "address_plausible"
    assert session.geometry_choices["handedness"] == "right"
    assert session.neural_proposal_slot == "empty_provider"


def test_session_preset_name_delegates_without_deep_chain() -> None:
    session = create_club_only_session(
        trial_id="TW_wiffle",
        model_id="double_pendulum",
        preset=MatchPreset.FAST_PREVIEW,
    )
    assert session.preset_name() == MatchPreset.FAST_PREVIEW.value
    assert session.as_dict()["preset"] == session.preset_name()


def test_clone_preserves_user_edits() -> None:
    session = create_club_only_session(
        trial_id="GW_wiffle",
        model_id="triple_pendulum",
        preset=MatchPreset.VERIFIED_FIT,
        user_edits={"notes": "original", "budget_scale": 0.5},
    )
    cloned = clone_club_only_session(session)
    assert cloned.session_id != session.session_id
    assert cloned.user_edits == session.user_edits
    assert cloned.trial_id == session.trial_id
    assert cloned.preset is session.preset


def test_preview_and_verified_display_status_are_honest() -> None:
    observation = build_calibrated_observation_fixture("TW_ProV1")
    preview = run_club_only_ui_match(
        create_club_only_session(
            trial_id=observation.trial_id,
            model_id="double_pendulum",
            preset=MatchPreset.FAST_PREVIEW,
        ),
        observation=observation,
    )
    verified_attempt = run_club_only_ui_match(
        create_club_only_session(
            trial_id=observation.trial_id,
            model_id="double_pendulum",
            preset=MatchPreset.VERIFIED_FIT,
        ),
        observation=observation,
    )
    preview_view = build_club_only_result_view(preview)
    verified_view = build_club_only_result_view(verified_attempt)
    assert preview_view.display_status is VerificationDisplayStatus.PREVIEW
    assert verified_view.display_status is not VerificationDisplayStatus.NATIVE_VERIFIED
    assert verified_view.native_g1_pass is False
    assert verified_view.qualification_blockers
    assert_unqualified_cannot_appear_verified(verified_view)


def test_unqualified_cannot_appear_verified() -> None:
    with pytest.raises(ValueError, match="unqualified"):
        assert_unqualified_cannot_appear_verified(
            type(
                "Bad",
                (),
                {
                    "native_g1_pass": False,
                    "display_status": VerificationDisplayStatus.NATIVE_VERIFIED,
                    "qualification_blockers": ("blocker",),
                },
            )()
        )


def test_observed_versus_inferred_legend_and_trial_clock() -> None:
    observation = build_calibrated_observation_fixture("GW_ProV11")
    result = run_club_only_ui_match(
        create_club_only_session(
            trial_id=observation.trial_id,
            model_id="double_pendulum",
            preset=MatchPreset.FAST_PREVIEW,
        ),
        observation=observation,
    )
    view = build_club_only_result_view(result)
    legend = build_observed_inferred_legend(view)
    roles = {entry.role for entry in legend}
    assert ObservationRole.OBSERVED in roles
    assert ObservationRole.INFERRED in roles
    assert all(
        "body" not in e.label.lower() or e.role is ObservationRole.INFERRED
        for e in legend
        if "body" in e.label.lower()
    )
    assert view.trial_clock_hz == pytest.approx(240.0)
    np.testing.assert_allclose(view.native_time_s, observation.native_time_s)


def test_cancel_and_resume_hooks() -> None:
    observation = build_calibrated_observation_fixture("TW_wiffle")
    session = create_club_only_session(
        trial_id=observation.trial_id,
        model_id="double_pendulum",
        preset=MatchPreset.FAST_PREVIEW,
    )

    def _cancel() -> bool:
        return True

    with pytest.raises(MatchCancelledError):
        run_club_only_ui_match(session, observation=observation, cancel_hook=_cancel)

    first = run_club_only_ui_match(session, observation=observation)
    assert first.checkpoint is not None
    resumed = run_club_only_ui_match(
        session,
        observation=observation,
        resume_checkpoint=first.checkpoint,
    )
    assert resumed.checkpoint is not None
    assert resumed.checkpoint.cache_key == first.checkpoint.cache_key


def test_keyboard_action_map_covers_required_operations() -> None:
    actions = keyboard_action_map()
    assert actions["preview"] == "P"
    assert actions["verified_fit"] == "V"
    assert actions["compare_candidates"] == "C"
    assert actions["inspect_evidence"] == "I"
    assert actions["clone_session"] == "L"
    assert actions["cancel"] == "Escape"


def test_ledger_row_uses_club_only_lane_and_named_blockers(tmp_path: Path) -> None:
    from src.shared.python.motion_matching.club_only.ui_integration import (
        write_club_only_result_package,
    )

    observation = build_calibrated_observation_fixture("TW_wiffle")
    result = run_club_only_ui_match(
        create_club_only_session(
            trial_id=observation.trial_id,
            model_id="double_pendulum",
            preset=MatchPreset.FAST_PREVIEW,
        ),
        observation=observation,
    )
    view = build_club_only_result_view(result)
    receipt = tmp_path / "ledger_receipt.json"
    write_club_only_result_package(view, receipt)
    row = publish_club_only_ledger_row(view, receipt_path=receipt)
    assert isinstance(row, LedgerRow)
    assert row.lane == "club_only"
    assert row.engine
    assert row.acceptance is not None
    assert row.acceptance.get("native_g1_pass") is False
    assert row.reason
    assert "native_g1" in row.reason or "software_contract" in row.reason


def test_results_browser_indexes_club_only_json_packages(tmp_path: Path) -> None:
    package = {
        "schema_version": UI_SCHEMA,
        "kind": "club_only_ui_result",
        "backend": "double_pendulum",
        "meta_session_id": "sess-1",
        "meta_dataset_id": "club-only-excel",
        "native_g1_pass": False,
        "display_status": VerificationDisplayStatus.PREVIEW.value,
    }
    path = tmp_path / "club_only_result.json"
    path.write_text(json.dumps(package), encoding="utf-8")
    browser = ResultsBrowser(tmp_path)
    indexed = browser.index(ResultFilter(extensions=(".json", ".h5", ".hdf5")))
    assert len(indexed) == 1
    assert indexed[0].kind == "club_only_ui_result"
    assert indexed[0].schema_version == UI_SCHEMA
    assert indexed[0].backend == "double_pendulum"
    assert indexed[0].metadata["session_id"] == "sess-1"


def test_pipeline_club_only_request_binds_existing_facades() -> None:
    from src.tools.motion_matching import pipeline

    req = pipeline.ClubOnlyMatchRequest(
        trial_id="TW_wiffle",
        model_id="double_pendulum",
        preset="fast_preview",
    )
    assert req.source_kind == "club_only_excel"
    session = req.to_session()
    assert isinstance(session, ClubOnlyUiSession)
    assert session.preset is MatchPreset.FAST_PREVIEW
    summary = pipeline.summarize_club_only_result(
        build_club_only_result_view(
            run_club_only_ui_match(
                session,
                observation=build_calibrated_observation_fixture("TW_wiffle"),
            )
        )
    )
    assert summary["display_status"] != "native_verified"
    assert summary["native_g1_pass"] is False


def test_evidence_payload_is_versioned_and_honest() -> None:
    payload = ui_integration_evidence_payload(REPO_ROOT)
    assert payload["schema_version"] == UI_SCHEMA
    assert payload["governing_issue"] == 10613
    assert payload["native_g1_pass"] is False
    assert payload["unique_trial_count"] == 4
    assert EVIDENCE.is_file()
    on_disk = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == UI_SCHEMA
    assert on_disk["native_g1_pass"] is False


def test_ledger_row_hashes_receipt_file_bytes(tmp_path: Path) -> None:
    from src.shared.python.motion_matching.club_only.ui_integration import (
        write_club_only_result_package,
    )

    observation = build_calibrated_observation_fixture("TW_wiffle")
    result = run_club_only_ui_match(
        create_club_only_session(
            trial_id=observation.trial_id,
            model_id="double_pendulum",
            preset=MatchPreset.FAST_PREVIEW,
        ),
        observation=observation,
    )
    view = build_club_only_result_view(result)
    receipt = tmp_path / "club_only_receipt.json"
    digest = write_club_only_result_package(view, receipt)
    assert receipt.is_file()
    assert digest == __import__("hashlib").sha256(receipt.read_bytes()).hexdigest()
    row = publish_club_only_ledger_row(view, receipt_path=receipt)
    assert row.sha256 == digest


def test_results_browser_default_index_includes_club_only_json(tmp_path: Path) -> None:
    package = {
        "schema_version": UI_SCHEMA,
        "kind": "club_only_ui_result",
        "backend": "driven_double_pendulum",
        "meta_session_id": "sess-default",
        "meta_dataset_id": "club-only-excel",
        "native_g1_pass": False,
        "display_status": VerificationDisplayStatus.PREVIEW.value,
    }
    (tmp_path / "club_only_result.json").write_text(
        json.dumps(package), encoding="utf-8"
    )
    indexed = ResultsBrowser(tmp_path).index()
    assert any(item.kind == "club_only_ui_result" for item in indexed)


def test_workbook_observation_loader_uses_selected_trial() -> None:
    from src.shared.python.motion_matching.club_only.ui_integration import (
        load_club_only_workbook_observation,
    )
    from src.shared.python.motion_matching.club_only.workbook_identity import (
        CLUB_DATA_RELATIVE,
    )

    workbook = REPO_ROOT / CLUB_DATA_RELATIVE
    if not workbook.is_file():
        pytest.skip("Club_Data.xlsx not present")
    obs_a = load_club_only_workbook_observation(REPO_ROOT, "TW_wiffle")
    obs_b = load_club_only_workbook_observation(REPO_ROOT, "GW_wiffle")
    assert obs_a.trial_id == "TW_wiffle"
    assert obs_b.trial_id == "GW_wiffle"
    assert len(obs_a.native_time_s) > 32
    assert not np.allclose(obs_a.face_xyz[0], obs_b.face_xyz[0])
