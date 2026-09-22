"""CO-09: club-only UI façade contracts (import, verified gate, legend, clone)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
)
from src.tools.motion_matching import club_only_ui as cui

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _write_minimal_workbook(path: Path, sheets: list[str]) -> Path:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    default = wb.active
    assert default is not None
    default.title = sheets[0]
    for name in sheets[1:]:
        wb.create_sheet(name)
    for ws in wb.worksheets:
        ws["A1"] = f"{ws.title} label"
    wb.save(path)
    return path


def test_import_lists_four_unique_canonical_trials(tmp_path: Path) -> None:
    path = _write_minimal_workbook(
        tmp_path / "club.xlsx",
        [*CANONICAL_TRIAL_SHEETS, "Filtering Experiments", "Noise"],
    )
    result = cui.import_club_only_workbook(path)
    assert result.ok is True
    assert result.trials == CANONICAL_TRIAL_SHEETS
    assert "Filtering Experiments" in " ".join(result.alias_conflicts)
    assert not result.missing_trials
    assert result.errors == ()


def test_import_dedups_alias_without_duplicating_trial(tmp_path: Path) -> None:
    path = _write_minimal_workbook(
        tmp_path / "alias.xlsx",
        ["TW_ProV1", "Filtering Experiments"],
    )
    result = cui.import_club_only_workbook(path)
    assert result.trials.count("TW_ProV1") == 1
    assert "TW_ProV1" in result.trials
    assert any("Filtering Experiments" in note for note in result.alias_conflicts)


def test_import_missing_path_and_empty_workbook_error_states(tmp_path: Path) -> None:
    missing = cui.import_club_only_workbook(tmp_path / "nope.xlsx")
    assert missing.ok is False
    assert missing.errors
    assert missing.trials == ()

    empty_path = tmp_path / "empty.xlsx"
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.title = "Unrelated"
    wb.save(empty_path)
    empty = cui.import_club_only_workbook(empty_path)
    assert empty.ok is False
    assert empty.missing_trials == CANONICAL_TRIAL_SHEETS
    assert empty.errors


def test_unqualified_and_fixture_cannot_appear_verified() -> None:
    for status in ("unqualified", "rejected", "unsupported", "missing_runtime"):
        assert cui.may_appear_as_verified(matrix_cell_status=status) is False
        label = cui.verified_display_label(
            matrix_cell_status=status, preset="verified_fit"
        )
        assert label != "VERIFIED"
        assert "verified" not in label.lower() or "not" in label.lower()

    assert cui.may_appear_as_verified(matrix_cell_status="scored") is True
    assert (
        cui.verified_display_label(matrix_cell_status="scored", preset="verified_fit")
        == "VERIFIED"
    )
    # Preview never claims verified even when matrix cell is scored.
    assert (
        cui.verified_display_label(matrix_cell_status="scored", preset="fast_preview")
        != "VERIFIED"
    )


def test_observed_versus_inferred_legend_text() -> None:
    legend = cui.observed_versus_inferred_legend()
    keys = {key for key, _ in legend}
    assert "observed_club" in keys
    assert "inferred_body" in keys
    text = " ".join(label for _, label in legend).lower()
    assert "observed" in text
    assert "inferred" in text or "plausible" in text
    assert "not measured" in text or "not a measurement" in text
    assert cui.BODY_MOTION_DISCLAIMER
    assert "plausible" in cui.BODY_MOTION_DISCLAIMER.lower()
    assert "not measured" in cui.BODY_MOTION_DISCLAIMER.lower()


def test_clone_session_preserves_user_edits() -> None:
    state = cui.ClubOnlySessionState(
        workbook_path=Path("Club_Data.xlsx"),
        trial_id="TW_wiffle",
        model_id="driven_double_pendulum",
        preset="fast_preview",
        user_notes="tighten grip prior",
        prior_edits={"shoulder_abduction_deg": 12.5, "notes": "edit-a"},
        cancel_requested=False,
    )
    cloned = state.clone()
    assert cloned is not state
    assert cloned.user_notes == state.user_notes
    assert cloned.prior_edits == state.prior_edits
    assert cloned.prior_edits is not state.prior_edits
    cloned.prior_edits["notes"] = "edit-b"
    cloned.user_notes = "changed"
    assert state.user_notes == "tighten grip prior"
    assert state.prior_edits["notes"] == "edit-a"


def test_cancel_resume_hooks_on_session() -> None:
    state = cui.ClubOnlySessionState(
        workbook_path=None,
        trial_id="TW_wiffle",
        model_id="driven_double_pendulum",
        preset="fast_preview",
    )
    assert state.cancel_check() is False
    state.request_cancel()
    assert state.cancel_requested is True
    assert state.cancel_check() is True
    state.clear_cancel()
    assert state.cancel_check() is False
    resumed = state.with_checkpoint_token("ckpt-1")
    assert resumed.checkpoint_token == "ckpt-1"
    assert state.checkpoint_token is None


def test_results_metadata_uses_club_only_lane_not_parallel_store() -> None:
    meta = cui.build_results_metadata(
        trial_id="GW_wiffle",
        model_id="driven_triple_pendulum",
        matrix_cell_status="unqualified",
        preset="verified_fit",
        workbook_sha256="a" * 64,
    )
    assert meta["lane"] == cui.CLUB_ONLY_LANE
    assert meta["source_kind"] == "club_only_excel"
    assert meta["trial_id"] == "GW_wiffle"
    assert meta["body_motion"] == "plausible_inferred_candidate"
    assert meta["appears_verified"] is False
    assert meta["matrix_cell_status"] == "unqualified"
    assert "store" not in meta.get("storage", "ledger")

    scored = cui.build_results_metadata(
        trial_id="TW_wiffle",
        model_id="driven_double_pendulum",
        matrix_cell_status="scored",
        preset="verified_fit",
        workbook_sha256="b" * 64,
    )
    assert scored["appears_verified"] is True
