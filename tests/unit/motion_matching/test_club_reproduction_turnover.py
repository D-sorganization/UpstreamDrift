"""CO-10 publish reproduction guide and final club-only turnover (#10614)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_matching.club_only.matrix_qualification import (
    MATRIX_SCHEMA,
    build_matrix_qualification_report,
)
from src.shared.python.motion_matching.club_only.reproduction import (
    REPRODUCTION_SCHEMA,
    assert_epic_not_closed_with_missing_fits,
    assert_no_fake_native_success,
    build_clean_environment_replay,
    build_reproduction_guide,
    build_saved_job_commands,
    reconcile_matrix_blockers,
    reproduction_evidence_payload,
    render_reproduction_guide_markdown,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
    CLUB_DATA_SHA256,
    WIFFLE_PROV1_SHA256,
)
from src.shared.python.tour_baselines.registry import (
    init_default_registry,
    list_golf_models,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAN_DIR = REPO_ROOT / "docs" / "plans" / "club_only_matching"
EVIDENCE = PLAN_DIR / "evidence" / "club_reproduction_turnover.json"
GUIDE_MD = PLAN_DIR / "REPRODUCTION_GUIDE.md"
TURNOVER_MD = PLAN_DIR / "TURNOVER.md"
MATRIX_EVIDENCE = PLAN_DIR / "evidence" / "club_matrix_qualification.json"

REQUIRED_EVIDENCE_LINKS = (
    "club_workbook_identity.json",
    "club_observation_contracts.json",
    "club_plausibility_acceptance.json",
    "club_starting_guesses.json",
    "club_pendulum_match.json",
    "club_body_candidates.json",
    "club_control_replay.json",
    "club_fast_matching.json",
    "club_matrix_qualification.json",
    "club_ui_integration.json",
    "club_reproduction_turnover.json",
)


def test_schema_and_governing_issue() -> None:
    guide = build_reproduction_guide(REPO_ROOT)
    assert guide.schema == REPRODUCTION_SCHEMA
    assert guide.governing_issue == 10614
    assert guide.parent_epic == 10602
    assert guide.companion_neural_epic == 10603
    assert guide.native_program_epic == 10363


def test_four_trials_and_full_model_roster() -> None:
    init_default_registry()
    guide = build_reproduction_guide(REPO_ROOT)
    assert tuple(guide.trial_ids) == CANONICAL_TRIAL_SHEETS
    roster_ids = {m.model_id for m in list_golf_models()}
    assert set(guide.model_ids) == roster_ids
    assert len(guide.matrix_cells) == len(CANONICAL_TRIAL_SHEETS) * len(roster_ids)


def test_raw_source_provenance_hashes() -> None:
    guide = build_reproduction_guide(REPO_ROOT)
    hashes = {entry["sha256"] for entry in guide.raw_source_provenance}
    assert CLUB_DATA_SHA256 in hashes
    assert WIFFLE_PROV1_SHA256 in hashes
    for entry in guide.raw_source_provenance:
        assert entry["relative_path"]
        assert len(entry["sha256"]) == 64


def test_saved_job_commands_are_exact_and_runnable() -> None:
    commands = build_saved_job_commands()
    names = {cmd.name for cmd in commands}
    assert {
        "workbook_identity",
        "matrix_qualification",
        "ui_integration",
        "fast_preview_match",
        "portable_package_roundtrip",
        "reproduction_freshness",
    } <= names
    for cmd in commands:
        assert cmd.argv
        assert "python" in cmd.argv[0].lower() or cmd.argv[0] == "python"
        assert cmd.cwd_relative == "."
        assert cmd.purpose


def test_clean_environment_replay_contract() -> None:
    replay = build_clean_environment_replay()
    assert replay.requires_clean_venv is True
    assert replay.portable_package_schema
    assert "export_portable_package" in replay.export_command
    assert "import_portable_package" in replay.import_command
    assert replay.rejects_measured_state_resets is True
    assert replay.matlab_release == "R2025b"


def test_assumptions_and_candidate_selection_documented() -> None:
    guide = build_reproduction_guide(REPO_ROOT)
    assert guide.assumptions
    joined = " | ".join(guide.assumptions).lower()
    assert "prior" in joined
    assert "inferred" in joined or "body" in joined
    assert "cm" in joined or "centimetre" in joined or "unit" in joined
    assert guide.candidate_selection["strategy"]
    assert "pareto" in guide.candidate_selection["strategy"].lower()
    assert guide.candidate_selection["preview_vs_verified"]


def test_matrix_blockers_reconciled_with_next_step_prompts() -> None:
    report = build_matrix_qualification_report()
    reconciled = reconcile_matrix_blockers(report)
    unresolved = [cell for cell in reconciled if cell.status != "scored"]
    assert unresolved
    for cell in unresolved:
        assert cell.next_step_prompt
        assert cell.blocker
        assert cell.owner
        assert (
            "python" in cell.next_step_prompt.lower()
            or "desk" in cell.next_step_prompt.lower()
        )
    scored = [cell for cell in reconciled if cell.status == "scored"]
    assert scored
    for cell in scored:
        assert cell.evidence_kind == "software_contract"
        assert cell.claims_native is False


def test_missing_qualification_blocks_promotion_and_epic_closure() -> None:
    guide = build_reproduction_guide(REPO_ROOT)
    assert guide.epic_closure_allowed is False
    assert guide.claims_native_qualification is False
    assert guide.native_g1_pass is False
    assert (
        "native_g1_qualification_requires_desk_native_receipt"
        in guide.qualification_blockers
    )
    assert_epic_not_closed_with_missing_fits(guide)
    assert_no_fake_native_success(guide)

    with pytest.raises(ValueError, match="epic|mandatory|missing"):
        assert_epic_not_closed_with_missing_fits(
            guide.__class__(
                **{
                    **guide.__dict__,
                    "epic_closure_allowed": True,
                }
            )
        )
    with pytest.raises(ValueError, match="native"):
        assert_no_fake_native_success(
            guide.__class__(
                **{
                    **guide.__dict__,
                    "native_g1_pass": True,
                    "claims_native_qualification": True,
                    "qualification_blockers": (),
                }
            )
        )


def test_separation_from_g3_and_neural_programs() -> None:
    guide = build_reproduction_guide(REPO_ROOT)
    assert guide.inherits_full_body_g3_success is False
    assert guide.inherits_neural_speed_success is False
    notes = " ".join(guide.program_separation_notes).lower()
    assert "g3" in notes
    assert "neural" in notes


def test_evidence_payload_freshness_matches_committed_receipt() -> None:
    payload = reproduction_evidence_payload(REPO_ROOT)
    assert EVIDENCE.is_file(), f"missing evidence receipt {EVIDENCE}"
    committed = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert committed["schema"] == REPRODUCTION_SCHEMA
    assert committed["governing_issue"] == 10614
    assert committed["native_g1_pass"] is False
    assert committed["epic_closure_allowed"] is False
    assert committed["cell_count"] == payload["cell_count"]
    assert committed["unresolved_count"] == payload["unresolved_count"]
    assert committed["profile_gate_freeze_hash"] == payload["profile_gate_freeze_hash"]
    assert set(committed["trial_ids"]) == set(payload["trial_ids"])
    assert set(committed["model_ids"]) == set(payload["model_ids"])


def test_docs_and_context_parity_freshness() -> None:
    assert GUIDE_MD.is_file(), f"missing operator guide {GUIDE_MD}"
    assert TURNOVER_MD.is_file()
    guide_text = GUIDE_MD.read_text(encoding="utf-8")
    turnover = TURNOVER_MD.read_text(encoding="utf-8")
    rendered = render_reproduction_guide_markdown(REPO_ROOT)
    assert "# Club-Only Reproduction Guide" in guide_text
    assert "CO-10" in guide_text or "#10614" in guide_text
    for name in REQUIRED_EVIDENCE_LINKS:
        assert name in guide_text, f"guide missing evidence link {name}"
        assert name in turnover or name.replace(".json", "") in turnover
    assert "REPRODUCTION_GUIDE.md" in turnover
    assert "CO-10" in turnover or "#10614" in turnover
    assert "shipped" in turnover.lower() or "complete" in turnover.lower()
    # Renderer and on-disk guide must share the same semantic body; Prettier may
    # pad markdown tables, so compare normalized non-table lines.
    assert _normalize_guide_body(guide_text) == _normalize_guide_body(rendered)
    # Matrix evidence must remain the CO-08 schema the guide reconciles.
    matrix = json.loads(MATRIX_EVIDENCE.read_text(encoding="utf-8"))
    assert matrix["schema"] == MATRIX_SCHEMA
    assert matrix["native_g1_pass"] is False


def _normalize_guide_body(text: str) -> list[str]:
    """Drop blank/table-padding differences while keeping command and prose lines."""
    lines: list[str] = []
    for raw in text.strip().splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("|"):
            # Keep cell text only; Prettier changes column padding.
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= {"-", ":"} for c in cells):
                continue
            lines.append("|".join(cells))
            continue
        lines.append(line)
    return lines


def test_portable_replay_command_mentions_ms105_jobs() -> None:
    commands = build_saved_job_commands()
    portable = next(c for c in commands if c.name == "portable_package_roundtrip")
    joined = " ".join(portable.argv)
    assert "export_portable_package" in joined or "MatchingJobSpec" in joined
    assert "motion_matching.jobs" in joined
