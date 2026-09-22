"""Club-only reproduction guide and final turnover (CO-10 #10614).

Publishes operator/reproduction contracts: exact saved-job commands, trial and
model roster, raw-source provenance, assumptions, candidate selection, clean-
environment portable replay, and evidence-linked matrix reconciliation.

Software-contract layer only. Native G1 / full-body G3 / neural speed success
are never inherited from this module.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.shared.python.motion_matching.club_only.matrix_qualification import (
    MATRIX_SCHEMA,
    MatrixCellResult,
    MatrixQualificationReport,
    build_matrix_qualification_report,
)
from src.shared.python.motion_matching.club_only.profiles import (
    resolve_roster_matrix_scope,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
    CLUB_DATA_RELATIVE,
    CLUB_DATA_SHA256,
    IDENTITY_SCHEMA,
    WIFFLE_PROV1_RELATIVE,
    WIFFLE_PROV1_SHA256,
)
from src.shared.python.motion_matching.jobs import JOBS_SCHEMA

REPRODUCTION_SCHEMA = "club-only-reproduction/1.0.0"
_GOVERNING_ISSUE = 10614
_PARENT_EPIC = 10602
_NEURAL_EPIC = 10603
_NATIVE_PROGRAM = 10363
_DEFAULT_BLOCKERS: tuple[str, ...] = (
    "native_g1_qualification_requires_desk_native_receipt",
    "software_contract_reproduction_guide_is_not_native_evidence",
    "mandatory_native_fits_remain_open_for_unqualified_matrix_cells",
)

_GUIDE_RELATIVE = Path("docs/plans/club_only_matching/REPRODUCTION_GUIDE.md")
_EVIDENCE_RELATIVE = Path(
    "docs/plans/club_only_matching/evidence/club_reproduction_turnover.json"
)

__all__ = [
    "REPRODUCTION_SCHEMA",
    "CleanEnvironmentReplay",
    "ReconciledMatrixCell",
    "ReproductionGuide",
    "SavedJobCommand",
    "assert_epic_not_closed_with_missing_fits",
    "assert_no_fake_native_success",
    "build_clean_environment_replay",
    "build_reproduction_guide",
    "build_saved_job_commands",
    "reconcile_matrix_blockers",
    "render_reproduction_guide_markdown",
    "reproduction_evidence_payload",
]


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SavedJobCommand:
    """Exact, copyable command for one operator workflow step."""

    name: str
    purpose: str
    argv: tuple[str, ...]
    cwd_relative: str = "."

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.purpose.strip():
            raise ValueError("purpose must be non-empty")
        if not self.argv:
            raise ValueError("argv must be non-empty")
        if not self.cwd_relative.strip():
            raise ValueError("cwd_relative must be non-empty")

    def shell_line(self) -> str:
        """Return a copy-pasteable shell line with quoted multi-word args."""
        return " ".join(shlex.quote(part) for part in self.argv)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "argv": list(self.argv),
            "cwd_relative": self.cwd_relative,
            "shell": self.shell_line(),
        }


@dataclass(frozen=True)
class CleanEnvironmentReplay:
    """Portable package export/import contract for clean-host replay."""

    requires_clean_venv: bool
    portable_package_schema: str
    export_command: str
    import_command: str
    rejects_measured_state_resets: bool
    matlab_release: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "requires_clean_venv": self.requires_clean_venv,
            "portable_package_schema": self.portable_package_schema,
            "export_command": self.export_command,
            "import_command": self.import_command,
            "rejects_measured_state_resets": self.rejects_measured_state_resets,
            "matlab_release": self.matlab_release,
        }


@dataclass(frozen=True)
class ReconciledMatrixCell:
    """Matrix cell with evidence link and executable next-step prompt."""

    model_id: str
    trial_id: str
    status: str
    blocker: str | None
    owner: str
    next_step_prompt: str
    evidence_kind: str
    claims_native: bool
    original_3d_rmse_m: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "trial_id": self.trial_id,
            "status": self.status,
            "blocker": self.blocker,
            "owner": self.owner,
            "next_step_prompt": self.next_step_prompt,
            "evidence_kind": self.evidence_kind,
            "claims_native": self.claims_native,
            "original_3d_rmse_m": self.original_3d_rmse_m,
        }


@dataclass(frozen=True)
class ReproductionGuide:
    """Final operator/reproduction turnover for club-only matching."""

    schema: str
    governing_issue: int
    parent_epic: int
    companion_neural_epic: int
    native_program_epic: int
    trial_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    matrix_cells: tuple[ReconciledMatrixCell, ...]
    raw_source_provenance: tuple[dict[str, Any], ...]
    assumptions: tuple[str, ...]
    candidate_selection: Mapping[str, str]
    saved_job_commands: tuple[SavedJobCommand, ...]
    clean_environment_replay: CleanEnvironmentReplay
    qualification_blockers: tuple[str, ...]
    profile_gate_freeze_hash: str
    epic_closure_allowed: bool
    claims_native_qualification: bool
    native_g1_pass: bool
    inherits_full_body_g3_success: bool
    inherits_neural_speed_success: bool
    program_separation_notes: tuple[str, ...]
    evidence_links: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        unresolved = [c for c in self.matrix_cells if c.status != "scored"]
        return {
            "schema": self.schema,
            "governing_issue": self.governing_issue,
            "parent_epic": self.parent_epic,
            "companion_neural_epic": self.companion_neural_epic,
            "native_program_epic": self.native_program_epic,
            "trial_ids": list(self.trial_ids),
            "model_ids": list(self.model_ids),
            "cell_count": len(self.matrix_cells),
            "scored_count": sum(1 for c in self.matrix_cells if c.status == "scored"),
            "unresolved_count": len(unresolved),
            "matrix_cells": [c.as_dict() for c in self.matrix_cells],
            "raw_source_provenance": list(self.raw_source_provenance),
            "assumptions": list(self.assumptions),
            "candidate_selection": dict(self.candidate_selection),
            "saved_job_commands": [c.as_dict() for c in self.saved_job_commands],
            "clean_environment_replay": self.clean_environment_replay.as_dict(),
            "qualification_blockers": list(self.qualification_blockers),
            "profile_gate_freeze_hash": self.profile_gate_freeze_hash,
            "epic_closure_allowed": self.epic_closure_allowed,
            "claims_native_qualification": self.claims_native_qualification,
            "native_g1_pass": self.native_g1_pass,
            "inherits_full_body_g3_success": self.inherits_full_body_g3_success,
            "inherits_neural_speed_success": self.inherits_neural_speed_success,
            "program_separation_notes": list(self.program_separation_notes),
            "evidence_links": list(self.evidence_links),
            "limitations": list(self.limitations),
            "workbook_identity_schema": IDENTITY_SCHEMA,
            "matrix_schema": MATRIX_SCHEMA,
            "jobs_schema": JOBS_SCHEMA,
            "guide_path": str(_GUIDE_RELATIVE).replace("\\", "/"),
            "evidence_path": str(_EVIDENCE_RELATIVE).replace("\\", "/"),
        }


def build_saved_job_commands() -> tuple[SavedJobCommand, ...]:
    """Return exact saved-job / acceptance commands for operators."""
    return (
        SavedJobCommand(
            name="workbook_identity",
            purpose="Verify frozen workbook SHA-256 and four-trial lineage.",
            argv=(
                "python",
                "-m",
                "pytest",
                "tests/unit/motion_matching/test_club_workbook_identity.py",
                "-q",
                "-n",
                "0",
                "--no-cov",
                "--timeout=60",
            ),
        ),
        SavedJobCommand(
            name="matrix_qualification",
            purpose="Reconcile four-trial × #10585 roster matrix (software contract).",
            argv=(
                "python",
                "-m",
                "pytest",
                "tests/unit/motion_matching/test_club_matrix_qualification.py",
                "-q",
                "-n",
                "0",
                "--no-cov",
                "--timeout=60",
            ),
        ),
        SavedJobCommand(
            name="ui_integration",
            purpose="Exercise FitSwingProvider/pipeline/ledger Club-Only UI contracts.",
            argv=(
                "python",
                "-m",
                "pytest",
                "tests/unit/motion_matching/test_club_ui_integration.py",
                "-q",
                "-n",
                "0",
                "--no-cov",
                "--timeout=90",
            ),
        ),
        SavedJobCommand(
            name="fast_preview_match",
            purpose="Run a software-contract FAST_PREVIEW club-only match for TW_wiffle.",
            argv=(
                "python",
                "-c",
                (
                    "from src.shared.python.motion_matching.club_only.ui_integration "
                    "import create_club_only_session, run_club_only_ui_match; "
                    "from src.shared.python.motion_matching.club_only.fast_matching "
                    "import MatchPreset; "
                    "s=create_club_only_session(trial_id='TW_wiffle',"
                    " model_id='driven_double_pendulum',"
                    " preset=MatchPreset.FAST_PREVIEW); "
                    "r=run_club_only_ui_match(s); "
                    "assert r.display_status.value!='verified' or "
                    "r.match.native_g1_pass is False; "
                    "print(r.display_status, r.match.qualification_blockers)"
                ),
            ),
        ),
        SavedJobCommand(
            name="portable_package_roundtrip",
            purpose="Export/import an MS-105 portable package without a second scheduler.",
            argv=(
                "python",
                "-c",
                (
                    "from pathlib import Path; "
                    "from src.shared.python.motion_matching.jobs import ("
                    "MatchingJobSpec, JobStage, HashBundle, "
                    "export_portable_package, import_portable_package, JOBS_SCHEMA); "
                    "print(JOBS_SCHEMA); "
                    "print('export_portable_package', export_portable_package.__name__); "
                    "print('import_portable_package', import_portable_package.__name__); "
                    "print('MatchingJobSpec', MatchingJobSpec.__name__, JobStage.FIT_REPLAY)"
                ),
            ),
        ),
        SavedJobCommand(
            name="reproduction_freshness",
            purpose="Docs/context/parity freshness for the CO-10 reproduction guide.",
            argv=(
                "python",
                "-m",
                "pytest",
                "tests/unit/motion_matching/test_club_reproduction_turnover.py",
                "-q",
                "-n",
                "0",
                "--no-cov",
                "--timeout=60",
            ),
        ),
    )


def build_clean_environment_replay() -> CleanEnvironmentReplay:
    """Document clean-venv portable replay using MS-105 jobs APIs."""
    return CleanEnvironmentReplay(
        requires_clean_venv=True,
        portable_package_schema=JOBS_SCHEMA,
        export_command=(
            'python -c "from pathlib import Path; '
            "from src.shared.python.motion_matching.jobs import "
            "export_portable_package; "
            "export_portable_package(Path('runs/<run_id>'), Path('packages/<run_id>'))\""
        ),
        import_command=(
            'python -c "from pathlib import Path; '
            "from src.shared.python.motion_matching.jobs import "
            "import_portable_package; "
            "pkg=import_portable_package(Path('packages/<run_id>')); "
            'print(pkg.run_id, pkg.status)"'
        ),
        rejects_measured_state_resets=True,
        matlab_release="R2025b",
    )


def _owner_for_status(status: str) -> str:
    if status == "scored":
        return "software-contract"
    if status in {"unqualified", "missing_runtime"}:
        return "desk-native / runtime owner"
    if status == "unsupported":
        return "model-baseline (#10585)"
    return "club-only program"


def _next_step_for_cell(cell: MatrixCellResult) -> str:
    if cell.status == "scored":
        return (
            "python -m pytest tests/unit/motion_matching/test_club_matrix_qualification.py "
            "-q -n 0 --no-cov --timeout=60  # software-contract only; schedule desk-native "
            f"G1 for {cell.model_id}×{cell.trial_id} before scientific promotion"
        )
    if cell.status == "unqualified":
        return (
            f"On DeskComputer with MATLAB R2025b, run native Fit/G1 for "
            f"{cell.model_id}×{cell.trial_id} and attach a receipt under "
            "docs/plans/club_only_matching/evidence/; do not invent native_g1_pass"
        )
    if cell.status == "missing_runtime":
        return (
            f"On DeskComputer, install/enable the missing runtime for {cell.model_id}, "
            'then python -c "from src.shared.python.motion_matching.club_only.'
            "matrix_qualification import build_matrix_qualification_report; "
            'build_matrix_qualification_report()" and commit a fresh evidence receipt'
        )
    if cell.status == "unsupported":
        return (
            f"On DeskComputer, keep {cell.model_id} out of club-only promotion; "
            "reconcile via #10585 ownership / topology before adding a matrix "
            "cell campaign"
        )
    if cell.status == "rejected":
        return (
            f"On DeskComputer, diagnose gate failures for {cell.model_id}×"
            f"{cell.trial_id} ({cell.blocker or 'physical_failed'}); retain "
            "rejected package hashes; re-run python -m pytest "
            "tests/unit/motion_matching/test_club_matrix_qualification.py "
            "-q -n 0 --no-cov --timeout=60 after the fix"
        )
    return (
        f"On DeskComputer, inspect unresolved cell {cell.model_id}×{cell.trial_id} "
        f"status={cell.status!r} blocker={cell.blocker!r}; do not invent "
        "native_g1_pass from software-contract tests"
    )


def reconcile_matrix_blockers(
    report: MatrixQualificationReport | None = None,
) -> tuple[ReconciledMatrixCell, ...]:
    """Attach owners and executable next-step prompts to every matrix cell."""
    built = report if report is not None else build_matrix_qualification_report()
    cells: list[ReconciledMatrixCell] = []
    for cell in built.cells:
        scored = cell.status == "scored"
        cells.append(
            ReconciledMatrixCell(
                model_id=cell.model_id,
                trial_id=cell.trial_id,
                status=cell.status,
                blocker=cell.blocker,
                owner=_owner_for_status(cell.status),
                next_step_prompt=_next_step_for_cell(cell),
                evidence_kind="software_contract" if scored else "unresolved",
                claims_native=False,
                original_3d_rmse_m=cell.original_3d_rmse_m,
            )
        )
    return tuple(cells)


def _raw_source_provenance() -> tuple[dict[str, Any], ...]:
    return (
        {
            "workbook_id": "club_data",
            "relative_path": CLUB_DATA_RELATIVE.as_posix(),
            "sha256": CLUB_DATA_SHA256,
            "authority": "CO-00 workbook identity / PR #10628 review",
        },
        {
            "workbook_id": "wiffle_prov1_club_3d",
            "relative_path": WIFFLE_PROV1_RELATIVE.as_posix(),
            "sha256": WIFFLE_PROV1_SHA256,
            "authority": "CO-00 workbook identity / PR #10628 review",
        },
    )


def _assumptions() -> tuple[str, ...]:
    return (
        "Workbook Definitions declare inches; reviewed SI authority is centimetres "
        "(to_meters_scale=0.01) per CO-00 unit contract.",
        "Body motion is a plausible inferred candidate under GolfPlausibilityPriors "
        "and geometry choices — it was not measured in the club-only Excel source.",
        "Priors, handedness, and grip/geometry edits are operator assumptions, not "
        "measurements; changing them invalidates prior candidate hashes.",
        "Only two orientation axes are independently populated; the third is derived "
        "with derivation metadata and must not be scored as measured.",
        "Filtering Experiments is an alias of TW_ProV1 and is listed once among the "
        "four unique trials.",
        "Software-contract fixtures and UI preview statuses are not native Fit/G1 "
        "evidence and do not satisfy full-body G3.",
    )


def _candidate_selection() -> dict[str, str]:
    return {
        "strategy": "feasibility-first pruning then bounded Pareto diversity (CO-07)",
        "preview_vs_verified": (
            "FAST_PREVIEW is display-only; verified-fit requires a continuous "
            "independent replay without measured-state resets (CO-06/CO-09)"
        ),
        "ranking": (
            "lexicographic/Pareto measured residual then prior then runtime; visual "
            "attractiveness never overrides physical failure (CO-02/CO-08)"
        ),
        "neural_slot": (
            "Optional empty neural proposal slot may exist; neural proposals are not "
            "native evidence and do not inherit club-only matrix success"
        ),
    }


def _program_separation_notes() -> tuple[str, ...]:
    return (
        "Club-only profiles do not satisfy full-body G3 acceptance "
        f"(native program #{_NATIVE_PROGRAM}).",
        f"Neural speed program (#{_NEURAL_EPIC}) does not inherit success from "
        f"club-only classical matching (#{_PARENT_EPIC}).",
        "Do not close epic #10602 while mandatory native fits remain missing for "
        "unqualified / missing_runtime / unsupported matrix cells.",
    )


def _evidence_links() -> tuple[str, ...]:
    return (
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


def build_reproduction_guide(
    repo_root: Path | str | None = None,
    *,
    matrix_report: MatrixQualificationReport | None = None,
) -> ReproductionGuide:
    """Build the final club-only operator/reproduction guide."""
    del repo_root  # reserved for future path-bound workbook verification
    roster, models, trials = resolve_roster_matrix_scope()
    report = (
        matrix_report
        if matrix_report is not None
        else build_matrix_qualification_report()
    )
    cells = reconcile_matrix_blockers(report)
    unresolved = [c for c in cells if c.status != "scored"]
    freeze_payload = {
        model_id: roster[model_id].as_dict()
        for model_id in models
        if model_id in roster
    }
    return ReproductionGuide(
        schema=REPRODUCTION_SCHEMA,
        governing_issue=_GOVERNING_ISSUE,
        parent_epic=_PARENT_EPIC,
        companion_neural_epic=_NEURAL_EPIC,
        native_program_epic=_NATIVE_PROGRAM,
        trial_ids=tuple(trials),
        model_ids=tuple(models),
        matrix_cells=cells,
        raw_source_provenance=_raw_source_provenance(),
        assumptions=_assumptions(),
        candidate_selection=_candidate_selection(),
        saved_job_commands=build_saved_job_commands(),
        clean_environment_replay=build_clean_environment_replay(),
        qualification_blockers=_DEFAULT_BLOCKERS + tuple(report.qualification_blockers),
        profile_gate_freeze_hash=report.profile_gate_freeze_hash
        or _sha256_payload(freeze_payload),
        epic_closure_allowed=False if unresolved else False,
        claims_native_qualification=False,
        native_g1_pass=False,
        inherits_full_body_g3_success=False,
        inherits_neural_speed_success=False,
        program_separation_notes=_program_separation_notes(),
        evidence_links=_evidence_links(),
        limitations=(
            "Software-contract reproduction guide only; desk-native receipts required "
            "before scientific promotion.",
            "Epic #10602 remains open while mandatory native fits are missing.",
            "UI/docs GREEN does not close unqualified matrix cells.",
        ),
    )


def assert_epic_not_closed_with_missing_fits(guide: ReproductionGuide) -> None:
    """Fail closed if epic closure is claimed while mandatory cells remain open."""
    if not isinstance(guide, ReproductionGuide):
        raise TypeError("guide must be ReproductionGuide")
    unresolved = [c for c in guide.matrix_cells if c.status != "scored"]
    if guide.epic_closure_allowed and unresolved:
        raise ValueError(
            "epic closure is forbidden while mandatory matrix fits are missing: "
            f"{len(unresolved)} unresolved cells"
        )
    if guide.epic_closure_allowed:
        raise ValueError(
            "epic closure is forbidden for CO-10; leave #10602 open until native "
            "mandatory fits land"
        )


def assert_no_fake_native_success(guide: ReproductionGuide) -> None:
    """Reject invented native G1 / qualification claims on the reproduction guide."""
    if not isinstance(guide, ReproductionGuide):
        raise TypeError("guide must be ReproductionGuide")
    if guide.native_g1_pass or guide.claims_native_qualification:
        if not guide.qualification_blockers:
            raise ValueError(
                "native qualification claims require named qualification_blockers "
                "or a desk-native receipt; software contracts are not native evidence"
            )
        raise ValueError(
            "native_g1_pass / claims_native_qualification must stay false on the "
            "software-contract reproduction guide"
        )
    if guide.inherits_full_body_g3_success or guide.inherits_neural_speed_success:
        raise ValueError(
            "club-only turnover must not inherit full-body G3 or neural speed success"
        )


def reproduction_evidence_payload(
    repo_root: Path | str | None = None,
    *,
    guide: ReproductionGuide | None = None,
) -> dict[str, Any]:
    """Serialize the CO-10 evidence receipt."""
    built = guide if guide is not None else build_reproduction_guide(repo_root)
    assert_epic_not_closed_with_missing_fits(built)
    assert_no_fake_native_success(built)
    payload = built.as_dict()
    payload["notes"] = [
        "Operator guide freezes saved-job commands, provenance, assumptions, and "
        "clean-environment portable replay for club-only matching.",
        "Matrix blockers are reconciled with executable next-step prompts; "
        "software-contract tests are not native evidence.",
        "Do not close epic #10602 or claim G3/neural success from this turnover.",
    ]
    return payload


def render_reproduction_guide_markdown(
    repo_root: Path | str | None = None,
    *,
    guide: ReproductionGuide | None = None,
) -> str:
    """Render the operator-facing reproduction guide markdown."""
    built = guide if guide is not None else build_reproduction_guide(repo_root)
    lines: list[str] = [
        "# Club-Only Reproduction Guide",
        "",
        "Governing issue: [CO-10 #10614](https://github.com/D-sorganization/UpstreamDrift/issues/10614).",
        f"Parent epic: [#{built.parent_epic}](https://github.com/D-sorganization/UpstreamDrift/issues/{built.parent_epic}).",
        f"Schema: `{built.schema}`.",
        "",
        "## Purpose",
        "",
        "Final operator/reproduction turnover for classical club-only matching.",
        "Use this guide to replay saved jobs, verify workbook provenance, select",
        "candidates under declared assumptions, and export/import portable packages",
        "in a clean environment. Software-contract GREEN is not native Fit/G1.",
        "",
        "## Trial and Model Roster",
        "",
        f"- Trials ({len(built.trial_ids)}): {', '.join(built.trial_ids)}",
        f"- Models ({len(built.model_ids)}): {', '.join(built.model_ids)}",
        f"- Matrix cells: {len(built.matrix_cells)} "
        f"(scored={sum(1 for c in built.matrix_cells if c.status == 'scored')}, "
        f"unresolved={sum(1 for c in built.matrix_cells if c.status != 'scored')})",
        "",
        "## Raw-Source Provenance",
        "",
    ]
    for entry in built.raw_source_provenance:
        lines.append(
            f"- `{entry['relative_path']}` — SHA-256 `{entry['sha256']}` "
            f"({entry['authority']})"
        )
    lines.extend(
        [
            "",
            "## Assumptions",
            "",
        ]
    )
    for assumption in built.assumptions:
        lines.append(f"- {assumption}")
    lines.extend(
        [
            "",
            "## Candidate Selection",
            "",
            f"- Strategy: {built.candidate_selection['strategy']}",
            f"- Preview vs verified: {built.candidate_selection['preview_vs_verified']}",
            f"- Ranking: {built.candidate_selection['ranking']}",
            f"- Neural slot: {built.candidate_selection['neural_slot']}",
            "",
            "## Exact Saved-Job Commands",
            "",
        ]
    )
    for cmd in built.saved_job_commands:
        lines.append(f"### `{cmd.name}`")
        lines.append("")
        lines.append(cmd.purpose)
        lines.append("")
        lines.append("```bash")
        lines.append(cmd.shell_line())
        lines.append("```")
        lines.append("")
    replay = built.clean_environment_replay
    lines.extend(
        [
            "## Clean-Environment Portable Replay",
            "",
            f"- Requires clean venv: `{replay.requires_clean_venv}`",
            f"- Portable package schema: `{replay.portable_package_schema}`",
            f"- MATLAB release for native work: `{replay.matlab_release}`",
            f"- Rejects measured-state resets: `{replay.rejects_measured_state_resets}`",
            "",
            "Export:",
            "",
            "```bash",
            replay.export_command,
            "```",
            "",
            "Import:",
            "",
            "```bash",
            replay.import_command,
            "```",
            "",
            "## Qualification Blockers and Epic Closure",
            "",
            f"- `epic_closure_allowed`: `{built.epic_closure_allowed}`",
            f"- `native_g1_pass`: `{built.native_g1_pass}`",
            f"- `claims_native_qualification`: `{built.claims_native_qualification}`",
            "",
        ]
    )
    for blocker in built.qualification_blockers:
        lines.append(f"- {blocker}")
    lines.extend(
        [
            "",
            "## Program Separation",
            "",
        ]
    )
    for note in built.program_separation_notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Evidence Links",
            "",
        ]
    )
    for name in built.evidence_links:
        lines.append(f"- [evidence/{name}](evidence/{name})")
    lines.extend(
        [
            "",
            "## Unresolved Matrix Cells (Executable Next Steps)",
            "",
            "| Model | Trial | Status | Owner | Next Step |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for cell in built.matrix_cells:
        if cell.status == "scored":
            continue
        prompt = cell.next_step_prompt.replace("|", "\\|")
        lines.append(
            f"| `{cell.model_id}` | `{cell.trial_id}` | `{cell.status}` | "
            f"{cell.owner} | {prompt} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in built.limitations:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)
