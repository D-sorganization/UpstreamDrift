"""Club-only UI/results integration over existing facades (CO-09 #10613).

Headless service used by Motion Matching GUI, pipeline summaries, results
browser indexing, and ledger rows. Reuses workbook identity, fast matching,
and FitSwingProvider registration without a parallel solver or result store.
Software-contract evidence only — native G1 stays blocked by named reasons.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_only.fast_matching import (
    FAST_MATCH_SCHEMA,
    EmptyNeuralProposalProvider,
    FastMatchOptions,
    FastMatchResult,
    MatchBudget,
    MatchCheckpoint,
    MatchPreset,
    MatchCancelledError,
    NeuralProposalProvider,
    budget_for_preset,
    run_fast_club_match,
)
from src.shared.python.motion_matching.club_only.observation import (
    ClubObservation,
    build_calibrated_observation_fixture,
)
from src.shared.python.motion_matching.club_only.profiles import get_club_only_profile
from src.shared.python.motion_matching.club_only.seeds import (
    VERIFIED_SEED_SOURCES,
    CandidateSeed,
    geometry_content_hash,
    profile_content_hash,
)
from src.shared.python.motion_matching.club_only.workbook_identity import (
    CANONICAL_TRIAL_SHEETS,
    CLUB_DATA_RELATIVE,
    CLUB_DATA_SHA256,
    EXPECTED_SAMPLE_COUNTS,
    IDENTITY_SCHEMA,
    NATIVE_SAMPLE_RATE_HZ,
    UNIT_AUTHORITY,
    build_club_workbook_identity,
    verify_workbook_hash,
)
from src.shared.python.motion_matching.ledger import default_ledger_path
from src.shared.python.motion_matching.ledger_schema import (
    ArtefactPaths,
    Ledger,
    LedgerRow,
    SharedMetrics,
)

UI_SCHEMA = "club-only-ui-integration/1.0.0"
_GOVERNING_ISSUE = 10613
_DEFAULT_BLOCKERS: tuple[str, ...] = (
    "native_g1_qualification_requires_desk_native_receipt",
    "software_contract_ui_integration_is_not_native_evidence",
)
_BODY_DISCLAIMER = (
    "Body motion is a plausible inferred candidate under golf priors and "
    "geometry choices — it was not measured in the club-only Excel source."
)
_CLUB_ONLY_MODEL_ALIASES: Mapping[str, str] = {
    "double_pendulum": "driven_double_pendulum",
    "triple_pendulum": "driven_triple_pendulum",
}

__all__ = [
    "UI_SCHEMA",
    "ClubOnlySourceKind",
    "ClubOnlyTrialListing",
    "ClubOnlyUiResult",
    "ClubOnlyUiSession",
    "ClubOnlyWorkbookCatalog",
    "LegendEntry",
    "MatchCancelledError",
    "ObservationRole",
    "ResultViewModel",
    "VerificationDisplayStatus",
    "assert_unqualified_cannot_appear_verified",
    "build_club_only_result_view",
    "build_observed_inferred_legend",
    "clone_club_only_session",
    "create_club_only_session",
    "import_club_only_workbook_catalog",
    "keyboard_action_map",
    "list_motion_matching_source_kinds",
    "load_club_only_workbook_observation",
    "publish_club_only_ledger_row",
    "resolve_club_only_model_id",
    "run_club_only_ui_match",
    "ui_integration_evidence_payload",
    "write_club_only_result_package",
]


class ClubOnlySourceKind(str, Enum):
    """Motion Matching source selection modes."""

    TOUR_AVERAGE = "tour_average"
    CLUB_ONLY_EXCEL = "club_only_excel"


class ObservationRole(str, Enum):
    """Legend roles — measured club vs inferred body/control."""

    OBSERVED = "observed"
    INFERRED = "inferred"


class VerificationDisplayStatus(str, Enum):
    """UI verification badge. Native verified requires a native receipt."""

    PREVIEW = "preview"
    SOFTWARE_VERIFIED_FIT = "software_verified_fit"
    UNQUALIFIED = "unqualified"
    NATIVE_VERIFIED = "native_verified"


@dataclass(frozen=True)
class ClubOnlyTrialListing:
    """One unique workbook trial for source selection."""

    trial_id: str
    sheet_name: str
    alias_sheets: tuple[str, ...]
    sample_count: int
    clock_hz: float
    events: Mapping[str, float]
    observation_coverage: Mapping[str, str]
    unit_authority_note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "sheet_name": self.sheet_name,
            "alias_sheets": list(self.alias_sheets),
            "sample_count": self.sample_count,
            "clock_hz": self.clock_hz,
            "events": dict(self.events),
            "observation_coverage": dict(self.observation_coverage),
            "unit_authority_note": self.unit_authority_note,
        }


@dataclass(frozen=True)
class ClubOnlyWorkbookCatalog:
    """Imported Club-Only Excel catalog with conflicts and coverage."""

    schema_version: str
    workbook_sha256: str
    unique_trials: tuple[ClubOnlyTrialListing, ...]
    conflicts: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workbook_sha256": self.workbook_sha256,
            "unique_trials": [t.as_dict() for t in self.unique_trials],
            "conflicts": list(self.conflicts),
        }


@dataclass(frozen=True)
class ClubOnlyUiSession:
    """Editable club-only matching session (cloneable; preserves user edits)."""

    session_id: str
    source_kind: ClubOnlySourceKind
    trial_id: str
    model_id: str
    preset: MatchPreset
    prior_choices: Mapping[str, Any]
    geometry_choices: Mapping[str, Any]
    user_edits: Mapping[str, Any]
    computation_budget: MatchBudget
    body_motion_disclaimer: str = _BODY_DISCLAIMER
    neural_proposal_slot: str = "empty_provider"

    def preset_name(self) -> str:
        """Return the match preset wire name without deep attribute chains."""
        return str(self.preset.value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source_kind": self.source_kind.value,
            "trial_id": self.trial_id,
            "model_id": self.model_id,
            "preset": self.preset_name(),
            "prior_choices": dict(self.prior_choices),
            "geometry_choices": dict(self.geometry_choices),
            "user_edits": dict(self.user_edits),
            "computation_budget": {
                "max_time_s": self.computation_budget.max_time_s,
                "max_evaluations": self.computation_budget.max_evaluations,
                "max_pareto": self.computation_budget.max_pareto,
            },
            "body_motion_disclaimer": self.body_motion_disclaimer,
            "neural_proposal_slot": self.neural_proposal_slot,
        }


@dataclass(frozen=True)
class ClubOnlyUiResult:
    """Raw match outcome bound to a UI session."""

    session: ClubOnlyUiSession
    match: FastMatchResult
    observation: ClubObservation
    checkpoint: MatchCheckpoint | None
    seed: CandidateSeed | None = None


@dataclass(frozen=True)
class LegendEntry:
    """One legend row for observed vs inferred presentation."""

    label: str
    role: ObservationRole
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "role": self.role.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ResultViewModel:
    """UI/results view model — never invents native verification."""

    session_id: str
    trial_id: str
    model_id: str
    preset: MatchPreset
    display_status: VerificationDisplayStatus
    native_g1_pass: bool
    qualification_blockers: tuple[str, ...]
    trial_clock_hz: float
    native_time_s: NDArray[np.float64]
    body_motion_disclaimer: str
    candidate_ids: tuple[str, ...]
    error_time_tradeoffs: tuple[Mapping[str, float], ...]
    infeasible_models: tuple[Mapping[str, str], ...]
    prior_choices: Mapping[str, Any]
    geometry_choices: Mapping[str, Any]
    schema_version: str = UI_SCHEMA
    seed_source: str | None = None
    is_synthetic_seed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "club_only_ui_result",
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "model_id": self.model_id,
            "backend": self.model_id,
            "preset": self.preset.value,
            "display_status": self.display_status.value,
            "native_g1_pass": self.native_g1_pass,
            "qualification_blockers": list(self.qualification_blockers),
            "trial_clock_hz": self.trial_clock_hz,
            "body_motion_disclaimer": self.body_motion_disclaimer,
            "candidate_ids": list(self.candidate_ids),
            "error_time_tradeoffs": [dict(x) for x in self.error_time_tradeoffs],
            "infeasible_models": [dict(x) for x in self.infeasible_models],
            "prior_choices": dict(self.prior_choices),
            "geometry_choices": dict(self.geometry_choices),
            "meta_session_id": self.session_id,
            "meta_dataset_id": "club-only-excel",
        }


def resolve_club_only_model_id(model_id: str) -> str:
    """Resolve club-only UI aliases to roster model ids (avoids ambiguous registry)."""
    if not model_id or not str(model_id).strip():
        raise ValueError("model_id must be non-empty")
    return _CLUB_ONLY_MODEL_ALIASES.get(model_id, model_id)


def list_motion_matching_source_kinds() -> tuple[ClubOnlySourceKind, ...]:
    """Source kinds shown in Motion Matching source selection."""
    return (
        ClubOnlySourceKind.TOUR_AVERAGE,
        ClubOnlySourceKind.CLUB_ONLY_EXCEL,
    )


def keyboard_action_map() -> Mapping[str, str]:
    """Keyboard shortcuts for club-only UI operations."""
    return {
        "preview": "P",
        "verified_fit": "V",
        "compare_candidates": "C",
        "inspect_evidence": "I",
        "clone_session": "L",
        "cancel": "Escape",
    }


def import_club_only_workbook_catalog(repo_root: Path | str) -> ClubOnlyWorkbookCatalog:
    """Import Club-Only Excel via existing workbook identity contracts."""
    identity = build_club_workbook_identity(repo_root)
    measured_coverage = {
        "butt_position": ObservationRole.OBSERVED.value,
        "head_position": ObservationRole.OBSERVED.value,
        "butt_orientation": ObservationRole.OBSERVED.value,
        "head_orientation": ObservationRole.OBSERVED.value,
    }
    workbook_sha = next(
        (m.sha256 for m in identity.manifests if m.sha256 == CLUB_DATA_SHA256),
        identity.manifests[0].sha256 if identity.manifests else "",
    )
    trials: list[ClubOnlyTrialListing] = []
    for record in identity.trials:
        if record.primary_sheet not in CANONICAL_TRIAL_SHEETS:
            continue
        trials.append(
            ClubOnlyTrialListing(
                trial_id=record.trial_id,
                sheet_name=record.primary_sheet,
                alias_sheets=tuple(record.alias_sheets),
                sample_count=int(record.numeric_sample_count),
                clock_hz=float(NATIVE_SAMPLE_RATE_HZ),
                events=dict(record.event_samples),
                observation_coverage=dict(measured_coverage),
                unit_authority_note=(
                    f"Declared {UNIT_AUTHORITY.declared_units}; reviewed SI uses "
                    f"{UNIT_AUTHORITY.reviewed_units} "
                    f"(×{UNIT_AUTHORITY.to_meters_scale:g})."
                ),
            )
        )
    trials_sorted = tuple(sorted(trials, key=lambda t: t.trial_id))
    conflicts = (
        (
            f"Unit conflict: workbook declares {UNIT_AUTHORITY.declared_units} "
            f"but reviewed authority is {UNIT_AUTHORITY.reviewed_units} "
            f"({UNIT_AUTHORITY.authority})."
        ),
        (
            "Ball label is not a calibrated force/impulse observation; "
            "CHS and ball type remain descriptive only."
        ),
        (
            "Filtering Experiments is an alias of TW_ProV1 — listed once among "
            "the four unique trials."
        ),
        _BODY_DISCLAIMER,
    )
    catalog = ClubOnlyWorkbookCatalog(
        schema_version=IDENTITY_SCHEMA,
        workbook_sha256=workbook_sha,
        unique_trials=trials_sorted,
        conflicts=conflicts,
    )
    if len(catalog.unique_trials) != 4:
        raise ValueError("club-only catalog must list exactly four unique trials")
    if {t.trial_id for t in catalog.unique_trials} != set(CANONICAL_TRIAL_SHEETS):
        raise ValueError("catalog trial ids must match CANONICAL_TRIAL_SHEETS")
    return catalog


def load_club_only_workbook_observation(
    repo_root: Path | str,
    trial_id: str,
) -> ClubObservation:
    """Load a selected Club-Only Excel trial via the existing workbook importer.

    Uses ``load_club_target_excel`` + ``club_target_to_observation`` so the GUI
    fits the workbook sheet the user selected — fixtures remain test-only.
    """
    if trial_id not in CANONICAL_TRIAL_SHEETS:
        raise ValueError(f"unknown trial_id={trial_id!r}")
    root = Path(repo_root)
    workbook = root / CLUB_DATA_RELATIVE
    if not workbook.is_file():
        raise FileNotFoundError(f"club workbook missing: {workbook}")
    verify_workbook_hash(workbook, CLUB_DATA_SHA256)
    from src.shared.python.motion_matching.club_only.adapters import (
        club_target_to_observation,
    )
    from src.shared.python.motion_matching.club_target import AlignOptions
    from src.shared.python.motion_matching.loaders.excel import load_club_target_excel

    sample_count = int(EXPECTED_SAMPLE_COUNTS[trial_id])
    duration_s = (sample_count - 1) / float(NATIVE_SAMPLE_RATE_HZ)
    opts = AlignOptions(
        sample_rate_hz=float(NATIVE_SAMPLE_RATE_HZ),
        simulation_time_s=duration_s,
        time_alignment="none",
    )
    target = load_club_target_excel(workbook, trial_id, opts)
    return club_target_to_observation(target, trial_id=trial_id)


def create_club_only_session(
    *,
    trial_id: str,
    model_id: str,
    preset: MatchPreset,
    prior_choices: Mapping[str, Any] | None = None,
    geometry_choices: Mapping[str, Any] | None = None,
    user_edits: Mapping[str, Any] | None = None,
    neural_proposal_slot: str = "empty_provider",
) -> ClubOnlyUiSession:
    """Create a new club-only UI session with explicit budget and disclaimer."""
    if trial_id not in CANONICAL_TRIAL_SHEETS:
        raise ValueError(f"unknown trial_id={trial_id!r}")
    if not isinstance(preset, MatchPreset):
        raise TypeError("preset must be MatchPreset")
    resolved = resolve_club_only_model_id(model_id)
    profile = get_club_only_profile(resolved)
    budget = budget_for_preset(preset)
    return ClubOnlyUiSession(
        session_id=str(uuid.uuid4()),
        source_kind=ClubOnlySourceKind.CLUB_ONLY_EXCEL,
        trial_id=trial_id,
        model_id=profile.model_id,
        preset=preset,
        prior_choices=dict(prior_choices or {}),
        geometry_choices=dict(geometry_choices or {}),
        user_edits=dict(user_edits or {}),
        computation_budget=budget,
        neural_proposal_slot=neural_proposal_slot,
    )


def clone_club_only_session(session: ClubOnlyUiSession) -> ClubOnlyUiSession:
    """Clone a session with a new id while preserving user edits."""
    if not isinstance(session, ClubOnlyUiSession):
        raise TypeError("session must be ClubOnlyUiSession")
    return replace(session, session_id=str(uuid.uuid4()))


def _synthetic_seed(observation: ClubObservation, model_id: str) -> CandidateSeed:
    profile = get_club_only_profile(model_id)
    times = np.asarray(observation.native_time_s, dtype=np.float64)
    nq = 4 if "double" in model_id else 6 if "triple" in model_id else 8
    q = np.zeros(nq, dtype=np.float64)
    geom = geometry_content_hash(
        club_type=observation.club_type,
        catalog_length_m=float(observation.catalog_length_m),
        tool_to_model_residual_m=0.0,
    )
    return CandidateSeed(
        seed_id=f"ui-{observation.trial_id}-{model_id}",
        trial_id=observation.trial_id,
        model_id=model_id,
        source="ui_synthetic",
        q=q,
        body_configuration_hash=hashlib.sha256(q.tobytes()).hexdigest(),
        observed_residual_m=0.05,
        prior_score=0.5,
        feasibility_reasons=("ui_synthetic_seed",),
        timestamps_s=times,
        geometry_hash=geom,
        profile_hash=profile_content_hash(profile),
        body_is_prior=True,
        is_kinematic_preview=True,
    )


def run_club_only_ui_match(
    session: ClubOnlyUiSession,
    *,
    observation: ClubObservation | None = None,
    seed: CandidateSeed | None = None,
    cancel_hook: Callable[[], bool] | None = None,
    resume_checkpoint: MatchCheckpoint | None = None,
    neural_provider: NeuralProposalProvider | None = None,
) -> ClubOnlyUiResult:
    """Run preview/verified fit through existing fast-match orchestration."""
    if not isinstance(session, ClubOnlyUiSession):
        raise TypeError("session must be ClubOnlyUiSession")
    if observation is None:
        raise ValueError(
            "observation is required; silent fixture fallback is prohibited"
        )
    if not isinstance(observation, ClubObservation):
        raise TypeError("observation must be ClubObservation")
    if observation.trial_id != session.trial_id:
        raise ValueError("observation.trial_id must match session.trial_id")
    if seed is None:
        raise ValueError(
            "seed is required; must come from CO-03 retrieval or constrained_ik sources (got None)"
        )
    if not isinstance(seed, CandidateSeed):
        raise TypeError("seed must be CandidateSeed")
    obs = observation
    profile = get_club_only_profile(session.model_id)
    seed_obj = seed
    geom_hash = seed_obj.geometry_hash
    prof_hash = seed_obj.profile_hash
    provider = neural_provider or EmptyNeuralProposalProvider()
    options = FastMatchOptions(
        preset=session.preset,
        budget=session.computation_budget,
        diversity_seed=_GOVERNING_ISSUE,
        cancel_check=cancel_hook,
        checkpoint=resume_checkpoint,
        neural_provider=provider,
    )
    match = run_fast_club_match(
        observation=obs,
        profile=profile,
        seed=seed_obj,
        geometry_hash=geom_hash,
        profile_hash=prof_hash,
        options=options,
    )
    return ClubOnlyUiResult(
        session=session,
        match=match,
        observation=obs,
        checkpoint=match.checkpoint,
        seed=seed_obj,
    )


def build_club_only_result_view(result: ClubOnlyUiResult) -> ResultViewModel:
    """Build a results view that cannot claim native verification without receipts."""
    if not isinstance(result, ClubOnlyUiResult):
        raise TypeError("result must be ClubOnlyUiResult")
    native_pass = bool(result.match.native_g1_pass)
    blockers = list(result.match.qualification_blockers) or list(_DEFAULT_BLOCKERS)
    seed = result.seed
    is_synthetic = seed is None or seed.source not in VERIFIED_SEED_SOURCES
    if is_synthetic:
        blockers.append("synthetic_seed_unqualified_for_verified_fit")
    if result.session.preset is MatchPreset.FAST_PREVIEW:
        status = VerificationDisplayStatus.PREVIEW
    elif is_synthetic:
        status = VerificationDisplayStatus.UNQUALIFIED
    elif native_pass and not blockers:
        status = VerificationDisplayStatus.NATIVE_VERIFIED
    else:
        status = VerificationDisplayStatus.SOFTWARE_VERIFIED_FIT
    if not native_pass and status is VerificationDisplayStatus.NATIVE_VERIFIED:
        status = VerificationDisplayStatus.UNQUALIFIED
    tradeoffs = tuple(
        {
            "club_fit": (
                float(sample.best_observation_fit_m)
                if sample.best_observation_fit_m is not None
                else float("nan")
            ),
            "plausibility": float(sample.failed_attempts),
            "runtime_s": float(sample.wall_s),
            "evaluations": float(sample.evaluations),
        }
        for sample in result.match.quality_vs_time
    )
    infeasible = tuple(
        {
            "model_id": result.session.model_id,
            "reason": (
                "; ".join(branch.rejection_reasons)
                if branch.rejection_reasons
                else "infeasible_branch"
            ),
            "branch_id": branch.branch_id,
        }
        for branch in result.match.rejected
    )
    candidate_ids = tuple(branch.branch_id for branch in result.match.pareto)
    if not candidate_ids and result.checkpoint is not None:
        candidate_ids = tuple(result.checkpoint.pareto_ids)
    view = ResultViewModel(
        session_id=result.session.session_id,
        trial_id=result.session.trial_id,
        model_id=result.session.model_id,
        preset=result.session.preset,
        display_status=status,
        native_g1_pass=native_pass,
        qualification_blockers=tuple(blockers),
        trial_clock_hz=float(NATIVE_SAMPLE_RATE_HZ),
        native_time_s=np.asarray(result.observation.native_time_s, dtype=np.float64),
        body_motion_disclaimer=result.session.body_motion_disclaimer,
        candidate_ids=candidate_ids,
        error_time_tradeoffs=tradeoffs,
        infeasible_models=infeasible,
        prior_choices=dict(result.session.prior_choices),
        geometry_choices=dict(result.session.geometry_choices),
        seed_source=seed.source if seed is not None else None,
        is_synthetic_seed=is_synthetic,
    )
    assert_unqualified_cannot_appear_verified(view)
    return view


def build_observed_inferred_legend(
    view: ResultViewModel,
) -> tuple[LegendEntry, ...]:
    """Legend for observed club frames versus inferred body candidates."""
    if not isinstance(view, ResultViewModel):
        raise TypeError("view must be ResultViewModel")
    return (
        LegendEntry(
            label="Observed club frames",
            role=ObservationRole.OBSERVED,
            detail="Butt/head position and orientation from Club-Only Excel.",
        ),
        LegendEntry(
            label="Predicted club frames",
            role=ObservationRole.OBSERVED,
            detail="Model-projected club observables compared to the trial clock.",
        ),
        LegendEntry(
            label="Inferred body candidate",
            role=ObservationRole.INFERRED,
            detail=view.body_motion_disclaimer,
        ),
        LegendEntry(
            label="Prior / geometry choices",
            role=ObservationRole.INFERRED,
            detail=(
                "User-selected priors and geometry are assumptions, not measurements."
            ),
        ),
    )


def assert_unqualified_cannot_appear_verified(view: Any) -> None:
    """DbC: software-contract results must not show as native verified."""
    native_pass = bool(getattr(view, "native_g1_pass", False))
    status = getattr(view, "display_status", None)
    blockers = tuple(getattr(view, "qualification_blockers", ()) or ())
    if status is VerificationDisplayStatus.NATIVE_VERIFIED and (
        not native_pass or blockers
    ):
        raise ValueError(
            "unqualified club-only result cannot appear native_verified "
            f"(native_g1_pass={native_pass}, blockers={blockers})"
        )


def _receipt_path_relative_to_repo(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix().replace("\\", "/")


def _append_row_to_matched_swing_ledger(repo_root: Path, row: LedgerRow) -> None:
    ledger_path = default_ledger_path(repo_root)
    if ledger_path.is_file():
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger = Ledger.model_validate(payload)
        rows = [
            existing
            for existing in ledger.rows
            if existing.sha256 != row.sha256
            and existing.receipt_path != row.receipt_path
        ]
    else:
        rows = []
    rows.append(row)
    rows.sort(key=lambda item: item.receipt_path)
    updated = Ledger(
        schema_version="1.0.0",
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        total_receipts=len(rows),
        rows=rows,
    )
    updated.write_json(ledger_path)


def publish_club_only_ledger_row(
    view: ResultViewModel,
    *,
    receipt_path: str | Path,
    repo_root: Path | str | None = None,
) -> LedgerRow:
    """Publish a ledger row whose sha256 matches the receipt file bytes."""
    if not isinstance(view, ResultViewModel):
        raise TypeError("view must be ResultViewModel")
    if view.is_synthetic_seed:
        raise ValueError(
            "cannot publish or append ledger row for synthetic seed; "
            "must come from verified CO-03 sources ('retrieval' or 'constrained_ik')"
        )
    path = Path(receipt_path)
    if not str(path):
        raise ValueError("receipt_path must be non-empty")
    if not path.is_file():
        raise FileNotFoundError(
            f"receipt_path must exist before ledger publish: {path}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reason = "; ".join(view.qualification_blockers) or "software_contract_only"
    if repo_root is not None:
        rel_path = _receipt_path_relative_to_repo(path, Path(repo_root))
    else:
        rel_path = path.as_posix().replace("\\", "/")
    row = LedgerRow(
        receipt_path=rel_path,
        sha256=digest,
        engine=view.model_id,
        lane="club_only",
        capture=view.trial_id,
        candidate_sha=view.candidate_ids[0] if view.candidate_ids else None,
        metrics=SharedMetrics(),
        acceptance={
            "display_status": view.display_status.value,
            "native_g1_pass": view.native_g1_pass,
            "preset": view.preset.value,
        },
        artefacts=ArtefactPaths(),
        reason=reason,
    )
    if repo_root is not None:
        _append_row_to_matched_swing_ledger(Path(repo_root), row)
    return row


def write_club_only_result_package(
    view: ResultViewModel,
    receipt_path: str | Path,
) -> str:
    """Write ``view.as_dict()`` JSON and return sha256 of the exact file bytes."""
    if not isinstance(view, ResultViewModel):
        raise TypeError("view must be ResultViewModel")
    path = Path(receipt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = view.as_dict()
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ui_integration_evidence_payload(repo_root: Path | str) -> dict[str, Any]:
    """Versioned evidence payload for CO-09 software-contract qualification."""
    catalog = import_club_only_workbook_catalog(repo_root)
    session = create_club_only_session(
        trial_id="TW_wiffle",
        model_id="driven_double_pendulum",
        preset=MatchPreset.FAST_PREVIEW,
        prior_choices={"pose_prior": "address_plausible"},
        geometry_choices={"handedness": "right"},
    )
    observation = build_calibrated_observation_fixture("TW_wiffle")
    seed = _synthetic_seed(observation, session.model_id)
    result = run_club_only_ui_match(session, observation=observation, seed=seed)
    view = build_club_only_result_view(result)
    return {
        "schema_version": UI_SCHEMA,
        "governing_issue": _GOVERNING_ISSUE,
        "parent_epic": 10602,
        "fast_match_schema": FAST_MATCH_SCHEMA,
        "workbook_identity_schema": IDENTITY_SCHEMA,
        "native_g1_pass": False,
        "unique_trial_count": len(catalog.unique_trials),
        "conflicts": list(catalog.conflicts),
        "source_kinds": [k.value for k in list_motion_matching_source_kinds()],
        "keyboard_actions": dict(keyboard_action_map()),
        "sample_session": session.as_dict(),
        "sample_result": view.as_dict(),
        "legend": [e.as_dict() for e in build_observed_inferred_legend(view)],
        "qualification_blockers": list(_DEFAULT_BLOCKERS),
    }
