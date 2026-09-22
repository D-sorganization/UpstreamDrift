"""Fail-closed audit of neural datasets, checkpoints and claims (NM-00 #10615)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.motion_matching.dataset import (
    SCHEMA_VERSION as SWEEP_SCHEMA_VERSION,
)
from src.shared.python.motion_matching.provenance import git_commit_short

from .catalog import (
    HISTORICAL_DESIGN_ISSUES,
    CatalogEntry,
    default_catalog,
)
from .claims import classify_training_claim, list_known_claim_ids
from .types import (
    ArtifactIdentity,
    ArtifactKind,
    ArtifactRole,
    ClaimStatus,
    Disposition,
)

AUDIT_SCHEMA = "neural-artifact-audit/1.0.0"


@dataclass(frozen=True)
class ParquetInspectResult:
    """Bounded parquet metadata + sample inspection (no full materialisation)."""

    path: Path
    exists: bool
    content_sha256: str | None
    num_row_groups: int
    num_rows: int
    sampled_rows: int
    schema_names: tuple[str, ...]
    metadata_summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "content_sha256": self.content_sha256,
            "num_row_groups": self.num_row_groups,
            "num_rows": self.num_rows,
            "sampled_rows": self.sampled_rows,
            "schema_names": list(self.schema_names),
            "metadata_summary": self.metadata_summary,
        }


@dataclass(frozen=True)
class ArtifactAuditReceipt:
    """Versioned inventory receipt for NM-00."""

    schema: str
    repo_root: str
    source_revision: str
    artifacts: tuple[ArtifactIdentity, ...]
    historical_design_issues: tuple[str, ...]

    def artifact(self, artifact_id: str) -> ArtifactIdentity:
        for row in self.artifacts:
            if row.artifact_id == artifact_id:
                return row
        raise KeyError(f"artifact not in receipt: {artifact_id}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "repo_root": self.repo_root,
            "source_revision": self.source_revision,
            "historical_design_issues": list(self.historical_design_issues),
            "artifacts": [a.as_dict() for a in self.artifacts],
        }

    def write_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2, allow_nan=False)
            f.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@precondition(
    lambda path, expected_sha256=None, max_rows=8: max_rows > 0,
    "max_rows must be positive",
)
@postcondition(
    lambda result: result.sampled_rows >= 0 and result.num_row_groups >= 0,
    "inspect result must report non-negative row counts",
)
def inspect_parquet_bounded(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    max_rows: int = 8,
) -> ParquetInspectResult:
    """Inspect parquet metadata/row groups and a bounded row sample.

    Fail-closed: when ``expected_sha256`` is provided and does not match the
    file bytes, raises ``ValueError`` with ``identity mismatch``.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"parquet not found: {target}")

    content_sha = _sha256_file(target)
    if expected_sha256 is not None and expected_sha256.lower() != content_sha.lower():
        raise ValueError(
            f"identity mismatch for {target}: expected sha256 "
            f"{expected_sha256}, got {content_sha}"
        )

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(target)
    schema_names = tuple(parquet_file.schema_arrow.names)
    num_rows = int(parquet_file.metadata.num_rows)
    num_row_groups = int(parquet_file.num_row_groups)
    take = min(max_rows, num_rows)
    if take > 0:
        # Use iter_batches to avoid loading an entire row group into memory.
        # Row-group sizes are writer-controlled; a single group can be hundreds
        # of megabytes. iter_batches streams rows within the batch_size limit.
        sampled_rows = 0
        for batch in parquet_file.iter_batches(batch_size=take):
            sampled_rows = int(batch.num_rows)
            break
    else:
        sampled_rows = 0

    summary = {
        "created_by": parquet_file.metadata.created_by,
        "format_version": parquet_file.metadata.format_version,
        "serialized_size": parquet_file.metadata.serialized_size,
    }
    return ParquetInspectResult(
        path=target.resolve(),
        exists=True,
        content_sha256=content_sha,
        num_row_groups=num_row_groups,
        num_rows=num_rows,
        sampled_rows=sampled_rows,
        schema_names=schema_names,
        metadata_summary=summary,
    )


@dataclass(frozen=True)
class _PresenceOutcome:
    """Varying presence / claim fields for a catalog-derived identity row."""

    exists: bool
    content_sha256: str | None
    disposition: Disposition
    claim_status: ClaimStatus
    blockers: tuple[str, ...]
    trial_ancestry: str | None = None
    schema_version: str | None = None


def _resolve_path(repo_root: Path, entry: CatalogEntry) -> Path:
    if entry.path_is_absolute:
        return entry.path_spec
    return (repo_root / entry.path_spec).resolve()


def _identity_from_entry(
    entry: CatalogEntry,
    path: Path | None,
    source_revision: str | None,
    outcome: _PresenceOutcome,
) -> ArtifactIdentity:
    """Build ArtifactIdentity from a catalog entry plus a presence outcome."""
    return ArtifactIdentity(
        artifact_id=entry.artifact_id,
        kind=entry.kind,
        role=entry.role,
        path=path,
        exists=outcome.exists,
        content_sha256=outcome.content_sha256,
        schema_version=(
            outcome.schema_version
            if outcome.schema_version is not None
            else entry.schema_version
        ),
        model_ids=(),
        engine=entry.engine,
        source_revision=source_revision,
        units=entry.units,
        seeds=(),
        trial_ancestry=outcome.trial_ancestry,
        required_channels=entry.required_channels,
        label_availability=entry.label_availability,
        is_synthetic_fixture=entry.is_synthetic_fixture,
        disposition=outcome.disposition,
        claim_status=outcome.claim_status,
        blockers=outcome.blockers,
        retrieval_instructions=entry.retrieval_instructions,
        notes=entry.notes,
        related_issues=entry.related_issues,
    )


def _quarantine_absent(
    entry: CatalogEntry,
    path: Path,
    source_revision: str,
    blocker: str,
) -> ArtifactIdentity:
    return _identity_from_entry(
        entry,
        path,
        source_revision,
        _PresenceOutcome(
            exists=False,
            content_sha256=None,
            disposition=Disposition.QUARANTINE,
            claim_status=ClaimStatus.UNSUPPORTED,
            blockers=(blocker,),
        ),
    )


def _audit_sweep_folder(
    entry: CatalogEntry,
    folder: Path,
    *,
    source_revision: str,
) -> ArtifactIdentity:
    trials = folder / "trials.parquet"
    timesteps = folder / "timesteps.parquet"
    if not trials.is_file() or not timesteps.is_file():
        return _quarantine_absent(
            entry,
            folder,
            source_revision,
            "sweep folder missing trials.parquet or timesteps.parquet",
        )

    trials_meta = inspect_parquet_bounded(trials, max_rows=4)
    timesteps_meta = inspect_parquet_bounded(timesteps, max_rows=4)
    blockers: list[str] = []
    channel_names = set(timesteps_meta.schema_names)
    for channel in entry.required_channels:
        if channel not in channel_names:
            blockers.append(f"missing required channel: {channel}")

    claim_status = (
        ClaimStatus.SOFTWARE_CONTRACT_ONLY
        if entry.is_synthetic_fixture
        else ClaimStatus.REPRODUCED_LOCALLY
    )
    disposition = Disposition.RETAIN if not blockers else Disposition.REPAIR
    if entry.is_synthetic_fixture:
        blockers.append(
            "synthetic fixture cannot certify native physical supervision",
        )

    combined = hashlib.sha256()
    combined.update((trials_meta.content_sha256 or "").encode("ascii"))
    combined.update((timesteps_meta.content_sha256 or "").encode("ascii"))

    return _identity_from_entry(
        entry,
        folder,
        source_revision,
        _PresenceOutcome(
            exists=True,
            content_sha256=combined.hexdigest(),
            disposition=disposition,
            claim_status=claim_status,
            blockers=tuple(blockers),
            trial_ancestry=(
                "synthetic-fixture" if entry.is_synthetic_fixture else None
            ),
            schema_version=entry.schema_version or SWEEP_SCHEMA_VERSION,
        ),
    )


def _audit_present_file(
    entry: CatalogEntry,
    path: Path,
    source_revision: str,
) -> ArtifactIdentity:
    content_sha = _sha256_file(path)
    disposition = Disposition.RETAIN
    claim_status = ClaimStatus.REPRODUCED_LOCALLY
    blockers: list[str] = []
    if entry.kind == ArtifactKind.CHECKPOINT:
        disposition = Disposition.MIGRATE
        claim_status = ClaimStatus.SOFTWARE_CONTRACT_ONLY
        blockers.append(
            "checkpoint present but not yet schema-qualified under NM-00; "
            "load only via weights_only helpers before any training claim"
        )
    elif entry.role == ArtifactRole.COMPACT_CORPUS:
        # Fail-closed: compact corpus requires parquet inspection and provenance
        # before any training-adjacent status. Hash alone is not sufficient.
        disposition = Disposition.MIGRATE
        claim_status = ClaimStatus.SOFTWARE_CONTRACT_ONLY
        try:
            inspect_result = inspect_parquet_bounded(
                path, expected_sha256=content_sha, max_rows=8
            )
            if inspect_result.num_rows == 0:
                blockers.append("parquet file is empty")
            if not inspect_result.schema_names:
                blockers.append("parquet schema has no columns")
        except (FileNotFoundError, ValueError, Exception) as exc:
            blockers.append(f"parquet inspection failed: {exc}")
        blockers.append(
            "compact corpus present but provenance (model release, seeds, "
            "geometry, contact settings) not yet established; "
            "do not cite as training evidence"
        )
    return _identity_from_entry(
        entry,
        path,
        source_revision,
        _PresenceOutcome(
            exists=True,
            content_sha256=content_sha,
            disposition=disposition,
            claim_status=claim_status,
            blockers=tuple(blockers),
        ),
    )


def _audit_catalog_entry(
    entry: CatalogEntry,
    repo_root: Path,
    *,
    source_revision: str,
) -> ArtifactIdentity:
    path = _resolve_path(repo_root, entry)

    if entry.is_sweep_folder:
        if path.is_dir():
            return _audit_sweep_folder(entry, path, source_revision=source_revision)
        return _quarantine_absent(
            entry, path, source_revision, f"absent sweep folder: {path}"
        )

    if not path.exists():
        return _quarantine_absent(
            entry, path, source_revision, f"absent documented path: {path}"
        )

    if path.is_file():
        return _audit_present_file(entry, path, source_revision)

    return _identity_from_entry(
        entry,
        path,
        source_revision,
        _PresenceOutcome(
            exists=True,
            content_sha256=None,
            disposition=Disposition.REPAIR,
            claim_status=ClaimStatus.SOFTWARE_CONTRACT_ONLY,
            blockers=("path exists but is not a regular file; repair inventory",),
        ),
    )


def _historical_claim_rows() -> Iterable[ArtifactIdentity]:
    for issue in HISTORICAL_DESIGN_ISSUES:
        yield ArtifactIdentity(
            artifact_id=f"claim.historical_design_{issue.lstrip('#')}",
            kind=ArtifactKind.TRAINING_CLAIM,
            role=ArtifactRole.HISTORICAL_ISSUE,
            path=None,
            exists=False,
            content_sha256=None,
            schema_version=None,
            model_ids=(),
            engine=None,
            source_revision=None,
            units=None,
            seeds=(),
            trial_ancestry=None,
            required_channels=(),
            label_availability={},
            is_synthetic_fixture=False,
            disposition=Disposition.QUARANTINE,
            claim_status=ClaimStatus.UNSUPPORTED,
            blockers=(
                f"{issue} supplies prior design notes only; not current "
                "speed/accuracy evidence",
            ),
            retrieval_instructions=(
                f"Read GitHub issue {issue} for historical design context only."
            ),
            notes="Historical design issue; do not cite as qualified training.",
            related_issues=(issue, "#10615"),
        )


@precondition(
    lambda repo_root: Path(repo_root).is_dir(),
    "repo_root must be an existing directory",
)
@postcondition(
    lambda result: result.schema == AUDIT_SCHEMA,
    "receipt must follow neural-artifact-audit schema",
)
@postcondition(
    lambda result: all(
        a.claim_status != ClaimStatus.NATIVE_QUALIFIED for a in result.artifacts
    ),
    "NM-00 audit must not mint native-qualified claims",
)
def audit_neural_artifacts(repo_root: Path | str) -> ArtifactAuditReceipt:
    """Inventory configured datasets, checkpoints and training claims.

    Fail-closed: absent documented corpora are quarantined; synthetic fixtures
    stay software-contract-only; plateau notes stay NOTE_ONLY.
    """
    root = Path(repo_root).resolve()
    revision = git_commit_short() or "unknown"
    rows: list[ArtifactIdentity] = []

    for entry in default_catalog():
        rows.append(_audit_catalog_entry(entry, root, source_revision=revision))

    for claim_id in list_known_claim_ids():
        rows.append(classify_training_claim(claim_id, repo_root=root))

    rows.extend(_historical_claim_rows())

    return ArtifactAuditReceipt(
        schema=AUDIT_SCHEMA,
        repo_root=str(root),
        source_revision=revision,
        artifacts=tuple(rows),
        historical_design_issues=HISTORICAL_DESIGN_ISSUES,
    )
