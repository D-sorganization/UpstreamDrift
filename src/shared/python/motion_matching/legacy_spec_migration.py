"""Migrate legacy golfer specs to SubjectProfile presets and canonical pins (#15673).

Right-handed legacy ``capsule``/``loft`` specs are upgraded to canonical-v2
skinned hands/body and flexible shaft, verified with :func:`require_canonical_golfer`,
and pinned as :class:`ModelDefinition` + :class:`GolferModelLock`. Left-handed or
non-migratable specs return ``status="deprecated_fallback"`` with an explicit reason.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from src.shared.python.core.handedness import subject_handedness
from src.shared.python.model_appearance.canonical_golfer_gate import (
    GolferModelLock,
    lock_for_spec,
    require_canonical_golfer,
)
from src.shared.python.motion_matching.execution.spec_revisions import (
    with_current_golfer_model,
)
from src.shared.python.motion_matching.model_library_contracts import (
    ModelDefinition,
    ModelLibraryManifest,
    SubjectProfile,
    _canonical_sha256,
    save_artifact,
    validate_immutable_ref,
)

SCHEMA_VERSION = "legacy-spec-migration/1.0.0"
MigrationStatus = Literal["migrated", "deprecated_fallback"]
DEFAULT_LEGACY_CONTROLLER_HASH = hashlib.sha256(b"legacy-ctrl-v1").hexdigest()
DEFAULT_LEGACY_SOLVER_HASH = hashlib.sha256(b"legacy-solver-v1").hexdigest()
_REQ_SECTIONS = ("bodies", "joints", "frames", "closure", "club")


@dataclass(frozen=True)
class LegacySpecMigrationResult:
    """Outcome of migrating one legacy golfer spec to canonical pins."""

    status: MigrationStatus
    reason: str
    source_spec_sha256: str
    subject_profile: SubjectProfile | None = None
    model_definition: ModelDefinition | None = None
    model_lock: GolferModelLock | None = None
    migrated_spec: Mapping[str, Any] | None = None
    receipt_sha256: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate status invariants, freeze migrated_spec, and compute receipt."""
        if self.status not in ("migrated", "deprecated_fallback"):
            raise ValueError(f"unknown migration status {self.status!r}")
        if not self.reason.strip():
            raise ValueError("migration result requires a non-empty reason")
        prof, mdef, lock = self.subject_profile, self.model_definition, self.model_lock
        spec = self.migrated_spec
        if self.status == "migrated":
            if None in (prof, mdef, lock, spec):
                raise ValueError("migrated result requires profile, model, lock, spec")
        elif mdef is not None or lock is not None:
            raise ValueError("deprecated_fallback cannot carry model_definition/lock")
        if spec is not None and not isinstance(spec, MappingProxyType):
            object.__setattr__(
                self, "migrated_spec", MappingProxyType(copy.deepcopy(dict(spec)))
            )
        if not self.receipt_sha256:
            payload = {
                "schema_version": self.schema_version,
                "status": self.status,
                "reason": self.reason,
                "source_spec_sha256": self.source_spec_sha256,
                "subject_sha256": prof.content_sha256 if prof else None,
                "model_sha256": mdef.content_sha256 if mdef else None,
                "model_lock": lock.to_dict() if lock else None,
            }
            object.__setattr__(self, "receipt_sha256", _canonical_sha256(payload))


@dataclass(frozen=True)
class BatchMigrationReport:
    """Deterministic summary and SHA-256 receipt for a batch of legacy specs."""

    source_paths: tuple[str, ...]
    results: tuple[LegacySpecMigrationResult, ...]
    total_count: int
    migrated_count: int
    deprecated_fallback_count: int
    receipt_sha256: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Verify batch counts and compute reproducible SHA-256 receipt."""
        n = len(self.results)
        mig_n = sum(1 for r in self.results if r.status == "migrated")
        ok = (
            len(self.source_paths) == n
            and self.total_count == n
            and self.migrated_count == mig_n
            and self.deprecated_fallback_count == (n - mig_n)
        )
        if not ok:
            raise ValueError("batch counts do not match results")
        if not self.receipt_sha256:
            entries = [
                {
                    "receipt_sha256": r.receipt_sha256,
                    "source_spec_sha256": r.source_spec_sha256,
                    "status": r.status,
                }
                for r in self.results
            ]
            payload = {
                "schema_version": self.schema_version,
                "total_count": n,
                "migrated_count": mig_n,
                "deprecated_fallback_count": n - mig_n,
                "entries": entries,
            }
            object.__setattr__(self, "receipt_sha256", _canonical_sha256(payload))


def _num(val: object, label: str) -> float:
    """Convert a scalar to float, rejecting booleans and non-numeric values."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{label} must be a positive finite number, got {val!r}")
    return float(val)


def _anthro_sha(spec: Mapping[str, Any], subj: Mapping[str, Any]) -> str:
    """Resolve or compute the canonical anthropometry SHA-256 digest."""
    anthro = spec.get("anthropometry")
    pools = [c for c in (subj, anthro, spec) if isinstance(c, Mapping)]
    for key in ("anthropometry_sha256", "de_leva_table_sha256"):
        for pool in pools:
            if (val := pool.get(key)) is not None:
                return validate_immutable_ref(val, label=key)
    if isinstance(anthro, Mapping) and anthro:
        return _canonical_sha256(anthro)
    return _canonical_sha256(subj)


def extract_subject_profile_from_legacy_spec(
    spec: Mapping[str, Any], *, subject_id: str | None = None
) -> SubjectProfile:
    """Extract stature, mass, handedness, and anthropometry SHA-256."""
    if not isinstance(spec, Mapping) or not isinstance(
        subj := spec.get("subject"), Mapping
    ):
        raise ValueError("legacy spec must contain a 'subject' mapping")
    h = _num(subj.get("stature_m", subj.get("height_m")), "stature_m")
    m = _num(subj.get("mass_kg"), "mass_kg")
    side = subject_handedness(spec)
    ref = _anthro_sha(spec, subj)
    sid = subject_id or next(
        (
            v.strip()
            for k in ("subject_id", "subject_preset", "name")
            if isinstance(v := subj.get(k), str) and v.strip()
        ),
        None,
    )
    if sid is None:
        key = {"handedness": side, "height_m": h, "mass_kg": m, "source_ref": ref}
        sid = f"legacy-{_canonical_sha256(key)[:12]}"
    return SubjectProfile(
        subject_id=sid, handedness=side, height_m=h, mass_kg=m, source_ref=ref
    )


def _fb(
    reason: str, sha: str, prof: SubjectProfile | None = None
) -> LegacySpecMigrationResult:
    """Build a deprecated_fallback migration result with an explicit reason."""
    return LegacySpecMigrationResult(
        status="deprecated_fallback",
        reason=reason,
        source_spec_sha256=sha,
        subject_profile=prof,
    )


def migrate_legacy_spec_to_canonical(
    spec: Mapping[str, Any],
    *,
    controller_hash: str,
    solver_hash: str,
    model_id: str | None = None,
    subject_id: str | None = None,
) -> LegacySpecMigrationResult:
    """Upgrade a right-handed legacy spec to canonical-v2 pins or return fallback."""
    ctrl = validate_immutable_ref(controller_hash, label="controller_hash")
    solv = validate_immutable_ref(solver_hash, label="solver_hash")
    src_sha = _canonical_sha256(spec) if isinstance(spec, Mapping) else ""
    try:
        prof = extract_subject_profile_from_legacy_spec(spec, subject_id=subject_id)
    except (ValueError, TypeError) as exc:
        return _fb(f"non-migratable legacy spec: {exc}", src_sha)
    if prof.handedness == "left":
        msg = "left-handed legacy spec uses deprecated fallback to avoid corrupting wrist geometry"
        return _fb(msg, src_sha, prof)
    miss = [k for k in _REQ_SECTIONS if not spec.get(k)]
    club = spec.get("club")
    if not miss and (not isinstance(club, Mapping) or not club.get("name")):
        miss.append("club.name")
    if miss:
        return _fb(f"non-migratable spec missing: {', '.join(miss)}", src_sha, prof)

    work = copy.deepcopy(dict(spec))
    if work.get("spec_revision") is None:
        work.pop("spec_revision", None)
    try:
        mig, _ = with_current_golfer_model(work, work["subject"])
        require_canonical_golfer(mig)
        cname = str(mig["club"]["name"]).replace("/", "_").replace("\\", "_")
        mid = model_id or f"legacy-{cname}-{src_sha[:12]}"
        mdef = ModelDefinition.from_spec(mig, model_id=mid, source_ref=src_sha)
        lock = lock_for_spec(mig, controller=ctrl, solver=solv)
    except (ValueError, KeyError, TypeError) as exc:
        return _fb(f"non-migratable legacy spec: {exc}", src_sha, prof)

    return LegacySpecMigrationResult(
        status="migrated",
        reason="migrated right-handed legacy spec to canonical-v2 skinned hands/body",
        source_spec_sha256=src_sha,
        subject_profile=prof,
        model_definition=mdef,
        model_lock=lock,
        migrated_spec=mdef.spec,
    )


def _migrate_path(p: Path, ctrl: str, solv: str) -> LegacySpecMigrationResult:
    """Load one spec JSON file and run migration, returning fallback on errors."""
    try:
        parsed = json.loads(p.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _fb(f"unreadable spec {p.name}: {exc}", "")
    if not isinstance(parsed, dict):
        return _fb(f"spec {p.name} must contain a JSON object", "")
    return migrate_legacy_spec_to_canonical(
        parsed, controller_hash=ctrl, solver_hash=solv
    )


def batch_migrate_legacy_specs(
    spec_paths: Sequence[Path | str],
    *,
    controller_hash: str = DEFAULT_LEGACY_CONTROLLER_HASH,
    solver_hash: str = DEFAULT_LEGACY_SOLVER_HASH,
    output_dir: Path | None = None,
) -> BatchMigrationReport:
    """Process spec paths deterministically and produce a reproducible SHA-256 report."""
    ctrl = validate_immutable_ref(controller_hash, label="controller_hash")
    solv = validate_immutable_ref(solver_hash, label="solver_hash")
    ordered = sorted((Path(p) for p in spec_paths), key=lambda p: p.as_posix())
    results = tuple(_migrate_path(p, ctrl, solv) for p in ordered)
    if output_dir is not None:
        for res in results:
            if res.subject_profile is not None:
                save_artifact(output_dir, res.subject_profile)
            if res.status == "migrated" and res.model_definition is not None:
                save_artifact(output_dir, res.model_definition)
        manifest = ModelLibraryManifest.rebuild_from_directory(output_dir)
        manifest.save(output_dir)
    mig_n = sum(1 for r in results if r.status == "migrated")
    return BatchMigrationReport(
        source_paths=tuple(p.as_posix() for p in ordered),
        results=results,
        total_count=len(results),
        migrated_count=mig_n,
        deprecated_fallback_count=len(results) - mig_n,
    )
