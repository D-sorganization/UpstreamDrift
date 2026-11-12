"""Registry and validation for tour-matching native evidence run scripts (#15015).

Evidence scripts under ``docs/development/simscape_tour_matching/native_evidence/``
must either import cleanly on the current branch or ship an ``evidence_script.json``
manifest that records archival status and the git ref where they last ran.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

MANIFEST_NAME = "evidence_script.json"
SCHEMA_VERSION = "motion-evidence-script/1"


class EvidenceScriptDisposition(str, Enum):
    """Whether an evidence script is runnable on the current checkout."""

    RUNNABLE = "runnable"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class EvidenceScriptManifest:
    """Parsed ``evidence_script.json`` beside a native evidence run."""

    script: str
    run_id: str
    status: EvidenceScriptDisposition
    required_git_ref: str
    missing_modules: tuple[str, ...]
    archived_reason: str = ""
    required_branch: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EvidenceScriptManifest:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported {MANIFEST_NAME} schema: {data.get('schema_version')!r}"
            )
        status = EvidenceScriptDisposition(str(data["status"]))
        modules = data.get("missing_modules") or []
        ref = str(data.get("required_git_ref", "")).strip()
        if status == EvidenceScriptDisposition.ARCHIVED and len(ref) != 40:
            raise ValueError(
                "archived manifest requires a 40-character required_git_ref"
            )
        if status == EvidenceScriptDisposition.ARCHIVED and not modules:
            raise ValueError("archived manifest requires missing_modules")
        return cls(
            script=str(data["script"]),
            run_id=str(data["run_id"]),
            status=status,
            required_git_ref=ref,
            missing_modules=tuple(str(m) for m in modules),
            archived_reason=str(data.get("archived_reason", "")),
            required_branch=str(data.get("required_branch", "")),
        )


@dataclass(frozen=True)
class EvidenceScriptValidation:
    """Outcome of validating one evidence run directory."""

    evidence_dir: Path
    disposition: EvidenceScriptDisposition
    script_path: Path
    detail: str
    valid: bool


def _module_importable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def load_manifest(evidence_dir: Path) -> EvidenceScriptManifest:
    manifest_path = evidence_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {MANIFEST_NAME} in {evidence_dir}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return EvidenceScriptManifest.from_mapping(data)


def validate_evidence_run_dir(
    evidence_dir: Path,
    *,
    repo_root: Path | None = None,
) -> EvidenceScriptValidation:
    """Fail closed unless the evidence script runs here or is explicitly archived."""
    manifest = load_manifest(evidence_dir)
    script_path = evidence_dir / manifest.script
    if not script_path.is_file():
        return EvidenceScriptValidation(
            evidence_dir=evidence_dir,
            disposition=manifest.status,
            script_path=script_path,
            detail=f"script file {manifest.script} not found",
            valid=False,
        )

    if manifest.status == EvidenceScriptDisposition.ARCHIVED:
        restored = [m for m in manifest.missing_modules if _module_importable(m)]
        if restored:
            return EvidenceScriptValidation(
                evidence_dir=evidence_dir,
                disposition=EvidenceScriptDisposition.ARCHIVED,
                script_path=script_path,
                detail=(
                    "archived manifest is stale: modules import on this branch "
                    f"({', '.join(restored)}); promote manifest to runnable"
                ),
                valid=False,
            )
        return EvidenceScriptValidation(
            evidence_dir=evidence_dir,
            disposition=EvidenceScriptDisposition.ARCHIVED,
            script_path=script_path,
            detail=(
                f"archived at {manifest.required_git_ref}; "
                f"{manifest.archived_reason or 'modules not on main'}"
            ),
            valid=True,
        )

    missing = [m for m in manifest.missing_modules if not _module_importable(m)]
    if missing:
        return EvidenceScriptValidation(
            evidence_dir=evidence_dir,
            disposition=EvidenceScriptDisposition.RUNNABLE,
            script_path=script_path,
            detail=f"declared runnable but cannot import: {', '.join(missing)}",
            valid=False,
        )

    _ = repo_root  # reserved for future PYTHONPATH checks
    return EvidenceScriptValidation(
        evidence_dir=evidence_dir,
        disposition=EvidenceScriptDisposition.RUNNABLE,
        script_path=script_path,
        detail="manifest marks script runnable and imports resolve",
        valid=True,
    )
