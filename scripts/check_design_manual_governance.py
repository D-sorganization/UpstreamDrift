#!/usr/bin/env python3
"""Fail closed when UpstreamDrift design-manual authorities drift."""

from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath
from typing import cast

from scripts.design_manual_contract import (
    EXPECTED_CONTRACTS,
    EXPECTED_POLICY_FIELDS,
    GENERATED_SUFFIXES,
    IMPACTED_PATHS,
    REQUIRED_EVIDENCE,
    REQUIRED_FORMATS,
    REQUIRED_UPDATE_FILES,
    DesignManualGovernanceError,
    DesignManualGovernanceSummary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PurePosixPath("scripts/config/design_manual_governance.json")


def _object(value: object, label: str, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DesignManualGovernanceError(f"{label} must be an object")
    result = cast(dict[str, object], value)
    actual = set(result)
    if actual != fields:
        raise DesignManualGovernanceError(
            f"{label} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return result


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise DesignManualGovernanceError(f"{label} must be an array")
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DesignManualGovernanceError(f"{label} must be a non-empty string")
    return value


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise DesignManualGovernanceError(
            f"{label} must be {expected!r}; got {value!r}"
        )


def _safe_path(value: object, label: str) -> PurePosixPath:
    text = _text(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in text
        or path.as_posix() != text
    ):
        raise DesignManualGovernanceError(f"{label} must be a normalized relative path")
    return path


def _verify_source(policy: dict[str, object]) -> tuple[str, PurePosixPath]:
    source = _object(
        policy["canonical_source"],
        "canonical source",
        {
            "manual_id",
            "format",
            "path",
            "repository_role",
            "existing_documentation_relationship",
        },
    )
    _equal(source["format"], "qmd", "canonical source format")
    _equal(
        source["repository_role"],
        "scientific-computation-and-product-authority",
        "repository role",
    )
    _equal(
        source["existing_documentation_relationship"],
        "separate-products-not-manual-authority",
        "existing documentation relationship",
    )
    manual_id = _text(source["manual_id"], "manual ID")
    source_path = _safe_path(source["path"], "source path")
    _equal(manual_id, "upstreamdrift", "manual ID")
    _equal(source_path, PurePosixPath("manuals/upstreamdrift"), "source path")
    return manual_id, source_path


def _verify_outputs(policy: dict[str, object]) -> None:
    outputs = _object(
        policy["generated_outputs"],
        "generated outputs",
        {"editable", "required_release_formats", "current_artifact_status"},
    )
    if outputs["editable"] is not False:
        raise DesignManualGovernanceError("generated outputs must not be editable")
    _equal(outputs["required_release_formats"], REQUIRED_FORMATS, "release formats")
    _equal(
        outputs["current_artifact_status"],
        "not-generated-not-approved",
        "artifact status",
    )


def _verify_freshness(policy: dict[str, object]) -> None:
    freshness = _object(
        policy["freshness"],
        "freshness",
        {"enforcement", "current_gate", "impacted_paths", "exemptions"},
    )
    _equal(freshness["enforcement"], "release-blocking", "freshness enforcement")
    _equal(freshness["current_gate"], "blocked-pending-UP-D7", "freshness gate")
    _equal(freshness["exemptions"], "structured-owned-expiring-only", "exemptions")
    paths = [
        _safe_path(item, "impacted path")
        for item in _array(freshness["impacted_paths"], "impacted paths")
    ]
    _equal(paths, list(map(PurePosixPath, IMPACTED_PATHS)), "impacted paths")


def _verify_publication(policy: dict[str, object]) -> bool:
    publication = _object(
        policy["publication"],
        "publication",
        {
            "default",
            "public_projection_allowed",
            "catalog_repository",
            "projection_manifest_path",
            "current_approval",
            "required_evidence",
        },
    )
    _equal(publication["default"], "deny-until-approved", "publication default")
    allowed = publication["public_projection_allowed"]
    if allowed is not False:
        raise DesignManualGovernanceError("public projection must remain blocked")
    _equal(
        publication["catalog_repository"],
        EXPECTED_CONTRACTS["owner_repository"],
        "catalog",
    )
    _equal(
        _safe_path(publication["projection_manifest_path"], "projection manifest"),
        PurePosixPath("manuals/upstreamdrift/publication-projection.json"),
        "projection manifest",
    )
    _equal(
        publication["current_approval"],
        "blocked-pending-UP-D5-D7-D8",
        "publication approval",
    )
    evidence = [
        _text(item, "evidence item")
        for item in _array(publication["required_evidence"], "required evidence")
    ]
    _equal(evidence, REQUIRED_EVIDENCE, "required evidence")
    return False


def _verify_quality_and_git(policy: dict[str, object]) -> None:
    quality = _object(
        policy["quality"],
        "quality",
        {"python_version", "python_formatter", "principles"},
    )
    _equal(quality["python_version"], "3.12", "Python version")
    _equal(quality["python_formatter"], "ruff format", "python formatter")
    _equal(quality["principles"], ["TDD", "DbC", "DRY", "LoD"], "principles")
    git = _object(
        policy["git"],
        "git",
        {"default_branch", "pull_request_target", "direct_push", "auto_merge"},
    )
    _equal(git["default_branch"], "main", "default branch")
    _equal(git["pull_request_target"], "main", "pull request target")
    _equal(git["direct_push"], False, "direct push")
    _equal(git["auto_merge"], False, "auto-merge")


def verify_governance_policy(policy: object) -> tuple[str, PurePosixPath, bool]:
    """Validate producer, contract, artifact, freshness, and release boundaries."""
    document = _object(policy, "governance policy", EXPECTED_POLICY_FIELDS)
    _equal(
        document["schema_version"],
        "upstreamdrift/design-manual-governance/1.0.0",
        "schema version",
    )
    program = _object(
        document["program"], "program", {"epic", "current_subepic", "next_subepic"}
    )
    _equal(
        program,
        {"epic": 9064, "current_subepic": 9066, "next_subepic": 9067},
        "program",
    )
    manual_id, source_path = _verify_source(document)
    _equal(document["contracts"], EXPECTED_CONTRACTS, "shared contracts")
    inventory = _object(
        document["calculation_inventory"], "inventory", {"path", "owner_subepic"}
    )
    _equal(
        _safe_path(inventory["path"], "inventory path"),
        PurePosixPath("manuals/upstreamdrift/calculation-registry.json"),
        "inventory path",
    )
    _equal(inventory["owner_subepic"], 9067, "inventory owner")
    _verify_outputs(document)
    _verify_freshness(document)
    allowed = _verify_publication(document)
    _verify_quality_and_git(document)
    _object(
        document["agent_context"],
        "agent context",
        {"required_update_files", "required_gate"},
    )
    return manual_id, source_path, allowed


def _valid_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_calculation_registry(registry: object) -> int:
    """Validate the shared registry envelope without copying its owned schema."""
    document = _object(
        registry,
        "calculation registry",
        {
            "schema_version",
            "manual_id",
            "repository",
            "release_status",
            "inventory_commit",
            "blockers",
            "calculations",
        },
    )
    _equal(document["schema_version"], "1.0.0", "registry schema version")
    _equal(document["manual_id"], "upstreamdrift", "registry manual ID")
    _equal(document["repository"], "D-sorganization/UpstreamDrift", "repository")
    calculations = _array(document["calculations"], "calculations")
    blockers = _array(document["blockers"], "blockers")
    status = _text(document["release_status"], "release status")
    if status == "approved":
        if (
            not calculations
            or blockers
            or not _valid_revision(document["inventory_commit"])
        ):
            raise DesignManualGovernanceError(
                "approved registry requires calculations, immutable commit, and no blockers"
            )
    elif status == "blocked-inventory-required":
        if calculations or not blockers or document["inventory_commit"] is not None:
            raise DesignManualGovernanceError(
                "blocked registry requires no calculations, null commit, and blockers"
            )
    else:
        raise DesignManualGovernanceError("release status is unsupported in UP-D0")
    for blocker in blockers:
        item = _object(blocker, "registry blocker", {"id", "owner", "resolution"})
        for field, value in item.items():
            _text(value, f"blocker {field}")
    return len(calculations)


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_manual_tree(root: Path, source_path: PurePosixPath) -> int:
    manual_root = root.joinpath(*source_path.parts)
    if not manual_root.is_dir():
        raise DesignManualGovernanceError(f"canonical source is missing: {source_path}")
    forbidden = sorted(
        path.relative_to(root).as_posix()
        for path in manual_root.rglob("*")
        if path.is_file() and path.suffix.lower() in GENERATED_SUFFIXES
    )
    if forbidden:
        raise DesignManualGovernanceError(
            f"editable generated artifacts are forbidden: {forbidden}"
        )
    qmd_paths = sorted(manual_root.rglob("*.qmd"))
    if not qmd_paths:
        raise DesignManualGovernanceError("canonical manual must contain QMD source")
    return len(qmd_paths)


def _verify_context(root: Path, policy: dict[str, object]) -> None:
    context = _object(
        policy["agent_context"],
        "agent context",
        {"required_update_files", "required_gate"},
    )
    paths = [
        _safe_path(item, "required update file")
        for item in _array(context["required_update_files"], "required update files")
    ]
    _equal(
        paths, list(map(PurePosixPath, REQUIRED_UPDATE_FILES)), "required update files"
    )
    for path in paths:
        if not root.joinpath(*path.parts).is_file():
            raise DesignManualGovernanceError(
                f"required update file is missing: {path}"
            )
    _equal(
        context["required_gate"],
        "python3 -m scripts.check_design_manual_governance",
        "required gate",
    )
    for name in ("AGENTS.md", "CLAUDE.md"):
        text = (root / name).read_text(encoding="utf-8")
        for phrase in (
            "manuals/upstreamdrift",
            "scripts.check_design_manual_governance",
        ):
            if phrase not in text:
                raise DesignManualGovernanceError(f"{name} is missing manual context")


def verify_repository(root: Path = REPO_ROOT) -> DesignManualGovernanceSummary:
    """Verify repository adoption while retaining the explicit release block."""
    policy = _object(
        _load(root.joinpath(*POLICY_PATH.parts)),
        "governance policy",
        EXPECTED_POLICY_FIELDS,
    )
    manual_id, source_path, allowed = verify_governance_policy(policy)
    inventory = _object(
        policy["calculation_inventory"], "inventory", {"path", "owner_subepic"}
    )
    registry_path = _safe_path(inventory["path"], "inventory path")
    registry = _load(root.joinpath(*registry_path.parts))
    calculation_count = verify_calculation_registry(registry)
    qmd_count = _verify_manual_tree(root, source_path)
    _verify_context(root, policy)
    for schema in (
        root / "schemas" / "calculation-registry.schema.json",
        root / "schemas" / "publication-projection.schema.json",
    ):
        if schema.exists():
            raise DesignManualGovernanceError(
                f"program-owned schema copy is forbidden: {schema.relative_to(root)}"
            )
    publication = cast(dict[str, object], policy["publication"])
    manifest_path = _safe_path(publication["projection_manifest_path"], "manifest path")
    if not allowed and root.joinpath(*manifest_path.parts).exists():
        raise DesignManualGovernanceError(
            "blocked publication must not have a projection manifest"
        )
    registry_object = cast(dict[str, object], registry)
    return DesignManualGovernanceSummary(
        manual_id=manual_id,
        canonical_qmd_count=qmd_count,
        calculation_count=calculation_count,
        release_status=_text(registry_object["release_status"], "release status"),
        public_projection_allowed=allowed,
    )


def main() -> int:
    """Run the governance gate with deterministic diagnostics."""
    try:
        summary = verify_repository()
    except (DesignManualGovernanceError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "Design-manual governance verified: "
        f"{summary.canonical_qmd_count} QMD sources, "
        f"{summary.calculation_count} registered calculations, "
        f"release={summary.release_status}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
