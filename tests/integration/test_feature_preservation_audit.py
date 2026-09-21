"""Integration and acceptance tests for feature preservation audit (#10533).

Epic: #10508
Issue: #10533 ([ORG-24] Reconcile, Audit, and Freeze Feature Preservation Across All Historical Boundaries)
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

from src.config.capability_migration import (
    CapabilityMigrationInventory,
    FixtureEntry,
    MigrationEntry,
    load_migration_inventory,
)
from src.shared.python.workspace.feature_preservation_audit import (
    AuditFailureError,
    AuditReport,
    AuditStatus,
    FeaturePreservationAuditor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_JSON = REPO_ROOT / "src" / "config" / "capability_migration.json"
MODELS_YAML = REPO_ROOT / "src" / "config" / "models.yaml"


@pytest.fixture(scope="module")
def auditor() -> FeaturePreservationAuditor:
    """Create a default auditor loaded with the repository's migration inventory."""
    return FeaturePreservationAuditor(
        repo_root=REPO_ROOT, inventory_path=MIGRATION_JSON
    )


# ============================================================================
# RED Test Cases (Verification of contract enforcement & failure detection)
# ============================================================================


def test_red_missing_baseline_capability_fails_audit(
    auditor: FeaturePreservationAuditor,
) -> None:
    """Audit must fail closed when a baseline capability is missing from the active catalog."""
    mock_auditor = copy.deepcopy(auditor)
    # Simulate a baseline entry silently disappearing
    if "c3d_viewer" in mock_auditor.inventory.entries:
        del mock_auditor.inventory.entries["c3d_viewer"]

    with pytest.raises(
        AuditFailureError, match="Missing baseline capability.*c3d_viewer"
    ):
        mock_auditor.audit_baseline_reconciliation()


def test_red_corrupted_golden_fixture_fails_audit(
    auditor: FeaturePreservationAuditor,
) -> None:
    """Audit must fail closed when a golden fixture's byte hash differs from golden truth."""
    mock_auditor = copy.deepcopy(auditor)
    # Mutate a fixture hash in the inventory
    if "pose.test_poses" in mock_auditor.inventory.fixtures:
        orig = mock_auditor.inventory.fixtures["pose.test_poses"]
        mock_auditor.inventory.fixtures["pose.test_poses"] = FixtureEntry(
            path=orig.path,
            kind=orig.kind,
            sha256="deadbeef" * 8,
            size_bytes=orig.size_bytes,
            support_status=orig.support_status,
        )

    with pytest.raises(
        AuditFailureError, match="Fixture hash mismatch.*pose.test_poses"
    ):
        mock_auditor.audit_preservation_fixtures()


def test_red_alias_cycle_fails_audit(auditor: FeaturePreservationAuditor) -> None:
    """Audit must fail closed when cyclic aliases are detected in migration inventory."""
    mock_auditor = copy.deepcopy(auditor)
    mock_auditor.inventory.entries["cycle_a"] = MigrationEntry(
        id="cycle_a",
        name="Cycle A",
        entity_kind="tile",
        primary_workspace="simulation",
        lifecycle="deprecated_alias",
        alias_target="cycle_b",
    )
    mock_auditor.inventory.entries["cycle_b"] = MigrationEntry(
        id="cycle_b",
        name="Cycle B",
        entity_kind="tile",
        primary_workspace="simulation",
        lifecycle="deprecated_alias",
        alias_target="cycle_a",
    )

    with pytest.raises(AuditFailureError, match="Alias cycle detected"):
        mock_auditor.audit_alias_graphs()


def test_red_unregistered_workspace_domain_fails_audit(
    auditor: FeaturePreservationAuditor,
) -> None:
    """Audit must fail closed when a capability declares an unknown workspace domain."""
    mock_auditor = copy.deepcopy(auditor)
    with pytest.raises(ValueError, match="Unknown primary_workspace"):
        mock_auditor.inventory.add_entry(
            MigrationEntry(
                id="invalid_workspace_entry",
                name="Invalid Domain",
                entity_kind="tile",
                primary_workspace="unregistered_domain_xyz",
            )
        )


# ============================================================================
# GREEN Acceptance Cases (Validation of preserved features, contracts, and report)
# ============================================================================


def test_green_reconcile_all_org01_baseline_capabilities(
    auditor: FeaturePreservationAuditor,
) -> None:
    """All baseline capabilities must be reconciled with zero unaccounted entries and valid lifecycles."""
    res = auditor.audit_baseline_reconciliation()
    assert res.status == AuditStatus.PASSED
    assert res.details["total_entries"] >= 159
    assert res.details["zero_silent_removals"] is True


def test_green_golden_preservation_fixtures_byte_exact_integrity(
    auditor: FeaturePreservationAuditor,
) -> None:
    """All supported preservation fixtures must match their golden baseline byte hashes exactly."""
    res = auditor.audit_preservation_fixtures()
    assert res.status == AuditStatus.PASSED
    assert res.details["verified_count"] >= 10
    assert res.details["mismatches"] == 0


def test_green_acyclic_alias_resolution_to_canonical_targets(
    auditor: FeaturePreservationAuditor,
) -> None:
    """All legacy aliases must resolve acyclically and transitively to active canonical targets."""
    res = auditor.audit_alias_graphs()
    assert res.status == AuditStatus.PASSED
    assert res.details["alias_count"] >= 2
    # Verify specific known aliases
    assert (
        res.details["resolved_targets"]["starting_pose_matcher"]
        == "motion_target_preview"
    )
    assert res.details["resolved_targets"]["putting_green_gui"] == "putting_green"


def test_green_five_workspaces_architectural_boundaries(
    auditor: FeaturePreservationAuditor,
) -> None:
    """All 5 core workspaces must have non-empty capability allocations and explicit boundaries."""
    res = auditor.audit_five_workspaces()
    assert res.status == AuditStatus.PASSED
    dist = res.details["workspace_distribution"]
    assert dist["simulation"] > 0
    assert dist["analysis"] > 0
    assert dist["capture"] > 0
    assert dist["putting"] > 0
    assert dist["training"] > 0
    assert dist["governance"] > 0


def test_green_external_dependencies_and_engine_qualification(
    auditor: FeaturePreservationAuditor,
) -> None:
    """External dependencies and physics engines must report honest qualification statuses."""
    res = auditor.audit_external_dependencies()
    assert res.status == AuditStatus.PASSED
    engines = res.details["engines"]
    assert "mujoco" in engines
    assert "pinocchio" in engines
    assert "drake" in engines
    assert "opensim" in engines
    assert "simscape" in engines


def test_green_saved_layouts_and_preferences_compatibility(
    auditor: FeaturePreservationAuditor,
) -> None:
    """Historical user layout presets must resolve aliases cleanly without dropping tiles."""
    layouts = auditor.inventory.saved_layouts
    assert "default" in layouts
    assert "analysis_compact" in layouts
    assert "simulation_suite" in layouts

    resolved_default = auditor.inventory.validate_saved_layout(layouts["default"])
    assert "motion_target_preview" in resolved_default
    assert "putting_green" in resolved_default


def test_green_generate_and_publish_feature_preservation_report(
    tmp_path: Path, auditor: FeaturePreservationAuditor
) -> None:
    """Auditor must generate and publish a complete, frozen feature preservation report."""
    report = auditor.generate_full_audit_report()
    assert report.is_approved is True
    assert report.total_capabilities >= 159
    assert report.active_capabilities > 0
    assert report.golden_fixtures_verified >= 10

    # Publish to a test output path
    test_md_path = tmp_path / "ORG24_FEATURE_PRESERVATION_AUDIT.md"
    published_path = auditor.publish_audit_report(output_path=test_md_path)
    assert published_path.is_file()

    content = published_path.read_text(encoding="utf-8")
    assert (
        "# ORG-24: Feature Preservation Audit & Historical Boundary Freeze" in content
    )
    assert "✅ **APPROVED (ALL GATES PASSED)**" in content
    assert "## Detailed Audit Section Findings" in content
    assert "## Scientific Release Governance Note" in content
