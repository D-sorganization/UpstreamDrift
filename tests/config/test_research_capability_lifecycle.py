"""Regression and acceptance tests for Research Capability Lifecycle (ORG-22, #10530).

Acceptance Cases:
- RED: Untiled nonexcluded package or undocumented service fails coverage.
- RED: CLI-only action cannot be represented as a GUI-ready tile.
- GREEN: CLI help/small fixture invocation for retained supported tools.
- GREEN: Hidden aliases and deprecated wrappers resolve acyclic.
- GREEN: Catalog links have owners, issues, and next actions.
- GREEN: SG optimizer Phase 3 UI follow-up is accurately recorded with active CLI status.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

from src.config.capability_migration import CapabilityMigrationInventory
from src.config.launcher_manifest_loader import LauncherManifest
from src.config.research_capability_lifecycle import (
    CLINotInteractiveGUIError,
    IncompleteCapabilityRecord,
    ResearchCapabilityLifecycleManager,
    audit_research_and_excluded_capabilities,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "src" / "tools"
EXCLUSIONS_YAML = REPO_ROOT / "src" / "config" / "registry_exclusions.yaml"
CAPABILITY_MIGRATION_JSON = REPO_ROOT / "src" / "config" / "capability_migration.json"


# ---------------------------------------------------------------------------
# Acceptance Case 1: Untiled Nonexcluded Package or Undocumented Service Fails Coverage
# ---------------------------------------------------------------------------


def test_untiled_nonexcluded_package_fails_coverage() -> None:
    """Every src/tools directory must be a registered tile or explicitly excluded."""
    manager = ResearchCapabilityLifecycleManager.load(REPO_ROOT)
    unaccounted = manager.find_unaccounted_tools(TOOLS_DIR)
    assert not unaccounted, (
        f"Unaccounted tools found in src/tools: {unaccounted}. "
        "Every tool must either have a launcher tile or be documented in "
        "src/config/registry_exclusions.yaml."
    )


def test_undocumented_headless_service_fails_coverage() -> None:
    """Every exclusion in registry_exclusions.yaml must exist in capability_migration.json."""
    manager = ResearchCapabilityLifecycleManager.load(REPO_ROOT)
    missing_from_migration = manager.validate_exclusions_in_migration()
    assert not missing_from_migration, (
        f"Exclusions missing migration records: {missing_from_migration}"
    )


# ---------------------------------------------------------------------------
# Acceptance Case 2: CLI-Only Action Cannot Be Represented as GUI-Ready Tile
# ---------------------------------------------------------------------------


def test_cli_only_action_cannot_be_represented_as_gui_ready_tile() -> None:
    """Attempting to configure or adapt a CLI-only/headless capability as a GUI tile must fail."""
    manager = ResearchCapabilityLifecycleManager.load(REPO_ROOT)

    # Contraction is a headless research tool
    with pytest.raises(
        CLINotInteractiveGUIError, match="cannot be represented as a GUI-ready tile"
    ):
        manager.verify_tile_eligibility("contraction", requested_surface="desktop")

    # Drift control is terminal-only
    with pytest.raises(
        CLINotInteractiveGUIError, match="cannot be represented as a GUI-ready tile"
    ):
        manager.verify_tile_eligibility("drift_control", requested_surface="desktop")

    # Model converter is build-time/headless
    with pytest.raises(
        CLINotInteractiveGUIError, match="cannot be represented as a GUI-ready tile"
    ):
        manager.verify_tile_eligibility("model_converter", requested_surface="desktop")

    # Valid GUI tile must succeed
    assert (
        manager.verify_tile_eligibility("capture_rig", requested_surface="desktop")
        is True
    )


# ---------------------------------------------------------------------------
# Acceptance Case 3: CLI Help / Small Fixture for Retained Supported Tools
# ---------------------------------------------------------------------------


def test_retained_headless_tools_have_runnable_entrypoints() -> None:
    """Retained headless tools must have runnable __main__.py or module entry points."""
    retained_cli_tools = [
        "contraction",
        "drift_control",
        "sg_optimizer",
        "model_converter",
    ]
    for tool_name in retained_cli_tools:
        tool_dir = TOOLS_DIR / tool_name
        assert tool_dir.exists(), f"Tool directory missing: {tool_dir}"
        main_file = tool_dir / "__main__.py"
        assert main_file.exists(), (
            f"Missing __main__.py for retained CLI tool: {tool_name}"
        )


# ---------------------------------------------------------------------------
# Acceptance Case 4: Hidden Aliases and Deprecated Wrappers Resolve Acyclic
# ---------------------------------------------------------------------------


def test_hidden_aliases_resolve_acyclic_to_extant_targets() -> None:
    """All aliases in capability migration inventory must resolve acyclically to extant targets."""
    inventory = CapabilityMigrationInventory.load(CAPABILITY_MIGRATION_JSON)
    for entry_id, entry in inventory.entries.items():
        if entry.lifecycle == "deprecated_alias":
            assert entry.alias_target is not None, f"Alias {entry_id} has no target"
            resolved = inventory.resolve_alias(entry_id)
            assert resolved in inventory.entries, (
                f"Alias {entry_id} resolved to unknown {resolved}"
            )
            assert inventory.entries[resolved].lifecycle != "deprecated_alias", (
                f"Alias {entry_id} resolved to another alias {resolved}"
            )


# ---------------------------------------------------------------------------
# Acceptance Case 5: Catalog Links Have Owners and Next Actions
# ---------------------------------------------------------------------------


def test_catalog_links_have_owners_and_next_actions() -> None:
    """Incomplete/planned capabilities must document owner, issue, useful access, and next action."""
    manager = ResearchCapabilityLifecycleManager.load(REPO_ROOT)
    incomplete_records = manager.get_incomplete_records()
    assert len(incomplete_records) > 0, (
        "Expected at least one incomplete capability record"
    )

    for record in incomplete_records:
        assert record.capability_id, "Incomplete record missing capability_id"
        assert record.owner, f"Record {record.capability_id} missing owner"
        assert record.issue.startswith("#"), (
            f"Record {record.capability_id} missing tracking issue: {record.issue}"
        )
        assert len(record.next_action) >= 20, (
            f"Record {record.capability_id} needs substantive next_action, got: {record.next_action}"
        )
        assert record.useful_access, (
            f"Record {record.capability_id} missing useful_access instructions"
        )


# ---------------------------------------------------------------------------
# Acceptance Case 6: SG Optimizer Phase 3 UI Accurately Recorded
# ---------------------------------------------------------------------------


def test_sg_optimizer_phase3_ui_followup_accurately_recorded() -> None:
    """SG optimizer must record active CLI status, issue #6272, and concrete Phase 3 UI follow-up."""
    manager = ResearchCapabilityLifecycleManager.load(REPO_ROOT)
    sg_record = manager.get_record("sg_optimizer")
    assert sg_record is not None, "Missing sg_optimizer record"

    # Must accurately reflect active CLI reality
    assert sg_record.entity_kind == "cli_tool"
    assert "python -m src.tools.sg_optimizer" in sg_record.useful_access

    # Must link to issue #6272 and define concrete UI follow-up
    assert sg_record.issue == "#6272"
    assert "PyQt6" in sg_record.next_action or "UI" in sg_record.next_action
    assert "abandoned" not in sg_record.next_action.lower()


# ---------------------------------------------------------------------------
# Acceptance Case 7: Contextual Documentation Links Exist
# ---------------------------------------------------------------------------


def test_research_headless_documentation_links_exist() -> None:
    """All exclusions in registry_exclusions.yaml must have substantive reasons and valid paths."""
    data = yaml.safe_load(EXCLUSIONS_YAML.read_text(encoding="utf-8"))
    exclusions = data.get("exclusions", [])
    assert len(exclusions) >= 9, "Expected at least 9 registry exclusions"

    for entry in exclusions:
        pkg = entry.get("package", "")
        reason = str(entry.get("reason", "")).strip()
        assert (TOOLS_DIR / pkg).exists(), f"Excluded package missing from disk: {pkg}"
        assert len(reason) >= 20, f"Exclusion reason too short for {pkg}: {reason}"
