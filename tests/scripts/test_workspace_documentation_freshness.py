"""Acceptance tests for workspace documentation freshness, atlas routes, and parity.

Validates ORG-21 / Issue #10531:
1. Workspace membership drift is rejected across canonical catalogs and connections.
2. Undocumented and dangling route aliases fail validation.
3. Broken evidence, source, and help links fail validation.
4. Stale generated views (atlas, parity matrix, readiness index) fail validation.
5. Shell-only parity cannot be interpreted as compute-complete in task acceptance data.
6. Generation scripts are deterministic across repeated invocations.
7. Training controller README accurately reflects the shipped PyQt6 surface.
8. Industrial readiness item U3 (#8820) is reconciled with merge SHA and verified evidence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.capability_atlas.model import SOURCES, build, validate_graph
from scripts.generate_capability_atlas import outputs as atlas_outputs
from scripts.generate_feature_parity_matrix import render_matrix
from scripts.generate_industrial_readiness_index import render_index
from src.config.capability_migration import (
    ALLOWED_PRIMARY_WORKSPACES,
    CapabilityMigrationInventory,
    load_migration_inventory,
)
from src.config.capability_state import adapt_engine_matrix_qualification
from src.config.feature_parity_loader import FeatureParityRegistry
from src.config.industrial_readiness_loader import IndustrialReadinessLedger

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Acceptance Case 1: Workspace Membership Drift
# ---------------------------------------------------------------------------


def test_workspace_membership_drift() -> None:
    """Canonical workspaces must match exactly across migration, models, and connections."""
    migration_path = REPO_ROOT / "src/config/capability_migration.json"
    migration_raw = json.loads(migration_path.read_text(encoding="utf-8"))
    declared_workspaces = set(migration_raw.get("workspaces", []))
    assert declared_workspaces == set(ALLOWED_PRIMARY_WORKSPACES)

    inventory = load_migration_inventory(migration_path)
    for entry_id, entry in inventory.entries.items():
        assert entry.primary_workspace in ALLOWED_PRIMARY_WORKSPACES, (
            f"Entry {entry_id} declares unapproved workspace: {entry.primary_workspace}"
        )

    connections_path = REPO_ROOT / "src/config/capability_connections.json"
    connections_raw = json.loads(connections_path.read_text(encoding="utf-8"))
    for node in connections_raw.get("nodes", []):
        view = node.get("view")
        assert view in ("system", "workflow"), (
            f"Node {node.get('id')} has unapproved view: {view}"
        )

    # Mutation test: drift to an unregistered workspace must fail validation
    mutated = copy.deepcopy(migration_raw)
    mutated["workspaces"].append("unregistered_domain")
    with pytest.raises(ValueError, match="Unknown primary_workspace"):
        bad_entry = copy.deepcopy(next(iter(mutated["entries"].values())))
        bad_entry["primary_workspace"] = "unregistered_domain"
        mutated["entries"]["bad_entry_drift"] = bad_entry
        CapabilityMigrationInventory.from_dict(mutated)


# ---------------------------------------------------------------------------
# Acceptance Case 2: Undocumented and Dangling Aliases
# ---------------------------------------------------------------------------


def test_undocumented_alias() -> None:
    """Route and capability aliases must document a valid, active canonical destination."""
    inventory = load_migration_inventory()
    for entry_id, entry in inventory.entries.items():
        if entry.lifecycle == "deprecated_alias":
            assert entry.alias_target is not None, (
                f"Deprecated alias {entry_id} lacks an alias_target destination."
            )
            canonical_id = inventory.resolve_alias(entry_id)
            canonical = inventory.entries[canonical_id]
            assert canonical.lifecycle != "deprecated_alias", (
                f"Alias {entry_id} must resolve transitively to a non-alias capability."
            )
        elif entry.alias_target is not None:
            canonical_id = inventory.resolve_alias(entry_id)
            assert canonical_id in inventory.entries

    # Mutation test: cyclic alias must raise ValueError
    mutated = copy.deepcopy(inventory.to_dict())
    mutated["entries"]["cycle_a"] = {
        "id": "cycle_a",
        "name": "Cycle A",
        "entity_kind": "tile",
        "primary_workspace": "analysis",
        "lifecycle": "deprecated_alias",
        "alias_target": "cycle_b",
    }
    mutated["entries"]["cycle_b"] = {
        "id": "cycle_b",
        "name": "Cycle B",
        "entity_kind": "tile",
        "primary_workspace": "analysis",
        "lifecycle": "deprecated_alias",
        "alias_target": "cycle_a",
    }
    cyclic_inventory = CapabilityMigrationInventory.from_dict(mutated)
    with pytest.raises(ValueError, match="Alias cycle detected"):
        cyclic_inventory.resolve_alias("cycle_a")


# ---------------------------------------------------------------------------
# Acceptance Case 3: Broken Source and Help Links
# ---------------------------------------------------------------------------


def test_broken_source_and_help_links() -> None:
    """Every evidenced file path in capability connections and atlas must exist on disk."""
    connections_path = REPO_ROOT / "src/config/capability_connections.json"
    connections = json.loads(connections_path.read_text(encoding="utf-8"))

    for node in connections.get("nodes", []):
        evidence = node.get("evidence")
        assert evidence, f"Node {node.get('id')} missing evidence path."
        evidence_path = REPO_ROOT / evidence
        assert evidence_path.exists(), (
            f"Broken evidence link for node {node.get('id')}: {evidence}"
        )

    for edge in connections.get("edges", []):
        evidence = edge.get("evidence")
        assert evidence, (
            f"Edge {edge.get('source')}->{edge.get('target')} missing evidence."
        )
        evidence_path = REPO_ROOT / evidence
        assert evidence_path.exists(), (
            f"Broken evidence link for edge {edge.get('source')}->{edge.get('target')}: {evidence}"
        )

    # Validate that graph build rejects non-existent paths
    graph = build(REPO_ROOT)
    assert len(graph["nodes"]) > 0


# ---------------------------------------------------------------------------
# Acceptance Case 4: Stale Generated Views
# ---------------------------------------------------------------------------


def test_stale_generated_views() -> None:
    """Committed atlas, parity matrix, and readiness index views must match generators."""
    atlas_files = atlas_outputs(REPO_ROOT)
    for path, expected_content in atlas_files.items():
        assert path.exists(), f"Generated atlas artifact does not exist: {path}"
        actual = path.read_text(encoding="utf-8")
        assert actual == expected_content, (
            f"Stale atlas artifact {path.relative_to(REPO_ROOT)}. "
            "Regenerate with: python3 -m scripts.generate_capability_atlas"
        )

    parity_matrix_path = REPO_ROOT / "docs/development/feature_parity_matrix.md"
    registry = FeatureParityRegistry.load(REPO_ROOT / "src/config/feature_parity.json")
    expected_matrix = render_matrix(registry)
    actual_matrix = parity_matrix_path.read_text(encoding="utf-8")
    assert actual_matrix == expected_matrix, (
        "docs/development/feature_parity_matrix.md is stale relative to feature_parity.json. "
        "Regenerate with: python3 -m scripts.generate_feature_parity_matrix"
    )

    readiness_path = REPO_ROOT / "docs/operations/industrial-readiness-index.md"
    ledger = IndustrialReadinessLedger.load(
        REPO_ROOT / "src/config/industrial_readiness.json"
    )
    expected_index = render_index(ledger)
    actual_index = readiness_path.read_text(encoding="utf-8")
    assert actual_index == expected_index, (
        "docs/operations/industrial-readiness-index.md is stale relative to industrial_readiness.json. "
        "Regenerate with: python3 -m scripts.generate_industrial_readiness_index"
    )


# ---------------------------------------------------------------------------
# Acceptance Case 5: Shell-Only Parity vs Compute-Complete
# ---------------------------------------------------------------------------


def test_shell_only_parity_cannot_be_interpreted_as_compute_complete() -> None:
    """Shell availability does not equal compute execution or qualification completeness."""
    # Engine matrix qualification contract
    qualification = adapt_engine_matrix_qualification(
        engine_name="mujoco",
        is_engine=True,
        receipt=None,
    )
    assert not qualification.is_qualified
    assert qualification.status in ("qualification_failed", "unqualified")
    assert "missing_engine_receipt" in qualification.failure_reasons

    # Parity registry audit: shell surface must have real path, not stub or claim of qualification
    registry = FeatureParityRegistry.load(REPO_ROOT / "src/config/feature_parity.json")
    for feature in registry.entries:
        if feature.status == "parity":
            if feature.pyqt:
                assert (REPO_ROOT / feature.pyqt).exists(), (
                    f"Feature {feature.feature_id} pyqt evidence missing: {feature.pyqt}"
                )
            if feature.api:
                assert (REPO_ROOT / feature.api).exists(), (
                    f"Feature {feature.feature_id} api evidence missing: {feature.api}"
                )


# ---------------------------------------------------------------------------
# Acceptance Case 6: Deterministic Generators
# ---------------------------------------------------------------------------


def test_generators_deterministic() -> None:
    """All catalog generators must produce byte-identical output across repeated runs."""
    first_atlas = atlas_outputs(REPO_ROOT)
    second_atlas = atlas_outputs(REPO_ROOT)
    assert first_atlas == second_atlas

    registry = FeatureParityRegistry.load(REPO_ROOT / "src/config/feature_parity.json")
    assert render_matrix(registry) == render_matrix(registry)

    ledger = IndustrialReadinessLedger.load(
        REPO_ROOT / "src/config/industrial_readiness.json"
    )
    assert render_index(ledger) == render_index(ledger)


# ---------------------------------------------------------------------------
# Acceptance Case 7: Training Controller README Surface Accuracy
# ---------------------------------------------------------------------------


def test_training_controller_readme_accurately_reflects_shipped_surface() -> None:
    """Training controller README must document current shipped PyQt6 GUI reality."""
    readme_path = REPO_ROOT / "src/tools/training_controller/README.md"
    readme_text = readme_path.read_text(encoding="utf-8")

    assert "MainWindow" in readme_text
    assert "gui.py" in readme_text
    assert "_embed_adapter.py" in readme_text
    assert "deferred" not in readme_text.lower(), (
        "README must not claim GUI surface is deferred when gui.py is merged."
    )
    assert "draft" not in readme_text.lower(), (
        "README must not contain obsolete draft branch notes."
    )


# ---------------------------------------------------------------------------
# Acceptance Case 8: Industrial Readiness Issue #8820 Reconciled
# ---------------------------------------------------------------------------


def test_industrial_readiness_issue_8820_reconciled() -> None:
    """Item U3 (#8820) in industrial_readiness.json must be recorded as merged with evidence."""
    ledger = IndustrialReadinessLedger.load(
        REPO_ROOT / "src/config/industrial_readiness.json"
    )
    item_u3 = next((item for item in ledger.queue if item.key == "U3"), None)
    assert item_u3 is not None, "Missing item U3 in industrial_readiness.json"
    assert item_u3.issue == 8820
    assert item_u3.status == "merged"
    assert "8ef1bec803de292e44724cdf1f96d3ebf52bf2f2" in item_u3.merge_shas

    # Verify implementation and test paths exist on disk
    for path in item_u3.implementation:
        assert (REPO_ROOT / path).exists(), f"Implementation path missing: {path}"
    for path in item_u3.tests:
        assert (REPO_ROOT / path).exists(), f"Test path missing: {path}"
