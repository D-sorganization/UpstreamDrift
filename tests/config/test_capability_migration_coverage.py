"""Capability baseline and migration coverage tests (Issue #10510 / ORG-01).

Validates that every capability (launcher tile, provider model, parity feature,
and excluded tool package) is accounted for with explicit migration metadata,
that legacy aliases resolve acyclically, that saved layouts remain valid,
and that preserved test fixture byte hashes remain unchanged.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from src.config.capability_migration import (
    ALLOWED_ENTITY_KINDS,
    ALLOWED_LIFECYCLES,
    ALLOWED_PRIMARY_WORKSPACES,
    ALLOWED_PROVIDER_AUTHORITIES,
    CapabilityMigrationInventory,
    FixtureEntry,
    MigrationEntry,
    load_migration_inventory,
)
from src.config.feature_parity_loader import FeatureParityRegistry
from src.config.launcher_manifest_loader import LauncherManifest
from src.shared.python.config.model_registry import ModelRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_YAML = REPO_ROOT / "src" / "config" / "models.yaml"
MANIFEST_JSON = REPO_ROOT / "src" / "config" / "launcher_manifest.json"
FEATURE_PARITY_JSON = REPO_ROOT / "src" / "config" / "feature_parity.json"
MIGRATION_JSON = REPO_ROOT / "src" / "config" / "capability_migration.json"

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def inventory() -> CapabilityMigrationInventory:
    """Load the machine-checkable capability migration baseline inventory."""
    return load_migration_inventory(MIGRATION_JSON)


class TestCapabilityMigrationCoverageRedCases:
    """RED cases verifying that unassigned, cyclic, or malformed entries fail validation."""

    def test_red_unclassified_local_tile_fails_coverage(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """An unclassified local tile from models.yaml must fail coverage until assigned."""
        mock_inventory = copy.deepcopy(inventory)
        # Remove an active local tile from the inventory to simulate an unclassified tile
        if "model_explorer" in mock_inventory.entries:
            del mock_inventory.entries["model_explorer"]

        with pytest.raises(ValueError, match="Unclassified local tile.*model_explorer"):
            mock_inventory.validate_local_coverage(models_path=MODELS_YAML)

    def test_red_unclassified_provider_only_model_fails_coverage(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """A discovered provider-only model must fail coverage until assigned."""
        mock_inventory = copy.deepcopy(inventory)
        # Register a synthetic provider model not present in the baseline inventory
        synthetic_provider_id = "mujoco_models-synthetic_unassigned_test_model"
        with pytest.raises(
            ValueError,
            match="Unclassified provider model.*synthetic_unassigned_test_model",
        ):
            mock_inventory.validate_provider_model_id(synthetic_provider_id)

    def test_red_unclassified_manifest_only_service_fails_coverage(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """A manifest-only web service must fail coverage until assigned."""
        mock_inventory = copy.deepcopy(inventory)
        synthetic_service_tile = {
            "id": "synthetic_unassigned_manifest_service",
            "name": "Synthetic Service",
            "category": "service",
            "web": {"mode": "route", "route": "/services/synthetic"},
        }
        with pytest.raises(
            ValueError,
            match="Unclassified manifest entry.*synthetic_unassigned_manifest_service",
        ):
            mock_inventory.validate_manifest_tile(synthetic_service_tile)

    def test_red_unclassified_excluded_cli_fails_coverage(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """An excluded CLI package under src/tools/ must fail coverage until assigned."""
        mock_inventory = copy.deepcopy(inventory)
        synthetic_cli = {
            "package": "synthetic_unassigned_cli_tool",
            "reason": "CLI tool for testing unassigned failure cases",
        }
        with pytest.raises(
            ValueError,
            match="Unclassified excluded CLI package.*synthetic_unassigned_cli_tool",
        ):
            mock_inventory.validate_excluded_cli_package(synthetic_cli)

    def test_red_alias_cycle_fails_with_actionable_diagnostics(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """An alias cycle must fail resolution with actionable diagnostic listing the cycle."""
        mock_inventory = copy.deepcopy(inventory)
        mock_inventory.entries["cycle_a"] = MigrationEntry(
            id="cycle_a",
            name="Cycle A",
            entity_kind="tile",
            primary_workspace="simulation",
            secondary_links=[],
            provider_authority="core",
            lifecycle="deprecated_alias",
            alias_target="cycle_b",
            evidence="src/config/models.yaml",
            acceptance_owner="tests/config/test_capability_migration_coverage.py",
        )
        mock_inventory.entries["cycle_b"] = MigrationEntry(
            id="cycle_b",
            name="Cycle B",
            entity_kind="tile",
            primary_workspace="simulation",
            secondary_links=[],
            provider_authority="core",
            lifecycle="deprecated_alias",
            alias_target="cycle_a",
            evidence="src/config/models.yaml",
            acceptance_owner="tests/config/test_capability_migration_coverage.py",
        )

        with pytest.raises(ValueError, match="Alias cycle detected:.*cycle_a.*cycle_b"):
            mock_inventory.resolve_alias("cycle_a")

    def test_red_removed_saved_layout_id_fails_with_actionable_diagnostics(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """A saved layout containing an unregistered / removed ID must fail with actionable diagnostic."""
        stale_saved_layout = [
            "putting_green",
            "nonexistent_removed_tile_12345",
        ]
        with pytest.raises(
            ValueError,
            match="Unregistered or removed layout model ID 'nonexistent_removed_tile_12345'",
        ):
            inventory.validate_saved_layout(stale_saved_layout)

    def test_red_duplicate_identity_fails_with_actionable_diagnostics(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """Duplicate identity declarations must fail validation with actionable message."""
        mock_inventory = copy.deepcopy(inventory)
        duplicate_entry = MigrationEntry(
            id="model_explorer",
            name="Duplicate Model Explorer",
            entity_kind="tile",
            primary_workspace="simulation",
            secondary_links=[],
            provider_authority="core",
            lifecycle="active_feature",
            alias_target=None,
            evidence="src/config/models.yaml",
            acceptance_owner="tests/config/test_capability_migration_coverage.py",
        )
        with pytest.raises(
            ValueError, match="Duplicate capability ID: 'model_explorer'"
        ):
            mock_inventory.add_entry(duplicate_entry)

    def test_red_missing_provenance_fails_with_actionable_diagnostics(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """Entries with nonexistent or empty evidence must fail with actionable diagnostic."""
        invalid_entry = MigrationEntry(
            id="missing_provenance_entry",
            name="Missing Provenance",
            entity_kind="tile",
            primary_workspace="simulation",
            secondary_links=[],
            provider_authority="core",
            lifecycle="active_feature",
            alias_target=None,
            evidence="nonexistent/path/to/missing_file.py",
            acceptance_owner="tests/config/test_capability_migration_coverage.py",
        )
        with pytest.raises(
            ValueError,
            match="Missing provenance evidence: 'nonexistent/path/to/missing_file.py'",
        ):
            inventory.validate_entry_provenance(invalid_entry, root=REPO_ROOT)


class TestCapabilityMigrationCoverageGreenCases:
    """GREEN cases verifying baseline completeness, resolution, and hash preservation."""

    def test_green_local_only_inventory_classifies_all_entries(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """Local-only catalog (all desktop models + web catalog only + features + exclusions) must be classified."""
        desktop_reg = ModelRegistry(
            config_path=MODELS_YAML, discovery_mode="local-only"
        )
        local_model_ids = {m.id for m in desktop_reg.get_all_models()}

        # Every local desktop model must exist in baseline inventory
        unclassified_local = local_model_ids - set(inventory.entries.keys())
        assert not unclassified_local, (
            f"Unclassified local desktop models: {unclassified_local}"
        )

        # Every entry must have valid fields
        for model_id in local_model_ids:
            entry = inventory.entries[model_id]
            assert entry.entity_kind in ALLOWED_ENTITY_KINDS
            assert entry.primary_workspace in ALLOWED_PRIMARY_WORKSPACES
            assert entry.lifecycle in ALLOWED_LIFECYCLES
            assert entry.provider_authority in ALLOWED_PROVIDER_AUTHORITIES
            assert entry.evidence, f"Entry {model_id} lacks evidence path"

    def test_green_provider_expanded_inventory_classifies_all_entries(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """All 104 observed manifest tiles (including discovered provider models) must be classified."""
        manifest = LauncherManifest.load()
        observed_ids = set(manifest.tile_ids)
        assert len(observed_ids) >= 104, (
            f"Expected at least 104 tiles, found {len(observed_ids)}"
        )

        unclassified = observed_ids - set(inventory.entries.keys())
        assert not unclassified, (
            f"Unclassified observed tiles in provider-expanded catalog: {unclassified}"
        )

    def test_green_old_starting_pose_matcher_and_putting_green_gui_ids_resolve(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """Legacy aliases starting_pose_matcher and putting_green_gui must resolve to retained targets."""
        assert (
            inventory.resolve_alias("starting_pose_matcher") == "motion_target_preview"
        )
        assert inventory.resolve_alias("putting_green_gui") == "putting_green"

        # Verify their lifecycle is deprecated_alias and alias_target is assigned
        spm = inventory.entries["starting_pose_matcher"]
        assert spm.lifecycle == "deprecated_alias"
        assert spm.alias_target == "motion_target_preview"

        pgg = inventory.entries["putting_green_gui"]
        assert pgg.lifecycle == "deprecated_alias"
        assert pgg.alias_target == "putting_green"

    def test_green_fixture_byte_hashes_unchanged(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """Representative test fixtures across capture, pose, model, replay, flight, export must match golden hashes."""
        assert len(inventory.fixtures) >= 12, (
            "Baseline must track representative preservation fixtures"
        )

        for fixture in inventory.fixtures.values():
            fixture_path = REPO_ROOT / fixture.path
            assert fixture_path.exists(), (
                f"Preservation fixture file missing: {fixture.path}"
            )
            computed_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
            assert computed_sha256 == fixture.sha256, (
                f"Fixture hash mismatch for {fixture.path}: expected {fixture.sha256}, got {computed_sha256}"
            )
            assert fixture_path.stat().st_size == fixture.size_bytes, (
                f"Fixture size mismatch for {fixture.path}: expected {fixture.size_bytes}, got {fixture_path.stat().st_size}"
            )

    def test_green_provider_absence_changes_only_availability_not_identity(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """Provider absence changes runtime availability, but preserves identity and primary workspace."""
        provider_model_id = "mujoco_unified"
        entry = inventory.entries[provider_model_id]
        original_workspace = entry.primary_workspace
        original_kind = entry.entity_kind

        # Test availability query when provider is offline
        availability = inventory.evaluate_availability(
            provider_model_id, provider_available=False
        )
        assert availability["available"] is False
        assert availability["id"] == provider_model_id
        assert availability["primary_workspace"] == original_workspace
        assert availability["entity_kind"] == original_kind

        # When provider is online
        availability_online = inventory.evaluate_availability(
            provider_model_id, provider_available=True
        )
        assert availability_online["available"] is True

    def test_green_every_entry_has_exactly_one_primary_workspace(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """Every entry in the inventory must have exactly one primary workspace and valid secondary links."""
        for entry_id, entry in inventory.entries.items():
            assert entry.primary_workspace in ALLOWED_PRIMARY_WORKSPACES, (
                f"Entry {entry_id} has invalid primary workspace: {entry.primary_workspace}"
            )
            assert isinstance(entry.secondary_links, list), (
                f"Entry {entry_id} secondary_links must be a list"
            )

    def test_green_saved_layouts_resolve_cleanly(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """Saved layouts with legacy aliases must resolve them to retained canonical targets."""
        layout = ["starting_pose_matcher", "putting_green_gui", "model_explorer"]
        resolved = inventory.validate_saved_layout(layout)
        assert resolved == ["motion_target_preview", "putting_green", "model_explorer"]

    def test_green_feature_parity_contracts_covered(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """All 45 feature contracts from feature_parity.json must be accounted for."""
        parity_reg = FeatureParityRegistry.load(FEATURE_PARITY_JSON)
        feature_ids = {entry.feature_id for entry in parity_reg.entries}
        unclassified_features = feature_ids - set(inventory.entries.keys())
        assert not unclassified_features, (
            f"Unclassified parity features: {unclassified_features}"
        )

    def test_green_excluded_tool_packages_covered(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """All excluded tool packages from registry_exclusions.yaml must be accounted for."""
        import yaml

        exclusions = yaml.safe_load(
            (REPO_ROOT / "src/config/registry_exclusions.yaml").read_text()
        )["exclusions"]
        for item in exclusions:
            pkg = item["package"]
            # Look up by either package name or pkg_cli
            found = (
                pkg in inventory.entries
                or f"cli_{pkg}" in inventory.entries
                or f"tools_{pkg}" in inventory.entries
            )
            assert found, (
                f"Excluded tool package '{pkg}' not accounted for in migration inventory"
            )

    def test_green_generated_human_inventory_matches_machine_disposition(
        self, inventory: CapabilityMigrationInventory
    ) -> None:
        """The committed markdown document docs/development/ORG01_CAPABILITY_BASELINE.md must document all capabilities."""
        from scripts.generate_capability_baseline import BASELINE_PATH

        assert BASELINE_PATH.is_file(), f"Missing {BASELINE_PATH}"
        content = BASELINE_PATH.read_text(encoding="utf-8")

        assert (
            "# ORG-01: Baseline Capability Inventory & Entity Identity Preservation"
            in content
        )
        assert "## Complete Capability Baseline Inventory" in content
        assert "## Representative Golden Preservation Fixtures" in content
        assert "## Legacy Alias Resolution Graph" in content

        # Every capability entry must appear in the inventory table
        for entry_id in inventory.entries:
            assert f"`{entry_id}`" in content, (
                f"Entry '{entry_id}' missing from {BASELINE_PATH.name}"
            )

        # Every preservation fixture must appear in the fixtures table
        for fixture in inventory.fixtures.values():
            assert f"`{fixture.path}`" in content, (
                f"Fixture '{fixture.path}' missing from {BASELINE_PATH.name}"
            )

        # Every alias must appear in the alias table
        for entry in inventory.entries.values():
            if entry.lifecycle == "deprecated_alias":
                assert (
                    f"`{entry.id}`" in content and f"`{entry.alias_target}`" in content
                )
