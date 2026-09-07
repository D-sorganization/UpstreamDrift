"""CI gate for the feature-parity registry (issue #7445 / epic #7462).

These tests ARE the parity CI gate. They fail when:
    1. A ``gap`` entry has no open-issue number
    2. A referenced pyqt/api/web file path does not exist
    3. A launcher-manifest tile id has no corresponding registry entry

Plus loader contract (DbC) unit tests mirroring the launcher-manifest
loader test conventions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.config.feature_parity_loader import (
    REGISTRY_PATH,
    FeatureParityEntry,
    FeatureParityRegistry,
)
from src.config.launcher_manifest_loader import LauncherManifest, MANIFEST_PATH

pytestmark = pytest.mark.parity

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def registry() -> FeatureParityRegistry:
    """Load the production feature-parity registry."""
    return FeatureParityRegistry.load()


@pytest.fixture(scope="module")
def manifest_tiles() -> list:
    """Static launcher-manifest tiles (provider tiles are dynamic)."""
    manifest = LauncherManifest.load(MANIFEST_PATH, include_provider_tiles=False)
    return list(manifest.tiles)


# =============================================================================
# 1. Registry loading and schema
# =============================================================================


class TestRegistryLoads:
    def test_registry_file_exists(self) -> None:
        assert REGISTRY_PATH.exists(), f"Missing registry: {REGISTRY_PATH}"

    def test_registry_loads_and_is_nonempty(
        self, registry: FeatureParityRegistry
    ) -> None:
        assert len(registry.entries) > 0

    def test_registry_is_valid_json_object(self) -> None:
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        assert isinstance(raw.get("features"), dict)

    def test_every_entry_has_valid_status(
        self, registry: FeatureParityRegistry
    ) -> None:
        for entry in registry.entries:
            assert entry.status in {"parity", "gap", "exempt", "api_only"}, (
                entry.feature_id
            )


# =============================================================================
# 2. CI gate: gap entries require an issue number
# =============================================================================


class TestGapEntriesHaveIssues:
    def test_every_gap_entry_has_positive_issue_number(
        self, registry: FeatureParityRegistry
    ) -> None:
        offenders = [
            e.feature_id
            for e in registry.gaps
            if not isinstance(e.issue, int) or e.issue <= 0
        ]
        assert not offenders, (
            f"Gap entries without a tracking issue: {offenders}. "
            "Every 'gap' status requires an open GitHub issue number."
        )

    def test_every_exempt_entry_has_reason(
        self, registry: FeatureParityRegistry
    ) -> None:
        offenders = [e.feature_id for e in registry.exemptions if not e.reason]
        assert not offenders, f"Exempt entries without a reason: {offenders}"

    def test_pending_exemptions_reference_decision_issue(
        self, registry: FeatureParityRegistry
    ) -> None:
        """Exemptions awaiting the G12 decision must say so in their reason."""
        offenders = [
            e.feature_id
            for e in registry.exemptions
            if e.pending_decision and "#7460" not in (e.reason or "")
        ]
        assert not offenders, (
            f"pending_decision exemptions must cite #7460 in reason: {offenders}"
        )


# =============================================================================
# 3. CI gate: referenced file paths must exist
# =============================================================================


class TestReferencedPathsExist:
    def test_all_referenced_paths_exist(self, registry: FeatureParityRegistry) -> None:
        missing = [
            f"{entry.feature_id}: {path}"
            for entry in registry.entries
            for path in entry.referenced_paths
            if not (REPO_ROOT / path).exists()
        ]
        assert not missing, (
            "Registry references non-existent files (update feature_parity.json "
            f"when files move): {missing}"
        )

    def test_referenced_paths_are_repo_relative(
        self, registry: FeatureParityRegistry
    ) -> None:
        offenders = [
            f"{entry.feature_id}: {path}"
            for entry in registry.entries
            for path in entry.referenced_paths
            if Path(path).is_absolute() or "\\" in path
        ]
        assert not offenders, f"Paths must be repo-relative with '/': {offenders}"


# =============================================================================
# 4. CI gate: every launcher-manifest tile id has a registry entry
# =============================================================================


class TestLauncherTileCoverage:
    def test_every_manifest_tile_is_covered(
        self, registry: FeatureParityRegistry, manifest_tiles: list
    ) -> None:
        covered = registry.covered_tile_ids
        uncovered = sorted(t.id for t in manifest_tiles if t.id not in covered)
        assert not uncovered, (
            f"Launcher tiles with no feature_parity.json entry: {uncovered}. "
            "Add or extend a registry entry listing these tile ids in 'tiles'."
        )

    def test_covered_tiles_reference_real_manifest_ids(
        self, registry: FeatureParityRegistry, manifest_tiles: list
    ) -> None:
        manifest_ids = {t.id for t in manifest_tiles}
        unknown = sorted(registry.covered_tile_ids - manifest_ids)
        assert not unknown, f"Registry references unknown launcher tile ids: {unknown}"


# =============================================================================
# 5. Loader contract (DbC) unit tests
# =============================================================================


class TestLoaderContracts:
    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid status"):
            FeatureParityEntry.from_dict("x.y", {"title": "X", "status": "done"})

    def test_gap_without_issue_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires a positive integer"):
            FeatureParityEntry.from_dict("x.y", {"title": "X", "status": "gap"})

    def test_gap_with_bool_issue_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires a positive integer"):
            FeatureParityEntry.from_dict(
                "x.y", {"title": "X", "status": "gap", "issue": True}
            )

    def test_exempt_without_reason_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires a non-empty 'reason'"):
            FeatureParityEntry.from_dict("x.y", {"title": "X", "status": "exempt"})

    def test_missing_title_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty title"):
            FeatureParityEntry.from_dict("x.y", {"status": "parity"})

    def test_empty_feature_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            FeatureParityEntry.from_dict("", {"title": "X", "status": "parity"})

    def test_rejects_non_dict_data_branch(self) -> None:
        with pytest.raises(TypeError, match="must be an object"):
            FeatureParityEntry.from_dict("x.y", ["not", "a", "dict"])  # type: ignore[arg-type]

    def test_rejects_non_gap_negative_issue_branch(self) -> None:
        with pytest.raises(ValueError, match="invalid issue number"):
            FeatureParityEntry.from_dict(
                "x.y", {"title": "X", "status": "parity", "issue": -1}
            )

    def test_rejects_non_gap_zero_issue_branch(self) -> None:
        with pytest.raises(ValueError, match="invalid issue number"):
            FeatureParityEntry.from_dict(
                "x.y", {"title": "X", "status": "parity", "issue": 0}
            )

    def test_rejects_empty_path_field_branch(self) -> None:
        with pytest.raises(ValueError, match="must be a non-empty string or null"):
            FeatureParityEntry.from_dict(
                "x.y", {"title": "X", "status": "parity", "pyqt": ""}
            )

    def test_rejects_non_list_tiles_branch(self) -> None:
        with pytest.raises(ValueError, match="must be a list of non-empty strings"):
            FeatureParityEntry.from_dict(
                "x.y", {"title": "X", "status": "parity", "tiles": "t1"}
            )

    def test_rejects_non_string_tile_branch(self) -> None:
        with pytest.raises(ValueError, match="must be a list of non-empty strings"):
            FeatureParityEntry.from_dict(
                "x.y", {"title": "X", "status": "parity", "tiles": [123]}
            )

    def test_rejects_empty_string_tile_branch(self) -> None:
        with pytest.raises(ValueError, match="must be a list of non-empty strings"):
            FeatureParityEntry.from_dict(
                "x.y", {"title": "X", "status": "parity", "tiles": [""]}
            )

    @pytest.mark.unit
    def test_pending_decision_on_non_exempt_rejected(self) -> None:
        with pytest.raises(ValueError, match="pending_decision is only valid"):
            FeatureParityEntry.from_dict(
                "x.y",
                {"title": "X", "status": "parity", "pending_decision": True},
            )

    @pytest.mark.unit
    def test_pending_decision_on_gap_rejected(self) -> None:
        with pytest.raises(ValueError, match="pending_decision is only valid"):
            FeatureParityEntry.from_dict(
                "x.y",
                {
                    "title": "X",
                    "status": "gap",
                    "issue": 7449,
                    "pending_decision": True,
                },
            )

    @pytest.mark.unit
    def test_pending_decision_on_exempt_allowed(self) -> None:
        entry = FeatureParityEntry.from_dict(
            "x.y",
            {
                "title": "X",
                "status": "exempt",
                "reason": "Desktop-only; pending #7460",
                "pending_decision": True,
            },
        )
        assert entry.pending_decision is True

    @pytest.mark.unit
    def test_falsy_pending_decision_on_non_exempt_allowed(self) -> None:
        # An explicit falsy pending_decision must not trip the exempt-scope check.
        entry = FeatureParityEntry.from_dict(
            "x.y",
            {"title": "X", "status": "parity", "pending_decision": False},
        )
        assert entry.pending_decision is False

    @pytest.mark.unit
    def test_non_string_notes_rejected(self) -> None:
        with pytest.raises(ValueError, match="'notes' must be a string"):
            FeatureParityEntry.from_dict(
                "x.y", {"title": "X", "status": "parity", "notes": 123}
            )

    @pytest.mark.unit
    def test_string_notes_accepted(self) -> None:
        entry = FeatureParityEntry.from_dict(
            "x.y",
            {"title": "X", "status": "parity", "notes": "clarifying note"},
        )
        assert entry.notes == "clarifying note"

    @pytest.mark.unit
    def test_valid_gap_entry_loads(self) -> None:
        entry = FeatureParityEntry.from_dict(
            "a.b",
            {
                "title": "A",
                "status": "gap",
                "issue": 7449,
                "pyqt": "src/x.py",
                "tiles": ["t1"],
            },
        )
        assert entry.issue == 7449
        assert entry.referenced_paths == ("src/x.py",)
        assert entry.tiles == ("t1",)

    def test_missing_registry_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            FeatureParityRegistry.load(tmp_path / "nope.json")

    def test_duplicate_tile_id_across_entries_rejected(self, tmp_path: Path) -> None:
        """Two entries claiming the same launcher tile must raise (not collapse)."""
        registry_file = tmp_path / "dup_tiles.json"
        registry_file.write_text(
            json.dumps(
                {
                    "version": "1.0.0",
                    "features": {
                        "a.b": {
                            "title": "A",
                            "status": "parity",
                            "tiles": ["shared_tile"],
                        },
                        "c.d": {
                            "title": "C",
                            "status": "parity",
                            "tiles": ["shared_tile"],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate launcher tile ids") as exc:
            FeatureParityRegistry.load(registry_file)
        assert "shared_tile" in str(exc.value)

    def test_unique_tile_ids_load_successfully(self, tmp_path: Path) -> None:
        """Distinct tile ids across entries load without error."""
        registry_file = tmp_path / "unique_tiles.json"
        registry_file.write_text(
            json.dumps(
                {
                    "version": "1.0.0",
                    "features": {
                        "a.b": {
                            "title": "A",
                            "status": "parity",
                            "tiles": ["tile_a"],
                        },
                        "c.d": {
                            "title": "C",
                            "status": "parity",
                            "tiles": ["tile_c"],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        loaded = FeatureParityRegistry.load(registry_file)
        assert loaded.covered_tile_ids == frozenset({"tile_a", "tile_c"})

    def test_by_status_rejects_unknown(self, registry: FeatureParityRegistry) -> None:
        with pytest.raises(ValueError, match="Unknown parity status"):
            registry.by_status("wishful")

    def test_get_returns_entry_or_none(self, registry: FeatureParityRegistry) -> None:
        assert registry.get("chat.transport") is not None
        assert registry.get("no.such.feature") is None
        with pytest.raises(ValueError, match="must be provided"):
            registry.get("")
