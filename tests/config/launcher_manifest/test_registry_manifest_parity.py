"""Field-level parity between models.yaml and launcher_manifest.json (#8089).

The native PyQt6 launcher reads ``src/config/models.yaml``; the API and the
Tauri/React dashboard read ``src/config/launcher_manifest.json``. Both used to
hand-author the same user-visible semantics and every shared tile had drifted.

``src/config/launcher_parity.py`` now declares which file owns which field.
These tests enforce that contract so a new drift cannot reach main:

* every registry-owned field agrees for every shared tile ID;
* the committed manifest carries exactly what the generator produces
  (freshness gate, mirroring ``scripts/sync_launcher_manifest --check``);
* display orders are unique (#8092);
* no two visible tiles share a display name and resolved artifact (#8090);
* no visible provider tile advertises an empty capability set (#8091).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from scripts.sync_launcher_manifest import build_manifest, is_in_sync, registry_value
from src.config.launcher_manifest_loader import (
    MANIFEST_PATH,
    REGISTRY_PATH,
    LauncherManifest,
    _derive_provider_capabilities,
)
from src.config.launcher_parity import (
    FIELD_EXCEPTIONS,
    MANIFEST_OWNED_FIELDS,
    REGISTRY_OWNED_FIELDS,
    is_exempt,
)
from src.shared.python.config.model_registry import ModelConfig, ModelRegistry

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def registry_models() -> dict[str, ModelConfig]:
    """Every model the native launcher can see, keyed by ID."""
    registry = ModelRegistry(config_path=REGISTRY_PATH)
    return {model.id: model for model in registry.get_all_models()}


@pytest.fixture(scope="module")
def raw_tiles() -> dict[str, dict]:
    """Tiles exactly as committed in launcher_manifest.json, keyed by ID."""
    data = json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))
    return {tile["id"]: tile for tile in data["tiles"]}


@pytest.fixture(scope="module")
def shared_ids(
    registry_models: dict[str, ModelConfig], raw_tiles: dict[str, dict]
) -> list[str]:
    """IDs declared in both manifests."""
    return sorted(set(registry_models) & set(raw_tiles))


class TestOwnershipContract:
    """The contract itself must stay coherent and documented."""

    def test_no_field_is_owned_twice(self) -> None:
        """A field is owned by exactly one manifest."""
        overlap = set(REGISTRY_OWNED_FIELDS) & set(MANIFEST_OWNED_FIELDS)
        assert not overlap, f"Fields claimed by both manifests: {sorted(overlap)}"

    def test_every_manifest_owned_field_has_a_reason(self) -> None:
        """Each manifest-owned field documents why it has no registry twin."""
        undocumented = [
            field
            for field, reason in MANIFEST_OWNED_FIELDS.items()
            if not reason.strip()
        ]
        assert not undocumented, f"Undocumented manifest-owned fields: {undocumented}"

    def test_every_exception_has_a_reason(self) -> None:
        """Each per-tile exception documents why the two surfaces differ."""
        undocumented = [
            key for key, reason in FIELD_EXCEPTIONS.items() if not reason.strip()
        ]
        assert not undocumented, f"Undocumented parity exceptions: {undocumented}"

    def test_exceptions_only_cover_registry_owned_fields(self) -> None:
        """An exception is only meaningful for a field that must otherwise agree."""
        stray = [key for key in FIELD_EXCEPTIONS if key[1] not in REGISTRY_OWNED_FIELDS]
        assert not stray, (
            f"Exceptions for fields that are not registry-owned: {stray}. "
            "Manifest-owned fields never need an exception."
        )

    def test_exceptions_name_real_tiles(
        self, raw_tiles: dict[str, dict], registry_models: dict[str, ModelConfig]
    ) -> None:
        """Exceptions cannot rot into references to deleted tiles."""
        known = set(raw_tiles) | set(registry_models)
        stray = sorted(
            {tile_id for tile_id, _ in FIELD_EXCEPTIONS if tile_id not in known}
        )
        assert not stray, f"Parity exceptions for unknown tiles: {stray}"

    def test_is_exempt_rejects_empty_arguments(self) -> None:
        """DbC: the helper validates its preconditions."""
        with pytest.raises(ValueError):
            is_exempt("", "path")
        with pytest.raises(ValueError):
            is_exempt("putting_green", "")


class TestRegistryOwnedFieldParity:
    """Every registry-owned field must agree for every shared tile."""

    def test_shared_ids_are_not_empty(self, shared_ids: list[str]) -> None:
        """Guard against a vacuous parity suite if either file is renamed."""
        assert len(shared_ids) >= 20, (
            f"Expected the two manifests to share tiles; found {shared_ids}"
        )

    def test_registry_owned_fields_agree(
        self,
        shared_ids: list[str],
        registry_models: dict[str, ModelConfig],
        raw_tiles: dict[str, dict],
    ) -> None:
        """No shared tile may disagree on a registry-owned field."""
        drift: list[str] = []
        for tile_id in shared_ids:
            model = registry_models[tile_id]
            tile = raw_tiles[tile_id]
            for field in REGISTRY_OWNED_FIELDS:
                if is_exempt(tile_id, field):
                    continue
                expected = registry_value(model, field)
                actual = tile.get(field)
                if not expected and not actual:
                    continue
                if expected != actual:
                    drift.append(
                        f"{tile_id}.{field}: models.yaml={expected!r} != "
                        f"launcher_manifest.json={actual!r}"
                    )
        assert not drift, (
            "Registry-owned fields drifted. Run "
            "`python3 -m scripts.sync_launcher_manifest` to regenerate the "
            "manifest from models.yaml:\n  " + "\n  ".join(drift)
        )

    def test_committed_manifest_is_fresh(self) -> None:
        """The committed JSON carries the generator's values (CI freshness gate).

        Compared as parsed JSON rather than bytes because the ``prettier``
        pre-commit hook re-flows short arrays onto a single line.
        """
        assert is_in_sync(), (
            "src/config/launcher_manifest.json is stale. Run "
            "`python3 -m scripts.sync_launcher_manifest`."
        )

    def test_generator_is_idempotent(self) -> None:
        """Regenerating an already-synced manifest is a no-op."""
        assert build_manifest() == build_manifest()


class TestDisplayOrderUniqueness:
    """Issue #8092 — duplicate display orders break deterministic ordering."""

    def test_orders_are_unique(self, raw_tiles: dict[str, dict]) -> None:
        """Every declared display order value appears exactly once."""
        counts = Counter(tile.get("order", 99) for tile in raw_tiles.values())
        dupes = {
            order: sorted(
                t["id"] for t in raw_tiles.values() if t.get("order", 99) == order
            )
            for order, count in counts.items()
            if count > 1
        }
        assert not dupes, f"Duplicate display orders: {dupes}"

    def test_rendered_order_is_stable_across_loads(self) -> None:
        """Unique orders make the rendered sequence independent of tie-breaking."""
        first = [tile.id for tile in LauncherManifest.load().tiles]
        second = [tile.id for tile in LauncherManifest.load().tiles]
        assert first == second


class TestNoDuplicateVisibleTiles:
    """Issue #8090 — two indistinguishable Putting Green tiles were visible."""

    def test_no_visible_duplicate_name_and_artifact(self) -> None:
        """A (display name, resolved artifact) pair identifies one visible tile."""
        manifest = LauncherManifest.load()
        seen: dict[tuple[str, str], list[str]] = defaultdict(list)
        for tile in manifest.visible_tiles:
            seen[(tile.name, (tile.path or "").replace("\\", "/"))].append(tile.id)
        dupes = {key: ids for key, ids in seen.items() if len(ids) > 1}
        assert not dupes, (
            "Visible tiles share a display name and resolved artifact — hide the "
            f"legacy alias with hidden/hidden_reason/hidden_owner: {dupes}"
        )

    def test_registry_has_no_visible_duplicate_name_and_artifact(
        self, registry_models: dict[str, ModelConfig]
    ) -> None:
        """The native launcher's own source must not contain duplicates either."""
        seen: dict[tuple[str, str], list[str]] = defaultdict(list)
        for model in registry_models.values():
            if getattr(model, "hidden", False):
                continue
            seen[(model.name, (model.path or "").replace("\\", "/"))].append(model.id)
        dupes = {key: ids for key, ids in seen.items() if len(ids) > 1}
        assert not dupes, f"Duplicate visible models in models.yaml: {dupes}"

    def test_documented_legacy_aliases_are_still_allowed(
        self, registry_models: dict[str, ModelConfig]
    ) -> None:
        """Hidden aliases remain resolvable and carry a reason and an owner."""
        alias = registry_models["putting_green_gui"]
        assert alias.hidden is True
        assert alias.hidden_reason and alias.hidden_reason.strip()
        assert alias.hidden_owner and alias.hidden_owner.strip()
        assert alias.path == registry_models["putting_green"].path


class TestProviderTileCapabilities:
    """Issue #8091 — provider exercise tiles advertised no capabilities."""

    def test_no_visible_tile_has_empty_capabilities(self) -> None:
        """Every visible tile can describe itself to filters and detail views."""
        manifest = LauncherManifest.load()
        empty = [tile.id for tile in manifest.visible_tiles if not tile.capabilities]
        assert not empty, f"Visible tiles with no capabilities: {empty}"

    def test_derived_capabilities_are_truthful(
        self, registry_models: dict[str, ModelConfig]
    ) -> None:
        """Derived tags only restate facts the registry entry already carries."""
        provider_ids = [
            model_id
            for model_id, model in registry_models.items()
            if "-" in model_id and model.engine_type and not model.capabilities
        ]
        assert provider_ids, "Expected provider-backed exercise models in the registry"
        for model_id in provider_ids:
            model = registry_models[model_id]
            derived = _derive_provider_capabilities(model)
            assert "model_asset" in derived
            assert str(model.engine_type).lower() in derived
            assert str(model.type).lower().split("-", 1)[0] in derived
            exercise = model_id.partition("-")[2]
            assert f"exercise_{exercise.lower()}" in derived
            assert len(derived) == len(set(derived)), "capability tags must be unique"

    def test_declared_capabilities_are_never_overridden(self) -> None:
        """A pack that declares capabilities keeps them verbatim."""
        model = ModelConfig(
            id="pack-squat",
            name="Squat",
            description="Squat model",
            type="urdf",
            path="exercises/squat",
            engine_type="pinocchio",
            capabilities=("muscle_analysis",),
            provider="pinocchio",
        )
        assert model.capabilities != _derive_provider_capabilities(model)

    def test_derive_rejects_missing_model(self) -> None:
        """DbC: the helper validates its precondition."""
        with pytest.raises(ValueError):
            _derive_provider_capabilities(None)  # type: ignore[arg-type]
