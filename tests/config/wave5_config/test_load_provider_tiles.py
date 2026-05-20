"""Unit tests for LauncherManifest._load_provider_tiles."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.config.launcher_manifest_loader import LauncherManifest
from src.shared.python.config.model_pack_manifest import LauncherPresentationMetadata
from src.shared.python.config.model_registry import ModelConfig


def _make_model(**overrides) -> ModelConfig:
    base = {
        "id": "prov1",
        "name": "Provider 1",
        "description": "desc",
        "type": "engine",
        "path": "src/launchers/p1.py",
    }
    base.update(overrides)
    return ModelConfig(**base)


def test_returns_empty_when_registry_missing(tmp_path):
    missing = tmp_path / "no-registry.yaml"
    out = LauncherManifest._load_provider_tiles(
        registry_path=missing, existing_ids=set()
    )
    assert out == []


def test_skips_models_with_existing_ids(tmp_path):
    registry_path = tmp_path / "reg.yaml"
    registry_path.write_text("models: []\n", encoding="utf-8")
    mock_registry = MagicMock()
    meta = LauncherPresentationMetadata(
        category="tool", logo="x.svg", status="provider_ready"
    )
    mock_registry.get_all_models.return_value = [
        _make_model(id="dup", launcher=meta, provider="acme"),
    ]
    with patch(
        "src.config.launcher_manifest_loader.ModelRegistry",
        return_value=mock_registry,
    ):
        out = LauncherManifest._load_provider_tiles(
            registry_path=registry_path, existing_ids={"dup"}
        )
    assert out == []


def test_skips_models_without_provider_metadata(tmp_path):
    registry_path = tmp_path / "reg.yaml"
    registry_path.write_text("models: []\n", encoding="utf-8")
    mock_registry = MagicMock()
    mock_registry.get_all_models.return_value = [
        _make_model(id="plain", provider="local"),
    ]
    with patch(
        "src.config.launcher_manifest_loader.ModelRegistry",
        return_value=mock_registry,
    ):
        out = LauncherManifest._load_provider_tiles(
            registry_path=registry_path, existing_ids=set()
        )
    assert out == []


def test_builds_tiles_for_provider_models(tmp_path):
    registry_path = tmp_path / "reg.yaml"
    registry_path.write_text("models: []\n", encoding="utf-8")
    mock_registry = MagicMock()
    meta = LauncherPresentationMetadata(
        category="tool", logo="x.svg", status="provider_ready"
    )
    mock_registry.get_all_models.return_value = [
        _make_model(id="new1", launcher=meta, provider="acme"),
        _make_model(id="new2", launcher=meta, provider="acme"),
    ]
    with (
        patch(
            "src.config.launcher_manifest_loader.ModelRegistry",
            return_value=mock_registry,
        ),
        patch(
            "src.config.launcher_manifest_loader.is_engine_runtime_available",
            return_value=True,
        ),
    ):
        out = LauncherManifest._load_provider_tiles(
            registry_path=registry_path, existing_ids=set()
        )
    assert {t.id for t in out} == {"new1", "new2"}


def test_full_load_with_provider_tiles(tmp_path, write_manifest, make_tile):
    manifest_path = write_manifest([make_tile(id="static")])
    registry_path = tmp_path / "reg.yaml"
    registry_path.write_text("models: []\n", encoding="utf-8")
    meta = LauncherPresentationMetadata(
        category="tool", logo="p.svg", status="provider_ready"
    )
    mock_registry = MagicMock()
    mock_registry.get_all_models.return_value = [
        _make_model(id="dynamic", launcher=meta, provider="acme"),
    ]
    with (
        patch(
            "src.config.launcher_manifest_loader.ModelRegistry",
            return_value=mock_registry,
        ),
        patch(
            "src.config.launcher_manifest_loader.is_engine_runtime_available",
            return_value=True,
        ),
    ):
        manifest = LauncherManifest.load(
            manifest_path,
            include_provider_tiles=True,
            registry_path=registry_path,
        )
    ids = {t.id for t in manifest.tiles}
    assert ids == {"static", "dynamic"}
