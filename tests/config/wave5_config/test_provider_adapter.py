"""Unit tests for provider-backed tile adaptation helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.config.launcher_manifest_loader import (
    _build_provider_tile,
    _has_provider_metadata,
    _legacy_launcher_metadata,
)
from src.shared.python.config.model_pack_manifest import LauncherPresentationMetadata
from src.shared.python.config.model_registry import ModelConfig


def _make_model(**overrides) -> ModelConfig:
    base = {
        "id": "mod1",
        "name": "Model 1",
        "description": "desc",
        "type": "engine",
        "path": "src/launchers/mod1.py",
    }
    base.update(overrides)
    return ModelConfig(**base)


class TestHasProviderMetadata:
    def test_local_provider_no_source_returns_false(self):
        assert _has_provider_metadata(_make_model(provider="local")) is False

    def test_empty_provider_no_source_returns_false(self):
        assert _has_provider_metadata(_make_model(provider="")) is False

    def test_none_provider_no_source_returns_false(self):
        assert _has_provider_metadata(_make_model(provider=None)) is False

    def test_non_local_provider_returns_true(self):
        assert _has_provider_metadata(_make_model(provider="acme")) is True

    def test_source_root_makes_provider_aware(self):
        assert (
            _has_provider_metadata(
                _make_model(provider="local", source_root="/srv/data")
            )
            is True
        )


class TestLegacyLauncherMetadata:
    def test_engine_uses_known_logo(self):
        meta = _legacy_launcher_metadata(_make_model(engine_type="drake"))
        assert meta.category == "physics_engine"
        assert meta.logo == "drake.svg"
        assert meta.status == "provider_ready"

    def test_engine_falls_back_to_default_logo(self):
        meta = _legacy_launcher_metadata(_make_model(engine_type="unknown_engine"))
        assert meta.category == "physics_engine"
        assert meta.logo == "golf_logo.svg"

    def test_no_engine_marks_external(self):
        # exercises lines 88-90
        meta = _legacy_launcher_metadata(_make_model())
        assert meta.category == "external"
        assert meta.logo == "golf_logo.svg"
        assert meta.status == "external"


class TestBuildProviderTile:
    def test_uses_existing_launcher_metadata(self):
        meta = LauncherPresentationMetadata(
            category="tool", logo="x.svg", status="provider_ready"
        )
        model = _make_model(launcher=meta, engine_type=None)
        # Force runtime available for predictable status
        with patch(
            "src.config.launcher_manifest_loader.is_engine_runtime_available",
            return_value=True,
        ):
            tile = _build_provider_tile(model)
        assert tile.id == "mod1"
        assert tile.category == "tool"
        assert tile.logo == "x.svg"
        assert tile.status == "provider_ready"

    def test_falls_back_to_legacy_metadata(self):
        model = _make_model(engine_type="drake")
        with patch(
            "src.config.launcher_manifest_loader.is_engine_runtime_available",
            return_value=True,
        ):
            tile = _build_provider_tile(model)
        assert tile.category == "physics_engine"
        assert tile.logo == "drake.svg"

    def test_missing_source_root_marks_provider_unavailable(self, tmp_path):
        missing = tmp_path / "nope"
        meta = LauncherPresentationMetadata(
            category="tool", logo="x.svg", status="provider_ready"
        )
        model = _make_model(
            launcher=meta,
            source_root=str(missing),
        )
        with patch(
            "src.config.launcher_manifest_loader.is_engine_runtime_available",
            return_value=True,
        ):
            tile = _build_provider_tile(model)
        assert tile.status == "provider_unavailable"

    def test_existing_source_root_keeps_status(self, tmp_path):
        meta = LauncherPresentationMetadata(
            category="tool", logo="x.svg", status="provider_ready"
        )
        model = _make_model(launcher=meta, source_root=str(tmp_path))
        with patch(
            "src.config.launcher_manifest_loader.is_engine_runtime_available",
            return_value=True,
        ):
            tile = _build_provider_tile(model)
        assert tile.status == "provider_ready"

    def test_runtime_unavailable_marks_status(self):
        meta = LauncherPresentationMetadata(
            category="physics_engine", logo="drake.svg", status="provider_ready"
        )
        model = _make_model(launcher=meta, engine_type="drake")
        with patch(
            "src.config.launcher_manifest_loader.is_engine_runtime_available",
            return_value=False,
        ):
            tile = _build_provider_tile(model)
        assert tile.status == "runtime_unavailable"


@pytest.mark.parametrize(
    "engine,expected",
    [
        ("drake", "drake.svg"),
        ("mujoco", "mujoco_humanoid.svg"),
        ("myosuite", "myosim.svg"),
        ("opensim", "opensim.svg"),
        ("pinocchio", "pinocchio.svg"),
        ("putting_green", "putting_green.svg"),
    ],
)
def test_engine_logo_mapping(engine, expected):
    meta = _legacy_launcher_metadata(_make_model(engine_type=engine))
    assert meta.logo == expected
