"""Shared fixtures for wave5_config tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _minimal_tile(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid tile dict for manifest tests."""
    base: dict[str, Any] = {
        "id": "alpha",
        "name": "Alpha",
        "description": "Alpha desc",
        "category": "tool",
        "type": "script",
        "path": "src/tools/alpha.py",
        "logo": "alpha.svg",
        "status": "utility",
        "order": 1,
    }
    base.update(overrides)
    return base


@pytest.fixture
def minimal_tile_dict() -> dict[str, Any]:
    """A minimal valid tile entry."""
    return _minimal_tile()


@pytest.fixture
def write_manifest(tmp_path: Path):
    """Factory writing a manifest JSON file and returning its path."""

    def _write(
        tiles: list[dict[str, Any]] | None = None,
        *,
        version: str = "1.0.0",
        description: str = "test manifest",
        extra: dict[str, Any] | None = None,
    ) -> Path:
        payload: dict[str, Any] = {
            "version": version,
            "description": description,
            "tiles": [] if tiles is None else tiles,
        }
        if extra:
            payload.update(extra)
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def make_tile():
    """Factory producing minimal tile dicts with overrides."""
    return _minimal_tile
