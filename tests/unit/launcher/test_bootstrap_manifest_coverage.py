"""Tests for launcher-manifest coverage validation in the tool bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.launchers.embedded_tool_bootstrap import missing_embeddable_manifest_tools
from src.shared.python.launcher_embed import (
    EMBEDDABLE_TOOL_REGISTRY,
    EmbedCapabilities,
    register_embeddable_tool,
)

pytestmark = [pytest.mark.unit]


@dataclass
class _Tile:
    id: str
    is_tool: bool


@dataclass
class _Manifest:
    tiles: tuple[_Tile, ...]


class _Tool:
    def __init__(self, tool_id: str) -> None:
        self.tool_id = tool_id

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(supports_embedded=True)

    def create_main_widget(self, parent: object) -> object:  # pragma: no cover
        return object()

    def cleanup(self) -> None:  # pragma: no cover
        pass

    def is_dirty(self) -> bool:  # pragma: no cover
        return False


def test_missing_tools_are_reported_sorted() -> None:
    manifest = _Manifest(
        tiles=(
            _Tile(id="zz_missing", is_tool=True),
            _Tile(id="aa_missing", is_tool=True),
            _Tile(id="some_engine", is_tool=False),
        )
    )
    missing = missing_embeddable_manifest_tools(manifest)
    assert missing == ["aa_missing", "zz_missing"]


def test_registered_tool_is_not_reported() -> None:
    tool_id = "coverage_check_tool"
    register_embeddable_tool(_Tool(tool_id))
    try:
        manifest = _Manifest(tiles=(_Tile(id=tool_id, is_tool=True),))
        assert missing_embeddable_manifest_tools(manifest) == []
    finally:
        EMBEDDABLE_TOOL_REGISTRY.pop(tool_id, None)


def test_engine_tiles_are_ignored() -> None:
    manifest = _Manifest(tiles=(_Tile(id="mujoco_unified", is_tool=False),))
    assert missing_embeddable_manifest_tools(manifest) == []
