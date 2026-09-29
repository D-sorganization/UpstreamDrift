"""Runtime ``isinstance`` checks against the :class:`EmbeddableTool` protocol."""

from __future__ import annotations

import pytest

from src.shared.python.launcher_embed import EmbedCapabilities, EmbeddableTool


class _FakeEmbeddableTool:
    """Complete protocol implementation."""

    tool_id = "fake"

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities()

    def create_main_widget(self, parent: object) -> object:
        return object()

    def cleanup(self) -> None:
        return None

    def is_dirty(self) -> bool:
        return False


class _MissingCleanupTool:
    """Implementation missing :meth:`cleanup`."""

    tool_id = "missing-cleanup"

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities()

    def create_main_widget(self, parent: object) -> object:
        return object()

    def is_dirty(self) -> bool:
        return False


@pytest.mark.unit
def test_complete_implementation_passes_isinstance() -> None:
    assert isinstance(_FakeEmbeddableTool(), EmbeddableTool)


@pytest.mark.unit
def test_missing_cleanup_fails_isinstance() -> None:
    assert not isinstance(_MissingCleanupTool(), EmbeddableTool)
