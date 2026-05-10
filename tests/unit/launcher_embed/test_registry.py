"""Tests for :mod:`shared.python.launcher_embed.registry`."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from launcher_embed import (
    EMBEDDABLE_TOOL_REGISTRY,
    EmbedCapabilities,
    EmbeddableTool,
    get_embeddable_tool,
    is_embeddable,
    register_embeddable_tool,
    unregister_embeddable_tool,
)


class _FakeEmbeddableTool:
    """Minimal :class:`EmbeddableTool` implementation for tests."""

    def __init__(
        self,
        tool_id: str,
        *,
        supports_embedded: bool = True,
    ) -> None:
        self.tool_id = tool_id
        self._supports_embedded = supports_embedded
        self.cleanup_calls = 0

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(supports_embedded=self._supports_embedded)

    def create_main_widget(self, parent: object) -> object:
        return object()

    def cleanup(self) -> None:
        self.cleanup_calls += 1

    def is_dirty(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterator[None]:
    """Snapshot the registry, run the test, then restore prior state."""
    snapshot = dict(EMBEDDABLE_TOOL_REGISTRY)
    EMBEDDABLE_TOOL_REGISTRY.clear()
    try:
        yield
    finally:
        EMBEDDABLE_TOOL_REGISTRY.clear()
        EMBEDDABLE_TOOL_REGISTRY.update(snapshot)


@pytest.mark.unit
def test_register_then_get_returns_same_instance() -> None:
    tool = _FakeEmbeddableTool("alpha")
    register_embeddable_tool(tool)
    assert get_embeddable_tool("alpha") is tool


@pytest.mark.unit
def test_is_embeddable_true_for_registered_supporting_tool() -> None:
    register_embeddable_tool(_FakeEmbeddableTool("alpha"))
    assert is_embeddable("alpha") is True


@pytest.mark.unit
def test_is_embeddable_false_for_unregistered() -> None:
    assert is_embeddable("ghost") is False


@pytest.mark.unit
def test_is_embeddable_false_when_capabilities_disable_embedding() -> None:
    register_embeddable_tool(_FakeEmbeddableTool("alpha", supports_embedded=False))
    assert is_embeddable("alpha") is False


@pytest.mark.unit
def test_duplicate_registration_raises() -> None:
    register_embeddable_tool(_FakeEmbeddableTool("alpha"))
    with pytest.raises(ValueError, match="already registered"):
        register_embeddable_tool(_FakeEmbeddableTool("alpha"))


@pytest.mark.unit
def test_empty_tool_id_raises() -> None:
    tool = _FakeEmbeddableTool("")
    with pytest.raises(ValueError, match="non-empty"):
        register_embeddable_tool(tool)


@pytest.mark.unit
def test_whitespace_tool_id_raises() -> None:
    tool = _FakeEmbeddableTool("   ")
    with pytest.raises(ValueError, match="non-empty"):
        register_embeddable_tool(tool)


@pytest.mark.unit
def test_unregister_then_get_returns_none() -> None:
    register_embeddable_tool(_FakeEmbeddableTool("alpha"))
    unregister_embeddable_tool("alpha")
    assert get_embeddable_tool("alpha") is None


@pytest.mark.unit
def test_unregister_unknown_raises() -> None:
    with pytest.raises(ValueError, match="not registered"):
        unregister_embeddable_tool("ghost")


@pytest.mark.unit
def test_get_embeddable_tool_unknown_returns_none() -> None:
    assert get_embeddable_tool("ghost") is None


@pytest.mark.unit
def test_registered_fake_satisfies_protocol() -> None:
    tool = _FakeEmbeddableTool("alpha")
    register_embeddable_tool(tool)
    fetched = get_embeddable_tool("alpha")
    assert isinstance(fetched, EmbeddableTool)
