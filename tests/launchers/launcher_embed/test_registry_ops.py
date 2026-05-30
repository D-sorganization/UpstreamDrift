"""Unit tests for the embeddable-tool registry."""

from __future__ import annotations

import pytest

from src.shared.python.launcher_embed import (
    EMBEDDABLE_TOOL_REGISTRY,
    EmbedCapabilities,
    get_embeddable_tool,
    is_embeddable,
    register_embeddable_tool,
    unregister_embeddable_tool,
)

pytestmark = [pytest.mark.unit]


class _Tool:
    def __init__(self, tool_id: str, supports: bool = True) -> None:
        self.tool_id = tool_id
        self._supports = supports

    def embed_capabilities(self) -> EmbedCapabilities:
        return EmbedCapabilities(supports_embedded=self._supports)

    def create_main_widget(self, parent):  # noqa: ANN001
        return object()

    def cleanup(self) -> None:
        pass

    def is_dirty(self) -> bool:
        return False


def test_register_and_get() -> None:
    tool = _Tool("alpha")
    register_embeddable_tool(tool)
    assert get_embeddable_tool("alpha") is tool
    assert "alpha" in EMBEDDABLE_TOOL_REGISTRY


def test_get_missing_returns_none() -> None:
    assert get_embeddable_tool("ghost") is None


def test_register_empty_id_raises() -> None:
    tool = _Tool("")
    with pytest.raises(ValueError, match="non-empty"):
        register_embeddable_tool(tool)


def test_register_whitespace_id_raises() -> None:
    tool = _Tool("   ")
    with pytest.raises(ValueError, match="non-empty"):
        register_embeddable_tool(tool)


def test_register_non_string_id_raises() -> None:
    class BadTool:
        tool_id = 42

        def embed_capabilities(self):  # noqa: ANN001
            return EmbedCapabilities()

        def create_main_widget(self, parent):  # noqa: ANN001
            return object()

        def cleanup(self) -> None:
            pass

        def is_dirty(self) -> bool:
            return False

    with pytest.raises(ValueError, match="non-empty"):
        register_embeddable_tool(BadTool())


def test_register_missing_id_attribute_raises() -> None:
    class NoId:
        def embed_capabilities(self):  # noqa: ANN001
            return EmbedCapabilities()

        def create_main_widget(self, parent):  # noqa: ANN001
            return object()

        def cleanup(self) -> None:
            pass

        def is_dirty(self) -> bool:
            return False

    # getattr default is empty string -> raises
    with pytest.raises(ValueError, match="non-empty"):
        register_embeddable_tool(NoId())  # type: ignore[arg-type]


def test_register_duplicate_raises() -> None:
    class _DifferentTool:
        tool_id = "dup"

        def embed_capabilities(self) -> EmbedCapabilities:
            return EmbedCapabilities()

        def create_main_widget(self, parent: object) -> object:
            return object()

        def cleanup(self) -> None:
            pass

        def is_dirty(self) -> bool:
            return False

    tool = _Tool("dup")
    register_embeddable_tool(tool)
    with pytest.raises(ValueError, match="already registered"):
        register_embeddable_tool(_DifferentTool())


def test_is_embeddable_true() -> None:
    register_embeddable_tool(_Tool("ok"))
    assert is_embeddable("ok") is True


def test_is_embeddable_unregistered_false() -> None:
    assert is_embeddable("missing") is False


def test_is_embeddable_false_when_caps_decline() -> None:
    register_embeddable_tool(_Tool("decline", supports=False))
    assert is_embeddable("decline") is False


def test_unregister_removes_entry() -> None:
    register_embeddable_tool(_Tool("removable"))
    unregister_embeddable_tool("removable")
    assert "removable" not in EMBEDDABLE_TOOL_REGISTRY


def test_unregister_missing_raises() -> None:
    with pytest.raises(ValueError, match="not registered"):
        unregister_embeddable_tool("ghost")
