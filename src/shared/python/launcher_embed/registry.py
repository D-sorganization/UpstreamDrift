"""Process-wide registry of embeddable tools.

The registry is a simple ``dict[str, EmbeddableTool]`` keyed by
``tool_id``. Registration enforces non-empty ids and rejects duplicates;
this matches the design-by-contract style used elsewhere in the codebase
(public functions raise :class:`ValueError` on contract violations).

The :func:`unregister_embeddable_tool` helper is intended primarily for
test fixtures that need to clear state between tests.
"""

from __future__ import annotations

from .contract import EmbeddableTool

__all__ = [
    "EMBEDDABLE_TOOL_REGISTRY",
    "get_embeddable_tool",
    "is_embeddable",
    "register_embeddable_tool",
    "unregister_embeddable_tool",
]


EMBEDDABLE_TOOL_REGISTRY: dict[str, EmbeddableTool] = {}
"""Module-level mapping from ``tool_id`` to its :class:`EmbeddableTool`.

Mutated by :func:`register_embeddable_tool` and
:func:`unregister_embeddable_tool`. Tests should clear it via a fixture
rather than mutating it directly.
"""


def register_embeddable_tool(tool: EmbeddableTool) -> None:
    """Register ``tool`` under its ``tool_id``.

    Args:
        tool: The tool to register. Must expose a non-empty ``tool_id``
            string and must not already be registered.

    Raises:
        ValueError: If ``tool.tool_id`` is empty or whitespace-only, or
            if a tool with the same id is already registered.
    """
    tool_id = getattr(tool, "tool_id", "")
    # DbC: id must be a non-empty, non-whitespace string.
    if not isinstance(tool_id, str) or not tool_id.strip():
        raise ValueError(
            "register_embeddable_tool: tool.tool_id must be a non-empty string"
        )
    if tool_id in EMBEDDABLE_TOOL_REGISTRY:
        existing = EMBEDDABLE_TOOL_REGISTRY[tool_id]
        cls_existing = type(existing)
        cls_tool = type(tool)
        q_existing = (
            f"{cls_existing.__module__}.{cls_existing.__qualname__}"
            if getattr(cls_existing, "__module__", "")
            else cls_existing.__name__
        )
        q_tool = (
            f"{cls_tool.__module__}.{cls_tool.__qualname__}"
            if getattr(cls_tool, "__module__", "")
            else cls_tool.__name__
        )
        if (
            cls_existing is cls_tool
            or q_existing == q_tool
            or q_existing.split(".")[-2:] == q_tool.split(".")[-2:]
        ):
            return
        raise ValueError(
            f"register_embeddable_tool: tool_id {tool_id!r} is already registered to {q_existing}; got {q_tool}"
        )
    EMBEDDABLE_TOOL_REGISTRY[tool_id] = tool


def get_embeddable_tool(tool_id: str) -> EmbeddableTool | None:
    """Return the registered tool for ``tool_id`` or ``None`` if absent."""
    return EMBEDDABLE_TOOL_REGISTRY.get(tool_id)


def is_embeddable(tool_id: str) -> bool:
    """Return ``True`` if ``tool_id`` is registered and supports embedding.

    Returns ``False`` when the tool is not registered or when its
    :class:`EmbedCapabilities` reports ``supports_embedded=False``.
    """
    tool = EMBEDDABLE_TOOL_REGISTRY.get(tool_id)
    if tool is None:
        return False
    return bool(tool.embed_capabilities().supports_embedded)


def unregister_embeddable_tool(tool_id: str) -> None:
    """Remove ``tool_id`` from the registry.

    Intended for test fixtures.

    Raises:
        ValueError: If ``tool_id`` is not currently registered.
    """
    if tool_id not in EMBEDDABLE_TOOL_REGISTRY:
        raise ValueError(
            f"unregister_embeddable_tool: tool_id {tool_id!r} is not registered"
        )
    del EMBEDDABLE_TOOL_REGISTRY[tool_id]
