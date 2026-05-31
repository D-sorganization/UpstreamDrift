"""Canonical-core app-shell registry tests for ADR-0013."""

from __future__ import annotations

import pytest

from src.shared.python.launcher_embed import EmbeddableTool

pytestmark = [pytest.mark.unit]


def test_canonical_core_descriptors_cover_dual_shells() -> None:
    from src.tools.canonical_core.registry import canonical_core_tools

    tools = canonical_core_tools()
    assert {tool.tool_id for tool in tools} == {
        "canonical_core_estimation",
        "canonical_core_comparison",
    }
    for tool in tools:
        assert tool.category == "biomechanics"
        assert tool.default_launch == "tab"
        assert tool.pyqt_adapter == "src.tools.canonical_core._embed_adapter"
        assert "pyqt6" in tool.shell_surfaces
        assert "react" in tool.shell_surfaces
        assert tool.web_route.startswith("/tools/canonical-core/")


def test_canonical_core_adapters_implement_embed_contract() -> None:
    from src.tools.canonical_core._embed_adapter import (
        CanonicalCoreToolEmbedAdapter,
    )
    from src.tools.canonical_core.registry import get_canonical_core_tool

    descriptor = get_canonical_core_tool("canonical_core_estimation")
    adapter = CanonicalCoreToolEmbedAdapter(descriptor)

    assert isinstance(adapter, EmbeddableTool)
    caps = adapter.embed_capabilities()
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    assert caps.requires_separate_qapplication is False


def test_canonical_core_adapters_self_register() -> None:
    from src.shared.python.launcher_embed import get_embeddable_tool
    from src.tools.canonical_core import _embed_adapter  # noqa: F401

    assert get_embeddable_tool("canonical_core_estimation") is not None
    assert get_embeddable_tool("canonical_core_comparison") is not None
