from __future__ import annotations

import importlib

import pytest

from src.shared.python.launcher_embed import EmbeddableTool, get_embeddable_tool
from src.tools.launch_monitor_analytics import _embed_adapter

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_adapter_is_pyqt_free_and_satisfies_protocol() -> None:
    adapter = _embed_adapter.LaunchMonitorAnalyticsEmbedAdapter()
    assert isinstance(adapter, EmbeddableTool)
    assert adapter.tool_id == "launch_monitor_analytics"
    assert adapter.embed_capabilities().min_size == (1100, 700)


def test_import_registers_adapter() -> None:
    importlib.reload(_embed_adapter)
    assert get_embeddable_tool("launch_monitor_analytics") is not None


def test_create_and_cleanup_are_idempotent(qapp) -> None:  # noqa: ANN001
    adapter = _embed_adapter.LaunchMonitorAnalyticsEmbedAdapter()
    widget = adapter.create_main_widget(None)
    assert adapter.create_main_widget(None) is widget
    adapter.cleanup()
    adapter.cleanup()
