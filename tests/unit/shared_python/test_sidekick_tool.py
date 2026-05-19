"""Tests for the SidekickTool embeddable-tool implementation.

Covers:
- Protocol conformance against
  :class:`src.shared.python.launcher_embed.EmbeddableTool`.
- ``embed_capabilities`` fields match the ADR-0013 spec.
- ``src/config/models.yaml`` contains a valid ``chat_assistant`` entry.
- ``create_main_widget(None)`` works headlessly with Qt mocked.

Issue #5468.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.shared.python.launcher_embed import EmbedCapabilities, EmbeddableTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_sidekick_tool() -> Any:
    """Import SidekickTool, skipping if deps are missing."""
    try:
        from src.shared.python.chat.sidekick_tool import SidekickTool

        return SidekickTool
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"SidekickTool not importable: {exc}")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sidekick_tool_is_embeddable() -> None:
    """SidekickTool satisfies the EmbeddableTool runtime-checkable Protocol."""
    SidekickTool = _import_sidekick_tool()
    tool = SidekickTool()
    assert isinstance(tool, EmbeddableTool), (
        "SidekickTool must satisfy the EmbeddableTool structural Protocol"
    )
    assert tool.tool_id == "chat_assistant"


# ---------------------------------------------------------------------------
# EmbedCapabilities fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sidekick_embed_capabilities_fields() -> None:
    """embed_capabilities() returns an EmbedCapabilities with expected values."""
    SidekickTool = _import_sidekick_tool()
    tool = SidekickTool()
    caps = tool.embed_capabilities()

    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.prefers_dock is True
    assert isinstance(caps.min_size, tuple)
    assert len(caps.min_size) == 2
    assert all(v > 0 for v in caps.min_size)
    assert caps.requires_separate_qapplication is False


# ---------------------------------------------------------------------------
# models.yaml entry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_models_yaml_has_chat_assistant_entry() -> None:
    """models.yaml contains a valid chat_assistant entry per ADR-0013."""
    yaml_path = Path(__file__).resolve().parents[3] / "src" / "config" / "models.yaml"
    assert yaml_path.exists(), f"models.yaml not found at {yaml_path}"

    with yaml_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    models = data.get("models", [])
    entry = next((m for m in models if m.get("id") == "chat_assistant"), None)
    assert entry is not None, (
        "models.yaml must contain an entry with id='chat_assistant'"
    )

    assert entry.get("type") == "chat_assistant"
    assert "name" in entry
    assert "description" in entry

    launcher = entry.get("launcher", {})
    assert launcher.get("category") == "assistant", (
        "launcher.category must be 'assistant'"
    )
    assert "status" in launcher


# ---------------------------------------------------------------------------
# create_main_widget — headless (Qt mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sidekick_create_widget_headless() -> None:
    """create_main_widget works headlessly when PyQt6 is mocked."""
    mock_widget_instance = MagicMock()
    mock_ChatDockWidget = MagicMock(return_value=mock_widget_instance)

    with (
        patch.dict(
            "sys.modules",
            {
                "PyQt6": MagicMock(),
                "PyQt6.QtWidgets": MagicMock(),
                "PyQt6.QtCore": MagicMock(),
                "PyQt6.QtWebSockets": MagicMock(),
            },
        ),
        patch(
            "src.shared.python.chat.sidekick_tool._create_chat_dock_widget",
            return_value=mock_widget_instance,
        ),
    ):
        SidekickTool = _import_sidekick_tool()
        tool = SidekickTool()
        widget = tool.create_main_widget(None)

    assert widget is mock_widget_instance


# ---------------------------------------------------------------------------
# cleanup / is_dirty
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sidekick_cleanup_is_idempotent() -> None:
    """cleanup() may be called multiple times without raising."""
    SidekickTool = _import_sidekick_tool()
    tool = SidekickTool()
    tool.cleanup()
    tool.cleanup()


@pytest.mark.unit
def test_sidekick_is_dirty_default_false() -> None:
    """is_dirty() returns False when no widget has been created."""
    SidekickTool = _import_sidekick_tool()
    tool = SidekickTool()
    assert tool.is_dirty() is False


# ---------------------------------------------------------------------------
# DbC precondition on create_main_widget
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sidekick_create_widget_rejects_non_widget_parent() -> None:
    """create_main_widget raises ValueError for an invalid parent type."""
    SidekickTool = _import_sidekick_tool()
    tool = SidekickTool()
    with pytest.raises((ValueError, TypeError)):
        tool.create_main_widget("not-a-widget")  # type: ignore[arg-type]
