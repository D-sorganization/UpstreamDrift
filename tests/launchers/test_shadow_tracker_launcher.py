"""Unit tests for Shadow Tracker Launcher Integration (ST-11, #10134).

Tests verify:
- ShadowTrackerAdapter implements EmbeddableTool and BackgroundableTool protocols.
- Adapter creates the tool widget cleanly or degrades gracefully when PyQt6 is missing.
- Launcher manifest contains shadow_tracker with honest 'utility' status and valid fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest

from src.config.launcher_manifest_loader import LauncherManifest
from src.shared.python.launcher_embed import (
    BackgroundableTool,
    EmbedCapabilities,
    EmbeddableTool,
    get_embeddable_tool,
    is_embeddable,
)
from src.tools.shadow_tracker._embed_adapter import ShadowTrackerAdapter

pytestmark = pytest.mark.unit


def test_shadow_tracker_adapter_satisfies_protocols() -> None:
    """ShadowTrackerAdapter must conform to EmbeddableTool and BackgroundableTool."""
    adapter = ShadowTrackerAdapter()

    assert isinstance(adapter, EmbeddableTool)
    assert isinstance(adapter, BackgroundableTool)
    assert adapter.tool_id == "shadow_tracker"
    assert adapter.display_name == "Shadow Tracker"

    caps = adapter.embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert adapter.is_dirty() is False
    assert adapter.can_background() is True


def test_shadow_tracker_adapter_creates_widget(qapp: Any) -> None:
    """Creating main widget returns a usable review widget or graceful fallback."""
    adapter = ShadowTrackerAdapter()
    widget = adapter.create_main_widget()
    assert widget is not None
    # Cleanup should be idempotent and not raise
    adapter.cleanup()


def test_launcher_manifest_has_shadow_tracker_tile() -> None:
    """The shared launcher manifest must register the shadow_tracker tile."""
    manifest = LauncherManifest.load()
    tile = manifest.get_tile("shadow_tracker")

    assert tile is not None
    assert tile.name == "Shadow Tracker"
    assert tile.category == "motion_capture"
    assert tile.status == "utility"  # Honest readiness, not false 'gui_ready'
    assert "video_import" in tile.capabilities
    assert "manual_mask_editing" in tile.capabilities
    assert "bundle_persistence" in tile.capabilities
    assert tile.path == "src/tools/shadow_tracker/__main__.py"
