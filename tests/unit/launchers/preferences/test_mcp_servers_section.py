"""Tests for the UD launcher's MCP servers preferences section.

The section is intentionally thin — its job is to embed the shared
:class:`McpServersPrefsWidget` (Tools PR #2914) in a launcher
preferences pane. Most behaviour is exercised by the Tools-side widget
tests; these tests verify the wrapper contract:

    * ``build_widget()`` returns an actual Qt widget.
    * Calling it twice returns the same instance (idempotent).
    * ``servers`` proxies to the underlying widget.
    * ``persist()`` errors when called before ``build_widget()``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6.QtWidgets")
# Until Tools PR #2914 merges and the vendor submodule is bumped, the
# shared widget package may not be importable in CI. Skip rather than
# fail in that interim window.
pytest.importorskip(
    "src.shared.python.ai.mcp.widgets",
    reason="Pending Tools PR #2914 / vendor bump",
)

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.launchers.preferences.mcp_servers_section import (
    McpServersSection,
)  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.mark.unit
class TestMcpServersSectionConstants:
    def test_section_id(self) -> None:
        assert McpServersSection.SECTION_ID == "mcp_servers"

    def test_section_label(self) -> None:
        assert McpServersSection.SECTION_LABEL == "MCP Servers"


@pytest.mark.unit
class TestMcpServersSectionWidget:
    def test_build_widget_returns_qwidget(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        section = McpServersSection(config_path=tmp_path / "mcp.json")
        widget = section.build_widget()
        # Must be a QWidget (specifically the shared McpServersPrefsWidget).
        from PyQt6.QtWidgets import QWidget

        assert isinstance(widget, QWidget)

    def test_build_widget_is_idempotent(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        section = McpServersSection(config_path=tmp_path / "mcp.json")
        first = section.build_widget()
        second = section.build_widget()
        assert first is second

    def test_servers_empty_before_build(self, tmp_path: Path) -> None:
        section = McpServersSection(config_path=tmp_path / "mcp.json")
        assert section.servers == []

    def test_persist_requires_build(self, tmp_path: Path) -> None:
        section = McpServersSection(config_path=tmp_path / "mcp.json")
        with pytest.raises(RuntimeError, match="build_widget"):
            section.persist()

    def test_persist_after_build_writes_file(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "mcp.json"
        section = McpServersSection(config_path=cfg_path)
        section.build_widget()
        out = section.persist()
        assert out == cfg_path
        assert cfg_path.exists()
