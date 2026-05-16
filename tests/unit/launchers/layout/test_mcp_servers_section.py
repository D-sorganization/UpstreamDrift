"""Tests for :mod:`src.launchers.preferences.mcp_servers_section`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.launchers.mcp_config_writer import McpServerConfig
from src.launchers.preferences.mcp_servers_section import McpServersSection


def test_section_build_widget_empty_file(qt_real, qapp, tmp_path: Path) -> None:  # noqa: ARG001
    """A missing config file renders an empty table without errors."""
    target = tmp_path / "mcp_servers.json"
    section = McpServersSection(config_path=target)
    widget = section.build_widget()
    assert widget.objectName() == "prefs_section_mcp_servers"
    assert section.servers == []


def test_section_loads_existing_config(qt_real, qapp, tmp_path: Path) -> None:  # noqa: ARG001
    target = tmp_path / "mcp_servers.json"
    target.write_text(
        json.dumps(
            {
                "version": 1,
                "servers": [
                    {
                        "name": "preloaded",
                        "command": "echo",
                        "args": ["hi"],
                        "env": {},
                        "enabled": True,
                    }
                ],
            }
        )
    )
    section = McpServersSection(config_path=target)
    section.build_widget()
    names = [s.name for s in section.servers]
    assert names == ["preloaded"]


def test_add_server_appends(qt_real, qapp, tmp_path: Path) -> None:  # noqa: ARG001
    target = tmp_path / "mcp_servers.json"
    section = McpServersSection(config_path=target)
    section.build_widget()
    section.add_server(McpServerConfig(name="new", command="cmd", args=[], env={}))
    assert any(s.name == "new" for s in section.servers)


def test_remove_server(qt_real, qapp, tmp_path: Path) -> None:  # noqa: ARG001
    target = tmp_path / "mcp_servers.json"
    section = McpServersSection(config_path=target)
    section.build_widget()
    section.add_server(McpServerConfig(name="r", command="x"))
    assert section.remove_server("r") is True
    assert section.remove_server("r") is False


def test_persist_writes_json(qt_real, qapp, tmp_path: Path) -> None:  # noqa: ARG001
    target = tmp_path / "mcp_servers.json"
    section = McpServersSection(config_path=target)
    section.build_widget()
    section.add_server(McpServerConfig(name="persisted", command="x"))
    section.persist()
    assert target.exists()
    data = json.loads(target.read_text())
    assert any(s["name"] == "persisted" for s in data["servers"])


def test_add_server_rejects_non_model(qt_real, qapp, tmp_path: Path) -> None:  # noqa: ARG001
    target = tmp_path / "mcp_servers.json"
    section = McpServersSection(config_path=target)
    section.build_widget()
    with pytest.raises(TypeError):
        section.add_server({"name": "nope"})  # type: ignore[arg-type]
