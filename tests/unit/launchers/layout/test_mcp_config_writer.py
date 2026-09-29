"""Tests for :mod:`src.launchers.mcp_config_writer`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.launchers.mcp_config_writer import (
    McpServerConfig,
    expand_env,
    read,
    validate_env_placeholders,
    write,
)


def test_validate_env_placeholders_passthrough() -> None:
    """Well-formed values (with or without placeholders) pass through."""
    assert validate_env_placeholders("plain value") == "plain value"
    assert validate_env_placeholders("${HOME}/cfg") == "${HOME}/cfg"
    assert validate_env_placeholders("$LITERAL") == "$LITERAL"


def test_validate_env_placeholders_empty_braces_rejected() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        validate_env_placeholders("${}")


def test_validate_env_placeholders_unterminated_rejected() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        validate_env_placeholders("prefix-${UNCLOSED")


def test_validate_env_placeholders_none_raises() -> None:
    with pytest.raises(ValueError):
        validate_env_placeholders(None)  # type: ignore[arg-type]


def test_server_config_validates_env_placeholders() -> None:
    with pytest.raises(Exception):  # noqa: B017 — pydantic wraps ValueError
        McpServerConfig(
            name="bad",
            command="echo",
            env={"X": "${}"},
        )


def test_write_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "mcp_servers.json"
    write(
        [McpServerConfig(name="alpha", command="echo", args=["hi"])],
        path=target,
    )
    assert target.exists()
    data = json.loads(target.read_text())
    assert data["servers"][0]["name"] == "alpha"
    assert data["version"] == 1


def test_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "mcp_servers.json"
    servers = [
        McpServerConfig(
            name="alpha",
            command="alpha-cmd",
            args=["--port", "1234"],
            env={"HOME": "${HOME}"},
        ),
        McpServerConfig(name="beta", command="beta-cmd"),
    ]
    write(servers, path=target)
    loaded = read(path=target)
    assert [s.name for s in loaded.servers] == ["alpha", "beta"]
    assert loaded.servers[0].args == ["--port", "1234"]
    assert loaded.servers[0].env == {"HOME": "${HOME}"}


def test_roundtrip_accepts_dicts(tmp_path: Path) -> None:
    target = tmp_path / "mcp_servers.json"
    write(
        [
            {"name": "from-dict", "command": "cmd", "args": [], "env": {}},
        ],
        path=target,
    )
    loaded = read(path=target)
    assert loaded.servers[0].name == "from-dict"


def test_read_missing_returns_empty(tmp_path: Path) -> None:
    target = tmp_path / "absent.json"
    loaded = read(path=target)
    assert loaded.servers == []


def test_read_malformed_json_raises(tmp_path: Path) -> None:
    target = tmp_path / "broken.json"
    target.write_text("not json {{{")
    with pytest.raises(ValueError):
        read(path=target)


def test_read_skips_invalid_entry(tmp_path: Path, caplog) -> None:  # noqa: ARG001
    target = tmp_path / "mixed.json"
    target.write_text(
        json.dumps(
            {
                "version": 1,
                "servers": [
                    {"name": "good", "command": "ok"},
                    {"command": "no-name"},  # invalid — missing name
                ],
            }
        )
    )
    loaded = read(path=target)
    assert [s.name for s in loaded.servers] == ["good"]


def test_duplicate_names_rejected(tmp_path: Path) -> None:
    target = tmp_path / "dup.json"
    with pytest.raises(ValueError, match="Duplicate"):
        write(
            [
                McpServerConfig(name="x", command="a"),
                McpServerConfig(name="x", command="b"),
            ],
            path=target,
        )


def test_write_validation_error_raises_valueerror(tmp_path: Path) -> None:
    target = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        write([{"command": "no-name-given"}], path=target)


def test_expand_env_substitutes_known() -> None:
    assert expand_env("${X}/y", environ={"X": "/tmp"}) == "/tmp/y"


def test_expand_env_keeps_unknown() -> None:
    assert expand_env("${UNSET}", environ={}) == "${UNSET}"


def test_expand_env_none_raises() -> None:
    with pytest.raises(ValueError):
        expand_env(None)  # type: ignore[arg-type]
