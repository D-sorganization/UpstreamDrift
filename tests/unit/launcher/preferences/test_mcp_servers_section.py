"""Unit tests for MCP server management preferences section.

Tests the pure-Python data model (McpServersConfig, McpServerEntry)
without requiring Qt — verifying load/save, validation, and env-var
placeholder behaviour.

Issue: #5642
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server_entry(
    name: str = "test-server",
    transport: str = "stdio",
    command: str = "npx",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    enabled: bool = True,
) -> dict:
    return {
        "name": name,
        "transport": transport,
        "command": command,
        "args": args or ["-y", "@modelcontextprotocol/server-filesystem"],
        "env": env or {},
        "enabled": enabled,
    }


# ---------------------------------------------------------------------------
# Test: instantiate the data model without Qt
# ---------------------------------------------------------------------------


class TestMcpServersConfigInstantiation:
    """McpServersConfig must be importable and instantiable without PyQt6."""

    def test_mcp_servers_section_creates_without_qt(self) -> None:
        """Import and instantiate McpServersConfig with no Qt on the path."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig()
        assert isinstance(cfg, McpServersConfig)
        assert cfg.servers == []


# ---------------------------------------------------------------------------
# Test: load config from JSON
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_config_from_json_file(self, tmp_path: Path) -> None:
        """McpServersConfig.load() reads servers list from a JSON file."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        data = {
            "servers": [
                _make_server_entry("fs-server"),
                _make_server_entry(
                    "brave-search", env={"BRAVE_API_KEY": "${BRAVE_API_KEY}"}
                ),
            ]
        }
        config_file = tmp_path / "mcp_servers.json"
        config_file.write_text(json.dumps(data), encoding="utf-8")

        cfg = McpServersConfig.load(config_file)
        assert len(cfg.servers) == 2
        assert cfg.servers[0]["name"] == "fs-server"
        assert cfg.servers[1]["name"] == "brave-search"

    def test_load_missing_file_returns_empty_config(self, tmp_path: Path) -> None:
        """Loading a non-existent file returns an empty McpServersConfig."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig.load(tmp_path / "nonexistent.json")
        assert cfg.servers == []


# ---------------------------------------------------------------------------
# Test: save config to JSON
# ---------------------------------------------------------------------------


class TestSaveConfig:
    def test_save_config_to_json_file(self, tmp_path: Path) -> None:
        """McpServersConfig.save() persists servers to a JSON file."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig()
        cfg.servers.append(_make_server_entry("my-server"))
        out_path = tmp_path / "mcp_servers.json"

        cfg.save(out_path)

        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert "servers" in written
        assert len(written["servers"]) == 1
        assert written["servers"][0]["name"] == "my-server"

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        """save() creates the parent directory if it does not exist."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig()
        nested = tmp_path / "a" / "b" / "mcp_servers.json"
        cfg.save(nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# Test: add server validation
# ---------------------------------------------------------------------------


class TestAddServer:
    def test_add_server_entry_validates_required_fields(self) -> None:
        """add_server() raises ValueError when name is missing."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig()
        with pytest.raises(ValueError, match="name"):
            cfg.add_server({"transport": "stdio", "command": "npx"})

    def test_add_server_stores_entry(self) -> None:
        """A valid server dict is appended to cfg.servers."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig()
        entry = _make_server_entry("new-server")
        cfg.add_server(entry)
        assert len(cfg.servers) == 1
        assert cfg.servers[0]["name"] == "new-server"

    def test_add_server_duplicate_name_raises(self) -> None:
        """Adding a second server with the same name raises ValueError."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig()
        cfg.add_server(_make_server_entry("dup"))
        with pytest.raises(ValueError, match="already exists"):
            cfg.add_server(_make_server_entry("dup"))


# ---------------------------------------------------------------------------
# Test: disable server
# ---------------------------------------------------------------------------


class TestDisableServer:
    def test_disable_server_sets_enabled_false(self) -> None:
        """disable_server('name') sets the enabled flag to False."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig()
        cfg.add_server(_make_server_entry("alpha", enabled=True))
        cfg.disable_server("alpha")
        assert cfg.servers[0]["enabled"] is False

    def test_disable_unknown_server_raises(self) -> None:
        """disable_server() with an unknown name raises ValueError."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig()
        with pytest.raises(ValueError, match="not found"):
            cfg.disable_server("ghost")


# ---------------------------------------------------------------------------
# Test: remove server
# ---------------------------------------------------------------------------


class TestRemoveServer:
    def test_remove_server_removes_by_name(self) -> None:
        """remove_server('name') deletes that entry from cfg.servers."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig()
        cfg.add_server(_make_server_entry("keep"))
        cfg.add_server(_make_server_entry("gone"))
        cfg.remove_server("gone")
        assert len(cfg.servers) == 1
        assert cfg.servers[0]["name"] == "keep"

    def test_remove_unknown_server_raises(self) -> None:
        """remove_server() with an unknown name raises ValueError."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig()
        with pytest.raises(ValueError, match="not found"):
            cfg.remove_server("phantom")


# ---------------------------------------------------------------------------
# Test: env var placeholder storage
# ---------------------------------------------------------------------------


class TestEnvVarPlaceholders:
    def test_env_var_values_stored_as_placeholders(self) -> None:
        """Env vars must be stored as '${VAR_NAME}' — never as raw secrets."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig()
        entry = _make_server_entry(
            "secret-server",
            env={"API_KEY": "${API_KEY}", "DB_PASS": "${DB_PASS}"},
        )
        cfg.add_server(entry)
        stored_env = cfg.servers[0]["env"]
        for key, val in stored_env.items():
            assert val.startswith("${") and val.endswith("}"), (
                f"env var {key!r} stored as raw value {val!r} — must be a placeholder"
            )

    def test_add_server_raises_on_raw_secret_in_env(self) -> None:
        """add_server() raises ValueError if an env value is not a placeholder."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        cfg = McpServersConfig()
        entry = _make_server_entry(
            "leaky",
            env={"API_KEY": "sk-supersecret-value"},
        )
        with pytest.raises(ValueError, match="placeholder"):
            cfg.add_server(entry)


# ---------------------------------------------------------------------------
# Test: default config file path
# ---------------------------------------------------------------------------


class TestDefaultConfigPath:
    def test_mcp_config_file_path_default(self) -> None:
        """McpServersConfig.default_path() returns ~/.upstreamdrift/mcp_servers.json."""
        from src.launchers.mcp_servers_preferences import McpServersConfig

        expected = Path.home() / ".upstreamdrift" / "mcp_servers.json"
        assert McpServersConfig.default_path() == expected
