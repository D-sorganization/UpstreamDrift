"""Tests for src.shared.python.config.config_utils (Issues #1949, #1744)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.shared.python.config.config_utils import (
    ConfigLoader,
    load_json_config,
    load_yaml_config,
    merge_configs,
    save_json_config,
    save_yaml_config,
    validate_config,
)

# ---------------------------------------------------------------------------
# load_json_config
# ---------------------------------------------------------------------------


class TestLoadJsonConfig:
    def test_missing_file_returns_default(self, tmp_path: Path) -> None:
        result = load_json_config(tmp_path / "missing.json", default={"key": "val"})
        assert result == {"key": "val"}

    def test_missing_file_no_default_returns_empty(self, tmp_path: Path) -> None:
        result = load_json_config(tmp_path / "missing.json")
        assert result == {}

    def test_loads_existing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        p.write_text('{"x": 42}')
        assert load_json_config(p) == {"x": 42}

    def test_create_if_missing(self, tmp_path: Path) -> None:
        p = tmp_path / "new.json"
        default = {"created": True}
        load_json_config(p, default=default, create_if_missing=True)
        assert p.exists()
        assert json.loads(p.read_text()) == default

    def test_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not valid json {{{")
        result = load_json_config(p)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# save_json_config
# ---------------------------------------------------------------------------


class TestSaveJsonConfig:
    def test_saves_file(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        save_json_config(p, {"a": 1})
        assert p.exists()
        assert json.loads(p.read_text())["a"] == 1

    def test_config_utils_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "cfg.json"
        save_json_config(p, {"b": 2})
        assert p.exists()

    def test_returns_true_on_success(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        result = save_json_config(p, {})
        assert result is True

    def test_config_utils_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        data = {"key": "value", "num": 3.14}
        save_json_config(p, data)
        loaded = load_json_config(p)
        assert loaded == data


# ---------------------------------------------------------------------------
# load_yaml_config / save_yaml_config
# ---------------------------------------------------------------------------


class TestLoadYamlConfig:
    def test_missing_file_returns_default(self, tmp_path: Path) -> None:
        result = load_yaml_config(tmp_path / "missing.yaml", default={"a": 1})
        assert result == {"a": 1}

    def test_loads_existing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.yaml"
        p.write_text("key: hello\nnum: 10\n")
        result = load_yaml_config(p)
        assert result["key"] == "hello"
        assert result["num"] == 10


class TestSaveYamlConfig:
    def test_saves_file(self, tmp_path: Path) -> None:
        p = tmp_path / "out.yaml"
        save_yaml_config(p, {"x": 5})
        assert p.exists()

    def test_config_utils_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.yaml"
        data = {"key": "value", "nested": {"a": 1}}
        save_yaml_config(p, data)
        loaded = load_yaml_config(p)
        assert loaded["key"] == "value"
        assert loaded["nested"]["a"] == 1


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------


class TestConfigLoader:
    def test_load_json_missing_uses_default(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path / "cfg.json", format="json")
        result = loader.load(default={"theme": "dark"})
        assert result == {"theme": "dark"}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        loader = ConfigLoader(p)
        loader.save({"color": "blue"})
        loaded = loader.load()
        assert loaded["color"] == "blue"

    def test_caching(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        loader = ConfigLoader(p)
        loader.save({"v": 1})
        loader.load()  # prime cache
        # Modify file directly
        p.write_text('{"v": 99}')
        # Should return cached value
        cached = loader.load(use_cache=True)
        assert cached["v"] == 1

    def test_clear_cache_reloads(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        loader = ConfigLoader(p)
        loader.save({"v": 1})
        loader.load()
        p.write_text('{"v": 99}')
        loader.clear_cache()
        fresh = loader.load()
        assert fresh["v"] == 99

    def test_get_simple_key(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        loader = ConfigLoader(p)
        loader.save({"name": "Alice"})
        assert loader.get("name") == "Alice"

    def test_get_dot_notation(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        loader = ConfigLoader(p)
        loader.save({"ui": {"theme": "light"}})
        assert loader.get("ui.theme") == "light"

    def test_get_missing_key_returns_default(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        loader = ConfigLoader(p)
        loader.save({})
        assert loader.get("nonexistent", default="fallback") == "fallback"

    def test_set_key(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.json"
        loader = ConfigLoader(p)
        loader.save({"a": 1})
        loader.set("b", 2)
        assert loader.get("b") == 2

    def test_config_utils_unsupported_format_raises(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path / "cfg.toml", format="toml")
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load()


# ---------------------------------------------------------------------------
# merge_configs
# ---------------------------------------------------------------------------


class TestMergeConfigs:
    def test_two_flat_dicts(self) -> None:
        result = merge_configs({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_later_overrides_earlier(self) -> None:
        result = merge_configs({"x": 1}, {"x": 99})
        assert result["x"] == 99

    def test_deep_merge(self) -> None:
        result = merge_configs(
            {"ui": {"theme": "dark", "size": 12}}, {"ui": {"theme": "light"}}
        )
        assert result["ui"]["theme"] == "light"
        assert result["ui"]["size"] == 12

    def test_empty_configs(self) -> None:
        result = merge_configs({}, {})
        assert result == {}

    def test_single_config(self) -> None:
        result = merge_configs({"a": 1})
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_all_required_keys_present(self) -> None:
        valid, missing = validate_config({"a": 1, "b": 2}, ["a", "b"])
        assert valid is True
        assert missing == []

    def test_missing_required_key(self) -> None:
        valid, missing = validate_config({"a": 1}, ["a", "b"])
        assert valid is False
        assert "b" in missing

    def test_empty_required_keys(self) -> None:
        valid, missing = validate_config({"a": 1}, [])
        assert valid is True
        assert missing == []

    def test_extra_keys_are_ignored(self) -> None:
        valid, missing = validate_config({"a": 1, "extra": "x"}, ["a"])
        assert valid is True
