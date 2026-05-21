"""Extra coverage for config_utils: ConfigLoader, dot-notation, merge/validate."""

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


class TestLoadJsonConfig:
    def test_none_path_swallowed_by_decorator_returns_default(self) -> None:
        # @log_errors(reraise=False, default_return={}) catches ValueError.
        assert load_json_config(None) == {}  # type: ignore[arg-type]

    def test_missing_returns_default_copy(self, tmp_path: Path) -> None:
        default = {"a": 1}
        result = load_json_config(tmp_path / "missing.json", default=default)
        assert result == default
        # mutation does not leak back
        result["a"] = 999
        assert default["a"] == 1

    def test_missing_no_default_returns_empty(self, tmp_path: Path) -> None:
        assert load_json_config(tmp_path / "missing.json") == {}

    def test_create_if_missing_writes_file(self, tmp_path: Path) -> None:
        path = tmp_path / "new.json"
        load_json_config(path, default={"x": 1}, create_if_missing=True)
        assert path.exists()
        assert json.loads(path.read_text())["x"] == 1

    def test_loads_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"key": "value"}))
        assert load_json_config(path)["key"] == "value"


class TestSaveJsonConfig:
    def test_none_path_swallowed_returns_none(self) -> None:
        # @log_errors(reraise=False) defaults default_return=None.
        assert save_json_config(None, {}) is None  # type: ignore[arg-type]

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "c.json"
        assert save_json_config(path, {"a": 1}) is True
        assert path.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        save_json_config(path, {"x": [1, 2, 3]})
        assert load_json_config(path) == {"x": [1, 2, 3]}


class TestYamlConfig:
    def test_load_none_path_swallowed_returns_default(self) -> None:
        assert load_yaml_config(None) == {}  # type: ignore[arg-type]

    def test_save_none_path_swallowed_returns_none(self) -> None:
        assert save_yaml_config(None, {}) is None  # type: ignore[arg-type]

    def test_load_missing_returns_default_copy(self, tmp_path: Path) -> None:
        default = {"a": 1}
        result = load_yaml_config(tmp_path / "missing.yaml", default=default)
        assert result == default
        result["a"] = 99
        assert default["a"] == 1

    def test_load_missing_no_default_returns_empty(self, tmp_path: Path) -> None:
        assert load_yaml_config(tmp_path / "missing.yaml") == {}

    def test_save_then_load_yaml_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        assert save_yaml_config(path, {"a": {"nested": True}}) is True
        assert load_yaml_config(path) == {"a": {"nested": True}}

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "c.yaml"
        save_yaml_config(path, {"a": 1})
        assert path.exists()


class TestConfigLoader:
    def test_none_path_raises(self) -> None:
        with pytest.raises(ValueError, match="path"):
            ConfigLoader(None)  # type: ignore[arg-type]

    def test_load_unsupported_format_raises(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path / "c.toml", format="toml")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load()

    def test_save_unsupported_format_raises(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path / "c.toml", format="toml")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.save({"a": 1})

    def test_save_none_config_raises(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path / "c.json")
        with pytest.raises(ValueError, match="config"):
            loader.save(None)  # type: ignore[arg-type]

    def test_get_none_key_raises(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path / "c.json")
        with pytest.raises(ValueError, match="key"):
            loader.get(None)  # type: ignore[arg-type]

    def test_set_none_key_raises(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path / "c.json")
        with pytest.raises(ValueError, match="key"):
            loader.set(None, "v")  # type: ignore[arg-type]

    def test_save_then_load_uses_cache(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path / "c.json")
        loader.save({"a": 1})
        # corrupt file on disk; cache should still serve
        (tmp_path / "c.json").write_text("garbage")
        cached = loader.load(use_cache=True)
        assert cached == {"a": 1}

    def test_clear_cache_forces_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"a": 1}))
        loader = ConfigLoader(path)
        loader.load()
        path.write_text(json.dumps({"a": 2}))
        loader.clear_cache()
        assert loader.load(use_cache=False)["a"] == 2

    def test_get_with_dot_notation(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"ui": {"theme": "dark"}}))
        loader = ConfigLoader(path)
        assert loader.get("ui.theme") == "dark"

    def test_get_missing_returns_default(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path / "c.json")
        assert loader.get("nope.nada", default="X") == "X"

    def test_get_partial_path_missing_returns_default(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"ui": "not-a-dict"}))
        loader = ConfigLoader(path)
        assert loader.get("ui.theme", default="fallback") == "fallback"

    def test_set_with_dot_notation_creates_nested(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path / "c.json")
        loader.set("ui.theme", "light")
        assert loader.get("ui.theme") == "light"

    def test_yaml_format(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path / "c.yaml", format="yaml")
        loader.save({"a": 1})
        loader.clear_cache()
        assert loader.load()["a"] == 1


class TestMergeConfigs:
    def test_empty_merge(self) -> None:
        assert merge_configs() == {}

    def test_later_overrides_earlier(self) -> None:
        assert merge_configs({"a": 1}, {"a": 2}) == {"a": 2}

    def test_preserves_non_overlapping_keys(self) -> None:
        result = merge_configs({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_deep_merge_dict_values(self) -> None:
        result = merge_configs({"x": {"a": 1, "b": 2}}, {"x": {"b": 99, "c": 3}})
        assert result == {"x": {"a": 1, "b": 99, "c": 3}}

    def test_non_dict_value_overrides_dict(self) -> None:
        result = merge_configs({"x": {"a": 1}}, {"x": "scalar"})
        assert result == {"x": "scalar"}


class TestValidateConfig:
    def test_none_config_raises(self) -> None:
        with pytest.raises(ValueError, match="config"):
            validate_config(None, required_keys=["a"])  # type: ignore[arg-type]

    def test_valid_when_all_present(self) -> None:
        ok, missing = validate_config({"a": 1, "b": 2}, required_keys=["a", "b"])
        assert ok is True
        assert missing == []

    def test_reports_missing_keys(self) -> None:
        ok, missing = validate_config({"a": 1}, required_keys=["a", "b", "c"])
        assert ok is False
        assert set(missing) == {"b", "c"}

    def test_optional_keys_are_ignored(self) -> None:
        ok, _ = validate_config({"a": 1}, required_keys=["a"], optional_keys=["b", "c"])
        assert ok is True
