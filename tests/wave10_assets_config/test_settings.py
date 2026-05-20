"""Tests for src.shared.python.config.settings (in-process settings store)."""

from __future__ import annotations

import pytest
from src.shared.python.config import settings as settings_mod


@pytest.fixture(autouse=True)
def _reset_store() -> None:
    """Reset the module-level settings store between tests."""
    settings_mod._settings_store.clear()
    yield
    settings_mod._settings_store.clear()


class TestGetSetting:
    def test_returns_default_when_missing(self) -> None:
        assert settings_mod.get_setting("missing", default=42) == 42

    def test_returns_none_default(self) -> None:
        assert settings_mod.get_setting("missing") is None

    def test_returns_stored_value(self) -> None:
        settings_mod._settings_store["k"] = "v"
        assert settings_mod.get_setting("k") == "v"

    def test_non_str_key_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            settings_mod.get_setting(123)  # type: ignore[arg-type]

    def test_non_str_key_message_mentions_str(self) -> None:
        with pytest.raises(TypeError, match="str"):
            settings_mod.get_setting(None)  # type: ignore[arg-type]


class TestLoadSettings:
    def test_empty_returns_empty_dict(self) -> None:
        assert settings_mod.load_settings() == {}

    def test_returns_snapshot_copy(self) -> None:
        settings_mod._settings_store["x"] = 1
        snap = settings_mod.load_settings()
        snap["x"] = 999
        # mutation does not leak back
        assert settings_mod._settings_store["x"] == 1

    def test_returns_dict_type(self) -> None:
        assert isinstance(settings_mod.load_settings(), dict)


class TestSaveSettings:
    def test_merge_inserts_new_keys(self) -> None:
        settings_mod.save_settings({"a": 1, "b": 2})
        assert settings_mod.get_setting("a") == 1
        assert settings_mod.get_setting("b") == 2

    def test_merge_overwrites_existing(self) -> None:
        settings_mod.save_settings({"a": 1})
        settings_mod.save_settings({"a": 2})
        assert settings_mod.get_setting("a") == 2

    def test_merge_preserves_unmentioned_keys(self) -> None:
        settings_mod.save_settings({"a": 1, "b": 2})
        settings_mod.save_settings({"a": 99})
        assert settings_mod.get_setting("b") == 2

    def test_non_dict_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            settings_mod.save_settings(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_empty_dict_is_noop(self) -> None:
        settings_mod.save_settings({"a": 1})
        settings_mod.save_settings({})
        assert settings_mod.get_setting("a") == 1


class TestModuleExports:
    def test_all_exports_resolve(self) -> None:
        for name in settings_mod.__all__:
            assert hasattr(settings_mod, name), f"missing export: {name}"

    def test_all_is_a_list(self) -> None:
        assert isinstance(settings_mod.__all__, list)
