"""Tests for config module public API imports.

Addresses issue #5478: config/__init__.py must export get_setting,
load_settings, and save_settings without crashing at import time.
"""

import importlib

import pytest


def test_config_module_imports_cleanly() -> None:
    """The config package must import without AttributeError or ImportError."""
    import sys
    print("SYS PATH:", sys.path)
    try:
        mod = importlib.import_module("src.shared.python.config")
        print("MOD:", mod)
        print("MOD FILE:", getattr(mod, "__file__", "NONE"))
        print("MOD DIR:", dir(mod))
    except Exception as e:
        print("IMPORT ERROR:", e)
        raise
    assert mod is not None


def test_get_setting_exists_and_callable() -> None:
    """get_setting must be exported from the config package and be callable."""
    from src.shared.python.config import get_setting

    assert callable(get_setting)


def test_load_settings_exists() -> None:
    """load_settings must be exported from the config package and be callable."""
    from src.shared.python.config import load_settings

    assert callable(load_settings)


def test_save_settings_exists() -> None:
    """save_settings must be exported from the config package and be callable."""
    from src.shared.python.config import save_settings

    assert callable(save_settings)


def test_get_setting_returns_default_when_key_missing() -> None:
    """get_setting(key, default) returns default when the key is absent."""
    from src.shared.python.config import get_setting

    result = get_setting("__nonexistent_key_5478__", default="fallback")
    assert result == "fallback"


def test_get_setting_no_default_returns_none_when_missing() -> None:
    """get_setting(key) returns None when the key is absent and no default given."""
    from src.shared.python.config import get_setting

    result = get_setting("__nonexistent_key_5478__")
    assert result is None


def test_load_settings_returns_dict() -> None:
    """load_settings() must return a dict."""
    from src.shared.python.config import load_settings

    result = load_settings()
    assert isinstance(result, dict)


def test_save_settings_accepts_dict() -> None:
    """save_settings(data) must accept a dict without raising."""
    from src.shared.python.config import save_settings

    # Should not raise when given a valid dict
    save_settings({"__test_key_5478__": "test_value"})


def test_get_setting_invalid_key_type_raises() -> None:
    """get_setting must raise TypeError when key is not a string (DbC)."""
    from src.shared.python.config import get_setting

    with pytest.raises(TypeError):
        get_setting(123)  # type: ignore[arg-type]


def test_save_settings_invalid_type_raises() -> None:
    """save_settings must raise TypeError when given a non-dict (DbC)."""
    from src.shared.python.config import save_settings

    with pytest.raises(TypeError):
        save_settings("not_a_dict")  # type: ignore[arg-type]
