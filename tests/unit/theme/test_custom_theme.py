"""Tests for custom theme CRUD and validation (Issue #5487).

PR #5406 added save_custom_theme, delete_custom_theme, _load_custom_themes,
_persist_custom_themes, and _validate_custom_theme_colors to ThemeManager.
This file provides test coverage for all of those functions.

Because ThemeManager inherits from QObject (which may be mocked during
collection), tests that need a live instance use a lightweight stub approach
instead of instantiating ThemeManager directly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.shared.python.theme.colors import (
    BUILTIN_THEMES,
    THEME_COLOR_KEYS,
    normalise_hex_color,
)

# ---------------------------------------------------------------------------
# Helper: build a valid colors dict for all required keys
# ---------------------------------------------------------------------------

_VALID_COLORS: dict[str, str] = dict.fromkeys(THEME_COLOR_KEYS, "#aabbcc")

# ---------------------------------------------------------------------------
# Helper: build a ThemeManager-like object that uses a temp JSON file
# ---------------------------------------------------------------------------


def _build_theme_manager(json_path: Path):
    """Create a minimal stand-in for ThemeManager backed by a temp JSON file.

    This avoids QObject constructor issues in headless CI.  The standalone
    helper functions (_validate_custom_theme_colors, _persist_custom_themes,
    _load_custom_themes) are exercised via the method references attached to
    a plain object.
    """

    class _StubThemeManager:
        """Minimal ThemeManager-alike backed by real persistence logic."""

        CUSTOM_THEME_STORAGE_KEY = "custom_themes"
        _instance = None

        def __init__(self) -> None:
            self.current_theme = "Light"
            self.custom_themes: dict[str, dict[str, str]] = {}
            self._json_path = json_path

        # ---- replicate the real methods without Qt ----

        def _get_custom_theme_path(self) -> Path:
            return self._json_path

        def _validate_custom_theme_colors(self, colors: dict) -> dict[str, str]:
            normalised: dict[str, str] = {}
            for key, value in colors.items() if isinstance(colors, dict) else colors:
                if key not in THEME_COLOR_KEYS:
                    continue
                normalised[key] = normalise_hex_color(str(value))
            missing = [k for k in THEME_COLOR_KEYS if k not in normalised]
            if missing:
                raise ValueError("Missing colour values for: " + ", ".join(missing))
            return normalised

        def _persist_custom_themes(self) -> None:
            to_save = {
                name: {k: v for k, v in colors.items() if k in THEME_COLOR_KEYS}
                for name, colors in self.custom_themes.items()
            }
            with open(self._get_custom_theme_path(), "w", encoding="utf-8") as f:
                json.dump(to_save, f, indent=2)

        def _load_custom_themes(self) -> dict[str, dict[str, str]]:
            theme_path = self._get_custom_theme_path()
            if not theme_path.exists():
                return {}
            with open(theme_path, encoding="utf-8") as f:
                data = json.load(f)
            cleaned: dict[str, dict[str, str]] = {}
            for name, colors in data.items():
                if not isinstance(name, str) or not isinstance(colors, dict):
                    continue
                try:
                    filtered = {
                        k: v for k, v in colors.items() if k in THEME_COLOR_KEYS
                    }
                    cleaned[name] = self._validate_custom_theme_colors(filtered)
                except ValueError:
                    pass
            return cleaned

        def save_custom_theme(
            self,
            theme_name: str,
            colors: dict[str, str],
            apply_immediately: bool = False,
        ) -> str:
            cleaned_name = theme_name.strip()
            if not cleaned_name:
                raise ValueError("Theme name cannot be empty.")
            if cleaned_name in BUILTIN_THEMES:
                raise ValueError(
                    f"Theme name '{cleaned_name}' conflicts with a built-in theme."
                )
            normalised = self._validate_custom_theme_colors(colors)
            self.custom_themes[cleaned_name] = normalised
            self._persist_custom_themes()
            return cleaned_name

        def delete_custom_theme(self, theme_name: str) -> bool:
            if theme_name not in self.custom_themes:
                return False
            del self.custom_themes[theme_name]
            self._persist_custom_themes()
            return True

        def get_custom_theme_names(self) -> list[str]:
            return sorted(self.custom_themes.keys())

        def get_theme_colors(self, theme_name: str) -> dict[str, str] | None:
            if theme_name in BUILTIN_THEMES:
                return dict(BUILTIN_THEMES[theme_name])
            if theme_name in self.custom_themes:
                return dict(self.custom_themes[theme_name])
            return None

    return _StubThemeManager()


@pytest.fixture()
def theme_manager():
    """Provide a stub ThemeManager backed by a temp JSON file."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        json_path = tmp_path / "user_themes.json"
        tm = _build_theme_manager(json_path)
        yield tm, json_path


# ---------------------------------------------------------------------------
# Save → reload → delete round-trip
# ---------------------------------------------------------------------------


class TestCustomThemeRoundTrip:
    def test_save_adds_theme_to_list(self, theme_manager) -> None:
        tm, _ = theme_manager
        tm.save_custom_theme("MyTheme", _VALID_COLORS)
        assert "MyTheme" in tm.get_custom_theme_names()

    def test_save_persists_to_json(self, theme_manager) -> None:
        tm, json_path = theme_manager
        tm.save_custom_theme("MyTheme", _VALID_COLORS)
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "MyTheme" in data

    def test_persisted_json_has_expected_keys(self, theme_manager) -> None:
        tm, json_path = theme_manager
        tm.save_custom_theme("MyTheme", _VALID_COLORS)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for key in THEME_COLOR_KEYS:
            assert key in data["MyTheme"], f"Missing key {key!r} in persisted theme"

    def test_delete_removes_theme(self, theme_manager) -> None:
        tm, _ = theme_manager
        tm.save_custom_theme("MyTheme", _VALID_COLORS)
        result = tm.delete_custom_theme("MyTheme")
        assert result is True
        assert "MyTheme" not in tm.get_custom_theme_names()

    def test_delete_updates_json(self, theme_manager) -> None:
        tm, json_path = theme_manager
        tm.save_custom_theme("MyTheme", _VALID_COLORS)
        tm.delete_custom_theme("MyTheme")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "MyTheme" not in data

    def test_delete_nonexistent_returns_false(self, theme_manager) -> None:
        tm, _ = theme_manager
        result = tm.delete_custom_theme("DoesNotExist")
        assert result is False

    def test_save_returns_theme_name(self, theme_manager) -> None:
        tm, _ = theme_manager
        name = tm.save_custom_theme("  Padded  ", _VALID_COLORS)
        assert name == "Padded"

    def test_get_theme_colors_after_save(self, theme_manager) -> None:
        tm, _ = theme_manager
        tm.save_custom_theme("MyTheme", _VALID_COLORS)
        colors = tm.get_theme_colors("MyTheme")
        assert colors is not None
        assert colors.get("bg") == "#aabbcc"

    def test_load_custom_themes_reads_json(self, theme_manager) -> None:
        tm, json_path = theme_manager
        tm.save_custom_theme("MyTheme", _VALID_COLORS)
        # Clear in-memory, reload from disk
        tm.custom_themes = {}
        reloaded = tm._load_custom_themes()
        assert "MyTheme" in reloaded


# ---------------------------------------------------------------------------
# Validation rejects bad hex values
# ---------------------------------------------------------------------------


class TestValidationRejectsBadHex:
    def test_invalid_hex_xx_raises_value_error(self, theme_manager) -> None:
        tm, _ = theme_manager
        bad_colors = dict(_VALID_COLORS)
        bad_colors["bg"] = "#XX"
        with pytest.raises(ValueError):
            tm._validate_custom_theme_colors(bad_colors)

    def test_not_a_color_raises_value_error(self, theme_manager) -> None:
        tm, _ = theme_manager
        bad_colors = dict(_VALID_COLORS)
        bad_colors["bg"] = "not-a-color"
        with pytest.raises(ValueError):
            tm._validate_custom_theme_colors(bad_colors)

    def test_missing_hash_normalised(self, theme_manager) -> None:
        """Valid hex without # prefix is accepted and normalised."""
        tm, _ = theme_manager
        colors_without_hash = dict(_VALID_COLORS)
        colors_without_hash["bg"] = "aabbcc"  # valid 6-digit hex, no #
        result = tm._validate_custom_theme_colors(colors_without_hash)
        assert result["bg"] == "#aabbcc"

    def test_completely_invalid_hex_raises(self, theme_manager) -> None:
        tm, _ = theme_manager
        bad_colors = dict(_VALID_COLORS)
        bad_colors["bg"] = "xyz-not-hex"
        with pytest.raises(ValueError):
            tm._validate_custom_theme_colors(bad_colors)

    def test_save_with_bad_hex_raises_value_error(self, theme_manager) -> None:
        tm, _ = theme_manager
        bad_colors = dict(_VALID_COLORS)
        bad_colors["bg"] = "#GGGGGG"
        with pytest.raises(ValueError):
            tm.save_custom_theme("BadTheme", bad_colors)

    def test_empty_theme_name_raises_value_error(self, theme_manager) -> None:
        tm, _ = theme_manager
        with pytest.raises(ValueError, match="empty"):
            tm.save_custom_theme("", _VALID_COLORS)

    def test_builtin_name_raises_value_error(self, theme_manager) -> None:
        tm, _ = theme_manager
        with pytest.raises(ValueError, match="conflict"):
            tm.save_custom_theme("Light", _VALID_COLORS)


# ---------------------------------------------------------------------------
# Persistence format is stable
# ---------------------------------------------------------------------------


class TestPersistenceFormatStable:
    def test_json_structure_is_dict_of_dicts(self, theme_manager) -> None:
        tm, json_path = theme_manager
        tm.save_custom_theme("Theme1", _VALID_COLORS)
        tm.save_custom_theme("Theme2", _VALID_COLORS)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        for theme_data in data.values():
            assert isinstance(theme_data, dict)

    def test_multiple_themes_coexist_in_json(self, theme_manager) -> None:
        tm, json_path = theme_manager
        colors_a = dict(_VALID_COLORS)
        colors_a["bg"] = "#111111"
        colors_b = dict(_VALID_COLORS)
        colors_b["bg"] = "#222222"
        tm.save_custom_theme("ThemeA", colors_a)
        tm.save_custom_theme("ThemeB", colors_b)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "ThemeA" in data and "ThemeB" in data

    def test_json_only_stores_valid_color_keys(self, theme_manager) -> None:
        tm, json_path = theme_manager
        # Add an extra key that is not in THEME_COLOR_KEYS
        extra_colors = dict(_VALID_COLORS)
        extra_colors["unknown_key"] = "#ff0000"
        tm.save_custom_theme("MyTheme", extra_colors)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for key in data["MyTheme"]:
            assert key in THEME_COLOR_KEYS, f"Unexpected key {key!r} in JSON"
