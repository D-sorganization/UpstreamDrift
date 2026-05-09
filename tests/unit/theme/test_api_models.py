"""Tests for src.shared.python.theme.api Pydantic models (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.theme.api import (
    ActiveThemeResponse,
    SaveCustomThemeRequest,
    SetActiveThemeRequest,
    ThemeColors,
    ThemeDefinition,
    ThemeListResponse,
    ThemeOperationResponse,
)

_COLORS = {
    "bg": "#1e1e2e",
    "group_bg": "#2a2a3e",
    "border": "#3d3d5c",
    "text": "#cdd6f4",
    "text_secondary": "#a6adc8",
    "label": "#6c7086",
    "focus": "#89b4fa",
    "input_bg": "#181825",
    "accent": "#89b4fa",
    "title_bg": "#313244",
    "title_border": "#45475a",
    "table_header": "#313244",
    "table_alt": "#252535",
    "button_hover": "#7487c8",
}


class TestThemeColors:
    def test_api_models_valid_construction(self) -> None:
        tc = ThemeColors(**_COLORS)
        assert tc.bg == _COLORS["bg"]

    def test_api_models_missing_field_raises(self) -> None:
        incomplete = {k: v for k, v in _COLORS.items() if k != "bg"}
        with pytest.raises((ValueError, TypeError, AssertionError)):
            ThemeColors(**incomplete)

    def test_all_fields_accessible(self) -> None:
        tc = ThemeColors(**_COLORS)
        assert tc.accent == _COLORS["accent"]
        assert tc.button_hover == _COLORS["button_hover"]


class TestThemeDefinition:
    def test_construction_with_name(self) -> None:
        td = ThemeDefinition(name="Dark", colors=_COLORS)
        assert td.name == "Dark"

    def test_default_is_builtin_false(self) -> None:
        td = ThemeDefinition(name="Custom", colors={})
        assert td.is_builtin is False

    def test_is_builtin_true(self) -> None:
        td = ThemeDefinition(name="Light", is_builtin=True, colors=_COLORS)
        assert td.is_builtin is True


class TestThemeListResponse:
    def test_empty_themes(self) -> None:
        resp = ThemeListResponse(themes={})
        assert resp.themes == {}

    def test_with_themes(self) -> None:
        td = ThemeDefinition(name="Dark", colors=_COLORS)
        resp = ThemeListResponse(themes={"Dark": td})
        assert "Dark" in resp.themes


class TestActiveThemeResponse:
    def test_api_models_construction(self) -> None:
        resp = ActiveThemeResponse(name="Dark", is_builtin=True, colors=_COLORS)
        assert resp.name == "Dark"
        assert resp.is_builtin is True

    def test_colors_stored(self) -> None:
        resp = ActiveThemeResponse(name="Light", is_builtin=True, colors=_COLORS)
        assert resp.colors["bg"] == _COLORS["bg"]


class TestSetActiveThemeRequest:
    def test_api_models_name_stored(self) -> None:
        req = SetActiveThemeRequest(name="Monokai")
        assert req.name == "Monokai"

    def test_missing_name_raises(self) -> None:
        with pytest.raises((ValueError, TypeError, AssertionError)):
            SetActiveThemeRequest()  # type: ignore[call-arg]


class TestSaveCustomThemeRequest:
    def test_default_apply_false(self) -> None:
        req = SaveCustomThemeRequest(name="MyTheme", colors=_COLORS)
        assert req.apply is False

    def test_apply_true(self) -> None:
        req = SaveCustomThemeRequest(name="MyTheme", colors=_COLORS, apply=True)
        assert req.apply is True

    def test_colors_stored(self) -> None:
        req = SaveCustomThemeRequest(name="x", colors={"bg": "#000"})
        assert req.colors["bg"] == "#000"


class TestThemeOperationResponse:
    def test_success_true(self) -> None:
        resp = ThemeOperationResponse(success=True, message="done")
        assert resp.solver_status == "success"

    def test_theme_name_optional(self) -> None:
        resp = ThemeOperationResponse(success=True, message="ok")
        assert resp.theme_name is None

    def test_theme_name_set(self) -> None:
        resp = ThemeOperationResponse(success=True, message="ok", theme_name="Dark")
        assert resp.theme_name == "Dark"
