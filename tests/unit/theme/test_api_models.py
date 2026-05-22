"""Tests for src.shared.python.theme.api Pydantic models (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.shared.python.theme.api import (
    ActiveThemeResponse,
    SaveCustomThemeRequest,
    SetActiveThemeRequest,
    ThemeColors,
    ThemeDefinition,
    ThemeListResponse,
    ThemeOperationResponse,
    create_theme_router,
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


class FakeThemeManager:
    def __init__(self) -> None:
        self.builtin = {"Dark": _COLORS, "EmptyBuiltIn": {}}
        self.custom = {"Custom": {**_COLORS, "accent": "#ff00ff"}, "EmptyCustom": {}}
        self.current = "Dark"
        self.saved_requests: list[tuple[str, dict[str, str], bool]] = []
        self.deleted_requests: list[str] = []
        self.changed_requests: list[str] = []
        self.save_error: ValueError | None = None

    def get_builtin_themes(self) -> list[str]:
        return list(self.builtin)

    def get_custom_theme_names(self) -> list[str]:
        return list(self.custom)

    def get_theme_colors(self, name: str) -> dict[str, str]:
        return self.builtin.get(name) or self.custom.get(name) or {}

    def get_current_theme_name(self) -> str:
        return self.current

    def get_current_colors(self) -> dict[str, str]:
        return self.get_theme_colors(self.current)

    def get_available_themes(self) -> list[str]:
        return [*self.builtin, *self.custom]

    def change_theme(self, name: str) -> None:
        self.changed_requests.append(name)
        self.current = name

    def save_custom_theme(self, name: str, colors: dict[str, str], apply: bool) -> str:
        if self.save_error is not None:
            raise self.save_error
        self.saved_requests.append((name, colors, apply))
        self.custom[name] = colors
        if apply:
            self.current = name
        return name

    def delete_custom_theme(self, name: str) -> bool:
        self.deleted_requests.append(name)
        return self.custom.pop(name, None) is not None


def make_theme_client(
    manager: FakeThemeManager | None = None,
) -> tuple[TestClient, FakeThemeManager]:
    manager = manager or FakeThemeManager()
    app = FastAPI()
    app.include_router(create_theme_router(manager, prefix="/themes"))
    return TestClient(app), manager


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


class TestThemeRouter:
    def test_lists_builtin_themes_and_filters_empty_color_sets(self) -> None:
        client, _manager = make_theme_client()

        response = client.get("/themes/builtin")

        assert response.status_code == 200
        assert response.json() == {
            "themes": {
                "Dark": {
                    "name": "Dark",
                    "is_builtin": True,
                    "colors": _COLORS,
                }
            }
        }

    def test_lists_custom_themes_and_filters_empty_color_sets(self) -> None:
        client, _manager = make_theme_client()

        response = client.get("/themes/custom")

        assert response.status_code == 200
        assert response.json()["themes"] == {
            "Custom": {
                "name": "Custom",
                "is_builtin": False,
                "colors": {**_COLORS, "accent": "#ff00ff"},
            }
        }

    def test_lists_all_themes_with_builtin_flags(self) -> None:
        client, _manager = make_theme_client()

        response = client.get("/themes/")

        assert response.status_code == 200
        themes = response.json()["themes"]
        assert themes["Dark"]["is_builtin"] is True
        assert themes["Custom"]["is_builtin"] is False
        assert "EmptyBuiltIn" not in themes
        assert "EmptyCustom" not in themes

    def test_get_active_theme_reports_builtin_status(self) -> None:
        client, manager = make_theme_client()
        manager.current = "Custom"

        response = client.get("/themes/active")

        assert response.status_code == 200
        assert response.json() == {
            "name": "Custom",
            "is_builtin": False,
            "colors": {**_COLORS, "accent": "#ff00ff"},
        }

    def test_set_active_theme_changes_available_theme(self) -> None:
        client, manager = make_theme_client()

        response = client.put("/themes/active", json={"name": "Custom"})

        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "message": "Active theme set to 'Custom'",
            "theme_name": "Custom",
        }
        assert manager.changed_requests == ["Custom"]
        assert manager.current == "Custom"

    def test_set_active_theme_rejects_unknown_theme_with_available_names(self) -> None:
        client, manager = make_theme_client()

        response = client.put("/themes/active", json={"name": "Missing"})

        assert response.status_code == 404
        assert response.json()["detail"] == (
            "Theme 'Missing' not found. "
            "Available: Dark, EmptyBuiltIn, Custom, EmptyCustom"
        )
        assert manager.changed_requests == []

    def test_save_custom_theme_delegates_name_colors_and_apply_flag(self) -> None:
        client, manager = make_theme_client()

        response = client.post(
            "/themes/custom",
            json={"name": "Solar", "colors": _COLORS, "apply": True},
        )

        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "message": "Theme 'Solar' saved successfully",
            "theme_name": "Solar",
        }
        assert manager.saved_requests == [("Solar", _COLORS, True)]
        assert manager.current == "Solar"

    def test_save_custom_theme_converts_manager_value_error_to_bad_request(
        self,
    ) -> None:
        manager = FakeThemeManager()
        manager.save_error = ValueError("reserved theme name")
        client, _manager = make_theme_client(manager)

        response = client.post(
            "/themes/custom",
            json={"name": "Dark", "colors": _COLORS},
        )

        assert response.status_code == 400
        assert response.json() == {"detail": "reserved theme name"}
        assert manager.saved_requests == []

    def test_delete_custom_theme_returns_success_for_existing_theme(self) -> None:
        client, manager = make_theme_client()

        response = client.delete("/themes/custom/Custom")

        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "message": "Theme 'Custom' deleted",
            "theme_name": "Custom",
        }
        assert manager.deleted_requests == ["Custom"]
        assert "Custom" not in manager.custom

    def test_delete_custom_theme_returns_not_found_for_missing_theme(self) -> None:
        client, manager = make_theme_client()

        response = client.delete("/themes/custom/Missing")

        assert response.status_code == 404
        assert response.json() == {"detail": "Custom theme 'Missing' not found"}
        assert manager.deleted_requests == ["Missing"]
