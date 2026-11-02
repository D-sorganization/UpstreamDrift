"""Unit tests for the theme API route and theme catalog (UD #14265)."""

from __future__ import annotations

import builtins
import importlib
import re
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.theme import _StaticThemeManager
from src.shared.python.theme.colors import BUILTIN_THEMES, resolve_theme_name

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCT_NAME = re.compile(
    r"(ms[-_ ]?)?(word|excel|office|outlook|powerpoint|corporate|sage)", re.IGNORECASE
)
BRAND_HEXES = {"#2b579a", "#217346", "#185abd", "#107c41"}


def test_no_theme_definitions_outside_vendor() -> None:
    """No theme-definitions/themes.json exists outside vendor/ (UD #14265)."""
    matches = list(REPO_ROOT.glob("**/theme-definitions/themes.json"))
    outside_vendor = [
        str(p.relative_to(REPO_ROOT)) for p in matches if "vendor" not in p.parts
    ]
    assert not outside_vendor, f"Found themes.json outside vendor/: {outside_vendor}"


def test_theme_route_imports_without_desktop_theme_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API route discovery must not require the PyQt6 desktop theme stack."""
    real_import = builtins.__import__

    def deny_desktop_theme_manager_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "src.shared.python.theme.theme_manager":
            raise ModuleNotFoundError("No module named 'PyQt6'")
        return real_import(name, globals, locals, fromlist, level)

    for module_name in (
        "src.api.routes.theme",
        "src.shared.python.theme.theme_manager",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(builtins, "__import__", deny_desktop_theme_manager_import)

    route_module = importlib.import_module("src.api.routes.theme")

    app = FastAPI()
    app.include_router(route_module.router)
    response = TestClient(app).get("/themes/active")

    assert response.status_code == 200
    assert response.json()["name"] == "Dark"
    assert response.json()["colors"]["bg"] == "#1a1d23"


def test_api_themes_returns_tools_catalog_without_product_names_or_brand_hexes() -> (
    None
):
    """GET /themes/ returns exactly the Tools catalog without product names or brand hexes."""
    route_module = importlib.import_module("src.api.routes.theme")

    app = FastAPI()
    app.include_router(route_module.router)
    client = TestClient(app)

    response = client.get("/themes/")
    assert response.status_code == 200

    data = response.json()
    themes = data.get("themes", {})

    # Catalog must match Tools built-in themes exactly
    assert set(themes.keys()) == set(BUILTIN_THEMES.keys())
    assert "Cobalt Light" in themes
    assert "Fern Light" in themes

    # No product-derived names in theme keys, names, or categories
    offending_names = [
        name
        for name, theme_def in themes.items()
        if PRODUCT_NAME.search(name)
        or (
            "corporate" in name.lower()
            or "office" in name.lower()
            or "sage" in name.lower()
        )
    ]
    assert not offending_names, f"Product-derived theme names found: {offending_names}"

    # No Microsoft brand hexes in colors
    offending_hexes = [
        f"{name}.{key}={val}"
        for name, theme_def in themes.items()
        for key, val in theme_def.get("colors", {}).items()
        if isinstance(val, str) and val.lower() in BRAND_HEXES
    ]
    assert not offending_hexes, f"Brand hexes found in theme colors: {offending_hexes}"


@pytest.mark.parametrize(
    ("legacy_id", "expected_display_name"),
    [
        ("corporate-blue", "Cobalt Light"),
        ("Corporate Blue", "Cobalt Light"),
        ("sage-professional", "Fern Light"),
        ("Sage Professional", "Fern Light"),
        ("office-blue", "Cobalt Light"),
        ("Office Blue", "Cobalt Light"),
        ("office-green", "Fern Light"),
        ("Office Green", "Fern Light"),
        ("ms-word", "Cobalt Light"),
        ("ms-excel", "Fern Light"),
        ("legal-pad", "Sandstone Yellow"),
        ("monokai", "Neon Warm Dark"),
        ("dracula", "Vampire Dark"),
        ("one-dark", "Frost Dark"),
        ("gitpod-dark", "Teal Dark"),
    ],
)
def test_saved_legacy_ids_resolve_to_renamed_themes(
    legacy_id: str,
    expected_display_name: str,
) -> None:
    """Saved legacy theme preference strings resolve to current Tools theme names."""
    assert resolve_theme_name(legacy_id) == expected_display_name


@pytest.mark.parametrize(
    ("legacy_id", "expected_name"),
    [
        ("corporate-blue", "Cobalt Light"),
        ("sage-professional", "Fern Light"),
        ("ms-word", "Cobalt Light"),
        ("ms-excel", "Fern Light"),
    ],
)
def test_static_theme_manager_loads_legacy_preference(
    legacy_id: str,
    expected_name: str,
) -> None:
    """_StaticThemeManager initializes with and serves colors for legacy theme IDs."""
    mgr = _StaticThemeManager(initial_theme=legacy_id)
    assert mgr.get_current_theme_name() == expected_name

    expected_colors = mgr.get_theme_colors(expected_name)
    assert mgr.get_current_colors() == expected_colors
    assert mgr.get_theme_colors(legacy_id) == expected_colors

    # Changing theme using legacy id also resolves
    mgr.change_theme("Light")
    assert mgr.get_current_theme_name() == "Light"
    mgr.change_theme(legacy_id)
    assert mgr.get_current_theme_name() == expected_name
