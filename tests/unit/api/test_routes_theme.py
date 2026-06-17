"""Unit tests for the theme API route."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


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
