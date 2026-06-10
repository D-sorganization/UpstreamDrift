"""Tests for local server UI diagnostics and API versioning (#2070)."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

local_server = pytest.importorskip("src.api.local_server")


def test_local_server_logs_ui_missing(monkeypatch, tmp_path, caplog) -> None:
    """Local server should warn when UI dist folder is missing."""
    missing_ui_path = tmp_path / "ui" / "dist"
    monkeypatch.setenv("GOLF_UI_DIST", str(missing_ui_path))

    local_server._startup_metrics.update(
        {
            "startup_time": None,
            "static_files_mounted": False,
            "ui_path": None,
            "engines_loaded": [],
            "errors": [],
        }
    )

    caplog.set_level(logging.WARNING)

    local_server.create_local_app()

    assert local_server._startup_metrics["ui_path"] == str(missing_ui_path)
    assert any(
        "UI build not found" in message
        for message in local_server._startup_metrics["errors"]
    )
    assert any("UI build not found" in record.message for record in caplog.records)


# ── API Versioning Tests ──────────────────────────────────────────


def test_api_version_constants() -> None:
    """Local server exposes API_VERSION and API_PREFIX constants (#2070)."""
    assert local_server.API_VERSION == "v1"
    assert local_server.API_PREFIX == "/api/v1"


def test_local_server_has_single_logger_assignment() -> None:
    """local_server should not keep a dead logging.getLogger overwrite (#3008)."""
    source = Path(local_server.__file__).read_text(encoding="utf-8")
    assert "logger = logging.getLogger(__name__)" not in source
    assert source.count("logger = get_logger(__name__)") == 1


def test_local_app_registers_versioned_routes(monkeypatch, tmp_path) -> None:
    """create_local_app registers routes under /api/v1/ prefix (#2070)."""
    missing_ui_path = tmp_path / "ui" / "dist"
    monkeypatch.setenv("GOLF_UI_DIST", str(missing_ui_path))

    local_server._startup_metrics.update(
        {
            "startup_time": None,
            "static_files_mounted": False,
            "ui_path": None,
            "engines_loaded": [],
            "errors": [],
        }
    )

    app = local_server.create_local_app()
    route_paths = [getattr(r, "path", "") for r in app.routes if hasattr(r, "path")]
    versioned = [p for p in route_paths if p.startswith("/api/v1/")]

    assert len(versioned) > 0, (
        f"No /api/v1/ routes found. Registered paths: {route_paths[:20]}"
    )


def test_local_app_keeps_legacy_routes(monkeypatch, tmp_path) -> None:
    """create_local_app keeps legacy /api/ routes for backward compatibility (#2070)."""
    missing_ui_path = tmp_path / "ui" / "dist"
    monkeypatch.setenv("GOLF_UI_DIST", str(missing_ui_path))

    local_server._startup_metrics.update(
        {
            "startup_time": None,
            "static_files_mounted": False,
            "ui_path": None,
            "engines_loaded": [],
            "errors": [],
        }
    )

    app = local_server.create_local_app()
    route_paths = [getattr(r, "path", "") for r in app.routes if hasattr(r, "path")]
    # Legacy routes start with /api/ but NOT /api/v1/
    legacy = [
        p for p in route_paths if p.startswith("/api/") and not p.startswith("/api/v1/")
    ]
    assert len(legacy) > 0, (
        f"No legacy /api/ routes found. Registered paths: {route_paths[:20]}"
    )


def test_local_app_description_mentions_versioning(monkeypatch, tmp_path) -> None:
    """FastAPI app description references the API versioning scheme (#2070)."""
    missing_ui_path = tmp_path / "ui" / "dist"
    monkeypatch.setenv("GOLF_UI_DIST", str(missing_ui_path))

    local_server._startup_metrics.update(
        {
            "startup_time": None,
            "static_files_mounted": False,
            "ui_path": None,
            "engines_loaded": [],
            "errors": [],
        }
    )

    app = local_server.create_local_app()
    assert "v1" in app.description
    assert "/api/v1/" in app.description


def test_local_app_boots_when_engine_manager_is_unavailable(
    monkeypatch, tmp_path
) -> None:
    missing_ui_path = tmp_path / "ui" / "dist"
    monkeypatch.setenv("GOLF_UI_DIST", str(missing_ui_path))

    local_server._startup_metrics.update(
        {
            "startup_time": None,
            "static_files_mounted": False,
            "ui_path": None,
            "engines_loaded": [],
            "errors": [],
        }
    )

    def _raise_import_error() -> object:
        raise ImportError("mocked missing optional engine dependency")

    monkeypatch.setattr(local_server, "_load_engine_manager_class", _raise_import_error)

    app = local_server.create_local_app()

    assert app.state.engine_manager.get_available_engines() == []
    assert any(
        "Engine manager unavailable" in message
        for message in local_server._startup_metrics["errors"]
    )


def test_local_server_startup_modules_keep_engine_manager_import_lazy() -> None:
    repo_root = Path(local_server.__file__).parents[2]
    startup_modules = [
        repo_root / "src" / "api" / "local_server.py",
        repo_root / "src" / "api" / "routes" / "engines.py",
        repo_root / "src" / "api" / "services" / "simulation_service.py",
        repo_root / "src" / "api" / "services" / "analysis_service.py",
        repo_root / "src" / "shared" / "python" / "engine_core" / "workflow_adapter.py",
    ]

    for module_path in startup_modules:
        source = module_path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(module_path))
        top_level_imports = [
            node
            for node in module.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "src.shared.python.engine_core.engine_manager"
        ]
        assert top_level_imports == [], (
            f"{module_path} imports EngineManager at module load time"
        )
