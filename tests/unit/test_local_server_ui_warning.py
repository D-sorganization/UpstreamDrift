"""Tests for local server UI diagnostics."""

from __future__ import annotations

import logging

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
