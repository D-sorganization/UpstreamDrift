"""The vendored Impact Explorer web bundle mount (`/impact-explorer-app`).

The launcher tile `rate_of_closure` declares a real web route
(`/tools/impact-explorer`); that page embeds the vendored Rate of Closure
React build when the API has mounted it. These tests pin the mount's
contract: present bundle -> served with html fallback; absent bundle ->
tolerated, recorded in startup metrics, never an exception. Degradation is
explicit, never silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI

from src.api import local_server

pytestmark = pytest.mark.unit


def test_resolver_points_into_the_vendored_tools_tree() -> None:
    """The dist path must live under vendor/ud-tools, never a UD copy."""
    dist = local_server._resolve_impact_explorer_dist_path()
    parts = dist.parts
    assert "vendor" in parts and "ud-tools" in parts, dist
    assert parts[-3:] == ("rate_of_closure", "web", "dist"), dist


def test_missing_bundle_is_tolerated_and_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        local_server,
        "_resolve_impact_explorer_dist_path",
        lambda: tmp_path / "nope" / "dist",
    )
    app = FastAPI()
    local_server._mount_impact_explorer_directory(app)
    assert local_server._startup_metrics["impact_explorer_web"] is False
    assert all(r.path != "/impact-explorer-app" for r in app.routes)


def test_present_bundle_is_mounted_with_html_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>roc</title>")
    monkeypatch.setattr(
        local_server, "_resolve_impact_explorer_dist_path", lambda: dist
    )
    app = FastAPI()
    local_server._mount_impact_explorer_directory(app)
    assert local_server._startup_metrics["impact_explorer_web"] is True
    mounts = [r for r in app.routes if getattr(r, "path", "") == "/impact-explorer-app"]
    assert len(mounts) == 1

    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/impact-explorer-app/")
    assert resp.status_code == 200
    assert "roc" in resp.text
