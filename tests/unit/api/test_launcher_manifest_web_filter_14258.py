"""Web launcher manifest hides tiles whose bundles are absent (#14258)."""

from __future__ import annotations

import pytest

local_server = pytest.importorskip("src.api.local_server")

pytestmark = pytest.mark.unit


def test_rate_of_closure_hidden_when_bundle_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(local_server._startup_metrics, "impact_explorer_web", False)
    manifest = {
        "version": "1",
        "tiles": [
            {"id": "rate_of_closure", "logo": "rate_of_closure.svg"},
            {"id": "chat_assistant", "logo": "sidekick.svg"},
        ],
    }
    filtered = local_server._filter_web_manifest_tiles(manifest)
    ids = [tile["id"] for tile in filtered["tiles"]]
    assert "rate_of_closure" not in ids
    assert "chat_assistant" in ids


def test_rate_of_closure_kept_when_bundle_mounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(local_server._startup_metrics, "impact_explorer_web", True)
    manifest = {
        "version": "1",
        "tiles": [{"id": "rate_of_closure", "logo": "rate_of_closure.svg"}],
    }
    filtered = local_server._filter_web_manifest_tiles(manifest)
    assert len(filtered["tiles"]) == 1
