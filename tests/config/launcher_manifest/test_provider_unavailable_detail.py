"""Provider-unavailable tiles must carry an explicit reason (issue #8852)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import launcher_manifest_loader as loader
from src.shared.python.config.tools_vendor_authority import ToolsVendorAuthority

pytestmark = pytest.mark.unit

_PIN = "aa" * 20
_FOUND = "bb" * 20
_STALE_REASON = f"Tools pin stale (expected {_PIN}, found {_FOUND})"


def _tools_model() -> SimpleNamespace:
    return SimpleNamespace(
        id="video_analyzer",
        provider="tools",
        source_root=None,
        engine_type=None,
    )


def _stub_authority(monkeypatch: pytest.MonkeyPatch, *, available: bool) -> None:
    result = ToolsVendorAuthority(
        root=Path("vendor/ud-tools"),
        expected_sha=_PIN,
        available=available,
        reason=None if available else _STALE_REASON,
    )
    monkeypatch.setattr(
        loader, "inspect_tools_vendor_authority", lambda _repo_root: result
    )


def test_stale_pin_degrades_with_explicit_detail(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _stub_authority(monkeypatch, available=False)

    with caplog.at_level("WARNING", logger=loader.logger.name):
        status, detail = loader._provider_status(
            _tools_model(), "provider_ready", Path("."), check_runtime=False
        )

    assert status == "provider_unavailable"
    assert detail == f"unavailable: {_STALE_REASON}"
    assert any(_STALE_REASON in record.getMessage() for record in caplog.records)


def test_available_authority_keeps_declared_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_authority(monkeypatch, available=True)

    status, detail = loader._provider_status(
        _tools_model(), "provider_ready", Path("."), check_runtime=False
    )

    assert status == "provider_ready"
    assert detail is None


def test_tile_serializes_status_detail_for_api_consumers() -> None:
    tile = loader.LauncherTile(
        id="video_analyzer",
        name="Video Analyzer",
        description="d",
        category="tool",
        type="tools",
        path="x",
        logo="golf_logo.svg",
        status="provider_unavailable",
        status_detail=f"unavailable: {_STALE_REASON}",
    )

    payload = tile.to_dict()

    assert payload["status"] == "provider_unavailable"
    assert payload["status_detail"] == f"unavailable: {_STALE_REASON}"
    round_tripped = loader.LauncherTile.from_dict(payload)
    assert round_tripped.status_detail == f"unavailable: {_STALE_REASON}"
