"""Tests for opt-in hosted OpenCap session downloads (#11407, ADR-0053)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from src.shared.python.motion_pipeline.sources.opencap_download import (
    OpenCapHostedSettings,
    download_opencap_session,
    get_opencap_hosted_settings,
)
from src.shared.python.motion_pipeline.sources.opencap_layout import (
    OpenCapSessionLayout,
)
from src.shared.python.motion_pipeline.sources.opencap_session import (
    load_opencap_session,
)
from tests.unit.motion_pipeline.sources.opencap_fixtures import (
    OPENCAP_AUGMENTED_MARKERS,
    SCALED_MODEL_OSIM,
    SESSION_METADATA_YAML,
    write_kinematics,
    write_trc,
)

pytestmark = pytest.mark.unit


def _mock_response(content: bytes | str, status: int = 200) -> MagicMock:
    """Helper creating a context-manager response object for urllib."""
    data = content.encode("utf-8") if isinstance(content, str) else content
    mock_resp = MagicMock()
    mock_resp.read.return_value = data
    mock_resp.status = status
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


def test_download_requires_affirmative_consent(tmp_path: Path) -> None:
    settings = OpenCapHostedSettings(enabled=True, api_token="valid_token")
    with pytest.raises(PermissionError, match="consent"):
        download_opencap_session(
            "test-session-123",
            tmp_path,
            consent_recorded=False,
            settings=settings,
        )


def test_download_is_disabled_by_default(tmp_path: Path) -> None:
    # Default settings has enabled=False
    settings = OpenCapHostedSettings(api_token="valid_token")
    assert not settings.enabled
    with pytest.raises(PermissionError, match="disabled by default"):
        download_opencap_session(
            "test-session-123",
            tmp_path,
            consent_recorded=True,
            settings=settings,
        )


def test_download_requires_api_token(tmp_path: Path) -> None:
    settings = OpenCapHostedSettings(enabled=True, api_token=None)
    with pytest.raises(ValueError, match="token"):
        download_opencap_session(
            "test-session-123",
            tmp_path,
            consent_recorded=True,
            settings=settings,
        )


def test_download_rejects_invalid_session_id(tmp_path: Path) -> None:
    settings = OpenCapHostedSettings(enabled=True, api_token="tok")
    for bad_id in ("", "   ", "../escape", "foo/bar", "foo\\bar"):
        with pytest.raises(ValueError, match="session_id"):
            download_opencap_session(
                bad_id,
                tmp_path,
                consent_recorded=True,
                settings=settings,
            )


def test_download_handles_unauthorized_error(tmp_path: Path) -> None:
    settings = OpenCapHostedSettings(enabled=True, api_token="bad_tok")
    http_error = urllib.error.HTTPError(
        url="https://api.opencap.ai/sessions/session-123/",
        code=401,
        msg="Unauthorized",
        hdrs=MagicMock(),
        fp=io.BytesIO(b'{"detail": "Invalid token"}'),
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(PermissionError, match="Unauthorized|Invalid"):
            download_opencap_session(
                "session-123",
                tmp_path,
                consent_recorded=True,
                settings=settings,
            )


def test_download_handles_not_found_error(tmp_path: Path) -> None:
    settings = OpenCapHostedSettings(enabled=True, api_token="tok")
    http_error = urllib.error.HTTPError(
        url="https://api.opencap.ai/sessions/missing-123/",
        code=404,
        msg="Not Found",
        hdrs=MagicMock(),
        fp=io.BytesIO(b'{"detail": "Not found"}'),
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(FileNotFoundError, match="not found"):
            download_opencap_session(
                "missing-123",
                tmp_path,
                consent_recorded=True,
                settings=settings,
            )


def test_download_session_writes_complete_layout(tmp_path: Path) -> None:
    # Prepare dummy file contents using canonical fixtures
    dummy_trc_path = tmp_path / "scratch" / "trial.trc"
    write_trc(dummy_trc_path, OPENCAP_AUGMENTED_MARKERS)
    trc_bytes = dummy_trc_path.read_bytes()

    dummy_mot_path = tmp_path / "scratch"
    write_kinematics(dummy_mot_path, "swing1")
    mot_bytes = (
        dummy_mot_path / "OpenSimData" / "Kinematics" / "swing1.mot"
    ).read_bytes()

    session_data: dict[str, Any] = {
        "id": "session-xyz-456",
        "trials": [
            {
                "id": "trial-neutral-id",
                "name": "neutral",
                "created_at": "2026-10-01T12:00:00Z",
                "results": [
                    {
                        "tag": "session_metadata",
                        "media": "https://media.opencap.ai/sessionMetadata.yaml",
                    },
                    {
                        "tag": "opensim_model",
                        "media": "https://media.opencap.ai/LaiUhlrich2022_scaled.osim",
                    },
                ],
            },
            {
                "id": "trial-swing-id",
                "name": "swing1",
                "created_at": "2026-10-01T12:05:00Z",
                "results": [
                    {
                        "tag": "marker_data",
                        "media": "https://media.opencap.ai/swing1.trc",
                    },
                    {
                        "tag": "ik_results",
                        "media": "https://media.opencap.ai/swing1.mot",
                    },
                ],
            },
        ],
    }

    def fake_urlopen(req: Any, *args: Any, **kwargs: Any) -> MagicMock:
        url = req.full_url if hasattr(req, "full_url") else str(req)
        # Check authorization header on API calls
        if "api.opencap.ai" in url:
            auth = req.headers.get("Authorization") or req.headers.get("authorization")
            assert auth == "Token secret_token"
            if "sessions/session-xyz-456" in url:
                return _mock_response(json.dumps(session_data))
        if "sessionMetadata.yaml" in url:
            return _mock_response(SESSION_METADATA_YAML)
        if "LaiUhlrich2022_scaled.osim" in url:
            return _mock_response(SCALED_MODEL_OSIM)
        if "swing1.trc" in url:
            return _mock_response(trc_bytes)
        if "swing1.mot" in url:
            return _mock_response(mot_bytes)
        raise ValueError(f"Unexpected URL requested: {url}")

    settings = OpenCapHostedSettings(
        enabled=True,
        api_token="secret_token",
    )

    dest = tmp_path / "downloads"
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        session_dir = download_opencap_session(
            "session-xyz-456",
            dest,
            consent_recorded=True,
            settings=settings,
        )

    assert session_dir.is_dir()
    layout = OpenCapSessionLayout.discover(session_dir)
    assert layout.trials == ["swing1"]
    assert layout.subject.mass_kg == pytest.approx(79.5)
    assert layout.subject.subject_id == "subject-01"
    model_file = layout.model_file()
    assert model_file is not None
    assert model_file.name == "LaiUhlrich2022_scaled.osim"

    # End-to-end integration with load_opencap_session
    loaded = load_opencap_session(session_dir)
    assert loaded.trial == "swing1"
    assert loaded.kinematics is not None
    assert len(loaded.observations.frames) > 0


def test_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCAP_API_TOKEN", "env_token_abc")
    monkeypatch.setenv("OPENCAP_HOSTED_ENABLED", "true")
    monkeypatch.setenv("OPENCAP_API_URL", "https://custom.opencap.ai/")

    settings = get_opencap_hosted_settings()
    assert settings.api_token == "env_token_abc"
    assert settings.enabled is True
    assert settings.api_url == "https://custom.opencap.ai/"
