"""Contract tests against the vendored Sidekick voice provider."""

from __future__ import annotations

import io
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import chat_ws
from src.shared.python.sidekick.voice import (
    TranscriptionResult,
    VoiceCapability,
    VoiceTranscriber,
)
from src.shared.python.sidekick import voice as voice_module

pytestmark = pytest.mark.unit


def _audio_payload() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\xe8\x03" * 16_000)
    return buffer.getvalue()


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(chat_ws.router, prefix="/api")
    return TestClient(app)


def test_voice_routes_adapt_provider_status_and_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    monkeypatch.setattr(voice_module, "sweep_stale_voice_temp_files", lambda: 0)
    capability = VoiceCapability(True, "test-engine", "tiny", "ready")
    monkeypatch.setattr(VoiceTranscriber, "detect_capability", lambda self: capability)

    status = client.get("/api/chat/voice/status")
    assert status.status_code == 200
    assert status.headers["cache-control"] == "no-store"
    assert status.json() == {
        "available": True,
        "engine": "test-engine",
        "model": "tiny",
        "max_bytes": chat_ws._MAX_VOICE_BYTES,
        "allowed_mimes": sorted(chat_ws._ALLOWED_VOICE_MIMES),
    }

    result = TranscriptionResult("hello", 1.0, 0.0, "test-engine")
    original_init = VoiceTranscriber.__init__

    def init_with_backend(self: VoiceTranscriber) -> None:
        original_init(self, backend_override=lambda _path: result)

    monkeypatch.setattr(VoiceTranscriber, "__init__", init_with_backend)
    response = client.post(
        "/api/chat/voice/transcribe",
        content=_audio_payload(),
        headers={"content-type": "audio/wav"},
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "text": "hello",
        "duration_s": 1.0,
        "language": None,
        "engine": "test-engine",
        "auto_send": False,
    }


def test_voice_transcription_reports_unavailable_provider_as_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice_module, "sweep_stale_voice_temp_files", lambda: 0)
    monkeypatch.setattr(
        VoiceTranscriber,
        "detect_capability",
        lambda self: VoiceCapability(False, "none", "tiny", "engine unavailable"),
    )
    response = _client().post(
        "/api/chat/voice/transcribe",
        content=_audio_payload(),
        headers={"content-type": "audio/wav"},
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Local Whisper engine is not installed."}
