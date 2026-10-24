"""Unit and failure-case tests for Sidekick subscription CLI adapters, host simulation actions, and local Whisper VTT (#13660)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import chat_ws
from src.api.services.chat_service import ChatService
from src.launchers.sidekick_host_port import LauncherHostActionPort

pytestmark = pytest.mark.unit


class _DummyOrchestrator:
    active_engine = "mujoco"

    def __init__(self) -> None:
        self.configured: dict[str, Any] = {}
        self.runs: int = 0

    def get_simulation_status(self) -> dict[str, Any]:
        return {"engine": self.active_engine, "running": False, "runs": self.runs}

    def configure_simulation(self, **kwargs: Any) -> None:
        self.configured.update(kwargs)

    def run_simulation(self, **_kwargs: Any) -> dict[str, Any]:
        self.runs += 1
        return {"frames": 500, "peak_torque_nm": 42.5}

    def summarize_simulation_metrics(self) -> dict[str, Any]:
        return {"peak_torque_nm": 42.5, "rms_tracking_error_rad": 0.012}


class _DummyLauncher:
    def __init__(self) -> None:
        self.orchestrator = _DummyOrchestrator()


def test_subscription_cli_adapters_load_without_paid_keys() -> None:
    """claude_code, codex_cli, and agy load without paid API keys."""
    with patch.object(ChatService, "_load_adapter"):
        svc = ChatService()

    claude_adapter = svc._load_subscription_cli_adapter(
        "claude_code", "claude-sonnet-4-5"
    )
    assert claude_adapter is not None
    assert type(claude_adapter).__name__ == "ClaudeCodeAdapter"

    codex_adapter = svc._load_subscription_cli_adapter("codex_cli", "gpt-5-codex")
    assert codex_adapter is not None
    assert type(codex_adapter).__name__ == "CodexCliAdapter"

    agy_adapter = svc._load_subscription_cli_adapter("agy", "gemini-2.5-pro")
    assert agy_adapter is not None
    assert type(agy_adapter).__name__ in {"AntigravityCliAdapter", "GeminiCliAdapter"}

    # Failure case: unknown provider returns None
    assert svc._load_subscription_cli_adapter("unknown_cli") is None


def test_launcher_host_simulation_capabilities_and_validation() -> None:
    """LauncherHostActionPort exposes simulation capabilities and rejects out-of-range parameters."""
    launcher = _DummyLauncher()
    port = LauncherHostActionPort(launcher)

    cap_ids = {cap.capability_id for cap in port.list_capabilities()}
    assert "host.simulation.get_status" in cap_ids
    assert "host.simulation.configure" in cap_ids
    assert "host.simulation.run" in cap_ids
    assert "host.simulation.summarize_metrics" in cap_ids

    status_res = port.invoke("host.simulation.get_status", {})
    assert status_res.ok is True
    assert status_res.value["engine"] == "mujoco"

    # Failure cases: negative timestep or excessive duration rejected by host
    bad_dt = port.invoke("host.simulation.configure", {"timestep": -0.01})
    assert bad_dt.ok is False
    assert "timestep" in (bad_dt.error or "")

    bad_dur = port.invoke(
        "host.simulation.configure", {"timestep": 0.002, "duration": 9999.0}
    )
    assert bad_dur.ok is False
    assert "duration" in (bad_dur.error or "")

    ok_cfg = port.invoke(
        "host.simulation.configure", {"timestep": 0.001, "duration": 2.5}
    )
    assert ok_cfg.ok is True
    assert launcher.orchestrator.configured["timestep"] == 0.001

    run_res = port.invoke("host.simulation.run", {})
    assert run_res.ok is True
    assert run_res.value["status"] == "completed"

    metrics_res = port.invoke("host.simulation.summarize_metrics", {})
    assert metrics_res.ok is True
    assert metrics_res.value["metrics"]["peak_torque_nm"] == 42.5


@dataclass(frozen=True)
class _FakeVoiceResult:
    text: str = "set timestep to one millisecond"
    duration_s: float = 1.4
    language: str = "en"
    engine: str = "faster-whisper"


def test_chat_voice_routes_status_and_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Voice status and transcribe endpoints enforce MIME/size checks and never auto-send."""
    from src.shared.python.sidekick import voice as voice_module

    monkeypatch.setattr(voice_module, "sweep_stale_voice_temp_files", lambda: 0)
    app = FastAPI()
    app.include_router(chat_ws.router, prefix="/api")
    client = TestClient(app)

    status_resp = client.get("/api/chat/voice/status")
    assert status_resp.status_code == 200
    assert status_resp.headers.get("cache-control") == "no-store"
    assert "max_bytes" in status_resp.json()

    # Failure case 1: empty audio body -> 400
    empty_resp = client.post(
        "/api/chat/voice/transcribe",
        content=b"",
        headers={"content-type": "audio/wav"},
    )
    assert empty_resp.status_code == 400

    # Failure case 2: unsupported MIME type -> 415
    mime_resp = client.post(
        "/api/chat/voice/transcribe",
        content=b"RIFF1234",
        headers={"content-type": "application/octet-stream"},
    )
    assert mime_resp.status_code == 415

    # Valid transcription via stubbed VoiceTranscriber
    import sys
    import types

    fake_mod = types.ModuleType("src.shared.python.sidekick.voice")

    class _FakeTranscriber:
        def status(self) -> dict[str, Any]:
            return {"available": True, "engine": "faster-whisper"}

        def transcribe_bytes(
            self, data: bytes, *, mime_type: str = "audio/wav"
        ) -> _FakeVoiceResult:
            assert data.startswith(b"RIFF")
            assert mime_type == "audio/wav"
            return _FakeVoiceResult()

    from src.shared.python.sidekick.voice import VoiceUnavailableError

    fake_mod.VoiceTranscriber = _FakeTranscriber  # type: ignore[attr-defined]
    fake_mod.VoiceUnavailableError = VoiceUnavailableError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.shared.python.sidekick.voice", fake_mod)

    ok_resp = client.post(
        "/api/chat/voice/transcribe",
        content=b"RIFF\x00\x00\x00\x00WAVEfmt ",
        headers={"content-type": "audio/wav"},
    )
    assert ok_resp.status_code == 200
    assert ok_resp.headers.get("cache-control") == "no-store"
    body = ok_resp.json()
    assert body["text"] == "set timestep to one millisecond"
    assert body["auto_send"] is False
