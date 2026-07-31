"""Tests for Sidekick app-state context injection into chat WebSocket.

Covers:
    * Recent app state is appended as a system message when the buffer
      is populated.
    * No injection happens when the buffer is empty.
    * ``UPSTREAMDRIFT_SIDEKICK_CONTEXT=0`` disables injection even when
      the buffer has events.
    * Sensitive keys/values are redacted in the dump.
    * The payload stays under the 4 KB soft cap regardless of pushed
      volume.

The chat_ws module lives under ``src/api/routes/`` whose package init
transitively imports ``src/shared/python/config/__init__.py`` — that
file currently re-exports names that don't exist in ``settings.py`` (a
pre-existing repo bug). To keep this unit test runnable against the
real ``chat_context`` integration without depending on that fix, we
load ``chat_ws.py`` via ``importlib.util.spec_from_file_location`` so
the package init is bypassed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from src.shared.python.ai import chat_context


def _load_chat_ws() -> ModuleType:
    """Load ``src/api/routes/chat_ws.py`` without running the package init."""
    repo_root = Path(__file__).resolve().parents[3]
    source = repo_root / "src" / "api" / "routes" / "chat_ws.py"
    spec = importlib.util.spec_from_file_location("_chat_ws_under_test", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load chat_ws spec from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_chat_ws = _load_chat_ws()


class _CapturingSession:
    """Minimal session double that records ``add_message`` calls."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append((role, content))


@pytest.fixture(autouse=True)
def _reset_buffer_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests start from a clean ring buffer and default env."""
    chat_context.reset_buffer()
    monkeypatch.delenv("UPSTREAMDRIFT_SIDEKICK_CONTEXT", raising=False)


# -- Helper-level integration -----------------------------------------


def test_populated_buffer_injects_recent_app_state_system_message() -> None:
    """When events are buffered, ``_maybe_inject_chat_context`` adds a system message."""
    chat_context.record_event("diagnostic", {"name": "engine_check", "status": "ok"})
    session = _CapturingSession()

    injected = _chat_ws._maybe_inject_chat_context(session)

    assert injected is not None
    assert injected.startswith("Recent app state:")
    assert "engine_check" in injected
    assert len(session.messages) == 1
    role, content = session.messages[0]
    assert role == "system"
    assert content == injected


def test_empty_buffer_yields_no_injection() -> None:
    """With an empty buffer, no message is added."""
    session = _CapturingSession()

    injected = _chat_ws._maybe_inject_chat_context(session)

    assert injected is None
    assert session.messages == []


def test_env_var_off_disables_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    """``UPSTREAMDRIFT_SIDEKICK_CONTEXT=0`` skips injection."""
    monkeypatch.setenv("UPSTREAMDRIFT_SIDEKICK_CONTEXT", "0")
    chat_context.record_event("diagnostic", {"name": "engine_check"})
    session = _CapturingSession()

    injected = _chat_ws._maybe_inject_chat_context(session)

    assert injected is None
    assert session.messages == []


def test_injection_skipped_when_session_lacks_add_message() -> None:
    """Session doubles without ``add_message`` are silently skipped."""
    chat_context.record_event("diagnostic", {"name": "engine_check"})

    injected = _chat_ws._maybe_inject_chat_context(object())

    assert injected is None


# -- Privacy / size cap -----------------------------------------------


def test_password_field_is_redacted_in_dump() -> None:
    """Leaf values keyed under ``password`` are scrubbed."""
    chat_context.record_event("auth", {"user": "alice", "password": "hunter2"})

    payload = chat_context.get_chat_context()
    events = payload["events"]
    assert len(events) == 1
    body = events[0]["payload"]
    assert body["password"] == "<redacted>"
    assert body["user"] == "alice"


def test_sensitive_keys_redacted_recursively() -> None:
    """Nested mappings/lists are scrubbed."""
    chat_context.record_event(
        "config",
        {
            "outer": {"api_key": "abc123", "value": 1},
            "tokens": ["abc"],
            "TOKEN": "xyz",
        },
    )
    payload = chat_context.get_chat_context()
    body = payload["events"][0]["payload"]
    assert body["outer"]["api_key"] == "<redacted>"
    assert body["outer"]["value"] == 1
    assert body["TOKEN"] == "<redacted>"


def test_value_with_home_path_redacted() -> None:
    """String values containing a /home/ path are scrubbed."""
    chat_context.record_event("fs", {"note": "wrote to /home/alice/x"})
    payload = chat_context.get_chat_context()
    assert payload["events"][0]["payload"]["note"] == "<redacted>"


def test_size_cap_drops_oldest_events() -> None:
    """Pushing 100 fat events still yields a dump <= 4 KB."""
    big = "x" * 500
    for i in range(100):
        chat_context.record_event("noise", {"i": i, "blob": big})

    payload = chat_context.get_chat_context()
    encoded = json.dumps(payload["events"])
    assert len(encoded.encode("utf-8")) <= 4096
    assert payload["count"] < 100


# -- Validation -------------------------------------------------------


def test_record_event_rejects_bad_inputs() -> None:
    with pytest.raises(TypeError):
        chat_context.record_event(123, {})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        chat_context.record_event("", {})
    with pytest.raises(TypeError):
        chat_context.record_event("cat", "not a mapping")  # type: ignore[arg-type]


def test_get_chat_context_rejects_bad_max_bytes() -> None:
    with pytest.raises(TypeError):
        chat_context.get_chat_context(max_bytes="big")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        chat_context.get_chat_context(max_bytes=0)


def test_format_context_section_returns_empty_for_no_events() -> None:
    assert chat_context.format_context_section({"events": []}) == ""
    assert chat_context.format_context_section({}) == ""


def test_chat_context_provider_round_trip() -> None:
    provider = chat_context.ChatContextProvider()
    provider.record("diag", {"k": 1})
    payload = provider.get()
    assert payload["count"] == 1


# -- Source-level integration assertion -------------------------------


def test_chat_ws_send_branch_calls_injection_helper() -> None:
    """Source-level guard: the WebSocket ``send`` handler invokes the helper.

    A direct WebSocket TestClient call would import ``src.api.routes`` which
    triggers an unrelated, pre-existing import bug in
    ``src/shared/python/config/__init__.py``. Until that is fixed in a
    separate change, this static check pins the wiring without spinning
    up the FastAPI app.
    """
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "src" / "api" / "routes" / "chat_ws.py").read_text()
    assert "_maybe_inject_chat_context(" in source
    assert (
        source.count("_maybe_inject_chat_context") >= 2
    ), "helper must be defined and at least one call-site must exist"
