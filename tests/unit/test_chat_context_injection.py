"""Tests for the context-injection watermark in chat_ws.

Verifies that ``_maybe_inject_chat_context`` only re-injects app-state
context into the conversation when the payload has changed, preventing
near-duplicate system messages from accumulating on every "send" action.

Coverage:
    * Identical context is not injected twice.
    * Changed context is re-injected after the state changes.
    * A session without a ``metadata`` dict still gets injected (graceful
      degradation — watermark is skipped, not errored).
    * Hash helper rejects non-string input.
    * history action: the existing ``test_chat_ws_context.py`` suite
      already tests the injection path; this file focuses specifically
      on deduplication semantics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from src.shared.python.ai import chat_context


# ── Module loader ────────────────────────────────────────────────────


def _load_chat_ws() -> ModuleType:
    """Load ``src/api/routes/chat_ws.py`` without running the package init.

    The ``src.api.routes`` package ``__init__`` transitively imports a
    module with a pre-existing bug (see test_chat_ws_context.py for
    explanation); loading via ``spec_from_file_location`` sidesteps it.
    """
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "src" / "api" / "routes" / "chat_ws.py"
    spec = importlib.util.spec_from_file_location("_chat_ws_dedup_under_test", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load chat_ws spec from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_chat_ws = _load_chat_ws()


# ── Session doubles ──────────────────────────────────────────────────


class _SessionWithMetadata:
    """Minimal session double that records injected messages and exposes metadata."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []
        self.metadata: dict[str, Any] = {}

    def add_message(self, role: str, content: str) -> None:
        self.messages.append((role, content))


class _SessionWithoutMetadata:
    """Session double that has ``add_message`` but no ``metadata`` dict."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append((role, content))


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_buffer_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset ring buffer and default env before every test."""
    chat_context.reset_buffer()
    monkeypatch.delenv("UPSTREAMDRIFT_SIDEKICK_CONTEXT", raising=False)


# ── Deduplication tests ───────────────────────────────────────────────


def test_same_context_injected_only_once() -> None:
    """Calling ``_maybe_inject_chat_context`` twice with identical state injects once."""
    chat_context.record_event("engine", {"status": "ok", "name": "mujoco"})
    session = _SessionWithMetadata()

    first = _chat_ws._maybe_inject_chat_context(session)
    second = _chat_ws._maybe_inject_chat_context(session)

    assert first is not None, "first injection should return the section string"
    assert second is None, "second injection with same context should be skipped"
    assert len(session.messages) == 1, "only one system message should be added"


def test_changed_context_triggers_reinjection() -> None:
    """After the app state changes, context is re-injected on the next call."""
    chat_context.record_event("engine", {"status": "ok"})
    session = _SessionWithMetadata()

    first = _chat_ws._maybe_inject_chat_context(session)
    assert first is not None

    # Simulate new app state arriving in the buffer.
    chat_context.record_event("engine", {"status": "error", "code": 42})

    second = _chat_ws._maybe_inject_chat_context(session)

    assert second is not None, "changed context should be re-injected"
    assert first != second, "the two injections should carry different payloads"
    assert len(session.messages) == 2


def test_hash_stored_in_metadata_after_injection() -> None:
    """After injection the watermark hash is persisted in ``session.metadata``."""
    chat_context.record_event("diag", {"val": 1})
    session = _SessionWithMetadata()

    _chat_ws._maybe_inject_chat_context(session)

    key = _chat_ws._CONTEXT_HASH_KEY
    assert key in session.metadata
    stored_hash = session.metadata[key]
    assert isinstance(stored_hash, str)
    assert len(stored_hash) == 16  # truncated hex digest


def test_three_identical_sends_inject_exactly_once() -> None:
    """Simulate three consecutive sends without state change — one injection."""
    chat_context.record_event("swing", {"phase": "backswing"})
    session = _SessionWithMetadata()

    results = [_chat_ws._maybe_inject_chat_context(session) for _ in range(3)]

    injected = [r for r in results if r is not None]
    assert len(injected) == 1
    assert len(session.messages) == 1


def test_session_without_metadata_still_receives_injection() -> None:
    """Sessions without ``metadata`` get the injection; watermark is gracefully skipped."""
    chat_context.record_event("engine", {"name": "pinocchio"})
    session = _SessionWithoutMetadata()

    first = _chat_ws._maybe_inject_chat_context(session)
    # No metadata dict, so no watermark is stored; injection fires again.
    second = _chat_ws._maybe_inject_chat_context(session)

    assert first is not None
    assert second is not None
    assert len(session.messages) == 2


# ── Hash helper tests ─────────────────────────────────────────────────


def test_context_section_hash_stable() -> None:
    """Same input always produces the same truncated digest."""
    h1 = _chat_ws._context_section_hash("hello world")
    h2 = _chat_ws._context_section_hash("hello world")
    assert h1 == h2


def test_context_section_hash_differs_for_different_inputs() -> None:
    """Different inputs produce different digests."""
    h1 = _chat_ws._context_section_hash("aaa")
    h2 = _chat_ws._context_section_hash("bbb")
    assert h1 != h2


def test_context_section_hash_rejects_non_str() -> None:
    """Non-string input raises ``TypeError``."""
    with pytest.raises(TypeError):
        _chat_ws._context_section_hash(42)  # type: ignore[arg-type]


# ── Env-var gate still respected ─────────────────────────────────────


def test_env_var_off_skips_even_fresh_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``UPSTREAMDRIFT_SIDEKICK_CONTEXT=0`` prevents any injection."""
    monkeypatch.setenv("UPSTREAMDRIFT_SIDEKICK_CONTEXT", "0")
    chat_context.record_event("diag", {"k": "v"})
    session = _SessionWithMetadata()

    result = _chat_ws._maybe_inject_chat_context(session)

    assert result is None
    assert session.messages == []
