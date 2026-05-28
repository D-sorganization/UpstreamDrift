"""TDD coverage for the Ollama-side latency fixes (profiler 2026-05-26).

These tests pin the *shape* of the optimisations so a future refactor can't
silently regress the first-chunk latency back to 70+ seconds. They do not
require a live Ollama instance — every test patches the httpx layer.

Invariants under test
---------------------
1. **Tool declarations are cached** at ``ChatService.__init__`` and the
   per-send worker reuses the cached list instead of re-walking the
   registry on every message.
2. **Ollama streaming POST sends ``keep_alive``** to prevent the model
   from unloading between idle periods (otherwise every coffee break
   costs a 3-5 s cold-load on the next message).
3. **Ollama streaming POST caps ``num_ctx``** to 4096 so prompt-eval
   doesn't grow with the default 8192-token KV cache that most
   llama3.1 manifests ship with.
4. **Tools are sent via Ollama's native ``tools`` field** instead of
   only being stuffed into the system prompt as text — this drops
   ~700 tokens off the prompt-eval budget on every message.
5. **The Ollama tool-declaration converter** matches the OpenAI
   function-calling wire format that llama3.1+ understands.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Invariant 5 — Tool-declaration converter wire format
# ---------------------------------------------------------------------------


def test_tool_declarations_to_ollama_handles_none() -> None:
    from src.shared.python.ai.adapters.ollama_adapter import (
        _tool_declarations_to_ollama,
    )

    assert _tool_declarations_to_ollama(None) == []


def test_tool_declarations_to_ollama_handles_empty() -> None:
    from src.shared.python.ai.adapters.ollama_adapter import (
        _tool_declarations_to_ollama,
    )

    assert _tool_declarations_to_ollama([]) == []


def test_tool_declarations_to_ollama_emits_openai_function_shape() -> None:
    from src.shared.python.ai.adapters.base import ToolDeclaration
    from src.shared.python.ai.adapters.ollama_adapter import (
        _tool_declarations_to_ollama,
    )

    td = ToolDeclaration(
        name="get_weather",
        description="Look up weather for a location",
        parameters={"location": {"type": "string"}},
        required=["location"],
    )

    result = _tool_declarations_to_ollama([td])

    assert result == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]


def test_tool_declarations_to_ollama_copies_outer_containers() -> None:
    """Outer dict/list copies are made so callers can rebuild safely.

    Inner schema dicts are intentionally *not* deep-copied — that would
    cost real CPU on the hot path with 28 tools, and the conversion
    output is only consumed by ``json.dumps`` immediately afterwards.
    """
    from src.shared.python.ai.adapters.base import ToolDeclaration
    from src.shared.python.ai.adapters.ollama_adapter import (
        _tool_declarations_to_ollama,
    )

    params = {"x": {"type": "number"}}
    req = ["x"]
    td = ToolDeclaration(name="t", description="d", parameters=params, required=req)

    out = _tool_declarations_to_ollama([td])

    # Mutating the outer ``required`` list on the result must not bleed
    # into the input ToolDeclaration.
    out[0]["function"]["parameters"]["required"].append("MUTATED")
    assert req == ["x"], "input required list was mutated by output mutation"

    # Same for outer properties dict — adding a key on the output must
    # not appear in the input.
    out[0]["function"]["parameters"]["properties"]["new"] = {"type": "string"}
    assert "new" not in params, "input properties dict was mutated"


# ---------------------------------------------------------------------------
# Invariant 1 — Tool declarations cached at __init__
# ---------------------------------------------------------------------------


def test_chat_service_caches_tool_declarations_at_init() -> None:
    from src.api.services.chat_service import ChatService

    svc = ChatService()
    assert hasattr(svc, "_tool_declarations"), (
        "ChatService must expose a cached _tool_declarations list"
    )
    assert isinstance(svc._tool_declarations, list)
    # Don't pin a specific count — that's brittle when new tools land —
    # but require *some* tools given register_golf_suite_tools registers
    # ~10+ in every supported configuration.
    assert len(svc._tool_declarations) > 0, (
        "_tool_declarations must be populated from the tool registry"
    )


def test_chat_service_tool_declarations_are_well_formed() -> None:
    from src.api.services.chat_service import ChatService

    svc = ChatService()
    for td in svc._tool_declarations:
        assert isinstance(td.name, str) and td.name
        assert isinstance(td.description, str)
        assert isinstance(td.parameters, dict)


def test_build_tool_declarations_is_idempotent() -> None:
    """Calling the snapshot helper twice must produce equivalent lists."""
    from src.api.services.chat_service import ChatService

    svc = ChatService()
    snap1 = svc._build_tool_declarations()
    snap2 = svc._build_tool_declarations()

    assert len(snap1) == len(snap2)
    for a, b in zip(snap1, snap2, strict=True):
        assert a.name == b.name
        assert a.description == b.description


# ---------------------------------------------------------------------------
# Invariants 2, 3, 4 — Ollama POST body shape
# ---------------------------------------------------------------------------


class _StreamResponse:
    """Minimal context-manager double for ``httpx.Client.stream`` returns."""

    def __init__(self) -> None:
        self.raise_for_status = MagicMock()

    def __enter__(self) -> _StreamResponse:
        return self

    def __exit__(self, *_a: Any) -> None:
        pass

    def iter_lines(self) -> Any:
        # Yield a single ``done: true`` chunk so the generator terminates.
        import json as _json

        yield _json.dumps({"message": {"content": "ok"}, "done": True})


def _capture_post_body() -> tuple[Any, list[dict[str, Any]]]:
    """Patch the adapter's HTTP client and capture every POST body."""
    captured: list[dict[str, Any]] = []

    fake_client = MagicMock()

    def _stream(method: str, url: str, **kwargs: Any) -> Any:
        captured.append({"method": method, "url": url, **kwargs})
        return _StreamResponse()

    fake_client.stream = _stream
    return fake_client, captured


def _drain(gen: Any) -> None:
    for _ in gen:
        pass


def test_ollama_stream_post_sets_keep_alive() -> None:
    from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
    from src.shared.python.ai.types import ConversationContext

    fake_client, captured = _capture_post_body()
    adapter = OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b")

    with patch.object(adapter, "_get_client", return_value=fake_client):
        _drain(adapter.stream_response("hi", ConversationContext(), []))

    assert captured, "no POST was issued"
    body = captured[0]["json"]
    assert body.get("keep_alive") == "30m", (
        f"expected keep_alive=30m, got {body.get('keep_alive')!r}"
    )


def test_ollama_stream_post_sets_num_ctx_in_options() -> None:
    from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
    from src.shared.python.ai.types import ConversationContext

    fake_client, captured = _capture_post_body()
    adapter = OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b")

    with patch.object(adapter, "_get_client", return_value=fake_client):
        _drain(adapter.stream_response("hi", ConversationContext(), []))

    body = captured[0]["json"]
    assert isinstance(body.get("options"), dict), (
        f"expected dict options, got {body.get('options')!r}"
    )
    assert body["options"].get("num_ctx") == 4096


def test_ollama_stream_post_includes_native_tools_when_supplied() -> None:
    from src.shared.python.ai.adapters.base import ToolDeclaration
    from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
    from src.shared.python.ai.types import ConversationContext

    fake_client, captured = _capture_post_body()
    adapter = OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b")
    tools = [
        ToolDeclaration(
            name="weather",
            description="d",
            parameters={"x": {"type": "string"}},
            required=["x"],
        )
    ]

    with patch.object(adapter, "_get_client", return_value=fake_client):
        _drain(adapter.stream_response("hi", ConversationContext(), tools))

    body = captured[0]["json"]
    assert "tools" in body, f"tools field missing from POST body: {body.keys()}"
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "weather"


def test_ollama_stream_post_omits_tools_when_empty() -> None:
    """No empty ``tools: []`` — Ollama interprets that as "tools mode off"."""
    from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
    from src.shared.python.ai.types import ConversationContext

    fake_client, captured = _capture_post_body()
    adapter = OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b")

    with patch.object(adapter, "_get_client", return_value=fake_client):
        _drain(adapter.stream_response("hi", ConversationContext(), []))

    body = captured[0]["json"]
    assert "tools" not in body, (
        "tools field must be omitted entirely when no tools are supplied"
    )


def test_ollama_stream_post_streaming_enabled() -> None:
    """Regression: ``stream: True`` must remain set."""
    from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
    from src.shared.python.ai.types import ConversationContext

    fake_client, captured = _capture_post_body()
    adapter = OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b")

    with patch.object(adapter, "_get_client", return_value=fake_client):
        _drain(adapter.stream_response("hi", ConversationContext(), []))

    body = captured[0]["json"]
    assert body.get("stream") is True
    assert body.get("model") == "llama3.1:8b"


# ---------------------------------------------------------------------------
# Regression — duplicate precondition check removed
# ---------------------------------------------------------------------------


def test_ollama_stream_response_rejects_none_message_exactly_once() -> None:
    from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter
    from src.shared.python.ai.types import ConversationContext

    adapter = OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b")

    with pytest.raises(ValueError, match="message must be provided"):
        list(adapter.stream_response(None, ConversationContext(), []))  # type: ignore[arg-type]
