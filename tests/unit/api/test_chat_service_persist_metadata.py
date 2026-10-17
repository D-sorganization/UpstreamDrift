"""Mid-stream persistence must not see runtime-only state (#12872)."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _make_service(tmp_path) -> Any:
    with patch("src.api.services.chat_service.ChatService._load_adapter"):
        from src.api.services.chat_service import ChatService

        svc = ChatService()
        svc.PERSIST_DIR = tmp_path / "chat_sessions"
        svc._adapter = MagicMock()
        return svc


def _collect(svc: Any, session_id: str) -> list[Any]:
    async def _run() -> list[Any]:
        return [item async for item in svc.stream_response(session_id)]

    return asyncio.run(_run())


@pytest.mark.unit
def test_persist_mid_stream_succeeds_without_runtime_objects(tmp_path) -> None:
    svc = _make_service(tmp_path)
    ctx = svc.get_or_create_session(None)
    svc.add_user_message(ctx.session_id, "hi")
    seen: dict[str, Any] = {}

    def _stream(*args: Any, **kwargs: Any) -> Any:
        adapter_ctx = args[1] if len(args) > 1 else kwargs["context"]
        # The adapter still receives the cooperative cancel flag...
        seen["adapter_flag"] = adapter_ctx.metadata.get("stop_event")
        # ...and the session is persisted while the stream is in flight.
        svc._persist_session(ctx.session_id)
        yield SimpleNamespace(content="ok", tool_call_delta=None)

    svc._adapter.stream_response.side_effect = _stream
    assert _collect(svc, ctx.session_id) == ["ok"]

    assert isinstance(seen["adapter_flag"], threading.Event)
    assert "stop_event" not in ctx.metadata
    stored = json.loads((svc.PERSIST_DIR / f"{ctx.session_id}.json").read_text())
    assert "stop_event" not in json.dumps(stored)


@pytest.mark.unit
def test_persist_rejects_non_serialisable_metadata_naming_key(tmp_path) -> None:
    svc = _make_service(tmp_path)
    ctx = svc.get_or_create_session(None)
    ctx.metadata["bad_key"] = threading.Event()

    from src.api.services.chat_service import _assert_json_serialisable

    with pytest.raises(ValueError, match="bad_key"):
        _assert_json_serialisable(ctx.metadata)
    # _persist_session logs and skips rather than raising or writing garbage.
    svc._persist_session(ctx.session_id)
    assert not (svc.PERSIST_DIR / f"{ctx.session_id}.json").exists()
