"""Canonical receipt registration cannot monopolize the HTTP event loop."""

import asyncio
from io import BytesIO
import threading
from types import SimpleNamespace
from typing import Any

from fastapi import HTTPException, UploadFile
import pytest

from src.api.routes import necromatcher as routes
from tests.unit.workspace.test_necromatcher_source_scope import make_scope

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_event_loop_progresses_while_registration_is_blocked(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, scope = make_scope(tmp_path)
    raw = b'{ "review": "exact bytes" }\n'
    library = SimpleNamespace()
    entered, release = threading.Event(), threading.Event()
    event_loop_thread = threading.get_ident()
    calls: list[Any] = []

    def registration(*args: Any) -> Any:
        calls.append(args)
        entered.set()
        assert threading.get_ident() != event_loop_thread, (
            "Registration ran on event loop"
        )
        assert release.wait(1), "Controlled registration was not released"
        return scope

    monkeypatch.setattr(routes, "import_fit_source_scope_review", registration)
    upload = UploadFile(BytesIO(raw), filename="review.json")
    task = asyncio.create_task(
        routes.import_source_scope_review("parent", upload, library)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        heartbeat: list[str] = []
        asyncio.get_running_loop().call_soon(heartbeat.append, "responsive")
        await asyncio.sleep(0)
        assert heartbeat == ["responsive"]
        assert not task.done(), "Registration finished before the control release"
        assert calls == [(library, "parent", raw)]
        release.set()
        assert await asyncio.wait_for(task, 1) == scope.to_record()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await upload.close()


@pytest.mark.asyncio
async def test_background_registration_preserves_canonical_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: Any) -> Any:
        raise ValueError("Reviewed source PNG identity differs")

    monkeypatch.setattr(routes, "import_fit_source_scope_review", reject)
    upload = UploadFile(BytesIO(b"{}"), filename="review.json")
    try:
        with pytest.raises(HTTPException) as caught:
            await routes.import_source_scope_review("parent", upload, SimpleNamespace())
        assert caught.value.status_code == 422
        assert caught.value.detail == "Reviewed source PNG identity differs"
    finally:
        await upload.close()
