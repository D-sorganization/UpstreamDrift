"""Focused coverage for Sidekick action audit sinks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sidekick.agent.action_audit import (
    JsonlActionAudit,
    MemoryActionAudit,
    redact_secrets,
)
from sidekick.agent.action_service import ActionDescriptor, ActionResult, RecordedCall

pytestmark = pytest.mark.unit


def _call(**params: Any) -> RecordedCall:
    return RecordedCall(
        timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        action_id="test.echo",
        params=params,
        descriptor=ActionDescriptor(
            action_id="test.echo",
            summary="Echo a value.",
            params_schema={"type": "object", "properties": {}},
            side_effects="read",
            reversible=False,
        ),
        result=ActionResult(ok=True, value={"ok": True}, undo_token="undo-1"),
        dry_run=False,
    )


def test_redact_secrets_masks_nested_sensitive_keys_without_mutating_input() -> None:
    payload = {"Token": "abc", "nested": {"password": "pw", "visible": 1}}

    redacted = redact_secrets(payload)

    assert redacted == {"Token": "***", "nested": {"password": "***", "visible": 1}}
    assert payload["Token"] == "abc"
    assert payload["nested"]["password"] == "pw"  # type: ignore[index]


def test_memory_audit_records_immutable_snapshot_view() -> None:
    audit = MemoryActionAudit()
    call = _call(value=1)

    audit(call)

    assert audit.records == (call,)
    assert isinstance(audit.records, tuple)


def test_jsonl_audit_writes_redacted_json_line(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "audit.jsonl"
    audit = JsonlActionAudit(path=path)

    audit(_call(api_key="secret", nested={"credential": "hidden"}))

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["action_id"] == "test.echo"
    assert record["summary"] == "Echo a value."
    assert record["params"] == {"api_key": "***", "nested": {"credential": "***"}}
    assert record["ok"] is True
    assert audit.tail[-1].action_id == "test.echo"


def test_jsonl_audit_degrades_to_tail_when_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audit = JsonlActionAudit(path=tmp_path / "audit.jsonl")

    def fail_open(*args: object, **kwargs: object) -> object:
        raise OSError("readonly")

    monkeypatch.setattr(Path, "open", fail_open)
    audit(_call(value=2))
    audit(_call(value=3))

    assert [call.params["value"] for call in audit.tail] == [2, 3]
