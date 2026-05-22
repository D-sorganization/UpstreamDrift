"""Tests for sidekick.agent.action_audit (epic #5967 / S6 / #5975)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sidekick.agent.action_audit import (
    JsonlActionAudit,
    MemoryActionAudit,
    redact_secrets,
)
from sidekick.agent.action_service import (
    ActionDescriptor,
    ActionResult,
    RecordedCall,
    SidekickActionService,
)
from datetime import datetime, timezone

UTC = timezone.utc  # noqa: UP017

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["password", "api_key", "secret", "token", "auth"])
def test_redact_secrets_masks_known_keys(key: str) -> None:
    out = redact_secrets({key: "shhh", "ok": 1})
    assert out[key] == "***"
    assert out["ok"] == 1


def test_redact_secrets_is_case_insensitive() -> None:
    out = redact_secrets({"API_KEY": "x", "Password": "y"})
    assert out["API_KEY"] == "***"
    assert out["Password"] == "***"


def test_redact_secrets_recurses_into_nested_dicts() -> None:
    out = redact_secrets({"outer": {"password": "p", "ok": 1}})
    assert out["outer"]["password"] == "***"
    assert out["outer"]["ok"] == 1


def test_redact_secrets_returns_plain_dict() -> None:
    out = redact_secrets({"a": 1})
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# MemoryActionAudit
# ---------------------------------------------------------------------------


def _make_call(action_id: str = "x.y", ok: bool = True) -> RecordedCall:
    return RecordedCall(
        timestamp=datetime.now(UTC),
        action_id=action_id,
        params={"value": 1},
        descriptor=None,
        result=ActionResult(ok=ok, error=None if ok else "fail"),
        dry_run=False,
    )


def test_memory_audit_records_calls_in_order() -> None:
    audit = MemoryActionAudit()
    audit(_make_call("a.first"))
    audit(_make_call("a.second"))
    assert [c.action_id for c in audit.records] == ["a.first", "a.second"]


def test_memory_audit_records_is_read_only_tuple() -> None:
    audit = MemoryActionAudit()
    audit(_make_call())
    assert isinstance(audit.records, tuple)


# ---------------------------------------------------------------------------
# JsonlActionAudit
# ---------------------------------------------------------------------------


def test_jsonl_audit_writes_one_line_per_call(tmp_path: Path) -> None:
    path = tmp_path / "actions.jsonl"
    audit = JsonlActionAudit(path=path)
    audit(_make_call("a.x"))
    audit(_make_call("a.y"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["action_id"] == "a.x"
    assert first["ok"] is True


def test_jsonl_audit_redacts_sensitive_params(tmp_path: Path) -> None:
    path = tmp_path / "actions.jsonl"
    audit = JsonlActionAudit(path=path)
    call = RecordedCall(
        timestamp=datetime.now(UTC),
        action_id="x.y",
        params={"username": "alice", "password": "hunter2"},
        descriptor=None,
        result=ActionResult(ok=True),
        dry_run=False,
    )
    audit(call)
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["params"]["password"] == "***"
    assert record["params"]["username"] == "alice"


def test_jsonl_audit_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "actions.jsonl"
    audit = JsonlActionAudit(path=path)
    audit(_make_call())
    assert path.exists()


def test_jsonl_audit_io_failure_degrades_to_memory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If file writes fail, the sink does not raise — it logs once and
    keeps a small in-memory tail so observability isn't lost."""
    path = tmp_path / "actions.jsonl"
    audit = JsonlActionAudit(path=path)
    # Make the directory read-only to provoke an OSError on write.
    path.parent.chmod(0o500)
    try:
        audit(_make_call("a.write_fail"))
    finally:
        path.parent.chmod(0o700)
    # The call still landed in the in-memory tail.
    assert any(c.action_id == "a.write_fail" for c in audit.tail)


def test_jsonl_audit_records_dry_run_flag(tmp_path: Path) -> None:
    path = tmp_path / "actions.jsonl"
    audit = JsonlActionAudit(path=path)
    call = RecordedCall(
        timestamp=datetime.now(UTC),
        action_id="x.y",
        params={},
        descriptor=None,
        result=ActionResult(ok=True),
        dry_run=True,
    )
    audit(call)
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["dry_run"] is True


def test_jsonl_audit_uses_real_descriptor_summary_when_present(
    tmp_path: Path,
) -> None:
    path = tmp_path / "actions.jsonl"
    audit = JsonlActionAudit(path=path)
    desc = ActionDescriptor(
        action_id="x.y",
        summary="A useful thing.",
        params_schema={"type": "object"},
        side_effects="read",
        reversible=False,
    )
    call = RecordedCall(
        timestamp=datetime.now(UTC),
        action_id="x.y",
        params={},
        descriptor=desc,
        result=ActionResult(ok=True),
        dry_run=False,
    )
    audit(call)
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["summary"] == "A useful thing."
    assert record["side_effects"] == "read"


# ---------------------------------------------------------------------------
# Integration with SidekickActionService
# ---------------------------------------------------------------------------


class _RecordingHandler:
    namespace = "t"

    def describe(self):  # type: ignore[no-untyped-def]
        return (
            ActionDescriptor(
                action_id="t.echo",
                summary="Echo",
                params_schema={"type": "object"},
                side_effects="read",
                reversible=False,
            ),
        )

    def invoke(self, action_id, params):  # type: ignore[no-untyped-def]
        return ActionResult(ok=True, value=params)


def test_service_wires_audit_sink(tmp_path: Path) -> None:
    audit = JsonlActionAudit(path=tmp_path / "a.jsonl")
    service = SidekickActionService(audit_sink=audit)
    service.register(_RecordingHandler())
    service.invoke("t.echo", {"x": 1})
    service.invoke("t.echo", {"x": 2})
    lines = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
