"""Tests for the launcher-side HistoryServiceAdapter.

Verifies the adapter is a thin LOD-compliant translator: each public method
maps to exactly one downstream call on the injected sidekick conversation
service, with DbC preconditions on inputs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.launchers.chat_history.chat_history_service import (
    HistoryServiceAdapter,
)


@pytest.fixture
def fake_service() -> MagicMock:
    """Return a MagicMock standing in for sidekick.chat.conversation."""
    svc = MagicMock()
    svc.list.return_value = [
        {"id": "c1", "title": "First", "archived": False, "timestamp": ""},
    ]
    svc.search.return_value = [
        {"id": "c1", "title": "First", "archived": False, "timestamp": ""},
    ]
    svc.load_as_context.return_value = {
        "messages": [{"role": "user", "content": "hi"}],
        "session_id": "c1",
    }
    return svc


@pytest.fixture
def adapter(fake_service: MagicMock) -> HistoryServiceAdapter:
    return HistoryServiceAdapter(service=fake_service)


def test_list_active_calls_service_list_with_archived_false(
    adapter: HistoryServiceAdapter, fake_service: MagicMock
) -> None:
    adapter.list_active()
    fake_service.list.assert_called_once_with(archived=False)


def test_list_archived_calls_service_list_with_archived_true(
    adapter: HistoryServiceAdapter, fake_service: MagicMock
) -> None:
    adapter.list_archived()
    fake_service.list.assert_called_once_with(archived=True)


def test_search_calls_service_search_with_query(
    adapter: HistoryServiceAdapter, fake_service: MagicMock
) -> None:
    adapter.search("hello world")
    fake_service.search.assert_called_once_with("hello world")


def test_search_precondition_rejects_empty_query(
    adapter: HistoryServiceAdapter,
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        adapter.search("")


def test_search_precondition_rejects_whitespace_query(
    adapter: HistoryServiceAdapter,
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        adapter.search("   ")


def test_archive_calls_service_archive(
    adapter: HistoryServiceAdapter, fake_service: MagicMock
) -> None:
    adapter.archive("c1")
    fake_service.archive.assert_called_once_with("c1")


def test_unarchive_calls_service_unarchive(
    adapter: HistoryServiceAdapter, fake_service: MagicMock
) -> None:
    adapter.unarchive("c1")
    fake_service.unarchive.assert_called_once_with("c1")


def test_delete_calls_service_delete(
    adapter: HistoryServiceAdapter, fake_service: MagicMock
) -> None:
    adapter.delete("c1")
    fake_service.delete.assert_called_once_with("c1")


def test_export_calls_service_export(
    adapter: HistoryServiceAdapter, fake_service: MagicMock
) -> None:
    adapter.export("c1", "/tmp/out.json")
    fake_service.export.assert_called_once_with("c1", "/tmp/out.json")


def test_load_as_context_calls_service(
    adapter: HistoryServiceAdapter, fake_service: MagicMock
) -> None:
    result = adapter.load_as_context("c1")
    fake_service.load_as_context.assert_called_once_with("c1")
    assert result["session_id"] == "c1"


def test_load_as_context_raises_when_not_found(
    adapter: HistoryServiceAdapter, fake_service: MagicMock
) -> None:
    fake_service.load_as_context.return_value = None
    with pytest.raises(KeyError, match="not found"):
        adapter.load_as_context("nope")


def test_restore_is_alias_for_unarchive(
    adapter: HistoryServiceAdapter, fake_service: MagicMock
) -> None:
    adapter.restore("c1")
    fake_service.unarchive.assert_called_once_with("c1")


def test_condense_to_memory_uses_condenser_when_available(
    adapter: HistoryServiceAdapter, fake_service: MagicMock
) -> None:
    fake_service.condense_to_memory.return_value = {"summary": "ok"}
    result = adapter.condense_to_memory(["c1", "c2"])
    fake_service.condense_to_memory.assert_called_once_with(["c1", "c2"])
    assert result == {"summary": "ok"}


def test_condense_to_memory_falls_back_when_unavailable() -> None:
    svc = MagicMock(
        spec=[
            "list",
            "search",
            "archive",
            "unarchive",
            "delete",
            "export",
            "load_as_context",
        ]
    )
    adapter = HistoryServiceAdapter(service=svc)
    result = adapter.condense_to_memory(["c1"])
    assert result["status"] == "stub"
    assert "not available" in result["message"]


def test_load_memory_reads_user_memory_file(tmp_path) -> None:
    memory_file = tmp_path / "user_memory.json"
    memory_file.write_text(
        '{"identity": {"name": "Alice"}, "preferences": {}, '
        '"projects": {}, "knowledge": {}}',
        encoding="utf-8",
    )
    svc = MagicMock()
    adapter = HistoryServiceAdapter(service=svc, memory_path=memory_file)
    mem = adapter.load_memory()
    assert mem["identity"]["name"] == "Alice"


def test_load_memory_returns_empty_skeleton_when_missing(tmp_path) -> None:
    svc = MagicMock()
    adapter = HistoryServiceAdapter(service=svc, memory_path=tmp_path / "nope.json")
    mem = adapter.load_memory()
    assert mem == {
        "identity": {},
        "preferences": {},
        "projects": {},
        "knowledge": {},
    }


def test_save_memory_writes_user_memory_file(tmp_path) -> None:
    memory_file = tmp_path / "user_memory.json"
    svc = MagicMock()
    adapter = HistoryServiceAdapter(service=svc, memory_path=memory_file)
    adapter.save_memory(
        {
            "identity": {"name": "Bob"},
            "preferences": {},
            "projects": {},
            "knowledge": {},
        }
    )
    assert memory_file.exists()
    import json

    data = json.loads(memory_file.read_text(encoding="utf-8"))
    assert data["identity"]["name"] == "Bob"


def test_save_memory_precondition_rejects_non_dict(tmp_path) -> None:
    svc = MagicMock()
    adapter = HistoryServiceAdapter(service=svc, memory_path=tmp_path / "m.json")
    with pytest.raises(TypeError, match="dict"):
        adapter.save_memory("not a dict")  # type: ignore[arg-type]


def test_reset_memory_removes_file(tmp_path) -> None:
    memory_file = tmp_path / "user_memory.json"
    memory_file.write_text("{}", encoding="utf-8")
    svc = MagicMock()
    adapter = HistoryServiceAdapter(service=svc, memory_path=memory_file)
    adapter.reset_memory()
    assert not memory_file.exists()


def test_reset_memory_noop_when_missing(tmp_path) -> None:
    svc = MagicMock()
    adapter = HistoryServiceAdapter(service=svc, memory_path=tmp_path / "absent.json")
    # Should not raise.
    adapter.reset_memory()
