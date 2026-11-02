"""Real ``ChatSessionManager`` lifecycle and signal tests (#14295).

Ordering is covered in ``test_chat_session_manager.py``; these cases cover
save/load/archive/delete/list behaviour, signal effects, and controlled
faults against the real Qt manager with temporary storage (offscreen; the
manager itself is never mocked). Evidence class: source-tree.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6.QtCore")

from src.shared.python.ai.gui import session_manager as sm  # noqa: E402

pytestmark = pytest.mark.unit


class _Spy:
    def __init__(self, signal) -> None:  # noqa: ANN001 - bound pyqtSignal
        self.calls: list[tuple] = []
        signal.connect(lambda *args: self.calls.append(args))


@pytest.fixture
def manager(tmp_path):
    return sm.ChatSessionManager(tmp_path / "sessions")


def _ctx(session_id: str = "s1", text: str = "How is my backswing?"):
    ctx = sm.ConversationContext(session_id=session_id)
    ctx.add_user_message(text)
    ctx.add_assistant_message("Looks fine.")
    return ctx


def test_save_persists_title_and_emits_once(manager, tmp_path) -> None:
    updated = _Spy(manager.sessions_updated)
    manager.save_session(_ctx(text="A" * 40))
    assert (tmp_path / "sessions" / "s1.json").exists()
    assert len(updated.calls) == 1
    loaded = manager.load_session("s1", emit=False)
    assert loaded is not None
    assert loaded.metadata["title"] == "A" * 30 + "..."


def test_load_emits_loaded_context_and_respects_emit_flag(manager) -> None:
    manager.save_session(_ctx())
    loaded_spy = _Spy(manager.session_loaded)
    ctx = manager.load_session("s1")
    assert ctx is not None
    assert len(loaded_spy.calls) == 1
    assert loaded_spy.calls[0][0].session_id == "s1"
    assert [m.content for m in ctx.messages] == [
        "How is my backswing?",
        "Looks fine.",
    ]
    manager.load_session("s1", emit=False)
    assert len(loaded_spy.calls) == 1


def test_missing_session_returns_none_without_signals(manager) -> None:
    loaded_spy = _Spy(manager.session_loaded)
    updated = _Spy(manager.sessions_updated)
    assert manager.load_session("ghost") is None
    assert manager.archive_session("ghost") is False
    assert manager.delete_session("ghost") is False
    with pytest.raises(KeyError):
        manager.unarchive_session("ghost")
    assert loaded_spy.calls == []
    assert updated.calls == []


def test_archive_roundtrip_is_persisted_and_listed(manager) -> None:
    manager.save_session(_ctx())
    assert manager.archive_session("s1") is True
    assert manager.is_archived("s1") is True
    assert manager.list_sessions()[0]["archived"] is True
    assert manager.search_sessions("backswing", include_archived=False) == []
    manager.unarchive_session("s1")
    # A fresh manager over the same storage sees the persisted flag.
    fresh = sm.ChatSessionManager(manager._storage_dir)
    assert fresh.is_archived("s1") is False


def test_delete_removes_file_and_listing_and_emits(manager) -> None:
    manager.save_session(_ctx("s1"))
    manager.save_session(_ctx("s2"))
    updated = _Spy(manager.sessions_updated)
    assert manager.delete_session("s1") is True
    assert len(updated.calls) == 1
    assert [s["id"] for s in manager.list_sessions()] == ["s2"]
    with pytest.raises(KeyError):
        manager.export_session("s1", "json")


def test_corrupt_session_file_is_skipped_not_fatal(manager) -> None:
    manager.save_session(_ctx("good"))
    (manager._storage_dir / "bad.json").write_text("{not json", encoding="utf-8")
    assert [s["id"] for s in manager.list_sessions()] == ["good"]
    loaded_spy = _Spy(manager.session_loaded)
    assert manager.load_session("bad") is None
    assert loaded_spy.calls == []


def test_failed_save_does_not_emit_sessions_updated(manager, monkeypatch) -> None:
    def _disk_full(self, path) -> None:  # noqa: ANN001
        raise OSError("No space left on device")

    monkeypatch.setattr(sm.ConversationContext, "save_to_file", _disk_full)
    updated = _Spy(manager.sessions_updated)
    manager.save_session(_ctx())
    assert updated.calls == []
    assert manager.list_sessions() == []
