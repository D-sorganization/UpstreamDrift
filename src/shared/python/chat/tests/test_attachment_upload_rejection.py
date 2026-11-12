"""File-upload rejection wiring for chat dock (Tools #5727 slice 3.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest


def _load_messaging_module():
    try:
        import PyQt6.QtCore  # noqa: F401
        import PyQt6.QtWidgets  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    from chat._qt.messaging import (
        ATTACHMENT_PENDING_PREFIX,
        ATTACHMENT_REJECTED_PREFIX,
        MessagingMixin,
        compute_attachment_intake,
        file_upload_error_detail,
        format_attachment_deduplicated_bubble,
        format_attachment_pending_bubble,
    )

    return (
        ATTACHMENT_PENDING_PREFIX,
        ATTACHMENT_REJECTED_PREFIX,
        MessagingMixin,
        compute_attachment_intake,
        file_upload_error_detail,
        format_attachment_deduplicated_bubble,
        format_attachment_pending_bubble,
    )


class _FakeBubble:
    def __init__(self) -> None:
        self.content = ""

    def set_content(self, text: str) -> None:
        self.content = text


def _upload_host_class(messaging_mixin: type) -> type:
    class _UploadHost(messaging_mixin):
        """Minimal host object exercising MessagingMixin attachment paths."""

        def __init__(self) -> None:
            self._attachment_manifest_by_sha: dict[str, dict[str, Any]] = {}
            self._attachment_bubbles_pending_delivery: list[
                tuple[str, str | None, _FakeBubble]
            ] = []
            self.sent_ws: list[dict[str, Any]] = []
            self.bubbles: list[tuple[str, str]] = []

        def _send_ws(self, payload: dict) -> None:
            self.sent_ws.append(payload)

        def _add_bubble(self, role: str, content: str) -> _FakeBubble:
            bubble = _FakeBubble()
            bubble.content = content
            self.bubbles.append((role, content))
            return bubble

    return _UploadHost


@pytest.mark.unit
def test_file_upload_error_detail_recognizes_unknown_action() -> None:
    *_, file_upload_error_detail, __, ___ = _load_messaging_module()
    detail = "Unknown action: file_upload"
    assert file_upload_error_detail(detail) is True
    assert file_upload_error_detail("Something else failed") is False


@pytest.mark.unit
def test_apply_file_upload_rejection_updates_pending_bubble() -> None:
    (
        _,
        ATTACHMENT_REJECTED_PREFIX,
        MessagingMixin,
        __,
        ___,
        ____,
        format_attachment_pending_bubble,
    ) = _load_messaging_module()
    host = _upload_host_class(MessagingMixin)()
    bubble = _FakeBubble()
    bubble.content = format_attachment_pending_bubble("notes.txt")
    host._attachment_bubbles_pending_delivery.append(("notes.txt", "deadbeef", bubble))

    detail = "Unknown action: file_upload"
    assert host._apply_file_upload_rejection(detail) is True
    assert bubble.content.startswith(ATTACHMENT_REJECTED_PREFIX)
    assert detail in bubble.content
    assert host._attachment_bubbles_pending_delivery == []


@pytest.mark.unit
def test_apply_file_upload_rejection_drops_manifest_so_retry_is_not_deduped(
    tmp_path: Path,
) -> None:
    (
        ATTACHMENT_PENDING_PREFIX,
        _,
        MessagingMixin,
        compute_attachment_intake,
        ___,
        format_attachment_deduplicated_bubble,
        ____,
    ) = _load_messaging_module()
    host = _upload_host_class(MessagingMixin)()
    path = tmp_path / "retry-me.txt"
    path.write_bytes(b"same payload every time")

    entry, _ = compute_attachment_intake(path)
    host._ingest_attachment_path(path)

    assert entry.sha256 in host._attachment_manifest_by_sha
    pending_bubble = host._attachment_bubbles_pending_delivery[0][2]
    assert pending_bubble.content.startswith(ATTACHMENT_PENDING_PREFIX)

    detail = "Unknown action: file_upload"
    host._apply_file_upload_rejection(detail)

    assert entry.sha256 not in host._attachment_manifest_by_sha

    before_ws = len(host.sent_ws)
    before_bubbles = len(host.bubbles)
    _, is_dup = host._ingest_attachment_path(path)
    assert is_dup is False
    assert len(host.sent_ws) == before_ws + 1
    assert len(host.bubbles) == before_bubbles + 1
    assert not any(
        format_attachment_deduplicated_bubble(path.name, delivered=False) in text
        for _, text in host.bubbles[before_bubbles:]
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


@pytest.mark.unit
def test_on_message_error_branch_calls_file_upload_rejection() -> None:
    """Wiring contract: server error messages must reach _apply_file_upload_rejection."""
    dock_source = (
        _repo_root() / "src/shared/python/chat/_chat_dock_widget_qt.py"
    ).read_text(encoding="utf-8")
    error_idx = dock_source.index('elif msg_type == "error":')
    error_block = dock_source[error_idx : error_idx + 500]
    assert "_apply_file_upload_rejection" in error_block
    assert 'msg_type == "error"' in dock_source
