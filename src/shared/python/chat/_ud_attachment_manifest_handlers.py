"""UD-only attachment manifest handlers for pre-#5727 file uploads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from chat._qt.bubbles import ChatMessageBubble
from chat._qt.messaging import (
    FILE_UPLOAD_BACKEND_DELIVERED,
    MAX_ATTACHMENT_BYTES,
    AttachmentIntakeEntry,
    MessagingMixin,
    file_upload_error_detail,
    format_attachment_rejection_bubble,
)


class AttachmentManifestHost(Protocol):
    """Minimal ChatDockWidget surface for attachment manifest overrides."""

    _attachment_manifest_by_sha: dict[str, dict[str, Any]]
    _attachment_bubbles_pending_delivery: list[
        tuple[str, str | None, ChatMessageBubble] | tuple[str, ChatMessageBubble]
    ]

    def _send_ws(self, payload: dict[str, Any]) -> None: ...

    def _add_bubble(self, role: str, content: str) -> ChatMessageBubble: ...


def drop_attachment_manifest_sha(host: AttachmentManifestHost, sha256: str) -> None:
    manifest = getattr(host, "_attachment_manifest_by_sha", None)
    if manifest is not None:
        manifest.pop(sha256, None)


def _pending_item_sha256(
    item: tuple[str, str | None, ChatMessageBubble] | tuple[str, ChatMessageBubble],
) -> tuple[str, str | None, ChatMessageBubble]:
    if len(item) == 2:
        filename, bubble = item
        return filename, None, bubble
    filename, sha256, bubble = item
    return filename, sha256, bubble


def apply_file_upload_rejection(host: AttachmentManifestHost, detail: str) -> bool:
    if FILE_UPLOAD_BACKEND_DELIVERED:
        return False
    if not file_upload_error_detail(detail):
        return False
    pending = getattr(host, "_attachment_bubbles_pending_delivery", None)
    if not pending:
        return False
    while pending:
        filename, sha256, bubble = _pending_item_sha256(pending.pop(0))
        bubble.set_content(format_attachment_rejection_bubble(filename, detail))
        if sha256:
            drop_attachment_manifest_sha(host, sha256)
    return True


def ingest_attachment_path(
    host: AttachmentManifestHost,
    path: Path,
    max_bytes: int = MAX_ATTACHMENT_BYTES,
) -> tuple[AttachmentIntakeEntry, bool]:
    entry, is_dup = MessagingMixin._ingest_attachment_path(host, path, max_bytes)
    if is_dup or FILE_UPLOAD_BACKEND_DELIVERED:
        return entry, is_dup
    pending = getattr(host, "_attachment_bubbles_pending_delivery", None) or []
    if not pending:
        return entry, is_dup
    last = pending[-1]
    if len(last) == 2 and last[0] == entry.filename:
        pending[-1] = (entry.filename, entry.sha256, last[1])
    return entry, is_dup
