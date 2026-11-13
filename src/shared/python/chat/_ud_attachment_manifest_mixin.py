"""UpstreamDrift-only attachment manifest helpers for pre-#5727 file uploads."""

from __future__ import annotations

import base64
from dataclasses import asdict
from pathlib import Path
from chat._qt.bubbles import ChatMessageBubble
from chat._qt.messaging import (
    FILE_UPLOAD_BACKEND_DELIVERED,
    MAX_ATTACHMENT_BYTES,
    AttachmentIntakeEntry,
    compute_attachment_intake,
    file_upload_error_detail,
    format_attachment_accepted_bubble,
    format_attachment_deduplicated_bubble,
    format_attachment_pending_bubble,
    format_attachment_rejection_bubble,
)


class UdAttachmentManifestMixin:
    """Release dedup manifest entries when the backend rejects ``file_upload``."""

    def _pending_attachment_bubbles(
        self,
    ) -> list[tuple[str, str | None, ChatMessageBubble]]:
        pending = getattr(self, "_attachment_bubbles_pending_delivery", None)
        if pending is None:
            pending = []
            self._attachment_bubbles_pending_delivery = pending
        return pending

    def _track_pending_attachment_bubble(
        self,
        filename: str,
        bubble: ChatMessageBubble,
        *,
        sha256: str | None = None,
    ) -> None:
        if FILE_UPLOAD_BACKEND_DELIVERED:
            return
        self._pending_attachment_bubbles().append((filename, sha256, bubble))

    def _drop_attachment_manifest_sha(self, sha256: str) -> None:
        manifest = getattr(self, "_attachment_manifest_by_sha", None)
        if manifest is not None:
            manifest.pop(sha256, None)

    def _apply_file_upload_rejection(self, detail: str) -> bool:
        if FILE_UPLOAD_BACKEND_DELIVERED:
            return False
        if not file_upload_error_detail(detail):
            return False
        pending = self._pending_attachment_bubbles()
        if not pending:
            return False
        while pending:
            filename, sha256, bubble = pending.pop(0)
            bubble.set_content(format_attachment_rejection_bubble(filename, detail))
            if sha256:
                self._drop_attachment_manifest_sha(sha256)
        return True

    def _ingest_attachment_path(
        self, path: Path, max_bytes: int = MAX_ATTACHMENT_BYTES
    ) -> tuple[AttachmentIntakeEntry, bool]:
        entry, data = compute_attachment_intake(path, max_bytes=max_bytes)
        manifest = getattr(self, "_attachment_manifest_by_sha", None)
        if manifest is None:
            manifest = {}
            self._attachment_manifest_by_sha = manifest
        is_dup = entry.sha256 in manifest
        if not is_dup:
            manifest[entry.sha256] = asdict(entry)
            b64 = base64.b64encode(data).decode("ascii")
            self._send_ws(
                {
                    "action": "file_upload",
                    "filename": entry.filename,
                    "sha256": entry.sha256,
                    "content": b64,
                }
            )
            if FILE_UPLOAD_BACKEND_DELIVERED:
                accepted = format_attachment_accepted_bubble(entry.filename)
                self._add_bubble("user", accepted)
            else:
                bubble = self._add_bubble(
                    "user", format_attachment_pending_bubble(entry.filename)
                )
                self._track_pending_attachment_bubble(
                    entry.filename, bubble, sha256=entry.sha256
                )
        else:
            text = format_attachment_deduplicated_bubble(
                entry.filename, delivered=FILE_UPLOAD_BACKEND_DELIVERED
            )
            self._add_bubble("user", text)
        return entry, is_dup


def install_ud_attachment_manifest_on_chat_dock() -> None:
    """Bind manifest-release behavior onto :class:`~chat.ChatDockWidget`."""
    from chat._chat_dock_widget_qt import ChatDockWidget

    for name in (
        "_pending_attachment_bubbles",
        "_track_pending_attachment_bubble",
        "_drop_attachment_manifest_sha",
        "_apply_file_upload_rejection",
        "_ingest_attachment_path",
    ):
        setattr(ChatDockWidget, name, getattr(UdAttachmentManifestMixin, name))


install_ud_attachment_manifest_on_chat_dock()
