"""UpstreamDrift-only attachment manifest hook for pre-#5727 file uploads."""

from __future__ import annotations

from pathlib import Path

from chat._qt.messaging import AttachmentIntakeEntry, MAX_ATTACHMENT_BYTES
from chat._ud_attachment_manifest_handlers import (
    apply_file_upload_rejection,
    drop_attachment_manifest_sha,
    ingest_attachment_path,
)


class UdAttachmentManifestMixin:
    """Release dedup manifest entries when the backend rejects ``file_upload``."""

    def _drop_attachment_manifest_sha(self, sha256: str) -> None:
        drop_attachment_manifest_sha(self, sha256)

    def _apply_file_upload_rejection(self, detail: str) -> bool:
        return apply_file_upload_rejection(self, detail)

    def _ingest_attachment_path(
        self, path: Path, max_bytes: int = MAX_ATTACHMENT_BYTES
    ) -> tuple[AttachmentIntakeEntry, bool]:
        return ingest_attachment_path(self, path, max_bytes=max_bytes)


def install_ud_attachment_manifest_on_chat_dock() -> None:
    """Bind manifest-release behavior onto :class:`~chat.ChatDockWidget`."""
    from chat._chat_dock_widget_qt import ChatDockWidget

    for name in (
        "_drop_attachment_manifest_sha",
        "_apply_file_upload_rejection",
        "_ingest_attachment_path",
    ):
        setattr(ChatDockWidget, name, getattr(UdAttachmentManifestMixin, name))


install_ud_attachment_manifest_on_chat_dock()
