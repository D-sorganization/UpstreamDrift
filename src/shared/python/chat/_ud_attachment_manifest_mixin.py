"""UpstreamDrift-only attachment manifest hook for pre-#5727 file uploads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chat._qt.messaging import (
    FILE_UPLOAD_BACKEND_DELIVERED,
    AttachmentIntakeEntry,
    MAX_ATTACHMENT_BYTES,
    MessagingMixin,
)
from chat._ud_attachment_manifest_handlers import (
    apply_file_upload_rejection,
    drop_attachment_manifest_sha,
    ingest_attachment_path,
)


class UdAttachmentManifestMixin:
    """Release dedup manifest entries when the backend rejects ``file_upload``."""

    def _track_pending_attachment_bubble(self, filename: str, bubble) -> None:
        """Track pending bubbles using the dock's pending-delivery list."""
        if FILE_UPLOAD_BACKEND_DELIVERED:
            return
        pending = getattr(self, "_attachment_bubbles_pending_delivery", None)
        if pending is None:
            pending = []
            self._attachment_bubbles_pending_delivery = pending
        pending.append((filename, bubble))

    def _drop_attachment_manifest_sha(self, sha256: str) -> None:
        drop_attachment_manifest_sha(self, sha256)

    def _apply_file_upload_rejection(self, detail: str) -> bool:
        return apply_file_upload_rejection(self, detail)

    def _ingest_attachment_path(
        self, path: Path, max_bytes: int = MAX_ATTACHMENT_BYTES
    ) -> tuple[AttachmentIntakeEntry, bool]:
        return ingest_attachment_path(self, path, max_bytes=max_bytes)


_MANIFEST_MIXIN_INSTALLED = False


def install_ud_attachment_manifest_on_chat_dock() -> None:
    """Bind manifest-release behavior onto :class:`~chat.ChatDockWidget`."""
    global _MANIFEST_MIXIN_INSTALLED
    if _MANIFEST_MIXIN_INSTALLED:
        return
    from chat._chat_dock_widget_qt import ChatDockWidget

    for name in (
        "_drop_attachment_manifest_sha",
        "_apply_file_upload_rejection",
        "_ingest_attachment_path",
    ):
        setattr(ChatDockWidget, name, getattr(UdAttachmentManifestMixin, name))
    _MANIFEST_MIXIN_INSTALLED = True


def _patch_chat_dock_lazy_load() -> None:
    """Hook the child-copy lazy loader without editing ``chat_dock_widget.py``."""
    from chat import chat_dock_widget as shell

    marker = "_ud_attachment_manifest_lazy_load_patched"
    original = shell._load_qt_module
    if getattr(original, marker, False):
        return

    def _load_qt_module() -> Any:
        qt_module = original()
        install_ud_attachment_manifest_on_chat_dock()
        return qt_module

    setattr(_load_qt_module, marker, True)
    shell._load_qt_module = _load_qt_module


_patch_chat_dock_lazy_load()
