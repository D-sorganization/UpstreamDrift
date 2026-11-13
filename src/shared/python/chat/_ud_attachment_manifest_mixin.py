"""UpstreamDrift-only attachment manifest hook for pre-#5727 file uploads.

Manifest release on ``file_upload`` rejection lives on
:class:`~chat._qt.messaging.MessagingMixin`; this module remains imported at
launcher startup so older call sites keep working until Tools slice #5727 lands.
"""

from __future__ import annotations


def install_ud_attachment_manifest_on_chat_dock() -> None:
    """Retain import-time registration hook; behavior is on ``MessagingMixin``."""


install_ud_attachment_manifest_on_chat_dock()
