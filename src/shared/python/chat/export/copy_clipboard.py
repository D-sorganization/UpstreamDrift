"""Per-message clipboard copy (Tools issue #2735).

The copier accepts any object implementing the
:class:`ClipboardWriterProtocol` so the contract is testable without
PyQt6. The :meth:`MessageClipboardCopier.from_qt_application` helper
returns a writer backed by :class:`QApplication.clipboard()` for the
real GUI path.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from typing import Literal, Protocol

from chat.service_base import ChatMessage

CopyMode = Literal["raw_text", "markdown", "code_only", "json"]

_FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


class ClipboardWriterProtocol(Protocol):
    """Minimal clipboard contract: ``set_text(text)``."""

    def set_text(self, text: str) -> None: ...


def _format_timestamp(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return str(ts)


def _render_markdown(message: ChatMessage) -> str:
    ts = _format_timestamp(message.timestamp)
    return f"## {message.role} - {ts}\n\n{message.content}\n"


def _extract_code_only(content: str) -> str:
    blocks = _FENCED_BLOCK.findall(content)
    return "\n\n".join(blocks)


def _render_json(message: ChatMessage) -> str:
    payload = asdict(message)
    return json.dumps(payload, default=str, indent=2)


class _QtClipboardAdapter:
    """Adapter that bridges :class:`ClipboardWriterProtocol` to PyQt6."""

    def __init__(self, qt_clipboard: object) -> None:
        self._clip = qt_clipboard

    def set_text(self, text: str) -> None:
        setter = getattr(self._clip, "setText", None)
        if setter is None:
            raise RuntimeError("Qt clipboard has no setText method")
        setter(text)


class MessageClipboardCopier:
    """Copy a :class:`ChatMessage` to a clipboard in one of several modes.

    The clipboard writer is injected at construction so the class is
    fully testable without instantiating a Qt application.
    """

    def __init__(self, clipboard_writer: ClipboardWriterProtocol) -> None:
        if clipboard_writer is None:
            raise ValueError("clipboard_writer is required")
        self._writer = clipboard_writer

    @classmethod
    def from_qt_application(cls) -> MessageClipboardCopier:
        """Build a copier backed by :class:`QApplication.clipboard()`.

        The PyQt6 import is deferred so non-GUI consumers (tests, REST)
        can import this module without pulling in Qt.
        """
        from PyQt6.QtWidgets import QApplication

        clip = QApplication.clipboard()
        if clip is None:
            raise RuntimeError("QApplication.clipboard() returned None")
        return cls(_QtClipboardAdapter(clip))

    def copy_message(self, message: ChatMessage, mode: CopyMode) -> str:
        """Copy ``message`` in ``mode``; return the text written.

        Pre:
            ``mode`` is one of ``raw_text | markdown | code_only | json``.
        Post:
            The clipboard writer has been called with the returned text.
        """
        if mode == "raw_text":
            text = message.content
        elif mode == "markdown":
            text = _render_markdown(message)
        elif mode == "code_only":
            text = _extract_code_only(message.content)
        elif mode == "json":
            text = _render_json(message)
        else:
            raise ValueError(
                "MessageClipboardCopier.copy_message: unknown mode "
                f"{mode!r}; expected one of raw_text, markdown, "
                "code_only, json"
            )
        self._writer.set_text(text)
        return text


__all__ = [
    "MessageClipboardCopier",
    "ClipboardWriterProtocol",
    "CopyMode",
]
