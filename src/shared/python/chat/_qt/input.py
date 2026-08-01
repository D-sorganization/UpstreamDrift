# ruff: noqa: E501
"""Input-widget helpers — Enter→submit keybinding installer.

Extracted from the monolithic ``_chat_dock_widget_qt`` module so the
keybinding logic is testable in isolation and the parent module fits in
the repo's 1500-line budget.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPlainTextEdit


def install_enter_submit(
    edit: QPlainTextEdit,
    on_submit: Callable[[], None],
) -> None:
    """Wire Enter→submit / Shift+Enter→newline on a ``QPlainTextEdit``.

    Implemented by monkey-patching ``keyPressEvent`` on the instance so we
    don't need a subclass (subclassed Python wrappers around Qt widgets
    can be reaped by the C++ side, leaving stale references — see Tools
    chat input keybindings tests). The original handler is preserved and
    called for non-submit keystrokes.

    DRY: every submit (button click + Enter key) funnels through the same
    ``on_submit`` callable so validation/queue logic lives in one place.

    DbC:
        Pre: ``edit`` is a non-None ``QPlainTextEdit``.
        Pre: ``on_submit`` is callable with zero args.
    """
    if edit is None:
        raise ValueError("install_enter_submit: edit must be provided")
    if not callable(on_submit):
        raise ValueError("install_enter_submit: on_submit must be callable")

    original_handler = edit.keyPressEvent

    def handler(event: Any) -> None:
        if event is None:
            original_handler(event)
            return
        key = event.key()
        if key in (int(Qt.Key.Key_Return), int(Qt.Key.Key_Enter)):
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if not shift:
                on_submit()
                event.accept()
                return
        original_handler(event)

    # Bind the new handler onto the instance.
    cast(Any, edit).keyPressEvent = handler
