"""Preferences subpage: Terminal — default-shell selection.

Surfaces Tools PR #2882's ``discover_shells()`` so users can pick a
default shell for the new OS-terminal tab. The discovery call is
imported lazily and stubbed gracefully when Tools is unavailable —
this keeps the prefs page renderable in headless tests where the
vendored Tools shim isn't on ``PYTHONPATH``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = ["TerminalSection", "discover_shells_safe"]


def discover_shells_safe() -> list[dict[str, str]]:
    """Return list of shells discovered by Tools, or a stub list on failure.

    Each entry is a ``{"id": ..., "label": ..., "path": ...}`` mapping.
    The fallback list contains generic host shells (cmd / powershell /
    bash / zsh) so the UI is still usable when Tools is missing.
    """
    try:
        from sidekick.terminal.shells import discover_shells  # type: ignore[import-not-found]
    except ImportError:
        logger.debug(
            "Sidekick terminal package not available — using fallback shell list"
        )
        return _fallback_shells()

    try:
        shells = list(discover_shells())
    except Exception as exc:  # noqa: BLE001 — fallback must not crash prefs
        logger.warning("discover_shells() raised %r — using fallback list", exc)
        return _fallback_shells()

    # Normalise the canonical Tools schema to dicts so callers don't have
    # to import Tools types (LoD).
    normalised: list[dict[str, str]] = []
    for shell in shells:
        if isinstance(shell, dict):
            normalised.append(
                {
                    "id": str(shell.get("id", "")),
                    "label": str(shell.get("label", "")),
                    "path": str(shell.get("path", "")),
                }
            )
        else:
            normalised.append(
                {
                    "id": str(getattr(shell, "id", "")),
                    "label": str(getattr(shell, "label", "")),
                    "path": str(getattr(shell, "path", "")),
                }
            )
    return normalised


def _fallback_shells() -> list[dict[str, str]]:
    import platform

    if platform.system() == "Windows":
        return [
            {"id": "cmd", "label": "Command Prompt", "path": "cmd.exe"},
            {"id": "powershell", "label": "PowerShell", "path": "powershell.exe"},
        ]
    return [
        {"id": "bash", "label": "Bash", "path": "/bin/bash"},
        {"id": "zsh", "label": "Zsh", "path": "/bin/zsh"},
        {"id": "sh", "label": "POSIX sh", "path": "/bin/sh"},
    ]


class TerminalSection:
    """Container for the Terminal prefs widget.

    Implemented as a class (not a free function) so callers can fetch
    the current selection through a stable ``selected_shell_id``
    property — the saved preference value lives in the launcher's
    settings adapter, this object only renders the widget tree.
    """

    SECTION_ID = "terminal"
    SECTION_LABEL = "Terminal"

    def __init__(
        self,
        *,
        get_default: Callable[[], str] | None = None,
        set_default: Callable[[str], None] | None = None,
    ) -> None:
        self._get_default = get_default or (lambda: "")
        self._set_default = set_default or (lambda _v: None)
        self._combo: Any | None = None

    def build_widget(self) -> Any:
        """Return the QWidget for embedding into the preferences dialog."""
        from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

        from . import build_prefs_section

        widget = QWidget()
        row = QHBoxLayout(widget)
        row.addWidget(QLabel("Default shell:"))

        combo = QComboBox()
        for shell in discover_shells_safe():
            combo.addItem(shell["label"], shell["id"])
        # Restore persisted selection if any
        default = self._get_default()
        if default:
            idx = combo.findData(default)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.currentIndexChanged.connect(self._on_changed)
        row.addWidget(combo)
        row.addStretch(1)
        self._combo = combo

        return build_prefs_section(self.SECTION_ID, self.SECTION_LABEL, [widget])

    def _on_changed(self, _idx: int) -> None:
        if self._combo is None:
            return
        data = self._combo.currentData()
        if isinstance(data, str):
            self._set_default(data)

    @property
    def selected_shell_id(self) -> str:
        if self._combo is None:
            return ""
        data = self._combo.currentData()
        return str(data) if isinstance(data, str) else ""
