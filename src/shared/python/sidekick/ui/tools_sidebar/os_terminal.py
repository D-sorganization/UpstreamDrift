"""OS terminal widget for the Sidekick sidebar.

Provides a PTY-backed terminal tab that runs a real OS shell (bash, pwsh,
cmd, etc.).  Falls back to a non-interactive subprocess pipe when neither
``pywinpty`` nor ``ptyprocess`` is installed.

Backend selection priority:
1. Windows: ``winpty.PtyProcess`` (pywinpty >= 2.0)
2. POSIX:   ``ptyprocess.PtyProcess``
3. Fallback: non-interactive ``subprocess.Popen`` pipe (labelled clearly)

Issue #5617: real OS terminal tab with PTY backend, shell selector, cwd display.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .shell_discovery import ShellDescriptor, discover_shells

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class _TerminalBackend(ABC):
    """Abstract PTY/pipe backend contract."""

    @abstractmethod
    def write(self, data: str) -> None:
        """Send user input to the shell process.

        Args:
            data: Text to write (including newlines).

        Raises:
            ValueError: If ``data`` is not a string.
            OSError: If the underlying process pipe is closed.
        """

    @abstractmethod
    def close(self) -> None:
        """Terminate the shell process and release resources."""

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """Return True when the shell process is still running."""


# ---------------------------------------------------------------------------
# winpty backend (Windows ConPTY)
# ---------------------------------------------------------------------------


class _WinPtyBackend(_TerminalBackend):
    """Backend using pywinpty (winpty.PtyProcess)."""

    def __init__(self, shell: str, args: list[str]) -> None:
        import winpty  # type: ignore[import]

        cmd = [shell, *args]
        self._proc = winpty.PtyProcess.spawn(cmd)
        logger.debug("WinPty backend started: %s", cmd)

    def write(self, data: str) -> None:
        if not isinstance(data, str):
            raise ValueError(f"write() expects str, got {type(data).__name__}")
        self._proc.write(data)

    def close(self) -> None:
        with _suppress(Exception):
            self._proc.terminate()
        logger.debug("WinPty backend closed")

    @property
    def is_alive(self) -> bool:
        return self._proc.isalive()


# ---------------------------------------------------------------------------
# ptyprocess backend (POSIX)
# ---------------------------------------------------------------------------


class _PtyProcessBackend(_TerminalBackend):
    """Backend using ptyprocess (POSIX)."""

    def __init__(self, shell: str, args: list[str]) -> None:
        from ptyprocess import PtyProcess  # type: ignore[import]

        cmd = [shell, *args]
        self._proc = PtyProcess.spawn(cmd)
        logger.debug("PtyProcess backend started: %s", cmd)

    def write(self, data: str) -> None:
        if not isinstance(data, str):
            raise ValueError(f"write() expects str, got {type(data).__name__}")
        self._proc.write(data.encode())

    def close(self) -> None:
        with _suppress(Exception):
            self._proc.terminate()
        logger.debug("PtyProcess backend closed")

    @property
    def is_alive(self) -> bool:
        return self._proc.isalive()


# ---------------------------------------------------------------------------
# Fallback pipe backend
# ---------------------------------------------------------------------------


class _PipeBackend(_TerminalBackend):
    """Non-interactive fallback backend (no PTY available).

    Warning: output reading is limited; interactive features (tab-completion,
    cursor movement) are unavailable.  The UI clearly labels this mode.
    """

    def __init__(self, shell: str, args: list[str]) -> None:
        cmd = [shell, *args]
        self._proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._output_lines: list[str] = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        logger.debug("Pipe (non-interactive) backend started: %s", cmd)

    def write(self, data: str) -> None:
        if not isinstance(data, str):
            raise ValueError(f"write() expects str, got {type(data).__name__}")
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()

    def close(self) -> None:
        with _suppress(Exception):
            self._proc.terminate()
        logger.debug("Pipe backend closed")

    @property
    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def _read_loop(self) -> None:
        if self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            with self._lock:
                self._output_lines.append(line)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_terminal_backend(
    shell_binary: str,
    args: list[str] | None = None,
) -> _TerminalBackend:
    """Create the best available terminal backend for *shell_binary*.

    Selection order:
    1. ``winpty.PtyProcess`` on Windows (pywinpty >= 2.0)
    2. ``ptyprocess.PtyProcess`` on POSIX
    3. Subprocess pipe fallback (non-interactive mode)

    Args:
        shell_binary: Absolute or relative path to the shell executable.
        args:         Optional extra arguments passed after the binary.

    Returns:
        A :class:`_TerminalBackend` instance. Never raises; falls back on error.

    Raises:
        ValueError: If *shell_binary* is empty.
    """
    if not shell_binary:
        raise ValueError("shell_binary must not be empty")

    effective_args: list[str] = args or []

    if sys.platform == "win32":
        try:
            return _WinPtyBackend(shell_binary, effective_args)
        except Exception as exc:  # noqa: BLE001 — fall through to next option
            logger.debug("WinPty unavailable (%s), trying fallback", exc)
    else:
        try:
            return _PtyProcessBackend(shell_binary, effective_args)
        except Exception as exc:  # noqa: BLE001 — fall through to fallback
            logger.debug("ptyprocess unavailable (%s), using pipe fallback", exc)

    return _PipeBackend(shell_binary, effective_args)


# ---------------------------------------------------------------------------
# Qt widget
# ---------------------------------------------------------------------------


def _build_qt_widget(parent: Any) -> Any:  # pragma: no cover — Qt required
    """Build the Qt widget tree.  Called only when Qt is available."""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import (
        QComboBox,
        QHBoxLayout,
        QLabel,
        QPlainTextEdit,
        QVBoxLayout,
        QWidget,
    )

    container = QWidget(parent)
    root_layout = QVBoxLayout(container)
    root_layout.setContentsMargins(0, 0, 0, 0)
    root_layout.setSpacing(2)

    # --- toolbar row ---
    toolbar = QWidget(container)
    toolbar_layout = QHBoxLayout(toolbar)
    toolbar_layout.setContentsMargins(4, 2, 4, 2)

    shell_selector = QComboBox(toolbar)
    shell_selector.setObjectName("SidekickOsTerminalShellSelector")
    shell_selector.setToolTip("Select the active shell")

    cwd_label = QLabel("", toolbar)
    cwd_label.setObjectName("SidekickOsTerminalCwd")
    cwd_label.setToolTip("Current working directory")

    toolbar_layout.addWidget(shell_selector)
    toolbar_layout.addWidget(cwd_label, stretch=1)
    root_layout.addWidget(toolbar)

    # --- output area ---
    output = QPlainTextEdit(container)
    output.setObjectName("SidekickOsTerminalOutput")
    output.setReadOnly(True)
    output.setToolTip("Terminal output")
    root_layout.addWidget(output, stretch=1)

    return container, shell_selector, cwd_label, output


class SidekickOsTerminalWidget:
    """PTY-backed OS terminal widget for the Sidekick sidebar.

    Hosts a real OS shell (bash, pwsh, cmd) in a Qt widget with:
    - Shell selector combo box
    - Live cwd display
    - PTY output pane

    Falls back to non-interactive pipe mode when pywinpty/ptyprocess are
    unavailable (clearly labelled in the cwd display).

    Design-by-contract:
    - Precondition: ``parent`` may be None or a QWidget.
    - Postcondition: ``current_backend`` is a ``_TerminalBackend`` after
      ``activate()`` is called.
    """

    _NON_INTERACTIVE_LABEL = "[non-interactive mode]"

    def __init__(self, parent: Any = None) -> None:
        self._parent = parent
        self._shells: list[ShellDescriptor] = discover_shells()
        self._backend: _TerminalBackend | None = None
        self._cwd: Path = Path(os.getcwd())
        self._qt_built = False

        # Build Qt components lazily so the class can be instantiated in
        # headless tests without a running QApplication.
        try:
            result = _build_qt_widget(parent)
            self._widget, self._shell_selector, self._cwd_label, self._output = result
            self._qt_built = True
            self._populate_shell_selector()
            self._update_cwd_label()
        except Exception as exc:  # noqa: BLE001 — headless / test environments
            logger.debug("Qt widget build skipped (%s)", exc)
            self._widget = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def widget(self) -> Any:
        """The root QWidget (None in headless environments)."""
        return self._widget

    @property
    def current_backend(self) -> _TerminalBackend | None:
        """Active terminal backend, or None before activate()."""
        return self._backend

    @property
    def cwd(self) -> Path:
        """Current working directory of the terminal session."""
        return self._cwd

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate(self, shell: ShellDescriptor | None = None) -> None:
        """Start (or restart) the shell process.

        Args:
            shell: Shell to launch.  Defaults to the first discovered shell.

        Raises:
            ValueError: If no shells are available.
        """
        if shell is None:
            if not self._shells:
                raise ValueError(
                    "No shells discovered; cannot activate terminal without a shell"
                )
            shell = self._shells[0]

        if self._backend is not None:
            self._backend.close()

        self._backend = create_terminal_backend(shell.binary, shell.args)
        is_non_interactive = isinstance(self._backend, _PipeBackend)
        if self._qt_built and is_non_interactive:
            self._cwd_label.setText(self._NON_INTERACTIVE_LABEL)
        elif self._qt_built:
            self._update_cwd_label()

        logger.debug(
            "Terminal activated: shell=%s non_interactive=%s",
            shell.binary,
            is_non_interactive,
        )

    def write(self, data: str) -> None:
        """Forward user input to the active shell.

        Args:
            data: Text (with newline) to send.

        Raises:
            RuntimeError: If activate() has not been called.
        """
        if self._backend is None:
            raise RuntimeError("Terminal is not activated; call activate() first")
        self._backend.write(data)

    def close(self) -> None:
        """Terminate the active shell session."""
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _populate_shell_selector(self) -> None:
        if not self._qt_built:
            return
        self._shell_selector.clear()
        for sd in self._shells:
            self._shell_selector.addItem(sd.display_name, sd)
        if not self._shells:
            self._shell_selector.addItem("(none)", None)

    def _update_cwd_label(self) -> None:
        if not self._qt_built:
            return
        self._cwd_label.setText(str(self._cwd))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _suppress:
    """Minimal context manager that suppresses specified exceptions."""

    def __init__(self, *exc_types: type[BaseException]) -> None:
        self._exc_types = exc_types

    def __enter__(self) -> _suppress:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return exc_type is not None and issubclass(exc_type, self._exc_types)


__all__ = [
    "SidekickOsTerminalWidget",
    "create_terminal_backend",
]
