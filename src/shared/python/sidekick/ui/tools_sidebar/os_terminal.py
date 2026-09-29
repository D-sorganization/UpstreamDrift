# mypy: warn-unused-ignores=False
"""Sidekick OS-level terminal widget (UpstreamDrift #5617).

This module provides a PTY-backed terminal surface for the Sidekick
sidebar. Unlike the renamed Python REPL widget, this tab launches a real
interactive shell (bash, zsh, pwsh, powershell, cmd, or a WSL distro) and
streams its output back into a read-only display.

Components:

* :class:`TerminalBackend` — Protocol every backend implements.
* :class:`PosixPtyBackend` — uses :mod:`ptyprocess` on POSIX.
* :class:`WindowsPtyBackend` — uses :mod:`winpty` (``pywinpty``) on Windows.
* :class:`SubprocessFallbackBackend` — non-interactive
  :mod:`subprocess` pipe fallback when no PTY library is installed.
* :func:`select_backend` — runtime backend factory.
* :class:`SidekickOsTerminalWidget` — Qt widget wiring it all together.

The widget never reaches into backend internals (LOD): it goes through the
``read``/``write``/``terminate``/``resize`` methods exclusively.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .appearance import (
    DEFAULT_DARK_PANEL_APPEARANCE,
    PanelAppearance,
    is_panel_appearance,
    panel_qss,
)
from .qt_compat import QtCore, QtWidgets, Signal
from .shell_discovery import ShellDescriptor, discover_shells

_logger = logging.getLogger(__name__)

SIDEKICK_OS_TERMINAL_OBJECT_NAME = "SidekickOsTerminalTab"

# ANSI / OSC parsing constants
#
# OSC 7 sequence shape: ``ESC ] 7 ; file://<host>/<path> BEL`` (or ESC \\).
# The hostname is empty or a token without slashes. We accept three shapes:
#   * ``file:///abs/path`` (POSIX, no host)
#   * ``file://host/abs/path`` (POSIX with host)
#   * ``file://host/C:/path`` and ``file://hostC:/path`` (Windows drive letter,
#     emitted both with and without a separator before the drive — be lenient).
_OSC7_PATTERN = re.compile(
    rb"\x1b\]7;file://[^/\x07\x1b]*?"
    rb"(?P<path>(?:/[A-Za-z]:/|[A-Za-z]:/|/)[^\x07\x1b]*)"
    rb"(?:\x07|\x1b\\)"
)
_ANSI_ESCAPE_RE = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]")
_OSC_ESCAPE_RE = re.compile(rb"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_READER_JOIN_TIMEOUT_SECONDS = 2.0


def _join_reader_thread(reader: threading.Thread | None) -> bool:
    """Join a backend reader thread without allowing teardown to hang."""
    if reader is None or reader is threading.current_thread():
        return True
    reader.join(timeout=_READER_JOIN_TIMEOUT_SECONDS)
    return not reader.is_alive()


def _close_stream(stream: object | None) -> None:
    """Best-effort close for subprocess pipes and PTY-like handles."""
    if stream is None:
        return
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:  # noqa: BLE001 - close is best-effort
            _logger.debug("Terminal stream close failed: %s", exc)


# ---------------------------------------------------------------------------
# Backend protocol + concrete backends
# ---------------------------------------------------------------------------


class TerminalBackend(Protocol):
    """Minimum API every terminal backend must expose."""

    is_running: bool

    def start(self) -> None: ...

    def write(self, data: bytes) -> None: ...

    def read(self, timeout: float = 0.0) -> bytes: ...

    def terminate(self) -> None: ...

    def resize(self, rows: int, cols: int) -> None: ...


@dataclass
class _BackendBase:
    """Shared backend state (command + cwd + running flag)."""

    command: tuple[str, ...]
    cwd: Path | None
    is_running: bool = False

    def _ensure_running(self) -> None:
        if not self.is_running:
            raise RuntimeError("backend is not running; call start() first")

    def _check_writeable(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if not data:
            raise ValueError("data must be non-empty")


class PosixPtyBackend(_BackendBase):
    """PTY backend backed by :mod:`ptyprocess` (POSIX only)."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Path | None,
    ) -> None:
        super().__init__(command=tuple(command), cwd=cwd)
        self._proc: object | None = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        from ptyprocess import PtyProcess  # type: ignore[import-not-found]

        cwd_value = str(self.cwd) if self.cwd is not None else None
        self._proc = PtyProcess.spawn(
            list(self.command),
            cwd=cwd_value,
            env=os.environ.copy(),
        )
        self.is_running = True
        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader.start()

    def _pump_stdout(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            while self.is_running:
                try:
                    chunk = proc.read(4096)  # type: ignore[attr-defined]
                    if not chunk:
                        continue
                    with self._lock:
                        self._buffer.extend(chunk)
                except EOFError:
                    break
        except Exception as exc:  # noqa: BLE001
            _logger.debug("PosixPtyBackend read thread error: %s", exc)
        finally:
            self.is_running = False

    def write(self, data: bytes) -> None:
        self._check_writeable(data)
        self._ensure_running()
        proc = self._proc
        assert proc is not None  # noqa: S101 - guarded by _ensure_running
        proc.write(data)  # type: ignore[attr-defined]

    def read(self, timeout: float = 0.0) -> bytes:  # noqa: ARG002
        self._ensure_running()
        with self._lock:
            data = bytes(self._buffer)
            self._buffer.clear()
        return data

    def terminate(self) -> None:
        if self._proc is None:
            self.is_running = False
            return
        proc = self._proc
        try:
            proc.terminate(force=True)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - terminate is best-effort
            _logger.debug("PosixPtyBackend terminate failed: %s", exc)
        self.is_running = False
        if _join_reader_thread(self._reader):
            self._reader = None
        self._proc = None

    def resize(self, rows: int, cols: int) -> None:
        if not self.is_running or self._proc is None:
            return
        try:
            self._proc.setwinsize(rows, cols)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - resize is cosmetic
            _logger.debug("PosixPtyBackend resize failed: %s", exc)


class WindowsPtyBackend(_BackendBase):
    """ConPTY backend backed by :mod:`winpty` (``pywinpty>=2``)."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Path | None,
    ) -> None:
        super().__init__(command=tuple(command), cwd=cwd)
        self._proc: object | None = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        from winpty import PtyProcess  # type: ignore[import-not-found]

        cwd_value = str(self.cwd) if self.cwd is not None else None
        # pywinpty 2.x accepts the command as a single string.
        cmdline = subprocess.list2cmdline(list(self.command))
        self._proc = PtyProcess.spawn(cmdline, cwd=cwd_value)
        self.is_running = True
        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader.start()

    def _pump_stdout(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            while self.is_running:
                try:
                    chunk = proc.read(4096)  # type: ignore[attr-defined]
                    if not chunk:
                        continue
                    if isinstance(chunk, str):
                        chunk = chunk.encode("utf-8", errors="replace")
                    with self._lock:
                        self._buffer.extend(chunk)
                except EOFError:
                    break
        except Exception as exc:  # noqa: BLE001
            _logger.debug("WindowsPtyBackend read thread error: %s", exc)
        finally:
            self.is_running = False

    def write(self, data: bytes) -> None:
        self._check_writeable(data)
        self._ensure_running()
        proc = self._proc
        assert proc is not None  # noqa: S101 - guarded by _ensure_running
        # pywinpty's write accepts ``str`` only; decode best-effort.
        proc.write(data.decode("utf-8", errors="replace"))  # type: ignore[attr-defined]

    def read(self, timeout: float = 0.0) -> bytes:  # noqa: ARG002
        self._ensure_running()
        with self._lock:
            data = bytes(self._buffer)
            self._buffer.clear()
        return data

    def terminate(self) -> None:
        if self._proc is None:
            self.is_running = False
            return
        proc = self._proc
        try:
            proc.terminate()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - terminate is best-effort
            _logger.debug("WindowsPtyBackend terminate failed: %s", exc)
        self.is_running = False
        if _join_reader_thread(self._reader):
            self._reader = None
        self._proc = None

    def resize(self, rows: int, cols: int) -> None:
        if not self.is_running or self._proc is None:
            return
        try:
            self._proc.setwinsize(rows, cols)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - resize is cosmetic
            _logger.debug("WindowsPtyBackend resize failed: %s", exc)


class SubprocessFallbackBackend(_BackendBase):
    """Non-interactive fallback when no PTY library is installed.

    This backend wires shell stdin/stdout through plain pipes. Many
    interactive features (line editing, colored prompts, job control) are
    unavailable, but simple commands and a labelled help banner keep the
    tab usable.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Path | None,
    ) -> None:
        super().__init__(command=tuple(command), cwd=cwd)
        self._proc: subprocess.Popen[bytes] | None = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        cwd_value = str(self.cwd) if self.cwd is not None else None
        self._proc = subprocess.Popen(  # noqa: S603 - command is caller-controlled
            list(self.command),
            cwd=cwd_value,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        self.is_running = True
        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader.start()

    def _pump_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                chunk = proc.stdout.read(1024)
                if not chunk:
                    break
                with self._lock:
                    self._buffer.extend(chunk)
        except Exception as exc:  # noqa: BLE001 - reader thread guards
            _logger.debug("SubprocessFallbackBackend read thread error: %s", exc)
        finally:
            self.is_running = False

    def write(self, data: bytes) -> None:
        self._check_writeable(data)
        self._ensure_running()
        proc = self._proc
        assert proc is not None and proc.stdin is not None  # noqa: S101
        proc.stdin.write(data)
        proc.stdin.flush()

    def read(self, timeout: float = 0.0) -> bytes:
        self._ensure_running()
        if timeout > 0:
            event = threading.Event()
            event.wait(timeout)
        with self._lock:
            data = bytes(self._buffer)
            self._buffer.clear()
        return data

    def terminate(self) -> None:
        proc = self._proc
        if proc is None:
            self.is_running = False
            return
        self.is_running = False
        try:
            _close_stream(proc.stdin)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception as exc:  # noqa: BLE001 - terminate is best-effort
            _logger.debug("SubprocessFallbackBackend terminate failed: %s", exc)
        finally:
            _close_stream(proc.stdout)
            if _join_reader_thread(self._reader):
                self._reader = None
            self._proc = None

    def resize(self, rows: int, cols: int) -> None:  # noqa: ARG002 - no-op for pipes
        return None


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def select_backend(
    *,
    command: Sequence[str],
    cwd: Path | None,
) -> tuple[TerminalBackend | None, str | None]:
    """Return the best available backend for the current host.

    Returns a ``(backend, hint)`` tuple. ``backend`` is ``None`` when no
    backend could be instantiated; in that case ``hint`` carries a human
    readable installation hint to surface in the UI. On success ``hint``
    may be a non-fatal note (e.g. "running without PTY").
    """
    if not command:
        raise ValueError("command must be non-empty")

    system = platform.system()
    if system in {"Linux", "Darwin"}:
        try:
            import ptyprocess  # type: ignore[import-not-found]  # noqa: F401

            return PosixPtyBackend(command=command, cwd=cwd), None
        except ImportError:
            _logger.debug("ptyprocess unavailable; falling back to subprocess pipes.")
    elif system == "Windows":
        try:
            import winpty  # type: ignore[import-not-found]  # noqa: F401

            return WindowsPtyBackend(command=command, cwd=cwd), None
        except ImportError:
            _logger.debug("pywinpty unavailable; falling back to subprocess pipes.")

    try:
        return (
            SubprocessFallbackBackend(command=command, cwd=cwd),
            _install_hint(system),
        )
    except Exception as exc:  # noqa: BLE001 - bottom of the stack
        _logger.debug("SubprocessFallbackBackend construction failed: %s", exc)
        return None, _install_hint(system)


def _install_hint(system: str) -> str:
    """Return the install hint for the missing PTY backend on ``system``."""
    if system == "Windows":
        return (
            "Interactive terminal features require pywinpty. "
            "Install with: pip install 'upstream-drift-tools[terminal]' "
            "(or directly: pip install 'pywinpty>=2.0')."
        )
    return (
        "Interactive terminal features require ptyprocess. "
        "Install with: pip install 'upstream-drift-tools[terminal]' "
        "(or directly: pip install ptyprocess)."
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def extract_osc7_cwd(data: bytes) -> Path | None:
    """Return the last OSC 7 cwd path emitted in ``data``, or ``None``.

    OSC 7 is the de-facto sequence shells use to advertise their cwd:
    ``ESC ] 7 ; file://<host><path> BEL`` (or terminated by ``ESC \\``).
    """
    last: bytes | None = None
    for match in _OSC7_PATTERN.finditer(data):
        last = match.group("path")
    if last is None:
        return None
    try:
        text = last.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text:
        return None
    if (
        platform.system() == "Windows"
        and text.startswith("/")
        and len(text) > 2
        and text[2] == ":"
    ):
        text = text[1:]
    return Path(text)


def strip_ansi(data: bytes) -> bytes:
    """Remove the common ANSI CSI / OSC sequences from ``data``."""
    stripped = _OSC_ESCAPE_RE.sub(b"", data)
    return _ANSI_ESCAPE_RE.sub(b"", stripped)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


class ShellDiscoveryThread(QtCore.QThread):
    """Thread for running shell discovery asynchronously to avoid UI freezing."""

    discovered = Signal(list)

    def __init__(
        self,
        shells_override: Sequence[ShellDescriptor] | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.shells_override = shells_override

    def run(self) -> None:
        try:
            res = list(self.shells_override or discover_shells())
            self.discovered.emit(res)
        except Exception as e:
            _logger.error("Failed to discover shells asynchronously: %s", e)
            self.discovered.emit([])


class SidekickOsTerminalWidget(QtWidgets.QWidget):
    """OS-level terminal tab backed by a PTY (or pipe fallback)."""

    def __init__(
        self,
        *,
        project_root: Path,
        shells: Sequence[ShellDescriptor] | None = None,
        autostart: bool = True,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if project_root is None:
            raise ValueError("project_root must be provided")
        super().__init__(parent)
        self.setObjectName(SIDEKICK_OS_TERMINAL_OBJECT_NAME)

        self._project_root = Path(project_root)
        self._backend: TerminalBackend | None = None
        self._install_hint: str | None = None
        self._current_shell: ShellDescriptor | None = None
        self._autostart: bool = autostart
        self._appearance: PanelAppearance = DEFAULT_DARK_PANEL_APPEARANCE
        self._discovery_thread: ShellDiscoveryThread | None = None
        self._is_shutting_down = False
        self.destroyed.connect(self._on_destroyed)

        # F2: command history ring (newest-first; index walks backward through it)
        self._history: list[str] = []
        self._history_index: int = -1  # -1 = not navigating
        self._history_scratch: str = ""  # stash live text while navigating

        if shells is not None:
            self._shells = list(shells)
            self._build_ui()
            self._populate_shell_selector()
            self._on_cwd_changed(self._project_root)
            if autostart and self._shells:
                self._start_backend(self._shells[0])
        else:
            self._shells = []
            self._build_ui()
            self._populate_shell_selector()
            self._on_cwd_changed(self._project_root)

            # Async shell discovery prevents UI freezing on startup
            # (slow WSL discovery was the original symptom).
            self._discovery_thread = ShellDiscoveryThread(None, self)
            self._discovery_thread.discovered.connect(self._on_shells_discovered)
            self._discovery_thread.start()

    def _on_shells_discovered(self, shells: list[ShellDescriptor]) -> None:
        """Callback for when shells are discovered asynchronously."""
        self._shells = shells
        self._populate_shell_selector()
        if self._autostart and self._shells and self._backend is None:
            self._start_backend(self._shells[0])

    # -- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(6)

        self._cwd_label = QtWidgets.QLabel(self)
        self._cwd_label.setObjectName("SidekickOsTerminalCwd")
        self._cwd_label.setToolTip(
            "Live current working directory of the running shell."
        )
        top_row.addWidget(self._cwd_label, stretch=1)

        self._shell_selector = QtWidgets.QComboBox(self)
        self._shell_selector.setObjectName("SidekickOsTerminalShellSelector")
        self._shell_selector.setToolTip(
            "Switch the active shell (bash, zsh, pwsh, powershell, cmd, WSL)."
        )
        self._shell_selector.currentIndexChanged.connect(self._on_selector_changed)
        top_row.addWidget(self._shell_selector)

        layout.addLayout(top_row)

        self._install_hint_label = QtWidgets.QLabel("", self)
        self._install_hint_label.setObjectName("SidekickOsTerminalInstallHint")
        self._install_hint_label.setWordWrap(True)
        self._install_hint_label.setVisible(False)
        layout.addWidget(self._install_hint_label)

        self._output = QtWidgets.QPlainTextEdit(self)
        self._output.setObjectName("SidekickOsTerminalOutput")
        self._output.setReadOnly(True)
        self._output.setToolTip("Streamed stdout / stderr from the running shell.")
        layout.addWidget(self._output, stretch=4)

        # F2: control row — Ctrl+C interrupt and Stop/Restart buttons.
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setSpacing(4)

        self._ctrlc_button = QtWidgets.QPushButton("⌃C", self)
        self._ctrlc_button.setObjectName("SidekickOsTerminalCtrlC")
        self._ctrlc_button.setToolTip(
            "Send SIGINT (Ctrl+C) to interrupt the running command."
        )
        self._ctrlc_button.setFixedWidth(36)
        self._ctrlc_button.clicked.connect(self._send_interrupt)
        ctrl_row.addWidget(self._ctrlc_button)

        self._stop_button = QtWidgets.QPushButton("↺", self)
        self._stop_button.setObjectName("SidekickOsTerminalStop")
        self._stop_button.setToolTip("Stop and restart the current shell session.")
        self._stop_button.setFixedWidth(36)
        self._stop_button.clicked.connect(self._on_restart)
        ctrl_row.addWidget(self._stop_button)

        ctrl_row.addStretch(1)
        layout.addLayout(ctrl_row)

        self._input = QtWidgets.QLineEdit(self)
        self._input.setObjectName("SidekickOsTerminalInput")
        self._input.setPlaceholderText(
            "Type a command and press Enter (↑↓ for history)"
        )
        self._input.setToolTip(
            "Send a single line to the shell's stdin (Enter submits; "
            "↑/↓ navigate command history)."
        )
        self._input.returnPressed.connect(self._on_submit)
        # F2: install an event filter so Up/Down traverse command history.
        self._input.installEventFilter(self)
        layout.addWidget(self._input)

        # Poll the backend on a short timer so output streams into the view.
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(80)
        self._poll_timer.timeout.connect(self._poll_backend)

        # Apply a visible border + colours so the terminal surfaces read as
        # distinct panels rather than borderless white space.
        self.apply_appearance(self._appearance)

    def apply_appearance(self, appearance: PanelAppearance) -> None:
        """Apply user-adjustable colours/border to the terminal surfaces.

        Single-value handoff (LOD): renders a validated
        :class:`PanelAppearance` via the shared panel stylesheet.
        """
        if not is_panel_appearance(appearance):
            raise TypeError("appearance must be a PanelAppearance")
        self._appearance = appearance
        self.setStyleSheet(panel_qss(self.objectName(), appearance))

    def appearance(self) -> PanelAppearance:
        """Return the currently applied appearance."""
        return self._appearance

    def _populate_shell_selector(self) -> None:
        self._shell_selector.blockSignals(True)
        self._shell_selector.clear()
        for shell in self._shells:
            self._shell_selector.addItem(shell.label, userData=shell.identifier)
        self._shell_selector.blockSignals(False)

    # -- Backend lifecycle --------------------------------------------------

    def _start_backend(self, shell: ShellDescriptor) -> None:
        backend, hint = select_backend(command=shell.command, cwd=self._project_root)
        self._backend = backend
        self._install_hint = hint
        self._current_shell = shell

        if backend is None:
            self._install_hint_label.setText(
                hint or "Terminal backend unavailable; install pty support."
            )
            self._install_hint_label.setVisible(True)
            return

        if hint:
            self._install_hint_label.setText(hint)
            self._install_hint_label.setVisible(True)
        else:
            self._install_hint_label.setVisible(False)

        try:
            backend.start()
        except Exception as exc:  # noqa: BLE001 - degrade to install hint
            _logger.debug("OS terminal backend failed to start: %s", exc)
            self._backend = None
            self._install_hint_label.setText(
                f"Failed to start {shell.label}: {exc}. {self._install_hint or ''}"
            )
            self._install_hint_label.setVisible(True)
            return

        self._poll_timer.start()

    def switch_shell(self, identifier: str) -> None:
        """Stop the current backend and start the shell named ``identifier``.

        Raises:
            ValueError: If no discovered shell matches ``identifier``.
        """
        target = next(
            (shell for shell in self._shells if shell.identifier == identifier),
            None,
        )
        if target is None:
            raise ValueError(f"unknown shell identifier: {identifier!r}")
        self._teardown_backend()
        self._on_cwd_changed(self._project_root)
        self._start_backend(target)

    def _teardown_backend(self) -> None:
        self._poll_timer.stop()
        backend = self._backend
        self._backend = None
        if backend is None:
            return
        try:
            backend.terminate()
        except Exception as exc:  # noqa: BLE001 - terminate is best-effort
            _logger.debug("OS terminal teardown failed: %s", exc)

    def shutdown(self) -> None:
        """Stop terminal runtime resources owned by this widget."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        try:
            discovery_thread = self._discovery_thread
            self._discovery_thread = None
            if discovery_thread is not None and discovery_thread.isRunning():
                discovery_thread.quit()
                discovery_thread.wait(2000)
            self._teardown_backend()
        finally:
            self._is_shutting_down = False

    # -- IO routing ---------------------------------------------------------

    def _on_submit(self) -> None:
        line = self._input.text()
        self._input.clear()
        # F2: commit to history (skip blank lines and exact duplicates of last entry).
        if line.strip() and (not self._history or self._history[0] != line):
            self._history.insert(0, line)
        self._history_index = -1
        self._history_scratch = ""

        if self._backend is None:
            return
        # Always send a single newline to the PTY regardless of host os.linesep.
        # On Windows os.linesep == "\r\n"; sending both bytes to ConPTY produces
        # a CR *and* LF which the shell interprets as two submits (F1).
        payload = (line + "\n").encode("utf-8", errors="replace")
        if not payload:
            return
        try:
            self._backend.write(payload)
        except Exception as exc:  # noqa: BLE001 - shell may exit
            _logger.debug("OS terminal write failed: %s", exc)

    def _send_interrupt(self) -> None:
        """Send SIGINT (Ctrl+C / ETX) to the running shell process (F2)."""
        if self._backend is None or not self._backend.is_running:
            return
        try:
            self._backend.write(b"\x03")
        except Exception as exc:  # noqa: BLE001 - shell may exit
            _logger.debug("OS terminal interrupt failed: %s", exc)

    def _on_restart(self) -> None:
        """Stop the current shell and restart the same shell type (F2)."""
        shell = self._current_shell
        if shell is None and self._shells:
            shell = self._shells[0]
        if shell is None:
            return
        self._teardown_backend()
        self._on_cwd_changed(self._project_root)
        self._start_backend(shell)

    def eventFilter(  # noqa: N802 - Qt API
        self, obj: QtCore.QObject, event: QtCore.QEvent
    ) -> bool:
        """Intercept Up/Down arrow on the input field for history navigation (F2)."""
        from .qt_compat import QtCore as _QtCore

        key_type = getattr(_QtCore.QEvent, "KeyPress", None)
        if key_type is not None and event.type() == key_type and obj is self._input:
            key_event = event  # type: ignore[assignment]
            key = getattr(key_event, "key", None)
            if callable(key):
                pressed = key()
                qt = _QtCore.Qt
                up_key = getattr(qt, "Key_Up", None)
                down_key = getattr(qt, "Key_Down", None)

                if up_key is not None and pressed == up_key:
                    self._navigate_history(direction=1)
                    return True
                if down_key is not None and pressed == down_key:
                    self._navigate_history(direction=-1)
                    return True

        return bool(super().eventFilter(obj, event))

    def _navigate_history(self, *, direction: int) -> None:
        """Move through command history. direction=1 goes older, -1 goes newer."""
        if not self._history:
            return

        if self._history_index == -1 and direction == 1:
            # Starting to navigate backward — stash whatever is in the input.
            self._history_scratch = self._input.text()

        new_index = self._history_index + direction

        if new_index < 0:
            # Walked past the newest entry — restore the live scratch text.
            self._history_index = -1
            self._input.setText(self._history_scratch)
            return

        if new_index >= len(self._history):
            # Can't go older than the oldest entry.
            return

        self._history_index = new_index
        self._input.setText(self._history[self._history_index])

    def _poll_backend(self) -> None:
        if self._backend is None or not self._backend.is_running:
            self._poll_timer.stop()
            return
        try:
            data = self._backend.read(timeout=0.0)
        except Exception as exc:  # noqa: BLE001 - poll degrades gracefully
            _logger.debug("OS terminal read failed: %s", exc)
            return
        if data:
            self._handle_output(data)

    def _handle_output(self, data: bytes) -> None:
        """Receive bytes from the backend and route to the UI/cwd slot."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        cwd = extract_osc7_cwd(data)
        if cwd is not None:
            self._on_cwd_changed(cwd)
        text = strip_ansi(data).decode("utf-8", errors="replace")
        if text:
            # Append raw decoded text without stripping trailing newlines (F3).
            # Per-chunk rstrip collapses blank lines and joins a prompt onto the
            # previous line when chunks split at a token boundary.  Moving the
            # cursor to End + insertPlainText lets the widget handle newlines
            # natively and preserves inter-chunk blank lines.
            cursor = self._output.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(text)
            self._output.setTextCursor(cursor)
            self._output.ensureCursorVisible()

    def _on_cwd_changed(self, path: Path) -> None:
        """Single slot for cwd updates (LOD: one path, no chains)."""
        if path is None:
            raise ValueError("path must be provided")
        self._cwd_label.setText(f"cwd: {path}")

    # -- Qt selector callback ----------------------------------------------

    def _on_selector_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._shells):
            return
        target = self._shells[index]
        if self._current_shell is not None and target.identifier == (
            self._current_shell.identifier
        ):
            return
        self.switch_shell(target.identifier)

    # -- Qt lifecycle -------------------------------------------------------

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        """Tear down the backend when Qt closes the widget."""
        self.shutdown()
        super().closeEvent(event)  # type: ignore[misc]

    def _on_destroyed(self, _obj: object | None = None) -> None:
        """Tear down runtime resources when Qt destroys the widget."""
        self.shutdown()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except RuntimeError:
            # The wrapped Qt object may already be deleted during interpreter
            # shutdown; backend teardown is best-effort in that path.
            return


__all__ = [
    "PosixPtyBackend",
    "SIDEKICK_OS_TERMINAL_OBJECT_NAME",
    "SidekickOsTerminalWidget",
    "SubprocessFallbackBackend",
    "TerminalBackend",
    "WindowsPtyBackend",
    "extract_osc7_cwd",
    "select_backend",
    "strip_ansi",
]
