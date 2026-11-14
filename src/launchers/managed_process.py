"""Launcher child-process lifecycle types (#14949, epic #14934).

Convention (per-repo, launcher scope):
- ``STARTING``: ``Popen`` returned but the child has not yet been confirmed alive.
- ``RUNNING``: child confirmed alive (still executing).
- ``EXITED`` / ``CRASHED``: terminal states with explicit ``exit_code``.
- ``RUNNING`` is never reported after a non-zero exit within the early window.
"""

from __future__ import annotations

import contextlib
from collections import deque
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass, field
from enum import Enum
import threading
import uuid
from typing import TYPE_CHECKING, Any

from src.shared.python.security.subprocess_utils import kill_process_tree

if TYPE_CHECKING:
    from subprocess import Popen


class ProcessKind(str, Enum):
    """Origin of a launcher-managed child process."""

    LOCAL = "local"
    WSL = "wsl"
    DOCKER = "docker"
    MATLAB = "matlab"
    API = "api"


class ProcessState(str, Enum):
    """Lifecycle state for a managed child process."""

    STARTING = "starting"
    RUNNING = "running"
    EXITED = "exited"
    CRASHED = "crashed"
    STOPPING = "stopping"
    KILLED = "killed"


class DuplicateKeyPolicy(str, Enum):
    """What to do when launching the same logical key twice."""

    FOCUS_EXISTING = "focus_existing"
    ALLOW_SECOND = "allow_second"
    REPLACE = "replace"


_STDERR_TAIL_LINES = 32


@dataclass
class ManagedProcess:
    """A single launcher-owned child process."""

    id: str
    key: str
    slot_name: str
    kind: ProcessKind
    state: ProcessState
    process: Popen[bytes]
    exit_code: int | None = None
    _stderr_tail: deque[str] = field(
        default_factory=lambda: deque(maxlen=_STDERR_TAIL_LINES)
    )
    _awaiting_output_drain: bool = False

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def last_stderr_tail(self) -> str:
        """Recent stderr lines only (stdout is never mixed into this tail)."""
        return "\n".join(self._stderr_tail)

    def record_output_line(self, line: str) -> None:
        if line.startswith("STDERR: "):
            self._stderr_tail.append(line.removeprefix("STDERR: "))
        elif line.startswith("STDERR:"):
            self._stderr_tail.append(line.removeprefix("STDERR:").lstrip())

    def to_public_dict(self) -> dict[str, Any]:
        poll = None if self._awaiting_output_drain else self.process.poll()
        return {
            "id": self.id,
            "key": self.key,
            "slot_name": self.slot_name,
            "kind": self.kind.value,
            "state": self.state.value,
            "pid": self.pid,
            "running": poll is None
            and self.state in {ProcessState.STARTING, ProcessState.RUNNING},
            "exit_code": self.exit_code if self.exit_code is not None else poll,
            "last_stderr_tail": self.last_stderr_tail,
        }


class RunningProcessesView(MutableMapping[str, "Popen[bytes]"]):
    """Read-only façade over tracked processes keyed by display ``slot_name``."""

    def __init__(
        self,
        registry: dict[str, ManagedProcess],
        lock: threading.RLock,
        attach: Callable[[str, Popen[bytes]], None],
        remove_slot: Callable[[str], None],
        clear_all: Callable[[], None],
    ) -> None:
        self._registry = registry
        self._lock = lock
        self._attach = attach
        self._remove_slot = remove_slot
        self._clear_all = clear_all

    def _slots(self) -> dict[str, Popen[bytes]]:
        return {m.slot_name: m.process for m in self._registry.values()}

    def __getitem__(self, key: str) -> Popen[bytes]:
        with self._lock:
            for managed in self._registry.values():
                if managed.slot_name == key:
                    return managed.process
        raise KeyError(key)

    def __setitem__(self, key: str, process: Popen[bytes]) -> None:
        self._attach(key, process)

    def __delitem__(self, key: str) -> None:
        self._remove_slot(key)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(self._slots())

    def __len__(self) -> int:
        with self._lock:
            return len(self._registry)

    def clear(self) -> None:
        self._clear_all()

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def items(self) -> list[tuple[str, Popen[bytes]]]:
        with self._lock:
            return list(self._slots().items())

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._slots().keys())

    def values(self) -> list[Popen[bytes]]:
        with self._lock:
            return list(self._slots().values())


def new_managed_id() -> str:
    return uuid.uuid4().hex


def reap_declined_child_spawn(process: Popen[bytes]) -> None:
    """Terminate a child whose registration was declined (duplicate policy)."""
    import subprocess

    try:
        if process.poll() is None:
            if not kill_process_tree(process.pid):
                process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
    except (OSError, ValueError):
        pass


def resolve_process_exit_code(process: Popen[bytes]) -> int:
    """Return a child process exit code without extra waiting if already exited."""
    poll = process.poll()
    if isinstance(poll, int):
        return poll
    waited = process.wait()
    return waited if isinstance(waited, int) else 1


def clear_awaiting_output_drain(
    registry: dict[str, ManagedProcess],
    lock: threading.RLock,
    managed_id: str,
) -> None:
    """Clear the output-drain hold on a tracked process if still registered."""
    with lock:
        current = registry.get(managed_id)
        if current is not None:
            current._awaiting_output_drain = False


def finalize_managed_exit(
    registry: dict[str, ManagedProcess],
    lock: threading.RLock,
    managed_id: str,
    process: Popen[bytes],
    notify: Callable[[ManagedProcess], None],
) -> int:
    """Finalize ManagedProcess terminal state and exit code, then notify observer."""
    return_code = resolve_process_exit_code(process)
    snapshot: ManagedProcess | None = None
    with lock:
        current = registry.get(managed_id)
        if current is not None:
            current._awaiting_output_drain = False
            current.exit_code = return_code
            if current.state in {ProcessState.STOPPING, ProcessState.KILLED}:
                terminal = ProcessState.KILLED
            elif return_code == 0:
                terminal = ProcessState.EXITED
            else:
                terminal = ProcessState.CRASHED
            current.state = terminal
            snapshot = current
    if snapshot is not None:
        notify(snapshot)
    return return_code


def drain_process_streams(
    slot_name: str,
    process: Popen[bytes],
    *,
    managed_id: str | None,
    registry: dict[str, ManagedProcess],
    lock: threading.RLock,
    emit_output: Callable[[str, str], None],
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    """Drain stdout and stderr pipes concurrently until EOF."""

    def _drain(stream: Any, *, stderr: bool) -> None:
        try:
            for raw_line in iter(stream.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                emitted = f"STDERR: {line}" if stderr else line
                if managed_id is not None:
                    with lock:
                        managed = registry.get(managed_id)
                        if managed is not None:
                            managed.record_output_line(emitted)
                emit_output(slot_name, emitted)
        except (RuntimeError, ValueError, OSError) as exc:
            if on_error is not None:
                on_error(exc)
        finally:
            with contextlib.suppress(OSError):
                stream.close()

    threads = [
        threading.Thread(
            target=_drain, args=(stream,), kwargs={"stderr": is_stderr}, daemon=True
        )
        for stream, is_stderr in ((process.stdout, False), (process.stderr, True))
        if stream is not None
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
