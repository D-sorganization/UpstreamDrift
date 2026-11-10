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

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def last_stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def record_output_line(self, line: str) -> None:
        if line.startswith("STDERR: "):
            self._stderr_tail.append(line.removeprefix("STDERR: "))
        elif line.startswith("STDERR:"):
            self._stderr_tail.append(line.removeprefix("STDERR:").lstrip())
        elif line and not line.startswith("["):
            self._stderr_tail.append(line)

    def to_public_dict(self) -> dict[str, Any]:
        poll = self.process.poll()
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
