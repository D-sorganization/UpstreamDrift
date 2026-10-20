"""The unit-gate hang watchdog must dump stacks without racing other threads.

Issue #13490: the root ``conftest.py`` armed
``faulthandler.register(SIGUSR1, all_threads=True)``. That C-level handler
walks every thread's frame stack without the GIL. On the xdist controller,
busy execnet receiver threads freed frames mid-walk and the whole session
segfaulted (exit 139), ejecting merge-queue batches. The handler must be a
Python-level signal handler (runs with the GIL held) that calls
``faulthandler.dump_traceback(all_threads=True)``.

These tests are structural: they record what ``_arm_hang_forensics``
installs instead of sending real signals or spawning the watchdog.
"""

from __future__ import annotations

import importlib.util
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONFTEST = ROOT / "conftest.py"
_FAKE_SIGUSR1 = 10


def _load_root_conftest() -> Any:
    spec = importlib.util.spec_from_file_location(
        "root_conftest_hang_forensics", CONFTEST
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakePluginManager:
    def getplugin(self, name: str) -> None:
        return None


class _FakeConfig:
    pluginmanager = _FakePluginManager()


class _FakePopen:
    calls: list[list[str]] = []

    def __init__(self, args: list[str], **kwargs: Any) -> None:
        _FakePopen.calls.append(args)
        stderr = kwargs.get("stderr")
        if stderr is not None:
            stderr.close()


@pytest.fixture
def armed(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Arm the forensics with signals, faulthandler and Popen recorded."""
    import faulthandler

    installed: dict[str, Any] = {}

    def _record_signal(signum: int, handler: Any) -> None:
        installed["signum"] = signum
        installed["handler"] = handler

    def _forbidden_register(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "faulthandler.register dumps all threads without the GIL and "
            "segfaulted the xdist controller (#13490)"
        )

    def _record_dump(file: Any = None, all_threads: bool = True) -> None:
        installed.setdefault("dumps", []).append((file, all_threads))

    monkeypatch.setattr(signal, "SIGUSR1", _FAKE_SIGUSR1, raising=False)
    monkeypatch.setattr(signal, "signal", _record_signal)
    monkeypatch.setattr(faulthandler, "register", _forbidden_register, raising=False)
    monkeypatch.setattr(faulthandler, "dump_traceback", _record_dump)
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    _FakePopen.calls = []

    module = _load_root_conftest()
    module._arm_hang_forensics(_FakeConfig())
    yield installed
    # Recover the pipe the handler writes to (via the recorded dump call) and
    # close it so the test leaks no file descriptor.
    if "handler" in installed:
        installed["handler"](_FAKE_SIGUSR1, None)
        installed["dumps"][-1][0].close()


def test_watchdog_installs_a_python_level_sigusr1_handler(
    armed: dict[str, Any],
) -> None:
    assert armed.get("signum") == _FAKE_SIGUSR1, (
        "the hang dump must be a Python-level signal.signal handler, not "
        "faulthandler.register"
    )
    assert callable(armed["handler"])
    assert len(_FakePopen.calls) == 1, "the watchdog subprocess must still be spawned"


def test_handler_dumps_every_thread_through_faulthandler(armed: dict[str, Any]) -> None:
    armed["handler"](_FAKE_SIGUSR1, None)
    dumps = armed.get("dumps", [])
    assert len(dumps) == 1
    dump_file, all_threads = dumps[0]
    assert all_threads is True
    assert dump_file.fileno() >= 0, "the dump must go to the watchdog pipe fd"
