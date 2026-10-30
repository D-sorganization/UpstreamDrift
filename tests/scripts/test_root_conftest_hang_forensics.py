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
    instances: list[_FakePopen] = []

    def __init__(self, args: list[str], **kwargs: Any) -> None:
        _FakePopen.calls.append(args)
        _FakePopen.instances.append(self)
        self.terminated = False
        self.waited = False
        stderr = kwargs.get("stderr")
        if stderr is not None:
            stderr.close()

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0


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
    _FakePopen.instances = []

    module = _load_root_conftest()
    cfg = _FakeConfig()
    module._arm_hang_forensics(cfg)
    installed["module"] = module
    installed["config"] = cfg
    yield installed
    # Recover the pipe the handler writes to (via the recorded dump call) and
    # close it so the test leaks no file descriptor.
    if "handler" in installed and callable(installed["handler"]):
        with pytest.MonkeyPatch.context():
            try:
                installed["handler"](_FAKE_SIGUSR1, None)
                installed["dumps"][-1][0].close()
            except ValueError:
                pass
    if hasattr(module, "_disarm_hang_forensics"):
        module._disarm_hang_forensics(cfg)


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


def test_pytest_unconfigure_disarms_watchdog_and_ignores_sigusr1(
    armed: dict[str, Any],
) -> None:
    """Issue #14109: disarm watchdog before Py_FinalizeEx resets SIGUSR1 to SIG_DFL."""
    armed["handler"](_FAKE_SIGUSR1, None)
    dump_file, _ = armed["dumps"][-1]
    assert not dump_file.closed

    module = armed["module"]
    assert hasattr(module, "pytest_unconfigure"), (
        "conftest.py must define pytest_unconfigure to disarm the hang watchdog"
    )
    module.pytest_unconfigure(armed["config"])

    assert armed.get("signum") == _FAKE_SIGUSR1
    assert armed.get("handler") == signal.SIG_IGN, (
        "pytest_unconfigure must set SIGUSR1 to SIG_IGN before interpreter shutdown"
    )
    assert dump_file.closed, "pytest_unconfigure must close the dump pipe"
    assert len(_FakePopen.instances) == 1
    assert _FakePopen.instances[0].terminated, (
        "pytest_unconfigure must terminate the watchdog subprocess"
    )
    assert _FakePopen.instances[0].waited, (
        "pytest_unconfigure must reap the watchdog subprocess"
    )


def test_watchdog_script_exits_immediately_on_stdin_eof() -> None:
    """Issue #14109: closing the parent pipe must make the watchdog exit without waiting 10s."""
    import os

    module = _load_root_conftest()
    proc = subprocess.Popen(
        [sys.executable, "-c", module._WATCHDOG_SRC, str(os.getpid())],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(b"thread dump line\n")
        proc.stdin.close()
        rc = proc.wait(timeout=2.0)
        assert rc == 0
        assert proc.stderr is not None
        err = proc.stderr.read().decode("utf-8", errors="replace")
        assert "[hang-watchdog] thread dump line" in err
    finally:
        proc.kill()
        proc.wait()
        if proc.stderr is not None:
            proc.stderr.close()
