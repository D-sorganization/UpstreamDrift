"""Canonical atomic promotion preserves snapshots across Windows sharing faults."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from threading import Event, Thread

import pytest

from src.shared.python.motion_matching.jobs import io_atomic

pytestmark = pytest.mark.unit


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows sharing contract")
def test_short_lived_reader_allows_atomic_promotion_after_release(tmp_path):
    destination = tmp_path / "request.json"
    staged = tmp_path / ".request.tmp"
    destination.write_text('{"revision": 1}')
    staged.write_text('{"revision": 2}')
    entered, finished = Event(), Event()
    errors = []

    def promote():
        entered.set()
        try:
            io_atomic._promote_temp(staged, destination)
        except OSError as exc:
            errors.append(exc)
        finally:
            finished.set()

    with destination.open() as previous:
        thread = Thread(target=promote)
        thread.start()
        assert entered.wait(2)
        Event().wait(0.06)
        assert json.load(previous) == {"revision": 1}
    thread.join(2)
    assert finished.is_set() and not errors
    assert json.loads(destination.read_text()) == {"revision": 2}
    assert not staged.exists()


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_verified_windows_sharing_error_retries_before_success(
    tmp_path, monkeypatch, winerror
):
    staged, destination = tmp_path / "stage", tmp_path / "destination"
    staged.write_bytes(b"new")
    destination.write_bytes(b"old")
    original = Path.replace
    attempts = []
    error = PermissionError("verified Windows sharing fault")
    error.winerror = winerror

    def replace(self, target):
        attempts.append(self)
        if len(attempts) == 1:
            raise error
        return original(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    io_atomic._promote_temp(staged, destination)
    assert len(attempts) == 2
    assert destination.read_bytes() == b"new" and not staged.exists()


@pytest.mark.parametrize("winerror", [None, 123])
def test_unrelated_permission_error_propagates_without_retry(
    tmp_path, monkeypatch, winerror
):
    staged, destination = tmp_path / "stage", tmp_path / "destination"
    staged.write_bytes(b"new")
    destination.write_bytes(b"old")
    error = PermissionError("permanent unrelated fault")
    if winerror is not None:
        error.winerror = winerror
    attempts = []

    def replace(self, target):
        attempts.append(self)
        raise error

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(PermissionError) as raised:
        io_atomic._promote_temp(staged, destination)
    assert raised.value is error and len(attempts) == 1
    assert not staged.exists() and destination.read_bytes() == b"old"


def test_permanent_windows_sharing_error_exhausts_finite_budget(tmp_path, monkeypatch):
    staged, destination = tmp_path / "stage", tmp_path / "destination"
    staged.write_bytes(b"new")
    destination.write_bytes(b"old")
    error = PermissionError("permanent sharing fault")
    error.winerror = 32
    attempts = []

    def replace(self, target):
        attempts.append(self)
        raise error

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(PermissionError) as raised:
        io_atomic._promote_temp(staged, destination)
    assert raised.value is error and 1 < len(attempts) <= 10
    assert not staged.exists() and destination.read_bytes() == b"old"


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_read_recovers_without_changing_file(tmp_path, monkeypatch, winerror):
    path = tmp_path / "request.json"
    path.write_text('{"status": "running"}', encoding="utf-8")
    original = Path.read_text
    attempts, delays = [], []
    error = PermissionError("verified sharing fault")
    error.winerror = winerror

    def read(self, *args, **kwargs):
        attempts.append(self)
        if len(attempts) < 3:
            raise error
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(io_atomic.time, "sleep", delays.append)
    assert io_atomic.read_text(path) == '{"status": "running"}'
    assert len(attempts) == 3 and delays == [0.01, 0.02]
    assert path.read_bytes() == b'{"status": "running"}'


def test_read_sharing_exhaustion_is_bounded(tmp_path, monkeypatch):
    error = PermissionError("persistent sharing fault")
    error.winerror = 32
    attempts, delays = [], []

    def read(self, *args, **kwargs):
        attempts.append(self)
        raise error

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(io_atomic.time, "sleep", delays.append)
    with pytest.raises(PermissionError) as raised:
        io_atomic.read_text(tmp_path / "request.json")
    assert raised.value is error
    assert len(attempts) == 6
    assert delays == [0.01, 0.02, 0.04, 0.08, 0.16]
    assert sum(delays) == pytest.approx(0.31)


@pytest.mark.parametrize(
    "error",
    [
        PermissionError(13, "unclassified"),
        FileNotFoundError("absent"),
        OSError("unrelated"),
    ],
)
def test_unclassified_read_error_propagates_immediately(tmp_path, monkeypatch, error):
    attempts, delays = [], []

    def read(self, *args, **kwargs):
        attempts.append(self)
        raise error

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(io_atomic.time, "sleep", delays.append)
    with pytest.raises(type(error)) as raised:
        io_atomic.read_text(tmp_path / "request.json")
    assert raised.value is error and len(attempts) == 1 and not delays


@pytest.mark.parametrize("native_code", [5, 32, 33])
def test_errno_only_read_uses_verified_fresh_windows_error(
    tmp_path, monkeypatch, native_code
):
    path = tmp_path / "request.json"
    path.write_text("valid", encoding="utf-8")
    original = Path.read_text
    native_state, attempts = [123], []
    error = PermissionError(13, "CRT omits winerror")

    def read(self, *args, **kwargs):
        assert native_state[0] == 0
        attempts.append(self)
        if len(attempts) == 1:
            native_state[0] = native_code
            raise error
        return original(self, *args, **kwargs)

    functions = (
        lambda code: native_state.__setitem__(0, code),
        lambda: native_state[0],
    )
    monkeypatch.setattr(io_atomic, "_windows_error_functions", lambda: functions)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(io_atomic.time, "sleep", lambda _: None)
    assert io_atomic.read_text(path) == "valid"
    assert len(attempts) == 2 and error.winerror == native_code


@pytest.mark.parametrize("code,explicit", [(0, None), (123, None), (32, 123)])
def test_stale_or_unrelated_native_error_cannot_enable_retry(
    tmp_path, monkeypatch, code, explicit
):
    native_state, attempts, delays = [32], [], []
    error = PermissionError(13, "unverified")
    if explicit is not None:
        error.winerror = explicit

    def read(self, *args, **kwargs):
        assert native_state[0] == 0
        attempts.append(self)
        native_state[0] = code
        raise error

    functions = (
        lambda value: native_state.__setitem__(0, value),
        lambda: native_state[0],
    )
    monkeypatch.setattr(io_atomic, "_windows_error_functions", lambda: functions)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(io_atomic.time, "sleep", delays.append)
    with pytest.raises(PermissionError) as raised:
        io_atomic.read_text(tmp_path / "request.json")
    assert raised.value is error and len(attempts) == 1 and not delays


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows sharing contract")
def test_owned_exclusive_windows_handle_releases_for_canonical_read(
    tmp_path, monkeypatch
):
    import ctypes
    from ctypes import wintypes

    path = tmp_path / "request.json"
    path.write_text('{"status": "running"}', encoding="utf-8")
    kernel = ctypes.WinDLL("kernel32", use_last_error=False)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes, close.restype = [wintypes.HANDLE], wintypes.BOOL
    handle = create(str(path), 0x80000000, 0, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value
    released, delays = [], []

    def release_on_backoff(delay):
        delays.append(delay)
        assert close(handle)
        released.append(True)

    try:
        with pytest.raises(PermissionError) as raised:
            path.read_text(encoding="utf-8")
        assert raised.value.errno == 13
        assert getattr(raised.value, "winerror", None) is None
        monkeypatch.setattr(io_atomic.time, "sleep", release_on_backoff)
        assert io_atomic.read_text(path) == '{"status": "running"}'
        assert delays == [0.01] and released == [True]
    finally:
        if not released:
            close(handle)


def test_non_windows_read_does_not_use_native_error_state(tmp_path, monkeypatch):
    path = tmp_path / "request.json"
    error = PermissionError(13, "non-Windows permission failure")
    delays = []
    monkeypatch.setattr(io_atomic.os, "name", "posix")
    assert io_atomic._windows_error_functions() is None

    def read(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(io_atomic.time, "sleep", delays.append)
    with pytest.raises(PermissionError) as raised:
        io_atomic.read_text(path)
    assert raised.value is error and not delays


def test_invalid_utf8_is_not_retried(tmp_path, monkeypatch):
    path = tmp_path / "request.json"
    path.write_bytes(b"\xff")
    delays = []
    monkeypatch.setattr(io_atomic.time, "sleep", delays.append)
    with pytest.raises(UnicodeDecodeError):
        io_atomic.read_text(path)
    assert not delays


@pytest.mark.parametrize("value", ["request.json", None])
def test_read_text_requires_path_before_any_io(value, monkeypatch):
    def bind():
        raise AssertionError("Invalid input reached native error binding")

    monkeypatch.setattr(io_atomic, "_windows_error_functions", bind)
    with pytest.raises(TypeError, match="pathlib.Path"):
        io_atomic.read_text(value)
