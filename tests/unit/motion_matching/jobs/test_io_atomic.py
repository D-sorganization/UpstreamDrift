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
