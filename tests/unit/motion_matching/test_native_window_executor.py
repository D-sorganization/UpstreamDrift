"""Persistent isolated execution preserves order and reports failures."""

import multiprocessing
import os
import time

import pytest

from src.shared.python.motion_matching.native_window_executor import (
    NativeWindowExecutor,
    WindowEvaluationError,
)

pytestmark = pytest.mark.unit


def deterministic_window(payload: bytes) -> tuple[int, int]:
    value = int(payload)
    if value < 0:
        raise ValueError("deliberate worker failure")
    time.sleep(0.02 if value == 1 else 0)
    return value * value, os.getpid()


@pytest.fixture(autouse=True)
def single_thread_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(name, "1")


@pytest.mark.parametrize("workers", [0, 2])
def test_order_repeated_evaluations_and_cleanup(workers: int) -> None:
    with NativeWindowExecutor(deterministic_window, workers=workers) as executor:
        first = executor.evaluate([b"1", b"2", b"3", b"4"])
        second = executor.evaluate([b"4", b"3", b"2", b"1"])
        assert [result[0] for result in first] == [1, 4, 9, 16]
        assert [result[0] for result in second] == [16, 9, 4, 1]
        pids = {result[1] for result in first + second}
        if workers:
            assert os.getpid() not in pids
            assert 1 <= len(pids) <= 2
        else:
            assert pids == {os.getpid()}
        assert executor.evaluate([]) == ()
    if workers:
        assert not pids.intersection(
            child.pid for child in multiprocessing.active_children()
        )
    executor.close()
    with pytest.raises(RuntimeError, match="closed"):
        executor.evaluate([b"1"])


@pytest.mark.parametrize("workers", [0, 2])
def test_failure_propagates_and_closes(workers: int) -> None:
    with NativeWindowExecutor(deterministic_window, workers=workers) as executor:
        pids = {item[1] for item in executor.evaluate([b"1", b"2"])}
        with pytest.raises(WindowEvaluationError, match="window 1") as error:
            executor.evaluate([b"1", b"-1", b"2"])
        assert isinstance(error.value.__cause__, ValueError)
        with pytest.raises(RuntimeError, match="closed"):
            executor.evaluate([b"3"])
    if workers:
        assert not pids.intersection(
            child.pid for child in multiprocessing.active_children()
        )


def test_contracts_reject_mutable_requests_and_bad_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for workers in (1, 3, -1, True):
        with pytest.raises(ValueError):
            NativeWindowExecutor(deterministic_window, workers=workers)
    with NativeWindowExecutor(deterministic_window) as executor:
        with pytest.raises(TypeError, match="bytes"):
            executor.evaluate([bytearray(b"1")])
    monkeypatch.delenv("OMP_NUM_THREADS")
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        NativeWindowExecutor(deterministic_window, workers=2)
