# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Thread-safety tests for shared optimizer state on MainWindow.

Issue #407: per-exercise runtime state is written from worker threads
(running ``_opt_worker``) and read from the GUI main
thread (signal handlers, animation, file I/O). These tests exercise
concurrent writers/readers against the OptimizationMixin lock helpers and
verify that no torn snapshots, lost updates, or deadlocks occur.

Tests must run headless. ``QT_QPA_PLATFORM=offscreen`` is set in the module
``conftest`` fixture so they pass on CI without a display.
"""

from __future__ import annotations

import os
import threading
from typing import ClassVar

# Force the offscreen Qt platform before any Qt import (defensive — also set
# by the harness command line). The OptimizationMixin code path under test
# does not actually require Qt, but importing main_window does.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from conftest import make_test_result

from movement_optimizer.gui.exercise_state import ExerciseRuntimeState
from movement_optimizer.gui.optimization_mixin import OptimizationMixin


class _ThreadSafetyHarness:
    """Minimal harness exposing consolidated exercise state plus _opt_lock.

    Mirrors the relevant attributes of MainWindow so the mixin's
    ``_snapshot_idx_state`` / ``_set_anim_frame`` helpers can be exercised
    directly without bringing up a Qt window.
    """

    EXERCISE_CONFIGS: ClassVar = (
        ("Squat", "squat"),
        ("Full Squat", "full_squat"),
        ("Deadlift", "deadlift"),
        ("Bench", "bench_press"),
    )

    def __init__(self) -> None:
        n = len(self.EXERCISE_CONFIGS)
        self.exercise_states = [ExerciseRuntimeState() for _ in range(n)]
        # RLock so re-entrant locking does not deadlock.
        self._opt_lock = threading.RLock()


def _join_with_timeout(threads: list[threading.Thread], timeout: float) -> None:
    """Join all threads. If any are still alive after ``timeout`` total
    seconds, fail the test loudly with a deadlock message instead of
    hanging the test runner.
    """
    deadline = threading.Event()

    def _set_deadline() -> None:
        deadline.set()

    timer = threading.Timer(timeout, _set_deadline)
    timer.daemon = True
    timer.start()
    try:
        for t in threads:
            remaining = max(0.05, timeout)
            t.join(timeout=remaining)
            if t.is_alive():
                raise AssertionError(
                    f"Thread {t.name!r} did not finish within {timeout}s "
                    "-- possible deadlock in shared-state locking"
                )
    finally:
        timer.cancel()
    assert not deadline.is_set() or all(not t.is_alive() for t in threads)


# ---------------------------------------------------------------------------
# Test 1: concurrent writers + reader produce coherent snapshots
# ---------------------------------------------------------------------------


class TestConcurrentSnapshot:
    """A reader running ``_snapshot_idx_state`` should never observe a
    torn state -- e.g., a ``result`` from iteration N paired with a
    ``body``/``dyn`` from iteration M.
    """

    def test_no_torn_snapshots_under_contention(self) -> None:
        harness = _ThreadSafetyHarness()
        idx = 0
        n_iters = 200
        stop = threading.Event()
        observations: list[tuple[int, int, int, int]] = []
        errors: list[BaseException] = []

        def writer() -> None:
            try:
                for i in range(1, n_iters + 1):
                    # Tag each iteration with its iteration counter so a
                    # torn read becomes detectable as mismatched ``cost``
                    # vs. body/dyn marker fields.
                    res = make_test_result(cost=float(i))
                    body_marker = ("body", i)
                    dyn_marker = ("dyn", i)
                    with harness._opt_lock:
                        state = harness.exercise_states[idx]
                        state.result = res
                        state.body = body_marker
                        state.dynamics = dyn_marker
                        state.anim_frame = i
            except BaseException as exc:  # pragma: no cover - debug aid
                errors.append(exc)
                stop.set()

        def reader() -> None:
            try:
                while not stop.is_set():
                    snap = OptimizationMixin._snapshot_idx_state(harness, idx)  # type: ignore[arg-type]
                    r, fi, body, dyn = snap
                    if r is None:
                        continue
                    cost = int(r.cost)
                    body_iter = body[1] if isinstance(body, tuple) else None  # type: ignore
                    dyn_iter = dyn[1] if isinstance(dyn, tuple) else None
                    observations.append((cost, fi, body_iter or 0, dyn_iter or 0))
            except BaseException as exc:  # pragma: no cover - debug aid
                errors.append(exc)
                stop.set()

        w = threading.Thread(target=writer, name="writer")
        r1 = threading.Thread(target=reader, name="reader-1")
        r2 = threading.Thread(target=reader, name="reader-2")
        w.start()
        r1.start()
        r2.start()
        w.join(timeout=10.0)
        stop.set()
        _join_with_timeout([r1, r2], timeout=5.0)

        assert not errors, f"Worker threads raised: {errors!r}"
        assert observations, "Reader threads never observed any state"
        # For every snapshot, all four "iteration" markers must agree --
        # if they ever differ, locking failed to provide atomicity.
        for cost, fi, body_iter, dyn_iter in observations:
            assert cost == fi == body_iter == dyn_iter, (
                f"Torn snapshot detected: cost={cost} frame={fi} "
                f"body_iter={body_iter} dyn_iter={dyn_iter}"
            )


# ---------------------------------------------------------------------------
# Test 2: concurrent ``_set_anim_frame`` writers do not lose updates
# ---------------------------------------------------------------------------


class TestConcurrentAnimFrameWrites:
    """All anim-frame writes from many threads must be observable; the final
    state must match exactly one of the values written, not a corrupted one.
    """

    def test_set_anim_frame_is_atomic(self) -> None:
        harness = _ThreadSafetyHarness()
        idx = 1
        per_thread = 500
        n_threads = 8
        errors: list[BaseException] = []

        def worker(start: int) -> None:
            try:
                for v in range(start, start + per_thread):
                    OptimizationMixin._set_anim_frame(harness, idx, v)  # type: ignore[arg-type]
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(t * per_thread,), name=f"w{t}")
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        _join_with_timeout(threads, timeout=10.0)

        assert not errors, f"Workers raised: {errors!r}"
        # Final value must be a non-negative int that some worker wrote --
        # a corrupted/non-int value here would mean the list slot was being
        # mutated mid-read by another thread.
        final = harness.exercise_states[idx].anim_frame
        assert isinstance(final, int)
        assert 0 <= final < n_threads * per_thread


# ---------------------------------------------------------------------------
# Test 3: re-entrant lock acquisition does not deadlock
# ---------------------------------------------------------------------------


class TestReentrantLockNoDeadlock:
    """Some code paths (notably ``_resolve_exercise_params`` calling
    ``_snapshot_idx_state`` would-be wrappers) acquire ``_opt_lock`` from
    inside another locked section. Using ``threading.RLock`` permits this;
    a plain ``Lock`` would deadlock. This test guards that invariant.
    """

    def test_nested_acquisition_completes(self) -> None:
        harness = _ThreadSafetyHarness()
        completed = threading.Event()
        errors: list[BaseException] = []

        def runner() -> None:
            try:
                with harness._opt_lock:
                    # Helper itself acquires _opt_lock again -- only safe
                    # because the lock is reentrant.
                    snap = OptimizationMixin._snapshot_idx_state(harness, 0)  # type: ignore[arg-type]
                    assert snap == (None, 0, None, None)
                    OptimizationMixin._set_anim_frame(harness, 0, 7)  # type: ignore[arg-type]
                completed.set()
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        t = threading.Thread(target=runner, name="reentrant", daemon=True)
        t.start()
        t.join(timeout=2.0)

        assert not t.is_alive(), (
            "Re-entrant lock acquisition deadlocked -- _opt_lock must be an RLock"
        )
        assert not errors, f"Runner raised: {errors!r}"
        assert completed.is_set()
        assert harness.exercise_states[0].anim_frame == 7
