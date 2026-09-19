"""Bound native render traffic; source timestamps remain authoritative.

This is a renderer backpressure adapter, not a UI playback transport. It drops
stale display samples when CORBA/rendering falls behind rather than queuing them.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def presentation_frames(
    time_s: NDArray[np.float64],
    fps: float,
    speed: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[int]:
    """Yield latest source sample at bounded cadence, including the endpoint."""
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(speed) or speed <= 0:
        raise ValueError("Presentation fps and speed must be finite and positive")
    started = clock()
    previous = -1
    while True:
        physical = time_s[0] + (clock() - started) * speed
        index = min(
            len(time_s) - 1, int(np.searchsorted(time_s, physical, side="right")) - 1
        )
        if index != previous:
            yield index
            previous = index
        if index == len(time_s) - 1:
            return
        sleep(1.0 / fps)


@contextmanager
def gepetto_playback_lock() -> Iterator[None]:
    """Only one owned candidate stream may write to the shared Gepetto scene."""
    uid = getattr(os, "getuid", lambda: "shared")()
    path = Path(tempfile.gettempdir()) / f"upstream-gepetto-{uid}.lock"
    with path.open("a+", encoding="utf-8") as lock_file:
        if sys.platform != "win32":
            import fcntl

            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "Gepetto candidate playback is already running; "
                    "reuse it or stop it before changing candidates"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        else:
            import msvcrt

            try:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    "Gepetto candidate playback is already running; "
                    "reuse it or stop it before changing candidates"
                ) from exc
            try:
                yield
            finally:
                lock_file.seek(0)
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
