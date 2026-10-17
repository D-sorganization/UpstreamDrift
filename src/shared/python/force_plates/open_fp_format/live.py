"""Optional live bridge from the open-fp Socket.IO feed to realtime IPC (OF-09).

The board's server pushes a Socket.IO event named ``live`` at about 30 Hz.  Each
message is a batch of the 200 Hz samples accumulated since the previous push
(upstream ``app/python/open_fp/pipeline.py``, ``UiBuffer.drain``, and
``app/python/open_fp/app.py``, ``tick``, at pinned commit
``53d682ce70b34bd2d85918b210ebdc81937e33f8``).  The documented JSON shape::

    {"t": [s...],                          # host monotonic seconds, 1e-4 rounding
     "fz": {"L": [N...], "R": [N...]},     # plate loads, 0.1 N rounding
     "cop": {"L": [[x, y]|null...],        # millimetres, 0.1 rounding; a row
             "R": [...], "C": [...]},      # is null where NaN (load < 20 N)
     "lead_pct": [pct|null...], "forces": [[8 cell loads N]...],
     "flags": int, "state": str, "bw_n": float, "lead_plate": str, "golfer": str}

An idle batch has empty lists.  This module implements that documented shape
only; open-fp is AGPL-3.0 and none of its code is imported or copied.

The bridge is strictly receive-only: it registers a ``live`` handler and never
emits anything to the board.  Valid batches are converted to SI units and queued
in a bounded queue (oldest dropped on overflow); :meth:`LiveBridge.pump` hands
the queued messages to the injected ``publish`` callable on
:data:`LIVE_CHANNEL`.  The ``python-socketio`` client is an optional extra
(``force-plate-live``) imported lazily, so importing this module needs no
network library.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from src.shared.python.force_plates.contracts import finite_positive
from src.shared.python.force_plates.open_fp_format._live_payload import (
    LiveBatch,
    LiveBatchError,
    negative_load_floor_n,
    parse_live_batch,
)
from src.shared.python.force_plates.open_fp_format.client import validated_base

__all__ = [
    "LIVE_CHANNEL",
    "BackoffPolicy",
    "LiveBridge",
    "LiveSocket",
    "backoff_delays",
    "default_socket_factory",
    "register_live_channel",
]

logger = logging.getLogger(__name__)

LIVE_CHANNEL = "force_plate/open_fp/live"
LIVE_EVENT = "live"
DISCONNECT_EVENT = "disconnect"
LIVE_SOURCE = "open-fp-live"
LIVE_CHANNEL_DESCRIPTION = (
    "Live open-fp vertical force-plate batches in SI units (unqualified, receive-only)"
)
LIVE_OWNER_TOOL_ID = "ground_reaction_plots"
DEFAULT_MAX_ATTEMPTS = 8


class LiveSocket(Protocol):
    """The slice of a Socket.IO client the bridge uses (no ``emit`` on purpose)."""

    def connect(self, url: str) -> Any: ...

    def on(self, event: str, handler: Callable[..., Any]) -> Any: ...

    def disconnect(self) -> Any: ...


def default_socket_factory() -> LiveSocket:
    """A ``socketio.Client``; ``ImportError`` names the extra when it is absent."""
    try:
        import socketio
    except ImportError as exc:
        raise ImportError(
            "the live open-fp bridge needs python-socketio; install the "
            "'force-plate-live' extra (pip install 'upstream-drift[force-plate-live]')"
        ) from exc
    return socketio.Client(reconnection=False)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """Exponential reconnect delays: ``base_s * factor**k`` capped at ``cap_s``."""

    base_s: float = 0.5
    factor: float = 2.0
    cap_s: float = 30.0

    def delays(self, attempts: int) -> list[float]:
        """The first ``attempts`` delays of this policy."""
        return backoff_delays(self.base_s, self.factor, self.cap_s, attempts)


def backoff_delays(
    base_s: float, factor: float, cap_s: float, attempts: int
) -> list[float]:
    """The first ``attempts`` capped geometric delays [s].

    ``base_s`` and ``cap_s`` must be positive finite with ``cap_s >= base_s``,
    ``factor`` finite and greater than one, ``attempts`` a non-negative int.
    """
    base = finite_positive(base_s, "base_s")
    ratio = finite_positive(factor, "factor")
    cap = finite_positive(cap_s, "cap_s")
    if ratio <= 1.0:
        raise ValueError(f"factor must exceed 1, got {factor!r}")
    if cap < base:
        raise ValueError(f"cap_s ({cap_s!r}) must be at least base_s ({base_s!r})")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
        raise TypeError(f"attempts must be a non-negative int, got {attempts!r}")
    return [min(cap, base * ratio**k) for k in range(attempts)]


def register_live_channel() -> None:
    """Register :data:`LIVE_CHANNEL` in both realtime registries (idempotent)."""
    from src.shared.python.realtime.api import register_channel
    from src.shared.python.realtime.channels import register_channel_hint

    register_channel(
        LIVE_CHANNEL, LIVE_CHANNEL_DESCRIPTION, owner_tool_id=LIVE_OWNER_TOOL_ID
    )
    register_channel_hint(LIVE_CHANNEL, "high")


class LiveBridge:
    """Receive-only bridge: validate ``live`` batches, queue them, publish them.

    ``publish(channel, message)`` receives plain-dict messages.
    ``force_floor_n`` is the plates' declared 1-sigma force floor [N]; loads
    down to ``-3`` floor sigmas are noise and are kept unclamped, as in the
    trial contract, and lower loads make a batch malformed.  Counters:
    ``dropped`` (batches lost to queue overflow) and ``malformed`` (batches
    refused by validation, never forwarded).  No threads: the socket library
    calls the handler, the owner calls :meth:`pump`.
    """

    def __init__(
        self,
        base_url: str,
        publish: Callable[[str, dict[str, Any]], object],
        *,
        max_queue: int,
        force_floor_n: float,
        socket_factory: Callable[[], LiveSocket] | None = None,
        backoff: BackoffPolicy | None = None,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        self._url = validated_base(base_url)
        if not callable(publish):
            raise TypeError("publish must be callable")
        self._publish = publish
        self._max_queue = _positive_int(max_queue, "max_queue")
        self._load_floor_n = negative_load_floor_n(
            finite_positive(force_floor_n, "force_floor_n")
        )
        self._factory = socket_factory or default_socket_factory
        self._backoff = backoff or BackoffPolicy()
        self._sleep = sleep
        self._queue: deque[LiveBatch] = deque()
        self._socket: LiveSocket | None = None
        self.dropped = 0
        self.malformed = 0
        self.connected = False

    def start(self, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> bool:
        """Connect, retrying with capped exponential backoff; ``False`` if it fails."""
        attempts = _positive_int(max_attempts, "max_attempts")
        delays = self._backoff.delays(attempts - 1)
        for attempt in range(attempts):
            if self._try_connect():
                return True
            if attempt < len(delays):
                self._sleep(delays[attempt])
        logger.warning("open-fp live: gave up after %d connection attempts", attempts)
        return False

    def reconnect(self, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> bool:
        """Drop the current socket (if any) and :meth:`start` again."""
        self.stop()
        return self.start(max_attempts)

    def stop(self) -> None:
        """Disconnect and forget the socket (idempotent)."""
        sock, self._socket = self._socket, None
        self.connected = False
        if sock is not None:
            sock.disconnect()

    def pump(self) -> int:
        """Publish every queued batch on :data:`LIVE_CHANNEL`; return the count."""
        sent = 0
        while self._queue:
            self._publish(LIVE_CHANNEL, self._message(self._queue.popleft()))
            sent += 1
        return sent

    def _try_connect(self) -> bool:
        sock = self._factory()
        sock.on(LIVE_EVENT, self._on_live)
        sock.on(DISCONNECT_EVENT, self._on_disconnect)
        try:
            sock.connect(self._url)
        except Exception as exc:  # noqa: BLE001 - any transport failure means retry
            logger.warning("open-fp live: connect to %s failed: %s", self._url, exc)
            return False
        self._socket = sock
        self.connected = True
        return True

    def _on_disconnect(self, *_args: object) -> None:
        self.connected = False

    def _on_live(self, payload: object) -> None:
        try:
            batch = parse_live_batch(payload, negative_floor_n=self._load_floor_n)
        except LiveBatchError as exc:
            self.malformed += 1
            logger.warning("open-fp live: dropped malformed batch: %s", exc)
            return
        if len(batch) == 0:
            return
        if len(self._queue) >= self._max_queue:
            self._queue.popleft()
            self.dropped += 1
        self._queue.append(batch)

    def _message(self, batch: LiveBatch) -> dict[str, Any]:
        return {
            "t_s": batch.t_s,
            "load_n": batch.load_n,
            "cop_m": batch.cop_m,
            "dropped": self.dropped,
            "source": LIVE_SOURCE,
            "qualified": False,
        }


register_live_channel()
