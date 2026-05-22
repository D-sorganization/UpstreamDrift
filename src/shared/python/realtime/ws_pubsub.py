"""WebSocket-based publish/subscribe backend.

Two backends are wired in this module:

* ``rust`` (default when the ``upstream_realtime`` wheel is importable) —
  the Tokio + tokio-tungstenite server in the ``upstream-realtime`` crate.
  Owns its own runtime and exposes a sync publish/subscribe ABI; no
  asyncio in the Python hot path.
* ``python`` — the legacy FastAPI/uvicorn autostart server, kept as a
  fallback for environments where the Rust wheel is unavailable.

Selection is controlled by the ``UD_REALTIME_BACKEND`` env var. When
unset, ``rust`` is used iff the wheel imports; otherwise the python
backend is used.

Reconnection (python backend only) is exponential (1s, 2s, 4s, ...,
capped at 30s).

Latency budget: < 50 ms one-hop (acceptance < 10 ms median, < 50 ms p99
for the rust backend — see issue #5214).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

from .protocol import Subscription, validate_channel

__all__ = ["WSPubSub"]


logger = logging.getLogger(__name__)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _has_rust_wheel() -> bool:
    """Return True iff the ``upstream_realtime`` wheel is importable."""
    try:
        import upstream_realtime  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _resolve_backend() -> str:
    """Resolve the realtime backend to use.

    Honours ``UD_REALTIME_BACKEND`` (``"rust"`` or ``"python"``). Defaults
    to ``"rust"`` when the wheel is present, otherwise ``"python"``.
    """
    requested = os.environ.get("UD_REALTIME_BACKEND", "").strip().lower()
    if requested == "rust":
        if not _has_rust_wheel():
            logger.warning(
                "UD_REALTIME_BACKEND=rust but upstream_realtime wheel "
                "not importable; falling back to python backend"
            )
            return "python"
        return "rust"
    if requested == "python":
        return "python"
    return "rust" if _has_rust_wheel() else "python"


REALTIME_BACKEND = _resolve_backend()


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        try:
            s.connect((host, port))
        except OSError:
            return False
        return True


class _BackoffSleeper:
    """1s → 2s → 4s → … → capped at 30s."""

    def __init__(self, cap: float = 30.0) -> None:
        self._cap = cap
        self._delay = 1.0

    def reset(self) -> None:
        self._delay = 1.0

    def wait(self, stop: threading.Event) -> bool:
        """Sleep for the current delay (with jitter), then double it.

        Applies 50-100% jitter to the computed delay to prevent a
        thundering-herd on server restart when many subscribers reconnect
        simultaneously.

        Returns False if ``stop`` was set during the wait.
        """
        jittered = self._delay * (0.5 + random.random() * 0.5)  # 50-100%
        ok = not stop.wait(jittered)
        self._delay = min(self._delay * 2.0, self._cap)
        return ok


class WSPubSub:
    """WebSocket pub-sub backend.

    Delegates to the Rust ``upstream_realtime`` server when available (and
    when ``UD_REALTIME_BACKEND`` is not pinned to ``python``); otherwise
    spawns a daemon FastAPI/uvicorn server on the configured port.

    Args:
        host: Host of the WS server. Defaults to 127.0.0.1.
        port: Port of the WS server. Defaults to 8765.
        autostart: If True, spawn the backend server on the configured port
            if nothing is listening. Defaults to True.
        backend: Override the module-level :data:`REALTIME_BACKEND` for this
            instance. ``None`` (default) uses the module-level value.
    """

    # Shared HTTP client for the python-backend publish path.
    # Declared at class level so mypy knows the attribute exists.
    # httpx is a soft/optional dep; imported lazily at first publish call.
    _http_client: Any = None

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        autostart: bool = True,
        backend: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.backend = (backend or REALTIME_BACKEND).lower()
        self._server_thread: threading.Thread | None = None
        self._rust_server = None  # upstream_realtime.Server | None
        if autostart:
            if self.backend == "rust":
                self._start_rust_server()
            elif not _port_in_use(host, port):
                self._spawn_server()

    # -- rust backend --------------------------------------------------------

    def _start_rust_server(self) -> None:
        try:
            import upstream_realtime  # type: ignore
        except Exception:  # noqa: BLE001
            logger.warning(
                "rust backend requested but upstream_realtime not importable; "
                "falling back to python backend"
            )
            self.backend = "python"
            if not _port_in_use(self.host, self.port):
                self._spawn_server()
            return
        try:
            srv = upstream_realtime.Server(self.host, self.port)
            self._rust_server = srv
            # Resolve port if caller passed 0 (OS-assigned).
            with contextlib.suppress(Exception):
                self.port = int(srv.bound_port())
        except Exception:
            logger.exception("failed to start rust upstream_realtime server")
            self._rust_server = None
            self.backend = "python"
            if not _port_in_use(self.host, self.port):
                self._spawn_server()

    def stop(self) -> None:
        """Tear down the Rust server (no-op for the python backend)."""
        if self._rust_server is not None:
            try:
                self._rust_server.stop()
            except Exception:
                logger.exception("rust upstream_realtime stop failed")
            self._rust_server = None

    # -- server bootstrap ----------------------------------------------------

    def _spawn_server(self) -> None:
        """Spawn a minimal FastAPI app exposing the realtime routes."""
        try:
            import uvicorn
            from fastapi import FastAPI
        except Exception:  # noqa: BLE001
            logger.warning("FastAPI/uvicorn not importable; cannot autostart WS server")
            return

        try:
            from src.api.routes.realtime import router as realtime_router
        except Exception:
            logger.exception("failed to import realtime router for autostart")
            return

        app = FastAPI(title="UpstreamDrift realtime (autostart)")
        app.include_router(realtime_router)

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            lifespan="on",
        )
        server = uvicorn.Server(config)

        def _run() -> None:
            try:
                server.run()
            except Exception:
                logger.exception("autostarted realtime server crashed")

        t = threading.Thread(target=_run, name="realtime-ws-server", daemon=True)
        t.start()
        self._server_thread = t

        # Wait briefly for the port to come up.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if _port_in_use(self.host, self.port):
                return
            time.sleep(0.05)
        logger.warning("autostarted WS server did not bind within 5s")

    # -- publish -------------------------------------------------------------

    def _publish_url(self) -> str:
        return f"http://{self.host}:{self.port}/realtime/publish"

    def _subscribe_url(self, channel: str) -> str:
        return f"ws://{self.host}:{self.port}/realtime/subscribe?channel={channel}"

    def publish(self, channel: str, payload: dict) -> None:
        validate_channel(channel)
        if not isinstance(payload, dict):
            raise TypeError(f"payload must be a dict, got {type(payload).__name__}")

        if self.backend == "rust" and self._rust_server is not None:
            self._rust_server.publish(channel, json.dumps(payload))
            return

        try:
            import httpx
        except Exception as exc:
            raise RuntimeError("httpx is required for WS pub-sub publish") from exc

        body = {"channel": channel, "payload": payload}

        # Reuse a class-level client to avoid per-publish connection overhead.
        if WSPubSub._http_client is None:
            WSPubSub._http_client = httpx.Client(timeout=2.0)
        r = WSPubSub._http_client.post(self._publish_url(), json=body)
        r.raise_for_status()

    # -- subscribe -----------------------------------------------------------

    def subscribe(
        self,
        channel: str,
        callback: Callable[[dict], None],
    ) -> Subscription:
        validate_channel(channel)

        if self.backend == "rust" and self._rust_server is not None:
            return self._subscribe_rust(channel, callback)

        stop = threading.Event()
        url = self._subscribe_url(channel)

        def _run() -> None:
            try:
                import websockets  # type: ignore
            except Exception:
                logger.exception("websockets package not importable")
                return

            backoff = _BackoffSleeper()

            async def _consume() -> None:
                while not stop.is_set():
                    try:
                        async with websockets.connect(url) as ws:
                            backoff.reset()
                            while not stop.is_set():
                                try:
                                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                                except asyncio.TimeoutError:
                                    continue
                                try:
                                    data = json.loads(msg)
                                except (TypeError, ValueError):
                                    continue
                                try:
                                    callback(data)
                                except Exception:
                                    logger.exception(
                                        "WS subscriber callback raised for %s",
                                        channel,
                                    )
                    except Exception as exc:  # noqa: BLE001
                        if stop.is_set():
                            return
                        logger.warning(
                            "WS subscribe to %s failed: %s; reconnecting", url, exc
                        )
                        if not backoff.wait(stop):
                            return

            try:
                asyncio.run(_consume())
            except Exception:
                logger.exception("WS subscribe loop crashed for %s", channel)

        thread = threading.Thread(
            target=_run,
            name=f"realtime-ws-sub-{channel}",
            daemon=True,
        )
        thread.start()

        def _unsubscribe() -> None:
            stop.set()
            if thread.is_alive() and threading.current_thread() is not thread:
                thread.join(timeout=2.0)

        return Subscription(
            channel=channel, callback=callback, _unsubscribe=_unsubscribe
        )

    # -- rust subscribe ------------------------------------------------------

    def _subscribe_rust(
        self,
        channel: str,
        callback: Callable[[dict], None],
    ) -> Subscription:
        """Subscribe via the in-process Rust broadcast channel.

        The Rust ``Subscriber`` exposes a blocking ``recv(timeout)`` that
        releases the GIL during the wait, so we can park a single Python
        thread per subscription without burning CPU.
        """
        assert self._rust_server is not None
        sub = self._rust_server.subscribe(channel)
        stop = threading.Event()

        def _run() -> None:
            while not stop.is_set():
                try:
                    payload_json = sub.recv(1.0)
                except Exception:
                    if stop.is_set():
                        return
                    logger.exception(
                        "rust upstream_realtime recv failed for %s", channel
                    )
                    # Brief backoff to avoid a hot loop on a broken
                    # subscriber; the broadcast channel is unrecoverable
                    # once the underlying sender is dropped.
                    if not stop.wait(1.0):
                        continue
                    return
                if payload_json is None:
                    continue
                try:
                    data = json.loads(payload_json)
                except (TypeError, ValueError):
                    continue
                try:
                    callback(data)
                except Exception:
                    logger.exception("rust subscriber callback raised for %s", channel)

        thread = threading.Thread(
            target=_run,
            name=f"realtime-ws-sub-rust-{channel}",
            daemon=True,
        )
        thread.start()

        def _unsubscribe() -> None:
            stop.set()
            if thread.is_alive() and threading.current_thread() is not thread:
                thread.join(timeout=2.0)

        return Subscription(
            channel=channel, callback=callback, _unsubscribe=_unsubscribe
        )
