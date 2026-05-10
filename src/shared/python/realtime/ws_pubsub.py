"""WebSocket-based publish/subscribe backend.

Uses the FastAPI server in :mod:`src.api.routes.realtime`. If no server is
already listening on the configured port, a minimal one is spawned in a
daemon thread.

Reconnection is exponential (1s, 2s, 4s, ..., capped at 30s).

Latency budget: < 50 ms one-hop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from collections.abc import Callable

from .protocol import Subscription, validate_channel

__all__ = ["WSPubSub"]


logger = logging.getLogger(__name__)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


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
        """Sleep for the current delay, then double it. Returns False if
        ``stop`` was set during the wait."""
        ok = not stop.wait(self._delay)
        self._delay = min(self._delay * 2.0, self._cap)
        return ok


class WSPubSub:
    """WebSocket pub-sub backend that talks to ``/realtime/*`` endpoints.

    Args:
        host: Host of the FastAPI server. Defaults to 127.0.0.1.
        port: Port of the FastAPI server. Defaults to 8765.
        autostart: If True, spawn a minimal FastAPI server on the configured
            port if nothing is listening. Defaults to True.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        autostart: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self._server_thread: threading.Thread | None = None
        if autostart and not _port_in_use(host, port):
            self._spawn_server()

    # -- server bootstrap ----------------------------------------------------

    def _spawn_server(self) -> None:
        """Spawn a minimal FastAPI app exposing the realtime routes."""
        try:
            import uvicorn
            from fastapi import FastAPI
        except Exception:
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
        try:
            import httpx
        except Exception as exc:
            raise RuntimeError("httpx is required for WS pub-sub publish") from exc

        body = {"channel": channel, "payload": payload}
        with httpx.Client(timeout=2.0) as client:
            r = client.post(self._publish_url(), json=body)
            r.raise_for_status()

    # -- subscribe -----------------------------------------------------------

    def subscribe(
        self,
        channel: str,
        callback: Callable[[dict], None],
    ) -> Subscription:
        validate_channel(channel)
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
                    except Exception as exc:
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
