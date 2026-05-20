"""Launcher-side readiness checks for the Sidekick chat backend."""

from __future__ import annotations

from dataclasses import dataclass
from http.client import HTTPConnection
from typing import Any

from src.api.config import get_server_host, get_server_port

READY_PATH = "/readyz"


@dataclass(frozen=True)
class SidekickApiReadiness:
    """Result of probing the local API process before enabling chat UI."""

    ready: bool
    url: str
    status_code: int | None = None
    detail: str = ""


def sidekick_api_ready_url() -> tuple[str, str, int]:
    """Return the readiness path and host/port used by the API server."""
    host = get_server_host()
    port = get_server_port()
    return READY_PATH, host, port


def check_sidekick_api_readiness(timeout_seconds: float = 0.25) -> SidekickApiReadiness:
    """Probe the local API readiness endpoint without blocking the UI long."""
    try:
        path, host, port = sidekick_api_ready_url()
    except ValueError as exc:
        return SidekickApiReadiness(
            ready=False,
            url=f"http://127.0.0.1:8000{READY_PATH}",
            detail=str(exc),
        )

    url = f"http://{host}:{port}{path}"
    connection: HTTPConnection | None = None
    try:
        connection = HTTPConnection(host, port, timeout=timeout_seconds)
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read(512).decode("utf-8", errors="replace")
        return SidekickApiReadiness(
            ready=response.status == 200,
            url=url,
            status_code=response.status,
            detail=body,
        )
    except OSError as exc:
        return SidekickApiReadiness(ready=False, url=url, detail=str(exc))
    finally:
        if connection is not None:
            connection.close()


def readiness_detail_for_log(readiness: SidekickApiReadiness) -> dict[str, Any]:
    """Return structured details suitable for launcher diagnostics/logging."""
    return {
        "ready": readiness.ready,
        "url": readiness.url,
        "status_code": readiness.status_code,
        "detail": readiness.detail[:300],
    }
