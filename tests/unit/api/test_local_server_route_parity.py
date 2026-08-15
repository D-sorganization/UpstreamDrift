"""Desktop (Tauri) vs web API route-parity tests (issue #8010).

``src/api/local_server.py`` is the backend the packaged Tauri desktop app
serves the React bundle from. It used to hand-mount a fixed subset of routers,
while ``npm run dev`` proxies to ``src.api.server`` — which mounts every
discovered router. The result: ~100 endpoints the React UI calls returned 404 in
the packaged app but worked fine in development, so the gap was invisible during
normal frontend work.

Both entry points now share ``route_registry.register_routes``. These tests pin
that: any endpoint reachable under ``/api/`` on the web server must be reachable
on the desktop server, and the desktop-only overrides must not be shadowed by
the shared routers.
"""

from __future__ import annotations

from collections import defaultdict

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

pytestmark = pytest.mark.unit

_SKIP_METHODS = frozenset({"HEAD", "OPTIONS"})


def _route_table(app: FastAPI) -> dict[tuple[str, str], list[str]]:
    table: dict[tuple[str, str], list[str]] = defaultdict(list)
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in sorted(route.methods or ()):
            if method in _SKIP_METHODS:
                continue
            endpoint = route.endpoint
            qualname = getattr(endpoint, "__qualname__", str(endpoint))
            table[(method, route.path)].append(f"{endpoint.__module__}.{qualname}")
    return table


@pytest.fixture(scope="module")
def local_app() -> FastAPI:
    from src.api.local_server import create_local_app

    return create_local_app()


@pytest.fixture(scope="module")
def web_app() -> FastAPI:
    from src.api.server import app

    return app


def test_desktop_serves_every_web_api_endpoint(
    local_app: FastAPI, web_app: FastAPI
) -> None:
    """No ``/api/...`` endpoint exists on the web server but not the desktop one."""
    local_paths = set(_route_table(local_app))
    missing = sorted(
        key
        for key in _route_table(web_app)
        if key[1].startswith("/api/") and key not in local_paths
    )
    assert not missing, (
        f"{len(missing)} endpoints 404 in the packaged desktop app but work "
        "under `npm run dev` (issue #8010):\n"
        + "\n".join(f"  {method} {path}" for method, path in missing)
    )


def test_desktop_mounts_both_versioned_and_legacy_prefixes(
    local_app: FastAPI,
) -> None:
    """Every router is reachable at ``/api/v1/`` and the legacy ``/api/``."""
    paths = {path for _method, path in _route_table(local_app)}
    for suffix in (
        "engines",
        "dataset/features",
        "simulation/actuators",
        "terrain/presets",
        "models",
    ):
        assert f"/api/{suffix}" in paths, f"legacy /api/{suffix} not mounted"
        assert f"/api/v1/{suffix}" in paths, f"versioned /api/v1/{suffix} not mounted"


@pytest.mark.parametrize(
    ("method", "path", "expected_module"),
    [
        ("GET", "/api/health", "src.api.local_server"),
        ("GET", "/api/launcher/manifest", "src.api.local_server"),
    ],
)
def test_desktop_overrides_win_first_match(
    local_app: FastAPI, method: str, path: str, expected_module: str
) -> None:
    """Desktop-specific handlers must be registered ahead of the shared routers.

    The local launcher manifest carries the launcher CSRF token and
    native-window state the desktop shell needs; the local health check reports
    the in-process engine manager. Registering the routers first would make both
    unreachable.
    """
    handlers = _route_table(local_app)[(method, path)]
    assert handlers, f"{method} {path} is not registered"
    assert handlers[0].startswith(expected_module), (
        f"{method} {path} resolves to {handlers[0]}; the desktop override is "
        "shadowed by a shared router"
    )
