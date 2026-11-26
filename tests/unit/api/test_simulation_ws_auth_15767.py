"""Security contracts for issue #15767.

Symmetric WebSocket auth on ``simulation_ws``, a fail-closed LocalUser ADMIN
bypass, and deprecation (not removal) of the unversioned and ``/api`` REST
prefixes.
"""

from __future__ import annotations

import os
import warnings
from unittest.mock import patch

import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.api.auth.middleware import LocalUser
from src.api.route_registry import ws_compatible_auth_dependency
from src.api.routes import simulation_ws

pytestmark = [pytest.mark.unit, pytest.mark.security]

_REMOTE_ENV = {
    "GOLF_SUITE_MODE": "remote",
    "GOLF_AUTH_DISABLED": "false",
    "ENVIRONMENT": "development",
    "UPSTREAM_DRIFT_ENV": "development",
}
_LOOPBACK_DEV = {
    "GOLF_SUITE_MODE": "remote",
    "GOLF_AUTH_DISABLED": "true",
    "API_HOST": "127.0.0.1",
    "ENVIRONMENT": "development",
    "UPSTREAM_DRIFT_ENV": "development",
}


def _include_mounts(app: FastAPI) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """Return ``(prefix, dependency names, child route paths)`` for each mount.

    FastAPI keeps ``include_router(..., dependencies=)`` on the include
    context rather than copying them onto each child route.
    """
    mounts: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def _walk(router: object, prefix: str) -> None:
        for route in getattr(router, "routes", ()):
            if type(route).__name__ != "_IncludedRouter":
                continue
            context = route.include_context
            child_prefix = f"{prefix}{context.prefix or ''}"
            names = tuple(
                getattr(getattr(dep, "dependency", None), "__name__", "")
                for dep in (context.dependencies or ())
            )
            child_paths = tuple(
                str(getattr(child, "path", ""))
                for child in context.included_router.routes
                if getattr(child, "path", None)
            )
            mounts.append((child_prefix, names, child_paths))
            _walk(context.included_router, child_prefix)

    _walk(app.router, "")
    return mounts


def _mounts_exposing(app: FastAPI, path: str) -> list[tuple[str, tuple[str, ...]]]:
    return [
        (prefix, names)
        for prefix, names, paths in _include_mounts(app)
        if path in paths
    ]


def test_server_mounts_simulation_ws_with_chat_auth_dependency() -> None:
    """Every simulation_ws prefix must carry the same auth dep as chat_ws."""
    from src.api.server import app

    simulated = _mounts_exposing(app, "/ws/simulate/{engine_type}")
    chat = _mounts_exposing(app, "/ws/chat/{session_id}")
    assert {prefix for prefix, _names in simulated} == {"", "/api", "/api/v1"}
    assert {prefix for prefix, _names in chat} == {"", "/api", "/api/v1"}
    assert all("_ws_compatible_auth_dependency" in names for _prefix, names in chat)
    assert all(
        "_ws_compatible_auth_dependency" in names for _prefix, names in simulated
    )


@pytest.mark.parametrize("prefix", ["", "/api", "/api/v1"])
def test_unauthenticated_simulation_ws_rejected_in_remote_mode(prefix: str) -> None:
    """Remote mode closes an unauthenticated simulation socket before accept."""
    app = FastAPI()
    app.include_router(
        simulation_ws.router,
        prefix=prefix,
        dependencies=[Depends(ws_compatible_auth_dependency)],
    )
    client = TestClient(app)
    url = f"{prefix}/ws/simulate/mujoco" if prefix else "/ws/simulate/mujoco"
    with patch.dict(os.environ, _REMOTE_ENV, clear=False):
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect(url):
                pass
    assert caught.value.code == 1008


def test_local_user_keeps_admin_on_loopback_outside_production() -> None:
    """The bypass remains available for a loopback development process."""
    with patch.dict(os.environ, _LOOPBACK_DEV, clear=False):
        user = LocalUser()
    assert user.role == "ADMIN"
    assert user.quota_remaining == float("inf")
    assert user.has_permission("admin") is True


def test_local_user_refuses_admin_off_loopback() -> None:
    """GOLF_AUTH_DISABLED must not mint ADMIN on a public bind address."""
    env = {**_LOOPBACK_DEV, "API_HOST": "0.0.0.0"}
    with patch.dict(os.environ, env, clear=False):
        with pytest.raises(RuntimeError, match="GOLF_AUTH_DISABLED"):
            LocalUser()


def test_local_user_refuses_admin_in_production_on_loopback() -> None:
    """Production refuses the ADMIN bypass even on loopback."""
    env = {**_LOOPBACK_DEV, "ENVIRONMENT": "production"}
    with patch.dict(os.environ, env, clear=False):
        with pytest.raises(RuntimeError, match="GOLF_AUTH_DISABLED"):
            LocalUser()


def test_split_production_env_refuses_bypass() -> None:
    """Either production spelling fails closed (UPSTREAM_DRIFT_ENV)."""
    from src.api.auth.middleware import assert_local_admin_bypass_allowed

    env = {
        **_LOOPBACK_DEV,
        "ENVIRONMENT": "development",
        "UPSTREAM_DRIFT_ENV": "production",
    }
    with patch.dict(os.environ, env, clear=False):
        with pytest.raises(RuntimeError, match="GOLF_AUTH_DISABLED"):
            assert_local_admin_bypass_allowed("127.0.0.1")


def test_auth_enabled_allows_non_loopback_bind() -> None:
    """A public bind is allowed when the ADMIN bypass flag is off."""
    from src.api.auth.middleware import assert_local_admin_bypass_allowed

    env = {
        "GOLF_AUTH_DISABLED": "false",
        "ENVIRONMENT": "production",
        "UPSTREAM_DRIFT_ENV": "production",
    }
    with patch.dict(os.environ, env, clear=False):
        assert_local_admin_bypass_allowed("0.0.0.0")


def test_blank_bind_host_is_rejected() -> None:
    """The startup contract requires a real bind address."""
    from src.api.auth.middleware import assert_local_admin_bypass_allowed

    with pytest.raises(ValueError, match="non-empty"):
        assert_local_admin_bypass_allowed("  ")
    with pytest.raises(TypeError):
        assert_local_admin_bypass_allowed(None)  # type: ignore[arg-type]


def test_server_lifespan_refuses_public_auth_bypass() -> None:
    """Startup raises before serving when the bypass is bound publicly."""
    from src.api.server import app

    env = {**_LOOPBACK_DEV, "API_HOST": "0.0.0.0"}
    with patch.dict(os.environ, env, clear=False):
        with pytest.raises(RuntimeError, match="GOLF_AUTH_DISABLED"):
            with TestClient(app):
                pass


def test_legacy_prefixes_warn_and_canonical_does_not() -> None:
    """Unversioned and /api prefixes warn; /api/v1 does not."""
    from src.api.versioning import warn_deprecated_rest_prefix

    with pytest.warns(DeprecationWarning, match="deprecated"):
        warn_deprecated_rest_prefix("", canonical="/api/v1")
    with pytest.warns(DeprecationWarning, match="/api"):
        warn_deprecated_rest_prefix("/api", canonical="/api/v1")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        warn_deprecated_rest_prefix("/api/v1", canonical="/api/v1")


def test_legacy_prefix_warning_rejects_bad_input() -> None:
    """Deprecation helpers fail fast on a malformed prefix."""
    from src.api.versioning import warn_deprecated_rest_prefix

    with pytest.raises(ValueError, match="absolute"):
        warn_deprecated_rest_prefix("/api", canonical="v1")
    with pytest.raises(TypeError):
        warn_deprecated_rest_prefix(None, canonical="/api/v1")  # type: ignore[arg-type]


def test_legacy_dependency_sets_deprecation_headers() -> None:
    """Legacy mounts keep working and advertise the successor prefix."""
    from src.api.versioning import legacy_rest_deprecation_dependency

    router = APIRouter()

    @router.get("/health")
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    app = FastAPI()
    app.include_router(
        router,
        dependencies=[legacy_rest_deprecation_dependency("/api/v1")],
    )
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["Deprecation"] == "true"
    assert response.headers["Link"] == '</api/v1>; rel="successor-version"'


def test_server_legacy_routes_carry_deprecation_dependency() -> None:
    """The live app deprecates root and /api health without removing them."""
    from src.api.server import app

    mounts = dict(_mounts_exposing(app, "/health"))
    assert set(mounts) >= {"", "/api", "/api/v1"}
    assert "mark_legacy_rest_deprecated" in mounts[""]
    assert "mark_legacy_rest_deprecated" in mounts["/api"]
    assert "mark_legacy_rest_deprecated" not in mounts["/api/v1"]


def test_deprecated_dependency_requires_absolute_successor() -> None:
    """A successor link must be an absolute prefix."""
    from src.api.versioning import legacy_rest_deprecation_dependency

    with pytest.raises(ValueError, match="absolute"):
        legacy_rest_deprecation_dependency("api/v1")
