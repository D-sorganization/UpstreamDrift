"""Regression tests for issue #7987 — unauthenticated capability install.

``POST /capabilities/{name}/install`` runs ``pip install`` in a subprocess and
echoes the output back to the caller. It was declared with
``dependencies=[Depends(OptionalAuth)]`` — passing the *class* rather than an
instance, so FastAPI treated ``OptionalAuth`` as a callable dependency,
instantiated it, and never invoked ``__call__``. The endpoint was therefore
reachable with no ``Authorization`` header in cloud mode, and the constructor's
``auto_error`` argument leaked into the OpenAPI schema as a query parameter.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import capabilities

pytestmark = pytest.mark.unit

# The API surface is where FastAPI dependencies are declared; scanning all of
# ``src`` costs ~16s for no extra coverage.
_API_ROOT = Path(__file__).resolve().parents[3] / "src" / "api"

_MUTATING_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("/capabilities/mujoco/install", "post"),
    ("/capabilities/refresh", "post"),
)


@pytest.fixture
def cloud_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client for the capabilities router with cloud-mode auth enabled."""
    monkeypatch.setenv("GOLF_SUITE_MODE", "cloud")
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "false")

    app = FastAPI()
    app.include_router(capabilities.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("path", [path for path, _ in _MUTATING_ENDPOINTS])
def test_mutating_capability_endpoints_reject_anonymous_callers(
    cloud_client: TestClient, path: str
) -> None:
    """No ``Authorization`` header in cloud mode must be a 401, not a 200."""
    response = cloud_client.post(path, json={"dry_run": True})

    assert response.status_code == 401, (
        f"{path} accepted an unauthenticated request "
        f"(status {response.status_code}); the install/refresh endpoints must "
        "never run unauthenticated in cloud mode"
    )


def test_install_rejects_malformed_bearer_token(cloud_client: TestClient) -> None:
    """A forged/garbage bearer token must not be accepted either."""
    response = cloud_client.post(
        "/capabilities/mujoco/install",
        json={"dry_run": True},
        headers={"Authorization": "Bearer not-a-real-token"},
    )

    assert response.status_code == 401


def test_install_is_reachable_when_auth_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local mode keeps working — the gate is cloud-mode only."""
    monkeypatch.setenv("GOLF_SUITE_MODE", "local")
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "true")

    app = FastAPI()
    app.include_router(capabilities.router)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/capabilities/mujoco/install", json={"dry_run": True, "timeout_seconds": 5}
    )

    assert response.status_code == 200


def test_auto_error_is_not_exposed_as_a_query_parameter() -> None:
    """``OptionalAuth.__init__``'s ``auto_error`` must not leak into the schema."""
    app = FastAPI()
    app.include_router(capabilities.router)

    schema = app.openapi()
    parameters = schema["paths"]["/capabilities/{name}/install"]["post"].get(
        "parameters", []
    )
    names = {param["name"] for param in parameters}

    assert "auto_error" not in names, (
        "'auto_error' in the schema means a security class was passed to "
        "Depends() without being instantiated"
    )


def _depends_class_violations(path: Path) -> list[str]:
    """Return ``Depends(SomeClass)`` call sites that pass a bare name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[str] = []

    security_class_names = {"OptionalAuth", "HTTPBearer", "APIKeyHeader", "OAuth2"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "Depends"):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Name) and arg.id in security_class_names:
            violations.append(f"{path}:{node.lineno}: Depends({arg.id})")

    return violations


def test_no_security_class_is_passed_to_depends_uninstantiated() -> None:
    """Architecture guard: ``Depends(OptionalAuth)`` silently disables auth.

    FastAPI instantiates any callable passed to ``Depends``. Handing it a
    security *class* produces an unconfigured instance whose ``__call__`` is
    never awaited, so the dependency becomes a no-op. The instance form
    ``Depends(OptionalAuth())`` is required.
    """
    violations: list[str] = []
    for path in _API_ROOT.rglob("*.py"):
        violations.extend(_depends_class_violations(path))

    assert not violations, (
        "security classes passed to Depends() without instantiation "
        f"(auth is silently disabled at these sites): {violations}"
    )
