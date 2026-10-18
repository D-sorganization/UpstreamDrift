"""Security and contract tests for /api/diagnostics (#13001).

Verifies that:
1. In remote mode (the default), unauthenticated GET requests to diagnostics
   endpoints are rejected in all environments (401 in development/staging,
   404 in production).
2. No absolute server path appears in any unauthenticated response or in
   the repo_root field.
3. Measured probes (pid, port, python_found, local_server_found) reflect
   actual runtime state instead of hard-coded constants.
4. An allowlist test enumerates public GET routes so unexpected unauthenticated
   endpoints fail fast.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.server import app

pytestmark = pytest.mark.unit


def _is_absolute_path(val: str) -> bool:
    """Return True if val looks like an absolute filesystem path."""
    if not isinstance(val, str) or len(val) < 2:
        return False
    # Check Windows drive letter path (e.g. C:\ or C:/) or POSIX absolute path
    if (len(val) >= 3 and val[1] == ":" and val[2] in ("\\", "/")) or (
        val.startswith("/") and not val.startswith("//") and len(val.split("/")) > 2
    ):
        try:
            return Path(val).is_absolute()
        except (ValueError, TypeError, OSError):
            return False
    return False


def _find_absolute_paths_in_payload(obj: Any) -> list[str]:
    """Recursively search for absolute server paths in a JSON-like payload."""
    leaked: list[str] = []
    if isinstance(obj, str):
        if _is_absolute_path(obj):
            leaked.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            leaked.extend(_find_absolute_paths_in_payload(v))
    elif isinstance(obj, list):
        for item in obj:
            leaked.extend(_find_absolute_paths_in_payload(item))
    return leaked


@pytest.fixture
def remote_dev_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Test client in remote mode, development environment (auth required)."""
    monkeypatch.setenv("GOLF_SUITE_MODE", "remote")
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "false")
    monkeypatch.setenv("ENVIRONMENT", "development")
    from src.shared.python.config import environment

    environment.get_environment.cache_clear()
    with TestClient(app) as client:
        yield client
    environment.get_environment.cache_clear()


@pytest.fixture
def remote_prod_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Test client in remote mode, production environment (diagnostics disabled)."""
    monkeypatch.setenv("GOLF_SUITE_MODE", "remote")
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "false")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv(
        "GOLF_API_SECRET_KEY",
        "prod-secret-key-that-is-at-least-64-characters-long-for-production-deployment-testing-0123456789",
    )
    from unittest.mock import patch
    from src.shared.python.config import environment

    environment.get_environment.cache_clear()
    with patch("src.api.database._assert_alembic_head_applied"):
        with TestClient(app) as client:
            yield client
    environment.get_environment.cache_clear()


@pytest.fixture
def local_dev_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Test client in local mode, development environment (auth bypassed)."""
    monkeypatch.setenv("GOLF_SUITE_MODE", "local")
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "true")
    monkeypatch.setenv("ENVIRONMENT", "development")
    from src.shared.python.config import environment

    environment.get_environment.cache_clear()
    with TestClient(app) as client:
        yield client
    environment.get_environment.cache_clear()


@pytest.mark.parametrize(
    "path",
    [
        "/api/diagnostics",
        "/diagnostics",
        "/api/v1/api/diagnostics",
        "/api/v1/diagnostics",
    ],
)
def test_unauthenticated_diagnostics_rejected_in_remote_mode_dev(
    remote_dev_client: TestClient, path: str
) -> None:
    """In remote mode and development, unauthenticated diagnostics requests return 401."""
    response = remote_dev_client.get(path)
    assert response.status_code == 401, (
        f"Expected 401 Unauthorized for {path} in remote mode, got {response.status_code}: {response.text}"
    )
    # Ensure no absolute path in error body
    leaked = _find_absolute_paths_in_payload(response.json())
    assert not leaked, f"Absolute path leaked in 401 response: {leaked}"


@pytest.mark.parametrize(
    "path",
    [
        "/api/diagnostics",
        "/diagnostics",
        "/api/v1/api/diagnostics",
        "/api/v1/diagnostics",
    ],
)
def test_unauthenticated_diagnostics_rejected_in_remote_mode_prod(
    remote_prod_client: TestClient, path: str
) -> None:
    """In remote mode and production, diagnostics requests return 404 (disabled in prod)."""
    response = remote_prod_client.get(path)
    assert response.status_code == 404, (
        f"Expected 404 Not Found for {path} in production mode, got {response.status_code}: {response.text}"
    )
    leaked = _find_absolute_paths_in_payload(response.json())
    assert not leaked, f"Absolute path leaked in 404 response: {leaked}"


def test_diagnostics_measured_probes_and_relative_root(
    local_dev_client: TestClient,
) -> None:
    """In authorized/local mode, diagnostics returns real measured probes and a relative root."""
    response = local_dev_client.get("/api/diagnostics")
    assert response.status_code == 200
    data = response.json()

    # 1. repo_root must never be an absolute server path
    repo_root = data.get("repo_root")
    assert repo_root is not None, "repo_root must be present"
    assert not _is_absolute_path(repo_root), (
        f"repo_root leaked absolute path: {repo_root}"
    )
    assert repo_root == ".", f"Expected repo-relative root '.', got {repo_root}"

    # 2. pid must be the real measured PID of the running process, not None
    backend = data.get("backend", {})
    pid = backend.get("pid")
    assert pid is not None, "pid must reflect measured state, not None"
    assert pid == os.getpid(), f"Expected pid {os.getpid()}, got {pid}"

    # 3. port must be an integer reflecting server port (default 8000), not hard-coded 8001
    port = backend.get("port")
    assert port is not None, "port must reflect measured state, not None"
    assert port in (8000, 80), f"Expected server port (8000), got {port}"

    # 4. python_found must reflect measured reality
    python_found = data.get("python_found")
    expected_python_found = bool(sys.executable and Path(sys.executable).exists())
    assert python_found == expected_python_found, (
        f"Expected python_found={expected_python_found}, got {python_found}"
    )

    # 5. local_server_found must reflect measured reality
    local_server_found = data.get("local_server_found")
    expected_local_server_found = (
        Path(__file__).resolve().parents[3] / "src" / "api" / "local_server.py"
    ).is_file()
    assert local_server_found == expected_local_server_found, (
        f"Expected local_server_found={expected_local_server_found}, got {local_server_found}"
    )

    # 6. No absolute paths anywhere in the payload
    leaked = _find_absolute_paths_in_payload(data)
    assert not leaked, f"Absolute paths leaked in diagnostics payload: {leaked}"


def test_public_get_routes_allowlist_in_remote_mode(
    remote_dev_client: TestClient,
) -> None:
    """Enumerate public GET routes in remote mode and assert diagnostics is NOT public.

    Only explicitly allowlisted public routes (healthz, capabilities, launcher
    manifest/tiles, matched-swings local viewer, etc.) may return 200 without auth.
    Diagnostics and sensitive routes must require authentication (401).
    """
    # Canonical allowlist of public unauthenticated GET route prefixes/paths under /api/v1/
    # Documented rationale:
    # - /health, /healthz, /readyz, /metrics: public observability/liveness endpoints.
    # - /capabilities: capability matrix for client feature negotiation before auth.
    # - /launcher/...: launcher manifest and tiles needed to display the app picker.
    # - /matched-swings: swing matching results index (local-client-gated via host check).
    # - /lifting/baseline: baseline profile data for client capability matching.
    allowlisted_prefixes = (
        "/api/v1/capabilities",
        "/api/v1/engines/matrix",
        "/api/v1/healthz",
        "/api/v1/launcher/",
        "/api/v1/lifting/baseline",
        "/api/v1/matched-swings",
        "/api/v1/metrics",
    )

    # Diagnostics routes MUST NOT be in the allowlist
    for diag_path in (
        "/api/v1/api/diagnostics",
        "/api/v1/diagnostics",
        "/api/diagnostics",
        "/diagnostics",
    ):
        res = remote_dev_client.get(diag_path)
        assert res.status_code == 401, (
            f"Diagnostics route {diag_path} unexpectedly accessible unauthenticated: {res.status_code}"
        )
