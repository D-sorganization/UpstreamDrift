"""Tests for the diagnostics parity routes (issue #7458).

Covers:
- ``GET /api/v1/diagnostics/full`` — desktop-grade report served to web UI
- ``GET /api/v1/integrations/health`` — integrations probes with server-side
  secret redaction
- Parity: the probe category set served by the API equals the category set
  the desktop diagnostics dialog renders (both derive from
  ``DIAGNOSTIC_CHECKS``).
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routes.diagnostics as diagnostics_routes
from src.launchers import integrations_health_data
from src.launchers.launcher_diagnostics import (
    DIAGNOSTIC_CHECKS,
    DiagnosticResult,
    LauncherDiagnostics,
)

pytestmark = pytest.mark.unit

FAKE_SECRET = "sk-test-DO-NOT-LEAK-1234567890"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(diagnostics_routes.router, prefix="/api/v1")
    return TestClient(app)


def _stub_check(name: str) -> Any:
    def check(self: LauncherDiagnostics) -> DiagnosticResult:
        result = DiagnosticResult(name=name, status="pass", message="stubbed")
        self.results.append(result)
        return result

    return check


@pytest.fixture
def stubbed_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every probe with a fast stub so endpoint tests stay quick."""
    for name in DIAGNOSTIC_CHECKS:
        monkeypatch.setattr(LauncherDiagnostics, f"check_{name}", _stub_check(name))


class TestFullDiagnosticsEndpoint:
    def test_report_shape(self, stubbed_checks: None) -> None:
        response = _client().get("/api/v1/diagnostics/full")
        assert response.status_code == 200
        body = response.json()
        assert set(body) >= {"summary", "categories", "checks", "recommendations"}
        assert body["summary"]["total_checks"] == len(DIAGNOSTIC_CHECKS)
        assert body["summary"]["status"] == "healthy"

    def test_categories_match_shared_enumeration(self, stubbed_checks: None) -> None:
        body = _client().get("/api/v1/diagnostics/full").json()
        assert body["categories"] == list(DIAGNOSTIC_CHECKS)
        served = {check["name"] for check in body["checks"]}
        assert served == set(DIAGNOSTIC_CHECKS)

    def test_probe_timeout_reported_as_warning(
        self, stubbed_checks: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slow_name = DIAGNOSTIC_CHECKS[0]

        def slow_check(self: LauncherDiagnostics) -> DiagnosticResult:
            time.sleep(1.0)
            return DiagnosticResult(name=slow_name, status="pass", message="late")

        monkeypatch.setattr(LauncherDiagnostics, f"check_{slow_name}", slow_check)
        monkeypatch.setattr(diagnostics_routes, "PROBE_TIMEOUT_SECONDS", 0.05)

        body = _client().get("/api/v1/diagnostics/full").json()
        by_name = {check["name"]: check for check in body["checks"]}
        assert by_name[slow_name]["status"] == "warning"
        assert "timed out" in by_name[slow_name]["message"]

    def test_timed_out_probe_cannot_publish_late_success(
        self, stubbed_checks: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slow_name = DIAGNOSTIC_CHECKS[0]
        instances: list[LauncherDiagnostics] = []
        published: list[tuple[str, str]] = []
        original_init = LauncherDiagnostics.__init__

        def tracking_init(self: LauncherDiagnostics, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            instances.append(self)

        def slow_check(self: LauncherDiagnostics) -> DiagnosticResult:
            time.sleep(0.15)
            result = DiagnosticResult(name=slow_name, status="pass", message="late")
            self._record_result(result)
            return result

        monkeypatch.setattr(LauncherDiagnostics, "__init__", tracking_init)
        monkeypatch.setattr(LauncherDiagnostics, f"check_{slow_name}", slow_check)
        monkeypatch.setattr(
            LauncherDiagnostics,
            "_publish_result",
            staticmethod(lambda result: published.append((result.name, result.status))),
        )
        monkeypatch.setattr(diagnostics_routes, "PROBE_TIMEOUT_SECONDS", 0.05)

        body = _client().get("/api/v1/diagnostics/full").json()
        time.sleep(0.25)

        by_name = {check["name"]: check for check in body["checks"]}
        assert by_name[slow_name]["status"] == "warning"
        assert instances
        recorded = [(result.name, result.status) for result in instances[0].results]
        assert (slow_name, "warning") in recorded
        assert (slow_name, "pass") not in recorded
        assert (slow_name, "warning") in published
        assert (slow_name, "pass") not in published

    def test_hidden_in_production(
        self, stubbed_checks: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(diagnostics_routes, "is_production", lambda: True)
        assert _client().get("/api/v1/diagnostics/full").status_code == 404


class TestDesktopApiParity:
    """Acceptance: API category set == desktop-rendered category set."""

    def test_every_category_has_a_probe_method(self) -> None:
        for name in DIAGNOSTIC_CHECKS:
            assert callable(getattr(LauncherDiagnostics, f"check_{name}", None)), (
                f"DIAGNOSTIC_CHECKS entry {name!r} has no "
                f"LauncherDiagnostics.check_{name} method"
            )

    def test_desktop_report_uses_same_enumeration(self, stubbed_checks: None) -> None:
        """run_all_checks (desktop dialog source) iterates DIAGNOSTIC_CHECKS."""
        desktop_report = LauncherDiagnostics().run_all_checks()
        desktop_names = [check["name"] for check in desktop_report["checks"]]
        assert desktop_names == list(DIAGNOSTIC_CHECKS)
        assert desktop_report["categories"] == list(DIAGNOSTIC_CHECKS)

    def test_api_and_desktop_serve_identical_category_sets(
        self, stubbed_checks: None
    ) -> None:
        api_body = _client().get("/api/v1/diagnostics/full").json()
        desktop_report = LauncherDiagnostics().run_all_checks()
        api_names = {check["name"] for check in api_body["checks"]}
        desktop_names = {check["name"] for check in desktop_report["checks"]}
        assert api_names == desktop_names == set(DIAGNOSTIC_CHECKS)


class TestIntegrationsHealthEndpoint:
    def test_report_shape_and_taxonomy(self) -> None:
        response = _client().get("/api/v1/integrations/health")
        assert response.status_code == 200
        body = response.json()
        assert set(body) >= {"generated_at", "records", "markdown"}
        assert body["markdown"].startswith("# Integration Health")
        valid = {
            "healthy",
            "configured",
            "warning",
            "error",
            "unconfigured",
            "unknown",
        }
        assert body["records"], "expected at least one integration record"
        for record in body["records"]:
            assert record["status"] in valid
            assert set(record) == {
                "kind",
                "name",
                "status",
                "last_checked",
                "last_error",
                "detail",
            }

    def test_secret_env_value_never_appears_in_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_SECRET)
        response = _client().get("/api/v1/integrations/health")
        assert response.status_code == 200
        assert FAKE_SECRET not in response.text
        anthropic = next(
            record
            for record in response.json()["records"]
            if record["kind"] == "api" and record["name"] == "anthropic"
        )
        assert anthropic["status"] == "configured"

    def test_hidden_in_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(diagnostics_routes, "is_production", lambda: True)
        assert _client().get("/api/v1/integrations/health").status_code == 404


class TestRedaction:
    def test_redact_secrets_strips_bearer_tokens(self) -> None:
        scrubbed = integrations_health_data.redact_secrets(
            "Authorization: Bearer abc.def.ghi done"
        )
        assert "abc.def.ghi" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_redact_secrets_strips_provider_env_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_SECRET)
        scrubbed = integrations_health_data.redact_secrets(
            f"probe failed using key {FAKE_SECRET} (HTTP 401)"
        )
        assert FAKE_SECRET not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_redact_secrets_none_returns_empty(self) -> None:
        assert integrations_health_data.redact_secrets(None) == ""

    def test_record_to_redacted_dict_scrubs_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_SECRET)
        record = integrations_health_data.IntegrationRecord(
            kind="api",
            name="gemini",
            status="error",
            last_error=f"401 with key {FAKE_SECRET}",
            detail=f"Bearer {FAKE_SECRET}",
        )
        payload = integrations_health_data.record_to_redacted_dict(record)
        assert FAKE_SECRET not in str(payload)
        assert payload["status"] == "error"
        assert payload["last_checked"] is None

    def test_record_to_redacted_dict_rejects_non_record(self) -> None:
        with pytest.raises(TypeError):
            integrations_health_data.record_to_redacted_dict({"kind": "api"})  # type: ignore[arg-type]
