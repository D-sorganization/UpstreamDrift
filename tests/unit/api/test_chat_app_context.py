"""Tests for the shared chat app-context schema and providers (issue #7453).

Parity contract: the desktop chat provider (``agent_context``, wired in
issue #5470) and the API-server provider (``make_app_state_provider``) must
emit the same schema. The desktop PyQt6 Sidekick sidebar is not reachable
headlessly, so parity is asserted against the provider *functions* both
wirings install plus the shared :class:`ChatAppContext` Pydantic model.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.api.services.chat_app_context import (
    LEGACY_CONTEXT_KEYS,
    ChatAppContext,
    SimulationRunContext,
    build_chat_app_context,
    make_app_state_provider,
)
from src.shared.python.app_state import HistoryStore, agent_context

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubEngineManager:
    def get_engine_info(self) -> dict[str, Any]:
        return {
            "current_engine": "mujoco",
            "available_engines": ["mujoco", "pendulum"],
            "engine_status": {"mujoco": "available"},
        }


class _BrokenEngineManager:
    def get_engine_info(self) -> dict[str, Any]:
        raise RuntimeError("engine registry exploded")


class _StubStats:
    def __init__(self, last_run: dict[str, Any] | None) -> None:
        self.last_run = last_run


class _StubSimulationService:
    def __init__(self, last_run: dict[str, Any] | None) -> None:
        self.stats = _StubStats(last_run)


_LAST_RUN = {
    "engine": "mujoco",
    "model": "golf_swing.urdf",
    "duration_seconds": 3.0,
    "status": "completed",
    "frames": 3000,
    "finished_at": "2026-06-12T00:00:00Z",
    "error": None,
    "analysis_summary": "analysis: kinematics",
}


def _store_with_event() -> HistoryStore:
    store = HistoryStore()
    store.append_event("simulation_run", {"engine": "mujoco"})
    return store


# ---------------------------------------------------------------------------
# Schema parity (desktop vs API)
# ---------------------------------------------------------------------------


class TestSchemaParity:
    def test_api_context_keys_superset_of_desktop_provider_keys(self) -> None:
        """Desktop provider (#5470) keys must all exist in ChatAppContext."""
        desktop_keys = set(agent_context(_store_with_event()))
        api_keys = set(build_chat_app_context(store=_store_with_event()).model_dump())
        assert desktop_keys <= api_keys
        assert desktop_keys == set(LEGACY_CONTEXT_KEYS)

    def test_server_and_local_server_providers_emit_identical_keys(self) -> None:
        """Both server wirings use make_app_state_provider — same keys."""
        store = _store_with_event()
        # Mirrors src/api/server.py wiring.
        cloud_provider = make_app_state_provider(
            lambda: _StubEngineManager(),
            lambda: _StubSimulationService(_LAST_RUN),
            lambda: store,
        )
        # Mirrors src/api/local_server.py wiring (lazy suppliers).
        local_provider = make_app_state_provider(
            lambda: _StubEngineManager(),
            lambda: _StubSimulationService(None),
            lambda: store,
        )
        assert set(cloud_provider()) == set(local_provider())

    def test_provider_output_validates_against_shared_schema(self) -> None:
        provider = make_app_state_provider(
            lambda: _StubEngineManager(),
            lambda: _StubSimulationService(_LAST_RUN),
            lambda: _store_with_event(),
        )
        ctx = ChatAppContext.model_validate(provider())
        assert ctx.active_engine == "mujoco"

    def test_desktop_payload_validates_against_shared_schema(self) -> None:
        """Legacy agent_context payload is accepted by the shared model."""
        ctx = ChatAppContext.model_validate(agent_context(_store_with_event()))
        assert ctx.events
        assert ctx.summary


# ---------------------------------------------------------------------------
# Builder behaviour
# ---------------------------------------------------------------------------


class TestBuildChatAppContext:
    def test_fills_engine_and_simulation_fields(self) -> None:
        ctx = build_chat_app_context(
            engine_manager=_StubEngineManager(),
            simulation_service=_StubSimulationService(_LAST_RUN),
            store=_store_with_event(),
        )
        assert ctx.engines_loaded == ["mujoco", "pendulum"]
        assert ctx.active_engine == "mujoco"
        assert ctx.active_model == "golf_swing.urdf"
        assert ctx.simulation is not None
        assert ctx.simulation.status == "completed"
        assert ctx.simulation.duration_seconds == 3.0
        assert ctx.analysis_summary == "analysis: kinematics"
        assert ctx.events  # legacy keys still populated

    def test_degrades_when_sources_missing(self) -> None:
        ctx = build_chat_app_context(store=_store_with_event())
        assert ctx.engines_loaded == []
        assert ctx.active_engine is None
        assert ctx.simulation is None

    def test_degrades_when_engine_manager_raises(self) -> None:
        ctx = build_chat_app_context(
            engine_manager=_BrokenEngineManager(),
            store=_store_with_event(),
        )
        assert ctx.engines_loaded == []
        assert ctx.active_engine is None

    def test_rejects_non_positive_max_events(self) -> None:
        with pytest.raises(ValueError, match="max_events"):
            build_chat_app_context(store=_store_with_event(), max_events=0)

    def test_provider_factory_rejects_non_callable_supplier(self) -> None:
        with pytest.raises(TypeError, match="engine_manager_supplier"):
            make_app_state_provider("nope", None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SimulationService last-run tracking
# ---------------------------------------------------------------------------


class TestSimulationServiceLastRun:
    def _service(self) -> Any:
        from unittest.mock import MagicMock

        from src.api.services.simulation_service import SimulationService

        return SimulationService(MagicMock())

    def _request(self) -> Any:
        from unittest.mock import MagicMock

        request = MagicMock()
        request.engine_type = "MUJOCO"
        request.model_path = "/tmp/models/golf_swing.urdf"
        request.duration = 3.0
        return request

    def test_begin_last_run_records_running_state(self) -> None:
        svc = self._service()
        svc._begin_last_run(self._request())
        last_run = svc.stats.last_run
        assert last_run is not None
        assert last_run["status"] == "running"
        assert last_run["engine"] == "mujoco"
        # Only the basename — never the full path (privacy).
        assert last_run["model"] == "golf_swing.urdf"
        assert "/tmp" not in str(last_run["model"])

    def test_finish_last_run_marks_completed(self) -> None:
        svc = self._service()
        svc._begin_last_run(self._request())
        svc._finish_last_run(
            status="completed", frames=3000, analysis_summary="analysis: kinematics"
        )
        last_run = svc.stats.last_run
        assert last_run["status"] == "completed"
        assert last_run["frames"] == 3000
        assert last_run["finished_at"]
        assert SimulationRunContext.model_validate(last_run).status == "completed"

    def test_finish_last_run_marks_failure(self) -> None:
        svc = self._service()
        svc._begin_last_run(self._request())
        svc._finish_last_run(status="failed", error="boom")
        assert svc.stats.last_run["status"] == "failed"
        assert svc.stats.last_run["error"] == "boom"

    def test_finish_without_begin_is_noop(self) -> None:
        svc = self._service()
        svc._finish_last_run(status="completed")
        assert svc.stats.last_run is None


# ---------------------------------------------------------------------------
# GET /chat/context endpoint
# ---------------------------------------------------------------------------


class TestChatContextEndpoint:
    def _client(self, with_services: bool) -> Any:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.routes import chat_ws

        app = FastAPI()
        app.include_router(chat_ws.router)
        if with_services:
            app.state.engine_manager = _StubEngineManager()
            app.state.simulation_service = _StubSimulationService(_LAST_RUN)
        return TestClient(app)

    def test_returns_schema_with_live_services(self) -> None:
        resp = self._client(with_services=True).get("/chat/context")
        assert resp.status_code == 200
        payload = resp.json()
        ctx = ChatAppContext.model_validate(payload)
        assert ctx.active_engine == "mujoco"
        assert ctx.active_model == "golf_swing.urdf"
        assert set(LEGACY_CONTEXT_KEYS) <= set(payload)

    def test_degrades_without_services(self) -> None:
        resp = self._client(with_services=False).get("/chat/context")
        assert resp.status_code == 200
        ctx = ChatAppContext.model_validate(resp.json())
        assert ctx.active_engine is None
        assert ctx.simulation is None
