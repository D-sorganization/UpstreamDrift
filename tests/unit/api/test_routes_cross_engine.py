"""Unit tests for the cross-engine robustness comparison API routes (issue #7455).

Tests use FastAPI TestClient with the task manager dependency overridden to a
real in-memory TaskManager (no background-task execution) so we can exercise
the route logic without starting real physics engines.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.cross_engine import router
from src.api.dependencies import get_task_manager
from src.api.task_manager import TaskManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def task_manager() -> TaskManager:
    """Fresh TaskManager for each test."""
    return TaskManager()


@pytest.fixture()
def app(task_manager: TaskManager) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_task_manager] = lambda: task_manager
    return test_app


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STUB_RESULT: dict[str, Any] = {
    "engines": {
        "pendulum_stub": {
            "metrics": {
                "total_energy_final": {
                    "mean": 0.5,
                    "std": 0.05,
                    "cv": 0.1,
                    "robustness_score": 0.9,
                }
            }
        }
    },
    "cv_summary": {"cv_total_energy_final": 0.1},
    "robustness_overall": 0.9,
    "config": {
        "t_end": 1.0,
        "dt": 0.01,
        "noise_amplitude": 0.05,
        "n_trials": 10,
        "seed": 42,
    },
}


# ---------------------------------------------------------------------------
# Tests — POST /analysis/cross-engine
# ---------------------------------------------------------------------------


class TestStartCrossEngineStudy:
    def test_valid_request_returns_task_id(self, client: TestClient) -> None:
        payload = {"engines": ["pendulum_stub"]}
        response = client.post("/analysis/cross-engine", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "started"
        assert len(data["task_id"]) > 0

    def test_multiple_engines_accepted(self, client: TestClient) -> None:
        payload = {
            "engines": ["pendulum_stub", "mujoco"],
            "config": {"n_trials": 5, "t_end": 0.5},
        }
        response = client.post("/analysis/cross-engine", json=payload)
        assert response.status_code == 200, response.text

    def test_unknown_engine_rejected(self, client: TestClient) -> None:
        payload = {"engines": ["not_a_real_engine"]}
        response = client.post("/analysis/cross-engine", json=payload)
        assert response.status_code == 422, response.text

    def test_empty_engines_list_rejected(self, client: TestClient) -> None:
        payload = {"engines": []}
        response = client.post("/analysis/cross-engine", json=payload)
        assert response.status_code == 422, response.text

    def test_task_id_stored_in_task_manager(
        self, client: TestClient, task_manager: TaskManager
    ) -> None:
        payload = {"engines": ["pendulum_stub"]}
        response = client.post("/analysis/cross-engine", json=payload)
        task_id = response.json()["task_id"]
        assert task_manager.exists(task_id)
        task_data = task_manager.get(task_id)
        assert task_data is not None
        assert task_data["engines"] == ["pendulum_stub"]

    def test_custom_config_accepted(self, client: TestClient) -> None:
        payload = {
            "engines": ["pendulum_stub"],
            "config": {
                "t_end": 2.0,
                "dt": 0.005,
                "noise_amplitude": 0.1,
                "n_trials": 20,
                "seed": 99,
            },
        }
        response = client.post("/analysis/cross-engine", json=payload)
        assert response.status_code == 200, response.text

    def test_invalid_config_dt_zero_rejected(self, client: TestClient) -> None:
        payload = {"engines": ["pendulum_stub"], "config": {"dt": 0.0}}
        response = client.post("/analysis/cross-engine", json=payload)
        assert response.status_code == 422, response.text

    def test_invalid_config_n_trials_zero_rejected(self, client: TestClient) -> None:
        payload = {"engines": ["pendulum_stub"], "config": {"n_trials": 0}}
        response = client.post("/analysis/cross-engine", json=payload)
        assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# Tests — GET /analysis/cross-engine/status/{task_id}
# ---------------------------------------------------------------------------


class TestGetCrossEngineStatus:
    def test_unknown_task_returns_404(self, client: TestClient) -> None:
        response = client.get("/analysis/cross-engine/status/does-not-exist")
        assert response.status_code == 404, response.text

    def test_known_task_returns_data(
        self, client: TestClient, task_manager: TaskManager
    ) -> None:
        task_manager.set(
            "test-task-001", {"status": "started", "engines": ["pendulum_stub"]}
        )
        response = client.get("/analysis/cross-engine/status/test-task-001")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "started"

    def test_completed_task_contains_result(
        self, client: TestClient, task_manager: TaskManager
    ) -> None:
        task_manager.set("done-task-002", {"status": "started"})
        task_manager.mark_completed("done-task-002", _STUB_RESULT)
        response = client.get("/analysis/cross-engine/status/done-task-002")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "completed"
        assert "result" in data
        assert data["result"]["robustness_overall"] == pytest.approx(0.9)

    def test_failed_task_contains_error(
        self, client: TestClient, task_manager: TaskManager
    ) -> None:
        task_manager.set("fail-task-003", {"status": "started"})
        task_manager.mark_failed("fail-task-003", "engine exploded")
        response = client.get("/analysis/cross-engine/status/fail-task-003")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "failed"
        assert "engine exploded" in data["error"]


# ---------------------------------------------------------------------------
# Tests — background worker (unit, no real physics)
# ---------------------------------------------------------------------------


class TestRunStudyBackground:
    def test_background_worker_marks_completed_on_success(self) -> None:
        """Worker calls mark_completed with valid result when service succeeds."""
        from src.api.routes.cross_engine import (
            CrossEnginePerturbationConfig,
            CrossEngineStudyRequest,
            _run_study_background,
        )

        tm = TaskManager()
        tm.set("bg-task-001", {"status": "started"})

        request = CrossEngineStudyRequest(
            engines=["pendulum_stub"],
            config=CrossEnginePerturbationConfig(t_end=0.5, n_trials=3),
        )
        # The background worker imports lazily; patch the source module, not the route.
        with (
            patch(
                "src.shared.python.analysis.cross_engine.run_cross_engine_study",
                return_value=_STUB_RESULT,
            ) as mock_study,
            patch(
                "src.shared.python.pendulum_simulator.cross_engine_perturbation.CrossEngineSimConfig"
            ) as mock_cfg,
        ):
            mock_cfg.return_value = object()
            _run_study_background("bg-task-001", request, tm)
            mock_study.assert_called_once()

        data = tm.get("bg-task-001")
        assert data is not None
        assert data["status"] == "completed"
        assert data["result"]["robustness_overall"] == pytest.approx(0.9)

    def test_background_worker_marks_failed_on_value_error(self) -> None:
        """Worker calls mark_failed when service raises ValueError."""
        from src.api.routes.cross_engine import (
            CrossEnginePerturbationConfig,
            CrossEngineStudyRequest,
            _run_study_background,
        )

        tm = TaskManager()
        tm.set("bg-task-002", {"status": "started"})

        request = CrossEngineStudyRequest(
            engines=["pendulum_stub"],
            config=CrossEnginePerturbationConfig(n_trials=2),
        )
        # Patch at the source so the lazy import inside the worker picks it up.
        with (
            patch(
                "src.shared.python.analysis.cross_engine.run_cross_engine_study",
                side_effect=ValueError("simulated failure"),
            ),
            patch(
                "src.shared.python.pendulum_simulator.cross_engine_perturbation.CrossEngineSimConfig"
            ) as mock_cfg,
        ):
            mock_cfg.return_value = object()
            _run_study_background("bg-task-002", request, tm)

        data = tm.get("bg-task-002")
        assert data is not None
        assert data["status"] == "failed"
        assert "simulated failure" in data["error"]
