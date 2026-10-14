"""Bounded, cancellable BunkerShot3D workbench jobs (issue #9545, slice 4).

Acceptance: a long solve leaves the client responsive, reports progress,
cancels and releases its worker; a failed or cancelled run can never be
shown as a fresh result. The job routes reuse the shared
:class:`~src.api.task_manager.TaskManager` for the job lifecycle, so these
tests pin:

* submit returns a job id, the shot budget and an input digest at once,
  and every refusal of the synchronous route is a 422 before any job exists;
* a completed job carries the job id and the input digest beside the same
  record the synchronous route returns;
* a cancelled job stops before its next shot, releases its worker and never
  gains a result, even when the shot in flight finishes after the cancel;
* a failed job reports why and carries no result; and
* concurrency is bounded: a submit beyond the worker slots is a 429.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.routes import bunker_workbench_jobs as jobs  # noqa: E402
from src.api.routes.bunker_workbench import (  # noqa: E402
    BunkerWorkbenchRequestV1,
    router as sync_router,
)
from src.api.task_manager import TaskManager  # noqa: E402
from src.tools.bunker_shot_gui.model import WorkbenchModel  # noqa: E402

pytestmark = [pytest.mark.contract, pytest.mark.integration]

PREFIX = "/tools/bunker-workbench/v1"
SUBMIT_EVALUATE = f"{PREFIX}/jobs/evaluate"
SUBMIT_COMPARE = f"{PREFIX}/jobs/compare"

NOMINAL: dict[str, Any] = {
    "handedness": "right",
    "design": {"name": "A", "grind_preset": "sm9_58_m"},
    "sand": {"preset": "firm"},
    "swing": {"clubhead_speed_mps": 25.0, "attack_angle_deg": -6.0},
}

PAIR: dict[str, Any] = {
    "handedness": "right",
    "left": {"name": "A", "grind_preset": "sm9_58_m"},
    "right": {"name": "B", "grind_preset": "sm9_58_m", "marketed_bounce_deg": 13.0},
    "sand": {"preset": "firm"},
    "swing": {"clubhead_speed_mps": 25.0, "attack_angle_deg": -6.0},
}

WAIT_S = 60.0


class _Gate:
    """Holds every shot until released, and counts the shots that began."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls = 0


@pytest.fixture(autouse=True)
def _drained_pool() -> Iterator[None]:
    """Each test starts and ends with every worker slot free."""
    yield
    assert jobs.JOB_POOL.drain(WAIT_S), "a bunker job worker never exited"


@pytest.fixture
def client() -> Iterator[TestClient]:
    """The sync and job routers over a fresh shared task manager."""
    app = FastAPI()
    app.include_router(sync_router)
    app.include_router(jobs.router)
    app.state.task_manager = TaskManager()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Gate]:
    """Every nominal shot waits for ``gate.release`` before it is solved."""
    held = _Gate()
    real = WorkbenchModel.shot_result

    def shot_result(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        held.calls += 1
        assert held.release.wait(WAIT_S), "the test never released the shot"
        return real(self, *args, **kwargs)

    monkeypatch.setattr(WorkbenchModel, "shot_result", shot_result)
    yield held
    held.release.set()


def _status(client: TestClient, job_id: str) -> dict[str, Any]:
    response = client.get(f"{PREFIX}/jobs/{job_id}")
    assert response.status_code == 200, response.text
    return response.json()


def _wait_for(client: TestClient, job_id: str, predicate: Any) -> dict[str, Any]:
    """Poll the job until ``predicate(status)`` holds."""
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        body = _status(client, job_id)
        if predicate(body):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached the expected state: {body}")


def _settled(body: dict[str, Any]) -> bool:
    return (
        body["status"] in {"completed", "failed", "cancelled"}
        and not body["worker_active"]
    )


def _submit(client: TestClient, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(path, json=payload)
    assert response.status_code == 202, response.text
    return response.json()


def _detail_code(response: Any) -> str:
    return str(response.json()["detail"]["code"])


# ---------------------------------------------------------------- submit


def test_the_job_routes_are_versioned_under_the_tools_prefix() -> None:
    paths = {route.path for route in jobs.router.routes}  # type: ignore[attr-defined]
    assert {
        SUBMIT_EVALUATE,
        SUBMIT_COMPARE,
        f"{PREFIX}/jobs/{{job_id}}",
        f"{PREFIX}/jobs/{{job_id}}/cancel",
    } <= paths


def test_submit_answers_at_once_with_the_shot_budget_and_input_digest(
    client: TestClient, gate: _Gate
) -> None:
    body = _submit(client, SUBMIT_COMPARE, PAIR)
    assert re.fullmatch(r"[0-9a-f-]{36}", body["job_id"])
    assert re.fullmatch(r"[0-9a-f]{64}", body["input_digest"])
    assert body["kind"] == "compare"
    assert body["status"] in {"pending", "running"}
    assert body["shots_planned"] == 2 * (1 + 5**2)
    assert body["result"] is None
    client.post(f"{PREFIX}/jobs/{body['job_id']}/cancel")


def test_the_input_digest_names_the_request_not_the_run() -> None:
    request = BunkerWorkbenchRequestV1.model_validate(NOMINAL)
    other = BunkerWorkbenchRequestV1.model_validate(
        {**NOMINAL, "swing": {"clubhead_speed_mps": 24.0}}
    )
    assert jobs.input_digest("evaluate", request) == jobs.input_digest(
        "evaluate", BunkerWorkbenchRequestV1.model_validate(NOMINAL)
    )
    assert jobs.input_digest("evaluate", request) != jobs.input_digest(
        "evaluate", other
    )
    assert jobs.input_digest("evaluate", request) != jobs.input_digest(
        "compare", request
    )


@pytest.mark.parametrize(
    ("path", "payload", "code"),
    [
        (
            SUBMIT_EVALUATE,
            {**NOMINAL, "handedness": "left"},
            "handedness_unsupported",
        ),
        (
            SUBMIT_COMPARE,
            {**PAIR, "right": {"name": "A"}},
            "design_names_not_distinct",
        ),
    ],
)
def test_a_refused_request_is_a_422_before_any_job_exists(
    client: TestClient, path: str, payload: dict[str, Any], code: str
) -> None:
    response = client.post(path, json=payload)
    assert response.status_code == 422
    assert _detail_code(response) == code
    assert client.app.state.task_manager.active_count() == 0  # type: ignore[attr-defined]


def test_an_unknown_job_is_a_404(client: TestClient) -> None:
    assert (
        client.get(f"{PREFIX}/jobs/{'0' * 8}-0000-0000-0000-{'0' * 12}").status_code
        == 404
    )
    assert client.get(f"{PREFIX}/jobs/not-a-job").status_code == 404


# -------------------------------------------------------------- complete


def test_a_completed_job_carries_its_id_digest_and_the_sync_record(
    client: TestClient,
) -> None:
    accepted = _submit(client, SUBMIT_EVALUATE, NOMINAL)
    done = _wait_for(client, accepted["job_id"], _settled)
    assert done["status"] == "completed", done
    assert done["progress_percent"] == 100.0
    result = done["result"]
    assert result["job_id"] == accepted["job_id"]
    assert result["input_digest"] == accepted["input_digest"]
    synchronous = client.post(f"{PREFIX}/evaluate", json=NOMINAL).json()
    assert result["record"]["digest"] == synchronous["digest"]


# ---------------------------------------------------------------- cancel


def test_a_cancelled_job_stops_before_its_next_shot_and_releases_its_worker(
    client: TestClient, gate: _Gate
) -> None:
    accepted = _submit(client, SUBMIT_COMPARE, PAIR)
    job_id = accepted["job_id"]
    _wait_for(client, job_id, lambda body: body["status"] == "running")
    response = client.post(f"{PREFIX}/jobs/{job_id}/cancel")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled"
    gate.release.set()
    done = _wait_for(client, job_id, _settled)
    assert done["status"] == "cancelled"
    assert done["result"] is None
    assert done["progress_percent"] < 100.0
    assert gate.calls <= 1


def test_a_shot_finishing_after_the_cancel_never_becomes_a_result(
    client: TestClient, gate: _Gate
) -> None:
    accepted = _submit(client, SUBMIT_EVALUATE, NOMINAL)
    job_id = accepted["job_id"]
    _wait_for(client, job_id, lambda body: body["status"] == "running")
    client.post(f"{PREFIX}/jobs/{job_id}/cancel")
    gate.release.set()
    done = _wait_for(client, job_id, _settled)
    assert done["status"] == "cancelled"
    assert done["result"] is None


def test_cancelling_a_finished_job_is_a_409(client: TestClient) -> None:
    accepted = _submit(client, SUBMIT_EVALUATE, NOMINAL)
    _wait_for(client, accepted["job_id"], _settled)
    response = client.post(f"{PREFIX}/jobs/{accepted['job_id']}/cancel")
    assert response.status_code == 409
    assert _detail_code(response) == "job_not_cancellable"


# ---------------------------------------------------------------- failed


def test_a_failed_job_says_why_and_carries_no_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("the solver diverged")

    monkeypatch.setattr(WorkbenchModel, "evaluate", explode)
    accepted = _submit(client, SUBMIT_EVALUATE, NOMINAL)
    done = _wait_for(client, accepted["job_id"], _settled)
    assert done["status"] == "failed"
    assert "the solver diverged" in done["error"]
    assert done["result"] is None


# ----------------------------------------------------------------- bound


def test_submits_beyond_the_worker_slots_are_refused_with_429(
    client: TestClient, gate: _Gate
) -> None:
    held = [
        _submit(client, SUBMIT_EVALUATE, NOMINAL)["job_id"]
        for _ in range(jobs.MAX_BUNKER_JOBS)
    ]
    response = client.post(SUBMIT_EVALUATE, json=NOMINAL)
    assert response.status_code == 429
    assert _detail_code(response) == "job_capacity_saturated"
    for job_id in held:
        client.post(f"{PREFIX}/jobs/{job_id}/cancel")
    gate.release.set()
    for job_id in held:
        _wait_for(client, job_id, _settled)
    accepted = _submit(client, SUBMIT_EVALUATE, NOMINAL)
    _wait_for(client, accepted["job_id"], _settled)
