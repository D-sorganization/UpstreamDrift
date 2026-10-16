"""NSX-40: create, poll and cancel the loop-nullspace job routes (#12572)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("highspy")
jsonschema = pytest.importorskip("jsonschema")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.routes import analysis_loop_nullspace as routes  # noqa: E402
from src.api.services import loop_nullspace_service as svc  # noqa: E402
from src.api.task_manager import TaskManager  # noqa: E402
from src.shared.python.loop_nullspace import facade  # noqa: E402
from tests.unit.loop_nullspace.report_support import (  # noqa: E402
    PlanarSource,
    load_schema,
)

pytestmark = [pytest.mark.unit, pytest.mark.integration]

BASE = "/analysis/loop-nullspace/jobs"
WAIT_S = 60.0
BODY: dict[str, Any] = {
    "source_id": "planar",
    "preset": "envelope_only",
    "frame_indices": [0, 3, 6, 9],
}


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(routes.router)
    from src.api.dependencies import get_task_manager

    manager = TaskManager()
    app.dependency_overrides[get_task_manager] = lambda: manager
    svc.register_source("planar", PlanarSource())
    svc.RESULT_CACHE.clear()
    yield TestClient(app)
    assert routes.JOB_POOL.drain(WAIT_S), "a loop-nullspace worker never exited"
    svc.unregister_source("planar")
    svc.RESULT_CACHE.clear()


def _wait(client: TestClient, job_id: str, *terminal: str) -> dict[str, Any]:
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        body = client.get(f"{BASE}/{job_id}").json()
        if body["status"] in terminal:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job never reached {terminal}: {body}")


def test_create_returns_a_job_id_and_the_job_completes(client) -> None:
    resp = client.post(BASE, json=BODY)
    assert resp.status_code == 202
    job = resp.json()
    assert job["cache_hit"] is False and len(job["cache_key"]) == 64
    done = _wait(client, job["job_id"], "completed")
    assert done["progress_percent"] == 100.0 and done["error"] is None
    jsonschema.validate(
        done["result"], load_schema(), cls=jsonschema.Draft202012Validator
    )
    assert done["result"]["service"]["cache_key"] == job["cache_key"]


def test_identical_request_is_served_from_the_cache(client, monkeypatch) -> None:
    first = client.post(BASE, json=BODY).json()
    _wait(client, first["job_id"], "completed")
    monkeypatch.setattr(
        svc.facade, "analyze", lambda *a, **k: pytest.fail("recomputed")
    )
    again = client.post(BASE, json=BODY)
    assert again.status_code == 202
    body = again.json()
    assert body["cache_hit"] is True and body["status"] == "completed"
    assert body["job_id"] != first["job_id"]
    assert body["result"]["service"]["cache_key"] == first["cache_key"]


def test_registry_version_change_is_a_miss(client, monkeypatch) -> None:
    first = client.post(BASE, json=BODY).json()
    _wait(client, first["job_id"], "completed")
    monkeypatch.setattr(svc, "_registry_version", lambda: "999.0.0")
    second = client.post(BASE, json=BODY).json()
    assert second["cache_hit"] is False and second["cache_key"] != first["cache_key"]
    _wait(client, second["job_id"], "completed")


class _Gate:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()


@pytest.fixture
def gated(monkeypatch) -> _Gate:
    """A facade that publishes partial invariants, then waits to be released."""
    gate = _Gate()
    real = facade.analyze

    def slow(trajectory, engine, preset, options, *, on_stage=None, should_cancel=None):
        def hook(stage, pct, partial):
            on_stage(stage, pct, partial)
            if stage == "invariants":
                gate.entered.set()
                assert gate.release.wait(WAIT_S)

        return real(
            trajectory,
            engine,
            preset,
            options,
            on_stage=hook,
            should_cancel=should_cancel,
        )

    monkeypatch.setattr(svc.facade, "analyze", slow)
    return gate


def test_poll_shows_partial_invariants_first(client, gated) -> None:
    job = client.post(BASE, json=BODY).json()
    assert gated.entered.wait(WAIT_S)
    mid = client.get(f"{BASE}/{job['job_id']}").json()
    assert mid["status"] == "running" and mid["stage"] == "invariants"
    assert mid["result"] is None
    assert mid["partial"]["invariants"]["frames"]
    assert 0.0 < mid["progress_percent"] < 100.0
    gated.release.set()
    _wait(client, job["job_id"], "completed")


def test_delete_cancels_a_running_job_and_it_never_gains_a_result(
    client, gated
) -> None:
    job = client.post(BASE, json=BODY).json()
    assert gated.entered.wait(WAIT_S)
    cancelled = client.delete(f"{BASE}/{job['job_id']}")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    gated.release.set()
    assert routes.JOB_POOL.drain(WAIT_S)
    final = client.get(f"{BASE}/{job['job_id']}").json()
    assert final["status"] == "cancelled" and final["result"] is None
    assert svc.lookup(svc.LoopNullspaceRequest(**BODY)) is None  # nothing cached


def test_delete_after_completion_is_a_conflict(client) -> None:
    job = client.post(BASE, json=BODY).json()
    _wait(client, job["job_id"], "completed")
    resp = client.delete(f"{BASE}/{job['job_id']}")
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "job_not_cancellable"


def test_failed_job_reports_why_and_has_no_result(client, monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("solver inconclusive")

    monkeypatch.setattr(svc.facade, "analyze", boom)
    job = client.post(BASE, json=BODY).json()
    failed = _wait(client, job["job_id"], "failed")
    assert "solver inconclusive" in failed["error"] and failed["result"] is None


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        ({"source_id": "missing"}, "unknown_source"),
        ({"preset": "nope"}, "unknown_preset"),
        (
            {"options": {"include_envelopes": False, "include_allocation": True}},
            "invalid_options",
        ),
    ],
)
def test_refusals_are_422_before_any_job_exists(client, patch, code) -> None:
    resp = client.post(BASE, json={**BODY, **patch})
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == code


def test_malformed_body_is_422(client) -> None:
    assert client.post(BASE, json={**BODY, "frame_indices": []}).status_code == 422
    assert client.post(BASE, json={**BODY, "extra": 1}).status_code == 422


@pytest.mark.parametrize("job_id", ["nope", "00000000-0000-0000-0000-000000000000"])
def test_unknown_job_is_404(client, job_id) -> None:
    assert client.get(f"{BASE}/{job_id}").status_code == 404
    assert client.delete(f"{BASE}/{job_id}").status_code == 404


def test_other_tasks_in_the_manager_are_not_served(client) -> None:
    from src.api.dependencies import get_task_manager

    manager = client.app.dependency_overrides[get_task_manager]()
    manager.set(
        "11111111-1111-1111-1111-111111111111", {"status": "running", "kind": "other"}
    )
    assert client.get(f"{BASE}/11111111-1111-1111-1111-111111111111").status_code == 404


def test_pool_saturation_is_429(client, gated, monkeypatch) -> None:
    monkeypatch.setattr(routes, "JOB_POOL", routes.BunkerJobPool(1))
    first = client.post(BASE, json=BODY).json()
    assert gated.entered.wait(WAIT_S)
    other = client.post(BASE, json={**BODY, "frame_indices": [0, 3, 6, 12]})
    assert other.status_code == 429
    gated.release.set()
    _wait(client, first["job_id"], "completed")
    assert routes.JOB_POOL.drain(WAIT_S)
