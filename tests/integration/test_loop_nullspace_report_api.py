"""NSX-40 acceptance (#12572): the NSX report end to end through the API.

The Pinocchio anthro ground-support fixture is passed through the FastAPI
routes by **server-side source id** (``anthro_ground_support``) and frame
indices -- no arrays are uploaded.  The source reuses the NSX-60 test support
(``tests/integration/loop_nullspace_lower_body_support.py``), so the report is
the same computation as the 200-frame benchmark.

The frames cover ground-supported stance, a flight frame with no contact
(489) and the airborne frame 510 whose balance cannot be closed: it must be
listed as refused with its residual, not repaired.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pinocchio")
pytest.importorskip("highspy")
jsonschema = pytest.importorskip("jsonschema")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.dependencies import get_task_manager  # noqa: E402
from src.api.routes import analysis_loop_nullspace as routes  # noqa: E402
from src.api.services import loop_nullspace_service as svc  # noqa: E402
from src.api.task_manager import TaskManager  # noqa: E402
from tests.integration.loop_nullspace_report_api_support import (  # noqa: E402
    SOURCE_ID,
    AnthroGroundSupportSource,
)
from tests.unit.loop_nullspace.report_support import load_schema  # noqa: E402

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.requires_pinocchio,
]

BASE = "/analysis/loop-nullspace/jobs"
FRAMES = [0, 60, 120, 300, 489, 510, 540, 597]
AIRBORNE = 510
WAIT_S = 600.0


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(routes.router)
    manager = TaskManager()
    app.dependency_overrides[get_task_manager] = lambda: manager
    svc.register_source(SOURCE_ID, AnthroGroundSupportSource())
    svc.RESULT_CACHE.clear()
    yield TestClient(app)
    routes.JOB_POOL.drain(WAIT_S)
    svc.unregister_source(SOURCE_ID)
    svc.RESULT_CACHE.clear()


def _body(preset: str = "envelope_only") -> dict:
    return {"source_id": SOURCE_ID, "preset": preset, "frame_indices": FRAMES}


def _wait(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        body = client.get(f"{BASE}/{job_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.2)
    raise AssertionError("the job did not finish")


@pytest.fixture(scope="module")
def finished(client) -> dict:
    submitted = client.post(BASE, json=_body())
    assert submitted.status_code == 202
    return _wait(client, submitted.json()["job_id"])


def test_job_completes_and_the_report_validates(finished) -> None:
    assert finished["status"] == "completed", finished["error"]
    report = finished["result"]
    jsonschema.validate(report, load_schema(), cls=jsonschema.Draft202012Validator)
    assert report["status"] == "complete"
    assert report["engine"]["name"] == "pinocchio"
    assert report["admission"]["verdict"] == "ADMITTED"


def test_airborne_frame_is_refused_with_its_residual_and_not_repaired(finished) -> None:
    traj = finished["result"]["trajectory"]
    refused = {r["index"]: r for r in traj["refused_frames"]}
    assert set(refused) == {AIRBORNE}
    assert 1e-9 < refused[AIRBORNE]["residual_scaled"] < 1e-6
    assert AIRBORNE not in traj["analyzed_frame_indices"]
    assert set(traj["analyzed_frame_indices"]) == set(FRAMES) - {AIRBORNE}
    assert finished["result"]["admission"]["repaired"] is False


def test_invariants_and_envelopes_cover_every_analysed_frame(finished) -> None:
    report = finished["result"]
    analysed = report["trajectory"]["analyzed_frame_indices"]
    assert [f["index"] for f in report["invariants"]["frames"]] == analysed
    assert [f["index"] for f in report["envelopes"]["frames"]] == analysed
    names = set(report["envelopes"]["quantities"])
    assert {"lambda_fx", "ground_fz", "right_foot_fz", "left_foot_fz"} <= names
    assert any(n.startswith("tau_") for n in names)
    assert report["registry_version"] == report["hypotheses"]["registry_version"]


def test_unavailable_invariants_say_why_and_carry_no_numbers(finished) -> None:
    frame = finished["result"]["invariants"]["frames"][0]["quantities"]
    wrench = frame["net_club_wrench"]
    assert wrench["allocation_dependence"] == "INVARIANT"  # club data was supplied
    for q in frame.values():
        if q["allocation_dependence"] == "UNAVAILABLE":
            assert q["values"] is None and q["reason"]


def test_identical_request_is_a_cache_hit(client, finished) -> None:
    again = client.post(BASE, json=_body())
    assert again.status_code == 202
    body = again.json()
    assert body["cache_hit"] is True and body["status"] == "completed"
    assert body["result"] == finished["result"]


def test_other_preset_misses_the_cache_and_adds_the_observation_rows(client) -> None:
    submitted = client.post(BASE, json=_body("envelope_grf_observed")).json()
    assert submitted["cache_hit"] is False
    done = _wait(client, submitted["job_id"])
    assert done["status"] == "completed", done["error"]
    hyps = {h["id"] for h in done["result"]["hypotheses"]["hypotheses"]}
    assert "H-GRF-OBSERVATION" in hyps
