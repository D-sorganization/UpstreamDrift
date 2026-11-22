"""Asynchronous native export jobs over REST and MCP (#15150 G8)."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.database import get_db_factory
from src.api.routes import agent_sim_run
from src.sim_run import native_export_jobs as jobs
from src.sim_run.agent_capabilities import run_agent_capability
from src.sim_run.mcp import SecurityPolicy, SimulationMcpServer
from src.sim_run.store import SimulationRunStore

pytestmark = pytest.mark.unit

_WAIT_S = 10.0
_MANIFEST = {"engines": [{"engine": "mujoco", "clips": {"face_on_1x": "a.mp4"}}]}


class _FakeRunner:
    """The exporter subprocess stand-in: records argv, returns or raises."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.error: Exception | None = None

    def __call__(self, argv: list[str]) -> dict[str, Any]:
        self.calls.append(list(argv))
        if self.error is not None:
            raise self.error
        return _MANIFEST


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Allowed roots, a REST store and a fake-runner queue as the shared one."""
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "b.npz").write_bytes(b"")
    monkeypatch.setenv("UD_NATIVE_EXPORT_INPUT_ROOTS", str(tmp_path / "in"))
    monkeypatch.setenv("UD_NATIVE_EXPORT_OUTPUT_ROOT", str(tmp_path / "out"))
    runs = SimulationRunStore(tmp_path / "runs")
    monkeypatch.setattr(agent_sim_run, "_default_store", lambda: runs)
    runner = _FakeRunner()
    queue = jobs.ExportJobQueue(runner, max_concurrency=1, retain=8)
    monkeypatch.setattr(jobs, "_QUEUE", queue)
    bundle = str(tmp_path / "in" / "b.npz")
    request = {"options": {"bundle": bundle, "out": "clips", "swing": "s1"}}
    return SimpleNamespace(root=tmp_path, runner=runner, queue=queue, request=request)


def _value(record: dict[str, Any]) -> dict[str, Any]:
    assert record["status"] == "succeeded", record
    return record["delivery"]["value"]


def test_invalid_request_is_rejected_before_queuing(env: SimpleNamespace) -> None:
    bad = {"options": {"bundle": "/etc/passwd", "out": "x", "swing": "s"}}
    record = run_agent_capability("native_export_run", bad)
    assert "outside the allowed input roots" in record["error"]["message"]
    assert env.queue.job_ids() == [] and env.runner.calls == []


def test_valid_request_returns_a_job_id_and_ends_succeeded(
    env: SimpleNamespace,
) -> None:
    started = _value(run_agent_capability("native_export_run", env.request))
    assert started["state"] == "queued" and "--bundle" in started["command_line"]
    assert env.queue.wait(started["job_id"], _WAIT_S)["state"] == "succeeded"
    request = {"job_id": started["job_id"]}
    status = _value(run_agent_capability("native_export_status", request))
    assert status["state"] == "succeeded" and status["manifest"] == _MANIFEST
    assert env.runner.calls[0][:2] == ["--bundle", env.request["options"]["bundle"]]


def test_exporter_failure_is_reported_on_the_job(env: SimpleNamespace) -> None:
    env.runner.error = RuntimeError("exporter exited 2")
    started = _value(run_agent_capability("native_export_run", env.request))
    done = env.queue.wait(started["job_id"], _WAIT_S)
    assert (done["state"], done["error"]) == ("failed", "exporter exited 2")
    assert done["manifest"] is None


def test_unknown_job_id_and_retention_bound(env: SimpleNamespace) -> None:
    record = run_agent_capability("native_export_status", {"job_id": "nope"})
    assert "unknown export job id 'nope'" in record["error"]["message"]
    missing = run_agent_capability("native_export_status", {})
    assert "job_id" in missing["error"]["message"]
    queue = jobs.ExportJobQueue(_FakeRunner(), max_concurrency=1, retain=1)
    queue.wait(queue.submit(["--a"], "a")["job_id"], _WAIT_S)
    second = queue.submit(["--b"], "b")["job_id"]
    assert queue.job_ids() == [second]
    with pytest.raises(ValueError, match="max_concurrency"):
        jobs.ExportJobQueue(_FakeRunner(), max_concurrency=0, retain=1)


def test_rest_and_mcp_reach_the_same_queue(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOLF_SUITE_MODE", "local")
    app = FastAPI()
    app.include_router(agent_sim_run.router)
    app.dependency_overrides[get_db_factory] = lambda: lambda: nullcontext(object())
    client = TestClient(app)
    path = "/tools/agent-sim-run/native-export"
    rest = client.post(f"{path}/run", json=env.request).json()
    store = SimulationRunStore(env.root / "mcp")
    server = SimulationMcpServer(store, None, SecurityPolicy())
    params = {"name": "native_export_run", "arguments": {"request": env.request}}
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params}
    mcp = json.loads(server.handle_request(call)["result"]["content"][0]["text"])
    ids = [_value(rest)["job_id"], _value(mcp)["job_id"]]
    assert sorted(ids) == sorted(env.queue.job_ids()) and len(set(ids)) == 2
    assert rest["input_sha256"] == mcp["input_sha256"]
    status = client.post(f"{path}/status", json={"job_id": ids[1]}).json()
    assert _value(status)["job_id"] == ids[1]
    assert [env.queue.wait(i, _WAIT_S)["state"] for i in ids] == ["succeeded"] * 2
    assert env.runner.calls[0] == env.runner.calls[1]


def test_subprocess_entry_writes_clips_and_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tools.native_viewer_export import cli
    from src.tools.native_viewer_export.core import ExportResult

    clip = tmp_path / "a.mp4"
    fake = [
        ExportResult("mujoco", {"face_on_1x": clip}, 3, flight={"carry_m": 1.0}),
        ExportResult("drake", {}, 0, skipped_reason="not installed"),
    ]
    monkeypatch.setattr(cli, "run_export", lambda *a, **k: fake)
    out = tmp_path / "manifest.json"
    argv = ["--manifest", str(out), "--bundle", "b.npz", "--out", "o", "--swing", "s"]
    assert jobs.main(argv) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))["engines"]
    assert rows[0]["clips"] == {"face_on_1x": str(clip)}
    assert rows[0]["flight"] == {"carry_m": 1.0}
    assert (rows[1]["skipped_reason"], rows[1]["clips"]) == ("not installed", {})
