"""Characterization tests for motion_pipeline API route behaviour (#12846).

Pin request handling (success, error mapping, temp-file cleanup, hooks) with a
stubbed ``MotionPipeline`` so ``create_app`` can be decomposed safely.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from src.shared.python.motion_pipeline import api as api_module  # noqa: E402
from src.shared.python.motion_pipeline.orchestrator import Stage  # noqa: E402

RUN = "/api/v1/motion-pipeline/run"
RUN_CONFIG = "/api/v1/motion-pipeline/run-config"


class _StubPipeline:
    """Records config/hooks/path; behaviour is driven by class attributes."""

    instances: list[_StubPipeline] = []
    raises: Exception | None = None

    def __init__(self, config: Any) -> None:
        self.config = config
        self.hooks: dict[Any, list[Any]] = {}
        self.ran_path: Path | None = None
        _StubPipeline.instances.append(self)

    def add_hook(self, stage: Any, hook: Any) -> None:
        self.hooks.setdefault(stage, []).append(hook)

    def run(self, path: Path) -> Any:
        self.ran_path = Path(path)
        assert self.ran_path.exists()
        if _StubPipeline.raises is not None:
            raise _StubPipeline.raises
        return SimpleNamespace(
            request_id="r1",
            success=False,
            message="stub failure",
            metadata={"k": 1},
            model_dump=dict,
        )

    def get_audit_log(self) -> list[dict[str, Any]]:
        return [{"stage": "x"}]


@pytest.fixture
def client() -> Any:
    _StubPipeline.instances = []
    _StubPipeline.raises = None
    with patch.object(api_module, "MotionPipeline", _StubPipeline):
        yield TestClient(api_module.create_app())


@pytest.mark.unit
def test_run_success_response_hooks_and_tempfile_cleanup(client: TestClient) -> None:
    r = client.post(
        RUN,
        files={"file": ("a.bvh", b"abc", "application/octet-stream")},
        data={"source_format": "auto", "ik_backend": "mujoco"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["request_id"] == "r1"
    assert body["success"] is False
    assert body["error"] == "stub failure"
    assert body["audit_log"] == [{"stage": "x"}]
    assert body["metadata"] == {"k": 1}
    pipe = _StubPipeline.instances[0]
    assert pipe.config.ik_backend == "mujoco"
    assert pipe.config.matching_backend == "mujoco"
    assert pipe.config.adapter.format == "auto"
    assert pipe.ran_path is not None
    assert pipe.ran_path.suffix == ".bvh"
    assert not pipe.ran_path.exists()
    assert set(pipe.hooks) == set(Stage)
    # Each hook logs without raising.
    payload = SimpleNamespace(stage=Stage.SCALING)
    pipe.hooks[Stage.SCALING][0](payload)


@pytest.mark.unit
def test_run_value_error_maps_to_400_and_cleans_up(client: TestClient) -> None:
    _StubPipeline.raises = ValueError("bad input")
    r = client.post(
        RUN,
        files={"file": ("a.c3d", b"abc", "application/octet-stream")},
        data={"source_format": "auto"},
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "bad input"
    ran = _StubPipeline.instances[0].ran_path
    assert ran is not None and not ran.exists()


@pytest.mark.unit
def test_run_runtime_error_maps_to_500(client: TestClient) -> None:
    _StubPipeline.raises = RuntimeError("kaput")
    r = client.post(
        RUN,
        files={"file": ("a.c3d", b"abc", "application/octet-stream")},
        data={"source_format": "passthrough"},
    )
    assert r.status_code == 500
    assert r.json()["detail"] == "kaput"


@pytest.mark.unit
def test_run_config_success_uses_request_config(client: TestClient) -> None:
    cfg = json.dumps({"source_format": "auto", "ik_backend": "drake"})
    r = client.post(
        RUN_CONFIG,
        files={"file": ("a.trc", b"abc", "application/octet-stream")},
        data={"config": cfg},
    )
    assert r.status_code == 200
    pipe = _StubPipeline.instances[0]
    assert pipe.config.ik_backend == "drake"
    assert pipe.hooks == {}
    assert pipe.ran_path is not None and not pipe.ran_path.exists()


@pytest.mark.unit
def test_run_config_error_mapping(client: TestClient) -> None:
    cfg = json.dumps({"source_format": "auto"})
    files = {"file": ("a.trc", b"abc", "application/octet-stream")}
    _StubPipeline.raises = ValueError("v")
    assert client.post(RUN_CONFIG, files=files, data={"config": cfg}).status_code == 400
    _StubPipeline.raises = RuntimeError("r")
    assert client.post(RUN_CONFIG, files=files, data={"config": cfg}).status_code == 500


@pytest.mark.unit
def test_openapi_operations_are_stable(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", RUN, RUN_CONFIG}
    assert paths["/health"]["get"]["operationId"] == "health_check_health_get"
    assert (
        paths[RUN]["post"]["operationId"]
        == "run_pipeline_api_v1_motion_pipeline_run_post"
    )
    assert (
        paths[RUN_CONFIG]["post"]["operationId"]
        == "run_pipeline_with_config_api_v1_motion_pipeline_run_config_post"
    )
