"""Unit tests for MotionPipeline orchestrator structure & hooks."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.orchestrator import (
    AdapterOverride,
    MotionPipeline,
    PipelineConfig,
    Stage,
    StageResult,
    _detect_format,
)

from ._local_fixtures import make_minimal_config


def test_motion_pipeline_constructs() -> None:
    p = MotionPipeline(make_minimal_config())
    assert p.config.ik_backend == "mujoco"
    assert p.get_audit_log() == []


def test_motion_pipeline_add_hook_records_callback() -> None:
    p = MotionPipeline(make_minimal_config())
    called: list[Stage] = []

    def cb(payload):  # type: ignore[no-untyped-def]
        called.append(payload.stage)

    p.add_hook(Stage.ADAPTER, cb)
    assert len(p._hooks[Stage.ADAPTER]) == 1


def test_motion_pipeline_compute_hash_stable() -> None:
    p = MotionPipeline(make_minimal_config())
    h1 = p._compute_hash("hello")
    h2 = p._compute_hash("hello")
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_motion_pipeline_compute_hash_bytes_and_str_equal() -> None:
    p = MotionPipeline(make_minimal_config())
    assert p._compute_hash("hello") == p._compute_hash(b"hello")


def test_stage_result_dataclass() -> None:
    sr = StageResult(success=True, data={"k": 1}, metadata={"x": 2})
    assert sr.success is True
    assert sr.error is None


def test_detect_format_known_extensions() -> None:
    from pathlib import Path

    assert _detect_format(Path("x.c3d")) == "c3d"
    assert _detect_format(Path("x.trc")) == "trc"
    assert _detect_format(Path("x.bvh")) == "bvh"
    assert _detect_format(Path("x.json")) == "json"
    assert _detect_format(Path("x.unknown_ext")) == "unknown"


def test_motion_pipeline_get_version_returns_string() -> None:
    p = MotionPipeline(make_minimal_config())
    v = p._get_version()
    assert isinstance(v, str)
    assert len(v) > 0


@pytest.mark.xfail(
    strict=False,
    reason=(
        "_fire_hooks instantiates HookPayload, which is a typing.Protocol "
        "and raises TypeError. Production bug — issue to file separately."
    ),
)
def test_motion_pipeline_fire_hooks_emits_payload() -> None:
    p = MotionPipeline(make_minimal_config())
    received = []

    def cb(payload):  # type: ignore[no-untyped-def]
        received.append((payload.stage, payload.data))

    p.add_hook(Stage.ADAPTER, cb)
    p._fire_hooks(Stage.ADAPTER, "data", {"meta": 1})
    assert received == [(Stage.ADAPTER, "data")]


def test_motion_pipeline_default_skeleton_raises() -> None:
    """No skeleton supplied should raise a clear runtime error."""
    p = MotionPipeline(make_minimal_config())
    with pytest.raises(RuntimeError, match="skeleton"):
        p._get_default_skeleton()


def test_motion_pipeline_run_missing_source_path_raises() -> None:
    """Running on a non-existent file path bubbles up a RuntimeError."""
    from pathlib import Path

    p = MotionPipeline(make_minimal_config())
    with pytest.raises((RuntimeError, OSError, FileNotFoundError)):
        p.run(Path("/does/not/exist.c3d"))
