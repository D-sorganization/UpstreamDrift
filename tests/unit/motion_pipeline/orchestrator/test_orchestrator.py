"""Unit tests for MotionPipeline orchestrator structure & hooks."""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.shared.python.motion_pipeline.orchestrator import (
    HookExecutionError,
    MotionPipeline,
    Stage,
    StageResult,
    _detect_format,
)

from ._local_fixtures import make_minimal_config


pytestmark = pytest.mark.unit

ORCHESTRATOR_LOGGER = "src.shared.python.motion_pipeline.orchestrator"


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


def test_motion_pipeline_fire_hooks_emits_payload() -> None:
    p = MotionPipeline(make_minimal_config())
    received = []

    def cb(payload):  # type: ignore[no-untyped-def]
        received.append((payload.stage, payload.data))

    p.add_hook(Stage.ADAPTER, cb)
    p._fire_hooks(Stage.ADAPTER, "data", {"meta": 1})
    assert received == [(Stage.ADAPTER, "data")]


def test_motion_pipeline_fire_hooks_lenient_logs_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    p = MotionPipeline(make_minimal_config())

    def broken_hook(payload):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"hook exploded for {payload.stage.value}")

    p.add_hook(Stage.ADAPTER, broken_hook)

    with caplog.at_level(
        logging.ERROR,
        logger="src.shared.python.motion_pipeline.orchestrator",
    ):
        p._fire_hooks(Stage.ADAPTER, "data", {"meta": 1})

    records = [
        record
        for record in caplog.records
        if record.name == "src.shared.python.motion_pipeline.orchestrator"
    ]
    assert records
    assert records[-1].exc_info is not None
    assert "Hook 'test_motion_pipeline_fire_hooks_lenient_logs_traceback" in caplog.text
    assert "adapter" in caplog.text
    assert "RuntimeError: hook exploded for adapter" in caplog.text


def _assert_unexpected_stage_failure_logs_traceback(
    caplog: pytest.LogCaptureFixture,
    run_stage: Callable[[MotionPipeline], StageResult],
    expected_error: str,
) -> None:
    pipeline = MotionPipeline(make_minimal_config())

    with caplog.at_level(logging.ERROR, logger=ORCHESTRATOR_LOGGER):
        result = run_stage(pipeline)

    assert result.success is False
    assert result.error == expected_error
    records = [
        record for record in caplog.records if record.name == ORCHESTRATOR_LOGGER
    ]
    assert records
    assert records[-1].exc_info is not None
    assert expected_error in caplog.text


def test_unexpected_pipeline_stage_failures_log_tracebacks(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text("{}", encoding="utf-8")

    def fail_load_source(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("adapter boom")

    def fail_preprocessing(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("preprocessing boom")

    def fail_scaling(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("scaling boom")

    def fail_make_solver(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("ik boom")

    monkeypatch.setattr(
        "src.shared.python.motion_pipeline.sources.loader.load_source",
        fail_load_source,
    )
    monkeypatch.setattr(
        "src.shared.python.motion_pipeline.preprocessing.apply_preprocessing",
        fail_preprocessing,
    )
    monkeypatch.setattr(
        "src.shared.python.motion_pipeline.scaling.scale_skeleton",
        fail_scaling,
    )
    monkeypatch.setattr(
        "src.shared.python.motion_pipeline.ik.base.make_ik_solver",
        fail_make_solver,
    )

    stage_cases: tuple[tuple[Callable[[MotionPipeline], StageResult], str], ...] = (
        (
            lambda pipeline: pipeline._run_adapter(source_path),
            "Adapter failed: adapter boom",
        ),
        (
            lambda pipeline: pipeline._run_preprocessing(object()),
            "Preprocessing failed: preprocessing boom",
        ),
        (
            lambda pipeline: pipeline._run_scaling(object(), SimpleNamespace()),
            "Scaling failed: scaling boom",
        ),
        (
            lambda pipeline: pipeline._run_inverse_kinematics(
                object(),
                SimpleNamespace(),
            ),
            "IK failed: ik boom",
        ),
    )

    for run_stage, expected_error in stage_cases:
        caplog.clear()
        _assert_unexpected_stage_failure_logs_traceback(
            caplog,
            run_stage,
            expected_error,
        )


def test_motion_pipeline_fire_hooks_strict_raises_diagnostic() -> None:
    p = MotionPipeline(make_minimal_config(strict_hooks=True))

    def broken_hook(payload):  # type: ignore[no-untyped-def]
        raise ValueError(f"bad hook payload: {payload.stage.value}")

    p.add_hook(Stage.PREPROCESSING, broken_hook)

    with pytest.raises(HookExecutionError) as excinfo:
        p._fire_hooks(Stage.PREPROCESSING, "data", {"meta": 1})

    assert excinfo.value.stage is Stage.PREPROCESSING
    assert excinfo.value.hook_name.endswith("broken_hook")
    assert isinstance(excinfo.value.original, ValueError)
    assert excinfo.value.__cause__ is excinfo.value.original
    assert "preprocessing" in str(excinfo.value)
    assert "ValueError: bad hook payload: preprocessing" in str(excinfo.value)


def test_motion_pipeline_default_skeleton_raises() -> None:
    """No skeleton supplied should raise a clear runtime error."""
    p = MotionPipeline(make_minimal_config())
    with pytest.raises(RuntimeError, match="skeleton"):
        p._get_default_skeleton()


def test_motion_pipeline_run_missing_source_path_raises() -> None:
    """A non-existent file path is a caller contract violation (#6932).

    Missing input is now classified as invalid input (InvalidInputError,
    a ValueError) rather than an internal RuntimeError, so the API maps it
    to 400.
    """
    from pathlib import Path

    from src.shared.python.motion_pipeline.orchestrator import InvalidInputError

    p = MotionPipeline(make_minimal_config())
    with pytest.raises((InvalidInputError, OSError, FileNotFoundError)):
        p.run(Path("/does/not/exist.c3d"))


# ---------------------------------------------------------------------------
# Error classification: invalid-input vs internal (#6932)
# ---------------------------------------------------------------------------


def test_unknown_source_type_is_invalid_input() -> None:
    from src.shared.python.motion_pipeline.orchestrator import InvalidInputError

    p = MotionPipeline(make_minimal_config())
    with pytest.raises(InvalidInputError):
        p.run(object())  # type: ignore[arg-type]


def test_invalid_input_error_is_value_error() -> None:
    from src.shared.python.motion_pipeline.orchestrator import InvalidInputError

    assert issubclass(InvalidInputError, ValueError)


def test_raise_stage_failure_maps_kind() -> None:
    from src.shared.python.motion_pipeline.orchestrator import InvalidInputError

    p = MotionPipeline(make_minimal_config())
    bad_input = StageResult(
        success=False, data=None, metadata={}, error="x", error_kind="invalid_input"
    )
    internal = StageResult(
        success=False, data=None, metadata={}, error="y", error_kind="internal"
    )
    with pytest.raises(InvalidInputError):
        p._raise_stage_failure("Stage", bad_input)
    with pytest.raises(RuntimeError):
        p._raise_stage_failure("Stage", internal)
