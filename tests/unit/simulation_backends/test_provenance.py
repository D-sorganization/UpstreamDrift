"""Focused tests for run provenance stamps."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from src.shared.python.engine_core.checkpoint import StateCheckpoint
from src.shared.python.simulation_backends import (
    PROVENANCE_META_KEY,
    ProvenanceStamp,
    Trace,
    attach_provenance_to_checkpoint,
    attach_provenance_to_trace,
)
from src.shared.python.simulation_backends.trace_io import read_trace, write_trace

pytestmark = pytest.mark.unit


def _stamp() -> ProvenanceStamp:
    return ProvenanceStamp(
        engine="ode",
        engine_version="1.2.3",
        model_hash="model-sha256",
        param_hash="param-sha256",
        git_commit="abc1234",
        solver_settings={"rtol": 1e-6, "method": "rk4"},
        seed=42,
        created_at="2026-05-30T12:00:00Z",
        convention="canonical-core",
        frame="world",
        units={"length": "m", "angle": "rad", "time": "s"},
    )


def test_provenance_stamp_is_frozen_and_serializes_deterministically() -> None:
    stamp = ProvenanceStamp(
        engine="mujoco",
        engine_version="3.6.0",
        model_hash="model",
        param_hash="params",
        git_commit="commit",
        solver_settings={"z": 3, "a": {"b": 2}},
        seed=None,
        created_at="2026-05-30T12:00:00Z",
        convention="canonical-core",
        frame="world",
        units={"time": "s", "length": "m"},
    )

    with pytest.raises(AttributeError):
        stamp.engine = "other"  # type: ignore[misc]

    assert isinstance(stamp.solver_settings, MappingProxyType)
    assert stamp.to_dict() == {
        "engine": "mujoco",
        "engine_version": "3.6.0",
        "model_hash": "model",
        "param_hash": "params",
        "git_commit": "commit",
        "solver_settings": {"a": {"b": 2}, "z": 3},
        "seed": None,
        "created_at": "2026-05-30T12:00:00Z",
        "convention": "canonical-core",
        "frame": "world",
        "units": {"length": "m", "time": "s"},
    }


def test_provenance_stamp_requires_caller_created_at() -> None:
    with pytest.raises(ValueError, match="created_at"):
        ProvenanceStamp(
            engine="ode",
            engine_version="1",
            model_hash="model",
            param_hash="params",
            git_commit="commit",
            solver_settings={},
            seed=1,
            created_at="",
            convention="canonical-core",
            frame="world",
            units={},
        )


def test_attach_provenance_to_trace_returns_copy_with_flat_meta(tmp_path) -> None:
    trace = Trace(
        t=np.array([0.0, 0.1]),
        q=np.zeros((2, 2)),
        v=np.ones((2, 2)),
        dt=0.1,
        backend="ode",
        meta={"scenario": "smoke"},
    )

    stamped = attach_provenance_to_trace(trace, _stamp())

    assert stamped is not trace
    assert trace.meta == {"scenario": "smoke"}
    assert stamped.meta["scenario"] == "smoke"
    assert stamped.meta["provenance_engine"] == "ode"
    assert stamped.meta["provenance_created_at"] == "2026-05-30T12:00:00Z"
    assert stamped.meta["provenance_units"] == ("{angle='rad', length='m', time='s'}")

    path = tmp_path / "trace.h5"
    write_trace(stamped, path)
    restored = read_trace(path)
    assert restored.meta["provenance_engine"] == "ode"
    assert restored.meta["provenance_seed"] == 42


def test_attach_provenance_to_checkpoint_metadata_is_nested() -> None:
    checkpoint = StateCheckpoint.create(
        engine_type="TestEngine",
        engine_state={"key": "value"},
        q=np.array([1.0, 2.0]),
        v=np.array([3.0, 4.0]),
        timestamp=1.5,
        metadata={"existing": "keep"},
    )

    stamped = attach_provenance_to_checkpoint(checkpoint, _stamp())

    assert stamped is not checkpoint
    assert checkpoint.metadata == {"existing": "keep"}
    assert stamped.metadata["existing"] == "keep"
    assert stamped.metadata[PROVENANCE_META_KEY]["engine"] == "ode"
    assert stamped.metadata[PROVENANCE_META_KEY]["created_at"] == (
        "2026-05-30T12:00:00Z"
    )
