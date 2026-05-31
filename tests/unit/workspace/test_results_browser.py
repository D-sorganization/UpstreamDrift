"""Focused tests for canonical result artifact indexing."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.simulation_backends import (
    ProvenanceStamp,
    Trace,
    attach_provenance_to_trace,
)
from src.shared.python.simulation_backends.trace_io import write_trace
from src.shared.python.workspace import ResultFilter, ResultsBrowser

pytestmark = pytest.mark.unit


def _stamp(engine: str) -> ProvenanceStamp:
    return ProvenanceStamp(
        engine=engine,
        engine_version="1.0.0",
        model_hash="model-sha",
        param_hash="param-sha",
        git_commit="abc123",
        solver_settings={"rtol": 1e-6},
        seed=7,
        created_at="2026-05-31T10:00:00Z",
        convention="canonical-core",
        frame="world",
        units={"length": "m", "time": "s"},
    )


def _trace(
    *,
    backend: str,
    session_id: str,
    dataset_id: str,
    with_provenance: bool = True,
) -> Trace:
    trace = Trace(
        t=np.array([0.0, 0.1]),
        q=np.zeros((2, 2)),
        v=np.ones((2, 2)),
        dt=0.1,
        backend=backend,
        meta={
            "project_id": "project-001",
            "subject_id": "subject-001",
            "session_id": session_id,
            "dataset_id": dataset_id,
            "label": f"{backend}-{session_id}",
        },
    )
    if not with_provenance:
        return trace
    return attach_provenance_to_trace(trace, _stamp(backend))


def test_results_browser_indexes_trace_metadata_and_provenance(tmp_path) -> None:
    write_trace(
        _trace(backend="ode", session_id="s1", dataset_id="d1"), tmp_path / "a.h5"
    )
    (tmp_path / "nested").mkdir()
    write_trace(
        _trace(
            backend="mujoco",
            session_id="s2",
            dataset_id="d2",
            with_provenance=False,
        ),
        tmp_path / "nested" / "b.hdf5",
    )

    artifacts = ResultsBrowser(tmp_path).index()

    assert [artifact.relative_path for artifact in artifacts] == [
        "a.h5",
        "nested/b.hdf5",
    ]
    assert artifacts[0].backend == "ode"
    assert artifacts[0].schema_version == "2.0.0"
    assert artifacts[0].metadata["session_id"] == "s1"
    assert artifacts[0].provenance["engine"] == "ode"
    assert artifacts[1].has_provenance is False


def test_results_browser_filters_by_session_backend_and_text(tmp_path) -> None:
    write_trace(
        _trace(backend="ode", session_id="s1", dataset_id="d1"), tmp_path / "a.h5"
    )
    write_trace(
        _trace(backend="mujoco", session_id="s2", dataset_id="d2"), tmp_path / "b.h5"
    )

    browser = ResultsBrowser(tmp_path)

    assert [
        item.relative_path for item in browser.index(ResultFilter(session_id="s2"))
    ] == ["b.h5"]
    assert [
        item.relative_path for item in browser.index(ResultFilter(backend="ode"))
    ] == ["a.h5"]
    assert [
        item.relative_path for item in browser.index(ResultFilter(text="mujoco-s2"))
    ] == ["b.h5"]
    assert len(browser.index(ResultFilter(has_provenance=True))) == 2


def test_results_browser_skips_non_hdf5_and_unreadable_hdf5(tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "broken.h5").write_text("not hdf5", encoding="utf-8")
    write_trace(
        _trace(backend="ode", session_id="s1", dataset_id="d1"), tmp_path / "ok.h5"
    )

    assert [item.relative_path for item in ResultsBrowser(tmp_path).index()] == [
        "ok.h5"
    ]
