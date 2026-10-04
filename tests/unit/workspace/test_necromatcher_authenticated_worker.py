"""Worker authentication closes before computation and result delivery."""

from dataclasses import asdict
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hypothesis_fixture import imported_capture
from src.shared.python.workspace import necromatcher_fit_jobs as jobs
from src.shared.python.workspace import necromatcher_fit_worker as worker
from src.shared.python.workspace.necromatcher_capture_identity import capture_identity

pytestmark = pytest.mark.unit


def mutate_same_stat(path: Path) -> None:
    before = path.stat()
    raw = bytearray(path.read_bytes())
    raw[-10] ^= 1
    path.write_bytes(raw)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


@pytest.fixture
def case(fit_case: Any, tmp_path: Path, monkeypatch: Any) -> Any:
    library, _, _ = fit_case
    capture = imported_capture(library, tmp_path / "source")
    state = SimpleNamespace(computations=0, reviews=0, validations=0, mutate=None)
    source = {
        "capture_id": capture.dataset_id,
        "frame_indices": [0, 1, 2],
        "q": [[0.0], [0.1], [0.2]],
        "evidence": {"original_fit": {"free_coordinates": ["q"]}},
    }
    native = SimpleNamespace(closure_residuals=lambda q: np.zeros(3))
    binding = SimpleNamespace(
        fit=source,
        plant=native,
        project=lambda i: None,
        review_inputs=lambda: ("camera", {"origin": "point"}),
    )

    class Review:
        def __init__(self, *args: Any) -> None:
            pass

        def __enter__(self) -> Any:
            state.reviews += 1
            return self

        def __exit__(self, *args: Any) -> None:
            state.reviews -= 1

        def frame(self, index: int) -> dict[str, Any]:
            return {"frame": {"index": index}}

    def validate(*args: Any) -> None:
        state.validations += 1
        capture_identity(library, capture.dataset_id)
        if state.mutate == state.validations:
            mutate_same_stat(Path(capture.path))

    def compute(*args: Any) -> Any:
        state.computations += 1
        assert state.reviews == 0, "CaptureReview remained open during computation"
        with library.authenticated_read():
            pass
        return SimpleNamespace(
            evaluate_source_times=lambda times: np.zeros((len(times), 1))
        )

    stamp = {"source_sha256": "source", "runtime_sha256": "runtime"}
    options = jobs.NativeRefitOptions((0, 2), 2, (1.0,))
    request = {
        "library_root": str(library.root),
        "source_fit_id": capture.dataset_id,
        "source_fit_hash": capture.metadata["hash"],
        "execution_stamp": stamp,
        "options": asdict(options),
    }
    monkeypatch.setattr(worker, "NecromatcherLibrary", lambda _: library)
    monkeypatch.setattr(worker, "fit_execution_stamp", lambda: stamp)
    monkeypatch.setattr(jobs, "_verify_scope_request", validate)
    monkeypatch.setattr(worker, "_load_refit_binding", lambda *_: (binding, None))
    monkeypatch.setattr(
        worker, "_range_provenance", lambda *_: {"range_source": "none"}
    )
    monkeypatch.setattr(worker, "CaptureReview", Review)
    monkeypatch.setattr(worker, "contact_schedule_binding", lambda *_: None)
    monkeypatch.setattr(
        worker,
        "read_capture_evidence",
        lambda r, a, indices, **kw: SimpleNamespace(
            source_times=np.asarray(indices) / 10
        ),
    )
    monkeypatch.setattr(worker, "_worker_inputs", lambda *_: ("inputs", None))
    monkeypatch.setattr(worker, "_compute_operation", compute)
    monkeypatch.setattr(
        worker,
        "_build_fit_payload",
        lambda *_: {"evidence": {"original_fit": {}}, "provenance": {}},
    )
    monkeypatch.setattr(
        worker, "_dense_reprojection_metrics", lambda *_: {"dense_rms_pixels": 0}
    )
    return library, request, state


def test_setup_closes_before_compute_and_final_context_closes_before_return(
    case: Any,
) -> None:
    library, request, state = case
    output = worker.compute_native_refit(request)
    assert state.computations == 1 and state.validations == 2
    assert state.reviews == 0
    assert output["evidence"]["original_fit"]["dense_rms_pixels"] == 0
    with library.authenticated_read():
        pass


@pytest.mark.parametrize("operation", ["fit", "author_initialization"])
def test_same_stat_setup_close_failure_never_computes(
    case: Any, operation: str
) -> None:
    library, request, state = case
    request["options"]["operation"] = operation
    state.mutate = 1
    with pytest.raises(ValueError, match="hash|changed"):
        worker.compute_native_refit(request)
    assert state.computations == 0 and state.reviews == 0
    with library.authenticated_read():
        pass


def test_same_stat_result_close_failure_prevents_response(case: Any) -> None:
    library, request, state = case
    state.mutate = 2
    with pytest.raises(ValueError, match="hash|changed"):
        worker.compute_native_refit(request)
    assert state.computations == 1 and state.reviews == 0
    with library.authenticated_read():
        pass


def test_computation_capture_mutation_rejected_by_fresh_result_context(
    case: Any, monkeypatch: Any
) -> None:
    library, request, state = case
    original = worker._compute_operation

    def compute(*args: Any) -> Any:
        result = original(*args)
        mutate_same_stat(Path(library.load_asset(request["source_fit_id"]).path))
        return result

    monkeypatch.setattr(worker, "_compute_operation", compute)
    with pytest.raises(ValueError, match="hash|changed"):
        worker.compute_native_refit(request)
    assert state.computations == 1 and state.reviews == 0
    with library.authenticated_read():
        pass
