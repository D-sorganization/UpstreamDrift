"""Candidate and delayed publication authentication close before side effects."""

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import pytest

from authenticated_refit_fixture import make_scoped_refit_case
from test_necromatcher_authenticated_refit_queue import Service
from src.shared.python.workspace import necromatcher_fit_jobs as jobs

pytestmark = pytest.mark.unit


def queued(fit_case: Any, tmp_path: Path, monkeypatch: Any) -> tuple[Any, Any, Path]:
    library, options, source = make_scoped_refit_case(fit_case, tmp_path)
    # Hold the unrelated source-stamp sentinel fixed while peers own disjoint files.
    stamp = jobs.fit_execution_stamp()
    monkeypatch.setattr(jobs, "fit_execution_stamp", lambda: deepcopy(stamp))
    response = deepcopy(source)
    response["provenance"].update(
        warm_start_fit_id="child",
        warm_start_fit_hash=library.load_asset("child").metadata["hash"],
    )

    def worker(*args: Any) -> dict:
        with library.authenticated_read():
            pass  # No queue state crosses the clean worker boundary.
        return {"fit": response}

    monkeypatch.setattr(jobs, "_execute_worker", worker)
    service = Service(library)
    _, root = jobs.start_native_refit(library, "child", "new", options, service)
    return library, service, root


def mutate_capture(library: Any) -> None:
    path = Path(library.load_asset("exact-capture").path)
    before = path.stat()
    raw = bytearray(path.read_bytes())
    raw[-10] ^= 1
    path.write_bytes(raw)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_candidate_close_mutation_prevents_candidate_write(
    fit_case: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    library, service, root = queued(fit_case, tmp_path, monkeypatch)
    original = jobs._verify_scope_request

    def verify(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        mutate_capture(library)

    monkeypatch.setattr(jobs, "_verify_scope_request", verify)
    with pytest.raises(ValueError, match="hash|changed"):
        service.work(lambda _: None, lambda: False)
    assert not (root / "candidate.json").exists()
    assert "new" not in {asset.dataset_id for asset in library.assets("practice")}


def test_delayed_publish_close_mutation_prevents_add_fit(
    fit_case: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    library, service, root = queued(fit_case, tmp_path, monkeypatch)
    result = service.work(lambda _: None, lambda: False)
    original = jobs._verify_scope_request
    writes = []

    def verify(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        mutate_capture(library)

    monkeypatch.setattr(jobs, "_verify_scope_request", verify)
    monkeypatch.setattr(library, "add_fit", lambda *args: writes.append(args))
    with pytest.raises(ValueError, match="hash|changed"):
        result.publish()
    assert writes == []
    assert (root / "candidate.json").exists()


def test_capture_mutation_between_candidate_and_publish_prevents_add_fit(
    fit_case: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    library, service, root = queued(fit_case, tmp_path, monkeypatch)
    result = service.work(lambda _: None, lambda: False)
    writes = []
    monkeypatch.setattr(library, "add_fit", lambda *args: writes.append(args))
    mutate_capture(library)
    with pytest.raises(ValueError, match="hash|changed"):
        result.publish()
    assert writes == []
    assert (root / "candidate.json").exists()


def test_candidate_and_delayed_publish_use_fresh_contexts_before_writes(
    fit_case: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    import cv2

    library, service, root = queued(fit_case, tmp_path, monkeypatch)
    original_decode, original_write = cv2.imdecode, jobs.atomic_write_json
    calls, writes = [], []

    def decode(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original_decode(*args, **kwargs)

    def write(path: Path, payload: Any) -> Path:
        if path.name == "candidate.json":
            with library.authenticated_read():
                pass
            writes.append(path)
        return original_write(path, payload)

    def add(*args: Any) -> None:
        with library.authenticated_read():
            pass
        writes.append(args)

    monkeypatch.setattr(cv2, "imdecode", decode)
    monkeypatch.setattr(jobs, "atomic_write_json", write)
    monkeypatch.setattr(library, "add_fit", add)
    result = service.work(lambda _: None, lambda: False)
    assert len(calls) == 3
    result.publish()
    assert len(calls) == 6
    assert len(writes) == 2 and writes[0] == root / "candidate.json"
