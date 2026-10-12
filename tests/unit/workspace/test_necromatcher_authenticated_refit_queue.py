"""Queue read authentication closes before requests and scheduling."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from authenticated_refit_fixture import make_scoped_refit_case
from src.shared.python.workspace import necromatcher_fit_jobs as jobs
from src.shared.python.workspace import necromatcher_fit as fit

pytestmark = pytest.mark.unit


class Service:
    def __init__(self, library: Any) -> None:
        self.library = library
        self.calls = []

    def start(self, spec: Any, work: Any) -> Any:
        # A fresh explicit context would reject if queue authentication leaked.
        with self.library.authenticated_read():
            pass
        self.calls.append(spec)
        self.work = work
        return SimpleNamespace()


def test_scoped_lineage_queue_decodes_each_original_once_per_submission(
    fit_case: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    import cv2

    library, options, payload = make_scoped_refit_case(fit_case, tmp_path)
    original = cv2.imdecode
    calls = []

    def decode(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "imdecode", decode)
    service = Service(library)
    _, root = jobs.start_native_refit(library, "child", "first", options, service)
    assert len(calls) == 3
    request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    assert request["source_scope"] == payload["provenance"]["source_fit_scope"]
    assert (
        request["source_scope_binding"]
        == payload["provenance"]["source_fit_scope_binding"]
    )
    assert request["options"] == json.loads(json.dumps(asdict(options)))
    jobs.start_native_refit(library, "child", "second", options, service)
    assert len(calls) == 6 and len(service.calls) == 2


def test_queue_close_mutation_prevents_run_root_request_and_service(
    fit_case: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    library, options, _ = make_scoped_refit_case(fit_case, tmp_path)
    original = fit.admit_refit_scope

    def admit(*args: Any, **kwargs: Any) -> Any:
        bound = original(*args, **kwargs)
        path = Path(library.load_asset("exact-capture").path)
        stat = path.stat()
        raw = bytearray(path.read_bytes())
        raw[-10] ^= 1
        path.write_bytes(raw)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        return bound

    # Recall itself invokes admission repeatedly; mutate only the final queue call.
    def final_admit(*args: Any, **kwargs: Any) -> Any:
        if len(args) == 6 and args[-1] is None:
            return admit(*args, **kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(fit, "admit_refit_scope", final_admit)
    service = Service(library)
    with pytest.raises(ValueError, match="hash|changed"):
        jobs.start_native_refit(library, "child", "new", options, service)
    assert service.calls == []
    assert not (library.root / "runs").exists()


def test_registered_scope_receipt_remains_fresh_before_queue(
    fit_case: Any, tmp_path: Path
) -> None:
    library, options, _ = make_scoped_refit_case(fit_case, tmp_path)
    Path(library.load_asset("review").path).write_bytes(b"{}")
    service = Service(library)
    with pytest.raises(ValueError, match="hash"):
        jobs.start_native_refit(library, "child", "new", options, service)
    assert service.calls == []
    assert not (library.root / "runs").exists()
