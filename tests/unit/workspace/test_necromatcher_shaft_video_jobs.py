"""Optional video evidence is hashed and rebound at every publication boundary."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Any

import pytest

from tests.unit.motion_matching.test_shaft_observations import evidence
from tests.unit.workspace.test_necromatcher_video_jobs import (
    _fake_export,
    _freeze_execution_stamp,
    _wait,
)
from src.shared.python.workspace.necromatcher_shaft_evidence import BoundShaftEvidence

pytestmark = pytest.mark.unit


@pytest.fixture
def video_inputs(fit_case: Any, monkeypatch: Any) -> Any:
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    saved = library.load_fit("source")
    _freeze_execution_stamp(jobs, monkeypatch)
    value = replace(
        evidence(),
        capture_id=saved["capture_id"],
        capture_sha256=saved["capture_hash"],
    )
    calls: list[str] = []
    state = {"changed": False}

    def bind(lib: Any, fit: Any, record: Any) -> BoundShaftEvidence:
        assert lib is library and fit["capture_id"] == record.capture_id
        calls.append(record.sha256)
        if state["changed"]:
            raise ValueError("Shaft original PNG hash mismatch")
        return BoundShaftEvidence(record, "sha256:" + "e" * 64, 1, 1)

    monkeypatch.setattr(jobs, "bind_fit_shaft_evidence", bind)
    return jobs, library, value, calls, state


def export_with_shaft(
    request_path: Any, budget: float, cancelled: Any
) -> dict[str, str]:
    _fake_export(request_path, budget, cancelled)
    request = json.loads(request_path.read_text())
    manifest_path = request_path.parent / "overlay" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shaft_overlay"] = {
        "schema": "necromatcher/shaft-video-overlay/1",
        "evidence": request["shaft_overlay"]["evidence"],
        "evidence_sha256": request["shaft_overlay"]["evidence_sha256"],
        "source_clock_sha256": request["shaft_overlay"]["source_clock_sha256"],
        "uncertainty_calibrated": False,
        "physical_geometry_qualified": False,
    }
    manifest_path.write_text(json.dumps(manifest))
    return {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}


def test_enabled_evidence_enters_hash_and_rebinds_on_download(
    video_inputs: Any, monkeypatch: Any
) -> None:
    jobs, library, value, calls, state = video_inputs
    monkeypatch.setattr(jobs, "_execute_worker", export_with_shaft)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source", value)["run_id"]
        assert _wait(session, run)["status"] == "succeeded"
        root = library.root / "video-runs" / run
        request = json.loads((root / "request.json").read_text())
        manifest = json.loads((root / "run_manifest.json").read_text())
        expected = jobs._digest(
            {
                "selected_frames": request["selected_frames"],
                "shaft_overlay": request["shaft_overlay"],
            }
        )
        assert manifest["hashes"]["controller_hash"] == expected
        assert len(calls) >= 4
        assert session.download(run).is_file()
        state["changed"] = True
        with pytest.raises(ValueError, match="PNG"):
            session.download(run)
    finally:
        session.close()


def test_source_change_after_worker_prevents_completion(
    video_inputs: Any, monkeypatch: Any
) -> None:
    jobs, library, value, _, state = video_inputs

    def changed(request_path: Any, budget: float, cancelled: Any) -> dict[str, str]:
        response = export_with_shaft(request_path, budget, cancelled)
        state["changed"] = True
        return response

    monkeypatch.setattr(jobs, "_execute_worker", changed)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source", value)["run_id"]
        assert _wait(session, run)["status"] == "failed"
        assert not (library.root / "video-runs" / run / "complete.json").exists()
    finally:
        session.close()


def test_omitted_enabled_overlay_manifest_cannot_publish(
    video_inputs: Any, monkeypatch: Any
) -> None:
    jobs, library, value, _, _ = video_inputs
    monkeypatch.setattr(jobs, "_execute_worker", _fake_export)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source", value)["run_id"]
        assert _wait(session, run)["status"] == "failed"
    finally:
        session.close()
