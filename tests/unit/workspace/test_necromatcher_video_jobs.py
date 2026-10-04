"""Async overlay exports reuse matching jobs and reject unverified downloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event
import time
from zipfile import ZipFile

import pytest

pytestmark = pytest.mark.unit


def _wait(session, run):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        view = session.view(run)
        if view["status"] not in {"pending", "running"}:
            return view
        Event().wait(0.01)
    raise AssertionError("Owned test job did not terminate")


def _fake_export(request_path, budget, cancelled):
    request = json.loads(request_path.read_text())
    root = request_path.parent / "overlay"
    root.mkdir()
    data = b"synthetic codec fixture; no native or scientific validation"
    (root / "overlay.mp4").write_bytes(data)
    manifest = {
        "schema": "necromatcher/source-overlay-video/1",
        "fit_id": request["source_fit_id"],
        "fit_hash": request["source_fit_hash"],
        "model_id": request["model_id"],
        "model_hash": request["model_hash"],
        "capture_id": request["capture_id"],
        "capture_hash": request["capture_hash"],
        "qualification": "monocular_research_hypothesis",
        "physical_time_qualified": False,
        "camera_qualified": False,
        "anatomy_qualified": False,
        "video": {"path": "overlay.mp4", "sha256": hashlib.sha256(data).hexdigest()},
        "pngs": [],
    }
    for index in request["selected_frames"]:
        name = f"frame-{index:06d}.png"
        (root / name).write_bytes(data)
        manifest["pngs"].append(
            {"path": name, "sha256": hashlib.sha256(data).hexdigest()}
        )
    (root / "manifest.json").write_text(json.dumps(manifest))
    return {
        "manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest()
    }


def test_video_job_success_reopens_and_checks_zip(fit_case, monkeypatch):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    monkeypatch.setattr(jobs, "_execute_worker", _fake_export)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source")["run_id"]
        result = _wait(session, run)
        assert result["status"] == "succeeded"
        assert result["acceptance"] == "rejected"
        assert result["download_available"] and result["execution_verified"]
        assert result["source_fit_id"] == "source"
        archive = session.download(run)
        with ZipFile(archive) as zipped:
            assert set(zipped.namelist()) == {
                "manifest.json",
                "overlay.mp4",
                "frame-000000.png",
                "frame-000002.png",
            }
    finally:
        session.close()
    reopened = jobs.NativeVideoSession(library)
    try:
        assert reopened.view(run)["download_available"]
        assert not reopened.view(run)["control_available"]
        assert reopened.download(run).is_file()
        (library.root / "video-runs" / run / "overlay" / "overlay.mp4").write_bytes(
            b"changed"
        )
        with pytest.raises(ValueError, match="hash|Hash"):
            reopened.download(run)
    finally:
        reopened.close()


def test_video_job_cancel_and_one_owned_active_job(fit_case, monkeypatch):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs
    from src.shared.python.motion_matching.jobs import JobCancelledError

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    entered = Event()

    def blocked(request_path, budget, cancelled):
        entered.set()
        while not cancelled():
            Event().wait(0.01)
        raise JobCancelledError("Owned export cancelled")

    monkeypatch.setattr(jobs, "_execute_worker", blocked)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source")["run_id"]
        assert entered.wait(3)
        assert session.view(run)["execution_started"]
        with pytest.raises(RuntimeError, match="already running"):
            session.submit("source")
        with pytest.raises(ValueError, match="successful|success|verified"):
            session.download(run)
        session.cancel(run)
        assert _wait(session, run)["status"] == "cancelled"
        assert not session.view(run)["download_available"]
        with pytest.raises(ValueError):
            session.download(run)
    finally:
        session.close()


def test_video_job_failure_has_no_download(fit_case, monkeypatch):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)

    def fail(*args):
        raise ValueError("No native evidence")

    monkeypatch.setattr(jobs, "_execute_worker", fail)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source")["run_id"]
        assert _wait(session, run)["status"] == "failed"
        with pytest.raises(ValueError):
            session.download(run)
        with pytest.raises(ValueError):
            session.view("../elsewhere")
    finally:
        session.close()


def test_video_job_requires_every_requested_still(fit_case, monkeypatch):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)

    def omit_stills(request_path, budget, cancelled):
        _fake_export(request_path, budget, cancelled)
        manifest_path = request_path.parent / "overlay" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["pngs"] = []
        manifest_path.write_text(json.dumps(manifest))
        return {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        }

    monkeypatch.setattr(jobs, "_execute_worker", omit_stills)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source")["run_id"]
        assert _wait(session, run)["status"] == "failed"
        assert not session.view(run)["download_available"]
    finally:
        session.close()


def test_video_job_reopen_cannot_cancel_another_sessions_worker(fit_case, monkeypatch):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs
    from src.shared.python.motion_matching.jobs import JobCancelledError

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    entered = Event()

    def blocked(request_path, budget, cancelled):
        entered.set()
        while not cancelled():
            Event().wait(0.01)
        raise JobCancelledError("Owned export cancelled")

    monkeypatch.setattr(jobs, "_execute_worker", blocked)
    owner, reader = jobs.NativeVideoSession(library), jobs.NativeVideoSession(library)
    try:
        run = owner.submit("source")["run_id"]
        assert entered.wait(3)
        status = reader.cancel(run)
        assert status["status"] == "running"
        assert not status["control_available"] and not status["execution_verified"]
        assert "unverified" in status["message"]
        assert owner.view(run)["status"] == "running"
        owner.cancel(run)
        assert _wait(owner, run)["status"] == "cancelled"
    finally:
        reader.close()
        owner.close()


def test_worker_cancellation_terminates_only_owned_process(tmp_path, monkeypatch):
    import subprocess
    import sys
    from src.shared.python.workspace import necromatcher_video_jobs as jobs
    from src.shared.python.motion_matching.jobs import JobCancelledError

    owned = []
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def launch(command, **kwargs):
        assert command[2:4] == [
            "-m",
            "src.shared.python.workspace.necromatcher_video_worker",
        ]
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(10)"], **kwargs
        )
        owned.append(process)
        return process

    monkeypatch.setattr(jobs, "secure_popen", launch)
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 3

    try:
        with pytest.raises(JobCancelledError):
            jobs._execute_worker(tmp_path / "request.json", 5.0, cancelled)
        assert len(owned) == 1 and owned[0].poll() is not None
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.communicate(timeout=5)


def test_worker_revalidates_launch_stamp_and_parent_identity(fit_case, monkeypatch):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs
    from src.shared.python.workspace import necromatcher_video_worker as worker

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    monkeypatch.setattr(jobs, "_execute_worker", _fake_export)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source")["run_id"]
        assert _wait(session, run)["status"] == "succeeded"
        path = library.root / "video-runs" / run / "request.json"
        request = json.loads(path.read_text())
        calls = []

        def export(bound_library, fit_id, destination, *, selected_frames):
            calls.append((bound_library.root, fit_id, destination, selected_frames))

        monkeypatch.setattr(worker, "export_fit_video", export)
        receipt = worker.execute(path)
        assert receipt["manifest_sha256"]
        assert calls == [(library.root, "source", path.parent / "overlay", (0, 2))]
        request["execution_stamp"]["runtime_sha256"] = "sha256:" + "0" * 64
        path.write_text(json.dumps(request))
        with pytest.raises(ValueError, match="source or runtime differs"):
            worker.execute(path)
        assert len(calls) == 1
    finally:
        session.close()


def test_download_rejects_changed_parent_and_missing_completion(fit_case, monkeypatch):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    monkeypatch.setattr(jobs, "_execute_worker", _fake_export)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source")["run_id"]
        assert _wait(session, run)["status"] == "succeeded"
        monkeypatch.setattr(
            library,
            "load_fit",
            lambda _: (_ for _ in ()).throw(ValueError("Parent changed")),
        )
        with pytest.raises(ValueError, match="Parent changed"):
            session.download(run)
        (library.root / "video-runs" / run / "complete.json").unlink()
        assert not session.view(run)["download_available"]
        with pytest.raises(ValueError, match="verified"):
            session.download(run)
    finally:
        session.close()


def test_json_polling_preserves_missing_and_malformed_errors(tmp_path):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    path = tmp_path / "request.json"
    with pytest.raises(FileNotFoundError):
        jobs._read(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        jobs._read(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        jobs._read(path)


def test_force_layer_is_validated_recorded_and_forwarded_to_the_worker(
    fit_case, monkeypatch
):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs
    from src.shared.python.workspace import necromatcher_video_worker as worker

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    monkeypatch.setattr(jobs, "_execute_worker", _fake_export)
    session = jobs.NativeVideoSession(library)
    layer = {
        "enabled": True,
        "kinds": ["contact"],
        "scale": 2.0,
        "segment_shading": False,
    }
    try:
        with pytest.raises(ValueError, match="kinds"):
            session.submit("source", force_layer={**layer, "kinds": ["bogus"]})
        run = session.submit("source", force_layer=layer)["run_id"]
        _wait(session, run)
        path = library.root / "video-runs" / run / "request.json"
        assert json.loads(path.read_text())["force_layer"] == layer
        calls = []
        monkeypatch.setattr(
            worker, "export_fit_video", lambda *args, **kwargs: calls.append(kwargs)
        )
        worker.execute(path)
        assert calls[0]["force_layer"].kinds == ("contact",)
        assert calls[0]["force_sampler_factory"] is worker.mujoco_force_sampler
    finally:
        session.close()


def test_default_submission_has_no_force_layer_in_its_request(fit_case, monkeypatch):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    seen = []

    def recording_export(request_path, budget, cancelled):
        seen.append(json.loads(request_path.read_text()))
        return _fake_export(request_path, budget, cancelled)

    monkeypatch.setattr(jobs, "_execute_worker", recording_export)
    session = jobs.NativeVideoSession(library)
    try:
        _wait(session, session.submit("source")["run_id"])
        assert len(seen) == 1 and "force_layer" not in seen[0]
    finally:
        session.close()
