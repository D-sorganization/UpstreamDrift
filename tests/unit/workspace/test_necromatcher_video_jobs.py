"""Async overlay exports reuse matching jobs and reject unverified downloads."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from threading import Event
import time
from zipfile import ZipFile

import pytest

pytestmark = pytest.mark.unit


def _freeze_execution_stamp(jobs, monkeypatch):
    """Isolate fake-worker availability tests from concurrent source edits."""
    stamp = jobs.fit_execution_stamp()
    monkeypatch.setattr(jobs, "fit_execution_stamp", lambda: stamp)


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


@pytest.mark.parametrize(
    "mutation",
    [
        "output",
        "parent",
        "parent_fit",
        "parent_capture",
        "missing",
        "extra",
        "symlink",
        "metadata",
        "malformed_metadata",
    ],
)
def test_polling_invalidates_changed_artifacts_without_hashing(
    fit_case, monkeypatch, mutation
):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    _freeze_execution_stamp(jobs, monkeypatch)

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    monkeypatch.setattr(jobs, "_execute_worker", _fake_export)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source")["run_id"]
        assert _wait(session, run)["download_available"]
        overlay = library.root / "video-runs" / run / "overlay"
        if mutation == "output":
            (overlay / "overlay.mp4").write_bytes(b"changed")
        elif mutation.startswith("parent"):
            identity = {
                "parent": "model-v1",
                "parent_fit": "source",
                "parent_capture": "capture-v1",
            }[mutation]
            Path(library.load_asset(identity).path).write_bytes(b"changed")
        elif mutation == "missing":
            (overlay / "frame-000000.png").unlink()
        elif mutation == "extra":
            (overlay / "extra.txt").write_text("undeclared")
        elif mutation == "metadata":
            actual_assets = library.assets

            def changed_assets(swing):
                assets = deepcopy(actual_assets(swing))
                next(
                    asset for asset in assets if asset.dataset_id == "model-v1"
                ).metadata["hash"] = "changed"
                return assets

            monkeypatch.setattr(library, "assets", changed_assets)
        elif mutation == "malformed_metadata":
            from src.shared.python.core.contracts.exceptions import StateError

            def malformed_assets(swing):
                raise StateError("Malformed relevant project metadata")

            monkeypatch.setattr(library, "assets", malformed_assets)
        else:
            target = overlay / "overlay.mp4"
            actual = Path.is_symlink
            # Windows privilege restrictions do not weaken the symlink guard test.
            monkeypatch.setattr(
                Path, "is_symlink", lambda path: path == target or actual(path)
            )
        view = session.view(run)
        assert view["execution_verified"]
        assert not view["download_available"]
        assert view["artifact_state"] == "changed_or_unverified"
    finally:
        session.close()


def test_polling_uses_metadata_and_ignores_unrelated_fit_versions(
    fit_case, monkeypatch
):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    _freeze_execution_stamp(jobs, monkeypatch)

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    monkeypatch.setattr(jobs, "_execute_worker", _fake_export)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source")["run_id"]
        assert _wait(session, run)["download_available"]
        library.add_fit("unrelated", "practice", source)

        def no_verification(*args, **kwargs):
            raise AssertionError("Polling must not fully hash parents or output")

        monkeypatch.setattr(jobs, "_parents", no_verification)
        monkeypatch.setattr(jobs, "_outputs", no_verification)
        monkeypatch.setattr(library, "load_asset", no_verification)
        for _ in range(3):
            assert session.view(run)["download_available"]
    finally:
        session.close()


def test_legacy_completion_requires_explicit_download_verification(
    fit_case, monkeypatch
):
    from src.shared.python.workspace import necromatcher_video_jobs as jobs

    _freeze_execution_stamp(jobs, monkeypatch)

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    monkeypatch.setattr(jobs, "_execute_worker", _fake_export)
    session = jobs.NativeVideoSession(library)
    try:
        run = session.submit("source")["run_id"]
        assert _wait(session, run)["download_available"]
        complete_path = library.root / "video-runs" / run / "complete.json"
        complete = json.loads(complete_path.read_text())
        complete.pop("artifact_baseline", None)
        complete_path.write_text(json.dumps(complete))
        view = session.view(run)
        assert view["execution_verified"]
        assert not view["download_available"]
        assert view["producer_source_commit"]
        assert session.download(run).is_file()
        assert session.view(run)["download_available"]
    finally:
        session.close()
