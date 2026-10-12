"""Owned research impact jobs fail closed before launch and after artifact changes."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from threading import Event
import json
import time
from zipfile import ZipFile
import pytest
from tests.unit.workspace.test_necromatcher_impact_contracts import geometry, selection
from tests.unit.workspace.test_necromatcher_impact_receipt import (
    extracted_state,
    trajectory_file,
)

pytestmark = pytest.mark.unit


class Library:
    def __init__(self, root):
        self.root = root
        self.assets = {}
        self.trace = SimpleNamespace(t=[0.0, 0.1], meta={})
        for name, kind in [
            ("replay", "authored_replay"),
            ("profile", "torque_profile"),
            ("fit", "kinematic_fit"),
            ("model", "native_model"),
            ("capture", "image_capture"),
        ]:
            path = root / (name + ".bin")
            path.write_bytes(name.encode())
            from src.shared.python.workspace.artifact_handoff import compute_file_sha256

            digest = compute_file_sha256(path)
            self.assets[name] = SimpleNamespace(
                kind=kind, session_id="swing", metadata={"hash": digest}, path=path
            )
            if name != "replay":
                self.trace.meta.update({name + "_id": name, name + "_hash": digest})

    def authenticated_read(self):
        return nullcontext()

    def load_asset(self, identity):
        from src.shared.python.workspace.artifact_handoff import compute_file_sha256

        asset = self.assets[identity]
        if compute_file_sha256(asset.path) != asset.metadata["hash"]:
            raise ValueError("Parent bytes changed")
        return asset

    def load_replay(self, identity):
        self.load_asset(identity)
        for name in ("profile", "fit", "model", "capture"):
            self.load_asset(name)
        return self.trace


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from src.shared.python.workspace import necromatcher_impact_jobs as jobs

    stamp = {
        "source_sha256": "sha256:" + "a" * 64,
        "runtime_sha256": "sha256:" + "b" * 64,
        "source_commit": "c" * 40,
    }
    monkeypatch.setattr(jobs, "impact_execution_stamp", lambda: dict(stamp))
    # Build the real interchange fixture before timing the asynchronous job double.
    from src.shared.python.workspace.necromatcher_impact_receipt import (
        export_replay_impact_receipt,
        load_replay_impact_receipt,
    )

    prepared = tmp_path / "prepared-wire"
    prepared.mkdir()
    trajectory = trajectory_file(prepared)
    receipt = export_replay_impact_receipt(
        extracted_state(), trajectory, prepared / "receipt.json"
    )
    load_replay_impact_receipt(receipt, trajectory)
    return jobs, Library(tmp_path)


def fake_worker(path, budget, cancelled, *, operation):
    from src.shared.python.workspace.artifact_handoff import compute_file_sha256

    request = json.loads(path.read_text())
    out = path.parent / "output"
    out.mkdir()
    from src.shared.python.workspace.necromatcher_impact_receipt import (
        export_replay_impact_receipt,
    )

    state = extracted_state()
    state.metadata.update(request["parents"])
    state.metadata.update(
        geometry=request["geometry"],
        selection=request["selection"],
        recorded_sample_index=request["selection"]["recorded_sample_index"],
    )
    state.clubhead_mass = request["geometry"]["mass_kg"]
    state.clubhead_moi = request["geometry"]["moi_kg_m2"]
    trajectory = trajectory_file(out)
    export_replay_impact_receipt(state, trajectory, out / "impact-receipt.json")
    result = {
        "schema": "necromatcher/impact-result/1",
        "parents": request["parents"],
        "geometry": request["geometry"],
        "selection": request["selection"],
        "scientific_qualified": False,
        "physical_source_time_qualified": False,
        "execution_stamp": request["execution_stamp"],
        "summary": {
            "carry_m": 10.0,
            "max_height_m": 2.0,
            "flight_time_s": 1.0,
            "landing_angle_deg": -15.0,
        },
        "replay_clock_policy": "authored_simulation_seconds",
    }
    (out / "result.json").write_text(json.dumps(result))
    return {"artifact_hashes": {p.name: compute_file_sha256(p) for p in out.iterdir()}}


def wait(session, run):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        view = session.view("replay", run)
        if view["status"] not in ("pending", "running"):
            return view
        time.sleep(0.01)
    raise AssertionError("Owned job did not close")


def test_success_recall_download_authenticates_all_bytes(setup, monkeypatch):
    jobs, library = setup
    monkeypatch.setattr(jobs, "execute_native_research_worker", fake_worker)
    session = jobs.NativeImpactSession(library)
    try:
        run = session.submit("replay", geometry(), selection(), 30.0)["run_id"]
        view = wait(session, run)
        assert view["status"] == "succeeded" and view["acceptance"] == "rejected"
        assert view["execution_verified"] and view["download_available"]
        assert view["scientific_qualified"] is False
        assert view["artifactsummary"]["summary"]["carry_m"] == 10.0
        assert view["artifactsummary"]["geometry"] == geometry().to_record()
        assert view["artifactsummary"]["selection"] == selection().to_record()
        assert view["artifactsummary"]["clockpolicy"] == "authored_simulation_seconds"
        with ZipFile(session.download("replay", run)) as archive:
            assert set(archive.namelist()) == {
                "trajectory.json",
                "impact-receipt.json",
                "result.json",
                "request.json",
            }
    finally:
        session.close()
    reopened = jobs.NativeImpactSession(library)
    try:
        assert reopened.view("replay", run)["download_available"]
    finally:
        reopened.close()


@pytest.mark.parametrize("budget", [True, 0, -1, 601, float("nan"), float("inf"), "30"])
def test_invalid_budget_rejected_before_any_run(setup, budget):
    jobs, library = setup
    session = jobs.NativeImpactSession(library)
    try:
        with pytest.raises((ValueError, TypeError)):
            session.submit("replay", geometry(), selection(), budget)
        assert not (library.root / "impact-runs").exists()
    finally:
        session.close()


def test_bad_selection_and_type_rejected_before_worker(setup):
    from dataclasses import replace

    jobs, library = setup
    session = jobs.NativeImpactSession(library)
    try:
        with pytest.raises(ValueError):
            session.submit("replay", {}, selection(), 30.0)
        with pytest.raises(IndexError):
            session.submit(
                "replay",
                geometry(),
                replace(selection(), recorded_sample_index=2),
                30.0,
            )
        assert not (library.root / "impact-runs").exists()
    finally:
        session.close()


@pytest.mark.parametrize(
    "fault", ["request", "artifact", "parent", "qualification", "complete", "manifest"]
)
def test_tamper_blocks_verified_download(setup, monkeypatch, fault):
    jobs, library = setup
    monkeypatch.setattr(jobs, "execute_native_research_worker", fake_worker)
    session = jobs.NativeImpactSession(library)
    try:
        run = session.submit("replay", geometry(), selection(), 30.0)["run_id"]
        assert wait(session, run)["execution_verified"]
        root = library.root / "impact-runs" / run
        if fault == "parent":
            library.assets["model"].path.write_bytes(b"tamper")
        elif fault == "artifact":
            (root / "output" / "trajectory.json").write_text("{}")
        elif fault == "request":
            (root / "request.json").write_text("{}")
        elif fault == "complete":
            (root / "complete.json").write_text("{}")
        elif fault == "manifest":
            (root / "run_manifest.json").write_text("{}")
        else:
            path = root / "output" / "impact-receipt.json"
            data = json.loads(path.read_text())
            data["state"]["metadata"]["scientific_qualified"] = True
            path.write_text(json.dumps(data))
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            session.download("replay", run)
    finally:
        session.close()


def test_cancel_blocks_publication_and_foreign_replay_control(setup, monkeypatch):
    from src.shared.python.motion_matching.jobs import JobCancelledError

    jobs, library = setup
    started = Event()

    def worker(path, budget, cancelled, *, operation):
        started.set()
        while not cancelled():
            time.sleep(0.005)
        raise JobCancelledError("cancelled")

    monkeypatch.setattr(jobs, "execute_native_research_worker", worker)
    session = jobs.NativeImpactSession(library)
    try:
        run = session.submit("replay", geometry(), selection(), 30.0)["run_id"]
        assert started.wait(1)
        with pytest.raises(ValueError):
            session.cancel("other", run)
        session.cancel("replay", run)
        assert wait(session, run)["status"] == "cancelled"
        with pytest.raises(RuntimeError):
            session.download("replay", run)
        assert not (library.root / "impact-runs" / run / "complete.json").exists()
    finally:
        session.close()


def test_budget_failure_is_failed_without_download(setup, monkeypatch):
    jobs, library = setup

    def worker(path, budget, cancelled, *, operation):
        assert budget == 0.1 and operation == "impact"
        raise RuntimeError("Native research exceeded wall execution budget")

    monkeypatch.setattr(jobs, "execute_native_research_worker", worker)
    session = jobs.NativeImpactSession(library)
    try:
        run = session.submit("replay", geometry(), selection(), 0.1)["run_id"]
        assert wait(session, run)["status"] == "failed"
        assert not session.view("replay", run)["download_available"]
    finally:
        session.close()


def test_orphan_running_state_has_no_controls_or_download(setup):
    from src.shared.python.motion_matching.jobs import (
        RunManifest,
        JobStatus,
        JobStage,
        AcceptanceState,
        HashBundle,
    )
    from src.shared.python.motion_matching.jobs.io_atomic import atomic_write_json

    jobs, library = setup
    run = "a" * 32
    root = library.root / "impact-runs" / run
    stamp = jobs.impact_execution_stamp()
    parents = jobs.impact_job_parents(library, "replay")
    request = {
        "kind": "necromatcher/impact-job/1",
        "run_id": run,
        "library_root": str(library.root),
        "replay_id": "replay",
        "parents": parents,
        "geometry": geometry().to_record(),
        "selection": selection().to_record(),
        "budget_wall_s": 30.0,
        "execution_stamp": stamp,
    }
    atomic_write_json(root / "request.json", request)
    manifest = RunManifest(
        run,
        JobStatus.RUNNING,
        JobStage.CANDIDATE,
        HashBundle(
            parents["capture_hash"],
            parents["model_hash"],
            stamp["runtime_sha256"],
            "sha256:" + "d" * 64,
            stamp["source_sha256"],
        ),
        AcceptanceState.PARTIAL,
        "fresh",
    )
    atomic_write_json(root / "run_manifest.json", manifest.to_dict())
    session = jobs.NativeImpactSession(library)
    try:
        view = session.view("replay", run)
        assert (
            view["control_available"] is False and view["execution_verified"] is False
        )
        with pytest.raises(RuntimeError):
            session.cancel("replay", run)
        with pytest.raises(RuntimeError):
            session.download("replay", run)
    finally:
        session.close()


def test_malformed_summary_never_publishes_success(setup, monkeypatch):
    jobs, library = setup

    def worker(path, budget, cancelled, *, operation):
        response = fake_worker(path, budget, cancelled, operation=operation)
        output = path.parent / "output" / "result.json"
        data = json.loads(output.read_text())
        data["summary"]["carry_m"] = True
        output.write_text(json.dumps(data))
        from src.shared.python.workspace.artifact_handoff import compute_file_sha256

        response["artifact_hashes"]["result.json"] = compute_file_sha256(output)
        return response

    monkeypatch.setattr(jobs, "execute_native_research_worker", worker)
    session = jobs.NativeImpactSession(library)
    try:
        run = session.submit("replay", geometry(), selection(), 30.0)["run_id"]
        assert wait(session, run)["status"] == "failed"
        assert not (library.root / "impact-runs" / run / "complete.json").exists()
    finally:
        session.close()


def test_stored_success_cannot_claim_scientific_acceptance(setup, monkeypatch):
    jobs, library = setup
    monkeypatch.setattr(jobs, "execute_native_research_worker", fake_worker)
    session = jobs.NativeImpactSession(library)
    try:
        run = session.submit("replay", geometry(), selection(), 30.0)["run_id"]
        wait(session, run)
    finally:
        session.close()
    path = library.root / "impact-runs" / run / "run_manifest.json"
    record = json.loads(path.read_text())
    record["acceptance"] = "accepted"
    path.write_text(json.dumps(record))
    recalled = jobs.NativeImpactSession(library)
    try:
        with pytest.raises(ValueError, match="acceptance"):
            recalled.view("replay", run)
    finally:
        recalled.close()
