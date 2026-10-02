"""Native/web job controls share the canonical executor and research outcomes."""

from threading import Event
from pathlib import Path
import json

import pytest

pytestmark = pytest.mark.unit


def test_refit_controls_bound_admission_cancel_and_reopen(fit_case, monkeypatch):
    from src.shared.python.workspace import NativeRefitSession, NativeRefitOptions
    from src.shared.python.workspace import necromatcher_refits as controls
    from src.shared.python.motion_matching.jobs import (
        MatchingJobSpec,
        HashBundle,
        JobStage,
        JobCancelledError,
    )

    library, _, _ = fit_case
    entered = Event()

    def launch(library, source, identity, options, service):
        root = library.root / "runs" / ("a" * 32)
        root.mkdir(parents=True)
        (root / "request.json").write_text(
            json.dumps({"source_fit_id": source, "new_fit_id": identity})
        )

        def work(progress, cancelled):
            entered.set()
            assert entered.wait(2)
            while not cancelled():
                Event().wait(0.01)
            raise JobCancelledError("Requested by operator")

        spec = MatchingJobSpec(
            root.name, "mujoco", JobStage.IK, root, HashBundle(*(["hash"] * 5))
        )
        return service.start(spec, work=work), root

    monkeypatch.setattr(controls, "start_native_refit", launch)
    session = NativeRefitSession(library)
    try:
        submitted = session.submit(
            "source", "new", NativeRefitOptions((0, 2), 2, (1.0,))
        )
        run = submitted["run_id"]
        assert entered.wait(2)
        assert session.view(run)["status"] == "running"
        assert session.view(run)["acceptance"] == "partial"
        with pytest.raises(RuntimeError, match="already running"):
            session.submit("other", "other-new", NativeRefitOptions((0, 2), 2, (1.0,)))
        session.cancel(run)
    finally:
        session.close()
    assert session.view(run)["status"] == "cancelled"
    fresh = NativeRefitSession(library)
    try:
        reopened = fresh.view(run)
        assert reopened["status"] == "cancelled"
        assert reopened["acceptance"] == "interrupted"
        assert reopened["new_fit_id"] == "new"
        assert "diagnostics_path" not in reopened
        with pytest.raises(ValueError):
            fresh.view("../../outside")
        with pytest.raises(KeyError):
            fresh.view("b" * 32)
    finally:
        fresh.close()


def test_api_refit_contract_rejects_invalid_options_and_unknown_job(fit_case):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes.necromatcher import router, get_library, get_refits
    from src.shared.python.workspace import NativeRefitSession

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    session = NativeRefitSession(library)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_library] = lambda: library
    app.dependency_overrides[get_refits] = lambda: session
    try:
        with TestClient(app) as client:
            assert client.get("/necromatcher/refits/" + "b" * 32).status_code == 404
            plan = client.get("/necromatcher/fits/source/refit-plan")
            assert plan.status_code == 200
            assert plan.json()["coordinate_order"] == ["hip"]
            invalid = client.post(
                "/necromatcher/fits/source/refits",
                json={
                    "new_fit_id": "new",
                    "frame_indices": [2, 0],
                    "knot_count": 2,
                    "coordinate_scales": [1.0],
                },
            )
            assert invalid.status_code == 422
            assert not (library.root / "runs").exists()
    finally:
        session.close()


def test_unowned_running_manifest_does_not_invent_worker_termination(fit_case):
    from src.shared.python.workspace import NativeRefitSession
    from src.shared.python.motion_matching.jobs import (
        RunManifest,
        JobStatus,
        JobStage,
        HashBundle,
        AcceptanceState,
        write_run_manifest,
    )

    library, _, _ = fit_case
    run_id = "c" * 32
    root = library.root / "runs" / run_id
    write_run_manifest(
        root,
        RunManifest(
            run_id,
            JobStatus.RUNNING,
            JobStage.IK,
            HashBundle(*(["hash"] * 5)),
            AcceptanceState.PARTIAL,
            "fresh",
        ),
    )
    (root / "request.json").write_text(
        json.dumps({"source_fit_id": "source", "new_fit_id": "new"})
    )
    session = NativeRefitSession(library)
    try:
        result = session.view(run_id)
        assert result["status"] == "running"
        assert result["acceptance"] == "partial"
        assert result["control_available"] is False
        assert "unverified" in result["message"]
        assert session.cancel(run_id) == result
    finally:
        session.close()


def test_api_submits_real_worker_and_reports_rejection_without_publishing(fit_case):
    import time
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes.necromatcher import router, get_library, get_refits
    from src.shared.python.workspace import NativeRefitSession

    library, source, _ = fit_case
    library.add_fit("source", "practice", source)
    session = NativeRefitSession(library)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_library] = lambda: library
    app.dependency_overrides[get_refits] = lambda: session
    try:
        with TestClient(app) as client:
            response = client.post(
                "/necromatcher/fits/source/refits",
                json={
                    "new_fit_id": "new",
                    "frame_indices": [0, 2],
                    "knot_count": 2,
                    "coordinate_scales": [1.0],
                },
            )
            assert response.status_code == 202
            run_id = response.json()["run_id"]
            deadline = time.monotonic() + 40
            result = response.json()
            while (
                result["status"] in {"pending", "running"}
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
                result = client.get("/necromatcher/refits/" + run_id).json()
            assert result["status"] == "failed"
            assert result["acceptance"] == "rejected"
            assert "native_definition" in result["message"]
            assert not any(x.dataset_id == "new" for x in library.assets("practice"))
    finally:
        session.close()
