"""Child measurements are authenticated separately from parent process budgets."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.shared.python.workspace import necromatcher_fit_worker as worker
from src.shared.python.workspace import necromatcher_fit_telemetry as telemetry


def _request(tmp_path):
    root = tmp_path / "runs" / ("a" * 32)
    root.mkdir(parents=True)
    request = {
        "library_root": str(tmp_path),
        "source_fit_id": "parent",
        "source_fit_hash": "sha256:" + "b" * 64,
        "execution_stamp": {"source_sha256": "c" * 64, "runtime_sha256": "d" * 64},
    }
    path = root / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path, request


def test_child_failure_persists_actual_worker_time_without_solver_counts(
    tmp_path, monkeypatch
):
    path, request = _request(tmp_path)
    monkeypatch.setattr(worker.sys, "argv", ["worker", str(path)])
    ticks = iter([20.0, 22.5])
    monkeypatch.setattr(worker.time, "perf_counter", lambda: next(ticks))

    def fail(_):
        raise ValueError("source admission failed")

    monkeypatch.setattr(worker, "compute_native_refit", fail)
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code == 1
    record = telemetry.read_worker_telemetry(path.parent, request)
    assert record.worker_elapsed_s == 2.5
    assert record.nfev is record.solver_elapsed_s is None
    assert record.termination_reason == "source admission failed"


def test_foreign_or_mutated_request_receipt_rejected(tmp_path):
    path, request = _request(tmp_path)
    telemetry.write_worker_telemetry(path, request, None, 1.0, "failed before result")
    changed = {**request, "source_fit_hash": "sha256:" + "e" * 64}
    with pytest.raises(ValueError, match="binding"):
        telemetry.read_worker_telemetry(path.parent, changed)
    data = json.loads(path.read_bytes())
    data["other"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        telemetry.read_worker_telemetry(path.parent, request)


def test_parent_timeout_receipt_keeps_child_durations_unknown(tmp_path):
    path, request = _request(tmp_path)
    telemetry.record_missing_worker_telemetry(path.parent, request, "wall guard")
    record = telemetry.read_worker_telemetry(path.parent, request)
    assert record.nfev is record.njev is None
    assert record.worker_elapsed_s is record.solver_elapsed_s is None
    assert record.unavailable_reason == "child_terminal_receipt_unavailable"


def test_no_telemetry_redirect_outside_owned_run(tmp_path):
    path, request = _request(tmp_path)
    stray = tmp_path / "request.json"
    stray.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="owned run"):
        telemetry.write_worker_telemetry(stray, request, None, 1.0, "failed")
    assert not (tmp_path / "worker-telemetry.json").exists()


def test_missing_worker_argument_remains_a_controlled_failure(monkeypatch):
    monkeypatch.setattr(worker.sys, "argv", ["worker"])
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code == 1


def test_completed_child_retains_distinct_solver_and_worker_scopes(
    tmp_path, monkeypatch, capsys
):
    from src.shared.python.estimation import SolverBackend, SolverTelemetry

    path, request = _request(tmp_path)
    measured = SolverTelemetry(
        3, 2, 0.125, None, "gtol", SolverBackend("scipy", "trf", "1.15")
    )
    fit = {
        "evidence": {
            "original_fit": {
                "solver_telemetry": measured.to_record(),
                "optimizer_ran": True,
            }
        },
        "qualification": "research",
    }
    monkeypatch.setattr(worker.sys, "argv", ["worker", str(path)])
    ticks = iter([100.0, 103.0])
    monkeypatch.setattr(worker.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(worker, "compute_native_refit", lambda _: fit)
    worker.main()
    response = json.loads(capsys.readouterr().out)
    stored = telemetry.read_worker_telemetry(path.parent, request)
    assert stored.nfev == 3 and stored.solver_elapsed_s == 0.125
    assert stored.worker_elapsed_s == 3.0
    assert stored.to_record()["worker_scope"] == (
        "child_entry_through_computed_payload_or_caught_failure"
        "_excluding_imports_telemetry_publication_and_transport_serialization"
    )
    assert (
        response["fit"]["evidence"]["original_fit"]["solver_telemetry"]
        == stored.to_record()
    )
    assert response["fit"]["qualification"] == "research"


@pytest.mark.parametrize("publication_fails", [False, True])
def test_parent_kill_preserves_original_failure_and_never_measures_child(
    tmp_path, monkeypatch, publication_fails
):
    from src.shared.python.workspace import necromatcher_fit_jobs as jobs

    path, request = _request(tmp_path)
    request["new_fit_id"] = "future"
    path.write_text(json.dumps(request), encoding="utf-8")
    stamp = request["execution_stamp"]
    monkeypatch.setattr(jobs, "fit_execution_stamp", lambda: stamp)
    fault = RuntimeError("wall execution budget exceeded")

    def fail(*_):
        raise fault

    monkeypatch.setattr(jobs, "_execute_worker", fail)
    if publication_fails:
        monkeypatch.setattr(
            jobs,
            "record_missing_worker_telemetry",
            lambda *_: (_ for _ in ()).throw(OSError("disk failure")),
        )
    spec = SimpleNamespace(
        run_root=path.parent, hashes=SimpleNamespace(solver_hash=stamp["source_sha256"])
    )
    operation = jobs._refit_work(
        SimpleNamespace(), SimpleNamespace(budget_wall_s=300), request, spec, "session"
    )
    with pytest.raises(RuntimeError) as error:
        operation(lambda _: None, lambda: False)
    assert error.value is fault
    if not publication_fails:
        stored = telemetry.read_worker_telemetry(path.parent, request)
        assert stored.nfev is stored.worker_elapsed_s is stored.solver_elapsed_s is None


@pytest.mark.parametrize("malformed", [True, 1.0])
def test_canonical_library_rejects_malformed_actual_counters(fit_case, malformed):
    from src.shared.python.estimation import SolverTelemetry

    library, path, payload = fit_case
    record = SolverTelemetry().to_record()
    record["nfev"] = malformed
    payload["evidence"]["original_fit"] = {"solver_telemetry": record}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="nfev"):
        library.add_fit("malformed", "practice", path)


def test_actual_counters_survive_immutable_library_recall_without_legacy_backfill(
    fit_case,
):
    from src.shared.python.estimation import SolverBackend, SolverTelemetry

    library, path, payload = fit_case
    original_bytes = path.read_bytes()
    library.add_fit("old", "practice", path)
    record = SolverTelemetry(
        2, 1, 0.2, 1.0, "early stop", SolverBackend("scipy", "trf", "1.15")
    )
    payload["evidence"]["original_fit"] = {"solver_telemetry": record.to_record()}
    path.write_text(json.dumps(payload), encoding="utf-8")
    library.add_fit("measured", "practice", path)
    recalled = library.load_fit("measured")
    assert (
        SolverTelemetry.from_record(
            recalled["evidence"]["original_fit"]["solver_telemetry"]
        )
        == record
    )
    assert recalled["qualification"] == "monocular_research_hypothesis"
    assert Path(library.load_asset("old").path).read_bytes() == original_bytes


@pytest.mark.parametrize("count", ["nfev", "njev"])
def test_unoptimized_import_cannot_claim_either_solver_counter(fit_case, count):
    from src.shared.python.estimation import SolverTelemetry

    library, path, payload = fit_case
    payload["evidence"]["original_fit"] = {
        "optimizer_ran": False,
        "solver_telemetry": SolverTelemetry(**{count: 0}).to_record(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="solver counts"):
        library.add_fit("unoptimized-counter", "practice", path)


def test_unoptimized_import_keeps_actual_worker_duration(fit_case):
    from src.shared.python.estimation import SolverTelemetry

    library, path, payload = fit_case
    record = SolverTelemetry(
        worker_elapsed_s=1.25, unavailable_reason="optimizer_not_run"
    )
    payload["evidence"]["original_fit"] = {
        "optimizer_ran": False,
        "solver_telemetry": record.to_record(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    library.add_fit("measured-initialization", "practice", path)
    assert (
        library.load_fit("measured-initialization")["evidence"]["original_fit"][
            "solver_telemetry"
        ]
        == record.to_record()
    )


def test_terminal_session_recall_preserves_failed_job_status(tmp_path, monkeypatch):
    from src.shared.python.workspace import NativeRefitSession

    path, request = _request(tmp_path)
    telemetry.record_missing_worker_telemetry(
        path.parent, request, "parent killed child"
    )
    session = NativeRefitSession(SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(
        session, "_view", lambda _: {"status": "failed", "acceptance": "rejected"}
    )
    result = session.view(path.parent.name)
    assert result["status"] == "failed" and result["acceptance"] == "rejected"
    assert result["solver_telemetry"]["nfev"] is None
    session.close()


def test_successful_computation_is_not_failed_by_optional_sidecar_fault(
    tmp_path, monkeypatch, capsys
):
    path, _ = _request(tmp_path)
    fit = {
        "evidence": {"original_fit": {"optimizer_ran": False}},
        "qualification": "research",
    }
    monkeypatch.setattr(worker.sys, "argv", ["worker", str(path)])
    monkeypatch.setattr(worker, "compute_native_refit", lambda _: fit)

    def fault(*_):
        raise OSError("sidecar disk fault")

    monkeypatch.setattr(worker, "write_worker_telemetry", fault)
    worker.main()
    response = json.loads(capsys.readouterr().out)
    assert response["fit"]["qualification"] == "research"
    record = response["fit"]["evidence"]["original_fit"]["solver_telemetry"]
    assert record["worker_elapsed_s"] is None
    assert record["unavailable_reason"] == "worker_telemetry_publication_failed"


def test_child_original_fault_retained_when_failure_sidecar_also_faults(
    tmp_path, monkeypatch, caplog
):
    path, _ = _request(tmp_path)
    monkeypatch.setattr(worker.sys, "argv", ["worker", str(path)])

    def compute(_):
        raise ValueError("original source fault")

    def publish(*_):
        raise OSError("sidecar disk fault")

    monkeypatch.setattr(worker, "compute_native_refit", compute)
    monkeypatch.setattr(worker, "write_worker_telemetry", publish)
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code == 1
    assert "original source fault" in caplog.text


def test_optional_emission_does_not_replace_malformed_payload_admission(
    tmp_path, monkeypatch, capsys
):
    path, _ = _request(tmp_path)
    fit = {"qualification": "malformed fixture must reach strict admission"}
    monkeypatch.setattr(worker.sys, "argv", ["worker", str(path)])
    monkeypatch.setattr(worker, "compute_native_refit", lambda _: fit)

    def publish(*_):
        raise OSError("sidecar disk fault")

    monkeypatch.setattr(worker, "write_worker_telemetry", publish)
    worker.main()
    assert json.loads(capsys.readouterr().out)["fit"] == fit
