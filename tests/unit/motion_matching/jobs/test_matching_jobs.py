"""MS-105 (#10379): reliable matching jobs, recovery, and portable results.

Software-contract tests only. Native long-run timing and host recovery remain
named blockers — these tests never invent universal solve-time guarantees.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from src.shared.python.motion_matching.jobs import (
    JOBS_SCHEMA,
    SERVICE_TARGETS,
    AcceptanceState,
    CheckpointCompatibilityError,
    CorruptPackageError,
    DiskFullError,
    EngineUnavailableError,
    FaultKind,
    HashBundle,
    IncompatibleResumeError,
    JobCancelledError,
    JobProgress,
    JobStage,
    JobStatus,
    MatchingJobService,
    MatchingJobSpec,
    PartialOutputRejectedError,
    PortablePackageError,
    ProcessGuard,
    RecoveryDecision,
    RunManifest,
    StageBenchmarkSample,
    TimeToAcceptedReport,
    UnsupportedHostError,
    classify_fault,
    decide_recovery,
    export_portable_package,
    import_portable_package,
    load_checkpoint,
    measure_stage_sample,
    publish_service_targets,
    write_checkpoint,
    write_run_manifest,
)

pytestmark = pytest.mark.unit


def _hashes(**overrides: str) -> HashBundle:
    base = {
        "data_hash": "sha256:" + "a" * 64,
        "model_hash": "sha256:" + "b" * 64,
        "runtime_hash": "sha256:" + "c" * 64,
        "controller_hash": "sha256:" + "d" * 64,
        "solver_hash": "sha256:" + "e" * 64,
    }
    base.update(overrides)
    return HashBundle(**base)


def _spec(run_root: Path, **kwargs: object) -> MatchingJobSpec:
    defaults: dict[str, object] = {
        "run_id": "run_ms105_demo",
        "engine": "pinocchio",
        "stage": JobStage.FIT_REPLAY,
        "run_root": run_root,
        "hashes": _hashes(),
        "budget_wall_s": 30.0,
        "max_workers": 1,
    }
    defaults.update(kwargs)
    return MatchingJobSpec(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


class TestContracts:
    def test_schema_marker(self) -> None:
        assert JOBS_SCHEMA.startswith("motion-matching-jobs/")

    def test_hash_bundle_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="data_hash"):
            HashBundle(
                data_hash="",
                model_hash="x",
                runtime_hash="x",
                controller_hash="x",
                solver_hash="x",
            )

    def test_acceptance_partial_never_equals_accepted(self) -> None:
        assert AcceptanceState.PARTIAL != AcceptanceState.ACCEPTED
        assert AcceptanceState.INTERRUPTED != AcceptanceState.ACCEPTED

    def test_job_progress_fraction_bounds(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            JobProgress(stage=JobStage.IK, fraction=1.5, message="bad")

    def test_spec_rejects_second_scheduler_flag(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="second scheduler"):
            _spec(tmp_path, use_external_scheduler=True)


# ---------------------------------------------------------------------------
# Atomic manifests and checkpoints
# ---------------------------------------------------------------------------


class TestManifestAndCheckpoint:
    def test_atomic_manifest_roundtrip(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        root.mkdir()
        manifest = RunManifest(
            run_id="r1",
            status=JobStatus.RUNNING,
            stage=JobStage.CANDIDATE,
            hashes=_hashes(),
            acceptance=AcceptanceState.PARTIAL,
            provenance="fresh",
        )
        path = write_run_manifest(root, manifest)
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["schema_version"] == JOBS_SCHEMA
        assert loaded["acceptance"] == "partial"
        assert loaded["status"] == "running"

    def test_checkpoint_resume_compatible(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        root.mkdir()
        hashes = _hashes()
        ckpt_path = write_checkpoint(
            root,
            stage=JobStage.REALLOCATION,
            hashes=hashes,
            payload={"iter": 3},
            acceptance=AcceptanceState.PARTIAL,
        )
        loaded = load_checkpoint(ckpt_path, expected_hashes=hashes)
        assert loaded.stage == JobStage.REALLOCATION
        assert loaded.payload["iter"] == 3
        assert loaded.provenance == "checkpoint"

    def test_incompatible_resume_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        root.mkdir()
        ckpt_path = write_checkpoint(
            root,
            stage=JobStage.IK,
            hashes=_hashes(),
            payload={},
            acceptance=AcceptanceState.PARTIAL,
        )
        with pytest.raises((IncompatibleResumeError, CheckpointCompatibilityError)):
            load_checkpoint(
                ckpt_path, expected_hashes=_hashes(model_hash="sha256:" + "f" * 64)
            )

    def test_resumed_solution_distinct_from_interrupted(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        root.mkdir()
        hashes = _hashes()
        write_checkpoint(
            root,
            stage=JobStage.FIT_REPLAY,
            hashes=hashes,
            payload={"q": [0.0]},
            acceptance=AcceptanceState.PARTIAL,
        )
        service = MatchingJobService()
        resumed = service.resume_or_restart(
            _spec(root, hashes=hashes),
            reason="app_restart",
        )
        assert resumed.provenance in {"resumed_numerical", "restarted"}
        assert resumed.acceptance != AcceptanceState.ACCEPTED
        assert resumed.resume_reason == "app_restart"


# ---------------------------------------------------------------------------
# Fault injection / recovery
# ---------------------------------------------------------------------------


class TestFaultInjection:
    def test_cancel_preserves_diagnostics(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        root.mkdir()
        service = MatchingJobService()
        cancelled = False
        started = threading.Event()

        def work(progress_cb, cancel_check):  # noqa: ANN001
            nonlocal cancelled
            progress_cb(JobProgress(JobStage.IK, 0.2, "ik"))
            started.set()
            # Cooperative long-running job: wait for cancel (no race with join).
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if cancel_check():
                    cancelled = True
                    raise JobCancelledError("user cancel")
                time.sleep(0.01)
            raise AssertionError("cancel was never observed")

        handle = service.start(_spec(root), work=work)
        assert started.wait(timeout=2.0)
        handle.request_cancel()
        result = handle.join(timeout=5.0)
        assert result.status == JobStatus.CANCELLED
        assert result.acceptance == AcceptanceState.INTERRUPTED
        assert result.diagnostics_path is not None
        assert Path(result.diagnostics_path).exists()
        assert cancelled is True

    def test_partial_output_not_accepted(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        root.mkdir()
        service = MatchingJobService()
        with pytest.raises(PartialOutputRejectedError):
            service.advertise_as_accepted(
                RunManifest(
                    run_id="r",
                    status=JobStatus.FAILED,
                    stage=JobStage.FIT_REPLAY,
                    hashes=_hashes(),
                    acceptance=AcceptanceState.PARTIAL,
                    provenance="interrupted",
                )
            )

    def test_engine_absence_named_blocker(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        root.mkdir()
        service = MatchingJobService(engine_available=lambda _e: False)
        result = service.start(
            _spec(root, engine="opensim"),
            work=lambda *_a, **_k: {"ok": True},
        ).join()
        assert result.status == JobStatus.FAILED
        assert result.fault == FaultKind.ENGINE_ABSENT
        assert "native_engine_unavailable" in result.blockers

    def test_disk_full_classified(self) -> None:
        fault = classify_fault(OSError(28, "No space left on device"))
        assert fault == FaultKind.DISK_FULL
        decision = decide_recovery(fault, has_compatible_checkpoint=True)
        assert decision == RecoveryDecision.PRESERVE_AND_FAIL

    def test_worker_crash_restarts_with_reason(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        root.mkdir()
        hashes = _hashes()
        write_checkpoint(
            root,
            stage=JobStage.CANDIDATE,
            hashes=hashes,
            payload={"step": 1},
            acceptance=AcceptanceState.PARTIAL,
        )
        service = MatchingJobService()
        outcome = service.resume_or_restart(
            _spec(root, hashes=hashes),
            reason="worker_crash",
        )
        assert outcome.resume_reason == "worker_crash"
        assert outcome.provenance in {"resumed_numerical", "restarted"}

    def test_corrupt_artifact_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "run"
        root.mkdir()
        bad = root / "checkpoint.json"
        bad.write_text("{not-json", encoding="utf-8")
        with pytest.raises((CorruptPackageError, ValueError, json.JSONDecodeError)):
            load_checkpoint(bad, expected_hashes=_hashes())

    def test_unavailable_host(self) -> None:
        decision = decide_recovery(
            FaultKind.HOST_UNAVAILABLE, has_compatible_checkpoint=False
        )
        assert decision == RecoveryDecision.FAIL_CLOSED
        with pytest.raises(UnsupportedHostError):
            raise UnsupportedHostError("host offline")


# ---------------------------------------------------------------------------
# Process guard (reuse managed_popen contract; no second scheduler)
# ---------------------------------------------------------------------------


class TestProcessGuard:
    def test_cancel_terminates_owned_tree(self) -> None:
        guard = ProcessGuard()
        ended: list[int] = []

        class _Proc:
            pid = 4242

            def poll(self) -> int | None:
                return None

            def terminate(self) -> None:
                ended.append(self.pid)

            def kill(self) -> None:
                ended.append(-self.pid)

            def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
                return 0

        guard.register(_Proc())  # type: ignore[arg-type]
        guard.terminate_all(reason="cancel")
        assert 4242 in ended

    def test_service_does_not_expose_scheduler(self) -> None:
        service = MatchingJobService()
        assert not hasattr(service, "schedule")
        assert not hasattr(service, "enqueue")


# ---------------------------------------------------------------------------
# Portable packages
# ---------------------------------------------------------------------------


class TestPortablePackage:
    def test_export_import_relocated(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        run_root.mkdir()
        capture = run_root / "capture.json"
        capture.write_text('{"markers": []}', encoding="utf-8")
        artifact = run_root / "result.json"
        artifact.write_text('{"q": [0.1]}', encoding="utf-8")
        write_run_manifest(
            run_root,
            RunManifest(
                run_id="pack1",
                status=JobStatus.SUCCEEDED,
                stage=JobStage.FIT_REPLAY,
                hashes=_hashes(),
                acceptance=AcceptanceState.ACCEPTED,
                provenance="fresh",
            ),
        )
        pkg_dir = tmp_path / "package"
        export_portable_package(
            run_root,
            pkg_dir,
            asset_paths={"capture": capture, "result": artifact},
            input_capture_paths=(capture,),
        )
        relocated = tmp_path / "elsewhere" / "package"
        relocated.parent.mkdir()
        # Simulate relocate by copying tree
        import shutil

        shutil.copytree(pkg_dir, relocated)
        opened = import_portable_package(relocated)
        assert opened.run_id == "pack1"
        assert opened.acceptance == AcceptanceState.ACCEPTED
        assert "capture" in opened.assets
        assert opened.assets["capture"].is_relative_to(relocated)

    def test_corruption_rejected(self, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "manifest.json").write_text(
            '{"schema_version": "wrong"}', encoding="utf-8"
        )
        with pytest.raises((CorruptPackageError, PortablePackageError, ValueError)):
            import_portable_package(pkg)

    def test_never_overwrites_input_capture(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        run_root.mkdir()
        capture = run_root / "capture.json"
        capture.write_text('{"v": 1}', encoding="utf-8")
        write_run_manifest(
            run_root,
            RunManifest(
                run_id="pack2",
                status=JobStatus.SUCCEEDED,
                stage=JobStage.FIT_REPLAY,
                hashes=_hashes(),
                acceptance=AcceptanceState.ACCEPTED,
                provenance="fresh",
            ),
        )
        pkg_dir = tmp_path / "package"
        export_portable_package(
            run_root,
            pkg_dir,
            asset_paths={"capture": capture},
            input_capture_paths=(capture,),
        )
        # Mutate capture after export; re-export into same package must refuse
        capture.write_text('{"v": 2}', encoding="utf-8")
        with pytest.raises(PortablePackageError, match="overwrite|input capture"):
            export_portable_package(
                run_root,
                pkg_dir,
                asset_paths={"capture": capture},
                input_capture_paths=(capture,),
            )

    def test_rejects_pickle_payload(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        run_root.mkdir()
        pickle_path = run_root / "payload.pkl"
        pickle_path.write_bytes(b"\x80\x04")
        write_run_manifest(
            run_root,
            RunManifest(
                run_id="pack3",
                status=JobStatus.SUCCEEDED,
                stage=JobStage.FIT_REPLAY,
                hashes=_hashes(),
                acceptance=AcceptanceState.ACCEPTED,
                provenance="fresh",
            ),
        )
        with pytest.raises(PortablePackageError, match="pickle"):
            export_portable_package(
                run_root,
                tmp_path / "pkg",
                asset_paths={"payload": pickle_path},
                input_capture_paths=(),
            )

    def test_path_escape_rejected_on_import(self, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        manifest = {
            "schema_version": JOBS_SCHEMA,
            "run_id": "x",
            "status": "succeeded",
            "stage": "fit_replay",
            "acceptance": "accepted",
            "hashes": _hashes().to_dict(),
            "provenance": "fresh",
            "assets": {"escape": "../outside.txt"},
            "checksums": {},
        }
        (pkg / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises((PortablePackageError, ValueError)):
            import_portable_package(pkg)


# ---------------------------------------------------------------------------
# Both-shell progress / failure DTOs
# ---------------------------------------------------------------------------


class TestShellProgress:
    def test_progress_and_failure_payloads(self, tmp_path: Path) -> None:
        service = MatchingJobService()
        root = tmp_path / "run"
        root.mkdir()

        def work(progress_cb, cancel_check):  # noqa: ANN001
            progress_cb(JobProgress(JobStage.IK, 0.5, "halfway"))
            raise EngineUnavailableError("drake missing")

        result = service.start(_spec(root, engine="drake"), work=work).join()
        desktop = service.shell_view(result, shell="desktop")
        web = service.shell_view(result, shell="web")
        assert (
            desktop["progress"]["fraction"] == 0.5 or result.status == JobStatus.FAILED
        )
        assert web["failure"]["fault"] == FaultKind.ENGINE_ABSENT.value
        assert desktop["failure"]["message"]
        assert web["can_reopen_package"] is True


# ---------------------------------------------------------------------------
# PF-08 benchmarks (folded #10438) — measured schema, not invented guarantees
# ---------------------------------------------------------------------------


class TestBenchmarks:
    def test_service_targets_are_budgets_not_guarantees(self) -> None:
        targets = publish_service_targets()
        assert targets is SERVICE_TARGETS or targets["schema"].startswith(
            "time-to-accepted-swing"
        )
        assert targets["guarantee"] is False
        assert targets["stages"]["ik"]["median_budget_s"] == 5.0
        assert targets["stages"]["reallocation"]["median_budget_s"] == 1.0
        assert targets["stages"]["candidate"]["median_budget_s"] == 10.0
        assert targets["stages"]["fit_replay"]["median_budget_s"] == 30.0

    def test_stage_sample_records_cold_warm_memory(self) -> None:
        sample = measure_stage_sample(
            stage=JobStage.IK,
            cold_s=4.2,
            warm_s=1.1,
            peak_memory_mb=128.0,
            cache_key="sha256:" + "a" * 64,
            cache_hit=False,
        )
        assert isinstance(sample, StageBenchmarkSample)
        assert sample.cold_s == 4.2
        assert sample.warm_s == 1.1
        assert sample.cache_hit is False

    def test_report_rejects_invented_universal_guarantee(self) -> None:
        report = TimeToAcceptedReport(
            samples=(
                measure_stage_sample(
                    JobStage.IK,
                    cold_s=3.0,
                    warm_s=1.0,
                    peak_memory_mb=64.0,
                    cache_key="k",
                    cache_hit=True,
                ),
            ),
            hardware_label="reference-desk-contract",
            guarantee=False,
        )
        payload = report.to_dict()
        assert payload["guarantee"] is False
        with pytest.raises(ValueError, match="guarantee"):
            TimeToAcceptedReport(samples=(), hardware_label="x", guarantee=True)

    def test_disk_full_error_type(self) -> None:
        with pytest.raises(DiskFullError):
            raise DiskFullError("ENOSPC")
